"""Compare dense and JK/DSE QED-SAPT0 components for cavity scans.

This script is a diagnostic companion to ``water_methylamine_qed_sapt.py`` and
``water_methylamine_qed_sapt_jk.py``.  It explicitly evaluates the dense
reference implementation and the JK/DSE path for the same water-methylamine
cluster and cavity conditions, then prints the component differences.

Examples
--------
Run the current single-point cavity condition:

    python water_methylamine_qed_sapt_dense_vs_jk.py

Scan several z-polarized coupling strengths:

    python water_methylamine_qed_sapt_dense_vs_jk.py --lambda-z-values 0.0 0.05 0.1 0.15
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time
from dataclasses import dataclass

import numpy as np
import psi4

from cqed_scf import CQEDConfig
from cqed_scf.sapt import QEDSAPT0Driver
from cqed_scf.sapt import qed_sapt_jk
from cqed_scf.sapt.dse_jk import DSEJK, PauliFierzJK


PSI4_OPTIONS = {
    "basis": "jun-cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}

DIMER = """
0 1
O   -0.687464896  -0.111744327  -0.019625472
H   -1.046121544   0.775938208   0.012706845
H    0.274042519   0.025850654  -0.003497262
--
0 1
N    2.787113199   0.125007400   0.008492726
H    3.082477630  -0.427630575  -0.786298137
H    3.097193694  -0.385713691   0.825352219
C    3.446448476   1.433371365  -0.031748912
H    3.135906054   2.015096325   0.832766508
H    4.537757766   1.394076393  -0.040704580
H    3.119736204   1.969288834  -0.919572724
symmetry c1
no_com
no_reorient
"""

COMPONENT_ORDER = (
    "Electrostatics",
    "Exchange",
    "Dispersion",
    "Exchange-Dispersion",
    "Induction",
    "Exchange-Induction",
    "Total QED-SAPT0 Energy",
)


@dataclass
class CaseResult:
    lambda_vector: np.ndarray
    omega: float
    include_cavity_terms: bool
    dense_components: dict[str, float]
    jk_components: dict[str, float]
    dense_seconds: float
    jk_seconds: float


def _build_config(lambda_vector, omega, psi4_options, include_debug=False):
    return CQEDConfig(
        lambda_vector=np.asarray(lambda_vector, dtype=float),
        omega=float(omega),
        psi4_options=psi4_options,
        reference="rhf",
        functional=None,
        density_fitting=False,
        charge=0,
        multiplicity=1,
        dispersion_policy="none",
        debug=include_debug,
    )


def _build_dense_driver(lambda_vector, omega, psi4_options, include_cavity_terms):
    dimer_geometry = psi4.geometry(DIMER)
    config = _build_config(lambda_vector, omega, psi4_options)
    return QEDSAPT0Driver(
        dimer_geometry=dimer_geometry,
        config=config,
        integral_backend="full_eri",
        include_cavity_terms=include_cavity_terms,
    )


def _dense_components(driver):
    return {
        "Electrostatics": float(driver.Eelst100),
        "Exchange": float(driver.Eexch100),
        "Dispersion": float(driver.Edisp200),
        "Exchange-Dispersion": float(driver.Eexchdisp200),
        "Induction": float(driver.Eind200),
        "Exchange-Induction": float(driver.Eexchind200),
        "Total QED-SAPT0 Energy": float(driver.E_SAPT0),
    }


def _build_native_jk(wfn):
    jk = psi4.core.JK.build(wfn.basisset())
    jk.set_memory(int(1e9))
    if hasattr(jk, "set_do_J"):
        jk.set_do_J(True)
    if hasattr(jk, "set_do_K"):
        jk.set_do_K(True)
    if hasattr(jk, "set_do_wK"):
        jk.set_do_wK(False)
    jk.initialize()
    return jk


def _shared_dse_operator(driver):
    d_ao_A = np.asarray(driver.monomer_A.d_ao)
    d_ao_B = np.asarray(driver.monomer_B.d_ao)
    if not np.allclose(d_ao_A, d_ao_B, atol=1e-12, rtol=1e-12):
        raise ValueError("Monomer DSE AO dipole operators differ in the shared dimer basis.")
    return d_ao_A


def _apply_cavity_one_body_terms(cache, driver, d_ao):
    if not driver.include_cavity_terms:
        return

    V_A_cavity = -float(driver.d_exp_el_A) * d_ao
    V_B_cavity = -float(driver.d_exp_el_B) * d_ao
    cache["V_A"].axpy(1.0, psi4.core.Matrix.from_array(np.ascontiguousarray(V_A_cavity)))
    cache["V_B"].axpy(1.0, psi4.core.Matrix.from_array(np.ascontiguousarray(V_B_cavity)))
    cache["nuclear_repulsion_energy"] = float(
        driver.nuc_rep + driver.d_exp_el_A * driver.d_exp_el_B
    )


def _compute_jk_components(driver):
    wfn_A = driver.monomer_A.wfn
    wfn_B = driver.monomer_B.wfn
    native_jk = _build_native_jk(wfn_A)
    dse_jk = DSEJK(d_ao=_shared_dse_operator(driver))
    pf_jk = PauliFierzJK(native_jk, dse_jk=dse_jk)

    cache = qed_sapt_jk.build_sapt_jk_cache(
        wfn_A,
        wfn_B,
        pf_jk,
        do_print=False,
    )
    _apply_cavity_one_body_terms(cache, driver, dse_jk.d_ao)

    elst = qed_sapt_jk.electrostatics(cache, do_print=False)
    exch = qed_sapt_jk.exchange(cache, do_print=False)
    ind = qed_sapt_jk.induction(cache, do_print=False, do_response=True)

    components = {
        "Electrostatics": float(elst["Elst10,r"]),
        # The dense reference compute_Exch100() is the S^2 expression, so use
        # the matching JK key rather than the S^inf-style Exch10 value.
        "Exchange": float(exch["Exch10(S^2)"]),
        "Induction": float(ind["Ind20,r"]),
        "Exchange-Induction": float(ind["Exch-Ind20,r"]),
    }

    # Dispersion terms are not implemented in the local JK helper path. Carry
    # the dense values so that the mixed total isolates the implemented JK/DSE
    # discrepancies without hiding which rows came from dense code.
    components["Dispersion"] = float(driver.Edisp200)
    components["Exchange-Dispersion"] = float(driver.Eexchdisp200)
    components["Total QED-SAPT0 Energy"] = sum(
        components[name]
        for name in (
            "Electrostatics",
            "Exchange",
            "Dispersion",
            "Exchange-Dispersion",
            "Induction",
            "Exchange-Induction",
        )
    )
    return components


def _run_case(lambda_vector, omega, psi4_options, include_cavity_terms, quiet):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(psi4_options)

    driver = _build_dense_driver(
        lambda_vector=lambda_vector,
        omega=omega,
        psi4_options=psi4_options,
        include_cavity_terms=include_cavity_terms,
    )

    dense_start = time.time()
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            driver.run()
    else:
        driver.run()
    dense_seconds = time.time() - dense_start

    dense = _dense_components(driver)

    jk_start = time.time()
    jk = _compute_jk_components(driver)
    jk_seconds = time.time() - jk_start

    return CaseResult(
        lambda_vector=np.asarray(lambda_vector, dtype=float),
        omega=float(omega),
        include_cavity_terms=include_cavity_terms,
        dense_components=dense,
        jk_components=jk,
        dense_seconds=dense_seconds,
        jk_seconds=jk_seconds,
    )


def _print_case(result):
    lam = result.lambda_vector
    print()
    print(
        "lambda = "
        f"[{lam[0]: .8f}, {lam[1]: .8f}, {lam[2]: .8f}], "
        f"omega = {result.omega:.8f}, "
        f"include_cavity_terms = {result.include_cavity_terms}"
    )
    print(f"Dense reference time: {result.dense_seconds:.2f} s")
    print(f"JK/DSE time:          {result.jk_seconds:.2f} s")
    print()
    print(f"{'Component':<25} {'Dense / Eh':>16} {'JK or mixed / Eh':>18} {'Delta / Eh':>14}")
    print("-" * 78)
    for name in COMPONENT_ORDER:
        dense = result.dense_components[name]
        jk = result.jk_components[name]
        print(f"{name:<25} {dense:16.10f} {jk:18.10f} {jk - dense:14.6e}")

    exind_delta = (
        result.jk_components["Exchange-Induction"]
        - result.dense_components["Exchange-Induction"]
    )
    total_delta = (
        result.jk_components["Total QED-SAPT0 Energy"]
        - result.dense_components["Total QED-SAPT0 Energy"]
    )
    print()
    print(f"Exch-Ind20,r delta: {exind_delta: .10e} Eh")
    print(f"Total delta:        {total_delta: .10e} Eh")


def _print_scan_summary(results):
    if len(results) <= 1:
        return

    print()
    print("Scan summary")
    print(f"{'lambda_x':>12} {'lambda_y':>12} {'lambda_z':>12} {'omega':>10} {'dExch-Ind/r':>16} {'dInd/r':>16} {'dTotal':>16}")
    print("-" * 101)
    for result in results:
        lam = result.lambda_vector
        delta_exind = (
            result.jk_components["Exchange-Induction"]
            - result.dense_components["Exchange-Induction"]
        )
        delta_ind = result.jk_components["Induction"] - result.dense_components["Induction"]
        delta_total = (
            result.jk_components["Total QED-SAPT0 Energy"]
            - result.dense_components["Total QED-SAPT0 Energy"]
        )
        print(
            f"{lam[0]:12.6f} {lam[1]:12.6f} {lam[2]:12.6f} "
            f"{result.omega:10.6f} {delta_exind:16.8e} "
            f"{delta_ind:16.8e} {delta_total:16.8e}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare dense and JK/DSE QED-SAPT0 terms for water-methylamine cavity conditions."
    )
    parser.add_argument(
        "--lambda-vector",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.1),
        metavar=("LX", "LY", "LZ"),
        help="Cavity coupling vector for a single calculation. Default: 0 0 0.1.",
    )
    parser.add_argument(
        "--lambda-z-values",
        nargs="+",
        type=float,
        help="Run a scan over z-polarized lambda values. Overrides --lambda-vector.",
    )
    parser.add_argument("--omega", type=float, default=0.1, help="Cavity frequency. Default: 0.1.")
    parser.add_argument("--basis", default=PSI4_OPTIONS["basis"], help="Psi4 basis. Default: jun-cc-pVDZ.")
    parser.add_argument(
        "--no-cavity-terms",
        action="store_true",
        help="Disable explicit cavity terms in QEDSAPT0Driver while retaining the CQED-SCF monomer reference.",
    )
    parser.add_argument(
        "--show-scf-output",
        action="store_true",
        help="Do not suppress CQED-SCF progress prints from the dense reference driver.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    psi4.core.be_quiet()

    psi4_options = dict(PSI4_OPTIONS)
    psi4_options["basis"] = args.basis

    if args.lambda_z_values is None:
        lambda_vectors = [np.asarray(args.lambda_vector, dtype=float)]
    else:
        lambda_vectors = [np.array([0.0, 0.0, value], dtype=float) for value in args.lambda_z_values]

    results = []
    for lambda_vector in lambda_vectors:
        result = _run_case(
            lambda_vector=lambda_vector,
            omega=args.omega,
            psi4_options=psi4_options,
            include_cavity_terms=not args.no_cavity_terms,
            quiet=not args.show_scf_output,
        )
        results.append(result)
        _print_case(result)

    _print_scan_summary(results)

    print()
    print("Note: JK dispersion and exchange-dispersion rows are carried from the dense reference.")
    print('Exchange uses JK key "Exch10(S^2)" to match dense compute_Exch100().')


if __name__ == "__main__":
    main()
