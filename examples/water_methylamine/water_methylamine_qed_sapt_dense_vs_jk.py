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

UNCOUPLED_COMPONENT_ORDER = (
    "Ind20,u (A<-B)",
    "Ind20,u (A->B)",
    "Ind20,u",
    "Exch-Ind20,u (A<-B)",
    "Exch-Ind20,u (A->B)",
    "Exch-Ind20,u",
)

COUPLED_COMPONENT_ORDER = (
    "Ind20,r (A<-B)",
    "Ind20,r (A->B)",
    "Ind20,r",
    "Exch-Ind20,r (A<-B)",
    "Exch-Ind20,r (A->B)",
    "Exch-Ind20,r",
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
    components = {
        "Electrostatics": float(driver.Eelst100),
        "Exchange": float(driver.Eexch100),
        "Dispersion": float(driver.Edisp200),
        "Exchange-Dispersion": float(driver.Eexchdisp200),
        "Induction": float(driver.Eind200),
        "Exchange-Induction": float(driver.Eexchind200),
        "Total QED-SAPT0 Energy": float(driver.E_SAPT0),
    }
    components.update(_dense_uncoupled_components(driver))
    components.update(_dense_coupled_components(driver))
    return components


def _dense_uncoupled_components(driver):
    w_B_MOA = 2.0 * np.einsum("rbab->ar", driver.v("rbab"))
    w_B_MOA += driver.V_B_AA[driver.slices["a"], driver.slices["r"]]
    x_B_MOA = w_B_MOA / (driver.eps("a", dim=2) - driver.eps("r"))

    w_A_MOB = 2.0 * np.einsum("saba->bs", driver.v("saba"))
    w_A_MOB += driver.V_A_BB[driver.slices["b"], driver.slices["s"]]
    x_A_MOB = w_A_MOB / (driver.eps("b", dim=2) - driver.eps("s"))

    ind_ab = 2.0 * np.einsum("ar,ar->", x_B_MOA, w_B_MOA)
    ind_ba = 2.0 * np.einsum("bs,bs->", x_A_MOB, w_A_MOB)
    exch_ind = _dense_uncoupled_exch_ind_components(driver, x_B_MOA.T, x_A_MOB.T)

    components = {
        "Ind20,u (A<-B)": float(ind_ab),
        "Ind20,u (A->B)": float(ind_ba),
        "Ind20,u": float(ind_ab + ind_ba),
    }
    components.update(exch_ind)
    return components


def _dense_coupled_components(driver):
    w_B_MOA = driver.diagnostic_chf_rhs("B")
    w_A_MOB = driver.diagnostic_chf_rhs("A")

    ind_ab = 2.0 * np.einsum("ar,ar->", driver.CPHF_ra.T, w_B_MOA)
    ind_ba = 2.0 * np.einsum("bs,bs->", driver.CPHF_sb.T, w_A_MOB)
    exch_ind = _dense_uncoupled_exch_ind_components(
        driver,
        driver.CPHF_ra,
        driver.CPHF_sb,
    )

    components = {
        "Ind20,r (A<-B)": float(ind_ab),
        "Ind20,r (A->B)": float(ind_ba),
        "Ind20,r": float(ind_ab + ind_ba),
    }
    components.update(
        {key.replace(",u", ",r"): value for key, value in exch_ind.items()}
    )
    return components


def _dense_uncoupled_exch_ind_components(driver, CPHF_ra, CPHF_sb):
    vt_abra = driver.vt("abra")
    vt_abar = driver.vt("abar")

    ExchInd20_ab = np.einsum("ra,abbr->", CPHF_ra, driver.vt("abbr"))
    ExchInd20_ab += 2 * np.einsum("rA,Ab,abar->", CPHF_ra, driver.s("ab"), vt_abar)
    ExchInd20_ab += 2 * np.einsum("ra,Ab,abrA->", CPHF_ra, driver.s("ab"), vt_abra)
    ExchInd20_ab -= np.einsum("rA,Ab,abra->", CPHF_ra, driver.s("ab"), vt_abra)

    vt_abbb = driver.vt("abbb")
    vt_abab = driver.vt("abab")
    ExchInd20_ab -= np.einsum("ra,Ab,abAr->", CPHF_ra, driver.s("ab"), vt_abar)
    ExchInd20_ab += 2 * np.einsum("ra,Br,abBb->", CPHF_ra, driver.s("br"), vt_abbb)
    ExchInd20_ab -= np.einsum("ra,Br,abbB->", CPHF_ra, driver.s("br"), vt_abbb)
    ExchInd20_ab -= 2 * np.einsum(
        "rA,Ab,Br,abaB->", CPHF_ra, driver.s("ab"), driver.s("br"), vt_abab
    )

    vt_abrb = driver.vt("abrb")
    ExchInd20_ab -= 2 * np.einsum(
        "ra,Ab,BA,abrB->", CPHF_ra, driver.s("ab"), driver.s("ba"), vt_abrb
    )
    ExchInd20_ab -= 2 * np.einsum(
        "ra,AB,Br,abAb->", CPHF_ra, driver.s("ab"), driver.s("br"), vt_abab
    )
    ExchInd20_ab -= 2 * np.einsum(
        "rA,AB,Ba,abrb->", CPHF_ra, driver.s("ab"), driver.s("ba"), vt_abrb
    )

    ExchInd20_ab += np.einsum(
        "ra,Ab,Br,abAB->", CPHF_ra, driver.s("ab"), driver.s("br"), vt_abab
    )
    ExchInd20_ab += np.einsum(
        "rA,Ab,Ba,abrB->", CPHF_ra, driver.s("ab"), driver.s("ba"), vt_abrb
    )
    ExchInd20_ab *= -2

    vt_absb = driver.vt("absb")
    vt_abbs = driver.vt("abbs")
    ExchInd20_ba = np.einsum("sb,absa->", CPHF_sb, driver.vt("absa"))
    ExchInd20_ba += 2 * np.einsum("sB,Ba,absb->", CPHF_sb, driver.s("ba"), vt_absb)
    ExchInd20_ba += 2 * np.einsum("sb,Ba,abBs->", CPHF_sb, driver.s("ba"), vt_abbs)
    ExchInd20_ba -= np.einsum("sB,Ba,abbs->", CPHF_sb, driver.s("ba"), vt_abbs)

    vt_abaa = driver.vt("abaa")
    ExchInd20_ba -= np.einsum("sb,Ba,absB->", CPHF_sb, driver.s("ba"), vt_absb)
    ExchInd20_ba += 2 * np.einsum("sb,As,abaA->", CPHF_sb, driver.s("as"), vt_abaa)
    ExchInd20_ba -= np.einsum("sb,As,abAa->", CPHF_sb, driver.s("as"), vt_abaa)
    ExchInd20_ba -= 2 * np.einsum(
        "sB,Ba,As,abAb->", CPHF_sb, driver.s("ba"), driver.s("as"), vt_abab
    )

    vt_abas = driver.vt("abas")
    ExchInd20_ba -= 2 * np.einsum(
        "sb,Ba,AB,abAs->", CPHF_sb, driver.s("ba"), driver.s("ab"), vt_abas
    )
    ExchInd20_ba -= 2 * np.einsum(
        "sb,BA,As,abaB->", CPHF_sb, driver.s("ba"), driver.s("as"), vt_abab
    )
    ExchInd20_ba -= 2 * np.einsum(
        "sB,BA,Ab,abas->", CPHF_sb, driver.s("ba"), driver.s("ab"), vt_abas
    )

    ExchInd20_ba += np.einsum(
        "sb,Ba,As,abAB->", CPHF_sb, driver.s("ba"), driver.s("as"), vt_abab
    )
    ExchInd20_ba += np.einsum(
        "sB,Ba,Ab,abAs->", CPHF_sb, driver.s("ba"), driver.s("ab"), vt_abas
    )
    ExchInd20_ba *= -2

    return {
        "Exch-Ind20,u (A<-B)": float(ExchInd20_ab),
        "Exch-Ind20,u (A->B)": float(ExchInd20_ba),
        "Exch-Ind20,u": float(ExchInd20_ab + ExchInd20_ba),
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


def _compute_jk_components(driver):
    wfn_A = driver.monomer_A.wfn
    wfn_B = driver.monomer_B.wfn
    native_jk = _build_native_jk(wfn_A)
    dse_jk = DSEJK(d_ao=_shared_dse_operator(driver), enabled=driver.include_cavity_terms)
    pf_jk = PauliFierzJK(native_jk, dse_jk=dse_jk)

    cache = qed_sapt_jk.build_sapt_jk_cache(
        wfn_A,
        wfn_B,
        pf_jk,
        do_print=False,
        d_exp_el_A=driver.d_exp_el_A,
        d_exp_el_B=driver.d_exp_el_B,
        include_cavity_terms=driver.include_cavity_terms,
        nuclear_repulsion_energy=driver.nuc_rep,
    )

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
        "Ind20,u (A<-B)": float(ind["Ind20,u (A<-B)"]),
        "Ind20,u (A->B)": float(ind["Ind20,u (A->B)"]),
        "Ind20,u": float(ind["Ind20,u"]),
        "Exch-Ind20,u (A<-B)": float(ind["Exch-Ind20,u (A<-B)"]),
        "Exch-Ind20,u (A->B)": float(ind["Exch-Ind20,u (A->B)"]),
        "Exch-Ind20,u": float(ind["Exch-Ind20,u"]),
        "Ind20,r (A<-B)": float(ind["Ind20,r (A<-B)"]),
        "Ind20,r (A->B)": float(ind["Ind20,r (A->B)"]),
        "Ind20,r": float(ind["Ind20,r"]),
        "Exch-Ind20,r (A<-B)": float(ind["Exch-Ind20,r (A<-B)"]),
        "Exch-Ind20,r (A->B)": float(ind["Exch-Ind20,r (A->B)"]),
        "Exch-Ind20,r": float(ind["Exch-Ind20,r"]),
        "Exchange S^inf": float(exch["Exch10"]),
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

    print()
    print("Uncoupled induction diagnostics")
    print(f"{'Component':<25} {'Dense / Eh':>16} {'JK / Eh':>18} {'Delta / Eh':>14}")
    print("-" * 78)
    for name in UNCOUPLED_COMPONENT_ORDER:
        dense = result.dense_components[name]
        jk = result.jk_components[name]
        print(f"{name:<25} {dense:16.10f} {jk:18.10f} {jk - dense:14.6e}")

    print()
    print("Coupled induction diagnostics")
    print(f"{'Component':<25} {'Dense / Eh':>16} {'JK / Eh':>18} {'Delta / Eh':>14}")
    print("-" * 78)
    for name in COUPLED_COMPONENT_ORDER:
        dense = result.dense_components[name]
        jk = result.jk_components[name]
        print(f"{name:<25} {dense:16.10f} {jk:18.10f} {jk - dense:14.6e}")

    print()
    print(f"Exch10 S^inf (JK only): {result.jk_components['Exchange S^inf']: .10f} Eh")

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
