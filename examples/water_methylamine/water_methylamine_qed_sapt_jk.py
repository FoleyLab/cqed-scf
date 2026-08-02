"""Compare dense QED-SAPT0 reference components with JK/DSE-backed components.

This companion to ``water_methylamine_qed_sapt.py`` uses the same molecule and
cavity parameters, but evaluates the SAPT pieces currently available through
the local Psi4-JK helper path:

* electrostatics
* exchange
* induction
* exchange-induction

Dispersion and exchange-dispersion are not yet implemented in
``qed_sapt_jk.py``. They are carried from the dense reference values below so
the total row remains useful, and the output table labels their source.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import psi4

from cqed_scf import CQEDConfig
from cqed_scf.sapt import QEDSAPT0Driver
from cqed_scf.sapt import qed_sapt_jk
from cqed_scf.sapt.dse_jk import DSEJK, PauliFierzJK
import time


REFERENCE_COMPONENTS = {
    "Electrostatics": -0.0086737093,
    "Exchange": 0.0030805286,
    "Dispersion": -0.0027151107,
    "Exchange-Dispersion": 0.0001966590,
    "Induction": -0.0017887767,
    "Exchange-Induction": 0.0008167150,
}
REFERENCE_COMPONENTS["Total QED-SAPT0 Energy"] = sum(REFERENCE_COMPONENTS.values())


PSI4_OPTIONS = {
    "basis": "jun-cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}

CONFIG = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.1]),
    omega=0.1,
    psi4_options=PSI4_OPTIONS,
    reference="rhf",
    functional=None,
    density_fitting=False,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=False,
)

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


def _prepare_qed_monomers():
    """Run the same CQED monomer preparation used by the dense reference."""
    dimer_geometry = psi4.geometry(DIMER)
    driver = QEDSAPT0Driver(
        dimer_geometry=dimer_geometry,
        config=CONFIG,
        integral_backend="full_eri",
        include_cavity_terms=True,
    )
    driver.prepare_monomers()
    return driver


def _build_native_jk(wfn):
    """Build a native Psi4 JK object for the shared dimer AO basis."""
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
    """Return the dipole-projected AO operator shared by the two ghost bases."""
    d_ao_A = np.asarray(driver.monomer_A.d_ao)
    d_ao_B = np.asarray(driver.monomer_B.d_ao)
    if not np.allclose(d_ao_A, d_ao_B, atol=1e-12, rtol=1e-12):
        raise ValueError("Monomer DSE AO dipole operators differ in the shared dimer basis.")
    return d_ao_A


def _compute_jk_components(driver):
    """Compute the QED-SAPT0 pieces currently available through the JK path."""
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

    return {
        "Electrostatics": elst["Elst10,r"],
        "Exchange": exch["Exch10(S^2)"],
        "Induction": ind["Ind20,r"],
        "Exchange-Induction": ind["Exch-Ind20,r"],
    }

def _print_comparison(jk_components):
    components = dict(jk_components)
    components["Dispersion"] = REFERENCE_COMPONENTS["Dispersion"]
    components["Exchange-Dispersion"] = REFERENCE_COMPONENTS["Exchange-Dispersion"]
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

    sources = {
        "Electrostatics": "JK+DSE",
        "Exchange": "JK+DSE",
        "Dispersion": "dense ref",
        "Exchange-Dispersion": "dense ref",
        "Induction": "JK+DSE",
        "Exchange-Induction": "JK+DSE",
        "Total QED-SAPT0 Energy": "mixed",
    }

    print("Water-methylamine QED-SAPT0: dense reference vs JK/DSE path")
    print()
    print(f"{'Component':<25} {'Reference / Eh':>16} {'JK path / Eh':>16} {'Delta / Eh':>14} {'Source':>10}")
    print("-" * 85)
    for name in (
        "Electrostatics",
        "Exchange",
        "Dispersion",
        "Exchange-Dispersion",
        "Induction",
        "Exchange-Induction",
        "Total QED-SAPT0 Energy",
    ):
        reference = REFERENCE_COMPONENTS[name]
        actual = components[name]
        print(
            f"{name:<25} {reference:16.10f} {actual:16.10f} "
            f"{actual - reference:14.6e} {sources[name]:>10}"
        )

    print()
    print("Note: dispersion and exchange-dispersion are not yet implemented in the")
    print("local JK helper path, so those rows are carried from the dense reference.")


def main():
    psi4.core.be_quiet()
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(PSI4_OPTIONS)

    # CQEDSCF currently prints SCF progress via Python print; keep this example
    # focused on the component comparison table.
    with contextlib.redirect_stdout(io.StringIO()):
        driver = _prepare_qed_monomers()

    start = time.time()
    jk_components = _compute_jk_components(driver)
    _print_comparison(jk_components)
    end = time.time()
    print(F"JK/DSE path component evaluation took {end - start:.2f} seconds.")

if __name__ == "__main__":
    main()
