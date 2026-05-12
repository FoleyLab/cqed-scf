import numpy as np
import psi4
import pytest

from cqed_scf import CQEDCalculator
from cqed_scf.drivers import bfgs_optimize
from cqed_scf.utils import ANGSTROM_TO_BOHR


@pytest.mark.slow
def test_lambda_zero_reproduces_psi4_rhf():
    """
    Regression test:
    CQED-RHF with lambda = 0 must reproduce
    Psi4 RHF optimized energy and geometry.
    """

    # =========================
    # Initial geometry (angstrom)
    # =========================
    h2o_string = """
    O  0.000000000000   0.000000000000  -0.068516219320
    H  0.000000000000  -0.790689573744   0.543701060715
    H  0.000000000000   0.790689573744   0.543701060715
    units angstrom
    no_reorient
    no_com
    symmetry c1
    """

    # =========================
    # Psi4 options
    # =========================
    psi4_options = {
        "basis": "cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-12,
        "d_convergence": 1e-12,
        "geom_maxiter": 50,
        "g_convergence": "gau_verytight"
    }

    psi4.set_options(psi4_options)
    psi4.core.clean()

    # =========================
    # Psi4 reference optimization
    # =========================
    mol_ref = psi4.geometry(h2o_string)

    E_ref, wfn_ref = psi4.optimize(
        "scf",
        molecule=mol_ref,
        return_wfn=True,
    )

    coords_ref_bohr = mol_ref.geometry().to_array()
    coords_ref_angstrom = coords_ref_bohr * psi4.constants.bohr2angstroms

    symbols = [mol_ref.symbol(i) for i in range(mol_ref.natom())]

    psi4.core.clean()

    # =========================
    # CQED-RHF optimization (lambda = 0)
    # =========================
    lambda_vector = [0.0, 0.0, 0.0]
    omega = 0.1  # irrelevant when lambda = 0

    calc = CQEDCalculator(
        lambda_vector=lambda_vector,
        psi4_options=psi4_options,
        omega=omega,
    )

    opt_cqed, _od = bfgs_optimize(
        calculator=calc,
        geometry=h2o_string,
        canonical="exact",
        gtol=1e-6,
        maxiter=50,
        debug=False,
    )

    coords_cqed_bohr = opt_cqed.x.reshape(-1, 3)
    coords_cqed_angstrom = coords_cqed_bohr / ANGSTROM_TO_BOHR

    # =========================
    # Energy comparison
    # =========================
    energy_diff = abs(opt_cqed.fun - E_ref)

    assert energy_diff < 1e-7, (
        f"Energy mismatch too large: {energy_diff:.3e} Ha"
    )

    # =========================
    # Geometry comparison
    # =========================
    geom_diff = coords_cqed_angstrom - coords_ref_angstrom

    max_diff = np.max(np.abs(geom_diff))
    rms_diff = np.sqrt(np.mean(geom_diff**2))

    assert max_diff < 1e-5, (
        f"Max geometry difference too large: {max_diff:.3e} Å"
    )

    assert rms_diff < 1e-6, (
        f"RMS geometry difference too large: {rms_diff:.3e} Å"
    )

