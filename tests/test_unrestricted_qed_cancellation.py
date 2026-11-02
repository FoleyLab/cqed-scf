"""Numerical cancellation gate for the explicit unrestricted QED terms.

These tests intentionally reconstruct both forms on the same converged QED-UHF
density.  They must remain in place before ``CQEDUSCF`` drops the cancelling
coherent-state one-electron shift, DSE Coulomb term, and scalar energy.
"""

import importlib.util
from pathlib import Path

import numpy as np
import psi4
import pytest

from cqed_scf import CQEDConfig
from cqed_scf import output
from cqed_scf.uscf import CQEDUSCF

REFERENCE_DIR = (
    Path(__file__).parents[1] / "examples" / "canonical" / "Unrestricted_Test_Examples"
)


def _load_system_definitions():
    spec = importlib.util.spec_from_file_location(
        "unrestricted_cancellation_systems", REFERENCE_DIR / "systems.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYSTEMS = _load_system_definitions()


def _q_pf(mints, lambda_vector):
    q = [np.asarray(matrix) for matrix in mints.ao_quadrupole()]
    lx, ly, lz = lambda_vector
    return (
        -0.5 * lx**2 * q[0]
        - lx * ly * q[1]
        - lx * lz * q[2]
        - 0.5 * ly**2 * q[3]
        - ly * lz * q[4]
        - 0.5 * lz**2 * q[5]
    )


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(("oh_radical", "z_only"), id="oh-doublet-z"),
        pytest.param(("nh_triplet", "general"), id="nh-triplet-general"),
    ],
)
def cancellation_data(request):
    system_name, lambda_name = request.param
    system = SYSTEMS.SYSTEMS[system_name]
    lambda_vector = SYSTEMS.LAMBDA_VECTORS[lambda_name]
    psi4_options = SYSTEMS.base_psi4_options("uhf")
    psi4_options.pop("reference")

    psi4.core.clean()
    psi4.core.clean_options()
    config = CQEDConfig(
        lambda_vector=lambda_vector,
        omega=SYSTEMS.OMEGA,
        psi4_options=psi4_options,
        reference="uhf",
        density_fitting=False,
        charge=system["charge"],
        multiplicity=system["multiplicity"],
        dispersion_policy="none",
        quiet=True,
    )
    engine = CQEDUSCF(system["geometry"], config)
    with output.quiet_context(True):
        _, results = engine.run()

    Da = results["Da"]
    Db = results["Db"]
    Dt = Da + Db
    Ca_occ = results["Ca"][:, : results["nalpha"]]
    Cb_occ = results["Cb"][:, : results["nbeta"]]

    mints = psi4.core.MintsHelper(engine.wfn.basisset())
    dipoles = [np.asarray(matrix) for matrix in mints.ao_dipole()]
    d = sum(component * matrix for component, matrix in zip(lambda_vector, dipoles))
    x = float(np.trace(d @ Dt))

    H = (
        np.asarray(mints.ao_kinetic())
        + np.asarray(mints.ao_potential())
        + _q_pf(mints, lambda_vector)
    )
    H_shift = -x * d
    J_DSE_a = d * np.trace(d @ Da)
    J_DSE_b = d * np.trace(d @ Db)
    J_DSE = J_DSE_a + J_DSE_b
    N_a = d @ Da @ d
    N_b = d @ Db @ d

    # run() finalizes its Psi4 builders; create a fresh JK object for this
    # independent reconstruction of the converged Focks.
    engine._build_jk()
    J_a, J_b, K_a, K_b, _, _ = engine._build_JK(Ca_occ, Cb_occ)
    engine._finalize_builders()
    J_total = J_a + J_b

    F_a_explicit = H + H_shift + J_total - K_a + J_DSE - N_a
    F_b_explicit = H + H_shift + J_total - K_b + J_DSE - N_b
    F_a_simplified = H + J_total - K_a - N_a
    F_b_simplified = H + J_total - K_b - N_b

    E_core_shift = float(np.trace(H_shift @ Dt))
    E_J_DSE = float(0.5 * np.trace(J_DSE @ Da) + 0.5 * np.trace(J_DSE @ Db))
    E_scalar = 0.5 * x**2

    E_nuc = engine.mol.nuclear_repulsion_energy()
    E_explicit = float(
        0.5 * np.trace((H + H_shift) @ Dt)
        + 0.5 * np.trace(F_a_explicit @ Da)
        + 0.5 * np.trace(F_b_explicit @ Db)
        + E_nuc
        + E_scalar
    )
    E_simplified = float(
        0.5 * np.trace(H @ Dt)
        + 0.5 * np.trace(F_a_simplified @ Da)
        + 0.5 * np.trace(F_b_simplified @ Db)
        + E_nuc
    )

    return {
        "x": x,
        "H_shift": H_shift,
        "J_DSE": J_DSE,
        "E_core_shift": E_core_shift,
        "E_J_DSE": E_J_DSE,
        "E_scalar": E_scalar,
        "F_a_explicit": F_a_explicit,
        "F_b_explicit": F_b_explicit,
        "F_a_simplified": F_a_simplified,
        "F_b_simplified": F_b_simplified,
        "E_explicit": E_explicit,
        "E_simplified": E_simplified,
    }


def test_individual_cancelling_terms_are_nonzero(cancellation_data):
    data = cancellation_data

    assert abs(data["x"]) > 1.0e-6
    assert np.linalg.norm(data["H_shift"]) > 1.0e-6
    assert np.linalg.norm(data["J_DSE"]) > 1.0e-6
    assert abs(data["E_core_shift"]) > 1.0e-8
    assert abs(data["E_J_DSE"]) > 1.0e-8
    assert abs(data["E_scalar"]) > 1.0e-8


def test_unrestricted_dse_fock_cancellation(cancellation_data):
    data = cancellation_data

    np.testing.assert_allclose(
        data["H_shift"] + data["J_DSE"],
        np.zeros_like(data["H_shift"]),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_unrestricted_dse_energy_cancellation_term_by_term(cancellation_data):
    data = cancellation_data
    x_squared = data["x"] ** 2

    assert data["E_core_shift"] == pytest.approx(-x_squared, abs=1.0e-12)
    assert data["E_J_DSE"] == pytest.approx(0.5 * x_squared, abs=1.0e-12)
    assert data["E_scalar"] == pytest.approx(0.5 * x_squared, abs=1.0e-12)
    assert (data["E_core_shift"] + data["E_J_DSE"] + data["E_scalar"]) == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_explicit_and_simplified_focks_and_energies_agree(cancellation_data):
    data = cancellation_data

    np.testing.assert_allclose(
        data["F_a_explicit"], data["F_a_simplified"], atol=1.0e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        data["F_b_explicit"], data["F_b_simplified"], atol=1.0e-12, rtol=0.0
    )
    assert data["E_explicit"] == pytest.approx(
        data["E_simplified"], abs=1.0e-12, rel=0.0
    )
