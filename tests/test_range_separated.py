import numpy as np
import psi4
import pytest

from cqed_rhf.calculator import CQEDRHFCalculator



geometry = """
0 1
O
H 1 0.9572
H 1 0.9572 2 104.5
symmetry c1
"""
psi4_options = {
    "basis": "6-311g",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

def test_h2o_wb97x_d_lambda0():

    psi4.core.clean()
    psi4.core.clean_options()

    psi4.set_memory("4 GB")
    psi4.set_options(psi4_options)
    lambda_vector = np.zeros(3)

    calc = CQEDRHFCalculator(
        lambda_vector=lambda_vector,
        psi4_options=psi4_options,
        omega=0.1,
        charge=0,
        multiplicity=1,
        density_fitting=True,
        functional="wb97x-d",
        debug=True,
    )

    E, grad, _ = calc.energy_and_gradient(geometry)

    # --- reset AGAIN before reference ---
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(psi4_options)

    mol = psi4.geometry(geometry)

    E_ref = psi4.energy("wb97x-d")
    grad_ref = psi4.gradient("wb97x-d")

    assert abs(E - E_ref) < 1e-10
    assert np.max(np.abs(grad - grad_ref)) < 1e-6

def test_h2o_wb97x_lambda0():

    psi4.core.clean()
    psi4.core.clean_options()

    psi4.set_memory("4 GB")
    psi4.set_options(psi4_options)
    lambda_vector = np.zeros(3)

    calc = CQEDRHFCalculator(
        lambda_vector=lambda_vector,
        psi4_options=psi4_options,
        omega=0.1,
        charge=0,
        multiplicity=1,
        density_fitting=True,
        functional="wb97x",
        debug=True,
    )

    E, grad, _ = calc.energy_and_gradient(geometry)

    # --- reset AGAIN before reference ---
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(psi4_options)

    mol = psi4.geometry(geometry)

    E_ref = psi4.energy("wb97x")
    grad_ref = psi4.gradient("wb97x")

    assert abs(E - E_ref) < 1e-10
    assert np.max(np.abs(grad - grad_ref)) < 1e-6

def test_mghplus_wb97x_lambda0():

    geometry = """
    1 1
    Mg
    H 1 1.4
    symmetry c1
    """

    psi4.core.clean()
    psi4.core.clean_options()

    psi4.set_memory("4 GB")
    psi4.set_options(psi4_options)
    lambda_vector = np.zeros(3)

    calc = CQEDRHFCalculator(
        lambda_vector=lambda_vector,
        psi4_options=psi4_options,
        omega=0.1,
        charge=0,
        multiplicity=1,
        density_fitting=True,
        functional="wb97x",
        debug=True,
    )

    E, grad, _ = calc.energy_and_gradient(geometry)

    # --- reset AGAIN before reference ---
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(psi4_options)

    mol = psi4.geometry(geometry)

    E_ref = psi4.energy("wb97x")
    grad_ref = psi4.gradient("wb97x")

    assert abs(E - E_ref) < 1e-10
    assert np.max(np.abs(grad - grad_ref)) < 1e-6
