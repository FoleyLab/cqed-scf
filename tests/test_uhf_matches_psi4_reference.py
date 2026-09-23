import numpy as np
import pytest

from cqed_scf import CQEDCalculator, CQEDConfig


def test_water_singlet_matches_psi4_reference():
    # Define water from psi4numpy tutorial example
    WATER_SINGLET_MOL_STR = """
    0 1
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """

    # Set options from psi4numpy tutorial example
    WATER_SINGLET_PSI4_OPTIONS = {'guess': 'core',
                    'basis': 'cc-pvdz',
                    'scf_type': 'pk',
                    'e_convergence': 1e-8,
                    'd_convergence': 5e-7}

    WATER_CONFIG = CQEDConfig(
        lambda_vector=np.array([0.0, 0.0, 0.0]),
        omega=0.0,
        psi4_options=WATER_SINGLET_PSI4_OPTIONS,
        reference="uhf",
        functional=None,
        density_fitting=False,
        charge=0,
        multiplicity=1,
        dispersion_policy="none",
        debug=False,
        quiet=True,  # SILENT: suppress all stdout (CQED-SCF + Psi4 engine output)
    )
        

    EXPECTED_WATER_SINGLET_ENERGY = -7.598979579E+01 # cc-pVDZ

    WATER_CALC = CQEDCalculator(config=WATER_CONFIG)
    WATER_ENERGY = WATER_CALC.energy(WATER_SINGLET_MOL_STR)
    assert np.isclose(WATER_ENERGY, EXPECTED_WATER_SINGLET_ENERGY, rtol=1e-6), f"Expected {EXPECTED_WATER_SINGLET_ENERGY}, but got {WATER_ENERGY}"

@pytest.mark.slow
def test_hydroxyl_doublet_matches_psi4_reference():

    HYDROXYL_DOUBLET_MOL_STR = """
    0 2
    O
    H 1 0.9697
    symmetry c1
    """
    HYDROXYL_CONFIG = CQEDConfig(
        lambda_vector=np.array([0.0, 0.0, 0.0]),
        omega=0.0,
        psi4_options={'basis': '6-311+G*', 'scf_type': 'pk', 'e_convergence': 1e-8, 'd_convergence': 5e-7},
        reference="uhf",
        functional=None,
        density_fitting=False,
        charge=0,
        multiplicity=2,
        dispersion_policy="none",
        debug=False,
        quiet=True,  # SILENT: suppress all stdout (CQED-SCF + Psi4 engine output)
    )

    EXPECTED_HYDROXYL_DOUBLET_ENERGY = -7.540608221E+01 # 6-311+G* 

    HYDROXYL_CALC = CQEDCalculator(config=HYDROXYL_CONFIG)
    HYDROXYL_ENERGY = HYDROXYL_CALC.energy(HYDROXYL_DOUBLET_MOL_STR)
    assert np.isclose(HYDROXYL_ENERGY, EXPECTED_HYDROXYL_DOUBLET_ENERGY, rtol=1e-6), f"Expected {EXPECTED_HYDROXYL_DOUBLET_ENERGY}, but got {HYDROXYL_ENERGY}"

def test_oxygen_triplet_matches_psi4_reference():

    OXYGEN_TRIPLET_MOL_STR = """
    0 3
    O
    O 1 1.210 
    symmetry c1
    """

    EXPECTED_OXYGEN_TRIPLET_ENERGY = -1.496590418E+02 # 6-311+G*

    OXYGEN_TRIPLET_CONFIG = CQEDConfig(
        lambda_vector=np.array([0.0, 0.0, 0.0]),
        omega=0.0,
        psi4_options={'basis': '6-311+G*', 'scf_type': 'pk', 'e_convergence': 1e-8, 'd_convergence': 5e-7},
        reference="uhf",
        functional=None,
        density_fitting=False,
        charge=0,
        multiplicity=3,
        dispersion_policy="none",
        debug=False,
        quiet=True,  # SILENT: suppress all stdout (CQED-SCF + Psi4 engine output)
    )

    OXYGEN_CALC = CQEDCalculator(config=OXYGEN_TRIPLET_CONFIG)
    OXYGEN_ENERGY = OXYGEN_CALC.energy(OXYGEN_TRIPLET_MOL_STR)
    assert np.isclose(OXYGEN_ENERGY, EXPECTED_OXYGEN_TRIPLET_ENERGY, rtol=1e-6), f"Expected {EXPECTED_OXYGEN_TRIPLET_ENERGY}, but got {OXYGEN_ENERGY}"



def test_imidogen_triplet_matches_psi4_reference():

    IMIDOGEN_TRIPLET_MOL_STR = """
    0 3
    N
    H 1 1.06
    symmetry c1
    """

    EXPECTED_IMIDOGEN_TRIPLET_ENERGY = -5.497317027E+01 # 6-311+G*

    IMIDOGEN_TRIPLET_CONFIG = CQEDConfig(
        lambda_vector=np.array([0.0, 0.0, 0.0]),
        omega=0.0,
        psi4_options={'basis': '6-311+G*', 'scf_type': 'pk', 'e_convergence': 1e-8, 'd_convergence': 5e-6},
        reference="uhf",
        functional=None,
        density_fitting=False,
        charge=0,
        multiplicity=3,
        dispersion_policy="none",
        debug=False,
        quiet=True,  # SILENT: suppress all stdout (CQED-SCF + Psi4 engine output)
    )

    IMIDOGEN_CALC = CQEDCalculator(config=IMIDOGEN_TRIPLET_CONFIG)
    IMIDOGEN_ENERGY = IMIDOGEN_CALC.energy(IMIDOGEN_TRIPLET_MOL_STR)
    assert np.isclose(IMIDOGEN_ENERGY, EXPECTED_IMIDOGEN_TRIPLET_ENERGY, rtol=1e-6), f"Expected {EXPECTED_IMIDOGEN_TRIPLET_ENERGY}, but got {IMIDOGEN_ENERGY}"
