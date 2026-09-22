import numpy as np
import pytest

from cqed_scf import CQEDCalculator, CQEDConfig

# same series of lambda vectors used for all tests in test_qed_uhf_matches_hilbert_reference.py
LAMBDA_VECTORS = {
    "zero" : np.array([0.0, 0.0, 0.0]),
    "z_only" : np.array([0.0, 0.0, 0.05]),
    "x_only" : np.array([0.05, 0.0, 0.0]),
    "general" : np.array([0.02, 0.03, 0.05]),
}

# oh radical first
def test_hydroxyl_doublet_matches_hilbert_reference():

    HYDROXYL_DOUBLET_MOL_STR = """
    0 2
    O     0.000000     0.000000     0.000000
    H     0.000000     0.000000     0.969300
    no_reorient
    no_com
    units angstrom
    symmetry c1
    """

    # Collect the hilbert energies at the 6-311++G** level of theory for the hydroxyl doublet with different lambda vectors
    EXPECTED_HYDROXYL_DOUBLET_ENERGY = {
        "zero": -75.414526952824,
        "z_only": -75.409799160801,
        "x_only": -75.410652176375,
        "general": -75.407743988861
    }


    # Loop over the different lambda vectors, build config, calculate energy, and assert that it matches the expected value
    for lambda_name, lambda_vector in LAMBDA_VECTORS.items():
        config = CQEDConfig(
            lambda_vector=lambda_vector,
            omega=0.0,
            psi4_options={'basis': '6-311++G**', 'scf_type': 'pk', 'e_convergence': 1e-8, 'd_convergence': 5e-7},
            reference="uhf",
            functional=None,
            density_fitting=False,
            charge=0,
            multiplicity=2,
            dispersion_policy="none",
            debug=False,
            quiet=True,  # SILENT: suppress all stdout (CQED-SCF + Psi4 engine output)
        )

        calc = CQEDCalculator(config=config)
        energy = calc.energy(HYDROXYL_DOUBLET_MOL_STR)

        expected_energy = EXPECTED_HYDROXYL_DOUBLET_ENERGY[lambda_name]
        assert np.isclose(energy, expected_energy, rtol=1e-6), f"Expected {expected_energy} for lambda '{lambda_name}', but got {energy}"
    

# oxygen triplet next
def test_oxygen_triplet_matches_hilbert_reference():

    OXYGEN_TRIPLET_MOL_STR = """
    0 3
    O     0.000000     0.000000     0.000000
    O     0.000000     0.000000     1.207520
    no_reorient
    no_com
    units angstrom
    symmetry c1
    """

    # Collect the hilbert energies at the 6-311++G** level of theory for the oxygen triplet with different lambda vectors
    EXPECTED_OXYGEN_TRIPLET_ENERGY = {
        "zero": -149.659551073945,
        "z_only": -149.650195876160,
        "x_only": -149.653447107661,
        "general": -149.647034901405
    }


    # Loop over the different lambda vectors, build config, calculate energy, and assert that it matches the expected value
    for lambda_name, lambda_vector in LAMBDA_VECTORS.items():
        config = CQEDConfig(
            lambda_vector=lambda_vector,
            omega=0.0,
            psi4_options={'basis': '6-311++G**', 'scf_type': 'pk', 'e_convergence': 1e-8, 'd_convergence': 5e-7},
            reference="uhf",
            functional=None,
            density_fitting=False,
            charge=0,
            multiplicity=2,
            dispersion_policy="none",
            debug=False,
            quiet=True,  # SILENT: suppress all stdout (CQED-SCF + Psi4 engine output)
        )

        calc = CQEDCalculator(config=config)
        energy = calc.energy(OXYGEN_TRIPLET_MOL_STR)

        expected_energy = EXPECTED_OXYGEN_TRIPLET_ENERGY[lambda_name]
        assert np.isclose(energy, expected_energy, rtol=1e-6), f"Expected {expected_energy} for lambda '{lambda_name}', but got {energy}"
    


# nh triplet next
def test_imidogen_triplet_matches_hilbert_reference():
    IMIDOGEN_TRIPLET_MOL_STR = """
    0 3
    N     0.000000     0.000000     0.000000
    H     0.000000     0.000000     1.036200
    no_reorient
    no_com
    units angstrom
    symmetry c1
    """

    # Collect the hilbert energies at the 6-311++G** level of theory for the imidogen triplet with different lambda vectors
    EXPECTED_IMIDOGEN_TRIPLET_ENERGY = {
        "zero": -54.978041641531,  
        "z_only": -54.972868426137,  
        "x_only": -54.973719375981,  
        "general": -54.970631182269  
    }


    # Loop over the different lambda vectors, build config, calculate energy, and assert that it matches the expected value
    for lambda_name, lambda_vector in LAMBDA_VECTORS.items():
        config = CQEDConfig(
            lambda_vector=lambda_vector,
            omega=0.0,
            psi4_options={'basis': '6-311++G**', 'scf_type': 'pk', 'e_convergence': 1e-8, 'd_convergence': 5e-7},
            reference="uhf",
            functional=None,
            density_fitting=False,
            charge=0,
            multiplicity=2,
            dispersion_policy="none",
            debug=False,
            quiet=True,  # SILENT: suppress all stdout (CQED-SCF + Psi4 engine output)
        )

        calc = CQEDCalculator(config=config)
        energy = calc.energy(IMIDOGEN_TRIPLET_MOL_STR)

        expected_energy = EXPECTED_IMIDOGEN_TRIPLET_ENERGY[lambda_name]
        assert np.isclose(energy, expected_energy, rtol=1e-6), f"Expected {expected_energy} for lambda '{lambda_name}', but got {energy}"