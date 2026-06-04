import numpy as np
import psi4

from cqed_scf import CQEDConfig
from cqed_scf.sapt import QEDSAPT0Driver


def _he_sapt_config():
    psi4_options = {
        "basis": "6-31g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-8,
    }

    return CQEDConfig(
        lambda_vector=np.array([0.0, 0.0, 0.0]),
        omega=0.07349864501573,
        psi4_options=psi4_options,
        reference="rhf",
        functional=None,
        density_fitting=False,
        charge=0,
        multiplicity=1,
        dispersion_policy="none",
        debug=False,
    )




def test_qedsapt0_driver_auto_extract_he_dimer_v_arbs():
    dimer = """
    He 0.0000000000 0.0000000000 0.0000000000
    --
    He 0.0000000000 0.0000000000 2.0000000000
    symmetry c1
    units angstrom
    no_reorient
    no_com
    """

    config = _he_sapt_config()

    psi4.core.clean()
    try:
        dimer_geometry = psi4.geometry(dimer)
        driver = QEDSAPT0Driver(
            dimer_geometry=dimer_geometry,
            config=config,
            integral_backend="full_eri",
        )

        monomers = driver.prepare_monomers()

        assert monomers[0] is driver.dimer
        assert monomers[1] is driver.monomer_A
        assert monomers[2] is driver.monomer_B
        assert tuple(monomer.label for monomer in monomers) == (
            "dimer",
            "monomer_A",
            "monomer_B",
        )

        driver.build_integrals(monomers)
        actual_varbs = driver.v("arbs")
        actual_sas = driver.s("as")
        actual_sbr = driver.s("br")
        actual_Elst100 = driver.compute_Elst100()
        actual_Exch100 = driver.compute_Exch100()
        actual_Edisp200 = driver.compute_Edisp200()
        actual_Eexchdisp200 = driver.compute_Eexchdisp200()

        expected_varbs = np.array(
            [
                [
                    [[-0.00132318, -0.01021403, -0.00274152]],
                    [[-0.01021403, -0.00095427, -0.02294565]],
                    [[-0.00274152, -0.02294565, 0.00143705]],
                ]
            ]
        )

        expected_sas = np.array(
            [
                [-0.855236157,  0.078947881,  0.509439526],
            ]
        )
        expected_sbr = np.array(
            [
                [-0.855236157,  0.078947881, -0.509439526],
            ]
        )
        expected_Elst100 = -5.213924649057e-04
        expected_Exch100 = 2.863988537340e-03
        expected_Edisp200 = -1.642253534355e-05
        expected_Eexchdisp200 = -6.875724637967e-06

        assert actual_varbs.shape == expected_varbs.shape
        assert actual_sas.shape == expected_sas.shape
        assert actual_sbr.shape == expected_sbr.shape

        np.testing.assert_allclose(actual_varbs, expected_varbs, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(np.abs(actual_sas), np.abs(expected_sas), atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(np.abs(actual_sbr), np.abs(expected_sbr), atol=1e-7, rtol=1e-7)
        assert np.isclose(actual_Elst100, expected_Elst100, atol=1e-9, rtol=1e-9)
        assert np.isclose(actual_Exch100, expected_Exch100, atol=1e-9, rtol=1e-9)
        assert np.isclose(actual_Edisp200, expected_Edisp200, atol=1e-9, rtol=1e-9)
        assert np.isclose(actual_Eexchdisp200, expected_Eexchdisp200, atol=1e-9, rtol=1e-9)
    finally:
        psi4.core.clean()


def _he_h2_sapt_config():
    psi4_options = {
        "basis": "6-31g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-8,
    }

    return CQEDConfig(
        lambda_vector=np.array([0.0, 0.0, 1.0]),
        omega=0.07349864501573,
        psi4_options=psi4_options,
        reference="rhf",
        functional=None,
        density_fitting=False,
        charge=0,
        multiplicity=1,
        dispersion_policy="none",
        debug=False,
    )


def test_qedsapt0_driver_he_h2():
    dimer = """
    He 0.0000000000 0.0000000000 0.0000000000
    --
    Li  0.0000000000 0.0000000000 2.0000000000
    H  0.0000000000 0.0000000000 3.0000000000
    symmetry c1
    units angstrom
    no_reorient
    no_com
    """

    config = _he_h2_sapt_config()

    psi4.core.clean()
    try:
        dimer_geometry = psi4.geometry(dimer)
        driver = QEDSAPT0Driver(
            dimer_geometry=dimer_geometry,
            config=config,
            integral_backend="full_eri",
        )

        monomers = driver.prepare_monomers()

        assert monomers[0] is driver.dimer
        assert monomers[1] is driver.monomer_A
        assert monomers[2] is driver.monomer_B
        assert tuple(monomer.label for monomer in monomers) == (
            "dimer",
            "monomer_A",
            "monomer_B",
        )

        driver.build_integrals(monomers)

    finally:
        psi4.core.clean()
