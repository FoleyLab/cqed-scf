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
    """ This unit test will test some of the core components of QED-SAPT driver, including:
        - testing slices of v, s,
        - testing the monomer classes behave as expected
        - running individual driver methods to compute energy components and comparing against expected output
        All expected output comes from 
    
    
    """
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
        actual_Eelst100 = driver.compute_Elst100()
        actual_Eexch100 = driver.compute_Exch100()
        actual_Edisp200 = driver.compute_Edisp200()
        actual_Eexchdisp200 = driver.compute_Eexchdisp200()
        actual_Eind200 = driver.compute_Eind200()
        actual_EexchInd200 = driver.compute_Eexchind200()

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
        expected_Eind200 = -1.246065163870e-04
        expected_ExchInd200 = 1.340706587265e-04

        assert actual_varbs.shape == expected_varbs.shape
        assert actual_sas.shape == expected_sas.shape
        assert actual_sbr.shape == expected_sbr.shape

        np.testing.assert_allclose(actual_varbs, expected_varbs, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(np.abs(actual_sas), np.abs(expected_sas), atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(np.abs(actual_sbr), np.abs(expected_sbr), atol=1e-7, rtol=1e-7)
        assert np.isclose(actual_Eelst100, expected_Elst100, atol=1e-9, rtol=1e-9)
        assert np.isclose(actual_Eexch100, expected_Exch100, atol=1e-9, rtol=1e-9)
        assert np.isclose(actual_Edisp200, expected_Edisp200, atol=1e-9, rtol=1e-9)
        assert np.isclose(actual_Eexchdisp200, expected_Eexchdisp200, atol=1e-9, rtol=1e-9)
        assert np.isclose(actual_Eind200, expected_Eind200, atol=1e-9, rtol=1e-9)
        assert np.isclose(actual_EexchInd200, expected_ExchInd200, atol=1e-9, rtol=1e-9)
    finally:
        psi4.core.clean()


def _water_water_sapt_config():
    psi4_options = {
        "basis": "jun-cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-12,
        "d_convergence": 1e-12,
    }

    return CQEDConfig(
        lambda_vector=np.array([0.0, 0.0, 0.0]),
        omega=0.0,
        psi4_options=psi4_options,
        reference="rhf",
        functional=None,
        density_fitting=False,
        charge=0,
        multiplicity=1,
        dispersion_policy="none",
        debug=False,
    )


def test_qedsapt0_driver_water_water():
    dimer = """
    O   -0.066999140   0.000000000   1.494354740
    H    0.815734270   0.000000000   1.865866390
    H    0.068855100   0.000000000   0.539142770
    --
    O    0.062547750   0.000000000  -1.422632080
    H   -0.406965400  -0.760178410  -1.771744500
    H   -0.406965400   0.760178410  -1.771744500
    symmetry c1
    """

    config = _water_water_sapt_config()

    psi4.core.clean()
    try:
        dimer_geometry = psi4.geometry(dimer)
        driver = QEDSAPT0Driver(
            dimer_geometry=dimer_geometry,
            config=config,
            integral_backend="full_eri",
        )

        driver.run()
        #Elst10: -0.013763635524
        expected_Elst10 = -1.376153034932e-02
        #Exch10: 0.010671445704
        expected_Exch10 = 1.067102949788e-02
        #Disp20: -0.002490829360
        expected_Disp20 = -2.490914697507e-03
        #Exch-Disp20: 0.000521337186
        expected_ExchDisp20 = 5.213383314441e-04
        #Ind20,r: -0.004267681293
        expected_Ind20 = -4.267259551623e-03
        #Exch-Ind20,r: 0.0023373573144267681293
        expected_ExchInd20 = 2.337176071937e-03
        #Total SAPT0: -0.006992005972
        expected_SAPT0 = -6.990160697192e-03

        assert np.isclose(driver.Eelst100, expected_Elst10, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.Eexch100, expected_Exch10, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.Edisp200, expected_Disp20, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.Eexchdisp200, expected_ExchDisp20, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.Eind200, expected_Ind20, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.Eexchind200, expected_ExchInd20, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.E_SAPT0, expected_SAPT0, atol=1e-9, rtol=1e-9)
        

    finally:
        psi4.core.clean()

def test_qedsapt0_driver_water_methanol():
    dimer = """
    O   -0.525329794  -0.050971084  -0.314516861
    H   -0.942006633   0.747901631   0.011252816
    H    0.403696525   0.059785981  -0.073568368
    --
    O    2.164777967   0.046634850   0.060313926
    H    2.532760791  -0.525442553   0.737842253
    C    2.629783038  -0.424995066  -1.201845184
    H    2.198965943   0.226098625  -1.954959216
    H    3.715746725  -0.374202205  -1.276157112
    H    2.301440420  -1.444864563  -1.400925818
    """

    config = _water_water_sapt_config()

    psi4.core.clean()
    try:
        dimer_geometry = psi4.geometry(dimer)
        driver = QEDSAPT0Driver(
            dimer_geometry=dimer_geometry,
            config=config,
            integral_backend="full_eri",
        )

        driver.run()
        
        expected_Elst100 = -2.022863936994e-02
        expected_Eexch100 = 2.181434270454e-02
        expected_Edisp200 = -5.073563201634e-03
        expected_Eexchdisp200 = 1.127969697164e-03
        expected_Eind200 = -9.309710760244e-03
        expected_Eexchind200 = 5.605149106206e-03
        expected_SAPT0 = -6.064451823912e-03
        assert np.isclose(driver.Eelst100, expected_Elst100, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.Eexch100, expected_Eexch100, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.Edisp200, expected_Edisp200, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.Eexchdisp200, expected_Eexchdisp200, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.Eind200, expected_Eind200, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.Eexchind200, expected_Eexchind200, atol=1e-9, rtol=1e-9)
        assert np.isclose(driver.E_SAPT0, expected_SAPT0, atol=1e-9, rtol=1e-9)


    finally:
        psi4.core.clean()