import numpy as np
import pytest
import psi4
import warnings

from cqed_scf import CQEDConfig
from cqed_scf.sapt import QEDSAPT0Driver, SAPTMonomer


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


_HE_DIMER = """
He 0.0000000000 0.0000000000 0.0000000000
--
He 0.0000000000 0.0000000000 2.0000000000
symmetry c1
units angstrom
no_reorient
no_com
"""


def _build_he_vt_driver(lambda_vector=None, include_cavity_terms=True):
    config = _he_sapt_config()
    if lambda_vector is not None:
        config = config.copy_with(lambda_vector=np.array(lambda_vector), omega=0.1)

    dimer_geometry = psi4.geometry(_HE_DIMER)
    driver = QEDSAPT0Driver(
        dimer_geometry=dimer_geometry,
        config=config,
        integral_backend="full_eri",
        include_cavity_terms=include_cavity_terms,
    )
    driver.prepare_monomers()
    driver.build_integrals()
    return driver


def _sum_vt_parts(parts):
    total = parts["eri"].copy()
    total += parts["potential_A"]
    total += parts["potential_B"]
    total += parts["constant"]
    return total




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

        assert monomers == (driver.monomer_A, driver.monomer_B)
        assert tuple(monomer.label for monomer in monomers) == ("monomer_A", "monomer_B")
        assert not isinstance(getattr(driver, "dimer", None), SAPTMonomer)
        assert driver.E_nuc_dimer is not None
        assert driver.dimer_mu_nuc.shape == (3,)
        assert driver.nuc_rep is not None

        driver.build_integrals()
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


def test_qedsapt0_driver_cavity_off_independent_of_no_cavity_backend():
    dimer = """
    He 0.0000000000 0.0000000000 0.0000000000
    --
    He 0.0000000000 0.0000000000 2.0000000000
    symmetry c1
    units angstrom
    no_reorient
    no_com
    """

    psi4.core.clean()
    try:
        dimer_geometry = psi4.geometry(dimer)
        config = _he_sapt_config()

        driver = QEDSAPT0Driver(
            dimer_geometry=dimer_geometry,
            config=config,
            integral_backend="full_eri",
            include_cavity_terms=False,
        )
        driver.run()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy_driver = QEDSAPT0Driver(
                dimer_geometry=dimer_geometry,
                config=config,
                integral_backend="no_cavity",
            )
        legacy_driver.run()

        assert legacy_driver.integral_backend == "full_eri"
        assert legacy_driver.include_cavity_terms is False
        np.testing.assert_allclose(driver.E_SAPT0, legacy_driver.E_SAPT0, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(driver.I_dimer, legacy_driver.I_dimer, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(driver.V_A, legacy_driver.V_A, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(driver.V_B, legacy_driver.V_B, atol=1e-12, rtol=1e-12)
    finally:
        psi4.core.clean()


def test_qedsapt0_driver_lambda_zero_cavity_on_matches_cavity_off():
    dimer = """
    He 0.0000000000 0.0000000000 0.0000000000
    --
    He 0.0000000000 0.0000000000 2.0000000000
    symmetry c1
    units angstrom
    no_reorient
    no_com
    """

    psi4.core.clean()
    try:
        dimer_geometry = psi4.geometry(dimer)
        config = _he_sapt_config()

        cavity_on = QEDSAPT0Driver(
            dimer_geometry=dimer_geometry,
            config=config,
            integral_backend="full_eri",
            include_cavity_terms=True,
        )
        cavity_on.run()

        cavity_off = QEDSAPT0Driver(
            dimer_geometry=dimer_geometry,
            config=config,
            integral_backend="full_eri",
            include_cavity_terms=False,
        )
        cavity_off.run()

        np.testing.assert_allclose(cavity_on.I_dimer, cavity_off.I_dimer, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(cavity_on.V_A, cavity_off.V_A, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(cavity_on.V_B, cavity_off.V_B, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(cavity_on.vt_nuc_rep, cavity_off.vt_nuc_rep, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(cavity_on.E_SAPT0, cavity_off.E_SAPT0, atol=1e-12, rtol=1e-12)
    finally:
        psi4.core.clean()


def test_qedsapt0_driver_vt_parts_sum_to_vt():
    psi4.core.clean()
    try:
        driver = _build_he_vt_driver(lambda_vector=[0.0, 0.0, 0.1])

        for string in ("abab", "abba", "abrs"):
            parts = driver.vt_parts(string)
            np.testing.assert_allclose(
                _sum_vt_parts(parts),
                driver.vt(string),
                atol=1e-12,
                rtol=1e-12,
            )
    finally:
        psi4.core.clean()


def test_qedsapt0_driver_vt_partitions_standard_cavity_total_relationship():
    psi4.core.clean()
    try:
        driver = _build_he_vt_driver(lambda_vector=[0.0, 0.0, 0.1])

        for string in ("abab", "abba", "abrs"):
            partitions = driver.vt_partitions(string)
            direct_cavity_parts = driver.vt_parts(string, context="cavity")
            for piece in ("eri", "potential_A", "potential_B", "constant", "total"):
                np.testing.assert_allclose(
                    partitions["standard"][piece] + partitions["cavity"][piece],
                    partitions["total"][piece],
                    atol=1e-12,
                    rtol=1e-12,
                )
                np.testing.assert_allclose(
                    partitions["cavity"][piece],
                    partitions["total"][piece] - partitions["standard"][piece],
                    atol=1e-12,
                    rtol=1e-12,
                )
                if piece != "total":
                    np.testing.assert_allclose(
                        direct_cavity_parts[piece],
                        partitions["cavity"][piece],
                        atol=1e-12,
                        rtol=1e-12,
                    )

            np.testing.assert_allclose(
                partitions["total"]["total"],
                driver.vt(string),
                atol=1e-12,
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                partitions["standard"]["total"],
                driver.vt(string, context="standard"),
                atol=1e-12,
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                partitions["cavity"]["total"],
                driver.vt(string, context="cavity"),
                atol=1e-12,
                rtol=1e-12,
            )

        elst_parts = driver.contract_vt_parts("abab", prefactor=4.0)
        elst_partitions = driver.vt_partitions("abab")
        for context in ("standard", "cavity", "total"):
            np.testing.assert_allclose(
                elst_parts[context]["total"],
                4.0 * np.einsum("abab->", elst_partitions[context]["total"]),
                atol=1e-12,
                rtol=1e-12,
            )
    finally:
        psi4.core.clean()


def test_qedsapt0_driver_lambda_zero_vt_cavity_parts_are_zero():
    psi4.core.clean()
    try:
        driver = _build_he_vt_driver(lambda_vector=[0.0, 0.0, 0.0])

        for string in ("abab", "abba", "abrs"):
            for value in driver.vt_partitions(string)["cavity"].values():
                np.testing.assert_allclose(value, np.zeros_like(value), atol=1e-12, rtol=1e-12)
    finally:
        psi4.core.clean()


def test_qedsapt0_driver_cavity_disabled_vt_standard_equals_total():
    psi4.core.clean()
    try:
        driver = _build_he_vt_driver(
            lambda_vector=[0.0, 0.0, 0.1],
            include_cavity_terms=False,
        )

        for string in ("abab", "abba", "abrs"):
            partitions = driver.vt_partitions(string)
            for piece in ("eri", "potential_A", "potential_B", "constant", "total"):
                np.testing.assert_allclose(
                    partitions["standard"][piece],
                    partitions["total"][piece],
                    atol=1e-12,
                    rtol=1e-12,
                )
                np.testing.assert_allclose(
                    partitions["cavity"][piece],
                    np.zeros_like(partitions["cavity"][piece]),
                    atol=1e-12,
                    rtol=1e-12,
                )
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


@pytest.mark.slow
def test_qedsapt0_driver_water_water_cavity_diagnostics():
    psi4_options = {
        "basis": "jun-cc-pVDZ",
        "scf_type": "pk",
        "e_convergence": 1e-12,
        "d_convergence": 1e-12,
    }

    config = CQEDConfig(
        lambda_vector=np.array([0.0, 0.0, 0.1]),
        omega=0.1,
        psi4_options=psi4_options,
        reference="rhf",
        functional=None,
        density_fitting=False,
        charge=0,
        multiplicity=1,
        dispersion_policy="none",
        debug=False,
    )

    dimer = """
    O   -0.066999140   0.000000000   1.494354740
    H    0.815734270   0.000000000   1.865866390
    H    0.068855100   0.000000000   0.539142770
    --
    O    0.062547750   0.000000000  -1.422632080
    H   -0.406965400  -0.760178410  -1.771744500
    H   -0.406965400   0.760178410  -1.771744500
    symmetry c1
    no_com
    no_reorient
    """

    psi4.core.clean()
    try:
        dimer_geometry = psi4.geometry(dimer)
        driver = QEDSAPT0Driver(
            dimer_geometry=dimer_geometry,
            config=config,
            integral_backend="full_eri",
            include_cavity_terms=True,
        )

        energy = driver.run()
        summary = driver.diagnostic_summary(print_output=False)
        elst100 = summary["vt"]["abab"]

        assert abs(summary["scalars"]["d_exp_A_residual"]) < 1e-10
        assert abs(summary["scalars"]["d_exp_B_residual"]) < 1e-10
        assert abs(elst100["checks"]["compute_Elst100_minus_diagnostic_total"]) < 1e-10
        assert abs(elst100["checks"]["standard_plus_cavity_minus_total"]) < 1e-10
        for piece in ("eri", "potential_A", "potential_B", "constant", "total"):
            assert piece in elst100["cavity"]
        assert abs(elst100["cavity"]["total"]) < 1e-10

        assert np.isfinite(energy)
        assert np.isfinite(driver.Eelst100)
        assert np.isfinite(driver.Eexch100)
        assert np.isfinite(driver.Edisp200)
        assert np.isfinite(driver.Eexchdisp200)
        assert np.isfinite(driver.Eind200)
        assert np.isfinite(driver.Eexchind200)
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
