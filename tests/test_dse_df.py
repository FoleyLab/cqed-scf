"""Phase 0/1 regression tests: DF-factorized Pauli-Fierz integrals.

The sharp test here is the asymmetry in tolerances.  Standard blocks carry
ordinary density-fitting error, but the DSE contribution is represented by a
*single exact* auxiliary row, so cavity blocks must agree with the dense
reference to machine precision.  A cavity block that only agrees to DF error
would mean the augmentation is wrong.
"""

import numpy as np
import psi4
import pytest

from cqed_scf import CQEDConfig
from cqed_scf.sapt import QEDSAPT0Driver, PauliFierzDF, build_df_ao_tensor
from cqed_scf.sapt import qed_sapt_jk
from cqed_scf.sapt.dse_jk import DSEJK, PauliFierzJK


BASIS = "cc-pvdz"

# Water + He: small enough for a dense N^4 reference, but with no symmetry and
# unequal monomers, so a transposed-pair convention error cannot pass unnoticed.
_WATER_HE = """
0 1
O 0.0 0.0 0.0
H 0.0 0.757 0.587
H 0.0 -0.757 0.587
--
0 1
He 0.0 0.0 3.4
symmetry c1
units angstrom
no_reorient
no_com
"""

# Every four-character string used by qed_sapt0.py, plus the two the tests use.
_ALL_STRINGS = (
    "abaa abab abar abas abba abbb abbr abbs abra abrb abrs absa absb absr "
    "rbab rsab saba arbs"
).split()


def _config(lambda_vector, basis=BASIS, convergence=1e-10):
    return CQEDConfig(
        lambda_vector=np.array(lambda_vector),
        omega=0.1,
        psi4_options={
            "basis": basis,
            "scf_type": "pk",
            "e_convergence": convergence,
            "d_convergence": convergence,
        },
        reference="rhf",
        functional=None,
        density_fitting=False,
        charge=0,
        multiplicity=1,
        dispersion_policy="none",
        debug=False,
    )


def _driver(lambda_vector=(0.0, 0.0, 0.1), include_cavity_terms=True, geometry=_WATER_HE):
    psi4.core.be_quiet()
    psi4.core.clean()
    driver = QEDSAPT0Driver(
        dimer_geometry=psi4.geometry(geometry),
        config=_config(lambda_vector),
        integral_backend="full_eri",
        include_cavity_terms=include_cavity_terms,
    )
    driver.prepare_monomers()
    driver.build_integrals()
    return driver


@pytest.fixture(scope="module")
def cavity_driver():
    return _driver()


@pytest.fixture(scope="module")
def cavity_df(cavity_driver):
    return PauliFierzDF.from_driver(cavity_driver, aux_basis_name=BASIS)


# --------------------------------------------------------------------------
# Phase 1: the factorization itself
# --------------------------------------------------------------------------


def test_df_ao_tensor_reproduces_exact_eris_to_fitting_error():
    """Anchors the (aux, zero, primary, primary) BraKet ordering and the metric.

    A wrong BraKet order raises; a wrong metric power gives errors orders of
    magnitude larger than the rms bound below.  The bound is deliberately on
    the *rms*: the max error over individual AO integrals is dominated by
    centres with sparse auxiliary sets (here the He atom, ~1e-1) and is a poor
    proxy for the accuracy of the contracted quantities we actually want.  The
    accuracy that matters is checked directly against Psi4's own SAPT0 in
    ``test_lambda_zero_dispersion_matches_psi4_sapt0``.
    """
    driver = _driver()
    wfn = driver.monomer_A.wfn
    aux = psi4.core.BasisSet.build(wfn.molecule(), "DF_BASIS_SAPT", "", "RIFIT", BASIS)

    B = build_df_ao_tensor(wfn.basisset(), aux)
    I_df = np.einsum("Ppq,Prs->pqrs", B, B, optimize=True)
    I_exact = np.asarray(psi4.core.MintsHelper(wfn.basisset()).ao_eri())

    assert np.sqrt(((I_df - I_exact) ** 2).mean()) < 1e-3

    # The fit is a genuine symmetric low-rank factorization, not an accident of
    # index ordering: sum_Q B^Q_pq B^Q_rs is symmetric under (pq) <-> (rs) and
    # under p <-> q by construction.
    np.testing.assert_allclose(I_df, I_df.transpose(2, 3, 0, 1), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(I_df, I_df.transpose(1, 0, 2, 3), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("string", _ALL_STRINGS)
def test_cavity_blocks_are_exact_under_the_single_auxiliary_row(string, cavity_driver, cavity_df):
    """The DSE row carries no fitting error, so this must hold to roundoff.

    This is the load-bearing assertion of the whole DF design: if the cavity
    kernel were not exactly rank one, or the pair assignment (0,2)/(1,3) were
    wrong, this would degrade to DF-level error rather than machine precision.
    """
    reference = cavity_driver.v(string, context="cavity")
    computed = cavity_df.v(string, context="cavity")

    assert computed.shape == reference.shape
    scale = max(np.abs(reference).max(), 1.0)
    assert np.abs(computed - reference).max() < 1e-12 * scale


@pytest.mark.parametrize("string", _ALL_STRINGS)
def test_standard_blocks_match_dense_to_fitting_error(string, cavity_driver, cavity_df):
    reference = cavity_driver.v(string, context="standard")
    computed = cavity_df.v(string, context="standard")

    assert computed.shape == reference.shape
    assert np.abs(computed - reference).max() < 1e-3


@pytest.mark.parametrize("string", _ALL_STRINGS)
def test_total_equals_standard_plus_cavity_by_auxiliary_partition(string, cavity_df):
    """standard/cavity/total are slices of one auxiliary index, so this is exact."""
    total = cavity_df.v(string, context="total")
    standard = cavity_df.v(string, context="standard")
    cavity = cavity_df.v(string, context="cavity")

    np.testing.assert_allclose(total, standard + cavity, rtol=0.0, atol=1e-13)


def test_v_is_derived_from_two_pair_blocks(cavity_df):
    """Pins the pair assignment: positions (0,2) and (1,3), not (0,1) and (2,3)."""
    string = "abrs"
    left = cavity_df.b(string[0] + string[2])   # (a, r)
    right = cavity_df.b(string[1] + string[3])  # (b, s)

    assert left.shape[1:] == (cavity_df.b("aa").shape[1], cavity_df.b("rr").shape[2])
    np.testing.assert_allclose(
        cavity_df.v(string),
        np.einsum("PAC,PBD->ABCD", left, right, optimize=True),
        rtol=0.0,
        atol=1e-13,
    )


def test_pair_blocks_are_read_only_views(cavity_df):
    """The three contexts overlap in one tensor; writing would couple them.

    Guards the mechanism, per docs/development/psi4_array_aliasing.md, rather
    than waiting for a physical consequence to look wrong.
    """
    block = cavity_df.b("ar")
    assert not block.flags.writeable
    with pytest.raises(ValueError):
        block[0, 0, 0] = 1.0


def test_cross_monomer_pairs_are_available(cavity_df):
    """Exchange-dispersion needs ab / as / br / sb, not just same-monomer pairs."""
    for pair in ("ab", "as", "br", "sb", "ra", "bs"):
        assert cavity_df.b(pair).ndim == 3


def test_b_rejects_bad_pairs_and_contexts(cavity_df):
    with pytest.raises(ValueError):
        cavity_df.b("a")
    with pytest.raises(ValueError):
        cavity_df.b("ax")
    with pytest.raises(ValueError):
        cavity_df.b("ar", context="bogus")
    with pytest.raises(ValueError):
        cavity_df.v("abr")


def test_disabled_cavity_has_no_extra_auxiliary_row():
    driver = _driver(include_cavity_terms=False)
    df = PauliFierzDF.from_driver(driver, aux_basis_name=BASIS)

    assert not df.cavity_is_active
    assert df.b("ar", context="cavity").shape[0] == 0
    # An empty auxiliary slice contracts to exact zeros with no special case.
    assert np.abs(df.v("abrs", context="cavity")).max() == 0.0
    np.testing.assert_allclose(
        df.v("abrs", context="total"), df.v("abrs", context="standard"), rtol=0.0, atol=0.0
    )


def test_from_driver_rejects_asymmetric_dipole_kernel(cavity_driver):
    """d_A == d_B is load-bearing; a mismatch must fail loudly, not silently."""
    perturbed = np.array(cavity_driver.d_B, copy=True)
    perturbed[0, 0] += 1.0e-3

    original = cavity_driver.d_B
    try:
        cavity_driver.d_B = perturbed
        with pytest.raises(RuntimeError, match="d_A == d_B"):
            PauliFierzDF.from_driver(cavity_driver, aux_basis_name=BASIS)
    finally:
        cavity_driver.d_B = original


# --------------------------------------------------------------------------
# Phase 0: audit fixes
# --------------------------------------------------------------------------


def test_disabled_cavity_allocates_no_cavity_eri_tensor():
    """Audit item A.1: nbf**4 doubles must not be spent representing zero."""
    driver = _driver(include_cavity_terms=False)

    assert driver.I_dimer_cavity is None
    # With no cavity term the total operator *is* the standard one.
    assert driver.I_dimer.base is driver.I_dimer_standard.base
    # And v() still returns correctly shaped exact zeros for the cavity context.
    cavity_block = driver.v("abrs", context="cavity")
    assert cavity_block.shape == driver.v("abrs", context="standard").shape
    assert np.abs(cavity_block).max() == 0.0


def test_integral_tensors_are_not_writeable():
    """Audit item A.1 guard: shared views must not be mutable."""
    driver = _driver()
    for name in ("I_dimer", "I_dimer_standard", "I_dimer_cavity"):
        array = getattr(driver, name)
        assert array is None or not array.flags.writeable


def test_exchange_dispersion_refuses_missing_amplitudes():
    """Audit item B: the t_rsab coupling must be explicit, not silent."""
    driver = _driver()
    assert not hasattr(driver, "t_rsab")

    with pytest.raises(RuntimeError, match="compute_Edisp200"):
        driver.compute_Eexchdisp200()


def test_exchange_dispersion_records_denominator_convention():
    """Audit item B: the canonical_denom choice must not leak silently."""
    driver = _driver()

    driver.compute_Edisp200(canonical_denom=False)
    assert driver.t_rsab_canonical_denom is False
    cqed_amplitudes = driver.t_rsab

    # Reusing cached amplitudes is fine when the convention matches ...
    driver.compute_Eexchdisp200(canonical_denom=False)
    assert driver.t_rsab is cqed_amplitudes

    # ... and asking for the other convention rebuilds rather than reusing.
    driver.compute_Eexchdisp200(canonical_denom=True)
    assert driver.t_rsab_canonical_denom is True
    assert driver.t_rsab is not cqed_amplitudes


# --------------------------------------------------------------------------
# Phase 0: the lambda = 0 oracle
# --------------------------------------------------------------------------


def test_lambda_zero_dispersion_matches_psi4_sapt0():
    """Pin the dense reference against an independent implementation.

    At ``lambda = 0`` QED-SAPT0 reduces to conventional SAPT0 exactly, so
    Psi4's own ``sapt0`` is a genuine external oracle for the dispersion terms
    -- not FISAPT, which is functional-group SAPT.  The residual difference is
    Psi4's density-fitting error against our exact-ERI reference.

    This anchors ``compute_Edisp200``/``compute_Eexchdisp200`` *before* the DF
    backend replaces their integral source, so a later regression can be
    attributed to the backend rather than to the formulas.
    """
    psi4.core.be_quiet()
    psi4.core.clean()

    psi4.geometry(_WATER_HE)
    psi4.set_options(
        {
            "basis": BASIS,
            "freeze_core": "false",
            "e_convergence": 1e-10,
            "d_convergence": 1e-10,
        }
    )
    psi4.energy("sapt0")
    reference = psi4.core.variables()

    psi4.core.clean()
    driver = _driver(lambda_vector=(0.0, 0.0, 0.0), include_cavity_terms=False)
    disp20 = driver.compute_Edisp200()
    exch_disp20 = driver.compute_Eexchdisp200()

    assert disp20 == pytest.approx(reference["SAPT DISP20 ENERGY"], abs=1e-6)
    assert exch_disp20 == pytest.approx(reference["SAPT EXCH-DISP20 ENERGY"], abs=1e-6)

    # Also confirm the first-order terms, which use the same v()/vt() machinery
    # the DF backend will replace.  Exchange must be compared in the S^2
    # convention -- Psi4 also reports an S^inf value (see docs/SAPT_NOTE.md).
    assert driver.compute_Elst100() == pytest.approx(
        reference["SAPT ELST10,R ENERGY"], abs=1e-5
    )
    assert driver.compute_Exch100() == pytest.approx(
        reference["SAPT EXCH10(S^2) ENERGY"], abs=1e-5
    )


# --------------------------------------------------------------------------
# Phase 2: the driver's "df" integral backend
# --------------------------------------------------------------------------


_COMPONENTS = ("Elst10", "Exch10", "Disp20", "ExchDisp20", "Ind20r", "ExchInd20r")


def _components(driver):
    return {
        "Elst10": driver.compute_Elst100(),
        "Exch10": driver.compute_Exch100(),
        "Disp20": driver.compute_Edisp200(),
        "ExchDisp20": driver.compute_Eexchdisp200(),
        "Ind20r": driver.compute_Eind200(),
        "ExchInd20r": driver.compute_Eexchind200(),
    }


def _backend_driver(backend, lambda_vector=(0.0, 0.0, 0.1), include_cavity_terms=True, **kwargs):
    psi4.core.be_quiet()
    psi4.core.clean()
    driver = QEDSAPT0Driver(
        dimer_geometry=psi4.geometry(_WATER_HE),
        config=_config(lambda_vector),
        integral_backend=backend,
        include_cavity_terms=include_cavity_terms,
        **kwargs,
    )
    driver.prepare_monomers()
    driver.build_integrals()
    return driver


@pytest.fixture(scope="module")
def dense_vs_df():
    dense = _components(_backend_driver("full_eri"))
    df_driver = _backend_driver("df")
    return dense, _components(df_driver), df_driver


@pytest.mark.parametrize("component", _COMPONENTS)
def test_df_backend_reproduces_dense_components(component, dense_vs_df):
    """Every existing component, not only the two new ones.

    This is why Phase 2 precedes the dispersion work: it exercises the
    augmented DF tensor against all six pinned components, which is far more
    diagnostic than validating it through Disp20 alone.
    """
    dense, df, _ = dense_vs_df
    assert df[component] == pytest.approx(dense[component], abs=5e-6)


def test_df_backend_total_matches_dense(dense_vs_df):
    dense, df, _ = dense_vs_df
    assert sum(df.values()) == pytest.approx(sum(dense.values()), abs=5e-6)


def test_df_backend_allocates_no_dense_ao_tensors():
    driver = _backend_driver("df")

    assert driver.I_dimer is None
    assert driver.I_dimer_standard is None
    assert driver.I_dimer_cavity is None

    # And the dense accessor refuses rather than returning something that would
    # silently become zeros downstream.
    with pytest.raises(RuntimeError, match="no dense AO ERI tensor"):
        driver._eri_for_context("total")


def test_cavity_eri_norm_agrees_between_backends():
    """The rank-one identity ||d_A (x) d_B|| = ||d_A|| ||d_B|| is exact.

    It keeps the diagnostic meaningful under the df backend, where the tensor
    is deliberately never materialized.
    """
    dense = _backend_driver("full_eri")
    df = _backend_driver("df")

    expected = np.linalg.norm(dense.I_dimer_cavity)
    assert dense._cavity_eri_norm() == pytest.approx(expected, rel=1e-12)
    assert df._cavity_eri_norm() == pytest.approx(expected, rel=1e-12)


def test_df_partitions_sum_correctly():
    driver = _backend_driver("df")
    partitions = driver.vt_partitions("abrs")
    np.testing.assert_allclose(
        partitions["total"]["total"],
        partitions["standard"]["total"] + partitions["cavity"]["total"],
        rtol=0.0,
        atol=1e-12,
    )


# -- fitting-role routing ---------------------------------------------------


def test_dispersion_uses_the_correlation_fitting_basis():
    """Behavioural check that the role wiring is real, not just declared.

    Changing only the correlation fitting basis must move the dispersion terms
    and leave the Coulomb-like terms bitwise unchanged; anything else means a
    term is drawing on the wrong tensor.
    """
    default = _backend_driver("df")
    swapped = _backend_driver("df", df_corr_fitting_role="JKFIT")

    a, b = _components(default), _components(swapped)

    for term in ("Elst10", "Exch10", "Ind20r", "ExchInd20r"):
        assert a[term] == b[term], f"{term} must not depend on the correlation fitting basis"
    for term in ("Disp20", "ExchDisp20"):
        assert a[term] != b[term], f"{term} must depend on the correlation fitting basis"


def test_correlation_tensor_is_built_lazily():
    """A caller wanting only first-order terms should not pay for RIFIT."""
    driver = _backend_driver("df")

    assert "scf" in driver._pf_df
    assert "corr" not in driver._pf_df

    driver.compute_Elst100()
    assert "corr" not in driver._pf_df

    driver.compute_Edisp200()
    assert "corr" in driver._pf_df


def test_correlation_fitting_basis_is_the_wrong_choice_for_electrostatics():
    """Assert that the naive approach fails, per SAPT_NOTE.md section 8.3.

    A single RIFIT tensor for every term is the obvious simplification, and it
    is wrong: Elst10 is a small residual of large cancelling terms, so a
    correlation-optimized fit degrades it by roughly two orders of magnitude.
    This test documents why the scf/corr role split exists, and will signal if
    that ever stops being true.
    """
    dense = _components(_backend_driver("full_eri"))
    jkfit = _components(_backend_driver("df"))
    rifit = _components(_backend_driver("df", df_scf_fitting_role="RIFIT"))

    err_jkfit = abs(jkfit["Elst10"] - dense["Elst10"])
    err_rifit = abs(rifit["Elst10"] - dense["Elst10"])

    assert err_rifit > 20 * err_jkfit
    assert err_jkfit < 1e-6


def test_backend_and_role_arguments_are_validated():
    with pytest.raises(ValueError, match="integral_backend"):
        QEDSAPT0Driver(
            dimer_geometry=psi4.geometry(_WATER_HE),
            config=_config((0.0, 0.0, 0.0)),
            integral_backend="bogus",
        )

    driver = _backend_driver("df")
    with pytest.raises(ValueError, match="df_role"):
        driver.v("abrs", df_role="bogus")

    dense = _backend_driver("full_eri")
    with pytest.raises(RuntimeError, match='integral_backend="df"'):
        dense._pauli_fierz_df("scf")


def test_df_backend_lambda_zero_cavity_on_matches_cavity_off():
    """The lambda = 0 control, now through the df path."""
    on = _components(_backend_driver("df", lambda_vector=(0.0, 0.0, 0.0), include_cavity_terms=True))
    off = _components(_backend_driver("df", lambda_vector=(0.0, 0.0, 0.0), include_cavity_terms=False))

    for term in _COMPONENTS:
        assert on[term] == pytest.approx(off[term], abs=1e-12)


# --------------------------------------------------------------------------
# Phase 3: Disp20
# --------------------------------------------------------------------------


def _water_he(separation=3.4, shift=0.0):
    return f"""
0 1
O 0.0 0.0 {0.0 + shift}
H 0.0 0.757 {0.587 + shift}
H 0.0 -0.757 {0.587 + shift}
--
0 1
He 0.0 0.0 {separation + shift}
symmetry c1
units angstrom
no_reorient
no_com
"""


def _geometry_driver(
    geometry, lambda_vector=(0.0, 0.0, 0.1), backend="full_eri", convergence=1e-10
):
    psi4.core.be_quiet()
    psi4.core.clean()
    driver = QEDSAPT0Driver(
        dimer_geometry=psi4.geometry(geometry),
        config=_config(lambda_vector, convergence=convergence),
        integral_backend=backend,
        include_cavity_terms=True,
    )
    driver.prepare_monomers()
    driver.build_integrals()
    return driver


@pytest.mark.parametrize("backend", ["full_eri", "df"])
@pytest.mark.parametrize("context", ["standard", "cavity", "total"])
def test_disp20_rsab_block_is_the_abrs_transpose(backend, context):
    """Why compute_Edisp200 builds only one two-electron block.

    b_(ra) = b_(ar)^T because the three-index tensor is symmetric in its
    orbital pair, so v('rsab') carries no information v('abrs') does not.  The
    dense backend satisfies the same identity through the eightfold ERI
    symmetry.  If this ever fails, the single-transform optimization in
    _dispersion_numerator() is invalid.
    """
    driver = _backend_driver(backend)
    direct = driver.v("rsab", context=context, df_role="corr")
    transposed = driver.v("abrs", context=context, df_role="corr").transpose(2, 3, 0, 1)

    scale = max(np.abs(direct).max(), 1.0)
    assert np.abs(direct - transposed).max() < 1e-13 * scale


@pytest.mark.parametrize("backend", ["full_eri", "df"])
def test_dispersion_partition_sums_to_the_total(backend):
    """The three-way split is exact; the two-way split would not be."""
    driver = _backend_driver(backend)
    partition = driver.dispersion_energy_partition()

    assert partition["total"] == pytest.approx(
        partition["standard"] + partition["cross"] + partition["cavity"], abs=1e-18
    )
    assert partition["total"] == pytest.approx(driver.compute_Edisp200(), abs=1e-15)


def test_dispersion_partition_is_not_two_way_additive():
    """Disp20 is quadratic in the numerator, so a cross term exists.

    Recorded as a test because "standard + cavity" is the obvious
    simplification and it is wrong: here the cross term is larger in magnitude
    than either part it connects.
    """
    driver = _backend_driver("full_eri")
    partition = driver.dispersion_energy_partition()

    assert abs(partition["cross"]) > 1e-12
    assert partition["total"] != pytest.approx(
        partition["standard"] + partition["cavity"], abs=1e-9
    )


def test_dispersion_partition_does_not_disturb_production_amplitudes():
    """It is a diagnostic; audit item B must stay closed."""
    driver = _backend_driver("full_eri")
    driver.compute_Edisp200()
    cached = driver.t_rsab

    driver.dispersion_energy_partition()

    assert driver.t_rsab is cached
    assert driver.t_rsab_context == "total"
    assert driver.t_rsab_canonical_denom is False


def test_cavity_dispersion_matches_its_closed_form():
    """The cavity dispersion is computable in closed form from d and eps alone.

    Its numerator is the bare outer product of monomer transition dipoles,
    d^A_ar d^B_bs, with no Coulomb kernel, so

        Disp20[cav,cav] = 4 sum (d^A_ar)^2 (d^B_bs)^2 / (e_a + e_b - e_r - e_s)

    exactly.  Both sides are free of fitting error, so this holds to roundoff
    and validates the cavity numerator algebra independently of any dimer
    two-electron integral.
    """
    driver = _backend_driver("full_eri")

    d_A = driver.Co_A.T @ driver.d_A @ driver.Cv_A
    d_B = driver.Co_B.T @ driver.d_B @ driver.Cv_B
    closed_form = 4 * np.einsum(
        "ar,bs,rsab->", d_A**2, d_B**2, driver._dispersion_denominator(False), optimize=True
    )

    assert driver.dispersion_energy_partition()["cavity"] == pytest.approx(
        closed_form, rel=1e-12
    )


def test_cavity_dispersion_numerator_does_not_decay_with_separation():
    """The structural claim of the single-mode long-wavelength approximation.

    The cavity kernel d (x) d contains no Coulomb operator, so the dispersion
    *numerator* has no R-dependence, while the standard numerator decays.
    Verified on the numerator specifically: the cavity *energy* does decay,
    but for an unrelated reason -- see
    test_dispersion_energy_is_translation_invariant.
    """
    near = _geometry_driver(_water_he(separation=8.0))
    far = _geometry_driver(_water_he(separation=12.0))

    cav_near = np.linalg.norm(near.v("abrs", context="cavity"))
    cav_far = np.linalg.norm(far.v("abrs", context="cavity"))
    std_near = np.linalg.norm(near.v("abrs", context="standard"))
    std_far = np.linalg.norm(far.v("abrs", context="standard"))

    # Cavity numerator: unchanged to six significant figures over 8 -> 12 Ang.
    assert cav_far == pytest.approx(cav_near, rel=1e-5)
    # Standard numerator: falls by more than a factor of three over the same range.
    assert std_far < std_near / 3.0


@pytest.mark.parametrize(
    "component", ["Elst10", "Exch10", "Ind20r", "ExchInd20r"]
)
def test_sapt_components_are_translation_invariant(component):
    """A rigid translation of the whole dimer is not a physical change."""
    here = _components(_geometry_driver(_water_he(shift=0.0), convergence=1e-12))
    there = _components(_geometry_driver(_water_he(shift=20.0), convergence=1e-12))

    assert there[component] == pytest.approx(here[component], abs=1e-12)


@pytest.mark.xfail(
    reason=(
        "Known defect, not introduced by the DF work: CQED-SCF orbital energies "
        "are not translation invariant, so the dispersion denominators depend on "
        "where the coordinate origin sits. The total CQED-SCF energy IS invariant "
        "(to 4e-14) and so are Elst10/Exch10/Ind20,r/Exch-Ind20,r; only the terms "
        "using bare monomer orbital-energy differences are affected. The "
        "HOMO-LUMO gap of a translated monomer grows as lambda^2 z^2, which "
        "points at the origin-dependent one-electron DSE (quadrupole) term and "
        "the exchange-like N[D] term built from the bare dipole matrix rather "
        "than the coherent-state fluctuation. See "
        "docs/development/QED_SAPT_DISPERSION_PLAN.md."
    ),
    strict=True,
)
@pytest.mark.parametrize("component", ["Disp20", "ExchDisp20"])
def test_dispersion_energy_is_translation_invariant(component):
    here = _components(_geometry_driver(_water_he(shift=0.0), convergence=1e-12))
    there = _components(_geometry_driver(_water_he(shift=20.0), convergence=1e-12))

    assert there[component] == pytest.approx(here[component], rel=1e-6)


def test_lambda_zero_dispersion_is_translation_invariant():
    """Control: with the cavity switched off the defect above disappears."""
    here = _components(
        _geometry_driver(_water_he(shift=0.0), lambda_vector=(0.0, 0.0, 0.0), convergence=1e-12)
    )
    there = _components(
        _geometry_driver(_water_he(shift=20.0), lambda_vector=(0.0, 0.0, 0.0), convergence=1e-12)
    )

    for component in ("Disp20", "ExchDisp20"):
        assert there[component] == pytest.approx(here[component], abs=1e-12)


# --------------------------------------------------------------------------
# Option 1: intrinsic per-monomer reference frame
# --------------------------------------------------------------------------


def _framed_driver(frame, shift=0.0, separation=3.4, lambda_vector=(0.0, 0.0, 0.1),
                   backend="full_eri", convergence=1e-12):
    psi4.core.be_quiet()
    psi4.core.clean()
    driver = QEDSAPT0Driver(
        dimer_geometry=psi4.geometry(_water_he(separation=separation, shift=shift)),
        config=_config(lambda_vector, convergence=convergence),
        integral_backend=backend,
        include_cavity_terms=True,
        monomer_reference_frame=frame,
    )
    driver.prepare_monomers()
    driver.build_integrals()
    return driver


_FRAMED_COMPONENTS = {}


def _framed_components(frame, shift=0.0, backend="full_eri", lambda_vector=(0.0, 0.0, 0.1)):
    """Cached component run: the frame tests reuse a handful of identical drivers."""
    key = (frame, shift, backend, lambda_vector)
    if key not in _FRAMED_COMPONENTS:
        _FRAMED_COMPONENTS[key] = _components(
            _framed_driver(
                frame,
                shift=shift,
                backend=backend,
                lambda_vector=lambda_vector,
                convergence=1e-12,
            )
        )
    return _FRAMED_COMPONENTS[key]


@pytest.mark.parametrize("backend", ["full_eri", "df"])
@pytest.mark.parametrize("component", _COMPONENTS)
def test_monomer_com_frame_makes_every_component_translation_invariant(component, backend):
    """The point of Option 1: results must not depend on where the dimer sits.

    Both integral backends are covered. The frame handling lives in
    build_integrals() *before* the backend branch and in the backend-agnostic
    v(..., frame=...) dispatch, so the two should behave identically -- but
    "should" is what this test is for. Under the default 'dimer' frame Disp20
    changes by a factor of 7 under this same translation (see
    test_dispersion_energy_is_translation_invariant).
    """
    here = _framed_components("monomer_com", shift=0.0, backend=backend)
    there = _framed_components("monomer_com", shift=20.0, backend=backend)

    assert there[component] == pytest.approx(here[component], abs=1e-11)


@pytest.mark.parametrize("component", ["Elst10", "Exch10", "Ind20r", "ExchInd20r"])
def test_monomer_com_frame_leaves_the_already_invariant_terms_unchanged(component):
    """Only the dispersion terms should move.

    Electrostatics, exchange and induction were already origin independent, so
    changing the reference frame must not perturb them.  Induction is the sharp
    one: it is origin independent only because its coupled response carries a
    compensating DSE Hessian term, so it stays right only if that Hessian is
    built in the same frame as the orbital energies feeding it.
    """
    dimer_frame = _framed_components("dimer")
    com_frame = _framed_components("monomer_com")

    assert com_frame[component] == pytest.approx(dimer_frame[component], abs=1e-12)


@pytest.mark.parametrize("component", ["Disp20", "ExchDisp20"])
def test_monomer_com_frame_does_change_the_dispersion_terms(component):
    """The correction is real, not a no-op."""
    dimer_frame = _framed_components("dimer")
    com_frame = _framed_components("monomer_com")

    assert abs(com_frame[component] - dimer_frame[component]) > 1e-9


def test_monomer_com_frame_preserves_the_shared_interaction_frame():
    """d_A == d_B must survive, or the rank-one DF augmentation breaks.

    Monomers A and B are solved in *different* intrinsic frames.  Their dipole
    matrices would differ by (z_A - z_B) * lambda * S if the interaction
    operator followed them there, which would both misstate the interaction and
    destroy the single-auxiliary-row factorization of Phases 1-2.
    """
    driver = _framed_driver("monomer_com")

    np.testing.assert_allclose(driver.d_A, driver.d_B, rtol=0.0, atol=1e-13)
    # The intrinsic-frame matrices, by contrast, genuinely differ from each
    # other and from the shared one -- that is what makes the split meaningful.
    assert np.abs(driver._d_intrinsic["A"] - driver._d_intrinsic["B"]).max() > 1e-6
    assert np.abs(driver._d_intrinsic["B"] - driver.d_A).max() > 1e-6


def test_monomer_com_frame_restores_the_dispersion_plateau():
    """With an intrinsic reference the cavity dispersion does plateau.

    This is audit item E finally answered in the affirmative: the cavity
    contribution tends to a finite non-zero constant, because its numerator has
    no Coulomb kernel and its denominators no longer drift with the origin.
    """
    near = _framed_driver("monomer_com", separation=12.0)
    far = _framed_driver("monomer_com", separation=50.0)

    cav_near = near.dispersion_energy_partition()["cavity"]
    cav_far = far.dispersion_energy_partition()["cavity"]
    std_near = near.dispersion_energy_partition()["standard"]
    std_far = far.dispersion_energy_partition()["standard"]

    assert cav_near != pytest.approx(0.0, abs=1e-9)
    # Cavity part: constant to nine significant figures over 12 -> 50 Ang.
    assert cav_far == pytest.approx(cav_near, rel=1e-9)
    # Standard part over the same range: gone.
    assert abs(std_far) < abs(std_near) / 1000.0


def test_monomer_com_frame_works_with_the_df_backend():
    dense = _framed_components("monomer_com", backend="full_eri")
    df = _framed_components("monomer_com", backend="df")

    for component in _COMPONENTS:
        assert df[component] == pytest.approx(dense[component], abs=5e-6)


def test_dimer_frame_remains_the_default_and_is_unchanged():
    """Option 1 is opt-in; the historical path must be bitwise untouched."""
    driver = _framed_driver("dimer")
    assert driver.monomer_reference_frame == "dimer"
    assert QEDSAPT0Driver(
        dimer_geometry=psi4.geometry(_water_he()),
        config=_config((0.0, 0.0, 0.0)),
    ).monomer_reference_frame == "dimer"

    # In the dimer frame the intrinsic matrices *are* the shared one, so every
    # frame correction in v() vanishes identically rather than merely being small.
    for side in ("A", "B"):
        assert np.abs(driver._d_intrinsic[side] - driver.d_A).max() == 0.0
    for string in ("raar", "rara", "sbbs", "sbsb"):
        np.testing.assert_array_equal(
            driver.v(string, frame="A"), driver.v(string)
        )


def test_monomer_reference_frame_is_validated():
    with pytest.raises(ValueError, match="monomer_reference_frame"):
        QEDSAPT0Driver(
            dimer_geometry=psi4.geometry(_water_he()),
            config=_config((0.0, 0.0, 0.0)),
            monomer_reference_frame="bogus",
        )
    with pytest.raises(ValueError, match="frame must be"):
        _framed_driver("dimer").v("abrs", frame="C")


def test_lambda_zero_is_insensitive_to_the_reference_frame():
    """Control: with no cavity the frame cannot matter at all."""
    dimer_frame = _components(_framed_driver("dimer", lambda_vector=(0.0, 0.0, 0.0)))
    com_frame = _components(_framed_driver("monomer_com", lambda_vector=(0.0, 0.0, 0.0)))

    for component in _COMPONENTS:
        assert com_frame[component] == pytest.approx(dimer_frame[component], abs=1e-12)


def test_results_metadata_exposes_the_dispersion_partition():
    """The partition must be reachable from the public facade, not only the driver.

    ``examples/qed_sapt0_water_methylamine/qed_sapt0_scan.py`` relies on this to
    stay on the documented CQEDConfig / CQEDCalculator interface.
    """
    driver = _backend_driver("full_eri")
    results = driver.run_components()

    partition = results.metadata["disp20_partition"]
    assert set(partition) == {"standard", "cross", "cavity", "total"}
    assert partition == driver.dispersion_energy_partition()
    assert (
        partition["standard"] + partition["cross"] + partition["cavity"]
        == pytest.approx(results.disp20, abs=1e-15)
    )


# --------------------------------------------------------------------------
# The same frame split, through the JK backend
# --------------------------------------------------------------------------
#
# The JK path in ``sapt/qed_sapt_jk.py`` consumes bare Psi4 wavefunctions and
# a caller-built JK object, so it cannot ask a driver which frame it is in.
# These tests hold it to the same standard as the dense/DF path above.
#
# Only four components are covered because dispersion is not implemented in
# the JK path yet (Phases 4-5 of docs/development/QED_SAPT_DISPERSION_PLAN.md).
# Induction is the sharp one here for the same reason as in the dense case.

_JK_COMPONENTS = ("Elst10", "Exch10", "Ind20r", "ExchInd20r")


def _jk_components_from_cache(cache):
    elst = qed_sapt_jk.electrostatics(cache, do_print=False)
    exch = qed_sapt_jk.exchange(cache, do_print=False)
    ind = qed_sapt_jk.induction(cache, do_print=False, do_response=True)
    return {
        "Elst10": float(elst["Elst10,r"]),
        # compute_Exch100() is the S^2 expression, so pair it with the
        # matching JK key rather than the S^inf-style Exch10 value.
        "Exch10": float(exch["Exch10(S^2)"]),
        "Ind20r": float(ind["Ind20,r"]),
        "ExchInd20r": float(ind["Exch-Ind20,r"]),
    }


_FRAMED_JK_COMPONENTS = {}


def _framed_jk_components(frame, shift=0.0):
    """JK-path components, memoized: each entry costs two monomer SCFs."""
    key = (frame, shift)
    if key not in _FRAMED_JK_COMPONENTS:
        driver = _framed_driver(frame, shift=shift)
        _FRAMED_JK_COMPONENTS[key] = _jk_components_from_cache(
            qed_sapt_jk.build_sapt_jk_cache_from_driver(driver, do_print=False)
        )
    return _FRAMED_JK_COMPONENTS[key]


def _legacy_jk_cache(driver):
    """The pre-frame-awareness wiring, as the shipped examples still do it.

    Kept verbatim so the default path has a regression anchor and so the
    guard has something realistic to fire on.
    """
    wfn_A = driver.monomer_A.wfn
    wfn_B = driver.monomer_B.wfn
    jk = psi4.core.JK.build(wfn_A.basisset())
    jk.set_memory(int(1e9))
    jk.set_do_J(True)
    jk.set_do_K(True)
    jk.set_do_wK(False)
    jk.initialize()
    dse_jk = DSEJK(
        d_ao=np.asarray(driver.monomer_A.d_ao),
        enabled=driver.include_cavity_terms,
    )
    return qed_sapt_jk.build_sapt_jk_cache(
        wfn_A,
        wfn_B,
        PauliFierzJK(jk, dse_jk=dse_jk),
        do_print=False,
        d_exp_el_A=driver.d_exp_el_A,
        d_exp_el_B=driver.d_exp_el_B,
        include_cavity_terms=driver.include_cavity_terms,
        nuclear_repulsion_energy=driver.nuc_rep,
    )


@pytest.mark.parametrize("component", _JK_COMPONENTS)
def test_jk_path_monomer_com_frame_is_translation_invariant(component):
    """The decisive test: a rigid 20 Ang shift is not a physical change.

    Matches the dense/DF result for the same four components.  With the DSE
    orbital Hessian left in the shared frame instead -- the state of this path
    before frame support -- Ind20,r drifts by 1.07e-5 Eh and Exch-Ind20,r by
    3.1e-6 Eh over this same translation.
    """
    here = _framed_jk_components("monomer_com", shift=0.0)
    there = _framed_jk_components("monomer_com", shift=20.0)

    assert there[component] == pytest.approx(here[component], abs=1e-11)


@pytest.mark.parametrize("component", _JK_COMPONENTS)
def test_jk_path_monomer_com_frame_leaves_every_component_unchanged(component):
    """None of the four JK components should move when the frame changes.

    All four were already origin independent, so ``monomer_com`` must
    reproduce the dimer-frame numbers exactly.  Induction is the sharp one:
    it is origin independent only because the drift in its orbital energies
    cancels against the -d_bb' d_ss' term of the DSE Hessian, so it survives
    only if that Hessian is built in the frame its orbital energies came from.
    """
    dimer_frame = _framed_jk_components("dimer")
    com_frame = _framed_jk_components("monomer_com")

    assert com_frame[component] == pytest.approx(dimer_frame[component], abs=1e-12)


@pytest.mark.parametrize("frame", ["dimer", "monomer_com"])
@pytest.mark.parametrize("component", _JK_COMPONENTS)
def test_jk_path_matches_the_dense_reference_in_either_frame(component, frame):
    """The JK path must not drift from the validated dense implementation.

    The dense-vs-JK agreement was ~1e-12 Eh before frame support existed; the
    intrinsic-frame Hessian must not cost any of it.
    """
    jk_values = _framed_jk_components(frame)
    dense_values = _framed_components(frame)

    assert jk_values[component] == pytest.approx(dense_values[component], abs=1e-11)


def test_jk_cache_refuses_divergent_monomer_frames_it_cannot_account_for():
    """Silent wrongness is the one outcome this path must not produce.

    ``build_sapt_jk_cache`` takes raw wavefunctions and has no driver to ask,
    but the frames are directly observable: two ghosted monomers in one dimer
    frame have bitwise identical coordinates, and under ``monomer_com`` they
    sit 6.3 bohr apart.
    """
    dimer_driver = _framed_driver("dimer")
    com_driver = _framed_driver("monomer_com")

    geom = lambda d, side: np.asarray(
        (d.monomer_A if side == "A" else d.monomer_B).wfn.molecule().geometry()
    )
    assert np.abs(geom(dimer_driver, "A") - geom(dimer_driver, "B")).max() == 0.0
    assert np.abs(geom(com_driver, "A") - geom(com_driver, "B")).max() > 6.0

    # The legacy wiring is fine in the dimer frame ...
    _legacy_jk_cache(dimer_driver)

    # ... and refused in the intrinsic one, rather than quietly returning a
    # cache whose induction is wrong by 1e-6 Eh.
    with pytest.raises(RuntimeError, match="different reference frames"):
        _legacy_jk_cache(com_driver)


def test_jk_cache_default_frame_path_is_bitwise_unchanged():
    """Frame support is opt-in; the dimer-frame numbers must not move at all.

    Compared against the legacy wiring rather than a stored constant, so this
    keeps holding if the underlying components are ever legitimately revised.
    """
    driver = _framed_driver("dimer")

    new = _jk_components_from_cache(
        qed_sapt_jk.build_sapt_jk_cache_from_driver(driver, do_print=False)
    )
    legacy = _jk_components_from_cache(_legacy_jk_cache(driver))

    for component in _JK_COMPONENTS:
        assert new[component] == legacy[component]


def test_jk_cache_reuses_one_dse_provider_when_the_frames_coincide():
    """The dimer-frame no-op is by identity, not by an equal-valued rebuild."""
    driver = _framed_driver("dimer")
    cache = qed_sapt_jk.build_sapt_jk_cache_from_driver(driver, do_print=False)

    shared = cache["dse_jk"]
    assert cache["dse_cphf_A"].dse_jk is shared
    assert cache["dse_cphf_B"].dse_jk is shared

    # Under monomer_com they must genuinely differ, or the split is a no-op.
    com_cache = qed_sapt_jk.build_sapt_jk_cache_from_driver(
        _framed_driver("monomer_com"), do_print=False
    )
    d_A = com_cache["dse_cphf_A"].d_ao
    d_B = com_cache["dse_cphf_B"].d_ao
    assert np.abs(d_A - d_B).max() > 1e-6
    assert np.abs(d_A - com_cache["dse_jk"].d_ao).max() > 1e-6
