"""Tier 5: the CQEDCalculator response API.

These cover the facade rather than the physics -- the physics is pinned by
test_qed_cis_layout / _sigma / _dense / _tier3.  What matters here is that the
public entry points run the reference, dispatch to the right engine, agree with
the low-level driver, and refuse clearly when asked for something unimplemented.
"""

import numpy as np
import psi4
import pytest

import cqed_scf
from cqed_scf import CQEDCalculator


WATER = """
0 1
O   0.000000000000   0.000000000000  -0.068516219320
H   0.000000000000  -0.790689573744   0.543701060715
H   0.000000000000   0.790689573744   0.543701060715
no_reorient
no_com
symmetry c1
"""

OPTIONS = {
    "basis": "sto-3g",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "save_jk": True,
}

OMEGA = 0.35


def _calculator(functional=None, coupling=0.05, quiet=True):
    psi4.core.clean()
    psi4.core.clean_options()
    return CQEDCalculator(
        lambda_vector=np.array([0.0, 0.0, coupling]),
        psi4_options=OPTIONS,
        omega=OMEGA,
        functional=functional,
        quiet=quiet,
    )


def test_package_exports_the_response_symbols():
    for name in ("QEDCIS", "QEDCISResults", "print_qed_cis_results", "davidson_solve"):
        assert hasattr(cqed_scf, name), f"cqed_scf.{name} is not exported"


@pytest.mark.slow
def test_cis_runs_end_to_end_from_a_geometry():
    results = _calculator().cis(WATER, nroots=5, print_results=False)

    assert results.eigenvalues.size == 5
    assert results.excitation_energies[0] == 0.0
    assert np.all(np.diff(results.eigenvalues) >= -1e-12)
    assert results.oscillator_strengths is not None
    assert results.total_energies[0] == pytest.approx(
        results.scf_energy + results.eigenvalues[0], abs=1e-12
    )


@pytest.mark.slow
def test_cis_accepts_precomputed_scf_results():
    """Reusing a reference must give the same answer as computing it inline."""

    calculator = _calculator()
    driver = calculator.response(WATER, n_photon=1)
    reused = calculator.cis(
        scf_results=driver.scf_results, nroots=4, print_results=False
    )
    fresh = calculator.cis(WATER, nroots=4, print_results=False)

    np.testing.assert_allclose(reused.eigenvalues, fresh.eigenvalues, atol=1e-9)


@pytest.mark.slow
def test_cis_solvers_agree():
    calculator = _calculator()
    driver = calculator.response(WATER, n_photon=1)

    davidson = calculator.cis(
        scf_results=driver.scf_results, nroots=5, solver="davidson",
        print_results=False,
    )
    dense = calculator.cis(
        scf_results=driver.scf_results, solver="dense", print_results=False
    )

    np.testing.assert_allclose(davidson.eigenvalues, dense.eigenvalues[:5], atol=1e-8)


@pytest.mark.slow
def test_response_returns_an_unsolved_driver():
    driver = _calculator().response(WATER, n_photon=2)

    assert driver.n_photon == 2
    assert driver.dimension == 3 * (1 + driver.n_ov)
    # nothing has been solved, and no O(N^4) transformation has happened
    assert driver.ovov is None and driver.oovv is None


@pytest.mark.slow
def test_tddft_dispatches_to_the_kohn_sham_path():
    results = _calculator(functional="B3LYP").tddft(WATER, nroots=4, print_results=False)

    assert results.eigenvalues.size == 4
    assert np.all(np.diff(results.eigenvalues) >= -1e-12)


@pytest.mark.slow
def test_tddft_and_cis_agree_for_a_kohn_sham_reference():
    """tddft() is cis() named for what it is; they must not diverge."""

    calculator = _calculator(functional="B3LYP")
    driver = calculator.response(WATER, n_photon=1)

    from_cis = calculator.cis(
        scf_results=driver.scf_results, nroots=4, print_results=False
    )
    from_tddft = calculator.tddft(
        scf_results=driver.scf_results, nroots=4, print_results=False
    )

    np.testing.assert_allclose(from_tddft.eigenvalues, from_cis.eigenvalues, atol=1e-12)


def test_tddft_refuses_non_tda_clearly():
    """Full linear response is Tier 4; the refusal must say so."""

    with pytest.raises(NotImplementedError, match="non-TDA|Tier 4"):
        _calculator(functional="B3LYP").tddft(WATER, tda=False)


def test_tddft_requires_a_functional():
    with pytest.raises(ValueError, match="Kohn-Sham|functional"):
        _calculator().tddft(WATER)


def test_cis_requires_a_geometry_or_scf_results():
    with pytest.raises(TypeError, match="geometry or scf_results"):
        _calculator().cis()


@pytest.mark.slow
def test_results_table_prints_without_error(capsys):
    # quiet=False: output.quiet_context would otherwise suppress the table
    _calculator(quiet=False).cis(WATER, nroots=3, print_results=True)

    printed = capsys.readouterr().out
    assert "QED-CIS Excited States" in printed
    assert "<b+b>" in printed
    assert "f (osc)" in printed


@pytest.mark.slow
def test_results_table_names_the_kohn_sham_method(capsys):
    _calculator(functional="B3LYP", quiet=False).cis(
        WATER, nroots=3, print_results=True
    )

    assert "QED-TDA-DFT Excited States" in capsys.readouterr().out
