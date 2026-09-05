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
    """An unsolved driver must still be able to describe itself.

    block_size and dimension prepare the orbital blocks on demand, like every
    other entry point on QEDCIS; querying the size of the problem should not
    require the caller to know to call build_orbital_blocks() first.
    """

    driver = _calculator().response(WATER, n_photon=2)

    assert driver.n_photon == 2
    assert driver.dimension == 3 * (1 + driver.n_ov)
    assert driver.block_size == 1 + driver.ndocc * driver.nvirt
    # nothing has been solved, and no O(N^4) transformation has happened
    assert driver.ovov is None and driver.oovv is None


@pytest.mark.slow
def test_cis_runs_with_a_kohn_sham_reference():
    results = _calculator(functional="B3LYP").cis(WATER, nroots=4, print_results=False)

    assert results.eigenvalues.size == 4
    assert np.all(np.diff(results.eigenvalues) >= -1e-12)


def test_tddft_is_reserved_for_linear_response_qed_tddft():
    """The name must not be attached to this method.

    QED-CIS on a QED-Kohn-Sham reference is a Fock-basis CI: it carries
    |Phi_i^a, n>=1> configurations that a product ansatz cannot represent,
    treats the DSE as a two-electron operator rather than at mean-field level,
    and relaxes the ground state.  It is not a Tamm-Dancoff approximation to
    the linear-response QED-TDDFT of Yang et al.  Reserving the name keeps
    anyone from citing these numbers as QED-TDDFT.
    """

    for calculator in (_calculator(), _calculator(functional="B3LYP")):
        with pytest.raises(NotImplementedError, match="Yang|linear-response"):
            calculator.tddft(WATER)


def test_tddft_refusal_points_at_the_method_that_does_work():
    with pytest.raises(NotImplementedError, match="cis\\(\\)"):
        _calculator(functional="B3LYP").tddft(WATER)


def test_cis_requires_a_geometry_or_scf_results():
    with pytest.raises(TypeError, match="geometry or scf_results"):
        _calculator().cis()


@pytest.mark.slow
def test_results_table_prints_without_error(capsys):
    # quiet=False: output.quiet_context would otherwise suppress the table
    _calculator(quiet=False).cis(WATER, nroots=3, print_results=True)

    printed = capsys.readouterr().out
    assert "QED-CIS Excited States (CQED-RHF reference)" in printed
    assert "<b+b>" in printed
    assert "f (osc)" in printed


@pytest.mark.slow
def test_results_table_names_the_reference_not_a_tddft_approximation(capsys):
    """The banner must not say TDA-DFT -- see the comparison in the .tex."""

    _calculator(functional="B3LYP", quiet=False).cis(
        WATER, nroots=3, print_results=True
    )

    printed = capsys.readouterr().out
    assert "QED-CIS Excited States (CQED-RKS reference)" in printed
    assert "TDA-DFT" not in printed
    assert "TDDFT" not in printed
