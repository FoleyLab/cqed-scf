"""Tier 3: Kohn-Sham references, QED-CIS-N convergence, and density fitting.

Note on naming: QED-CIS on a QED-Kohn-Sham reference is NOT the Tamm-Dancoff
approximation to linear-response QED-TDDFT (TDA-PF, Yang et al., JCP 155,
064107 (2021)).  It is a Fock-basis configuration interaction; see
docs/qed_cis_formalism.tex, "Relationship to linear-response QED-TDDFT".
The psi4 tdscf comparison below is valid only at lambda = 0, where both
reduce to ordinary TDA.

The Kohn-Sham two-electron block carries an exchange-correlation kernel that has
no MO-integral form, so it is routed through Psi4's own ``twoel_Hx_full`` --
the same routine Psi4's TDSCF engine uses -- and combined the same way:

    A X = F X + C_o^T ( 2 J_like - K_like ) C_v

with the XC kernel already inside ``J_like`` and the hybrid exchange fraction
already inside ``K_like``.  Two independent anchors check that we inherited that
convention correctly:

* for a Hartree-Fock reference the new engine must agree with the JK engine
  validated in Tier 2 -- this checks the combination without involving any
  functional at all;
* at lambda = 0 a KS run must reproduce Psi4's own ``tdscf_excitations``
  (tda=True) on the same reference -- this checks the XC kernel itself,
  including its factor.
"""

import numpy as np
import psi4
import pytest

from cqed_scf.response import JKERIEngine, Psi4HxERIEngine, QEDCIS
from cqed_scf.scf import CQEDSCF


WATER = """
0 1
         O            0.000000000000     0.000000000000    -0.068516219320
         H            0.000000000000    -0.790689573744     0.543701060715
         H            0.000000000000     0.790689573744     0.543701060715
no_reorient
no_com
symmetry c1
"""

OMEGA = 0.07349864501573


def _run(lambda_vector, functional=None, density_fitting=False, basis="sto-3g",
         omega=OMEGA, n_photon=1, **cis_kwargs):
    psi4.core.clean()
    psi4.core.clean_options()
    options = {
        "basis": basis,
        "scf_type": "df" if density_fitting else "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
        "save_jk": True,
    }
    if functional is not None:
        options.update({"points": 3, "dft_spherical_points": 590})

    scf = CQEDSCF(
        geometry=WATER,
        lambda_vector=np.array(lambda_vector),
        psi4_options=options,
        omega=omega,
        density_fitting=density_fitting,
        method="rhf" if functional is None else "rks",
        functional=functional,
    )
    energy, results = scf.run()
    driver = QEDCIS(scf_results=results, n_photon=n_photon, **cis_kwargs)
    return energy, results, driver


def _psi4_tda(functional, basis="sto-3g", states=3):
    """Excitation energies from Psi4's own TDA on the cavity-free reference."""

    from psi4.driver.procrouting.response.scf_response import tdscf_excitations

    psi4.core.clean()
    psi4.core.clean_options()
    options = {"basis": basis, "scf_type": "pk", "e_convergence": 1e-10,
               "d_convergence": 1e-10, "save_jk": True}
    if functional is not None:
        options.update({"points": 3, "dft_spherical_points": 590})
    psi4.set_options(options)
    psi4.geometry(WATER)
    _, wfn = psi4.energy("scf" if functional is None else functional, return_wfn=True)
    computed = tdscf_excitations(wfn, states=states, triplets="NONE", tda=True)
    return np.sort([state["EXCITATION ENERGY"] for state in computed])


def _undo_photon_replication(excitations, count):
    """At omega = 0 every root is doubled by the photon basis; [::2] halves every
    multiplicity uniformly and preserves physical degeneracies."""

    np.testing.assert_allclose(excitations[0::2], excitations[1::2], atol=1e-10)
    return excitations[::2][1 : 1 + count]


# ---------------------------------------------------------------------------
# the combination convention, checked without any functional
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_psi4_hx_engine_agrees_with_jk_engine_for_hartree_fock():
    """Anchors Psi4HxERIEngine against the Tier-2-validated JK path."""

    _, results, _ = _run([0.0, 0.0, 0.05])

    jk_driver = QEDCIS(scf_results=results, n_photon=1, integral_backend="jk")
    jk_driver.build_orbital_blocks()
    hx_driver = QEDCIS(scf_results=results, n_photon=1, integral_backend="psi4_hx")
    hx_driver.build_orbital_blocks()

    assert isinstance(jk_driver.eri_engine, JKERIEngine)
    assert isinstance(hx_driver.eri_engine, Psi4HxERIEngine)

    rng = np.random.default_rng(23)
    X = rng.normal(size=(3, jk_driver.ndocc, jk_driver.nvirt))

    np.testing.assert_allclose(
        hx_driver.eri_engine.ov_sigma(X), jk_driver.eri_engine.ov_sigma(X), atol=1e-9
    )


# ---------------------------------------------------------------------------
# Kohn-Sham references
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("functional", ["PBE", "B3LYP"])
def test_ks_zero_coupling_matches_psi4_tda(functional):
    """The XC kernel itself, including its factor.

    PBE is a pure functional (no K-like term at all); B3LYP is a hybrid, so the
    two cases exercise both branches of the J/K split.
    """

    _, _, driver = _run([0.0, 0.0, 0.0], functional=functional, omega=0.0)
    results = driver.kernel(solver="dense")

    expected = _psi4_tda(functional)
    ours = _undo_photon_replication(results.excitation_energies, expected.size)

    np.testing.assert_allclose(ours, expected, atol=1e-6)


@pytest.mark.slow
def test_ks_reference_is_auto_detected():
    _, _, driver = _run([0.0, 0.0, 0.05], functional="B3LYP")
    driver.build_orbital_blocks()

    assert driver.is_ks
    assert isinstance(driver.eri_engine, Psi4HxERIEngine)


@pytest.mark.slow
def test_ks_hamiltonian_is_symmetric_and_davidson_matches_dense():
    _, _, driver = _run([0.0, 0.0, 0.05], functional="B3LYP")

    H = driver.build_dense_hamiltonian()
    assert np.max(np.abs(H - H.T)) < 1e-9

    dense = driver.kernel(solver="dense")
    davidson = driver.kernel(nroots=5, solver="davidson", tol=1e-9)
    assert davidson.davidson.converged
    np.testing.assert_allclose(davidson.eigenvalues, dense.eigenvalues[:5], atol=1e-8)


@pytest.mark.slow
def test_ks_cavity_shifts_the_spectrum():
    """A sanity check that the cavity is actually doing something for DFT."""

    _, _, field_free = _run([0.0, 0.0, 0.0], functional="B3LYP")
    _, _, coupled = _run([0.0, 0.0, 0.05], functional="B3LYP")

    free = field_free.kernel(solver="dense").excitation_energies
    cavity = coupled.kernel(solver="dense").excitation_energies

    assert np.max(np.abs(free[:6] - cavity[:6])) > 1e-5


# ---------------------------------------------------------------------------
# QED-CIS-N convergence
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_photon_truncation_converges():
    """Convergence of the ground state with respect to N_ph.

    Tracked through the GROUND-STATE energy, which is unambiguously root 0 at
    every N_ph.  Comparing roots by index across different N_ph is invalid:
    adding a photon sector inserts new states into the spectrum, and with
    omega = 0.0735 Eh the two-photon manifold falls below water's electronic
    excitations, so a given index refers to a different physical state at
    N_ph = 1 and N_ph = 2.

    Assertions are on the shape of the convergence rather than an absolute
    threshold: the ground state must decrease monotonically (each larger space
    contains the smaller one, so this is variational), and the increments must
    shrink by at least an order of magnitude across the sweep.
    """

    _, results, _ = _run([0.0, 0.0, 0.05])

    ground = {}
    for n_photon in (1, 2, 3, 4):
        driver = QEDCIS(scf_results=results, n_photon=n_photon)
        ground[n_photon] = float(driver.kernel(solver="dense").eigenvalues[0])

    # variational: a larger photon space can only lower the ground state
    for n_photon in (1, 2, 3):
        assert ground[n_photon + 1] <= ground[n_photon] + 1e-12

    increments = [abs(ground[n + 1] - ground[n]) for n in (1, 2, 3)]

    assert increments[1] <= increments[0] + 1e-14
    assert increments[2] <= increments[1] + 1e-14
    assert increments[2] < increments[0] / 10.0, (
        f"photon truncation is not converging: increments {increments}"
    )


@pytest.mark.slow
def test_photon_truncation_only_adds_states_above():
    """Adding photon sectors lowers roots variationally; it must not remove any."""

    _, results, _ = _run([0.0, 0.0, 0.05])

    one = QEDCIS(scf_results=results, n_photon=1).kernel(solver="dense")
    two = QEDCIS(scf_results=results, n_photon=2).kernel(solver="dense")

    assert two.eigenvalues.size > one.eigenvalues.size
    assert two.eigenvalues[0] <= one.eigenvalues[0] + 1e-10


# ---------------------------------------------------------------------------
# density fitting
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_density_fitting_reproduces_conventional_integrals():
    """DF flows through psi4.core.JK untouched; the DSE terms are DF-free
    already, being separable."""

    _, _, conventional = _run([0.0, 0.0, 0.05])
    _, _, fitted = _run([0.0, 0.0, 0.05], density_fitting=True)

    exact = conventional.kernel(nroots=5, solver="davidson", tol=1e-9)
    approx = fitted.kernel(nroots=5, solver="davidson", tol=1e-9)

    np.testing.assert_allclose(
        approx.excitation_energies, exact.excitation_energies, atol=2e-4
    )


@pytest.mark.slow
def test_density_fitting_with_ks_reference_runs():
    _, _, driver = _run([0.0, 0.0, 0.05], functional="PBE", density_fitting=True)
    results = driver.kernel(nroots=4, solver="davidson", tol=1e-9)

    assert results.davidson.converged
    assert np.all(np.diff(results.eigenvalues) >= -1e-12)
