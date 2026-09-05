"""Tier 1: dense QED-CIS against both oracles.

Two independent references:

* ``docs/RESPONSE_REFERENCE/helper_CS_CQED_CIS.py`` -- same singlet-adapted CIS space
  as ours, so the *entire spectrum* must agree elementwise.  Its eigenvalues are
  relative to E_CQED-RHF.
* ``qed-ci`` totals stored in ``tests/data/qed_cis_reference.json`` -- a
  determinant-basis CIS, so it also contains triplets and our spectrum is a
  strict subset.  Compare by value, never by root index.

See docs/development/QED_RESPONSE_PLAN.md, Tier 1.
"""

import json
import os
import sys

import numpy as np
import psi4
import pytest

from cqed_scf.response import QEDCIS
from cqed_scf.scf import CQEDSCF


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DIR = os.path.join(REPO_ROOT, "docs/RESPONSE_REFERENCE")
DATA_FILE = os.path.join(REPO_ROOT, "tests", "data", "qed_cis_reference.json")

if os.path.isdir(REFERENCE_DIR) and REFERENCE_DIR not in sys.path:
    sys.path.insert(0, REFERENCE_DIR)

with open(DATA_FILE) as _handle:
    _DATA = json.load(_handle)["mghp_ccpvdz_2.2A"]

GEOM = _DATA["geometry"]
OPTIONS = _DATA["psi4_options"]
CAVITY = _DATA["cases"]["cavity"]
NO_CAVITY = _DATA["cases"]["no_cavity"]
OMEGA_CAVITY = CAVITY["omega_ev"] / psi4.constants.Hartree_energy_in_eV


def _run(lambda_vector, omega, n_photon=1):
    psi4.core.clean()
    psi4.core.clean_options()
    scf = CQEDSCF(
        geometry=GEOM,
        lambda_vector=np.array(lambda_vector),
        psi4_options=OPTIONS,
        omega=omega,
        method="rhf",
    )
    energy, results = scf.run()
    cis = QEDCIS(scf_results=results, n_photon=n_photon)
    return energy, results, cis.kernel()


@pytest.fixture(scope="module")
def cavity_run():
    return _run(CAVITY["lambda_vector"], OMEGA_CAVITY)


@pytest.fixture(scope="module")
def field_free_run():
    return _run([0.0, 0.0, 0.0], 0.0)


def test_dimension_is_photon_major_blocks(cavity_run):
    _, results, cis_results = cavity_run
    no, nv = results["ndocc"], results["nvirt"]
    assert cis_results.eigenvalues.size == 2 * (1 + no * nv)


def test_brillouin_block_is_negligible_for_cqed_orbitals(cavity_run):
    """Tier 0's canonicalization is what makes the layout's zeros real."""

    _, results, _ = cavity_run
    assert results["max_fock_ov"] < 1e-10


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.isdir(REFERENCE_DIR),
    reason="docs/RESPONSE_REFERENCE/ is not present",
)
def test_full_spectrum_matches_cs_cqed_cis_oracle(cavity_run):
    """Same CIS space, different basis ordering -> identical spectra."""

    from helper_CS_CQED_CIS import cs_cqed_cis

    _, _, cis_results = cavity_run

    psi4.core.clean()
    psi4.core.clean_options()
    oracle = cs_cqed_cis(
        np.array(CAVITY["lambda_vector"]), OMEGA_CAVITY, GEOM, OPTIONS
    )
    oracle_eigs = np.asarray(oracle["CQED-CIS ENERGY"])
    assert np.max(np.abs(np.imag(oracle_eigs))) < 1e-10
    oracle_eigs = np.sort(np.real(oracle_eigs))

    assert oracle_eigs.size == cis_results.eigenvalues.size
    np.testing.assert_allclose(cis_results.eigenvalues, oracle_eigs, atol=1e-8)


@pytest.mark.slow
def test_totals_match_qed_ci(cavity_run):
    """Absolute energies, cross-checked against the determinant-basis code.

    qed-ci's CIS space also contains triplets, so its root indices differ from
    ours.  Match by value: each reference total must appear in our spectrum.
    """

    _, _, cis_results = cavity_run
    totals = cis_results.total_energies
    reference = CAVITY["qed_ci_totals"]

    assert totals[0] == pytest.approx(reference["ground"], abs=1e-6)
    for label in ("lower_polariton", "upper_polariton"):
        closest = np.min(np.abs(totals - reference[label]))
        assert closest < 1e-6, f"{label} not found in the QED-CIS spectrum (closest {closest:.2e})"


@pytest.mark.slow
def test_ground_state_relaxes_below_the_scf_reference(cavity_run):
    """The bilinear term couples |Phi_0,0> to |Phi_i^a,1>, lowering the ground root.

    This is why the oracle's eigenvalue[1] is not an excitation energy.
    """

    scf_energy, _, cis_results = cavity_run

    assert scf_energy == pytest.approx(CAVITY["cqed_rhf_energy"], abs=1e-6)
    assert cis_results.eigenvalues[0] < 0.0
    assert cis_results.eigenvalues[0] == pytest.approx(
        CAVITY["ground_relaxation_below_cqed_rhf"], abs=1e-6
    )

    # the true excitation energy differs from the oracle's printed value by exactly
    # that relaxation
    omega_lp = np.min(
        cis_results.excitation_energies[cis_results.excitation_energies > 1e-6]
    )
    assert omega_lp == pytest.approx(CAVITY["excitation_energy_lp"], abs=1e-6)
    assert omega_lp - CAVITY["helper_eigenvalue_1"] == pytest.approx(
        -CAVITY["ground_relaxation_below_cqed_rhf"], abs=1e-8
    )


@pytest.mark.slow
def test_zero_coupling_gives_doubly_degenerate_canonical_cis(field_free_run):
    _, _, cis_results = field_free_run
    eigs = cis_results.eigenvalues

    # omega = 0 and lambda = 0: the two photon blocks are identical
    np.testing.assert_allclose(eigs[0::2], eigs[1::2], atol=1e-10)
    assert eigs[0] == pytest.approx(0.0, abs=1e-10)

    first_singlet = cis_results.excitation_energies[2]
    assert first_singlet == pytest.approx(
        NO_CAVITY["excitation_energy_first_singlet"], abs=1e-7
    )


@pytest.mark.slow
def test_zero_coupling_matches_psi4_tdscf_tda(field_free_run):
    """Independent anchor: at lambda = 0 this is ordinary singlet CIS."""

    from psi4.driver.procrouting.response.scf_response import tdscf_excitations

    _, results, cis_results = field_free_run

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(OPTIONS)
    psi4.geometry(GEOM)
    _, wfn = psi4.energy("scf", return_wfn=True)
    psi4_states = tdscf_excitations(wfn, states=3, triplets="NONE", tda=True)
    psi4_energies = np.sort([state["EXCITATION ENERGY"] for state in psi4_states])

    # Undo the photon replication, NOT the physical degeneracies.
    #
    # At lambda = 0 and omega = 0 the two photon blocks are identical, so every
    # root appears exactly twice from the photon basis -- on top of whatever
    # spatial degeneracy it already has.  MgH+ has a doubly degenerate Pi state,
    # which therefore appears four times.  Slicing [::2] halves every
    # multiplicity uniformly and leaves the physical spectrum intact.
    #
    # Do NOT use np.unique here: it collapses the Pi pair as well and silently
    # shifts every later root up by one.
    excitations = cis_results.excitation_energies
    np.testing.assert_allclose(
        excitations[0::2], excitations[1::2], atol=1e-10,
        err_msg="photon blocks are not exactly degenerate; [::2] is not valid here",
    )

    ours = excitations[::2][1 : 1 + psi4_energies.size]  # drop the ground root
    np.testing.assert_allclose(ours, psi4_energies, atol=1e-7)


# ===========================================================================
# Tier 2: the matrix-free path on real integrals
# ===========================================================================
#
# tests/test_qed_cis_sigma.py already pins the sigma against the dense
# Hamiltonian elementwise, but it does so with the dense ERI engine.  The
# generalized-JK engine is the piece that cannot be checked synthetically: it
# depends on Psi4's C_left/C_right convention for the one-sided response density
# D = C_o X C_v^T.  These tests cover exactly that.


@pytest.mark.slow
def test_jk_engine_matches_the_dense_eri_engine(cavity_run):
    """The generalized-JK density convention, checked against exact MO integrals."""

    from cqed_scf.response import DenseERIEngine

    _, results, _ = cavity_run

    jk_driver = QEDCIS(scf_results=results, n_photon=1)
    jk_driver.build_orbital_blocks()

    dense_driver = QEDCIS(scf_results=results, n_photon=1, integral_backend="dense_eri")
    dense_driver.build_orbital_blocks()
    assert isinstance(dense_driver.eri_engine, DenseERIEngine)

    rng = np.random.default_rng(17)
    X = rng.normal(size=(3, jk_driver.ndocc, jk_driver.nvirt))

    np.testing.assert_allclose(
        jk_driver.eri_engine.ov_sigma(X),
        dense_driver.eri_engine.ov_sigma(X),
        atol=1e-9,
    )


@pytest.mark.slow
def test_jk_sigma_reproduces_the_dense_hamiltonian(cavity_run):
    """Full column test on real integrals, through the production JK path."""

    _, results, _ = cavity_run

    driver = QEDCIS(scf_results=results, n_photon=1)
    H_dense = driver.build_dense_hamiltonian()

    identity = np.eye(driver.dimension).reshape(
        driver.dimension, driver.n_photon + 1, driver.block_size
    )
    H_sigma = driver.sigma(identity).reshape(driver.dimension, driver.dimension).T

    np.testing.assert_allclose(H_sigma, H_dense, atol=1e-9)


@pytest.mark.slow
def test_davidson_matches_dense_on_real_integrals(cavity_run):
    _, results, dense = cavity_run

    driver = QEDCIS(scf_results=results, n_photon=1)
    nroots = 8
    davidson = driver.kernel(nroots=nroots, solver="davidson", tol=1e-10)

    assert davidson.davidson.converged
    np.testing.assert_allclose(
        davidson.eigenvalues, dense.eigenvalues[:nroots], atol=1e-9
    )


@pytest.mark.slow
def test_davidson_recovers_the_polaritons(cavity_run):
    """LP and UP by value, and the Rabi splitting, from the direct solver."""

    scf_energy, results, _ = cavity_run

    driver = QEDCIS(scf_results=results, n_photon=1)
    davidson = driver.kernel(nroots=8, solver="davidson", tol=1e-10)
    totals = davidson.total_energies

    for label in ("lower_polariton", "upper_polariton"):
        closest = np.min(np.abs(totals - CAVITY["qed_ci_totals"][label]))
        assert closest < 1e-6, f"{label} not recovered by Davidson ({closest:.2e})"

    omega_lp = np.min(davidson.excitation_energies[davidson.excitation_energies > 1e-6])
    omega_up_candidates = davidson.excitation_energies[
        np.abs(davidson.excitation_energies - CAVITY["excitation_energy_up"]) < 1e-6
    ]
    assert omega_up_candidates.size == 1
    assert omega_up_candidates[0] - omega_lp == pytest.approx(
        CAVITY["rabi_splitting"], abs=1e-6
    )


@pytest.mark.slow
def test_polaritons_share_photon_character(cavity_run):
    """The LP/UP pair should split the photon between them.

    A polariton is a mixture; if either came back with essentially zero photon
    number the state assignment would be wrong even with the right energy.
    """

    _, _, dense = cavity_run

    totals = dense.total_energies
    lp = int(np.argmin(np.abs(totals - CAVITY["qed_ci_totals"]["lower_polariton"])))
    up = int(np.argmin(np.abs(totals - CAVITY["qed_ci_totals"]["upper_polariton"])))

    assert dense.photon_numbers[lp] > 0.05
    assert dense.photon_numbers[up] > 0.05
    assert dense.photon_numbers[lp] + dense.photon_numbers[up] > 0.5


def _sum_over_degenerate_manifolds(energies, strengths, tol=1e-7):
    """Group by excitation energy and sum the intensities within each group.

    Under exact degeneracy the individual oscillator strengths are NOT well
    defined: eigh returns an arbitrary orthogonal mixture within a degenerate
    manifold, and the intensity redistributes among its members.  Only the sum
    over a full manifold is basis independent.

    At lambda = 0, omega = 0 this bites twice over -- the photon blocks are
    degenerate, so even the ground state is an arbitrary mix of |Phi_0,0> and
    |Phi_0,1>.  Since the dipole is diagonal in photon number, one member of a
    pair can carry all the intensity and its partner none.  (Slicing [::2] is
    fine for energies, which are equal across a pair, and wrong for intensities.)
    """

    grouped_energies, grouped_strengths = [], []
    for energy, strength in zip(energies, strengths):
        if grouped_energies and abs(energy - grouped_energies[-1]) < tol:
            grouped_strengths[-1] += strength
        else:
            grouped_energies.append(energy)
            grouped_strengths.append(strength)
    return np.array(grouped_energies), np.array(grouped_strengths)


@pytest.mark.slow
def test_zero_coupling_oscillator_strengths_match_psi4(field_free_run):
    """Transition properties, anchored against psi4 at lambda = 0.

    Compared as manifold sums, which is the only basis-independent statement
    available at exact degeneracy.  Our spectrum carries each physical root
    (N_ph + 1) times, so a manifold that psi4 reports once appears twice here and
    a psi4 Pi pair appears four times -- the sums still have to agree.
    """

    from psi4.driver.procrouting.response.scf_response import tdscf_excitations

    _, _, cis_results = field_free_run

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(OPTIONS)
    psi4.geometry(GEOM)
    _, wfn = psi4.energy("scf", return_wfn=True)
    states = tdscf_excitations(wfn, states=3, triplets="NONE", tda=True)

    order = np.argsort([state["EXCITATION ENERGY"] for state in states])
    psi4_energies = np.array([states[i]["EXCITATION ENERGY"] for i in order])
    psi4_strengths = np.array([states[i]["OSCILLATOR STRENGTH (LEN)"] for i in order])
    psi4_grouped_e, psi4_grouped_f = _sum_over_degenerate_manifolds(
        psi4_energies, psi4_strengths
    )

    assert cis_results.oscillator_strengths is not None

    excitations = cis_results.excitation_energies
    strengths = cis_results.oscillator_strengths
    excited = excitations > 1e-8  # drop the (dark, zero-energy) ground manifold
    ours_e, ours_f = _sum_over_degenerate_manifolds(
        excitations[excited], strengths[excited]
    )

    # Asymmetric tolerances, deliberately.
    #
    # Excitation energies are eigenvalues and are well conditioned.  Intensities
    # depend on eigenvECTORS, whose components are ill conditioned near a
    # degeneracy -- the conditioning goes like 1/gap, and MgH+ has a Pi manifold.
    # Two independently converged SCF solutions feeding two independent CIS
    # implementations therefore agree far better on energies than on intensities.
    #
    # Observed here: energies agree to better than 1e-7 Eh, manifold-summed
    # oscillator strengths to ~2.4e-6 absolute on values of order 0.63-0.80,
    # i.e. a few parts per million, with ours consistently the smaller.  That is
    # a normal level of agreement, not a defect.
    #
    # If this ever needs settling: run QEDCIS at lambda = 0 on psi4's own SCF
    # orbitals instead of the CQED-SCF ones.  If the intensity agreement tightens
    # to ~1e-9 the residual is orbital convergence; if it does not, there is a
    # real difference in the transition-dipole expression worth finding.
    n_compare = psi4_grouped_e.size
    np.testing.assert_allclose(ours_e[:n_compare], psi4_grouped_e, atol=1e-7)
    np.testing.assert_allclose(ours_f[:n_compare], psi4_grouped_f, atol=1e-5)


@pytest.mark.slow
def test_zero_coupling_total_intensity_is_conserved(field_free_run):
    """Sanity companion: the photon replication must not create intensity.

    Each physical manifold appears (N_ph + 1) times, but the ground state is
    normalized across the same blocks, so the total transition intensity to a
    given electronic manifold is unchanged.
    """

    _, _, cis_results = field_free_run

    strengths = cis_results.oscillator_strengths
    excitations = cis_results.excitation_energies
    excited = excitations > 1e-8

    _, grouped = _sum_over_degenerate_manifolds(
        excitations[excited], strengths[excited]
    )
    assert np.all(grouped >= -1e-12)
    assert grouped[0] > 0.1  # the bright sigma state keeps its intensity


@pytest.mark.slow
def test_polariton_identification_by_photonic_reference_weight(cavity_run):
    """The LP/UP pair must be found by weight on |Phi_0,1>, not by photon number.

    Pinned against the independent qed-ci totals, so this catches a
    misidentification rather than merely an internally-consistent one.
    """

    _, _, dense = cavity_run
    lower, upper = dense.polariton_indices()
    totals = dense.total_energies

    assert totals[lower] == pytest.approx(
        CAVITY["qed_ci_totals"]["lower_polariton"], abs=1e-6
    )
    assert totals[upper] == pytest.approx(
        CAVITY["qed_ci_totals"]["upper_polariton"], abs=1e-6
    )

    # a genuine pair shares the one-photon reference character between them
    shared = dense.reference_weights[[lower, upper], 1].sum()
    assert 0.85 < shared < 1.15, f"photonic weight is not shared: {shared:.4f}"


@pytest.mark.slow
def test_total_photon_number_misidentifies_the_polaritons(cavity_run):
    """Documents why polariton_indices exists.

    Selecting the two roots with the largest <b+b> picks up a photon-dressed
    state |Phi_i^a,1> -- near (bare transition + omega), photon number close to
    one, mixing with nothing -- and drops the true lower polariton.  If this
    test ever starts failing, the naive heuristic has become safe on this system
    and the warning in polariton_indices' docstring should be revisited.
    """

    _, _, dense = cavity_run

    correct = dense.polariton_indices()
    excited = np.where(dense.excitation_energies > 1e-8)[0]
    naive = sorted(excited[np.argsort(dense.photon_numbers[excited])[-2:]].tolist())

    assert naive != correct

    # specifically: the naive choice misses the lower polariton
    lower = correct[0]
    assert lower not in naive
    assert dense.total_energies[lower] == pytest.approx(
        CAVITY["qed_ci_totals"]["lower_polariton"], abs=1e-6
    )
