"""Tier 1: dense QED-CIS against both oracles.

Two independent references:

* ``RESPONSE_REFERENCE/helper_CS_CQED_CIS.py`` -- same singlet-adapted CIS space
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
REFERENCE_DIR = os.path.join(REPO_ROOT, "RESPONSE_REFERENCE")
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
    reason="RESPONSE_REFERENCE/ is not present (it is untracked)",
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
