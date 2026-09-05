"""Tier 0: cqed_scf.scf and the docs/RESPONSE_REFERENCE CQED-RHF are the same theory.

`scf.py` uses a simplified coherent-state formulation:

    F = H0 + Q_PF + 2J - K - N            E = Tr[(F+H)D] + Enuc

while `docs/RESPONSE_REFERENCE/helper_CQED_RHF.py` carries two extra terms:

    F = H0 + Q_PF + d_PF + 2J - K + 2M - N     E = Tr[(F+H)D] + Enuc + d_c

These are algebraically identical.  Writing d = lambda . mu_el (AO),
x = lambda . <mu_el> = 2 Tr(dD), n = lambda . mu_nuc:

    2M = x d,  d_PF = -x d                  ->  2M + d_PF = 0
    Tr[2M D] = +x^2/2
    Tr[2 d_PF D] = -x^2
    d_c = x^2/2                             ->  the three sum to zero

The QED-CIS working equations are written against the reference's Fock operator
and orbital energies, so they only transfer to this package if the two Fock
matrices agree.  These tests pin that down rather than leaving it as folklore.

See docs/development/QED_RESPONSE_PLAN.md, section 0.1.
"""

import json
import os
import sys

import numpy as np
import psi4
import pytest

from cqed_scf.scf import CQEDSCF


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DIR = os.path.join(REPO_ROOT, "docs/RESPONSE_REFERENCE")
DATA_FILE = os.path.join(REPO_ROOT, "tests", "data", "qed_cis_reference.json")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_DIR),
    reason="docs/RESPONSE_REFERENCE/ is not present",
)

if os.path.isdir(REFERENCE_DIR) and REFERENCE_DIR not in sys.path:
    sys.path.insert(0, REFERENCE_DIR)


GEOM = """
0 1
         O            0.000000000000     0.000000000000    -0.068516219320
         H            0.000000000000    -0.790689573744     0.543701060715
         H            0.000000000000     0.790689573744     0.543701060715
no_reorient
no_com
symmetry c1
"""

OPTIONS = {
    "basis": "sto-3g",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
}

LAMBDA = np.array([0.0, 0.0, 0.05])


@pytest.fixture(scope="module")
def both_scf():
    from helper_CQED_RHF import cqed_rhf

    psi4.core.clean()
    psi4.core.clean_options()
    reference = cqed_rhf(LAMBDA, GEOM, OPTIONS)

    psi4.core.clean()
    psi4.core.clean_options()
    calc = CQEDSCF(
        geometry=GEOM,
        lambda_vector=LAMBDA,
        psi4_options=OPTIONS,
        omega=0.1,
        method="rhf",
    )
    energy, results = calc.run()
    return reference, energy, results


def test_total_energy_matches_reference(both_scf):
    reference, energy, _ = both_scf
    assert energy == pytest.approx(reference["CQED-RHF ENERGY"], abs=1e-9)


def test_density_matches_reference(both_scf):
    reference, _, results = both_scf
    np.testing.assert_allclose(
        results["density"], reference["CQED-RHF DENSITY MATRIX"], atol=1e-8
    )


def test_orbital_energies_match_reference(both_scf):
    """Loose-ish tolerance is expected and is the reference's problem, not ours.

    helper_CQED_RHF has the same break-before-diagonalize pattern this package
    just fixed, so its returned eps belong to the Fock of the previous density.
    It runs without DIIS, so the staleness is mild, but it caps agreement near
    the density convergence threshold.
    """

    reference, _, results = both_scf
    np.testing.assert_allclose(
        results["orbital_energies"], reference["CQED-RHF EPS"], atol=1e-8
    )


def test_fock_cancellation_identity(both_scf):
    """2M + d_PF = 0, which is why scf.py can omit both terms."""

    reference, _, results = both_scf
    D = np.asarray(reference["CQED-RHF DENSITY MATRIX"])
    mints = results["mints"]

    mu_ao = [np.asarray(m) for m in mints.ao_dipole()]
    d = sum(LAMBDA[i] * mu_ao[i] for i in range(3))

    x = 2.0 * np.einsum("pq,pq->", d, D)              # lambda . <mu_el>
    n = float(np.dot(LAMBDA, reference["NUCLEAR DIPOLE MOMENT"]))

    M = d * np.einsum("pq,pq->", d, D)                # J-like DSE mean field
    d_PF = (n - (x + n)) * d                          # coherent-state 1e shift

    np.testing.assert_allclose(2.0 * M + d_PF, np.zeros_like(d), atol=1e-12)


def test_energy_cancellation_identity(both_scf):
    """Tr[2M D] + Tr[2 d_PF D] + d_c = 0, which is why scf.py can omit d_c."""

    reference, _, results = both_scf
    D = np.asarray(reference["CQED-RHF DENSITY MATRIX"])
    mints = results["mints"]

    mu_ao = [np.asarray(m) for m in mints.ao_dipole()]
    d = sum(LAMBDA[i] * mu_ao[i] for i in range(3))

    x = 2.0 * np.einsum("pq,pq->", d, D)
    n = float(np.dot(LAMBDA, reference["NUCLEAR DIPOLE MOMENT"]))

    M = d * np.einsum("pq,pq->", d, D)
    d_PF = (n - (x + n)) * d
    d_c = 0.5 * n**2 - n * (x + n) + 0.5 * (x + n) ** 2

    # d_c as computed here must be the reference's own dipole energy.
    assert d_c == pytest.approx(reference["DIPOLE ENERGY"], abs=1e-12)

    total = (
        np.einsum("pq,pq->", 2.0 * M, D)
        + np.einsum("pq,pq->", 2.0 * d_PF, D)
        + d_c
    )
    assert total == pytest.approx(0.0, abs=1e-12)

    # and each piece is what the algebra says it is
    assert np.einsum("pq,pq->", 2.0 * M, D) == pytest.approx(0.5 * x**2, abs=1e-12)
    assert np.einsum("pq,pq->", 2.0 * d_PF, D) == pytest.approx(-(x**2), abs=1e-12)
    assert d_c == pytest.approx(0.5 * x**2, abs=1e-12)


@pytest.mark.slow
def test_mghp_cqed_rhf_energy_matches_oracle_inference():
    """Cross-check the two QED-CIS oracles against each other through the SCF.

    qed-ci reports total energies; CS_CQED_CIS.py reports energies relative to
    E_CQED-RHF.  Subtracting one from the other predicts E_CQED-RHF without
    either code having reported it.  If this passes, the two oracles share a
    reference and the energy-zero bookkeeping in the plan is right.
    """

    with open(DATA_FILE) as handle:
        data = json.load(handle)

    case = data["mghp_ccpvdz_2.2A"]
    cavity = case["cases"]["cavity"]
    omega = cavity["omega_ev"] / psi4.constants.Hartree_energy_in_eV

    psi4.core.clean()
    psi4.core.clean_options()
    calc = CQEDSCF(
        geometry=case["geometry"],
        lambda_vector=np.array(cavity["lambda_vector"]),
        psi4_options=case["psi4_options"],
        omega=omega,
        method="rhf",
    )
    energy, _ = calc.run()

    assert energy == pytest.approx(cavity["cqed_rhf_energy"], abs=1e-6)
