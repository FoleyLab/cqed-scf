"""Tier 0: the SCF must return canonical CQED orbitals consistent with its Fock.

Before the canonicalization fix, ``CQEDSCF.run()`` broke out of the SCF loop
*before* re-diagonalizing, so ``results["orbital_energies"]`` were eigenvalues
of the previous, DIIS-extrapolated Fock rather than of the returned
``results["F"]``.  Response theory assumes canonical orbitals -- in particular
``F_ia = 0``, which is what makes the CQED Brillouin block vanish -- so this
inconsistency would silently cap agreement with the QED-CIS oracles.

See docs/development/QED_RESPONSE_PLAN.md, sections 0.2 and 0.4.
"""

import numpy as np
import psi4
import pytest

from cqed_scf.scf import CQEDSCF


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

OMEGA = 0.07349864501573


@pytest.fixture(scope="module")
def scf_results():
    psi4.core.clean()
    psi4.core.clean_options()
    calc = CQEDSCF(
        geometry=GEOM,
        lambda_vector=np.array([0.0, 0.0, 0.05]),
        psi4_options=OPTIONS,
        omega=OMEGA,
        method="rhf",
    )
    _, results = calc.run()
    return results


def test_results_dict_exposes_the_response_contract(scf_results):
    """Response drivers must be able to rebuild the Hamiltonian from the dict alone."""

    for key in ("omega", "lambda_vector", "fock_mo", "max_fock_ov"):
        assert key in scf_results, f"results dict is missing {key!r}"

    assert scf_results["omega"] == pytest.approx(OMEGA)
    np.testing.assert_allclose(scf_results["lambda_vector"], [0.0, 0.0, 0.05])


def test_orbital_energies_diagonalize_the_returned_fock(scf_results):
    """This is the regression test for the stale-eps bug."""

    F = scf_results["F"]
    S = np.asarray(scf_results["mints"].ao_overlap())
    eps = scf_results["orbital_energies"]

    A = psi4.core.Matrix.from_array(S)
    A.power(-0.5, 1.0e-16)
    A = np.asarray(A)

    eps_from_F = np.linalg.eigvalsh(A @ F @ A)
    np.testing.assert_allclose(eps, eps_from_F, atol=1e-12)


def test_mo_fock_is_diagonal_and_brillouin_block_vanishes(scf_results):
    C = scf_results["coefficients"]
    F = scf_results["F"]
    eps = scf_results["orbital_energies"]
    ndocc = scf_results["ndocc"]

    F_mo = C.T @ F @ C
    np.testing.assert_allclose(np.diag(F_mo), eps, atol=1e-12)

    off_diagonal = F_mo - np.diag(np.diag(F_mo))
    assert np.max(np.abs(off_diagonal)) < 1e-10

    # The CQED Brillouin condition: <Phi_0|H|Phi_i^a> = sqrt(2) F_ia = 0.
    max_fock_ov = np.max(np.abs(F_mo[:ndocc, ndocc:]))
    assert max_fock_ov < 1e-10
    assert scf_results["max_fock_ov"] == pytest.approx(max_fock_ov, abs=1e-14)


def test_coefficients_are_orthonormal_and_consistent_with_the_density(scf_results):
    C = scf_results["coefficients"]
    D = scf_results["density"]
    S = np.asarray(scf_results["mints"].ao_overlap())
    ndocc = scf_results["ndocc"]

    np.testing.assert_allclose(C.T @ S @ C, np.eye(C.shape[1]), atol=1e-10)

    Cocc = C[:, :ndocc]
    np.testing.assert_allclose(D, Cocc @ Cocc.T, atol=1e-12)
    np.testing.assert_allclose(scf_results["Co"], Cocc, atol=1e-12)
    np.testing.assert_allclose(scf_results["Cv"], C[:, ndocc:], atol=1e-12)


def test_lambda_zero_reduces_to_the_psi4_reference():
    """Sanity anchor: with no coupling the CQED orbitals are the RHF orbitals."""

    psi4.core.clean()
    psi4.core.clean_options()
    calc = CQEDSCF(
        geometry=GEOM,
        lambda_vector=np.zeros(3),
        psi4_options=OPTIONS,
        omega=OMEGA,
        method="rhf",
    )
    energy, results = calc.run()

    assert energy == pytest.approx(results["energy_psi4"], abs=1e-9)
    np.testing.assert_allclose(
        results["orbital_energies"],
        results["canonical_orbital_energies"],
        atol=1e-8,
    )


def test_canonical_orbital_energies_are_not_aliased_to_the_cqed_ones(scf_results):
    """np.asarray on a Psi4 Vector is a view, and the wfn is written to at the end.

    _update_wfn_with_cqed() writes the CQED orbital energies into
    wfn.epsilon_a().  If results["canonical_orbital_energies"] were captured as
    a view of that same buffer, it would silently become a duplicate of
    results["orbital_energies"] -- and QED-SAPT reads it via monomer.py.
    """

    eps_cqed = scf_results["orbital_energies"]
    eps_canonical = scf_results["canonical_orbital_energies"]

    # With lambda = (0, 0, 0.05) the cavity must actually move the orbitals.
    assert np.max(np.abs(eps_cqed - eps_canonical)) > 1e-6
