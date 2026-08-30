"""Tier 2: the matrix-free sigma reproduces the dense Hamiltonian exactly.

The column test is the decisive one: apply sigma to every unit vector, assemble
the implied matrix, and compare elementwise against build_dense_hamiltonian().
If it passes, every sign, factor and index in the sigma is right -- there is no
residual doubt to chase later through a Davidson convergence failure.

Like tests/test_qed_cis_layout.py these run on synthetic orbital blocks, so they
need no SCF and no psi4 integrals.  The dense ERI engine stands in for the
production JK path; tests/test_qed_cis_dense.py covers the JK route on real
integrals.
"""

import numpy as np
import pytest

from cqed_scf.davidson import davidson_solve
from cqed_scf.response import DenseERIEngine, QEDCIS


def _build(no=3, nv=4, n_photon=1, seed=11, d_scale=0.05, omega=0.1745, F_ov=None):
    rng = np.random.default_rng(seed)
    nmo = no + nv

    B = rng.normal(size=(6, nmo, nmo)) * 0.1
    B = 0.5 * (B + B.transpose(0, 2, 1))
    eri = np.einsum("Ppq,Prs->pqrs", B, B)

    d = rng.normal(size=(nmo, nmo)) * d_scale
    d = 0.5 * (d + d.T)
    eps = np.sort(rng.normal(size=nmo))

    cis = QEDCIS(n_photon=n_photon, integral_backend="dense_eri")
    cis.ndocc, cis.nvirt, cis.nmo, cis.n_ov = no, nv, nmo, no * nv
    cis.omega, cis.scf_energy = omega, 0.0
    cis.F_oo, cis.F_vv = np.diag(eps[:no]), np.diag(eps[no:])
    cis.F_ov = np.zeros((no, nv)) if F_ov is None else F_ov
    cis.d_oo, cis.d_ov, cis.d_vv = d[:no, :no], d[:no, no:], d[no:, no:]
    cis.ovov, cis.oovv = eri[:no, no:, :no, no:], eri[:no, :no, no:, no:]
    cis._prepared = True
    return cis


def _implied_matrix(cis):
    identity = np.eye(cis.dimension).reshape(
        cis.dimension, cis.n_photon + 1, cis.block_size
    )
    return cis.sigma(identity).reshape(cis.dimension, cis.dimension).T


@pytest.mark.parametrize(
    "kwargs, label",
    [
        ({}, "canonical CQED-HF orbitals"),
        ({"n_photon": 0}, "N_ph = 0"),
        ({"n_photon": 3, "no": 2, "nv": 3}, "N_ph = 3"),
        ({"d_scale": 0.0}, "zero coupling"),
    ],
)
def test_sigma_reproduces_the_dense_hamiltonian(kwargs, label):
    cis = _build(**kwargs)
    np.testing.assert_allclose(
        _implied_matrix(cis), cis.build_dense_hamiltonian(), atol=1e-12, err_msg=label
    )


def test_sigma_reproduces_the_dense_hamiltonian_with_live_brillouin_block():
    """Non-canonical orbitals: F_ov != 0 must flow through the sigma too."""

    rng = np.random.default_rng(5)
    cis = _build(F_ov=rng.normal(size=(3, 4)) * 0.01)
    np.testing.assert_allclose(
        _implied_matrix(cis), cis.build_dense_hamiltonian(), atol=1e-12
    )


def test_sigma_batches_consistently():
    """One batched call must equal many single calls -- batching is the whole
    point of the JK path, so a broadcasting slip here would be costly."""

    cis = _build(n_photon=2)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(7, cis.n_photon + 1, cis.block_size))

    np.testing.assert_allclose(
        cis.sigma(X), np.array([cis.sigma(X[k]) for k in range(7)]), atol=1e-13
    )


def test_sigma_accepts_a_single_unbatched_vector():
    cis = _build()
    X = np.zeros((cis.n_photon + 1, cis.block_size))
    X[0, 1] = 1.0
    assert cis.sigma(X).shape == X.shape


def test_diagonal_matches_the_dense_diagonal():
    cis = _build()
    np.testing.assert_allclose(
        cis.hamiltonian_diagonal(), np.diag(cis.build_dense_hamiltonian()), atol=1e-13
    )


@pytest.mark.parametrize(
    "kwargs, label",
    [
        ({}, "standard"),
        ({"n_photon": 3, "no": 2, "nv": 3, "omega": 0.21}, "N_ph = 3"),
        ({"d_scale": 0.0}, "zero coupling, degenerate"),
    ],
)
def test_davidson_matches_dense_eigh(kwargs, label):
    cis = _build(**kwargs)
    dense = cis.kernel(solver="dense")
    nroots = min(6, cis.dimension)
    davidson = cis.kernel(nroots=nroots, solver="davidson", tol=1e-10)

    assert davidson.davidson.converged
    np.testing.assert_allclose(
        davidson.eigenvalues, dense.eigenvalues[:nroots], atol=1e-9, err_msg=label
    )


def test_davidson_does_one_eri_action_per_iteration():
    """Cost claim: all trial vectors and all photon blocks go into one action."""

    cis = _build(n_photon=2)
    result = cis.kernel(nroots=4, solver="davidson", tol=1e-10)
    assert cis.eri_engine.n_builds == result.davidson.n_iterations


def test_photonic_seed_is_present_in_the_guess():
    """The seed is cheap insurance rather than a proven necessity (see the
    initial_guess docstring), but it must actually be in the guess."""

    cis = _build(n_photon=2)
    guess = cis.initial_guess(nroots=3)
    for n in range(cis.n_photon + 1):
        index = n * cis.block_size
        assert np.any(guess[:, index] == 1.0), f"|Phi_0,{n}> missing from the guess"


def test_photon_dominated_root_is_found_without_its_seed():
    """Documented behavior, not an aspiration.

    Constructed so an 86%-photon root ranks 32nd of 66 by diagonal and falls
    outside the guess, with an engine that supplies no two-electron diagonal --
    which is the production JK situation.  Davidson recovers it anyway.  This
    test records that; if it ever starts failing, the photonic seeding in
    initial_guess is doing real work and its docstring should be upgraded.
    """

    class _NoDiagonalEngine(DenseERIEngine):
        def ov_diagonal(self):
            return None

    no, nv = 4, 8
    rng = np.random.default_rng(9)
    nmo = no + nv
    B = rng.normal(size=(6, nmo, nmo)) * 0.55
    B = 0.5 * (B + B.transpose(0, 2, 1))
    eri = np.einsum("Ppq,Prs->pqrs", B, B)
    eps = np.concatenate([np.linspace(-0.30, -0.20, no), np.linspace(0.02, 0.35, nv)])
    d = rng.normal(size=(nmo, nmo)) * 0.04
    d = 0.5 * (d + d.T)

    cis = QEDCIS(n_photon=1, integral_backend="dense_eri")
    cis.ndocc, cis.nvirt, cis.nmo, cis.n_ov = no, nv, nmo, no * nv
    cis.omega, cis.scf_energy = 0.62, 0.0
    cis.F_oo, cis.F_vv = np.diag(eps[:no]), np.diag(eps[no:])
    cis.F_ov = np.zeros((no, nv))
    cis.d_oo, cis.d_ov, cis.d_vv = d[:no, :no], d[:no, no:], d[no:, no:]
    cis.ovov, cis.oovv = eri[:no, no:, :no, no:], eri[:no, :no, no:, no:]
    cis._prepared = True
    cis._eri_engine = _NoDiagonalEngine(cis.ovov, cis.oovv)

    dense = cis.kernel(solver="dense")
    assert dense.photon_numbers[2] > 0.8  # the root really is photon dominated

    diagonal = cis.hamiltonian_diagonal()
    photonic = cis.block_size
    indices = np.argsort(diagonal)[:8]
    assert photonic not in set(indices.tolist())  # and really is outside the guess

    bare = np.zeros((len(indices), cis.dimension))
    for row, index in enumerate(indices):
        bare[row, index] = 1.0

    result = davidson_solve(cis._matvec, diagonal, bare, 4, tol=1e-10)
    assert result.converged
    np.testing.assert_allclose(result.eigenvalues, dense.eigenvalues[:4], atol=1e-9)


def test_results_carry_polaritonic_character():
    cis = _build(n_photon=2)
    results = cis.kernel(solver="dense")

    weights = results.reference_weights.sum(axis=1) + results.singles_weights.sum(axis=1)
    np.testing.assert_allclose(weights, 1.0, atol=1e-12)
    assert results.photon_numbers.min() > -1e-12
    assert results.photon_numbers.max() < results.n_photon + 1e-12
    # no cartesian dipole was supplied, so transition properties are absent
    assert results.transition_dipoles is None


def test_unknown_solver_is_rejected():
    with pytest.raises(ValueError, match="dense.*davidson|davidson"):
        _build().kernel(solver="lanczos")


# ---------------------------------------------------------------------------
# Tier 3: the Kohn-Sham column-build path
#
# A KS reference has no explicit MO-integral form for its two-electron block --
# the XC kernel is not an integral -- so the block is assembled by applying the
# integral engine to unit vectors.  These tests exercise that plumbing with a
# mock engine, so they cover the transpose convention and the interaction with
# lazy MO integrals without needing Psi4 or a quadrature grid.
# ---------------------------------------------------------------------------


class _MockKernelEngine:
    """Stands in for Psi4HxERIEngine: an action with no explicit integral form."""

    requires_column_build = True

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)
        self.n_builds = 0

    def ov_sigma(self, X):
        self.n_builds += 1
        X = np.asarray(X)
        flat = X.reshape(-1, self.matrix.shape[0])
        return (flat @ self.matrix.T).reshape(X.shape)

    def ov_diagonal(self):
        return None


def _build_ks(matrix, n_photon=1, no=3, nv=4, seed=11):
    rng = np.random.default_rng(seed)
    nmo = no + nv
    d = rng.normal(size=(nmo, nmo)) * 0.05
    d = 0.5 * (d + d.T)
    eps = np.sort(rng.normal(size=nmo))

    cis = QEDCIS(n_photon=n_photon)
    cis.ndocc, cis.nvirt, cis.nmo, cis.n_ov = no, nv, nmo, no * nv
    cis.omega, cis.scf_energy = 0.1745, 0.0
    cis.F_oo, cis.F_vv = np.diag(eps[:no]), np.diag(eps[no:])
    cis.F_ov = np.zeros((no, nv))
    cis.d_oo, cis.d_ov, cis.d_vv = d[:no, :no], d[:no, no:], d[no:, no:]
    cis.is_ks = True
    cis._eri_engine = _MockKernelEngine(matrix)
    cis._prepared = True
    return cis


def test_column_build_recovers_the_two_electron_block_not_its_transpose():
    """Pinned with a deliberately NON-symmetric probe.

    A symmetric probe cannot distinguish M from M^T, so it would pass even with
    the transpose convention inverted -- and the error would only surface later
    as an asymmetric Hamiltonian on a real KS run.
    """

    nov = 3 * 4
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(nov, nov))

    recovered = _build_ks(matrix)._two_electron_block()

    np.testing.assert_allclose(recovered, matrix, atol=1e-13)
    assert not np.allclose(recovered, matrix.T)  # the probe really is asymmetric


def test_ks_path_sigma_reproduces_the_dense_hamiltonian():
    nov = 3 * 4
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(nov, nov))
    matrix = 0.5 * (matrix + matrix.T)

    cis = _build_ks(matrix)
    np.testing.assert_allclose(
        _implied_matrix(cis), cis.build_dense_hamiltonian(), atol=1e-12
    )


def test_ks_path_davidson_matches_dense():
    nov = 3 * 4
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(nov, nov))
    matrix = 0.5 * (matrix + matrix.T)

    cis = _build_ks(matrix)
    dense = cis.kernel(solver="dense")
    davidson = cis.kernel(nroots=5, solver="davidson", tol=1e-10)

    assert davidson.davidson.converged
    np.testing.assert_allclose(davidson.eigenvalues, dense.eigenvalues[:5], atol=1e-9)


def test_ks_path_never_materialises_mo_integrals():
    """The O(N^4) transformation must stay unbuilt on the matrix-free path."""

    nov = 3 * 4
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(nov, nov))
    matrix = 0.5 * (matrix + matrix.T)

    cis = _build_ks(matrix)
    cis.kernel(nroots=4, solver="davidson", tol=1e-10)
    cis.build_dense_hamiltonian()

    assert cis.ovov is None and cis.oovv is None
