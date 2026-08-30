"""Tier 2: the block Davidson solver, exercised away from any cavity physics.

Degenerate clusters are the point here.  At lambda = 0 every QED-CIS root is
(N_ph+1)-fold degenerate and molecular symmetry multiplies on top of that, so a
solver that quietly returns one representative of a cluster is useless.
"""

import numpy as np
import pytest

from cqed_scf.davidson import davidson_solve


def _matrix_with_degeneracies(n=400, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigenvalues = np.sort(rng.normal(size=n)) * 5.0
    eigenvalues[3:6] = eigenvalues[3]    # 3-fold cluster
    eigenvalues[10:12] = eigenvalues[10]  # 2-fold cluster
    A = Q @ np.diag(eigenvalues) @ Q.T
    return 0.5 * (A + A.T)


@pytest.fixture(scope="module")
def problem():
    A = _matrix_with_degeneracies()
    return A, np.linalg.eigvalsh(A)


@pytest.mark.parametrize("nroots", [1, 3, 8, 15])
def test_matches_dense_eigh(problem, nroots):
    A, exact = problem
    diagonal = np.diag(A)
    guess = np.eye(A.shape[0])[np.argsort(diagonal)[: 2 * nroots]]

    result = davidson_solve(lambda V: V @ A, diagonal, guess, nroots, tol=1e-9)

    assert result.converged
    np.testing.assert_allclose(result.eigenvalues, exact[:nroots], atol=1e-9)


def test_degenerate_cluster_returned_with_full_multiplicity(problem):
    """A cluster must come back repeated, not collapsed to one representative."""

    A, exact = problem
    diagonal = np.diag(A)
    guess = np.eye(A.shape[0])[np.argsort(diagonal)[:24]]

    # 12 roots, so that BOTH clusters (indices 3-5 and 10-11) lie inside the
    # requested set -- asking for 8 would put the 2-fold cluster out of range.
    result = davidson_solve(lambda V: V @ A, diagonal, guess, 12, tol=1e-9)

    assert result.converged
    assert np.sum(np.abs(result.eigenvalues - exact[3]) < 1e-9) == 3
    assert np.sum(np.abs(result.eigenvalues - exact[10]) < 1e-9) == 2


def test_eigenvectors_are_orthonormal(problem):
    A, _ = problem
    diagonal = np.diag(A)
    guess = np.eye(A.shape[0])[np.argsort(diagonal)[:16]]

    result = davidson_solve(lambda V: V @ A, diagonal, guess, 8, tol=1e-9)
    V = result.eigenvectors
    np.testing.assert_allclose(V.T @ V, np.eye(8), atol=1e-10)


def test_residuals_actually_reach_tolerance(problem):
    """Guards the normalization bug: corrections must be normalized before the
    linear-dependence test, or residuals plateau around 1e-6 while the
    eigenvalues look converged and the solver reports failure."""

    A, _ = problem
    diagonal = np.diag(A)
    guess = np.eye(A.shape[0])[np.argsort(diagonal)[:16]]

    result = davidson_solve(lambda V: V @ A, diagonal, guess, 8, tol=1e-9)

    assert result.converged
    assert np.max(result.residual_norms) < 1e-9


def test_tiny_max_subspace_is_clamped_not_obeyed(problem):
    """A subspace with no room to expand after a restart cannot converge, so the
    solver raises the ceiling rather than failing silently."""

    A, exact = problem
    diagonal = np.diag(A)
    guess = np.eye(A.shape[0])[np.argsort(diagonal)[:16]]

    result = davidson_solve(
        lambda V: V @ A, diagonal, guess, 8, tol=1e-9, max_subspace=1
    )

    assert result.converged
    np.testing.assert_allclose(result.eigenvalues, exact[:8], atol=1e-9)


def test_undersized_guess_is_topped_up(problem):
    A, exact = problem
    diagonal = np.diag(A)
    result = davidson_solve(
        lambda V: V @ A, diagonal, np.eye(A.shape[0])[:1], 3, tol=1e-9
    )
    assert result.converged
    np.testing.assert_allclose(result.eigenvalues, exact[:3], atol=1e-9)


def test_rejects_impossible_root_counts(problem):
    A, _ = problem
    diagonal = np.diag(A)
    with pytest.raises(ValueError):
        davidson_solve(lambda V: V @ A, diagonal, np.eye(A.shape[0])[:4], 0)
    with pytest.raises(ValueError):
        davidson_solve(
            lambda V: V @ A, diagonal, np.eye(A.shape[0])[:4], A.shape[0] + 1
        )
