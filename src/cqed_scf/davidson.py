"""Block Davidson-Liu eigensolver for real symmetric, matrix-free problems.

Written for QED-CIS but deliberately free of any cavity physics: it needs only a
batched matrix-vector product, an approximate diagonal for preconditioning, and
a set of guess vectors.

Design notes that matter for polaritonic spectra:

* **Preconditioned corrections are normalized before the independence test.**
  Testing raw norms makes the solver mistake a small correction for a dependent
  one and stagnate just short of tolerance -- eigenvalues look converged to
  1e-13 while residuals plateau around 1e-6.
* **Restarts are thick.**  Collapsing to exactly ``nroots`` vectors discards the
  nearly-converged directions just outside the requested set and the solver
  thrashes; ``max_subspace`` is clamped upward to leave room to expand after a
  restart.
* **Degeneracies are the normal case, not an edge case.**  At ``lambda = 0``
  every root is ``(N_ph + 1)``-fold degenerate, and molecular point-group
  degeneracies multiply on top of that.  The expansion space is therefore
  orthonormalized with a double Gram-Schmidt pass and a linear-dependence drop
  threshold, and the solver expands more vectors than requested roots.
* **Do not collapse degenerate roots.**  The solver returns exactly ``nroots``
  eigenpairs including repeats; deduplicating a spectrum cannot distinguish a
  basis replication from a physically degenerate state.
* **Tight thresholds.**  Rabi splittings scale with lambda and can be ~1e-4 Eh,
  so the default residual tolerance is 1e-8 rather than the 1e-4 typical of
  TDDFT drivers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class DavidsonResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray      # (dim, nroots), columns
    converged: bool
    n_iterations: int
    residual_norms: np.ndarray
    n_matvec: int = 0
    subspace_size: int = 0
    history: list = field(default_factory=list)


def _orthonormalize(candidates, basis, drop_tol):
    """Double Gram-Schmidt `candidates` against `basis` and against each other.

    `basis` is (k, dim) with orthonormal rows; `candidates` is a list of (dim,).
    Vectors that lose their norm are dropped -- that is the linear-dependence
    guard that keeps degenerate blocks from poisoning the subspace matrix.
    """

    kept = []
    for vector in candidates:
        v = np.array(vector, dtype=float, copy=True)

        # Normalize BEFORE projecting.  The drop test below must measure linear
        # independence -- what fraction of the direction is new -- not the
        # magnitude of the vector.  Preconditioned residuals shrink as the
        # solver converges, so testing raw norms makes the solver mistake a
        # small correction for a dependent one and stagnate just short of the
        # tolerance.
        norm = np.linalg.norm(v)
        if norm == 0.0:
            continue
        v /= norm

        for _ in range(2):  # twice is enough; once is not, near degeneracies
            if basis is not None and len(basis):
                v -= basis.T @ (basis @ v)
            for u in kept:
                v -= u * (u @ v)
        norm = np.linalg.norm(v)
        if norm > drop_tol:
            kept.append(v / norm)
    return kept


def davidson_solve(
    matvec: Callable[[np.ndarray], np.ndarray],
    diagonal: np.ndarray,
    guess: np.ndarray,
    nroots: int,
    tol: float = 1e-8,
    max_iterations: int = 100,
    max_subspace: Optional[int] = None,
    drop_tol: float = 1e-6,
    denominator_floor: float = 1e-8,
) -> DavidsonResult:
    """Lowest `nroots` eigenpairs of a real symmetric operator.

    Parameters
    ----------
    matvec
        Batched product: takes ``(nvec, dim)`` and returns ``(nvec, dim)``.
        Batching is the point -- one call per iteration lets the caller build
        all its J/K matrices together.
    diagonal
        Approximate diagonal, used only for preconditioning.  It does not have
        to be exact; an orbital-energy-difference diagonal converges fine.
    guess
        ``(nguess, dim)`` starting vectors.  Under-sized or rank-deficient
        guesses are topped up from the lowest diagonal entries.
    max_subspace
        Expansion-space ceiling.  Clamped upward to a value that leaves room to
        expand after a thick restart -- a smaller request is a performance knob
        that would otherwise turn into a convergence failure.
    """

    diagonal = np.asarray(diagonal, dtype=float).ravel()
    dim = diagonal.size
    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be at least 1")
    if nroots > dim:
        raise ValueError(f"nroots={nroots} exceeds the problem dimension {dim}")

    # Thick restart: on collapse we keep more Ritz vectors than we return, which
    # preserves the nearly-converged directions just outside the requested set.
    # Collapsing to exactly nroots discards them and the solver thrashes.
    n_keep = min(max(2 * nroots, nroots + 4), dim)

    if max_subspace is None:
        max_subspace = min(dim, max(20 * nroots, nroots + 30))
    max_subspace = min(int(max_subspace), dim)
    # A subspace with no room to expand after a restart stagnates: the
    # preconditioner keeps regenerating the direction the restart just threw
    # away, and residuals plateau well above tolerance even though the
    # eigenvalues look converged.  Clamp to a workable minimum rather than
    # letting a too-small value fail silently.
    max_subspace = max(max_subspace, min(dim, max(4 * nroots + 10, n_keep + 2 * nroots)))

    guess = np.atleast_2d(np.asarray(guess, dtype=float))
    if guess.shape[1] != dim:
        raise ValueError(f"guess vectors have length {guess.shape[1]}, expected {dim}")

    new = _orthonormalize(list(guess), None, drop_tol)
    if len(new) < nroots:
        # top up with the lowest-diagonal unit vectors we are not already using
        for index in np.argsort(diagonal):
            if len(new) >= nroots:
                break
            unit = np.zeros(dim)
            unit[index] = 1.0
            new.extend(_orthonormalize([unit], np.array(new), drop_tol))
    if len(new) < nroots:
        raise RuntimeError("could not build enough linearly independent guess vectors")

    V = np.zeros((0, dim))
    W = np.zeros((0, dim))
    n_matvec = 0
    history = []
    residual_norms = np.full(nroots, np.inf)
    theta = np.zeros(nroots)
    X = np.zeros((nroots, dim))
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        new_block = np.array(new)
        W_new = np.asarray(matvec(new_block))
        if W_new.shape != new_block.shape:
            raise ValueError(
                f"matvec returned shape {W_new.shape}, expected {new_block.shape}"
            )
        n_matvec += new_block.shape[0]

        V = np.vstack([V, new_block])
        W = np.vstack([W, W_new])

        subspace = V @ W.T
        subspace = 0.5 * (subspace + subspace.T)  # symmetrize away roundoff
        eigenvalues, vectors = np.linalg.eigh(subspace)

        theta = eigenvalues[:nroots]
        y = vectors[:, :nroots]
        X = y.T @ V                       # Ritz vectors, (nroots, dim)
        residuals = y.T @ W - theta[:, None] * X
        residual_norms = np.linalg.norm(residuals, axis=1)
        history.append((iteration, V.shape[0], float(np.max(residual_norms))))

        if np.max(residual_norms) < tol:
            converged = True
            break

        # collapse before expanding, so the new directions are orthogonalized
        # against the collapsed basis rather than the discarded one
        if V.shape[0] + nroots > max_subspace:
            # Exact thick restart, no extra matvec: Ritz vectors are already
            # orthonormal (eigh gives orthonormal columns, V orthonormal rows),
            # and their images are the same linear combination of the old ones.
            keep = min(n_keep, V.shape[0])
            Y = vectors[:, :keep]
            V = Y.T @ V
            W = Y.T @ W

        corrections = []
        for k in range(nroots):
            if residual_norms[k] < tol:
                continue
            denominator = theta[k] - diagonal
            small = np.abs(denominator) < denominator_floor
            if np.any(small):
                denominator = denominator.copy()
                denominator[small] = denominator_floor
            corrections.append(residuals[k] / denominator)

        new = _orthonormalize(corrections, V, drop_tol)
        if not new:
            break  # stagnated: no new independent directions

    return DavidsonResult(
        eigenvalues=theta.copy(),
        eigenvectors=X.T.copy(),
        converged=converged,
        n_iterations=iteration,
        residual_norms=residual_norms,
        n_matvec=n_matvec,
        subspace_size=V.shape[0],
        history=history,
    )
