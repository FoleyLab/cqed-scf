"""Psi4 JK-like scaffolds for future Pauli-Fierz SAPT0 terms."""

import numpy as np
from psi4 import core


def _asarray(matrix):
    """Return a NumPy view/copy for Psi4 matrices and array-like inputs."""
    if hasattr(matrix, "np"):
        return np.asarray(matrix.np)
    return np.asarray(matrix)


def _core_matrix(array):
    """Build a Psi4 matrix from a C-contiguous NumPy array."""
    return core.Matrix.from_array(np.ascontiguousarray(array))


class DSEJK:
    """No-op JK-like placeholder for future DSE J/K builds.

    This class mimics the small subset of ``psi4.core.JK`` used by the local
    SAPT JK routines. The actual Pauli-Fierz/DSE physics is intentionally not
    implemented yet.
    """

    def __init__(
        self,
        d_ao=None,
        *,
        j_scale=1.0,
        k_scale=1.0,
        enabled=True,
        return_core_matrices=True,
        metadata=None,
    ):
        self.d_ao = d_ao
        self.j_scale = j_scale
        self.k_scale = k_scale
        self.enabled = enabled
        self.return_core_matrices = return_core_matrices
        self.metadata = {} if metadata is None else dict(metadata)
        self._C_left = []
        self._C_right = []
        self._J = []
        self._K = []

    def C_clear(self):
        """Clear queued coefficient matrices and previous J/K results."""
        self._C_left = []
        self._C_right = []
        self._J = []
        self._K = []

    def C_add(self, C):
        """Queue a symmetric JK build with the same matrix on both sides."""
        self.C_left_add(C)
        self.C_right_add(C)

    def C_left_add(self, C):
        """Queue a left coefficient matrix."""
        self._C_left.append(_asarray(C))

    def C_right_add(self, C):
        """Queue a right coefficient matrix."""
        self._C_right.append(_asarray(C))

    def compute(self):
        """Build zero DSE J/K matrices for each queued density-like product."""
        C_right = self._C_right if self._C_right else self._C_left
        if len(self._C_left) != len(C_right):
            raise ValueError(
                "DSEJK requires matching C_left and C_right build lists; "
                f"got {len(self._C_left)} left and {len(C_right)} right."
            )

        self._J = []
        self._K = []
        for C_left, C_right_i in zip(self._C_left, C_right):
            D = np.ascontiguousarray(C_left @ C_right_i.T)
            J, K = self.jk_from_density(D)
            self._J.append(J)
            self._K.append(K)

    def J(self):
        """Return the most recently built DSE J matrices."""
        return self._J

    def K(self):
        """Return the most recently built DSE K matrices."""
        return self._K

    def jk_from_density(self, D):
        """Return zero J/K matrices with the same shape as ``D``.

        TODO: Implement the Pauli-Fierz/DSE Coulomb-like contribution:
            J_DSE[pq] = d[pq] * sum_rs d[rs] D[rs]

        TODO: Implement the Pauli-Fierz/DSE exchange-like contribution:
            K_DSE[pq] = sum_rs d[pr] d[qs] D[rs]
        """
        D_array = _asarray(D)
        J = np.zeros_like(D_array)
        K = np.zeros_like(D_array)

        if self.return_core_matrices:
            return _core_matrix(J), _core_matrix(K)
        return np.ascontiguousarray(J), np.ascontiguousarray(K)

    def is_active(self):
        """Return whether this scaffold should be considered by callers."""
        return bool(self.enabled and self.d_ao is not None)


class PauliFierzJK:
    """Thin wrapper combining native ERI JK with future DSE JK terms."""

    def __init__(self, eri_jk, dse_jk=None):
        self.eri_jk = eri_jk
        self.dse_jk = dse_jk

    def __getattr__(self, name):
        return getattr(self.eri_jk, name)

    def C_clear(self):
        self.eri_jk.C_clear()
        if self.dse_jk is not None:
            self.dse_jk.C_clear()

    def C_add(self, C):
        self.eri_jk.C_add(C)
        if self.dse_jk is not None:
            self.dse_jk.C_add(C)

    def C_left_add(self, C):
        self.eri_jk.C_left_add(C)
        if self.dse_jk is not None:
            self.dse_jk.C_left_add(C)

    def C_right_add(self, C):
        self.eri_jk.C_right_add(C)
        if self.dse_jk is not None:
            self.dse_jk.C_right_add(C)

    def compute(self):
        self.eri_jk.compute()
        if self.dse_jk is not None:
            self.dse_jk.compute()

    def J(self):
        eri_J = self.eri_jk.J()
        if self.dse_jk is None:
            return eri_J
        return self._sum_matrix_lists(eri_J, self.dse_jk.J())

    def K(self):
        eri_K = self.eri_jk.K()
        if self.dse_jk is None:
            return eri_K
        return self._sum_matrix_lists(eri_K, self.dse_jk.K())

    def print_header(self):
        self.eri_jk.print_header()
        if self.dse_jk is not None:
            core.print_out("  Pauli-Fierz/DSE JK scaffold enabled (zero contribution).\n")

    def native_jk(self):
        """Return the bare Psi4 JK object for code that requires C++ JK."""
        return self.eri_jk

    @staticmethod
    def _sum_matrix_lists(eri_mats, dse_mats):
        if len(eri_mats) != len(dse_mats):
            raise ValueError(
                "Native JK and DSEJK returned different numbers of matrices: "
                f"{len(eri_mats)} != {len(dse_mats)}."
            )

        summed = []
        for eri_mat, dse_mat in zip(eri_mats, dse_mats):
            total = eri_mat.clone()
            total.axpy(1.0, dse_mat)
            summed.append(total)
        return summed


class DSECPHF:
    """No-op placeholder for future DSE response-Hessian terms."""

    def __init__(self, d_ao=None, Cocc=None, Cvir=None, *, enabled=True, metadata=None):
        self.d_ao = d_ao
        self.Cocc = Cocc
        self.Cvir = Cvir
        self.enabled = enabled
        self.metadata = {} if metadata is None else dict(metadata)

    def hx_array(self, X):
        """Return a zero response contribution with the same shape as ``X``."""
        # TODO: Add the DSE analogue of 4J - K - K.T in occupied-virtual space.
        return np.zeros_like(_asarray(X))

    def hx_matrix(self, X_matrix):
        """Return a zero Psi4 matrix matching ``X_matrix``."""
        return _core_matrix(self.hx_array(X_matrix))

    def is_active(self):
        """Return whether this response scaffold should be considered."""
        return bool(self.enabled and self.d_ao is not None)
