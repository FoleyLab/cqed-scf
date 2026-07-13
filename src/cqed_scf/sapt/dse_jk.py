"""JK-like helpers for factorizable Pauli-Fierz dipole self-energy terms."""

import numpy as np
from psi4 import core
import opt_einsum as oe


def _asarray(matrix):
    """Return a NumPy view/copy for Psi4 matrices and array-like inputs."""
    if hasattr(matrix, "np"):
        return np.asarray(matrix.np)
    return np.asarray(matrix)


def _core_matrix(array):
    """Build a Psi4 matrix from a C-contiguous NumPy array."""
    return core.Matrix.from_array(np.ascontiguousarray(array))


class DSEJK:
    """JK-like provider for the separable dipole-dipole operator.

    The class mimics the small subset of ``psi4.core.JK`` used by the local
    SAPT JK routines while evaluating
    ``J[pq] = d[pq] sum_rs d[rs] D[rs]`` and
    ``K[pq] = sum_rs d[pr] d[qs] D[rs]`` without constructing a four-index
    tensor.
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
        self.d_ao = None if d_ao is None else np.asarray(d_ao, dtype=float)
        if self.d_ao is not None:
            self._validate_d_ao(self.d_ao)
        self.j_scale = float(j_scale)
        self.k_scale = float(k_scale)
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
        self._C_left.append(self._validate_coeff(C, "C_left"))

    def C_right_add(self, C):
        """Queue a right coefficient matrix."""
        self._C_right.append(self._validate_coeff(C, "C_right"))

    def compute(self):
        """Build DSE J/K matrices for each queued density-like product."""
        C_right = self._C_right if self._C_right else self._C_left
        if len(self._C_left) != len(C_right):
            raise ValueError(
                "DSEJK requires matching C_left and C_right build lists; "
                f"got {len(self._C_left)} left and {len(C_right)} right."
            )

        self._J = []
        self._K = []
        for C_left, C_right_i in zip(self._C_left, C_right):
            self._validate_coeff_pair(C_left, C_right_i)
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
        """Return DSE J/K matrices for a generalized AO density."""
        D_array = self._validate_density(D)

        if not self.is_active():
            J = np.zeros_like(D_array)
            K = np.zeros_like(D_array)
        else:
            d_ao = self._require_d_ao()
            scalar = oe.contract("pq,pq->", d_ao, D_array, optimize="optimal")
            J = self.j_scale * scalar * d_ao
            K = self.k_scale * (d_ao @ D_array @ d_ao.T)

        if self.return_core_matrices:
            return _core_matrix(J), _core_matrix(K)
        return np.ascontiguousarray(J), np.ascontiguousarray(K)

    def is_active(self):
        """Return whether this DSE provider should be considered by callers."""
        return bool(self.enabled and self.d_ao is not None)

    @staticmethod
    def _validate_d_ao(d_ao):
        if d_ao.ndim != 2 or d_ao.shape[0] != d_ao.shape[1]:
            raise ValueError(f"d_ao must be a square 2D array; got shape {d_ao.shape}.")

    @staticmethod
    def _validate_coeff(C, name):
        C_array = _asarray(C)
        if C_array.ndim != 2:
            raise ValueError(f"{name} must be a 2D coefficient matrix; got shape {C_array.shape}.")
        return np.asarray(C_array, dtype=float)

    def _validate_coeff_pair(self, C_left, C_right):
        if C_left.shape[0] != C_right.shape[0]:
            raise ValueError(
                "DSEJK queued coefficient matrices must have the same row dimension; "
                f"got {C_left.shape} and {C_right.shape}."
            )
        if C_left.shape[1] != C_right.shape[1]:
            raise ValueError(
                "DSEJK generalized density requires matching left/right column counts; "
                f"got {C_left.shape[1]} and {C_right.shape[1]}."
            )
        if self.d_ao is not None and C_left.shape[0] != self.d_ao.shape[0]:
            raise ValueError(
                "DSEJK coefficient row dimension must match d_ao; "
                f"got {C_left.shape[0]} and {self.d_ao.shape[0]}."
            )

    def _validate_density(self, D):
        D_array = np.asarray(_asarray(D), dtype=float)
        if D_array.ndim != 2 or D_array.shape[0] != D_array.shape[1]:
            raise ValueError(f"D must be a square 2D density matrix; got shape {D_array.shape}.")
        if self.d_ao is not None and D_array.shape != self.d_ao.shape:
            raise ValueError(
                "DSEJK density shape must match d_ao; "
                f"got {D_array.shape} and {self.d_ao.shape}."
            )
        return D_array

    def _require_d_ao(self):
        if self.d_ao is None:
            raise ValueError("DSEJK requires d_ao when enabled=True.")
        return self.d_ao


class PauliFierzJK:
    """Adapter combining native ERI JK results with active DSE JK terms."""

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
        self._call_synchronized("C_add", C)

    def C_left_add(self, C):
        self._call_synchronized("C_left_add", C)

    def C_right_add(self, C):
        self._call_synchronized("C_right_add", C)

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
            state = "active" if self.dse_jk.is_active() else "inactive"
            core.print_out(f"  Pauli-Fierz/DSE JK adapter enabled ({state}).\n")

    def native_jk(self):
        """Return the bare Psi4 JK object for code that requires C++ JK."""
        return self.eri_jk

    def _call_synchronized(self, method_name, *args):
        try:
            getattr(self.eri_jk, method_name)(*args)
            if self.dse_jk is not None:
                getattr(self.dse_jk, method_name)(*args)
        except Exception:
            self.C_clear()
            raise

    @staticmethod
    def _sum_matrix_lists(eri_mats, dse_mats):
        if len(eri_mats) != len(dse_mats):
            raise ValueError(
                "Native JK and DSEJK returned different numbers of matrices: "
                f"{len(eri_mats)} != {len(dse_mats)}."
            )

        summed = []
        for eri_mat, dse_mat in zip(eri_mats, dse_mats):
            if eri_mat.shape != dse_mat.shape:
                raise ValueError(
                    "Native JK and DSEJK returned matrices with different shapes: "
                    f"{eri_mat.shape} != {dse_mat.shape}."
                )
            total = eri_mat.clone()
            total.axpy(1.0, dse_mat)
            summed.append(total)
        return summed


class DSECPHF:
    """Matrix-free DSE contribution to the RHF occupied-virtual Hessian."""

    def __init__(
        self,
        d_ao=None,
        Cocc=None,
        Cvir=None,
        *,
        dse_jk=None,
        enabled=True,
        metadata=None,
    ):
        if dse_jk is None:
            dse_jk = DSEJK(
                d_ao=d_ao,
                enabled=enabled,
                return_core_matrices=False,
            )
        self.dse_jk = dse_jk
        self.d_ao = self.dse_jk.d_ao
        self.Cocc = None if Cocc is None else self._validate_coeff(Cocc, "Cocc")
        self.Cvir = None if Cvir is None else self._validate_coeff(Cvir, "Cvir")
        self.enabled = enabled
        self.metadata = {} if metadata is None else dict(metadata)

    def hx_array(self, X):
        """Return the DSE Hessian action in Psi4's ``(nocc, nvir)`` convention."""
        X_array = np.asarray(_asarray(X), dtype=float)
        if X_array.ndim != 2:
            raise ValueError(f"CPHF trial matrix must be 2D; got shape {X_array.shape}.")
        if not self.is_active():
            return np.zeros_like(X_array)

        Cocc, Cvir = self._coefficients()
        expected_shape = (Cocc.shape[1], Cvir.shape[1])
        if X_array.shape != expected_shape:
            raise ValueError(
                "CPHF trial matrix shape must be (nocc, nvir); "
                f"got {X_array.shape}, expected {expected_shape}."
            )

        D_ov = np.ascontiguousarray(Cocc @ X_array @ Cvir.T)
        J_dse, K_dse = self.dse_jk.jk_from_density(D_ov)
        J_dse = _asarray(J_dse)
        K_dse = _asarray(K_dse)
        G_dse = 4.0 * J_dse - K_dse - K_dse.T
        return np.ascontiguousarray(Cocc.T @ G_dse @ Cvir)

    def hx_matrix(self, X_matrix):
        """Return the DSE Hessian action as a Psi4 matrix."""
        return _core_matrix(self.hx_array(X_matrix))

    def is_active(self):
        """Return whether this response contribution should be considered."""
        return bool(self.enabled and self.dse_jk is not None and self.dse_jk.is_active())

    @staticmethod
    def _validate_coeff(C, name):
        C_array = np.asarray(_asarray(C), dtype=float)
        if C_array.ndim != 2:
            raise ValueError(f"{name} must be a 2D coefficient matrix; got shape {C_array.shape}.")
        return C_array

    def _coefficients(self):
        if self.Cocc is None or self.Cvir is None:
            raise ValueError("DSECPHF requires both Cocc and Cvir when active.")
        if self.Cocc.shape[0] != self.Cvir.shape[0]:
            raise ValueError(
                "Cocc and Cvir must have the same AO row dimension; "
                f"got {self.Cocc.shape} and {self.Cvir.shape}."
            )
        if self.d_ao is not None and self.Cocc.shape[0] != self.d_ao.shape[0]:
            raise ValueError(
                "DSECPHF coefficient row dimension must match d_ao; "
                f"got {self.Cocc.shape[0]} and {self.d_ao.shape[0]}."
            )
        return self.Cocc, self.Cvir
