import numpy as np
import pytest
from psi4 import core

from cqed_scf.sapt.dse_jk import DSECPHF, DSEJK, PauliFierzJK


class FakeJK:
    def __init__(self):
        self._C_left = []
        self._C_right = []
        self._J = []
        self._K = []

    def C_clear(self):
        self._C_left = []
        self._C_right = []
        self._J = []
        self._K = []

    def C_add(self, C):
        self.C_left_add(C)
        self.C_right_add(C)

    def C_left_add(self, C):
        self._C_left.append(C)

    def C_right_add(self, C):
        self._C_right.append(C)

    def compute(self):
        self._J = []
        self._K = []
        for C_left in self._C_left:
            nrow = C_left.shape[0]
            self._J.append(core.Matrix.from_array(np.ones((nrow, nrow))))
            self._K.append(core.Matrix.from_array(2.0 * np.ones((nrow, nrow))))

    def J(self):
        return self._J

    def K(self):
        return self._K

    def print_header(self):
        pass


class MismatchedFakeJK(FakeJK):
    def compute(self):
        self._J = [core.Matrix.from_array(np.ones((3, 3)))]
        self._K = [core.Matrix.from_array(np.ones((3, 3)))]


def test_dsejk_compute_matches_rank_one_exchange_formula():
    d = np.eye(2)
    C_L = np.array([[1.0], [0.0]])
    C_R = np.array([[0.0], [1.0]])
    D = C_L @ C_R.T

    dse_jk = DSEJK(d_ao=d)
    dse_jk.C_left_add(C_L)
    dse_jk.C_right_add(C_R)
    dse_jk.compute()

    expected_J = np.einsum("pq,rs,rs->pq", d, d, D)
    expected_K = d @ D @ d.T

    np.testing.assert_allclose(dse_jk.J()[0].np, expected_J)
    np.testing.assert_allclose(dse_jk.K()[0].np, expected_K)


def test_dsejk_applies_scales_and_supports_disabled_zero_builds():
    d = np.array([[1.0, 0.2], [0.3, -0.4]])
    D = np.array([[0.1, 0.5], [-0.2, 0.7]])

    dse_jk = DSEJK(d_ao=d, j_scale=2.0, k_scale=-0.5, return_core_matrices=False)
    J, K = dse_jk.jk_from_density(D)

    np.testing.assert_allclose(J, 2.0 * np.einsum("pq,pq->", d, D) * d)
    np.testing.assert_allclose(K, -0.5 * (d @ D @ d.T))

    disabled = DSEJK(d_ao=d, enabled=False, return_core_matrices=False)
    J0, K0 = disabled.jk_from_density(D)
    np.testing.assert_allclose(J0, np.zeros_like(D))
    np.testing.assert_allclose(K0, np.zeros_like(D))


def test_dsejk_validates_shapes():
    with pytest.raises(ValueError, match="d_ao must be a square"):
        DSEJK(d_ao=np.ones((2, 3)))

    dse_jk = DSEJK(d_ao=np.eye(2))
    dse_jk.C_left_add(np.ones((2, 2)))
    dse_jk.C_right_add(np.ones((2, 1)))
    with pytest.raises(ValueError, match="matching left/right column counts"):
        dse_jk.compute()

    with pytest.raises(ValueError, match="density shape must match d_ao"):
        dse_jk.jk_from_density(np.eye(3))


def test_pauli_fierz_jk_adds_dse_without_mutating_native():
    native_jk = FakeJK()
    d = np.eye(2)
    dse_jk = DSEJK(d_ao=d)
    pf_jk = PauliFierzJK(native_jk, dse_jk=dse_jk)

    pf_jk.C_add(np.eye(2))
    pf_jk.compute()

    J_native_before = native_jk.J()[0].np.copy()
    K_native_before = native_jk.K()[0].np.copy()
    J_pf = pf_jk.J()[0]
    K_pf = pf_jk.K()[0]
    D = np.eye(2)
    expected_J = J_native_before + np.einsum("pq,rs,rs->pq", d, d, D)
    expected_K = K_native_before + d @ D @ d.T

    np.testing.assert_allclose(J_pf.np, expected_J)
    np.testing.assert_allclose(K_pf.np, expected_K)
    np.testing.assert_allclose(native_jk.J()[0].np, J_native_before)
    np.testing.assert_allclose(native_jk.K()[0].np, K_native_before)
    assert pf_jk.native_jk() is native_jk


def test_pauli_fierz_jk_rejects_mismatched_native_and_dse_shapes():
    pf_jk = PauliFierzJK(MismatchedFakeJK(), dse_jk=DSEJK(d_ao=np.eye(2)))
    pf_jk.C_add(np.eye(2))
    pf_jk.compute()

    with pytest.raises(ValueError, match="different shapes"):
        pf_jk.J()


def test_pauli_fierz_jk_clears_queues_when_coefficient_addition_fails():
    native_jk = FakeJK()
    pf_jk = PauliFierzJK(native_jk, dse_jk=DSEJK(d_ao=np.eye(2)))

    with pytest.raises(ValueError, match="2D coefficient matrix"):
        pf_jk.C_left_add(np.ones(2))

    assert native_jk._C_left == []
    assert pf_jk.dse_jk._C_left == []


def test_dse_cphf_matches_dense_orbital_hessian_block_and_is_linear():
    d = np.array(
        [
            [0.8, 0.1, -0.2],
            [0.1, -0.4, 0.3],
            [-0.2, 0.3, 0.5],
        ]
    )
    Cocc = np.array([[1.0], [0.0], [0.0]])
    Cvir = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    X = np.array([[0.2, -0.7]])

    dse_cphf = DSECPHF(d_ao=d, Cocc=Cocc, Cvir=Cvir)
    hx = dse_cphf.hx_array(X)

    d_oo = Cocc.T @ d @ Cocc
    d_ov = Cocc.T @ d @ Cvir
    d_vv = Cvir.T @ d @ Cvir
    dense = (
        4.0 * d_ov * np.einsum("OV,OV->", d_ov, X)
        - d_oo @ X @ d_vv
        - d_ov @ X.T @ d_ov
    )

    np.testing.assert_allclose(hx, dense)
    np.testing.assert_allclose(dse_cphf.hx_array(3.0 * X), 3.0 * hx)


def test_dse_cphf_disabled_returns_zero_matrix():
    dse_cphf = DSECPHF(d_ao=np.eye(2), enabled=False)
    X = core.Matrix.from_array(np.arange(6.0).reshape(2, 3))

    hx = dse_cphf.hx_matrix(X)

    np.testing.assert_allclose(hx.np, np.zeros((2, 3)))
