import numpy as np
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


def test_dsejk_compute_returns_zero_core_matrices():
    dse_jk = DSEJK(d_ao=np.eye(2))
    dse_jk.C_left_add(np.array([[1.0], [0.0]]))
    dse_jk.C_right_add(np.array([[0.0], [1.0]]))

    dse_jk.compute()

    assert len(dse_jk.J()) == 1
    assert len(dse_jk.K()) == 1
    np.testing.assert_allclose(dse_jk.J()[0].np, np.zeros((2, 2)))
    np.testing.assert_allclose(dse_jk.K()[0].np, np.zeros((2, 2)))


def test_pauli_fierz_jk_adds_zero_dse_without_mutating_native():
    native_jk = FakeJK()
    dse_jk = DSEJK(d_ao=np.eye(2))
    pf_jk = PauliFierzJK(native_jk, dse_jk=dse_jk)

    pf_jk.C_add(np.eye(2))
    pf_jk.compute()

    J_native_before = native_jk.J()[0].np.copy()
    J_pf = pf_jk.J()[0]
    K_pf = pf_jk.K()[0]

    np.testing.assert_allclose(J_pf.np, np.ones((2, 2)))
    np.testing.assert_allclose(K_pf.np, 2.0 * np.ones((2, 2)))
    np.testing.assert_allclose(native_jk.J()[0].np, J_native_before)
    assert pf_jk.native_jk() is native_jk


def test_dse_cphf_returns_zero_matrix():
    dse_cphf = DSECPHF(d_ao=np.eye(2))
    X = core.Matrix.from_array(np.arange(6.0).reshape(2, 3))

    hx = dse_cphf.hx_matrix(X)

    np.testing.assert_allclose(hx.np, np.zeros((2, 3)))
