import numpy as np
import pytest
from psi4 import core

from cqed_scf.sapt import qed_sapt_jk
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


def test_dse_cavity_terms_match_dense_vt_convention():
    d_A = np.array([[1.0, 0.2], [0.2, -0.4]])
    d_B = np.array([[0.3, -0.1], [-0.1, 0.8]])

    V_A, V_B, constant = qed_sapt_jk._dse_cavity_terms(
        (2, 2),
        d_ao_A=d_A,
        d_ao_B=d_B,
        d_exp_el_A=1.25,
        d_exp_el_B=-0.5,
    )

    np.testing.assert_allclose(V_A.np, -1.25 * d_B)
    np.testing.assert_allclose(V_B.np, 0.5 * d_A)
    assert constant == pytest.approx(-0.625)


def test_dse_cavity_terms_disable_cleanly_and_validate_shapes():
    V_A, V_B, constant = qed_sapt_jk._dse_cavity_terms(
        (2, 2),
        d_ao=np.ones((2, 2)),
        d_exp_el_A=1.0,
        d_exp_el_B=2.0,
        include_cavity_terms=False,
    )

    np.testing.assert_allclose(V_A.np, np.zeros((2, 2)))
    np.testing.assert_allclose(V_B.np, np.zeros((2, 2)))
    assert constant == 0.0

    with pytest.raises(ValueError, match="must match the shared AO shape"):
        qed_sapt_jk._dse_cavity_terms(
            (2, 2),
            d_ao=np.ones((3, 3)),
            d_exp_el_A=1.0,
            d_exp_el_B=2.0,
        )


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


# --------------------------------------------------------------------------
# Reference frames: the internal / interaction split
# --------------------------------------------------------------------------


class FakeMolecule:
    def __init__(self, geometry):
        self._geometry = np.asarray(geometry, dtype=float)

    def geometry(self):
        return self._geometry


class FakeWavefunction:
    """Just enough of a wavefunction for the frame guard, which reads only geometry."""

    def __init__(self, geometry):
        self._molecule = FakeMolecule(geometry)

    def molecule(self):
        return self._molecule


_DIMER_GEOM = np.array([[0.0, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, 0.0, 6.4]])


def _guard(geom_A, geom_B, **kwargs):
    kwargs.setdefault("d_ao_intrinsic_A", None)
    kwargs.setdefault("d_ao_intrinsic_B", None)
    return qed_sapt_jk._check_shared_reference_frame(
        FakeWavefunction(geom_A), FakeWavefunction(geom_B), **kwargs
    )


def test_frame_guard_is_silent_when_both_monomers_share_the_dimer_frame():
    """Ghosted monomers in one dimer frame have bitwise identical coordinates."""
    assert _guard(_DIMER_GEOM, _DIMER_GEOM.copy()) is None


def test_frame_guard_fires_on_divergent_monomer_frames():
    """This is what monomer_reference_frame="monomer_com" produces."""
    shifted = _DIMER_GEOM - np.array([0.0, 0.0, 6.3])

    with pytest.raises(RuntimeError, match="different reference frames") as excinfo:
        _guard(_DIMER_GEOM, shifted)

    # The message must name the likely cause and the way out, or it just
    # relocates the confusion.
    message = str(excinfo.value)
    assert "monomer_com" in message
    assert "d_ao_intrinsic_A" in message
    assert "6.300e+00" in message


def test_frame_guard_stands_down_once_intrinsic_operators_are_supplied():
    """Divergent frames are legitimate -- the guard is about unhandled ones."""
    shifted = _DIMER_GEOM - np.array([0.0, 0.0, 6.3])
    d = np.eye(3)

    assert _guard(shifted, _DIMER_GEOM, d_ao_intrinsic_A=d, d_ao_intrinsic_B=d) is None
    # Either one alone is enough to signal that the caller knows about frames.
    assert _guard(shifted, _DIMER_GEOM, d_ao_intrinsic_A=d) is None
    assert _guard(shifted, _DIMER_GEOM, d_ao_intrinsic_B=d) is None


def test_frame_guard_rejects_mismatched_centre_counts():
    with pytest.raises(RuntimeError, match="different numbers of centres"):
        _guard(_DIMER_GEOM, _DIMER_GEOM[:2])


def test_with_d_ao_preserves_every_other_setting():
    original = DSEJK(
        d_ao=np.array([[1.0, 0.2], [0.2, -0.5]]),
        j_scale=2.0,
        k_scale=-3.0,
        enabled=True,
        return_core_matrices=False,
        metadata={"frame": "dimer"},
    )
    replacement = np.array([[0.0, 1.0], [1.0, 4.0]])

    copy = original.with_d_ao(replacement)

    assert copy is not original
    np.testing.assert_array_equal(copy.d_ao, replacement)
    assert copy.j_scale == original.j_scale
    assert copy.k_scale == original.k_scale
    assert copy.enabled == original.enabled
    assert copy.return_core_matrices == original.return_core_matrices
    assert copy.metadata == original.metadata
    # The original must be untouched: the two frames coexist.
    np.testing.assert_array_equal(original.d_ao, [[1.0, 0.2], [0.2, -0.5]])


def test_with_d_ao_returns_self_for_an_equal_operator():
    """The single-frame path must be a no-op by identity, not merely by value.

    build_sapt_jk_cache asserts on this to keep the default dimer-frame path
    bitwise unchanged rather than re-deriving an equal-but-separate provider.
    """
    d_ao = np.array([[1.0, 0.2], [0.2, -0.5]])
    original = DSEJK(d_ao=d_ao)

    assert original.with_d_ao(d_ao) is original
    assert original.with_d_ao(d_ao.copy()) is original
    assert original.with_d_ao(original.d_ao) is original
    assert original.with_d_ao(d_ao + 1e-13) is not original


def test_with_d_ao_round_trips_through_an_inactive_operator():
    original = DSEJK(d_ao=np.eye(2))

    disabled = original.with_d_ao(None)
    assert disabled is not original
    assert disabled.d_ao is None
    assert not disabled.is_active()
    assert original.is_active()


def test_frame_guard_stands_down_when_the_cavity_is_off():
    """No cavity means no dipole operator anywhere, so no frame dependence.

    Without this the guard would reject a perfectly well-defined lambda = 0
    calculation purely on the geometry of its references.
    """
    shifted = _DIMER_GEOM - np.array([0.0, 0.0, 6.3])

    assert _guard(_DIMER_GEOM, shifted, include_cavity_terms=False) is None
    with pytest.raises(RuntimeError, match="different reference frames"):
        _guard(_DIMER_GEOM, shifted, include_cavity_terms=True)
