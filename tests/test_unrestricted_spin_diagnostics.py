import numpy as np
import pytest

from cqed_scf.uscf import _spin_diagnostics


def test_restricted_singlet_has_zero_spin():
    Cocc = np.eye(3)[:, :2]

    s_squared, contamination = _spin_diagnostics(Cocc, Cocc, np.eye(3), 2, 2)

    assert s_squared == pytest.approx(0.0, abs=1.0e-14)
    assert contamination == pytest.approx(0.0, abs=1.0e-14)


def test_pure_doublet_has_expected_spin():
    Ca_occ = np.eye(3)[:, :2]
    Cb_occ = np.eye(3)[:, :1]

    s_squared, contamination = _spin_diagnostics(
        Ca_occ, Cb_occ, np.eye(3), 2, 1
    )

    assert s_squared == pytest.approx(0.75, abs=1.0e-14)
    assert contamination == pytest.approx(0.0, abs=1.0e-14)


def test_complex_orbitals_use_hermitian_overlap_norm():
    phase = np.exp(0.37j)
    Ca_occ = np.array([[phase], [0.0j]])
    Cb_occ = np.array([[1.0j], [0.0j]])

    s_squared, contamination = _spin_diagnostics(
        Ca_occ, Cb_occ, np.eye(2, dtype=complex), 1, 1
    )

    assert isinstance(s_squared, float)
    assert isinstance(contamination, float)
    assert s_squared == pytest.approx(0.0, abs=1.0e-14)
    assert contamination == pytest.approx(0.0, abs=1.0e-14)


@pytest.mark.parametrize("nalpha, nbeta", [(5, 4), (9, 7), (3, 3)])
def test_contamination_is_relative_to_target_spin(nalpha, nbeta):
    nmo = max(nalpha, nbeta) + 1
    Ca_occ = np.eye(nmo)[:, :nalpha]
    Cb_occ = np.eye(nmo)[:, :nbeta]

    s_squared, contamination = _spin_diagnostics(
        Ca_occ, Cb_occ, np.eye(nmo), nalpha, nbeta
    )
    target = 0.5 * abs(nalpha - nbeta)

    assert contamination == pytest.approx(
        s_squared - target * (target + 1.0), abs=1.0e-14
    )
