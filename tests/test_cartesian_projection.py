import numpy as np
import psi4
import pytest

from cqed_scf import CQEDCalculator
from cqed_scf.drivers import project_cartesian_gradient_remove_translation_rotation
from cqed_scf.utils import AMU_TO_AU, ANGSTROM_TO_BOHR

H2O_GEOM = """
O  0.000000000000   0.000000000000  -0.068516219320
H  0.000000000000  -0.790689573744   0.543701060715
H  0.000000000000   0.790689573744   0.543701060715
units angstrom
no_reorient
no_com
symmetry c1
"""

PSI4_OPTIONS_WB97X = {
    "basis": "cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none",
}


def test_project_cartesian_gradient_removes_translation_and_rotation():
    coords_angstrom = np.array(
        [
            [0.000000000000, 0.000000000000, -0.068516219320],
            [0.000000000000, -0.790689573744, 0.543701060715],
            [0.000000000000, 0.790689573744, 0.543701060715],
        ]
    )
    coords_bohr = coords_angstrom * ANGSTROM_TO_BOHR
    grad = np.array(
        [
            [0.012, -0.018, 0.021],
            [-0.031, 0.044, -0.016],
            [0.023, -0.027, 0.035],
        ]
    )
    masses = np.array([15.999, 1.008, 1.008]) * AMU_TO_AU

    grad_proj, info = project_cartesian_gradient_remove_translation_rotation(
        coords_bohr,
        grad,
        masses,
        return_diagnostics=True,
    )

    assert grad_proj.shape == grad.shape
    assert np.linalg.norm(grad_proj) <= np.linalg.norm(grad) + 1e-12
    assert np.linalg.norm(info["net_force_proj"]) < 1e-10
    assert np.linalg.norm(info["torque_proj"]) < 1e-10
    assert info["rank"] == 6


def test_energy_and_projected_gradient_wraps_energy_and_gradient(monkeypatch):
    calc = CQEDCalculator(
        lambda_vector=[0.0, 0.0, 0.0],
        psi4_options={},
        omega=0.0,
        charge=0,
        multiplicity=1,
    )
    energy = -76.0
    coupling = np.array([0.1, 0.2, 0.3])
    grad = np.array(
        [
            [0.012, -0.018, 0.021],
            [-0.031, 0.044, -0.016],
            [0.023, -0.027, 0.035],
        ]
    )

    def fake_energy_and_gradient(geometry, canonical="psi4"):
        assert geometry == H2O_GEOM
        assert canonical == "exact"
        return energy, grad.copy(), coupling.copy()

    monkeypatch.setattr(calc, "energy_and_gradient", fake_energy_and_gradient)

    projected_energy, projected_grad, projected_coupling = (
        calc.energy_and_projected_gradient(H2O_GEOM, canonical="exact")
    )

    mol = psi4.geometry(H2O_GEOM)
    coords_bohr = mol.geometry().to_array()
    masses = np.array([mol.mass(i) for i in range(mol.natom())]) * AMU_TO_AU
    expected_grad = project_cartesian_gradient_remove_translation_rotation(
        coords_bohr,
        grad,
        masses,
    )
    sqrtm = np.sqrt(masses)

    assert projected_energy == energy
    np.testing.assert_allclose(projected_coupling, coupling)
    np.testing.assert_allclose(projected_grad, expected_grad, atol=1e-12, rtol=1e-12)
    assert np.linalg.norm(projected_grad / sqrtm[:, None]) <= (
        np.linalg.norm(grad / sqrtm[:, None]) + 1e-12
    )


@pytest.mark.slow
def test_energy_and_projected_gradient_matches_manual_projection_for_cqed_water():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(PSI4_OPTIONS_WB97X)

    calc = CQEDCalculator(
        lambda_vector=[0.0, 0.05, 0.05],
        psi4_options=PSI4_OPTIONS_WB97X,
        omega=0.0,
        density_fitting=True,
        charge=0,
        multiplicity=1,
        functional="wb97x",
    )

    energy_full, full_grad, coupling_full = calc.energy_and_gradient(
        H2O_GEOM,
        canonical="psi4",
    )
    energy_proj, proj_grad, coupling_proj = calc.energy_and_projected_gradient(
        H2O_GEOM,
        canonical="psi4",
    )

    mol = psi4.geometry(H2O_GEOM)
    coords_bohr = mol.geometry().to_array()
    masses = np.array([mol.mass(i) for i in range(mol.natom())]) * AMU_TO_AU
    manual_proj_grad, info = project_cartesian_gradient_remove_translation_rotation(
        coords_bohr,
        full_grad,
        masses,
        return_diagnostics=True,
    )
    sqrtm = np.sqrt(masses)

    np.testing.assert_allclose(energy_proj, energy_full, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(coupling_proj, coupling_full, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(proj_grad, manual_proj_grad, atol=1e-10, rtol=1e-10)
    assert proj_grad.shape == full_grad.shape
    assert np.linalg.norm(proj_grad / sqrtm[:, None]) <= (
        np.linalg.norm(full_grad / sqrtm[:, None]) + 1e-12
    )
    assert np.linalg.norm(info["net_force_proj"]) < 1e-10
    assert np.linalg.norm(info["torque_proj"]) < 1e-10
    assert info["rank"] == 6
