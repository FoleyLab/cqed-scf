import numpy as np
import psi4
import pytest
from scipy.optimize import OptimizeResult

from cqed_scf import CQEDCalculator
import cqed_scf.drivers as drivers
from cqed_scf.drivers import (
    bfgs_optimize,
    build_reference_translation_rotation_projectors,
    project_cartesian_gradient_remove_translation_rotation,
)
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


def test_step_projector_removes_inverse_hessian_rigid_mode_mixing():
    """Projecting g alone is insufficient when B^-1 mixes subspaces.

    This regression constructs a deliberately dense positive-definite inverse
    Hessian.  Its unprojected action on an already-projected gradient contains
    rigid translation/rotation.  Applying the distinct Cartesian step
    projector must remove that contamination to numerical precision.
    """
    coords_angstrom = np.array(
        [
            [0.000000000000, 0.000000000000, -0.068516219320],
            [0.000000000000, -0.790689573744, 0.543701060715],
            [0.000000000000, 0.790689573744, 0.543701060715],
        ]
    )
    coords_bohr = coords_angstrom * ANGSTROM_TO_BOHR
    masses = np.array([15.999, 1.008, 1.008]) * AMU_TO_AU
    projectors = build_reference_translation_rotation_projectors(
        coords_bohr,
        masses,
    )

    rng = np.random.default_rng(8)
    raw_gradient = rng.normal(size=coords_bohr.size)
    gradient_projected = projectors["gradient_projector"] @ raw_gradient

    # A dense SPD matrix is a stand-in for a BFGS inverse Hessian that has
    # learned coupling between internal and rigid Cartesian directions.
    mixing = rng.normal(size=(coords_bohr.size, coords_bohr.size))
    inverse_hessian = mixing.T @ mixing + np.eye(coords_bohr.size)
    step_before_projection = -inverse_hessian @ gradient_projected
    step_after_projection = (
        projectors["step_projector"] @ step_before_projection
    )

    sqrtm_flat = np.repeat(np.sqrt(masses), 3)
    rigid_basis = projectors["rigid_basis_mass_weighted"]
    rigid_before = rigid_basis.T @ (sqrtm_flat * step_before_projection)
    rigid_after = rigid_basis.T @ (sqrtm_flat * step_after_projection)

    assert np.linalg.norm(rigid_before) > 1e-6
    assert np.linalg.norm(rigid_after) < 1e-10
    np.testing.assert_allclose(
        projectors["step_projector"],
        projectors["gradient_projector"].T,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        projectors["step_projector"] @ projectors["step_projector"],
        projectors["step_projector"],
        atol=1e-12,
        rtol=1e-12,
    )


def test_projected_bfgs_maps_optimizer_steps_and_inverse_hessian(monkeypatch):
    """The optimizer variable may drift, but physical coordinates may not."""

    class FakeCalculator:
        charge = 0
        multiplicity = 1

        def energy_and_gradient(self, geometry, canonical="psi4"):
            # A deliberately non-rigid gradient exercises the right-hand
            # projection applied before SciPy's inverse Hessian.
            return -76.0, np.array(
                [
                    [0.012, -0.018, 0.021],
                    [-0.031, 0.044, -0.016],
                    [0.023, -0.027, 0.035],
                ]
            ), np.zeros(3)

    rng = np.random.default_rng(19)
    optimizer_displacement = rng.normal(size=9)
    mixing = rng.normal(size=(9, 9))
    optimizer_hess_inv = mixing.T @ mixing + np.eye(9)

    def fake_minimize(fun, x0, jac, method, callback, options):
        # Exercise both cached adapters at the reference and at a trial point.
        fun(x0)
        jac(x0)
        callback(optimizer_displacement)
        return OptimizeResult(
            x=optimizer_displacement.copy(),
            fun=fun(optimizer_displacement),
            jac=jac(optimizer_displacement),
            hess_inv=optimizer_hess_inv.copy(),
            success=True,
        )

    monkeypatch.setattr(drivers, "minimize", fake_minimize)

    result, _ = bfgs_optimize(
        calculator=FakeCalculator(),
        geometry=H2O_GEOM,
        project_tr_rot=True,
    )

    mol = psi4.geometry(H2O_GEOM)
    x_reference = mol.geometry().to_array().reshape(-1)
    masses = np.array([mol.mass(i) for i in range(mol.natom())]) * AMU_TO_AU
    projectors = build_reference_translation_rotation_projectors(
        x_reference.reshape(-1, 3),
        masses,
    )

    expected_x = (
        x_reference
        + projectors["step_projector"] @ optimizer_displacement
    )
    expected_hess_inv = (
        projectors["step_projector"]
        @ optimizer_hess_inv
        @ projectors["gradient_projector"]
    )

    np.testing.assert_allclose(result.x, expected_x, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        result.hess_inv,
        expected_hess_inv,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.optimizer_displacement,
        optimizer_displacement,
        atol=0.0,
        rtol=0.0,
    )

    sqrtm_flat = np.repeat(np.sqrt(masses), 3)
    rigid_basis = projectors["rigid_basis_mass_weighted"]
    physical_displacement = result.x - x_reference
    np.testing.assert_allclose(
        rigid_basis.T @ (sqrtm_flat * physical_displacement),
        0.0,
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(result.constraint_residual, 0.0, atol=1e-10)


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
