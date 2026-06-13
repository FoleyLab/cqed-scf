import numpy as np

from cqed_scf.drivers import project_cartesian_gradient_remove_translation_rotation
from cqed_scf.utils import AMU_TO_AU, ANGSTROM_TO_BOHR


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
