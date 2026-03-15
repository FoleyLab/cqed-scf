import numpy as np
import psi4

from cqed_rhf.scf import CQEDSCF
from cqed_rhf.gradients import CQEDRHFGradient


GEOM = """
0 1
O 0.0 0.0 -0.124721924329
H 0.0 -1.429937284075 0.989898061465
H 0.0 1.429937284075 0.989898061465
units bohr
no_reorient
no_com
symmetry c1
"""


def energy(geom, lam):

    options = {
        "basis": "sto-3g",
        "scf_type": "df",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    calc = CQEDSCF(
        geometry=geom,
        lambda_vector=lam,
        psi4_options=options,
        omega=0.07349864501573,
        density_fitting=True,
        functional="PBE",
    )

    E, _ = calc.run()
    return E


def test_fd_gradient():

    psi4.core.clean()
    psi4.core.clean_options()

    lam = np.array([0.0, 0.0, 0.05])

    calc = CQEDSCF(
        geometry=GEOM,
        lambda_vector=lam,
        psi4_options={
            "basis": "sto-3g",
            "scf_type": "df",
        },
        omega=0.07349864501573,
        density_fitting=True,
        functional="PBE",
    )

    E, data = calc.run()

    grad = CQEDRHFGradient(lam).compute(data)
    grad = np.array(grad)

    mol = psi4.geometry(GEOM)
    coords = np.array(mol.geometry())

    delta = 1e-4
    fd_grad = np.zeros_like(coords)

    for A in range(coords.shape[0]):
        for xyz in range(3):

            coords_p = coords.copy()
            coords_m = coords.copy()

            coords_p[A, xyz] += delta
            coords_m[A, xyz] -= delta

            mol_p = mol.clone()
            mol_m = mol.clone()

            mol_p.set_geometry(psi4.core.Matrix.from_array(coords_p))
            mol_m.set_geometry(psi4.core.Matrix.from_array(coords_m))

            Ep = energy(mol_p.create_psi4_string_from_molecule(), lam)
            Em = energy(mol_m.create_psi4_string_from_molecule(), lam)

            fd_grad[A, xyz] = (Ep - Em) / (2 * delta)

    diff = fd_grad - grad

    rms = np.sqrt(np.mean(diff**2))

    assert rms < 1e-5
