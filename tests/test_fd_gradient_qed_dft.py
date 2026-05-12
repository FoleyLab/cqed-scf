import numpy as np
import psi4

from cqed_scf.calculator import CQEDCalculator


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

# testing functional
test_func = "wb97x"

options = {
    "basis": "sto-3g",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}
lam_vec = np.array([0, 0, 0.0])

def energy(geom, lam, functional_choice):
    calc = CQEDCalculator(
        lambda_vector=lam,
        psi4_options=options,
        omega=0.07349864501573,
        density_fitting=True,
        functional=functional_choice,
    )
    return calc.energy(geom)


def test_fd_gradient_pbe():

    psi4.core.clean()
    psi4.core.clean_options()

    calc = CQEDCalculator(
        lambda_vector=lam_vec,
        psi4_options=options,
        omega=0.1,
        charge=0,
        multiplicity=1,
        density_fitting=True,
        functional="PBE",
        debug=True,
    )

    E, grad, _ = calc.energy_and_gradient(GEOM)

    # --- reset before FD reference ---
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(options)
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

            Ep = energy(mol_p.create_psi4_string_from_molecule(), lam_vec, "PBE")
            Em = energy(mol_m.create_psi4_string_from_molecule(), lam_vec, "PBE")

            fd_grad[A, xyz] = (Ep - Em) / (2 * delta)

    diff = fd_grad - grad

    rms = np.sqrt(np.mean(diff**2))

    assert rms < 1e-5

def test_fd_gradient_wb97x():

    psi4.core.clean()
    psi4.core.clean_options()

    psi4.set_memory("4 GB")
    psi4.set_options(options)

    calc = CQEDCalculator(
        lambda_vector=lam_vec,
        psi4_options=options,
        omega=0.1,
        charge=0,
        multiplicity=1,
        density_fitting=True,
        functional=test_func,
        debug=True,
    )

    # analytic grad from our code
    E, grad, _ = calc.energy_and_gradient(GEOM)

    # --- reset before FD reference ---
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(options)
    mol = psi4.geometry(GEOM)
    coords = np.array(mol.geometry())

    # analytic grad from psi4
    grad_psi4 = psi4.gradient("wb97x")


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

            Ep = energy(mol_p.create_psi4_string_from_molecule(), lam_vec, test_func)
            Em = energy(mol_m.create_psi4_string_from_molecule(), lam_vec, test_func)

            fd_grad[A, xyz] = (Ep - Em) / (2 * delta)

    diff1 = fd_grad - grad
    diff2 = fd_grad - grad_psi4
    diff3 = grad - grad_psi4

    rms1 = np.sqrt(np.mean(diff1**2))
    rms2 = np.sqrt(np.mean(diff2**2))
    rms3 = np.sqrt(np.mean(diff3**2))

    assert rms3 < 1e-5
    #assert rms2 < 1e-5
    #assert rms1 < 1e-5
