import numpy as np
import psi4

from cqed_rhf.scf import CQEDSCF
from cqed_rhf.calculator import CQEDRHFCalculator


GEOM = """
1 1
H 0.0 0.0 -0.124721924329
H 0.0 -1.429937284075 0.989898061465
H 0.0 1.429937284075 0.989898061465
units bohr
no_reorient
no_com
symmetry c1
"""

test_func = "wb97x"

options = {
    "basis": "6-311g",
    "scf_type": "df",
    "e_convergence": 1e-12,
    "d_convergence": 1e-10,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none",
}

lam_vec = np.array([0.0, 0.0, 0.0])


def coords_to_bohr_geom(symbols, coords_bohr, charge=1, multiplicity=1):
    lines = [f"{charge} {multiplicity}"]
    for s, xyz in zip(symbols, coords_bohr):
        lines.append(f"{s} {xyz[0]:.16f} {xyz[1]:.16f} {xyz[2]:.16f}")
    lines.append("units bohr")
    lines.append("no_reorient")
    lines.append("no_com")
    lines.append("symmetry c1")
    return "\n".join(lines)


def energy(geom, lam, functional_choice):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(options)

    calc = CQEDSCF(
        geometry=geom,
        lambda_vector=lam,
        psi4_options=options,
        omega=0.07349864501573,
        density_fitting=True,
        functional=functional_choice,
    )

    E, _ = calc.run()
    return E


def test_fd_gradient_wb97x():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(options)

    calc = CQEDRHFCalculator(
        lambda_vector=lam_vec,
        psi4_options=options,
        omega=0.1,
        charge=1,
        multiplicity=1,
        density_fitting=True,
        functional=test_func,
        debug=False,
    )

    E, grad, _ = calc.energy_and_gradient(GEOM)

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(options)

    mol = psi4.geometry(GEOM)
    coords = np.array(mol.geometry())
    symbols = [mol.symbol(i) for i in range(mol.natom())]

    grad_psi4 = np.array(psi4.gradient("wb97x"))

    delta = 5e-4
    fd_grad = np.zeros_like(coords)

    for A in range(coords.shape[0]):
        for xyz in range(3):
            coords_p = coords.copy()
            coords_m = coords.copy()

            coords_p[A, xyz] += delta
            coords_m[A, xyz] -= delta

            geom_p = coords_to_bohr_geom(symbols, coords_p)
            geom_m = coords_to_bohr_geom(symbols, coords_m)

            Ep = energy(geom_p, lam_vec, test_func)
            Em = energy(geom_m, lam_vec, test_func)

            fd_grad[A, xyz] = (Ep - Em) / (2 * delta)

    diff_fd_vs_analytic = fd_grad - grad
    diff_fd_vs_psi4 = fd_grad - grad_psi4
    diff_analytic_vs_psi4 = grad - grad_psi4

    rms1 = np.sqrt(np.mean(diff_fd_vs_analytic**2))
    rms2 = np.sqrt(np.mean(diff_fd_vs_psi4**2))
    rms3 = np.sqrt(np.mean(diff_analytic_vs_psi4**2))

    assert rms3 < 1e-6
    assert rms2 < 1e-5
    assert rms1 < 1e-5
