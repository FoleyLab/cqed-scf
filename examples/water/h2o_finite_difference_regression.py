import numpy as np
import psi4

from cqed_rhf import CQEDSCF
from cqed_rhf import CQEDRHFGradient


# ---------------------------------------------------------
# Geometry (fixed frame)
# ---------------------------------------------------------

geometry = """
0 1
O  0.000000  0.000000  0.000000
H  0.000000  0.757000  0.586000
H  0.000000 -0.757000  0.586000
no_reorient
no_com
symmetry c1
"""


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "6-311g",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}


# ---------------------------------------------------------
# CQED parameters
# ---------------------------------------------------------

lambda_vector = np.array([0.05, 0.05, 0.05])  # start with lambda = 0
omega = 0.0


# ---------------------------------------------------------
# Compute analytic gradient
# ---------------------------------------------------------

calc = CQEDSCF(
    geometry=geometry,
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    functional="PBE",
)

E, scf_data = calc.run()

gradient_engine = CQEDRHFGradient(lambda_vector, canonical="psi4", debug=False)

grad_qed = gradient_engine.compute(scf_data)

grad_analytic = np.array(grad_qed)


# ---------------------------------------------------------
# Finite difference gradient
# ---------------------------------------------------------

delta = 1e-4

mol = psi4.geometry(geometry)
coords = np.array(mol.geometry())

natom = coords.shape[0]

grad_fd = np.zeros_like(coords)

print("\nRunning finite difference gradient check\n")

for atom in range(natom):
    for xyz in range(3):

        coords_p = coords.copy()
        coords_m = coords.copy()

        coords_p[atom, xyz] += delta
        coords_m[atom, xyz] -= delta

        mol_p = mol.clone()
        mol_m = mol.clone()

        mol_p.set_geometry(psi4.core.Matrix.from_array(coords_p))
        mol_m.set_geometry(psi4.core.Matrix.from_array(coords_m))

        geom_p = mol_p.create_psi4_string_from_molecule()
        geom_m = mol_m.create_psi4_string_from_molecule()

        calc_p = CQEDSCF(
            geometry=geom_p,
            lambda_vector=lambda_vector,
            psi4_options=psi4_options,
            omega=omega,
            density_fitting=True,
            functional="PBE",
        )

        calc_m = CQEDSCF(
            geometry=geom_m,
            lambda_vector=lambda_vector,
            psi4_options=psi4_options,
            omega=omega,
            density_fitting=True,
            functional="PBE",
        )

        Ep, _ = calc_p.run()
        Em, _ = calc_m.run()

        grad_fd[atom, xyz] = (Ep - Em) / (2 * delta)


# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

diff = grad_fd - grad_analytic

rms = np.sqrt(np.mean(diff**2))
max_err = np.max(np.abs(diff))

print("\n==============================")
print("Finite Difference Validation")
print("==============================")

print("Analytic gradient:")
print(grad_analytic)

print("\nFinite difference gradient:")
print(grad_fd)

print("\nDifference:")
print(diff)

print("\nRMS error:", rms)
print("Max error:", max_err)


tol = 1e-5

if rms < tol:
    print("\nPASS: analytic gradient validated")
else:
    print("\nWARNING: gradient discrepancy detected")
