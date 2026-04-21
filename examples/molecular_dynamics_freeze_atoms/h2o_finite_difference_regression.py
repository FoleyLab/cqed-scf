import numpy as np
import psi4

from cqed_rhf.calculator import CQEDRHFCalculator
from cqed_rhf import CQEDRHFGradient
from cqed_rhf.scf import CQEDSCF


# ---------------------------------------------------------
# Geometry (fixed frame)
# ---------------------------------------------------------

geometry = """
0 1
C   2.8948252698  2.5912813816  0.2470756673
C   2.5537006291  2.5014046660 -1.0626123399
C   1.9521790572  1.2984619119 -1.4129636179
C   2.6202434391  1.6970197119  1.2068537007
H   3.5382447893  3.4704332497  0.3255964120
H   2.8778947116  3.3115067684 -1.7246463335
H   2.9390023637  1.5862128526  2.3872432154
N   1.4659553978  1.1573493493 -2.8082487684
O   1.7368463769  2.0602473282 -3.6581567049
O   0.6979675702  0.2266864234 -2.8670851244
C   2.1983094489  0.4403896320  0.8256146446
H   2.1239748411 -0.3148677292  1.4015782607
C   1.7594673232  0.2515204047 -0.3900522507
H   1.3839532617 -0.6819842681 -0.7692934504
no_reorient
no_com
symmetry c1
"""


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",
    "e_convergence": 1e-9,
    "d_convergence": 1e-9,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none"
}
# CQED parameters
# ---------------------------------------------------------
#0.07878123598
#0.0551632153
#0.02739592187
lambda_vector = np.array([0.07878123598, 0.0551632153, 0.02739592187])  # start with lambda = 0
omega = 0.01


# ---------------------------------------------------------
# Compute analytic gradient using wb97x functional 
# ---------------------------------------------------------

calc_a = CQEDRHFCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    functional="wb97x",
    charge = 0,
    multiplicity = 1,
    debug = True,
)

E_a, gradient_a, _ = calc_a.energy_and_gradient(geometry, canonical="psi4")

# ---------------------------------------------------------
# Compute analytic gradient using wb97x functional 
# ---------------------------------------------------------


calc_b = CQEDRHFCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    functional=None,
    charge = 0,
    multiplicity = 1,
    debug = True,
)

E_b, gradient_b, _ = calc_b.energy_and_gradient(geometry, canonical="psi4")

# ---------------------------------------------------------
# Finite difference gradient
# ---------------------------------------------------------

delta = 1e-4

mol = psi4.geometry(geometry)
coords = np.array(mol.geometry())

natom = coords.shape[0]

grad_fd_a = np.zeros_like(coords)
grad_fd_b = np.zeros_like(coords)

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
        print(geom_p)
        print(geom_m)


        Ep = calc_a.energy(geom_p)
        Em = calc_a.energy(geom_m)
        grad_fd_a[atom, xyz] = (Ep - Em) / (2 * delta)
        Ep = calc_b.energy(geom_p)
        Em = calc_b.energy(geom_m)
        grad_fd_b[atom, xyz] = (Ep - Em) / (2 * delta)

# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

diff_a = grad_fd_a - gradient_a
diff_b = grad_fd_b - gradient_b

rms_a = np.sqrt(np.mean(diff_a**2))
max_err_a = np.max(np.abs(diff_a))

rms_b = np.sqrt(np.mean(diff_b**2))
max_err_b = np.max(np.abs(diff_b    ))

print("\n==============================")
print("Finite Difference Validation")
print("==============================")

print("Analytic gradient a:")
print(gradient_a)    

print("\nFinite difference gradient a:")
print(grad_fd_a)    
print("\nDifference a:")
print(diff_a)

print("\nRMS error a:", rms_a)
print("Max error a:", max_err_a)

print("\nAnalytic gradient b:")
print(gradient_b)

print("\nFinite difference gradient b:")
print(grad_fd_b)

print("\nDifference b:")
print(diff_b)

print("\nRMS error b:", rms_b)
print("Max error b:", max_err_b)

tol = 1e-5

if rms_a < tol and rms_b < tol:
    print("\nPASS: analytic gradient validated")
else:
    print("\nWARNING: gradient discrepancy detected")
