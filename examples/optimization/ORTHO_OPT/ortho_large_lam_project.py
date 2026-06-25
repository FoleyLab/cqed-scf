"""
Geometry optimization of water in an optical cavity using CQED-RHF + BFGS,
with Cartesian projected-gradient removal of translation/rotation.
"""

import numpy as np
import psi4
#psi4.core.be_quiet()

from cqed_scf import CQEDCalculator
from cqed_scf.drivers import bfgs_optimize
from cqed_scf.utils import write_xyz, ANGSTROM_TO_BOHR, generate_field_vector_from_theta_and_phi
# =========================
# Psi4 geometry (angstrom)
# =========================

meta_string = """
1 1
C  -1.7945865812  1.9431226633  0.7747202691
C  -0.3807030738  1.9909804434  0.7393407453
C   0.2754776320  0.8214661994  0.7166873524
C  -2.5087518690  0.7425863375  0.7852236838
H  -2.3394937770  2.8799272043  0.8074200799
H   0.1396872251  2.9331742460  0.7404716377
H  -3.5828451907  0.7656074819  0.8410960100
N   1.7454037382  0.7890854013  0.7748560225
O   2.2059739123 -0.0668023873  1.4873396357
O   2.3137609813  1.5993722512  0.0996512993
C  -1.8294097410 -0.4319165263  0.7004837424
H  -2.3507875651 -1.3761782233  0.6800847785
C  -0.3709727732 -0.4842459133  0.6040103725
H   0.0728034849 -1.2338707145  1.2593968739
Br  0.0509427435 -1.0878317702 -1.2210545712
units angstrom
no_reorient
no_com
symmetry c1
"""

# =========================
# Psi4 options
# =========================

psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none"
}

psi4.set_options(psi4_options)

# =========================
# Cavity parameters
# =========================

theta_central = 70 # 70° from z-axis
phi_central = 31 # 31° from x-axis in xy-plane
d_alpha = 1.0 # deviation angle in degrees

# pre-compute different field vectors for finite differences of QED-RHF energy wrt theta and phi
field_vector_center = generate_field_vector_from_theta_and_phi(theta_central, phi_central)
vec_mag = np.linalg.norm(field_vector_center)
print(F"Field Magnitude before scaling is {vec_mag}")
print(F"Field Vector is")
print(field_vector_center)

lambda_direction = np.asarray(field_vector_center, dtype=float)
lambda_direction /= np.linalg.norm(lambda_direction)
lam_mag = 0.1
omega = 0.06615


# Extract symbols once (for XYZ writing)
mol = psi4.geometry(meta_string)
symbols = [mol.symbol(i) for i in range(mol.natom())]


lambda_vector = (lam_mag * lambda_direction).tolist()
xyz_file = f"nitro_cavity_opt_projected_df_lam_{lam_mag:.2f}.xyz"

mag = np.linalg.norm(lambda_vector)
print(F"Mag after scaling {mag}")
# Clear old trajectory if it exists for this lambda magnitude.
open(xyz_file, "w").close()

print(f"Running optimization for |lambda| = {lam_mag:.2f} with vector {lambda_vector}")

calc = CQEDCalculator(
        lambda_vector=lambda_vector,
        psi4_options=psi4_options,
        omega=omega,
        density_fitting=True,
        charge=1,
        multiplicity=1,
        functional="wb97x",  # try None for RHF
)

opt_result, _ = bfgs_optimize( #trajectory_file='nitro_opt_ortho.npz',
        calculator=calc,
        geometry=meta_string,
        canonical="psi4",
        gtol=1e-6,
        maxiter=50,
        debug=True,
        project_tr_rot=True,
        projection_debug=True,
)

coords_opt_bohr = opt_result.x.reshape(-1, 3)
coords_opt_angstrom = coords_opt_bohr / ANGSTROM_TO_BOHR

write_xyz(
        xyz_file,
        symbols,
        coords_opt_angstrom,
        comment=f"FINAL OPTIMIZED | lambda = {lam_mag:.2f} | E = {opt_result.fun:.10f} Ha",
        mode="a",
)

#results.append((lam_mag, opt_result.success, opt_result.fun, xyz_file))




# =========================
# Summary
# =========================

print("\nOptimization finished.")
print(f"Converged: {opt_result.success}")
print(f"Final energy (Ha): {opt_result.fun:.10f}")
print(f"XYZ trajectory written to: {xyz_file}")
