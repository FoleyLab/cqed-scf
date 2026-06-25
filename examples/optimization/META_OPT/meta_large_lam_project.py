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
C  -0.9376946656  2.0045984403  0.8307475274
C   0.4513119772  1.9306534433  0.6812514961
C   1.1341665883  0.7144353559  0.7111550570
C   0.4874811932 -0.4656967098  0.7927491056
C  -1.6359626038  0.8456634229  0.8700458385
H  -1.4313549188  2.9639031729  0.8808900447
H   1.0304332237  2.8331982445  0.5490051040
H   1.0296335462 -1.4038602089  0.8057466366
H  -2.7132614129  0.8565926242  0.9334943503
N   2.6082399177  0.7111610943  0.5894043365
O   3.1792751227 -0.2463563094  1.0374118795
O   3.0695511634  1.6568638112  0.0103301087
C  -0.9790259495 -0.4644901176  0.7032285915
H  -1.4456776513 -1.2595647120  1.2766155152
Br -1.2389488175 -0.8755026340 -1.2006179623
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
