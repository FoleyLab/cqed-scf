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
C  -0.9310246030  1.9871639795  0.8481743899
C   0.4499108358  1.9158476851  0.6672377386
C   1.1420484721  0.7016737360  0.7109174760
C   0.4926318839 -0.4732796314  0.7950434271
C  -1.6389987123  0.8295503248  0.9020765748
H  -1.4288751153  2.9419267395  0.9050650253
H   1.0255608595  2.8164551012  0.5104825528
H   1.0341834234 -1.4100933803  0.8157972484
H  -2.7188030883  0.8400783934  0.9798095944
N   2.6146749090  0.7136264304  0.5995293285
O   3.2106846421 -0.1022797266  1.2498432272
O   3.0588971189  1.5287007288 -0.1650518559
C  -0.9779421486 -0.4714446737  0.6860156122
H  -1.4525753110 -1.2945919014  1.2093323505
Br -1.2722064520 -0.7217348828 -1.2428150560
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

theta_central = 65 # 70° from z-axis
phi_central = 78 # 31° from x-axis in xy-plane
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
