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
C  -0.4938685756  1.2145704097  0.8312462388
C   0.8602944754  1.2394211824  0.8122824256
C   1.5285918783  0.0168222333  0.7209992534
H  -1.0652938171  2.1272845525  0.9117808171
H   1.4238850587  2.1542127271  0.8664459944
N   2.9947097608  0.0476786098  0.6985003515
O   3.5337089623 -0.5079436707  1.6152933492
O   3.4719748311  0.6200117008 -0.2422204847
C  -0.4462688230 -1.2580969452  0.6165455508
H  -0.9820709513 -2.1903923003  0.5457602961
C   0.9039364276 -1.2393499089  0.6465699283
H   1.5069990569 -2.1323893623  0.6149241845
C  -1.2499060989 -0.0305920798  0.6696779219
H  -2.0545828851 -0.1198692639  1.4033344274
BR -2.1661900611  0.2061525939 -1.0457006502
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
