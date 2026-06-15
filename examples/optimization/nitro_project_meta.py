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
C           -0.929257263947     2.021527608578     0.744707683350
C            0.476075706053     1.968481358578     0.682883583350 
C            1.153033166053     0.732862858578     0.671089073350 
C            0.486309286053    -0.455398891422     0.707696283350 
C           -1.646688783947     0.850023888578     0.786483593350 
H           -1.430027043947     2.980198348578     0.754644003350 
H            1.068570756053     2.878318968578     0.644324213350 
H            1.030908186053    -1.394630481422     0.699715393350 
H           -2.730391873947     0.862207158578     0.834726773350 
N            2.627601876053     0.732774608578     0.609077593350 
O            3.188360516053    -0.377859281422     0.588451963350 
O            3.186221516053     1.845711198578     0.586422223350 
C           -0.982368843947    -0.464026221422     0.760065283350 
H           -1.395507033947    -1.190671951422     1.465426213350 
BR          -1.494673453947    -1.187920261422    -1.064256256650 
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

lambda_direction = np.asarray(field_vector_center, dtype=float)
lambda_direction /= np.linalg.norm(lambda_direction)
lambda_magnitudes = np.linspace(0.02, 0.10, 5)
omega = 0.06615

# Extract symbols once (for XYZ writing)
mol = psi4.geometry(meta_string)
symbols = [mol.symbol(i) for i in range(mol.natom())]

print("Starting projected gradient BFGS optimization sweep for fixed direction lambda vectors...\n")

results = []
for lam_mag in lambda_magnitudes:
    lambda_vector = (lam_mag * lambda_direction).tolist()
    xyz_file = f"nitro_cavity_opt_projected_df_lam_{lam_mag:.2f}.xyz"

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

    opt_result, _ = bfgs_optimize(trajectory_file='nitro_opt_ortho.npz',
        calculator=calc,
        geometry=meta_string,
        canonical="psi4",
        gtol=1e-5,
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

    results.append((lam_mag, opt_result.success, opt_result.fun, xyz_file))

print("\nOptimization sweep finished.")
for lam_mag, success, energy, xyz_file in results:
    print(
        f"|lambda| = {lam_mag:.2f} | Converged: {success} | "
        f"Final energy (Ha): {energy:.10f} | XYZ: {xyz_file}"
    )
