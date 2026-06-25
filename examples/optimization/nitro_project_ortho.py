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

ortho_string = """
1 1
C           -1.804928163307     1.957993763262     0.703312273806 
C           -0.379708783307     1.994122833262     0.698532703806 
C            0.296125016693     0.817793533262     0.710271493806 
C           -2.520286433307     0.755089873262     0.736288843806 
H           -2.344947113307     2.899196893262     0.691895063806 
H            0.158564066693     2.933869823262     0.699142733806 
H           -3.601954283307     0.764862203262     0.746931053806 
N            1.767881836693     0.820900013262     0.771891313806 
O            2.315054046693    -0.296733496738     0.879853723806 
O            2.340645916693     1.923356243262     0.711986073806 
C           -1.829967733307    -0.442167236738     0.756258983806 
H           -2.356763623307    -1.389967436738     0.789740873806 
C           -0.361341153307    -0.491572936738     0.714148383806 
H            0.119338216693    -1.238105076738     1.350400383806 
Br          -0.151212663307    -1.224162306738    -1.170925976194 
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
mol = psi4.geometry(ortho_string)
symbols = [mol.symbol(i) for i in range(mol.natom())]

print("Starting projected gradient BFGS optimization sweep for fixed direction lambda vectors...\n")

results = []
for lam_mag in lambda_magnitudes:
    lambda_vector = (lam_mag * lambda_direction).tolist()
    xyz_file = f"nitro_cavity_opt_projected_df_lam_{lam_mag:.2f}.xyz"
    traj_file_para = f"nitro_cavity_opt_projected_df_lam_{lam_mag:.2f}.npz"

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

    opt_result, _ = bfgs_optimize(
        calculator=calc,
        geometry=ortho_string,
        canonical="psi4",
        gtol=1e-5,
        maxiter=50,
        debug=True,
        project_tr_rot=True,
        projection_debug=True,
        trajectory_file=traj_file_para,
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

    results.append((lam_mag, opt_result.success, opt_result.fun, xyz_file, traj_file_para))

print("\nOptimization sweep finished.")
for lam_mag, success, energy, xyz_file, traj_file_para in results:
    print(
        f"|lambda| = {lam_mag:.2f} | Converged: {success} | "
        f"Final energy (Ha): {energy:.10f} | XYZ: {xyz_file} | Trajectory: {traj_file_para}"
    )
