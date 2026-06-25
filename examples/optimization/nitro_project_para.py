
import numpy as np
import psi4
#psi4.core.be_quiet()

from cqed_scf import CQEDCalculator
from cqed_scf.drivers import bfgs_optimize
from cqed_scf.utils import write_xyz, ANGSTROM_TO_BOHR, generate_field_vector_from_theta_and_phi

# =========================
# Psi4 geometry (angstrom)
# =========================

para_string = """
1 1
C         -0.511618296797     1.244386024531     0.732140048697
C          0.856500593203     1.251903714531     0.717948218697
C          1.524118723203     0.024661924531     0.713927788697
H         -1.071804396797     2.172682314531     0.745925708697
H          1.436128963203     2.163921874531     0.712099008697
N          3.008539583203     0.046097104531     0.698823798697
O          3.575097303203    -1.082768165469     0.699174708697
O          3.542114363203     1.190870854531     0.689202018697
C         -0.475464946797    -1.253402765469     0.742118638697
H         -1.008574426797    -2.197377955469     0.762945788697
C          0.892227703203    -1.221407805469     0.728065818697
H          1.498048423203    -2.116244695469     0.729653208697
C         -1.267906576797    -0.015841805469     0.712127418697
H         -2.116520796797    -0.025293215469     1.403161498697
Br        -2.114966986797    -0.034666925469    -1.121874081303
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

lambda_direction = np.asarray(field_vector_center, dtype=float)
lambda_direction /= np.linalg.norm(lambda_direction)
lambda_magnitudes = np.linspace(0.02, 0.10, 5)
omega = 0.06615

# Extract symbols once (for XYZ writing)
mol = psi4.geometry(para_string)
symbols = [mol.symbol(i) for i in range(mol.natom())]

print("Starting projected gradient BFGS optimization sweep for fixed direction lambda vectors...\n")

results = []
for lam_mag in lambda_magnitudes:
    lambda_vector = (lam_mag * lambda_direction).tolist()
    xyz_file = f"nitro_cavity_opt_projected_df_para_65_78_lam_{lam_mag:.2f}.xyz"
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
        geometry=para_string,
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
