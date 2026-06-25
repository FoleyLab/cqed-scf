"""
Geometry optimization of water in an optical cavity using CQED-RHF + BFGS,
with Cartesian projected-gradient removal of translation/rotation.
"""

import numpy as np
import psi4
#psi4.core.be_quiet()

from cqed_scf import CQEDCalculator
from cqed_scf.drivers import bfgs_optimize
from cqed_scf.utils import write_xyz, ANGSTROM_TO_BOHR

# =========================
# Psi4 geometry (angstrom)
# =========================

h2o_string = """
0 1
C                  0.51932475    1.23303451   -0.03194925
C                  1.94454413    1.26916358   -0.03672882
C                  2.62037793    0.09283428   -0.02499003
C                 -0.19603352    0.03013062    0.00102732
H                 -0.02069420    2.17423764   -0.04336646
H                  2.48281698    2.20891057   -0.03611879
H                 -1.27770137    0.03990295    0.01166953
N                  4.09213475    0.09594076    0.03662979
O                  4.63930696   -1.02169275    0.14459220
O                  4.66489883    1.19839699   -0.02327545
C                  0.49428518   -1.16712649    0.02099746
H                 -0.03251071   -2.11492669    0.05447935
C                  1.96291176   -1.21653219   -0.02111314
H                  2.44359113   -1.96306433    0.61513886
units angstrom
no_reorient
no_com
symmetry c1
"""

# =========================
# Psi4 options
# =========================

psi4_options = {
    "basis": "6-31G",
    "scf_type": "pk",
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

lambda_vector = [0.078, 0.055, 0.027]  # polarization along z
omega = 0.06615    

# =========================
# CQED-DFT calculator
# =========================

calc = CQEDCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    charge=0,
    multiplicity=1,
    functional="wb97x",  # try None for RHF
)

# =========================
# Prepare XYZ output
# =========================

xyz_file = "nitro_cavity_opt_projected_df.xyz"

# Clear old trajectory if it exists
open(xyz_file, "w").close()

# Extract symbols once (for XYZ writing)
mol = psi4.geometry(h2o_string)
symbols = [mol.symbol(i) for i in range(mol.natom())]

# =========================
# Run BFGS optimization
# =========================

print("Starting projected-gradient BFGS optimization of H2O in cavity...\n")

opt_result, _ = bfgs_optimize(
    calculator=calc,
    geometry=h2o_string,
    canonical="psi4",
    gtol=1e-5,
    maxiter=50,
    debug=True,
    project_tr_rot=True,
    projection_debug=True,
)

# =========================
# Write final optimized geometry
# =========================

coords_opt_bohr = opt_result.x.reshape(-1, 3)
coords_opt_angstrom = coords_opt_bohr / ANGSTROM_TO_BOHR

write_xyz(
    xyz_file,
    symbols,
    coords_opt_angstrom,
    comment=f"FINAL OPTIMIZED | E = {opt_result.fun:.10f} Ha",
    mode="a",
)

# =========================
# Summary
# =========================

print("\nOptimization finished.")
print(f"Converged: {opt_result.success}")
print(f"Final energy (Ha): {opt_result.fun:.10f}")
print(f"XYZ trajectory written to: {xyz_file}")
