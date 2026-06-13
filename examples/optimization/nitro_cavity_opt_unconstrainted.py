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
C   0.5264654535  1.2626157057 -0.0366495512
C   1.9125214242  1.2608986510 -0.0346902530
C   2.5811043670  0.0541075112  0.0491291684
C  -0.1524735176  0.0548466950  0.0301613626
H  -0.0186354870  2.1941720966 -0.0872146527
H   2.4635502655  2.1816778524 -0.0926719680
H  -1.2292624512  0.0545482324  0.0380943757
N   4.0440230582  0.0786783018  0.1089609742
O   4.6536523015 -0.9633602273  0.4128981586
O   4.5899111068  1.1581100643 -0.1672816051
C   0.5410908538 -1.1469405479  0.0769938890
H   0.0090948930 -2.0819492506  0.1145705607
C   1.9273912525 -1.1636781981  0.0959249393
H   2.4888190913 -2.0845174360  0.1587671718
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
    project_tr_rot=False,
    projection_debug=False,
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
