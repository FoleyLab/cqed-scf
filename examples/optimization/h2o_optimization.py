"""
Geometry optimization of water in an optical cavity
using CQED-RHF + BFGS, with XYZ trajectory output.
"""

import numpy as np
import psi4
#psi4.core.be_quiet()

from cqed_rhf import CQEDRHFCalculator
from cqed_rhf.drivers import bfgs_optimize
from cqed_rhf.utils import write_xyz, ANGSTROM_TO_BOHR

# =========================
# Psi4 geometry (angstrom)
# =========================

h2o_string = """
O  0.000000000000   0.000000000000  -0.068516219320
H  0.000000000000  -0.790689573744   0.543701060715
H  0.000000000000   0.790689573744   0.543701060715
units angstrom
no_reorient
no_com
symmetry c1
"""

# =========================
# Psi4 options
# =========================

psi4_options = {
    "basis": "cc-pVDZ",
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

lambda_vector = [0., 0.3, 0.3]   # polarization along z
omega = 0.0                       # cavity frequency (a.u.)

# =========================
# CQED-DFT calculator
# =========================

calc = CQEDRHFCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    charge=0,
    multiplicity=1,
    functional="wb97x",  # try None for RHF
)
#calc = CQEDRHFCalculator(
#    lambda_vector=lambda_vector,
#    psi4_options=psi4_options,
#    omega=omega,
#    density_fitting=True,
#)

# =========================
# Prepare XYZ output
# =========================

xyz_file = "h2o_cavity_opt_df.xyz"

# Clear old trajectory if it exists
open(xyz_file, "w").close()

# Extract symbols once (for XYZ writing)
mol = psi4.geometry(h2o_string)
symbols = [mol.symbol(i) for i in range(mol.natom())]

# =========================
# Run BFGS optimization
# =========================

print("Starting BFGS optimization of H2O in cavity...\n")

opt_result, _ = bfgs_optimize(
    calculator=calc,
    geometry=h2o_string,
    canonical="psi4",   # use exact gradients for optimization
    gtol=1e-5,
    maxiter=50,
    debug=True,          # <-- enables XYZ writing + detailed output
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

