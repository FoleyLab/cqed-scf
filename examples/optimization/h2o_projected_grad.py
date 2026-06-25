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
O  -0.0000000000  0.0010939293 -0.0581699841
H   0.0000000000 -0.7540033053  0.5389372499
H   0.0000000000  0.7529093760  0.5381186372
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

lambda_vector = [0., 0.05, 0.05]   # polarization along z
omega = 0.0                       # cavity frequency (a.u.)

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
# Compute Gradient
# =========================

Energy, Projected_Gradient, _ = calc.energy_and_projected_gradient(h2o_string)

print(F"Energy is {Energy:12.12f}")
print("Projected Gradient")
print(Projected_Gradient)
