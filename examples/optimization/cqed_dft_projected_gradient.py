"""
CQED-DFT energy and Cartesian projected gradient.
"""

import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

geometry = """
0 1
O
H 1 0.9572
H 1 0.9572 2 104.5
symmetry c1
"""


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "6-311g",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none",
}


# ---------------------------------------------------------
# Build CQED configuration
# ---------------------------------------------------------

config = CQEDConfig(
    lambda_vector=np.array([0.0, 0.05, 0.05]),
    omega=0.1,
    psi4_options=psi4_options,
    reference="rks",
    functional="wb97x",
    density_fitting=True,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=False,
    quiet=False,  # NORMAL: gradient-path prints aren't routed via output yet (Stage B)
)


# ---------------------------------------------------------
# Run CQED-DFT energy + projected gradient
# ---------------------------------------------------------

calc = CQEDCalculator(config=config)
energy, projected_gradient, coupling = calc.energy_and_projected_gradient(geometry)

print("\nCQED-DFT projected gradient")
print("===========================")
print(f"Reference   : {config.reference}")
print(f"Functional  : {config.functional}")
print(f"Energy      : {energy:.12f} Eh")
print(f"Effective g : {coupling:.12e}")
print("Projected gradient / Eh Bohr^-1:")
print(projected_gradient)
