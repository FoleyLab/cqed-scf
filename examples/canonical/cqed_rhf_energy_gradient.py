"""
CQED-RHF energy and nuclear gradient using CQEDConfig and CQEDCalculator.
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
    "basis": "cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}


# ---------------------------------------------------------
# Build CQED configuration
# ---------------------------------------------------------

config = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.05]),
    omega=0.1,
    psi4_options=psi4_options,
    reference="rhf",
    functional=None,
    density_fitting=False,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=False,
)


# ---------------------------------------------------------
# Run CQED-RHF energy + gradient
# ---------------------------------------------------------

calc = CQEDCalculator(config=config)
energy, gradient, coupling = calc.energy_and_gradient(geometry)

print("\nCQED-RHF energy and gradient")
print("============================")
print(f"Energy      : {energy:.12f} Eh")
print(f"Effective g : {coupling:.12e}")
print("Gradient / Eh Bohr^-1:")
print(gradient)
