"""
CQED-DFT water regression using the new CQEDConfig architecture.

This example runs CQED-DFT with lambda = 0 using wB97X-D and compares
the energy and gradient directly against Psi4. Since lambda = 0, the
CQED result should reduce to ordinary Psi4 DFT.

The calculator internally strips the dispersion-corrected functional
for the CQED-SCF step when needed, then adds the dispersion correction
post-SCF according to the config/dispersion policy.
"""

import numpy as np
import psi4

from cqed_scf import CQEDConfig, CQEDCalculator


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
}


# ---------------------------------------------------------
# Build CQED configuration
# ---------------------------------------------------------

config = CQEDConfig(
    lambda_vector=np.zeros(3),
    omega=0.1,
    psi4_options=psi4_options,
    reference="rks",
    functional="wb97x-d",
    density_fitting=True,
    charge=0,
    multiplicity=1,
    dispersion_policy="post_scf",
    debug=True,
)


# ---------------------------------------------------------
# Build calculator from config
# ---------------------------------------------------------

calc = CQEDCalculator(config=config)


# ---------------------------------------------------------
# Run CQED-DFT energy + gradient
# ---------------------------------------------------------

E, grad, g = calc.energy_and_gradient(geometry)


# ---------------------------------------------------------
# Psi4 reference calculation
# ---------------------------------------------------------

psi4.core.clean()
psi4.core.clean_options()
psi4.set_options(psi4_options)

mol = psi4.geometry(geometry)

E_ref = psi4.energy("wb97x-d", molecule=mol)
grad_ref = np.asarray(psi4.gradient("wb97x-d", molecule=mol))


# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

energy_diff = E - E_ref
grad_max_diff = np.max(np.abs(grad - grad_ref))

print("\nCQED-DFT lambda = 0 regression")
print("==============================")
print(f"Reference       : {config.reference}")
print(f"Functional      : {config.functional}")
print(f"Base functional : {config.base_scf_functional}")
print(f"Dispersion mode : {config.dispersion_policy}")
print()
print(f"CQED energy     : {E:.12f} Eh")
print(f"Psi4 energy     : {E_ref:.12f} Eh")
print(f"Energy diff     : {energy_diff:.3e} Eh")
print()
print(f"Max grad diff   : {grad_max_diff:.3e} Eh/Bohr")
print(f"Effective g     : {g:.12e}")

assert abs(energy_diff) < 1e-10
assert grad_max_diff < 1e-6

print("\nPASS: CQED-DFT lambda = 0 reproduces Psi4 wB97X-D.")
