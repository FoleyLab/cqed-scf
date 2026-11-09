"""
CQED-UHF single-point energy using CQEDConfig and CQEDCalculator.
"""

import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig



# Define water from psi4numpy tutorial example
mol_str = """
0 2
O
H 1 0.9697
symmetry c1
"""

# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "6-311+G*",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}


# ---------------------------------------------------------
# Build CQED configuration
# ---------------------------------------------------------

config = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.0]),
    omega=0.0,
    psi4_options=psi4_options,
    reference="uhf",
    functional=None,
    density_fitting=False,
    charge=0,
    multiplicity=2,
    dispersion_policy="none",
    debug=False,
    quiet=True,  # SILENT: suppress all stdout (CQED-SCF + Psi4 engine output)
)

print(config)
# ---------------------------------------------------------
# Run CQED-RHF energy
# ---------------------------------------------------------

calc = CQEDCalculator(config=config)
energy = calc.energy(mol_str)

expected_energy = -7.540608221E+01 #-75.9897957793742762
assert np.isclose(energy, expected_energy)
