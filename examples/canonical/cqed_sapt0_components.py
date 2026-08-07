"""
QED-SAPT0 total energy and components using CQEDConfig and CQEDCalculator.
"""

import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig


# ---------------------------------------------------------
# Dimer geometry
# ---------------------------------------------------------

dimer = """
0 1
He 0.0000000000 0.0000000000 0.0000000000
--
0 1
He 0.0000000000 0.0000000000 2.0000000000
units angstrom
symmetry c1
no_reorient
no_com
"""


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "6-31g",
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
    debug=True,   # VERBOSE: add SCF/component detail (debug route via output.echo)
    quiet=False,  # keep normal verbosity so the verbose detail is emitted
)


# ---------------------------------------------------------
# Run QED-SAPT0 components
# ---------------------------------------------------------

calc = CQEDCalculator(config=config)
dimer_geometry = psi4.geometry(dimer)

components = calc.sapt0_components(
    dimer_geometry,
    integral_backend="full_eri",
    include_cavity_terms=True,
)

print("\nQED-SAPT0 components")
print("====================")
print(f"Electrostatics       : {components.elst10: .12f} Eh")
print(f"Exchange             : {components.exch10: .12f} Eh")
print(f"Dispersion           : {components.disp20: .12f} Eh")
print(f"Exchange-dispersion  : {components.exch_disp20: .12f} Eh")
print(f"Induction            : {components.ind20: .12f} Eh")
print(f"Exchange-induction   : {components.exch_ind20: .12f} Eh")
print("-" * 48)
print(f"Total QED-SAPT0      : {components.total: .12f} Eh")
