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
O      -0.702196054     -0.056060256      0.009942262
H      -1.022193224      0.846775782     -0.011488714
H       0.257521062      0.042121496      0.005218999
--
0 1
O       2.799169805      0.022179384     -0.000734005
H       3.175791420     -0.416200682      0.765390377
H       3.171434122     -0.454033591     -0.746136507
no_reorient
no_com
units angstrom
symmetry c1
"""


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "jun-cc-pVDZ",
    "scf_type": "df",
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
    reference="rhf",
    functional=None,
    density_fitting=True,
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
    integral_backend="df",
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
