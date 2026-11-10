"""
QED-SAPT0 for a helium dimer inside an optical cavity (cqed-scf only).

Psi4 has no native cavity SAPT0, so this example shows cqed-scf on its own.
The system is tiny (two He atoms, 6-31G), so the dense ``full_eri`` backend
and a conventional ``pk`` SCF are the simplest and most accurate choice; no
density fitting is needed.

``monomer_reference_frame="monomer_com"`` solves each monomer's CQED-SCF
reference with that monomer's own centre of mass at the origin.  CQED orbital
energies are not translation invariant, so this keeps the cavity-mediated
dispersion independent of where the dimer happens to sit in space.
"""

import time

import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig

EH_TO_KCAL = 627.5094740631

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
    "scf_type": "pk",  # conventional SCF: exact integrals, cheap for He2
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

# ---------------------------------------------------------
# Build CQED configuration
# ---------------------------------------------------------

lambda_vector = np.array([0.0, 0.0, 0.05])
omega = 0.1

config = CQEDConfig(
    lambda_vector=lambda_vector,
    omega=omega,
    psi4_options=psi4_options,
    reference="rhf",
    functional=None,
    density_fitting=False,  # exact (non-fitted) SAPT0 components
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=False,
    quiet=True,
)

# ---------------------------------------------------------
# Run QED-SAPT0 components
# ---------------------------------------------------------

calc = CQEDCalculator(config=config)
dimer_geometry = psi4.geometry(dimer)

t0 = time.perf_counter()
components = calc.sapt0_components(
    dimer_geometry,
    integral_backend="full_eri",
    include_cavity_terms=True,
    monomer_reference_frame="monomer_com",
)
elapsed = time.perf_counter() - t0

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

ROWS = [
    ("Electrostatics", components.elst10),
    ("Exchange", components.exch10),
    ("Induction", components.ind20),
    ("Exchange-induction", components.exch_ind20),
    ("Dispersion", components.disp20),
    ("Exchange-dispersion", components.exch_disp20),
]

width = 58
print()
print("QED-SAPT0 components: He ... He, 2.0 Angstrom")
print(f"basis={psi4_options['basis']}, scf_type={psi4_options['scf_type']}, "
      f"backend=full_eri, frame=monomer_com")
print(f"lambda = {lambda_vector.tolist()}, omega = {omega} Eh")
print("=" * width)
print(f"{'Component':<24}{'Energy (Eh)':>18}{'(kcal/mol)':>16}")
print("-" * width)
for label, value in ROWS:
    print(f"{label:<24}{value:18.12f}{value * EH_TO_KCAL:16.6f}")
print("-" * width)
print(f"{'Total QED-SAPT0':<24}{components.total:18.12f}{components.total * EH_TO_KCAL:16.6f}")
print("=" * width)
print(f"\nWall time: {elapsed:.2f} s (includes the CQED-SCF solves)")
