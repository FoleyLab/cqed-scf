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

width = 58
print()
print("QED-SAPT0 components: He ... He, 2.0 Angstrom")
print(f"basis={psi4_options['basis']}, scf_type={psi4_options['scf_type']}, "
      f"backend=full_eri, frame=monomer_com")
print(f"lambda = {lambda_vector.tolist()}, omega = {omega} Eh")
print("=" * 58)
print(f"{'Component':<24}{'Energy (Eh)':>18}{'(kcal/mol)':>16}")
print("-" * 58)
print(f"{'Electrostatics':<24}{components.elst10:18.12f}{components.elst10 * EH_TO_KCAL:16.6f}")
print(f"{'Exchange':<24}{components.exch10:18.12f}{components.exch10 * EH_TO_KCAL:16.6f}")
print(f"{'Induction':<24}{components.ind20:18.12f}{components.ind20 * EH_TO_KCAL:16.6f}")
print(f"{'Exchange-induction':<24}{components.exch_ind20:18.12f}{components.exch_ind20 * EH_TO_KCAL:16.6f}")
print(f"{'Dispersion':<24}{components.disp20:18.12f}{components.disp20 * EH_TO_KCAL:16.6f}")
print(f"{'Exchange-dispersion':<24}{components.exch_disp20:18.12f}{components.exch_disp20 * EH_TO_KCAL:16.6f}")
print("-" * 58)
print(f"{'Total QED-SAPT0':<24}{components.total:18.12f}{components.total * EH_TO_KCAL:16.6f}")
print("=" * 58)
print(f"\nWall time: {elapsed:.2f} s (includes the CQED-SCF solves)")
