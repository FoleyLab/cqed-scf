import os
import sys
import numpy as np
import psi4

from ase import Atoms
from ase.io import read, write
from ase.vibrations import Vibrations
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Bohr

# Import your second package modules
from cqed_scf import CQEDCalculator
from cqed_scf.utils import write_xyz, ANGSTROM_TO_BOHR

# ==========================================
# 1. PSI4 OPTIONS & CAVITY PARAMETERS
# ==========================================
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

lambda_vector = [0., 0.05, 0.05] 
omega = 0.0                      

# Instantiate your global CQED-DFT calculator instance
cqed_calc = CQEDCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    charge=0,
    multiplicity=1,
    functional="wb97x",
)

# ==========================================
# 2. CUSTOM ASE CALCULATOR WRAPPER
# ==========================================
class CQED_DFT_Gradient(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, cqed_calculator_instance):
        super().__init__()
        self.backend_calc = cqed_calculator_instance

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # 1. Rebuild the geometry string from ASE's current displacement positions
        pos = atoms.get_positions()
        syms = atoms.get_chemical_symbols()
        
        mol_str = "\n".join([f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for s, p in zip(syms, pos)])
        mol_str += "\nunits angstrom\nsymmetry c1\nno_reorient\nno_com"

        # 2. Call your package function
        energy_hartree, grad_hartree_bohr, _ = self.backend_calc.energy_and_gradient(mol_str)

        # 3. Convert units back to ASE standard (eV and eV/Angstrom)
        # Energy: Hartree -> eV
        self.results['energy'] = energy_hartree * Hartree

        # Forces: Force = -Gradient, Hartree/Bohr -> eV/Angstrom
        grad_array = np.array(grad_hartree_bohr).reshape(-1, 3)
        forces_ev_ang = -grad_array * (Hartree / Bohr)
        self.results['forces'] = forces_ev_ang

# ==========================================
# 3. INITIALIZE GEOMETRY (Pre-Optimized H2O)
# ==========================================
# Instantiated precisely matching your starting h2o_string coordinates
atoms = Atoms('H2O', positions=[
    [-0.0000000000,  0.0010939293, -0.0581699841], # O
    [ 0.0000000000, -0.7540033053,  0.5389372499], # H
    [ 0.0000000000,  0.7529093760,  0.5381186372]  # H
])

# Attach the new custom wrapper
atoms.calc = CQED_DFT_Gradient(cqed_calc)

# ==========================================
# 4. RUN FREQUENCY ANALYSIS
# ==========================================
print("Starting Numerical Frequency Calculation (CQED-DFT)...")
print("This will require 18 separate gradient evaluations (3 atoms x 3 axes x 2 directions).")

# Vibrations will create an 'h2o_freq' folder to house step data
vib = Vibrations(atoms, name='h2o_freq')
vib.run()

# ==========================================
# 5. PRINT RESULTS
# ==========================================
print(f"\n{'-'*50}")
print("FREQUENCY SUMMARY")
print(f"{'-'*50}")

# Prints the harmonic frequencies in cm^-1
vib.summary()

# Writes all modes to trajectory files (e.g., h2o_freq.0.traj) for visualization
vib.write_mode(-1) 
print("\nVibrational modes saved to trajectory files.")
