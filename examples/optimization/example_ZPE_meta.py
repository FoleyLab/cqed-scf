import os
import numpy as np
from cqed_scf import CQEDCalculator
from cqed_scf.drivers import bfgs_optimize,project_cartesian_gradient_remove_translation_rotation
from cqed_scf.utils import write_xyz, ANGSTROM_TO_BOHR, generate_field_vector_from_theta_and_phi
from ase import Atoms
from ase.io import write, read
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Bohr

# Make sure to import your custom functions
# from your_module import CQEDCalculator, project_cartesian_gradient_remove_translation_rotation

# ==========================================
# 1. USER PARAMETERS & OPTIONS
# ==========================================

psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none"
}

omega = 0.06615
lam_mag = 0.1
lambda_direction = np.array([0.80547379, 0.48397748, 0.34202014]) # Replace with your generated vector
lambda_direction /= np.linalg.norm(lambda_direction)
lambda_vector = (lam_mag * lambda_direction).tolist()

meta_string = """
C           -0.929257263947     2.021527608578     0.744707683350
C            0.476075706053     1.968481358578     0.682883583350 
C            1.153033166053     0.732862858578     0.671089073350 
C            0.486309286053    -0.455398891422     0.707696283350 
C           -1.646688783947     0.850023888578     0.786483593350 
H           -1.430027043947     2.980198348578     0.754644003350 
H            1.068570756053     2.878318968578     0.644324213350 
H            1.030908186053    -1.394630481422     0.699715393350 
H           -2.730391873947     0.862207158578     0.834726773350 
N            2.627601876053     0.732774608578     0.609077593350 
O            3.188360516053    -0.377859281422     0.588451963350 
O            3.186221516053     1.845711198578     0.586422223350 
C           -0.982368843947    -0.464026221422     0.760065283350 
H           -1.395507033947    -1.190671951422     1.465426213350 
Br          -1.494673453947    -1.187920261422    -1.064256256650
"""

symbols = []
positions = []
for line in meta_string.strip().split('\n'):
    parts = line.split()
    if len(parts) == 4:
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

molecule = Atoms(symbols=symbols, positions=positions)

# ==========================================
# 2. MODIFIED ASE CALCULATOR
# ==========================================

class CQED_ASE_Calculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, charge=1, multiplicity=1, project_forces=True, debug=False, **cqed_kwargs):
        super().__init__()
        self.charge = charge
        self.multiplicity = multiplicity
        
        # New attributes to control projection and debugging
        self.project_forces = project_forces 
        self.debug = debug
        self.cqed_kwargs = cqed_kwargs
        
        self.calc = CQEDCalculator(
            charge=self.charge,
            multiplicity=self.multiplicity,
            **self.cqed_kwargs
        )

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        syms = atoms.get_chemical_symbols()
        pos_angstrom = atoms.get_positions()
        masses = atoms.get_masses()
        
        geom_str = f"{self.charge} {self.multiplicity}\n"
        geom_lines = [f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}" for s, p in zip(syms, pos_angstrom)]
        geom_str += "\n".join(geom_lines)
        geom_str += "\nunits angstrom\nno_reorient\nno_com\nsymmetry c1"

        # 1. Get raw energy and gradient
        E_hartree, grad_au, _ = self.calc.energy_and_gradient(geom_str, canonical="psi4")
        grad_raw = np.array(grad_au).reshape(-1, 3)

        # 2. Apply projection if the flag is True
        if self.project_forces:
            # Convert coordinates from Angstrom to Bohr for your projection function
            # ASE's `Bohr` constant is ~0.529 (value of 1 Bohr in Angstroms)
            coords_bohr = pos_angstrom / Bohr 
            
            grad_proj, proj_info = project_cartesian_gradient_remove_translation_rotation(
                coords_bohr,
                grad_raw,
                masses,
                return_diagnostics=True,
            )
            final_grad = grad_proj.reshape(-1, 3)
            
            if self.debug:
                print("\n--- Gradient Projection Diagnostics ---")
                print(f"raw |grad|       = {proj_info['raw_grad_norm']:.6e}")
                print(f"projected |grad| = {proj_info['proj_grad_norm']:.6e}")
                print(f"raw |net force|  = {np.linalg.norm(proj_info['net_force_raw']):.6e}")
                print(f"proj |net force| = {np.linalg.norm(proj_info['net_force_proj']):.6e}")
                print(f"raw |torque|     = {np.linalg.norm(proj_info['torque_raw']):.6e}")
                print(f"proj |torque|    = {np.linalg.norm(proj_info['torque_proj']):.6e}")
                print("---------------------------------------\n")
                
        else:
            # If projection is off (e.g., during frequencies), use raw gradient
            final_grad = grad_raw

        # 3. Convert to ASE units (eV and eV/Angstrom)
        self.results['energy'] = E_hartree * Hartree
        self.results['forces'] = -final_grad * (Hartree / Bohr)

# ==========================================
# 3. RUN OPTIMIZATION (Projected Forces)
# ==========================================

print("Phase 1: Starting ASE Geometry Optimization with PROJECTED forces...")

# Attach calculator with projection turned ON
molecule.calc = CQED_ASE_Calculator(
    charge=1,
    multiplicity=1,
    project_forces=True,  # <-- IMPORTANT: Projection is ON
    debug=True,
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    functional="wb97x"
)

print(f"Energy before optimization: {molecule.get_potential_energy() / Hartree:.10f} Ha")

opt = BFGS(molecule, trajectory='qed_opt.traj', logfile='qed_opt.log')
opt.run(fmax=0.01)

write('qed_opt_final.xyz', molecule)
print("\nOptimization Complete!\n")
print(f"Energy after optimization: {molecule.get_potential_energy() / Hartree:.10f} Ha")

# ==========================================
# 4. RUN FREQUENCY ANALYSIS (Raw Forces)
# ==========================================

print("Phase 2: Starting Numerical Frequency Calculation with RAW forces...")

# Flip the switch: Turn projection OFF for the finite difference displacements
molecule.calc.project_forces = False  # <-- IMPORTANT: Projection is now OFF

vib = Vibrations(molecule, name='qed_freq')
vib.run()

print(f"\n{'-'*50}")
print("FREQUENCY SUMMARY")
print(f"{'-'*50}")
vib.summary()
vib.write_mode(-1)

# Save and print the Zero-Point Energy (ZPE)
# ASE returns ZPE in eV from `get_zero_point_energy()`
zpe_eV = vib.get_zero_point_energy()
zpe_Ha = zpe_eV / Hartree
print(f"\nZero-point energy (ZPE): {zpe_eV:.8f} eV ({zpe_Ha:.10f} Ha)")
np.savez_compressed('qed_freq_zpe.npz', zpe_eV=zpe_eV, zpe_Ha=zpe_Ha)