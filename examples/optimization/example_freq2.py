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
    "basis": "sto-3g",
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

ortho_string = """
C           -1.804928163307     1.957993763262     0.703312273806 
C           -0.379708783307     1.994122833262     0.698532703806 
C            0.296125016693     0.817793533262     0.710271493806 
C           -2.520286433307     0.755089873262     0.736288843806 
H           -2.344947113307     2.899196893262     0.691895063806 
H            0.158564066693     2.933869823262     0.699142733806 
H           -3.601954283307     0.764862203262     0.746931053806 
N            1.767881836693     0.820900013262     0.771891313806 
O            2.315054046693    -0.296733496738     0.879853723806 
O            2.340645916693     1.923356243262     0.711986073806 
C           -1.829967733307    -0.442167236738     0.756258983806 
H           -2.356763623307    -1.389967436738     0.789740873806 
C           -0.361341153307    -0.491572936738     0.714148383806 
H            0.119338216693    -1.238105076738     1.350400383806 
Br          -0.151212663307    -1.224162306738    -1.170925976194 
"""

symbols = []
positions = []
for line in ortho_string.strip().split('\n'):
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

opt = BFGS(molecule, trajectory='qed_opt.traj', logfile='qed_opt.log')
opt.run(fmax=0.01)

write('qed_opt_final.xyz', molecule)
print("\nOptimization Complete!\n")

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