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
lambda_direction = np.array([0.18843198, 0.88650279, 0.42261826]) # Replace with your generated vector
lambda_direction /= np.linalg.norm(lambda_direction)
lambda_vector = (lam_mag * lambda_direction).tolist()

meta_string = """
C         -0.511618296797     1.244386024531     0.732140048697
C          0.856500593203     1.251903714531     0.717948218697
C          1.524118723203     0.024661924531     0.713927788697
H         -1.071804396797     2.172682314531     0.745925708697
H          1.436128963203     2.163921874531     0.712099008697
N          3.008539583203     0.046097104531     0.698823798697
O          3.575097303203    -1.082768165469     0.699174708697
O          3.542114363203     1.190870854531     0.689202018697
C         -0.475464946797    -1.253402765469     0.742118638697
H         -1.008574426797    -2.197377955469     0.762945788697
C          0.892227703203    -1.221407805469     0.728065818697
H          1.498048423203    -2.116244695469     0.729653208697
C         -1.267906576797    -0.015841805469     0.712127418697
H         -2.116520796797    -0.025293215469     1.403161498697
Br        -2.114966986797    -0.034666925469    -1.121874081303
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