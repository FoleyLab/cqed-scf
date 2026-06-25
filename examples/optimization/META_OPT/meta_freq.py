import os
import numpy as np
import psi4

from ase import Atoms
from ase.vibrations import Vibrations
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Bohr

from cqed_scf import CQEDCalculator
from cqed_scf.utils import ANGSTROM_TO_BOHR


# ==========================================
# 1. PSI4 OPTIONS & CAVITY PARAMETERS
# ==========================================
psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none",
}

psi4.set_options(psi4_options)

lambda_vector = [0.08054737872487508, 0.048397747841675776, 0.03420201433256689]
omega = 0.0

cqed_calc = CQEDCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    charge=1,
    multiplicity=1,
    functional="wb97x",
)


# ==========================================
# 2. ASE CALCULATOR USING FULL QED GRADIENT
# ==========================================
class CQED_DFT_Gradient(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, cqed_calculator_instance):
        super().__init__()
        self.backend_calc = cqed_calculator_instance

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        pos = atoms.get_positions()
        syms = atoms.get_chemical_symbols()

        mol_str = "\n".join(
            f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}"
            for s, p in zip(syms, pos)
        )
        mol_str += "\n1 1\nunits angstrom\nsymmetry c1\nno_reorient\nno_com"

        energy_hartree, grad_hartree_bohr, _ = self.backend_calc.energy_and_gradient(mol_str)

        self.results["energy"] = energy_hartree * Hartree

        grad_array = np.asarray(grad_hartree_bohr).reshape(-1, 3)

        # ASE wants forces, not gradients.
        # Force = -gradient.
        self.results["forces"] = -grad_array * (Hartree / Bohr)


# ==========================================
# 3. MASS-WEIGHTED RIGID-BODY PROJECTOR
# ==========================================
def rigid_body_projector_mw(coords_bohr, masses_au, svd_tol=1e-10):
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    masses_au = np.asarray(masses_au, dtype=float)

    natom = coords_bohr.shape[0]
    total_mass = np.sum(masses_au)

    com = np.sum(coords_bohr * masses_au[:, None], axis=0) / total_mass
    x = coords_bohr - com
    sqrtm = np.sqrt(masses_au)

    modes = []

    # Mass-weighted translations
    for axis in range(3):
        mode = np.zeros((natom, 3))
        mode[:, axis] = sqrtm
        modes.append(mode.reshape(-1))

    # Mass-weighted rotations
    for axis in range(3):
        axis_vec = np.eye(3)[axis]
        mode = np.cross(axis_vec[None, :], x)
        mode *= sqrtm[:, None]
        modes.append(mode.reshape(-1))

    B = np.column_stack(modes)

    U, s, _ = np.linalg.svd(B, full_matrices=False)
    keep = s > svd_tol
    Q = U[:, keep]

    Pmw = np.eye(3 * natom) - Q @ Q.T

    return Pmw, Q, s


def project_and_diagonalize_hessian(atoms, vib):
    """
    Extract ASE Cartesian Hessian, convert to atomic units, mass-weight,
    project out translations/rotations, and diagonalize.
    """

    natom = len(atoms)

    # ASE Hessian from finite differences of forces.
    # Units are eV / Angstrom^2.
    H_ase = vib.get_vibrations().get_hessian()
    H_ase = np.asarray(H_ase).reshape(3 * natom, 3 * natom)

    # ASE's Hessian from forces corresponds to d^2E/dx^2.
    # Convert eV / Angstrom^2 -> Hartree / bohr^2.
    H_cart_au = H_ase * (Bohr**2 / Hartree)

    H_cart_au = 0.5 * (H_cart_au + H_cart_au.T)

    coords_bohr = atoms.get_positions() * ANGSTROM_TO_BOHR

    # ASE masses are amu. Convert amu -> electron masses.
    masses_au = atoms.get_masses() * 1822.888486209

    Pmw, Q, singular_values = rigid_body_projector_mw(coords_bohr, masses_au)

    sqrtm_vec = np.repeat(np.sqrt(masses_au), 3)
    Minv_sqrt = np.diag(1.0 / sqrtm_vec)

    H_mw = Minv_sqrt @ H_cart_au @ Minv_sqrt

    # This is the physically meaningful post-projection.
    H_locked_mw = Pmw @ H_mw @ Pmw
    H_locked_mw = 0.5 * (H_locked_mw + H_locked_mw.T)

    evals, evecs = np.linalg.eigh(H_locked_mw)

    # Convert angular frequency eigenvalues from a.u. to cm^-1.
    au_time = 2.4188843265857e-17  # seconds
    c_cm_s = 2.99792458e10

    omega_au = np.sign(evals) * np.sqrt(np.abs(evals))
    freq_cm = omega_au / (2.0 * np.pi * c_cm_s * au_time)

    return {
        "H_cart_au": H_cart_au,
        "H_mw": H_mw,
        "H_locked_mw": H_locked_mw,
        "evals": evals,
        "evecs": evecs,
        "freq_cm": freq_cm,
        "projector_rank": Q.shape[1],
        "singular_values": singular_values,
    }


# ==========================================
# 4. INITIALIZE LOCKED/OPTIMIZED GEOMETRY
# ==========================================

# Explicitly list each atom symbol in order so ASE maps them correctly
atom_symbols = ['C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'N', 'O', 'O', 'C', 'H', 'Br']

atoms = Atoms(symbols=atom_symbols, positions=[
    [-0.9376890934,  2.0046028135,  0.8307471810],  # C
    [ 0.4513151727,  1.9306572334,  0.6812504606],  # C
    [ 1.1341609892,  0.7144350472,  0.7111639609],  # C
    [ 0.4874805347, -0.4657001325,  0.7927522454],  # C
    [-1.6359638022,  0.8456692723,  0.8700412625],  # C
    [-1.4313566969,  2.9639016554,  0.8808893574],  # H
    [ 1.0304354877,  2.8332018435,  0.5490050614],  # H
    [ 1.0296379234, -1.4038627571,  0.8057457834],  # H
    [-2.7132672813,  0.8565882206,  0.9334964469],  # H
    [ 2.6082323378,  0.7111592245,  0.5894037493],  # N
    [ 3.1792788435, -0.2463508291,  1.0374037935],  # O
    [ 3.0695502992,  1.6568558543,  0.0103306394],  # O
    [-0.9790260749, -0.4644886153,  0.7032360728],  # C
    [-1.4456799215, -1.2595684166,  1.2766182300],  # H
    [-1.2389420038, -0.8755014914, -1.2006266108]   # Br
])

atoms.calc = CQED_DFT_Gradient(cqed_calc)


# ==========================================
# 5. RUN ASE FINITE DIFFERENCES
# ==========================================
print("Starting numerical frequency calculation using full QED-DFT gradients.")
print("Important: gradients are not projected during finite differences.")

vib = Vibrations(atoms, name="meta_freq")
vib.run()


# ==========================================
# 6. STANDARD ASE SUMMARY, UNPROJECTED
# ==========================================
print(f"\n{'-' * 50}")
print("ASE FREQUENCY SUMMARY FROM RAW CARTESIAN HESSIAN")
print(f"{'-' * 50}")
vib.summary()


# ==========================================
# 7. PROJECTED LOCKED-MOLECULE ANALYSIS
# ==========================================
results = project_and_diagonalize_hessian(atoms, vib)

freq_cm = results["freq_cm"]

# remove near-zero translation/rotation modes
vib_freq_cm = freq_cm[np.abs(freq_cm) > 1.0]

# optionally keep only positive modes
vib_freq_cm = vib_freq_cm[vib_freq_cm > 0.0]

cm_to_ev = 1.239841984e-4
zpe_ev = 0.5 * np.sum(vib_freq_cm) * cm_to_ev
zpe_hartree = zpe_ev / Hartree

print("Projected ZPE / eV     =", zpe_ev)
print("Projected ZPE / Hartree =", zpe_hartree)


print(f"\n{'-' * 50}")
print("LOCKED-MOLECULE PROJECTED FREQUENCY SUMMARY")
print(f"{'-' * 50}")
print(f"Rigid-body projector rank: {results['projector_rank']}")
print("Rigid-body singular values:")
print(results["singular_values"])

print("\nAll projected mass-weighted Hessian frequencies / cm^-1:")
for i, f in enumerate(freq_cm):
    print(f"{i:4d}  {f:16.8f}")

print("\nNonzero vibrational frequencies / cm^-1:")
nonzero = np.abs(freq_cm) > 1.0
for i, f in enumerate(freq_cm[nonzero]):
    print(f"{i:4d}  {f:16.8f}")


# ==========================================
# 8. SAVE ARRAYS FOR INSPECTION
# ==========================================
np.save("meta_H_cart_au.npy", results["H_cart_au"])
np.save("meta_H_mw.npy", results["H_mw"])
np.save("meta_H_locked_mw.npy", results["H_locked_mw"])
np.save("meta_projected_freq_cm.npy", results["freq_cm"])

vib.write_mode(-1)

print("\nSaved:")
print("  meta_H_cart_au.npy")
print("  meta_H_mw.npy")
print("  meta_H_locked_mw.npy")
print("  meta_projected_freq_cm.npy")
print("  ASE mode trajectory files")
