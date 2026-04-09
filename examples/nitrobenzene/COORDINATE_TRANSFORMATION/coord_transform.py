import numpy as np
import re

def parse_psi4_snippet(filename, is_shifted=False):
    """Parses coordinates, masses, dipoles, and optionally rot-consts from file."""
    coords = []
    masses = []
    symbols = []
    dipole_total_au = np.zeros(3)
    rot_consts = None

    with open(filename, 'r') as f:
        lines = f.readlines()

    # 1. Parse Geometry Block
    # Look for lines with: Symbol  X  Y  Z  Mass
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            try:
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                masses.append(float(parts[4]))
            except ValueError:
                continue

    # 2. Parse Dipole Table (Total a.u. column)
    for i, line in enumerate(lines):
        if "Dipole X" in line:
            dipole_total_au[0] = float(line.split()[-1])
        elif "Dipole Y" in line:
            dipole_total_au[1] = float(line.split()[-1])
        elif "Dipole Z" in line:
            dipole_total_au[2] = float(line.split()[-1])

    # 3. Parse Rotational Constants (Only if in shifted file)
    if is_shifted:
        for line in lines:
            if "Rotational constants:" in line:
                # Extracts numbers using regex: A, B, and C
                matches = re.findall(r"=\s+([-+]?\d*\.\d+|\d+)", line)
                rot_consts = [float(x) for x in matches]

    return np.array(symbols), np.array(coords), np.array(masses), dipole_total_au, rot_consts

# --- Physical Constants ---
# B = h / (8 * pi^2 * c * I)
H = 6.62607015e-34      # J*s
C_LIGHT = 29979245800.0 # cm/s
AMU_TO_KG = 1.66053906660e-27
ANG_TO_M = 1e-10
BOHR_TO_ANG = 0.52917721090  # Psi4 uses this for a.u. conversions
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
TOTAL_CHARGE = 1.0      # This species is a cation

# --- Load Data ---
file_unshifted = "nitro_complex_coordinates.txt"
file_shifted = "nitro_complex_coordinates_com.txt"

syms, coords_un, masses, dip_un_au, _ = parse_psi4_snippet(file_unshifted)
_, coords_sh_psi4, _, dip_sh_psi4_au, rot_psi4 = parse_psi4_snippet(file_shifted, is_shifted=True)

# 1. Compute Center of Mass (COM)
total_mass = np.sum(masses)
com = np.sum(masses[:, np.newaxis] * coords_un, axis=0) / total_mass
manual_shifted_coords = coords_un - com

# 2. Compute Rotational Constants (cm^-1)
I = np.zeros((3, 3))
for m, (x, y, z) in zip(masses, manual_shifted_coords):
    I[0, 0] += m * (y**2 + z**2)
    I[1, 1] += m * (x**2 + z**2)
    I[2, 2] += m * (x**2 + y**2)
    I[0, 1] -= m * x * y
    I[0, 2] -= m * x * z
    I[1, 2] -= m * y * z
I[1,0], I[2,0], I[2,1] = I[0,1], I[0,2], I[1,2]

evals = np.linalg.eigvalsh(I)
evals.sort()
conv = AMU_TO_KG * (ANG_TO_M**2)
manual_rot = [H / (8 * np.pi**2 * C_LIGHT * (Ip * conv)) for Ip in evals]

# 3. Predict Shifted Dipole (a.u.)
# Formula: mu_new = mu_old - (Q * Shift_Vector)
# Shift vector must be in Bohr (atomic units) for the math to work with dipoles in a.u.
com_bohr = com * ANG_TO_BOHR
predicted_dip_au = dip_un_au - (TOTAL_CHARGE * com_bohr)

# --- Comparison Output ---
print(f"{'Property':<25} | {'Manual Value':<20} | {'Psi4 Value':<20} | {'Match?'}")
print("-" * 85)

# Rotational Constants check
abc = ['A', 'B', 'C']
for i in range(3):
    match = np.isclose(manual_rot[i], rot_psi4[i], atol=1e-4)
    print(f"Rot Const {abc[i]} (cm^-1)  | {manual_rot[i]:<20.5f} | {rot_psi4[i]:<20.5f} | {match}")

# Dipole components check
comps = ['X', 'Y', 'Z']
for i in range(3):
    match = np.isclose(predicted_dip_au[i], dip_sh_psi4_au[i], atol=1e-4)
    print(f"Dipole {comps[i]} (a.u.)      | {predicted_dip_au[i]:<20.7f} | {dip_sh_psi4_au[i]:<20.7f} | {match}")

# Coordinates check (just the first atom as a sample)
coord_match = np.allclose(manual_shifted_coords, coords_sh_psi4, atol=1e-5)
print(f"Geometry Shift Match      | {'Verified':<20} | {'Verified':<20} | {coord_match}")

print("\nComputed Center of Mass (Angstroms):", com)
print("\nComputed Charge-Scaled COM:", TOTAL_CHARGE * com)
