import numpy as np
import psi4
from cqed_scf.calculator import CQEDCalculator

# -------------------------
# Psi4 setup
# -------------------------
psi4.set_memory("2 GB")
psi4.set_num_threads(2)

psi4_options = {
    "basis": "aug-cc-pVDZ",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

# -------------------------
# QED parameters
# -------------------------
lambda_vector = np.array([0.0, 0.0, 0.05])
omega = 0.07349864501573

# -------------------------
# Scan range
# -------------------------
d_vals = np.linspace(2.0, 10.0, 41)

# -------------------------
# Calculator
# -------------------------
calc = CQEDCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    functional="wb97x-d",   # IMPORTANT: base functional only
    debug=False,
)

# -------------------------
# Storage
# -------------------------
energies = []
distances = []

# -------------------------
# Scan loop
# -------------------------
for d in d_vals:

    geom = f"""
    0 1
    H  -0.370000000  0.000000000  0.000000000
    H   0.370000000  0.000000000  0.000000000
    H  -0.370000000  0.000000000  {d}
    H   0.370000000  0.000000000  {d}
    symmetry c1
    no_reorient
    nocom
    """

    print(f"Running d = {d:.3f}")

    E = calc.energy(geom)

    energies.append(E)
    distances.append(d)

# -------------------------
# Convert to arrays
# -------------------------
distances = np.array(distances)
energies = np.array(energies)

# -------------------------
# Save results
# -------------------------
np.savetxt(
    "h2_dimer_scan_qed_dispersion_corrected.dat",
    np.column_stack([distances, energies]),
    header="d (Angstrom)    Energy (Hartree)"
)

print("\nScan complete!")
