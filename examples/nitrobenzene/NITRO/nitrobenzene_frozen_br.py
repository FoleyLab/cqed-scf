"""
Molecular dynamics of ortho-nitrobenzene in a cavity
using CQED-DFT, with the bromine atom frozen.

Tracks molecular orientation using NitrobenzeneOrientation.
"""

import numpy as np
import psi4
psi4.core.be_quiet()

from cqed_rhf.utils import write_xyz, ANGSTROM_TO_BOHR
from cqed_rhf.calculator import CQEDRHFCalculator
from cqed_rhf.drivers import velocity_verlet_md
from cqed_rhf.observables.nitrobenzene_orientation import NitrobenzeneOrientation


# =========================
# Geometry
# =========================

nitro_string = """
1 1
C                  0.51932475    1.23303451   -0.03194925
C                  1.94454413    1.26916358   -0.03672882
C                  2.62037793    0.09283428   -0.02499003
C                 -0.19603352    0.03013062    0.00102732
H                 -0.02069420    2.17423764   -0.04336646
H                  2.48281698    2.20891057   -0.03611879
H                 -1.27770137    0.03990295    0.01166953
N                  4.09213475    0.09594076    0.03662979
O                  4.63930696   -1.02169275    0.14459220
O                  4.66489883    1.19839699   -0.02327545
C                  0.49428518   -1.16712649    0.02099746
H                 -0.03251071   -2.11492669    0.05447935
C                  1.96291176   -1.21653219   -0.02111314
H                  2.44359113   -1.96306433    0.61513886
Br                 2.17304025   -1.94912156   -1.90618750
units angstrom
no_reorient
no_com
symmetry c1
"""

# =========================
# Cavity parameters
# =========================

field_vector = np.array([0.078, 0.055, 0.027])
omega = 0.06615

# =========================
# Psi4 options
# =========================

psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

psi4.set_memory("8 GB")
psi4.core.set_output_file("psi4_md.out", False)

# =========================
# Calculator (CQED-DFT)
# =========================

calculator = CQEDRHFCalculator(
    lambda_vector=field_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    functional="wb97x-d",   # now using dispersion-corrected DFT
    charge=1,
    multiplicity=1,
    debug=False,
)

# =========================
# Build orientation tracker
# =========================

mol = psi4.geometry(nitro_string)
symbols = [mol.symbol(i) for i in range(mol.natom())]
coords_bohr = mol.geometry().to_array()

orientation_tracker = NitrobenzeneOrientation(
    symbols=symbols,
    coords_bohr=coords_bohr,
    field_vector=field_vector,
)

# =========================
# Freeze bromine atom
# =========================

freeze_atoms = [i for i, s in enumerate(symbols) if s == "Br"]

print("Frozen atoms:", freeze_atoms)

# =========================
# Initial velocities
# =========================

np.random.seed(42)

natom = len(symbols)
velocities = 0.001 * np.random.randn(natom, 3)

# remove net translation
velocities -= velocities.mean(axis=0)

# =========================
# Run MD
# =========================

traj, observer_data = velocity_verlet_md(
    calculator=calculator,
    geometry=nitro_string,
    velocities=velocities,
    dt=5.0,
    nsteps=10,
    canonical="psi4",
    observers=[orientation_tracker],
    freeze_atoms=freeze_atoms,
    debug=True,
)

# =========================
# Analyze orientation
# =========================

orientation_history = observer_data[orientation_tracker]

phi = np.array([d["phi_deg"] for d in orientation_history])
theta = np.array([d["theta_deg"] for d in orientation_history])

print("\nOrientation evolution:")
for i in range(len(phi)):
    print(f"Step {i:2d} | phi = {phi[i]:7.2f} deg | theta = {theta[i]:7.2f} deg")

print("\nFinal orientation:")
print(f"  phi   = {phi[-1]:.2f} deg")
print(f"  theta = {theta[-1]:.2f} deg")

# =========================
# Write trajectory
# =========================

xyz_file = "nitrobenzene_md_frozen_br.xyz"

for i, frame in enumerate(traj):

    write_xyz(
        filename=xyz_file,
        symbols=symbols,
        coords_angstrom=frame["coords"],
        comment=(
            f"Step {frame['step']}  "
            f"E={frame['energy']:.10f}  "
            f"phi={phi[i]:.3f}  "
            f"theta={theta[i]:.3f}"
        ),
        mode="w" if i == 0 else "a",
    )

print(f"\nTrajectory written to: {xyz_file}")
