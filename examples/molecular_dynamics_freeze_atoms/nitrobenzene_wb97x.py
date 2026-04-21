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
C           -1.901922841579     1.172760393927    -0.047772892327
C           -0.476703461579     1.208889463927    -0.052552462327
C            0.199130338421     0.032560163927    -0.040813672327
C           -2.617281111579    -0.030143496073    -0.014796322327
H           -2.441941791579     2.113963523927    -0.059190102327
H            0.061569388421     2.148636453927    -0.051942432327
H           -3.698948961579    -0.020371166073    -0.004154112327
N            1.670887158421     0.035666643927     0.020806147673
O            2.218059368421    -1.081966866073     0.128768557673
O            2.243651238421     1.138122873927    -0.039099092327
C           -1.926962411579    -1.227400606073     0.005173817673
H           -2.453758301579    -2.175200806073     0.038655707673
C           -0.458335831579    -1.276806306073    -0.036936782327
H            0.022343538421    -2.023338446073     0.599315217673
no_com
no_reorient
symmetry c1
"""

# =========================
# Cavity parameters
# =========================

field_vector = np.array([0.07878123598, 0.0551632153, 0.02739592187])
omega = 0.06615

# =========================
# Psi4 options
# =========================

psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-9,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none"
}

psi4.set_memory("8 GB")
psi4.core.set_output_file("psi4_md_wb97x.out", False)

# =========================
# Calculator (CQED-DFT)
# =========================

calculator = CQEDRHFCalculator(
    lambda_vector=field_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    functional="wb97x",   # now using dispersion-corrected DFT
    charge=0,
    multiplicity=1,
    debug=True,
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

#freeze_atoms = [i for i, s in enumerate(symbols) if s == "BR"]

#print("Frozen atoms:", freeze_atoms)
#for i in freeze_atoms:
#    print(f"Atom index {i} ({symbols[i]}) is FROZEN")
# =========================
# Initial velocities
# =========================

np.random.seed(42)

natom = len(symbols)
#velocities = 0.001 * np.random.randn(natom, 3)
velocities = np.zeros((natom, 3))

# remove net translation
#velocities -= velocities.mean(axis=0)

# =========================
# Run MD
# =========================

traj, observer_data = velocity_verlet_md(
    calculator=calculator,
    geometry=nitro_string,
    velocities=velocities,
    dt=25.0,
    nsteps=4000,
    canonical="psi4",
    observers=[orientation_tracker],
    freeze_atoms=None,
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

xyz_file = "nitrobenzene_direction_A_wb97x_4000_ts.xyz"

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
