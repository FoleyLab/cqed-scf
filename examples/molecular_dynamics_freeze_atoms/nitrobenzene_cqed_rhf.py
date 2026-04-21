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

#nitro_string = """
#1 1
#C                  0.51932475    1.23303451   -0.03194925
#C                  1.94454413    1.26916358   -0.03672882
#C                  2.62037793    0.09283428   -0.02499003
#C                 -0.19603352    0.03013062    0.00102732
#H                 -0.02069420    2.17423764   -0.04336646
#H                  2.48281698    2.20891057   -0.03611879
#H                 -1.27770137    0.03990295    0.01166953
#N                  4.09213475    0.09594076    0.03662979
#O                  4.63930696   -1.02169275    0.14459220
#O                  4.66489883    1.19839699   -0.02327545
#C                  0.49428518   -1.16712649    0.02099746
#H                 -0.03251071   -2.11492669    0.05447935
#C                  1.96291176   -1.21653219   -0.02111314
#H                  2.44359113   -1.96306433    0.61513886
#Br                 2.17304025   -1.94912156   -1.90618750
#units angstrom
#no_reorient
#no_com
#symmetry c1
#"""

# nitro - Br+ complex v1: Br+ is above center of mass of nitrobenzene
#nitro_string = """
#1 1
#         C           -1.885946869708     1.189583649926    -1.034153341072
#         C           -0.498436149708     1.207756989926    -1.010014881072
#         C            0.177022900292     0.001244419926    -0.911256431072
#         C           -2.570320809708    -0.018930810074    -0.960503561072
#         H           -2.433490749708     2.122014689926    -1.110929671072
#         H            0.061742700292     2.131508869926    -1.065576711072
#         H           -3.654760229708    -0.026965280074    -0.979966721072
#         N            1.653234990292     0.012187309926    -0.884870101072
#         O            2.221533450292    -1.057060330074    -0.798993391072
#         O            2.208198820292     1.089819739926    -0.950637981072
#         C           -1.871287949708    -1.217279030074    -0.861964221072
#         H           -2.407539509708    -2.157651160074    -0.804674201072
#         C           -0.483684069708    -1.215092210074    -0.836258311072
#         H            0.087632740292    -2.130465300074    -0.759830641072
#         BR           0.000000000292    -0.000000000074     1.425572788928
#no_com
#no_reorient
#symmetry c1
#"""

# nitro - Br+ complex v2: Br+ is above center of aromatic ring
#nitro_string = """
#1 1
# C           -1.468931365599     1.178697647085    -1.610476189598
# C           -0.081420645599     1.196870987085    -1.586337729598
# C            0.594038404401    -0.009641582915    -1.487579279598
# C           -2.153305305599    -0.029816812915    -1.536826409598
# H           -2.016475245599     2.111128687085    -1.687252519598
# H            0.478758204401     2.120622867085    -1.641899559598
# H           -3.237744725599    -0.037851282915    -1.556289569598
# N            2.070250494401     0.001301307085    -1.461192949598
# O            2.638548954401    -1.067946332915    -1.375316239598
# O            2.625214324401     1.078933737085    -1.526960829598
# C           -1.454272445599    -1.228165032915    -1.438287069598
# H           -1.990524005599    -2.168537162915    -1.380997049598
# C           -0.066668565599    -1.225978212915    -1.412581159598
# H            0.504648244401    -2.141351302915    -1.336153489598
# BR          -0.650118399399     0.016971049385     2.324047987802
#no_com
#no_reorient
#symmetry c1
#"""
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
#field_vector = np.array([0.038, 0.082, 0.042])
omega = 0.06615

# =========================
# Psi4 options
# =========================

psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-9
}
#    "dft_radial_points": 99,
#    "dft_spherical_points": 590,
#    "dft_pruning_scheme": "none"
#}

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
    functional=None,   # now using dispersion-corrected DFT
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

xyz_file = "nitrobenzene_direction_A_cqedrhf_4000_ts.xyz"

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
