import numpy as np
import psi4

from cqed_rhf import CQEDRHFCalculator
from cqed_rhf.utils import write_xyz
from cqed_rhf.drivers import velocity_verlet_md
from cqed_rhf.utils import write_xyz, ANGSTROM_TO_BOHR

# -----------------------------
# Geometry
# -----------------------------
water = """
0 1
O  0.000000  0.000000  0.000000
H  0.000000  0.757000  0.587000
H  0.000000 -0.757000  0.587000
units angstrom
no_reorient
symmetry c1
"""

mol = psi4.geometry(water)
symbols = [mol.symbol(i) for i in range(mol.natom())]
coords_bohr = mol.geometry().to_array()

# -----------------------------
# Psi4 options
# -----------------------------
psi4_options = {
    "basis": "6-311G",
    "scf_type": "df",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}



psi4.set_memory("4 GB")
psi4.core.set_output_file("psi4_test.out", False)


# ----------------------------
# Cavity / field parameters
# ----------------------------
field_vector = np.array([0.078, 0.055, 0.027])
omega = 0.06615  


# -----------------------------
# set up CQED-DFT calculator
# -----------------------------
print("\nRunning CQED-DFT (λ = 0)")

calc = CQEDRHFCalculator(
    lambda_vector=field_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    functional="PBE",
    charge=0,
    multiplicity=1,
    debug=False,
)
# ----------------------------
# Run MD
# ----------------------------
traj, observer_data = velocity_verlet_md(
    calculator=calc,
    geometry=water,
    dt=10.0,              # atomic units
    nsteps=50,
    canonical="psi4",
    observers=None,
    debug=True,
)

xyz_file = "water.xyz"
# write coords and theta and phi to trajectory file
for i, frame in enumerate(traj):

    write_xyz(
        filename=xyz_file,
        symbols=symbols,
        coords_angstrom=frame["coords"],
        comment=(
            f"Step {frame['step']}  "
            f"E={frame['energy']:.10f}  "
        ),
        mode="w" if i == 0 else "a",
    )

