import numpy as np
import psi4

from cqed_rhf.utils import write_xyz
from cqed_rhf import CQEDRHFCalculator
from cqed_rhf.drivers import velocity_verlet_md
from cqed_rhf.observables.nitrobenzene_orientation import NitrobenzeneOrientation
from cqed_rhf.utils import write_xyz, ANGSTROM_TO_BOHR

BOHR_TO_ANG = 0.52917721092
# ----------------------------
# Molecular geometry (ortho)
# ----------------------------
nitro_string = """
0 1
         C           -1.885946870148     1.189583649403    -0.119726128149
         C           -0.498436150781     1.207756989720    -0.095587674731
         C            0.177022900609     0.001244416404     0.003170781601
         C           -2.570320810975    -0.018930814056    -0.046076352491
         H           -2.433490753919     2.122014686561    -0.196502462215
         H            0.061742696521     2.131508866859    -0.151149504465
         H           -3.654760225882    -0.026965279238    -0.065539508188 
         N            1.653234989788     0.012187306043     0.029557107619  
         O            2.221533450190    -1.057060327273     0.115433821878  
         O            2.208198822578     1.089819744626    -0.036210769804    
         C           -1.871287950510    -1.217279026698     0.052462992898    
         H           -2.407539508613    -2.157651159151     0.109753013519    
         C           -0.483684066407    -1.215092211065     0.078168902711    
         H            0.087632738087    -2.130465296765     0.154596572701 
units angstrom
no_reorient
symmetry c1
"""


# ----------------------------
# Cavity / field parameters
# ----------------------------
field_vector = np.array([0.078, 0.055, 0.027])
omega = 0.06615  


# ----------------------------
# Psi4 options
# ----------------------------
psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",          # density fitting
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}

psi4.set_memory("24 GB")
psi4.core.set_output_file("psi4_md.out", False)


# ----------------------------
# Build calculator
# ----------------------------
#calculator = CQEDRHFCalculator(
#    lambda_vector=field_vector, 
#    psi4_options=psi4_options,
#    omega=omega,
#    density_fitting=True,
#    charge=0,
#    multiplicity=1
#)

calculator = CQEDRHFCalculator(
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
# Build orientation tracker
# ----------------------------
# We need initial coords + symbols for setup
mol = psi4.geometry(nitro_string)
symbols = [mol.symbol(i) for i in range(mol.natom())]
coords_bohr = mol.geometry().to_array()

orientation_tracker = NitrobenzeneOrientation(
    symbols=symbols,
    coords_bohr=coords_bohr,
    field_vector=field_vector,
)


# ----------------------------
# Run MD
# ----------------------------
traj, observer_data = velocity_verlet_md(
    calculator=calculator,
    geometry=nitro_string,
    dt=10.0,              # atomic units
    nsteps=5,
    canonical="psi4",
    observers=[orientation_tracker],
    debug=True,
)


# ----------------------------
# Inspect results
# ----------------------------
orientation_history = observer_data[orientation_tracker]

phi = np.array([d["phi_deg"] for d in orientation_history])
theta = np.array([d["theta_deg"] for d in orientation_history])

for i in range(len(phi)):
    
    print(f" phi   at step {i} is {phi[i]:.2f} deg")
    print(f" theta at step {i} is {theta[i]:.2f} deg")

print("\nFinal orientation:")
print(f"  phi   = {phi[-1]:.2f} deg")
print(f"  theta = {theta[-1]:.2f} deg")


xyz_file = "nitrobenzene.xyz"
# write coords and theta and phi to trajectory file
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
