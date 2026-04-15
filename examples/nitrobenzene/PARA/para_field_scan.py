import numpy as np
import psi4

from cqed_rhf.utils import write_xyz
from cqed_rhf.calculator import CQEDRHFCalculator
from cqed_rhf.drivers import velocity_verlet_md
from cqed_rhf.observables.nitrobenzene_orientation import NitrobenzeneOrientation
from cqed_rhf.utils import write_xyz, ANGSTROM_TO_BOHR

#<-- Change this to a number of theta and phi values
#<-- you want to compute the scan over!
num_phi_vals = 24
num_theta_vals = 24

def generate_field_vector_from_theta_and_phi(theta, phi):
    """
    Generate a unit field vector from spherical coordinates.

    Parameters:
    -----------
    theta : float
        Polar angle in degrees (0° = +z axis, 90° = xy-plane, 180° = -z axis)
    phi : float
        Azimuthal angle in degrees (0° = +x axis, 90° = +y axis)

    Returns:
    --------
    array : Field vector [x, y, z] as a unit vector

    Spherical to Cartesian conversion:
        x = sin(θ) cos(φ)
        y = sin(θ) sin(φ)
        z = cos(θ)
    """
    # Convert degrees to radians
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    # Compute Cartesian components
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)

    return np.array([x, y, z])

# Example: Generate field vector
theta = 45  # 45° from z-axis
phi = 45.0    # 30° from x-axis in xy-plane

# compute field vector and print it
field_vector = generate_field_vector_from_theta_and_phi(theta, phi) * 0.1  # scale by field strength
print(f"Field vector for θ={theta}°, φ={phi}°: {field_vector}")

# ----------------------------
# Molecular geometry (para)
# ----------------------------
para_string = """
1 1
         C           -0.511618296797     1.244386024531     0.732140048697
         C            0.856500593203     1.251903714531     0.717948218697
         C            1.524118723203     0.024661924531     0.713927788697
         H           -1.071804396797     2.172682314531     0.745925708697
         H            1.436128963203     2.163921874531     0.712099008697
         N            3.008539583203     0.046097104531     0.698823798697
         O            3.575097303203    -1.082768165469     0.699174708697
         O            3.542114363203     1.190870854531     0.689202018697
         C           -0.475464946797    -1.253402765469     0.742118638697
         H           -1.008574426797    -2.197377955469     0.762945788697
         C            0.892227703203    -1.221407805469     0.728065818697
         H            1.498048423203    -2.116244695469     0.729653208697
         C           -1.267906576797    -0.015841805469     0.712127418697
         H           -2.116520796797    -0.025293215469     1.403161498697
         BR          -2.114966986797    -0.034666925469    -1.121874081303
units angstrom
no_reorient
no_com
symmetry c1
"""


# ----------------------------
# Psi4 options
# ----------------------------
psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",          # density fitting
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none"
}

psi4.set_memory("24 GB")
psi4.core.set_output_file("bad_para_qeddft_wb97x-d_scan.out", False)
omega = 0.06615  # cavity frequency in atomic units (corresponding to ~1.8 eV)

# ----------------------------
# Build calculator
# ----------------------------
calculator = CQEDRHFCalculator(
    lambda_vector=field_vector, # molecule_string=ortho_string, #<-- CQEDRHFCalculator doesn't take molecule_string currently?
    psi4_options=psi4_options,
    omega=0.06615,
    charge=1,
    multiplicity=1,
    density_fitting=True,
    functional="wb97x-d",
    debug=False,
)

# ----------------------------
mol = psi4.geometry(para_string)
psi4.set_options(psi4_options)
# compute psi4 rhf energy for initial geometry
e_rhf_1 = psi4.energy("scf", molecule=mol)
print(f"Initial RHF energy: {e_rhf_1:.6f} Hartree")
# use calculator to run a single point and get initial energy and forces
e_cqed_1 = calculator.energy(para_string)
print(f"Initial CQED-RHF energy: {e_cqed_1:.6f} Hartree")


# loop over theta from 0 to 180 and phi from 0 to 360 and compute energy for each orientation in increments of 2 degree
# print theta, phi, x, y, z, energy to a file or stdout in a formatted way
theta_list = np.linspace(0, 180, num_theta_vals) #np.arange(0, 181, num_theta_vals)
phi_list = np.linspace(0, 360, num_phi_vals) #np.arange(0, 361, num_phi_vals)

print(F"Printing theta list which has {num_theta_vals} vals")
print(theta_list)

print(F"Printing phi list which has {num_phi_vals} vals")
print(phi_list)

print(F"Total size of grid is {num_theta_vals * num_phi_vals}")

# Note that when theta = 0, all phi values give the same field vector (0, 0, 0.1) since the field is pointing along the z-axis. Similarly, when theta = 180, all phi values give the same field vector (0, 0, -0.1). The variation in energy will be more pronounced at intermediate theta values where the field vector changes direction in the xy-plane as phi varies.
# so we don't need to repeat these calculation for all phi values at theta = 0 and theta = 180, we can just compute it once for each of these theta values and print the result. This will save computational time while still capturing the key orientations of the field.
field_vector_theta_0 = generate_field_vector_from_theta_and_phi(0, 0) * 0.1
field_vector_theta_180 = generate_field_vector_from_theta_and_phi(180, 0) * 0.1

# update calculator with field vector for theta = 0 and compute energy
calculator.lambda_vector = field_vector_theta_0
c_cqed_theta_0 = calculator.energy(para_string)

# update calculator with field vector for theta = 180 and compute energy
calculator.lambda_vector = field_vector_theta_180
c_cqed_theta_180 = calculator.energy(para_string)


# open file for writing
with open("bad_para_field_scan_qeddft_wb97x-d.txt", "w") as f:
    for theta in theta_list:
        for phi in phi_list:
            field_vector = generate_field_vector_from_theta_and_phi(theta, phi) * 0.1  # scale by field strength

            if theta == 0:
                field_vector = field_vector_theta_0
                e_cqed = c_cqed_theta_0
            elif theta == 180:
                field_vector = field_vector_theta_180
                e_cqed = c_cqed_theta_180
            else:   
                calculator.lambda_vector = field_vector
                e_cqed = 0.0
		#e_cqed = calculator.energy(para_string)
            print(f"{theta:3f} {phi:3f} {field_vector[0]: .4f} {field_vector[1]: .4f} {field_vector[2]: .4f} {e_cqed:.12f}")
            f.write(f"{theta:3f} {phi:3f} {field_vector[0]: .4f} {field_vector[1]: .4f} {field_vector[2]: .4f} {e_cqed:.12f}\n")

f.close()


