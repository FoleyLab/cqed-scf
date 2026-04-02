import numpy as np
import psi4

from cqed_rhf import CQEDSCF
from cqed_rhf import CQEDRHFGradient


# ---------------------------------------------------------
# Geometry (fixed frame)
# ---------------------------------------------------------


BOHR_TO_ANG = 0.52917721092
# ----------------------------
# Molecular geometry (ortho)
# ----------------------------
geometry = """
C           -1.804928163307     1.957993763262     0.703312273806
C           -0.379708783307     1.994122833262     0.698532703806
C            0.296125016693     0.817793533262     0.710271493806
C           -2.520286433307     0.755089873262     0.736288843806
H           -2.344947113307     2.899196893262     0.691895063806
H            0.158564066693     2.933869823262     0.699142733806
H           -3.601954283307     0.764862203262     0.746931053806
N            1.767881836693     0.820900013262     0.771891313806
O            2.315054046693    -0.296733496738     0.879853723806
O            2.340645916693     1.923356243262     0.711986073806
C           -1.829967733307    -0.442167236738     0.756258983806
H           -2.356763623307    -1.389967436738     0.789740873806
C           -0.361341153307    -0.491572936738     0.714148383806
H            0.119338216693    -1.238105076738     1.350400383806
BR          -0.151212663307    -1.224162306738    -1.170925976194
1 1
units angstrom
no_reorient
no_com
symmetry c1
"""


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "sto-3g",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "dft_radial_points": 150,
    "dft_spherical_points": 770
}


# ---------------------------------------------------------
# CQED parameters
# ---------------------------------------------------------

lambda_vector = np.array([0.05, 0.05, 0.05])  # start with lambda = 0
omega = 0.0


# ---------------------------------------------------------
# Compute analytic gradient
# ---------------------------------------------------------

calc = CQEDSCF(
    geometry=geometry,
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    functional="PBE",
)

E, scf_data = calc.run()

gradient_engine = CQEDRHFGradient(lambda_vector, canonical="psi4", debug=False)

results = gradient_engine.compute(scf_data)
grad_qed = results["total_grad"]

grad_analytic = np.array(grad_qed)


# ---------------------------------------------------------
# Finite difference gradient
# ---------------------------------------------------------

delta = 1e-4

mol = psi4.geometry(geometry)
coords = np.array(mol.geometry())

natom = coords.shape[0]

grad_fd = np.zeros_like(coords)

print("\nRunning finite difference gradient check\n")

for atom in range(natom):
    for xyz in range(3):

        coords_p = coords.copy()
        coords_m = coords.copy()

        coords_p[atom, xyz] += delta
        coords_m[atom, xyz] -= delta

        mol_p = mol.clone()
        mol_m = mol.clone()

        mol_p.set_geometry(psi4.core.Matrix.from_array(coords_p))
        mol_m.set_geometry(psi4.core.Matrix.from_array(coords_m))

        geom_p = mol_p.create_psi4_string_from_molecule()
        geom_m = mol_m.create_psi4_string_from_molecule()

        calc_p = CQEDSCF(
            geometry=geom_p,
            lambda_vector=lambda_vector,
            psi4_options=psi4_options,
            omega=omega,
            density_fitting=True,
            functional="PBE",
        )

        calc_m = CQEDSCF(
            geometry=geom_m,
            lambda_vector=lambda_vector,
            psi4_options=psi4_options,
            omega=omega,
            density_fitting=True,
            functional="PBE",
        )

        Ep, _ = calc_p.run()
        Em, _ = calc_m.run()

        grad_fd[atom, xyz] = (Ep - Em) / (2 * delta)


# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

diff = grad_fd - grad_analytic

rms = np.sqrt(np.mean(diff**2))
max_err = np.max(np.abs(diff))

print("\n==============================")
print("Finite Difference Validation")
print("==============================")

print("Analytic gradient:")
print(grad_analytic)

print("\nFinite difference gradient:")
print(grad_fd)

print("\nDifference:")
print(diff)

print("\nRMS error:", rms)
print("Max error:", max_err)


tol = 1e-5

if rms < tol:
    print("\nPASS: analytic gradient validated")
else:
    print("\nWARNING: gradient discrepancy detected")
