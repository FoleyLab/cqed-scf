import numpy as np
import psi4

from cqed_rhf import CQEDSCF
from cqed_rhf import CQEDRHFGradient


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

geometry = """
0 1
O
H 1 0.9572
H 1 0.9572 2 104.5
symmetry c1
"""

# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "6-311g",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

# ---------------------------------------------------------
# Run CQED-DFT (lambda = 0)
# ---------------------------------------------------------

lambda_vector = np.zeros(3)

calc = CQEDSCF(
    geometry=geometry,
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=0.1,
    density_fitting=True,
    functional="wB97x",
    debug=True,
)

print("\nRunning wB97x-D (λ = 0)\n")

E_qed, scf_data = calc.run()

gradient_engine = CQEDRHFGradient(lambda_vector, canonical="psi4", debug=False)

grad_qed = gradient_engine.compute(scf_data)


# ---------------------------------------------------------
# Run reference Psi4 calculation
# ---------------------------------------------------------

psi4.set_options(psi4_options)

mol = psi4.geometry(geometry)

print("\nRunning Psi4 reference wB97x-D\n")

E_ref, wfn_ref = psi4.energy("wB97X-D", return_wfn=True)
E_no_disp, wfn_no_disp = psi4.energy("wB97X", return_wfn=True)

try:
   E_disp = psi4.variable("DISPERSION CORRECTION ENERGY")
except KeyError:
   E_disp = 0.0
# subtract dispersion correction
#E_ref -= E_disp

grad_ref = psi4.gradient("wB97X-D", ref_wfn=wfn_ref).np

grad_no_disp = psi4.gradient("wB97X", ref_wfn=wfn_no_disp).np

disp_grad = grad_ref - grad_no_disp

# ---------------------------------------------------------
# Compare results
# ---------------------------------------------------------

print("\n==============================")
print(" Energy Comparison")
print("==============================")

print(f"CQED-wB97X-D              energy : {E_qed:20.12f}")
print(f"Psi4 wB97X-D - Dispersion energy : {E_ref:20.12f}")

dE = abs(E_qed - E_ref)

print(f"Energy difference : {dE:12.6e}")


print("\n==============================")
print(" Gradient Comparison")
print("==============================")

#grad_qed = np.zeros((3,3))
grad_qed = np.array(grad_qed) + disp_grad
grad_ref = np.array(grad_ref)

grad_diff = grad_qed - grad_ref

rms = np.sqrt(np.mean(grad_diff**2))
max_err = np.max(np.abs(grad_diff))

print(f"Gradient RMS error : {rms:12.6e}")
print(f"Gradient max error : {max_err:12.6e}")

print("\nCQED gradient:")
print(grad_qed)

print("\nPsi4 gradient:")
print(grad_ref)

print("\nDispersion gradient:")
print(disp_grad)


# ---------------------------------------------------------
# Regression pass/fail
# ---------------------------------------------------------

E_tol = 1e-8
G_tol = 1e-6

print("\n==============================")
print(" Regression Result")
print("==============================")

if dE < E_tol and rms < G_tol:
    print("PASS: CQED-DFT reproduces Psi4 wB97X-D.")
else:
    print("FAIL: Results differ from Psi4 reference.")
