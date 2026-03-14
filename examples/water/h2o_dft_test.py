import numpy as np
import psi4
from cqed_rhf import CQEDRHFCalculator

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


# -----------------------------
# Lambda = 0 (no cavity)
# -----------------------------
lambda_vector = np.zeros(3)

omega = 0.1


# -----------------------------
# Run CQED-DFT
# -----------------------------
print("\nRunning CQED-DFT (λ = 0)")

calc = CQEDRHFCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    functional="PBE",
    charge=0,
    multiplicity=1,
    debug=False,
)

E_qed, grad_qed, g_qed = calc.energy_and_gradient(water)
# -----------------------------
# Run Psi4 PBE
# -----------------------------
print("\nRunning reference Psi4 PBE")

psi4.set_options(psi4_options)

mol = psi4.geometry(water)

E_ref, wfn = psi4.energy("pbe", return_wfn=True)

#grad_ref = np.asarray(psi4.core.scfgrad(wfn))
grad_ref = psi4.gradient("pbe", ref_wfn=wfn).np

# -----------------------------
# Compare energies
# -----------------------------
print("\nEnergy comparison")

print("CQED-DFT energy :", E_qed)
print("Psi4 PBE energy :", E_ref)

print("Energy difference :", abs(E_qed - E_ref))


# -----------------------------
# Compare gradients
# -----------------------------
print("\nGradient comparison")

print("CQED gradient:\n", grad_qed)
print("\nPsi4 gradient:\n", grad_ref)

grad_diff = grad_qed - grad_ref

print("\nGradient difference:\n", grad_diff)

print("\nGradient RMS difference:",
      np.linalg.norm(grad_diff))




