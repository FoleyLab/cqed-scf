import numpy as np
import psi4
from cqed_rhf.calculator import CQEDRHFCalculator

# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

geometry = """
0 1
         H           -0.370000000000     0.000000000000    -1.000000000000    
         H            0.370000000000     0.000000000000    -1.000000000000    
         H           -0.370000000000     0.000000000000     1.000000000000    
         H            0.370000000000     0.000000000000     1.000000000000    
symmetry c1
nocom
no_reorient
"""

# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "aug-cc-pvdz",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

# ---------------------------------------------------------
# Run CQED-DFT (lambda = 0)
# ---------------------------------------------------------

lambda_vector = np.zeros(3)

calc = CQEDRHFCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=0.1,
    charge=0,
    multiplicity=1,
    density_fitting=True,
    functional="wb97x-d",
    debug=True,
)



E, grad, _ = calc.energy_and_gradient(geometry)
mol = psi4.geometry(geometry)
psi4.set_options(psi4_options)
E_ref = psi4.energy("wb97x-d")
grad_ref = psi4.gradient("wb97x-d")

print("\nCQED-DFT (λ = 0) wB97x-D")
print("==============================")
print(f"Energy: {E:.10f} Eh")
print(f"Reference Energy: {E_ref:.10f} Eh")
print(f"Energy Difference: {E - E_ref:.2e} Eh")
print(f"Max Gradient Difference: {np.max(np.abs(grad - grad_ref)):.2e} Eh/Bohr")
assert abs(E - E_ref) < 1e-10
assert np.max(np.abs(grad - grad_ref)) < 1e-6
