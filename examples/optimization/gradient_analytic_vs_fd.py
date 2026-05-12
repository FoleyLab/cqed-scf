import numpy as np

from cqed_scf import CQEDCalculator
from cqed_scf.utils import (
    build_psi4_geometry,
    finite_difference_gradient,
)

# =========================
# System definition
# =========================

symbols = ["O", "H", "H"]

# Geometry in BOHR
coords = np.array(
    [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 1.809],
        [1.751000, 0.000000, -0.453000],
    ]
)

# Light–matter coupling
lambda_vector = [0.0, 0.0, 0.05]

# Psi4 options
psi4_options = {
    "basis": "cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-8,
    "d_convergence": 1e-6,
}

# =========================
# Calculator
# =========================

calc = CQEDCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=0.1,
)

# =========================
# Analytic gradient
# =========================

geometry = build_psi4_geometry(coords, symbols, units="bohr")

#E, grad_analytic, g = calc.energy_and_gradient(geometry)
E, grad_exact, _ = calc.energy_and_gradient(geometry, canonical="exact")
E, grad_df, _ = calc.energy_and_gradient(geometry, canonical="psi4")

print("CQED-RHF Energy (Ha):")
print(f"{E: .10f}\n")

print("Analytic Gradient (Ha/Bohr):")
print(grad_exact)
print()

print("Analytic Gradient with DF (Ha/Bohr):")
print(grad_df)
print()
# =========================
# Numerical gradient
# =========================

grad_fd = finite_difference_gradient(
    calculator=calc,
    coords=coords,
    symbols=symbols,
    delta=1.0e-4,
)

print("Finite Difference Gradient (Ha/Bohr):")
print(grad_fd)
print()

# =========================
# Comparison
# =========================

diff = grad_exact - grad_fd

print("Gradient Difference (Analytic - FD):")
print(diff)
print()

print("Norm of difference:")
print(f"{np.linalg.norm(diff):.6e}")

diff = grad_exact - grad_df
print("Gradient Difference (Exact - DF):")
print(diff)
print()

print("Norm of difference:")
print(f"{np.linalg.norm(diff):.6e}")
