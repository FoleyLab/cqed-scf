# CQED-SCF

An open-source package for mean-field and DFT-based quantum electrodynamics calculations
(Pauli–Fierz Hamiltonian) with analytic nuclear gradients, molecular
dynamics, and geometry optimization, built on top of Psi4.  

## Features

- CQED-RHF and CQED-RKS energy and analytic nuclear gradients
- Exact and DF-based gradient backends
- Velocity-Verlet molecular dynamics
- BFGS geometry optimization
- Finite-difference gradient validation
- Psi4-native geometry handling
- CQED-RKS currently supports pure and global hybrid functionals; support for range-separated hybrids coming soon!

## Requirements

- Python >= 3.9
- Psi4 >= 1.7
- NumPy
- SciPy
- opt_einsum
- pytest

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/cqed-scf.git
cd cqed-scf
pip install -e .


## Quick example
```python
from cqed_scf import CQEDCalculator

geom = """
O  0.000000  0.000000 -0.068516
H  0.000000 -0.790690  0.543701
H  0.000000  0.790690  0.543701
units angstrom
no_reorient
no_com
symmetry c1
"""

psi4_options = {
    "basis": "6-311g",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

# ---------------------------------------------------------
# Run CQED-DFT (lambda = 0.05 along z)
# ---------------------------------------------------------

lambda_vector = np.array([0, 0, 0.05])

calc = CQEDSCF(
    geometry=geom,
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=0.0,
    density_fitting=True,
    functional="PBE0",
    debug=False,
)

E, grad, g = calc.energy_and_gradient(geom)
```

## Notes

This code is intended for research use. Please validate results
carefully and cite appropriately.
