# CQED-SCF

A Python package for mean-field and DFT-based quantum electrodynamics calculations
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
git clone https://github.com/YOUR_USERNAME/cqed-rhf.git
cd cqed-rhf
pip install -e .


## Quick example
```python
from cqed_rhf import CQEDRHFCalculator

geom = """
O  0.000000  0.000000 -0.068516
H  0.000000 -0.790690  0.543701
H  0.000000  0.790690  0.543701
units angstrom
no_reorient
no_com
symmetry c1
"""

calc = CQEDRHFCalculator(
    lambda_vector=[0.0, 0.0, 0.05],
    psi4_options={"basis": "cc-pVDZ"}
)

E, grad, g = calc.energy_and_gradient(geom)
```

## Notes

This code is intended for research use. Please validate results
carefully and cite appropriately.
