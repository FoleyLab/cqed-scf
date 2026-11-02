<p align="center">
  <img src="docs/logos/cqed-scf-logo.svg#gh-light-mode-only" width="420" alt="CQED-SCF">
  <img src="docs/logos/cqed-scf-logo-dark.svg#gh-dark-mode-only" width="420" alt="CQED-SCF">
</p>

# CQED-SCF

An open-source package for mean-field, DFT, open-shell, and SAPT-based
molecular cavity quantum electrodynamics calculations using the Pauli–Fierz
Hamiltonian, built on top of Psi4. The package provides CQED-RHF/RKS and
CQED-UHF/UKS energies, restricted analytic nuclear gradients, QED-SAPT0
interaction components, QED-CIS response calculations, molecular dynamics, and
geometry optimization.

## Features

### CQED mean-field methods

- CQED-RHF and CQED-RKS energies and analytic nuclear gradients
- CQED-UHF and CQED-UKS energies for open-shell systems
- Pure, global-hybrid, and range-separated restricted Kohn-Sham references
- Unrestricted Kohn-Sham support through Psi4 `VBase`
- Psi4 JK-backed unrestricted Coulomb and exchange builds
- Spin-resolved orbital, density, Fock, and diagnostic data
- `<S^2>` and spin-contamination diagnostics for unrestricted references

### Gradients and dynamics

- Exact and density-fitted gradient backends
- Velocity-Verlet molecular dynamics
- BFGS geometry optimization
- Finite-difference gradient validation
- Rigid translation/rotation projected gradients
- Psi4-native geometry handling

### QED-SAPT0

- QED-SAPT0 interaction energies and named components: electrostatics,
  exchange, dispersion, exchange-dispersion, induction, and
  exchange-induction
- Dense and density-fitted integral workflows
- JK/DSE-based diagnostic workflows
- Dispersion partitioning into Coulomb-Coulomb, Coulomb-DSE, and DSE-DSE
  contributions
- Restricted RHF/RKS monomer reference workflows

### Response methods

- QED-CIS calculations on CQED-HF and CQED-KS references
- Dense and Davidson QED-CIS solvers
- Photon-number-resolved polaritonic states and transition properties

Unrestricted UHF/UKS support currently provides energies and spin diagnostics.
Analytic unrestricted gradients, unrestricted QED-CIS, and unrestricted response
methods are not yet implemented.

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
```

## Quick example
```python
import numpy as np

from cqed_scf import CQEDCalculator, CQEDConfig

geom = """
0 1
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

config = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.05]),
    omega=0.0,
    psi4_options=psi4_options,
    reference="rks",
    functional="PBE0",
    density_fitting=True,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=False,
)

calc = CQEDCalculator(config=config)
E, grad, g = calc.energy_and_gradient(geom)
```

## QED-SAPT0 component example
```python
import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig

dimer = """
0 1
He 0.0 0.0 0.0
--
0 1
He 0.0 0.0 2.0
units angstrom
symmetry c1
no_reorient
no_com
"""

config = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.05]),
    omega=0.1,
    psi4_options={
        "basis": "6-31g",
        "scf_type": "pk",
        "e_convergence": 1e-10,
        "d_convergence": 1e-8,
    },
    reference="rhf",
    functional=None,
    density_fitting=False,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
)

calc = CQEDCalculator(config=config)
dimer_geometry = psi4.geometry(dimer)

total = calc.sapt0_energy(dimer_geometry)
components = calc.sapt0_components(dimer_geometry)

print(f"Total QED-SAPT0 energy: {total:.12f} Eh")
print(components.summary())
```

## Parameter ownership

- Put standard Psi4 electronic-structure settings in `psi4_options`: basis set,
  SCF type, convergence thresholds, and DFT grid settings.
- Put CQED package settings in `CQEDConfig`: `lambda_vector`, `omega`,
  reference, functional, density-fitting flag, charge, multiplicity,
  dispersion policy, and debug flag.
- Include charge and multiplicity in geometry strings too. This keeps examples
  unambiguous when they are run directly with Psi4.
- Public SAPT0 examples use component names `elst10`, `exch10`, `disp20`,
  `exch_disp20`, `ind20`, `exch_ind20`, and `total`.

## Future TO-DOs

- Mark or reorganize older scaffold-style examples that intentionally use
  lower-level interfaces such as `QEDSAPT0Driver`, direct SAPT intermediates, or
  SAPT JK helpers. These should remain available for development, but they can
  be labeled more clearly.
- Revisit `QEDSAPT0Results` serialization. Its `total` property is computed
  from components; if future workflows need serialized totals, consider adding a
  dedicated `total_energy` field or removing redundant total metadata.

## Notes

This code is intended for research use. Please validate results
carefully and cite appropriately.
