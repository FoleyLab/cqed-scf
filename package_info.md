# CQED-SCF Package Architecture and Data Flow

## Overview

The `cqed_scf` package is designed as a modular electronic structure framework for cavity quantum electrodynamics (CQED) methods, with an emphasis on:

- extensibility,
- reusable infrastructure,
- modular response theory,
- and future many-body methods such as QED-SAPT and QED-TDDFT.

The architecture separates:

1. **user-facing orchestration**
2. **reference wavefunction generation**
3. **analytic derivatives**
4. **response theory**
5. **intermolecular interaction methods**

into distinct but interoperable modules.

The long-term design goal is to make all higher-level methods operate on a consistent electronic-structure interface independent of:

- RHF vs UHF
- RKS vs UKS
- density fitting vs full ERIs
- cavity vs cavity-free calculations
- future correlated methods

---

# Directory Structure

```text
cqed_scf/
│
├── calculator.py
├── references.py
├── scf.py
├── uscf.py
├── gradients.py
├── ugradients.py
├── response.py
│
├── sapt/
│   ├── qed_sapt0.py
│   ├── monomer.py
│   ├── components.py
│   └── results.py
│
└── observables/

High-Level Design Philosophy

The package follows a layered architecture:

User Input
    ↓
CQEDConfig
    ↓
CQEDCalculator
    ↓
SCF / Gradient / Response Engines
    ↓
SAPT / TDDFT / Dynamics / Optimization

The central design principle is:

Each layer consumes standardized electronic structure data from the layer below it.

CQEDConfig
Purpose

CQEDConfig is the central configuration object for the package.

It replaces scattered keyword arguments and loosely structured options dictionaries with a single consistent object.

Responsibilities

CQEDConfig stores:

cavity parameters
reference type
functional
Psi4 options
charge/multiplicity
density fitting settings
debug settings
future response options
Example
config = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.05]),
    omega=0.0735,
    functional="wb97x-d",
    density_fitting=True,
    psi4_options={
        "basis": "6-311g",
        "scf_type": "df",
    },
)
Key Design Features
Automatic Reference Classification

The config object determines:

RHF vs UHF
RKS vs UKS

through helper properties such as:

config.is_restricted
config.is_unrestricted
config.is_dft
config.is_hf
Dispersion Handling

Dispersion-corrected functionals are internally separated into:

config.functional
config.base_scf_functional

Example:

functional = "wb97x-d"
base_scf_functional = "wb97x"

This allows:

CQED-SCF to operate on the orbital functional only,
while dispersion corrections are added externally.
CQEDCalculator
Purpose

CQEDCalculator is the user-facing facade.

It is responsible for:

dispatching calculations,
constructing SCF engines,
constructing gradient engines,
and managing higher-level workflows.
Responsibilities

The calculator:

accepts user geometry,
manages Psi4 state,
handles dispersion corrections,
delegates SCF and gradients,
exposes clean APIs:
energy()
energy_and_gradient()
Delegation Logic

Internally:

CQEDCalculator
    ↓
CQEDSCF or CQEDUSCF
    ↓
CQEDRHFGradient or CQEDUGradient

depending on the reference type stored in CQEDConfig.

SCF Engines
Restricted Engine
scf.py

contains:

CQEDSCF

which currently supports:

RHF
RKS
cavity-modified SCF
Responsibilities

The SCF engine:

builds AO integrals,
constructs cavity Hamiltonian terms,
builds JK objects,
constructs XC potentials,
iterates the SCF equations,
and returns a structured results dictionary.
SCF Result Dictionary

The SCF engine returns a dictionary containing:

results = dict(
    energy_scf=E,
    density=D,
    coefficients=C,
    orbital_energies=eps,
    mints=self.mints,
    wfn=self.wfn,
    ...
)

This dictionary is the primary data interface for all downstream methods.

Unrestricted Engine
Future Module
uscf.py

will contain:

CQEDUSCF

for:

UHF
UKS
spin-polarized cavity calculations

The package architecture already anticipates this extension.

Gradient Engines
Restricted Gradients
gradients.py

contains analytic nuclear gradients for restricted references.

Responsibilities

The gradient engine:

consumes SCF results,
uses stored integrals and wavefunctions,
builds derivative integrals,
solves CPHF-like equations,
computes analytic gradients.
Data Flow
CQEDSCF
    ↓
results dictionary
    ↓
CQEDRHFGradient.compute(results)
Response Theory Module
Purpose
response.py

is intended to become the shared infrastructure layer for:

CPHF
CPKS
TDHF
TDDFT
QED response theory
SAPT induction/dispersion
Key Long-Term Goal

The package is being designed so that:

response theory becomes reusable infrastructure rather than duplicated code.

Future Response Workflow
SCF Results
    ↓
Response Solver
    ↓
AX = b

where:

A = orbital Hessian / response kernel
X = orbital response vector
b = perturbation vector
Density Fitting Strategy

The package is designed to eventually avoid explicit ERI tensor construction.

Instead:

JK.build()

and DF intermediates will be reused to evaluate:

σ = A[X]

iteratively.

This is critical for:

large systems,
TDDFT,
SAPT,
and response calculations.
SAPT Architecture

The sapt/ subpackage is intentionally separated from SCF logic.

SAPTMonomer
Purpose

SAPTMonomer wraps monomer SCF references in a standardized container.

It stores:

geometry
charge/multiplicity
config
SCF result dictionary

and exposes convenience properties.

Convenience Properties

Example:

@property
def C(self):
    return self.scf_results.get("coefficients")

allows:

monomer.C

instead of:

monomer.scf_results["coefficients"]
Why This Matters

This abstraction layer:

isolates SAPT from SCF implementation details,
simplifies future unrestricted support,
simplifies DF support,
simplifies future correlated methods.
Recommended Property Pattern

The monomer object should gradually expose:

occupied orbitals
virtual orbitals
orbital energies
DF tensors
response intermediates

through standardized interfaces.

Example:

@property
def Cocc(self):
    return self.C[:, :self.ndocc]
SAPT Driver Workflow

The future SAPT workflow is intended to look like:

Dimer Geometry
    ↓
Monomer Partitioning
    ↓
Monomer CQED-SCF Calculations
    ↓
SAPTMonomer Objects
    ↓
SAPT Component Evaluations
Planned SAPT Components

Initially:

Electrostatics
Exchange
Induction
Dispersion

at the SAPT0 level.

Initial Implementation Strategy

The first implementation will use:

Full AO ERIs

for simplicity and transparency.

Later versions will migrate to:

Density-fitted tensor contractions

using the shared response infrastructure.

Interoperability Philosophy

A major architectural goal is:

Higher-level methods should never directly depend on low-level SCF implementation details.

Instead:

SCF → standardized results

becomes the interface contract.

This allows:

unrestricted references,
DF implementations,
response methods,
SAPT,
TDDFT,
and future correlated methods

to coexist cleanly.

Long-Term Vision

The package is evolving toward a unified CQED electronic structure platform supporting:

CQED-HF
CQED-DFT
CQED gradients
CQED response theory
QED-TDDFT
QED-SAPT
cavity-modified intermolecular interactions
cavity-modified molecular dynamics
cavity-modified geometry optimization

all sharing a common electronic structure infrastructure.
