# AGENTS.md

## Project overview

`cqed-scf` is a Psi4-based Python package for molecular cavity quantum
electrodynamics using the Pauli–Fierz Hamiltonian. The preferred user-facing
interface is `CQEDConfig` plus `CQEDCalculator`:

```python
from cqed_scf import CQEDCalculator, CQEDConfig

config = CQEDConfig(
    lambda_vector=(0.0, 0.0, 0.05),
    omega=0.1,
    psi4_options={"basis": "cc-pVDZ", "scf_type": "df"},
    reference="rhf",
    functional=None,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
)
calc = CQEDCalculator(config=config)
energy = calc.energy(geometry)
```

Use the high-level calculator for ordinary calculations. Use the lower-level
SCF, SAPT, response, or gradient classes only when a workflow needs diagnostic
data or functionality not exposed by the facade.

The package is research software. Preserve numerical conventions, validate
against the existing tests and examples, and do not silently change a physics
definition to make an example pass.

## Environment and validation

Requirements include Python >= 3.9, Psi4, NumPy, SciPy, and `opt-einsum`.
Psi4 is intentionally not listed as a pip dependency in `pyproject.toml`.

From the repository root:

```bash
pip install -e .
pytest
```

For focused validation, prefer the smallest relevant test file, for example:

```bash
pytest tests/test_qedsapt0_driver.py
pytest tests/test_unrestricted_plumbing.py
pytest tests/test_fd_gradient_qed_dft.py
pytest tests/test_qed_cis_dense.py
```

Canonical examples live under `examples/canonical/`. New examples should use
`CQEDConfig` and `CQEDCalculator` unless they intentionally demonstrate a
lower-level diagnostic interface.

## Configuration rules

Keep standard electronic-structure options in `psi4_options`:

- `basis`
- `scf_type` (`"pk"` or `"df"`)
- SCF convergence thresholds
- DFT grid settings
- other ordinary Psi4 options

Keep CQED settings in `CQEDConfig`:

- `lambda_vector`: three Cartesian coupling components, in atomic units;
- `omega`: photon frequency, in Hartree;
- `reference`: `"rhf"`, `"uhf"`, `"rks"`, or `"uks"`;
- `functional`: required for RKS/UKS and omitted for RHF/UHF;
- `density_fitting`;
- `charge` and `multiplicity`;
- `dispersion_policy`: `"none"`, `"post_scf"`, or `"scf"`.

The geometry and configuration must agree on charge and multiplicity. Include a
leading charge/multiplicity line in every geometry string, even when Psi4 could
infer it. Do not put `reference` inside `psi4_options`; set it on
`CQEDConfig`.

`reference=None` defaults to RKS when a functional is supplied and RHF
otherwise. Do not rely on multiplicity alone to select an unrestricted method;
use `reference="uhf"` or `reference="uks"` explicitly.

For known dispersion-augmented functionals such as `wb97x-d`, the default
`dispersion_policy="post_scf"` strips the dispersion suffix for the CQED-SCF
reference and adds the correction afterward. Set the policy deliberately when
reproducing a published protocol.

## Single-point energies

### QED-RHF

Use `reference="rhf"` and `functional=None`:

```python
config = CQEDConfig(
    lambda_vector=(0.0, 0.0, 0.05),
    omega=0.1,
    psi4_options={"basis": "cc-pVDZ", "scf_type": "pk"},
    reference="rhf",
    functional=None,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
)
energy = CQEDCalculator(config=config).energy(geometry)
```

Reference example: `examples/canonical/cqed_rhf_energy.py`.

### QED-UHF

Use `reference="uhf"` and an appropriate open-shell multiplicity:

```python
config = CQEDConfig(
    lambda_vector=(0.0, 0.0, 0.05),
    omega=0.1,
    psi4_options={"basis": "6-31g", "scf_type": "df"},
    reference="uhf",
    functional=None,
    charge=0,
    multiplicity=2,
    dispersion_policy="none",
)
energy = CQEDCalculator(config=config).energy(open_shell_geometry)
```

Unrestricted energies and spin diagnostics are implemented. Unrestricted
analytic gradients, QED-CIS, and response methods are not implemented.

### QED-RKS

Use `reference="rks"` and provide the functional:

```python
config = CQEDConfig(
    lambda_vector=(0.0, 0.0, 0.05),
    omega=0.1,
    psi4_options={"basis": "6-311g", "scf_type": "df"},
    reference="rks",
    functional="PBE0",
    density_fitting=True,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
)
energy = CQEDCalculator(config=config).energy(geometry)
```

The same path supports pure, global-hybrid, and supported range-separated
restricted Kohn–Sham functionals. See
`examples/canonical/cqed_dft_energy_gradient.py` for a complete configuration.

### QED-UKS

Use `reference="uks"`, a functional, and the correct multiplicity:

```python
config = CQEDConfig(
    lambda_vector=(0.0, 0.0, 0.05),
    omega=0.1,
    psi4_options={"basis": "6-31g", "scf_type": "df"},
    reference="uks",
    functional="PBE0",
    density_fitting=True,
    charge=0,
    multiplicity=2,
    dispersion_policy="none",
)
energy = CQEDCalculator(config=config).energy(open_shell_geometry)
```

The unrestricted path uses Psi4 `VBase` and JK-backed Coulomb/exchange builds.
Use `tests/test_uhf_jk_backend.py` and the unrestricted examples as references
when changing this path.

### QED-SAPT0

QED-SAPT0 takes a two-fragment Psi4 molecule. Use the restricted RHF or RKS
monomer reference path:

```python
components = calc.sapt0_components(
    psi4.geometry(dimer),
    integral_backend="full_eri",  # or "df"
    include_cavity_terms=True,
    monomer_reference_frame="monomer_com",
)

print(components.elst10)
print(components.exch10)
print(components.ind20)
print(components.exch_ind20)
print(components.disp20)
print(components.exch_disp20)
print(components.total)
```

`calc.sapt0_energy(...)` returns only the total. `calc.sapt0_components(...)`
returns a `QEDSAPT0Results` object containing the six named components and
metadata. Use `integral_backend="df"` for larger calculations and
`"full_eri"` for dense verification. The DF path uses JKFIT-like fitting for
Coulomb terms and RIFIT-like fitting for correlation/dispersion terms.

For intermolecular distance scans, prefer
`monomer_reference_frame="monomer_com"`. CQED orbital gaps are origin
dependent even though the total energy and density are translation invariant;
the intrinsic monomer frame avoids an artificial long-range dispersion error.

### QED-CIS

QED-CIS is called through `calc.cis(...)` on a restricted CQED-HF or CQED-KS
reference:

```python
results = calc.cis(
    geometry,
    nroots=6,
    n_photon=1,
    solver="davidson",  # use "dense" for small-system verification
)
```

The result contains eigenvalues relative to the SCF reference, excitation
energies, photon numbers, block weights, transition dipoles, and oscillator
strengths. This is configuration interaction in a composite electron–photon
Fock basis, not linear-response QED-TDDFT. `calc.tddft()` is deliberately not
implemented and should not be used as an alias for QED-CIS.

## Energy scans

For a scan, build one calculator for fixed electronic and cavity settings, then
loop over geometries and record the returned values. Rebuild the Psi4 geometry
for each point and keep charge/multiplicity consistent.

```python
rows = []
for distance, geometry in scan_geometries:
    energy = calc.energy(geometry)
    rows.append({"distance": distance, "energy": energy})
```

For QED-SAPT0, record each `QEDSAPT0Results` field and the selected backend.
For QED-CIS, record excitation energies and state-character data rather than
calling the result a ground-state energy. Track states by photon character,
block weights, or overlaps when a scan can reorder roots; root index alone is
only safe in simple weak-coupling cases.

Use the scan examples under `examples/qed_sapt0_water_methylamine/`,
`examples/h2_dimer/`, and `examples/qed_cis/` as starting points.

## QED-SAPT0 energy decomposition

### Standard six components

The ordinary public decomposition is:

| Result field | Meaning |
|---|---|
| `elst10` | electrostatics |
| `exch10` | exchange, S² convention |
| `ind20` | induction-response |
| `exch_ind20` | exchange induction-response |
| `disp20` | dispersion |
| `exch_disp20` | exchange dispersion |

The public total is the sum of these six components. QED-SAPT0 in this package
does not add Psi4's `delta_HF` correction, so compare component by component
with conventional SAPT0 unless the protocols explicitly match.

### Dispersion standard/cavity partition

`Disp20` is quadratic in the interaction numerator. The result metadata exposes
the exact three-way partition:

```python
partition = components.metadata["disp20_partition"]
standard = partition["standard"]
cross = partition["cross"]
cavity = partition["cavity"]
total = partition["total"]
assert np.isclose(standard + cross + cavity, total)
```

Here `standard` is Coulomb–Coulomb, `cross` contains both Coulomb–DSE cross
orderings, and `cavity` is DSE–DSE. Do not expect `standard + cavity` to equal
the total; the cross term is required.

### Advanced lower-level operator contexts

For diagnostics that are not exposed by the facade, construct a
`QEDSAPT0Driver` directly:

```python
import psi4
from cqed_scf.sapt import QEDSAPT0Driver

dimer_mol = psi4.geometry(dimer)
driver = QEDSAPT0Driver(
    dimer_geometry=dimer_mol,
    config=config,
    integral_backend="full_eri",  # or "df"
    include_cavity_terms=True,
    monomer_reference_frame="monomer_com",
)
driver.prepare_monomers()
driver.build_integrals()

for context in ("standard", "cavity", "total"):
    eri_like_block = driver.v("abrs", context=context, df_role="corr")
    interaction_block = driver.vt("abab", context=context, df_role="scf")
    print(context, eri_like_block.shape, interaction_block.shape)
```

The valid operator contexts are:

- `"standard"`: ordinary Coulomb/exchange and nuclear interaction terms;
- `"cavity"`: cavity/DSE contribution only;
- `"total"`: standard plus cavity operator.

Use `df_role="scf"` for electrostatics, exchange, and induction-like
quantities; use `df_role="corr"` for dispersion and exchange-dispersion
quantities. The dense `full_eri` path does not use the fitting role, but keeping
the role explicit makes a diagnostic harness portable between backends.

The driver also provides useful context-aware primitives:

```python
parts = driver.vt_partitions("abab")
for context in ("standard", "cavity", "total"):
    print(context, parts[context]["eri"])
    print(context, parts[context]["potential_A"])
    print(context, parts[context]["potential_B"])
    print(context, parts[context]["constant"])
    print(context, parts[context]["total"])

driver.diagnostic_summary(print_output=True)
driver.diagnostic_chf_rhs("A", context="cavity")
driver.diagnostic_chf_matrix("A", context="cavity")
```

Important implementation details for an all-six-component context study:

1. The current public `compute_Elst100()`, `compute_Exch100()`,
   `compute_Eind200()`, and `compute_Eexchind200()` methods do not accept a
   `context` keyword. Do not invent one at a call site and do not claim that
   these methods already return standard/cavity decompositions.
2. The current `compute_Edisp200(..., context=...)` path is context-aware, but
   the driver caches dispersion amplitudes. Rebuild amplitudes for each context
   and verify `driver.t_rsab_context` before evaluating exchange-dispersion.
3. The current exchange-dispersion routine consumes the cached amplitudes and
   its operator contractions are not a complete public context API. A robust
   diagnostic harness must thread `context` through every `v()`/`vt()` call and
   through the corresponding dispersion-amplitude and response intermediates.
4. Induction and exchange-induction require context-specific response solves.
   Reusing a total-context CPHF solution for a standard- or cavity-context
   contraction is invalid.
5. Standard plus cavity equals total at the operator/tensor level. It is not
   automatically an additive identity for response-dependent energies. For
   `Disp20`, use the existing `dispersion_energy_partition()` three-way
   decomposition instead of a naive two-way sum.

When implementing or extending this diagnostic path, use a fresh driver or
explicitly clear/rebuild context-sensitive caches for every context. Compare
the total-context values with `driver.run_components()` and use
`include_cavity_terms=False` and/or `lambda_vector=(0, 0, 0)` as controls.
Relevant implementation and regression material is in
`src/cqed_scf/sapt/qed_sapt0.py`, `tests/test_qedsapt0_driver.py`,
`tests/test_dse_df.py`, `examples/canonical/cqed_sapt0_components.py`, and
`examples/canonical/cqed_sapt0_dispersion_partition.py`.

## Gradients and geometry optimization

Restricted QED-RHF and QED-RKS support analytic nuclear gradients:

```python
energy, gradient, coupling = calc.energy_and_gradient(geometry)
```

The gradient is in Hartree/Bohr. `canonical="psi4"` uses the Psi4-backed
canonical path; `canonical="exact"` selects the exact alternative where
supported.

Use the BFGS driver for optimization:

```python
from cqed_scf.drivers import bfgs_optimize

result, observer_data = bfgs_optimize(
    calculator=calc,
    geometry=geometry,
    canonical="psi4",
    gtol=1e-5,
    maxiter=50,
    debug=True,
)
optimized_coords_bohr = result.x
```

For a fixed Cartesian translation/rotation-free optimization subspace, pass
`project_tr_rot=True`. This projects both the gradient and BFGS coordinate
steps; projecting only the gradient is insufficient because the inverse Hessian
can reintroduce rigid modes. See
`examples/optimization/h2o_optimization_projected.py`.

Optimization currently requires a restricted reference because unrestricted
analytic gradients are not implemented. Keep `no_reorient` and, when needed,
`no_com` explicit so Psi4 does not silently alter the coordinate frame.

## Molecular dynamics

Use the velocity-Verlet driver with a restricted gradient-capable calculator:

```python
from cqed_scf.drivers import velocity_verlet_md

trajectory, observer_data = velocity_verlet_md(
    calculator=calc,
    geometry=geometry,
    dt=10.0,       # atomic units
    nsteps=50,
    canonical="psi4",
    observers=None,
    debug=True,
)
```

The returned trajectory stores energies, Cartesian coordinates in Angstrom and
Bohr, velocities, forces, and the effective cavity coupling. Use
`freeze_atoms=[...]` for frozen nuclei and observers with an
`observe(coords_bohr)` method for custom measurements. Write frames with
`cqed_scf.utils.write_xyz` when a persistent trajectory is needed.

MD also requires restricted analytic gradients. Check timestep stability and
energy drift for every new system; do not treat a short trajectory as evidence
of production-level numerical stability.

## Unsupported or intentionally reserved features

- Unrestricted analytic gradients are not implemented.
- Unrestricted QED-CIS and unrestricted response are not implemented.
- Linear-response QED-TDDFT is not implemented; `tddft()` is reserved for that
  distinct formalism.
- QED-SAPT0 currently uses restricted RHF/RKS monomer references.
- The lower-level SAPT context primitives are diagnostic interfaces, not a
  stable high-level result schema.

## Development conventions

- Prefer `apply_patch` for source and documentation edits.
- Preserve unrelated user changes and inspect `git status` before editing.
- Add or update a focused regression test for numerical or API changes.
- Keep public examples aligned with the current `CQEDConfig` /
  `CQEDCalculator` interface.
- Run formatting/linting tools only when configured by the repository; do not
  introduce a new formatter as part of an unrelated change.
- For physics changes, document the operator convention and the relevant
  limiting-case control, especially λ=0, cavity-off, dense-vs-DF, or
  translation/origin behavior.

## Useful references

- `README.md`: package overview and public examples.
- `examples/README.md`: canonical versus diagnostic examples.
- `docs/qed_sapt0_formalism.tex`: QED-SAPT0 formalism and cavity partitions.
- `docs/SAPT_NOTE.md`: SAPT implementation notes.
- `tests/test_qedsapt0_driver.py`: SAPT driver and component regressions.
- `tests/test_dse_df.py`: dense/DF operator-context and DSE checks.
- `tests/test_unrestricted_plumbing.py`: UHF/UKS configuration and guards.
- `tests/test_cartesian_projection.py`: projected-gradient optimization.
- `tests/test_qed_cis_dense.py`: QED-CIS dense/Davidson validation.
