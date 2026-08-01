# Examples

The canonical user-facing examples live in `examples/canonical/`. They use the
current public interface:

1. Build a `CQEDConfig`.
2. Build a `CQEDCalculator(config=config)`.
3. Call a high-level calculator method.

Use this style for new examples unless the example is intentionally testing or
demonstrating a lower-level development interface.

## Canonical User Examples

- `canonical/cqed_rhf_energy.py`: CQED-RHF single-point energy.
- `canonical/cqed_rhf_energy_gradient.py`: CQED-RHF energy and nuclear gradient.
- `canonical/cqed_dft_energy_gradient.py`: CQED-DFT energy and nuclear gradient.
- `canonical/cqed_dft_projected_gradient.py`: CQED-DFT energy and projected gradient.
- `canonical/cqed_sapt0_components.py`: QED-SAPT0 total energy and components.

## Parameter Ownership

- Put standard Psi4 electronic-structure settings in `psi4_options`: basis set,
  SCF type, convergence thresholds, and DFT grid settings.
- Put package-level calculation settings in `CQEDConfig`: cavity coupling
  vector, photon frequency, reference, functional, density-fitting flag, charge,
  multiplicity, dispersion policy, and debug flag.
- Include charge and multiplicity in geometry strings too. This keeps examples
  unambiguous when they are run directly with Psi4.

## Diagnostic and Development Examples

Some examples intentionally use lower-level objects such as `QEDSAPT0Driver`,
SAPT JK helpers, or direct component/intermediate access. Those are development
or diagnostic examples and should not be rewritten to hide the interface they
are demonstrating.

In particular, these SAPT JK examples intentionally preserve their lower-level
interface:

- `water_methylamine/water_methylamine_qed_sapt_dense_vs_jk.py`
- `water_methylamine/water_methylamine_qed_sapt_jk.py`
