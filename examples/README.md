# Examples

Examples are organized by method into dedicated subfolders. Canonical user-facing
examples use `CQEDConfig` and `CQEDCalculator`; lower-level diagnostic interfaces live in
`diagnostics/`; development scaffolding for upcoming features lives in `development/`.

| Subfolder | Content | Key files |
|---|---|---|
| `qed-rhf/` | CQED-RHF energy and gradient | examples/qed-rhf/cqed_rhf_energy.py; examples/qed-rhf/cqed_rhf_energy_gradient.py |
| `qed-rks/` | QED-DFT (RKS/PBE0/wB97X-D), projected gradients, H2 dimer example | examples/qed-rks/cqed_dft_energy_gradient.py; examples/qed-rks/h2_dimer_example.py; examples/qed-rks/h2_dimer_qed_dft_wb97x.ipynb |
| `qed-sapt0/` | Total energy decomposition + dispersion partitioning; scans/ subfolder | examples/qed-sapt0/conventional_sapt0_control.py; examples/qed-sapt0/cqed_sapt0_components.py; examples/qed-sapt0/cqed_sapt0_dispersion_partition.py |
| `qed-uhf/` | CQED-UHF open-shell energies with psi4 reference baselines | examples/qed-uhf/cqed_uhf_h2o_singlet.py; examples/qed-uhf/cqed_uhf_oh_doublet.py |
| `qed-uks/` | Placeholder — pending unreleased unrestricted engine (src/cqed_scf/uscf.py) | (see folder) |
| `qed-cis/` | Polaritonic excited states on CQED-HF/KS references | (see folder) |
| `optimization/` | BFGS, projected gradients, frequency analysis, nitrobenzene isomers | examples/optimization/cqed_dft_projected_gradient.py; examples/optimization/example_ZPE_meta.py; examples/optimization/example_ZPE_ortho.py |
| `molecular_dynamics_freeze_atoms/` | Velocity-Verlet MD with frozen atoms (water + nitrobenzene) | examples/molecular_dynamics_freeze_atoms/h2o_md_basic.py; examples/molecular_dynamics_freeze_atoms/md_water_freeze_oxygen.ipynb; examples/molecular_dynamics_freeze_atoms/nitrobenzene_frozen_br.py |

## Diagnostic and Development Examples

`diagnostics/` contains lower-level scripts that intentionally use development interfaces such as
`QEDSAPT0Driver`, SAPT JK helpers, or direct component/intermediate access. These should not be
rewritten to hide the interface they demonstrate.

Diagnostic scripts and data:
- `examples/diagnostics/benzene_water_dft_interaction_scan.py`
- `examples/diagnostics/benzene_water_dft_serial_hardcoded.py`
- `examples/diagnostics/benzene_water_future_sapt0_template.py`
- `examples/diagnostics/finite_difference_gradient_validation.py`
- `examples/diagnostics/geometries.json`
- `examples/diagnostics/h2_dimer_fci_reference_notebook.ipynb`
- `examples/diagnostics/h2_dimer_wb97xd_lz0_control.py`
- `examples/diagnostics/h2o_dft_lz0_regression.py`
- `examples/diagnostics/h2o_freq_post_projection.py`
- `examples/diagnostics/h2o_pbe_lz0_regression.py`
- `examples/diagnostics/h2o_scf_lz0_regression.py`
- `examples/diagnostics/h2o_wb97xd_lz0_regression.py`
- `examples/diagnostics/he_dimer_sapt_example.py`
- `examples/diagnostics/he_dimer_sapt_jk_test.py`
- `examples/diagnostics/nitrobenzene_cqed_rhf.py`
- `examples/diagnostics/nitrobenzene_dft_lz_scan.ipynb`
- `examples/diagnostics/nitrobenzene_dft_optimization.ipynb`
- `examples/diagnostics/nitrobenzene_direction_A_wb97x_4_ts.xyz`
- `examples/diagnostics/nitrobenzene_direction_A_wb97x_d_4_ts.xyz`
- `examples/diagnostics/nitrobenzene_wb97x.py`
- `examples/diagnostics/nitrobenzene_wb97x_d.py`
- `examples/diagnostics/nitrobenzene_wb97xd_lz0_regression.py`
- `examples/diagnostics/reference_h2_dimer_method_comparison.py`
- `examples/diagnostics/water_methylamine_qed_sapt_dense_vs_jk.py`
- `examples/diagnostics/water_methylamine_qed_sapt_jk.py`
- `examples/diagnostics/water_water_qed_sapt.py`
- `examples/diagnostics/water_water_qeddft_scan.ipynb`
- `examples/diagnostics/water_water_sapt.py`

`development/unrestricted_reference/` contains scaffolding for the forthcoming unrestricted engine:
- `examples/development/unrestricted_reference/README.md`
- `examples/development/unrestricted_reference/hilbert_reference_energies.json`
- `examples/development/unrestricted_reference/make_reference_table.py`
- `examples/development/unrestricted_reference/oh_doublet_psi4.py`
- `examples/development/unrestricted_reference/psi4_reference_energies.json`
- `examples/development/unrestricted_reference/run_hilbert_qed_uhf_uks.py`
- `examples/development/unrestricted_reference/run_psi4_uhf_uks.py`
- `examples/development/unrestricted_reference/systems.py`
- `examples/development/unrestricted_reference/validate_protocol_restricted.py`

## Parameter Ownership

- Put standard Psi4 electronic-structure settings in `psi4_options`: basis set, SCF type,
  convergence thresholds, and DFT grid settings.
- Put package-level calculation settings in `CQEDConfig`: cavity coupling vector, photon
  frequency, reference, functional, density-fitting flag, charge, multiplicity, dispersion
  policy, and debug flag.
- Include charge and multiplicity in geometry strings too. This keeps examples unambiguous when
  they are run directly with Psi4.
