# Examples

Examples are organized by method into dedicated subfolders. Canonical user-facing
examples use `CQEDConfig` and `CQEDCalculator`; lower-level diagnostic interfaces live in
`diagnostics/`; development scaffolding for upcoming features lives in `development/`.

| Subfolder | Content | Key files |
|---|---|---|
| `qed-rhf/` | CQED-RHF energy and gradient | examples/qed-rhf/cqed_rhf_energy.py; examples/qed-rhf/cqed_rhf_energy_gradient.py |
| `qed-rks/` | QED-DFT (RKS/PBE0/wB97X-D), projected gradients, H2 dimer example | examples/qed-rks/cqed_dft_energy_gradient.py; examples/qed-rks/h2_dimer_example.py; examples/qed-rks/h2_dimer_qed_dft_wb97x.ipynb |
| `qed-sapt0/` | QED-SAPT0 components: cavity-free comparison with native Psi4 SAPT0, He dimer (dense), benzene dimer (density-fitted); scans/ subfolder | examples/qed-sapt0/sapt0_water_methylamine_density_fitting_vs_native_psi4.py; examples/qed-sapt0/cqed_sapt0_he_dimer_full_eri.py; examples/qed-sapt0/cqed_sapt0_benzene_dimer_density_fitting.py |
| `qed-uhf/` | CQED-UHF open-shell energies with psi4 reference baselines | examples/qed-uhf/cqed_uhf_h2o_singlet.py; examples/qed-uhf/cqed_uhf_oh_doublet.py |
| `qed-uks/` | Placeholder — pending unreleased unrestricted engine (src/cqed_scf/uscf.py) | (see folder) |
| `qed-cis/` | Polaritonic excited states on CQED-HF/KS references | (see folder) |
| `optimization/` | BFGS, projected gradients, frequency analysis, nitrobenzene isomers | examples/optimization/cqed_dft_projected_gradient.py; examples/optimization/example_ZPE_meta.py; examples/optimization/example_ZPE_ortho.py |
| `molecular_dynamics_freeze_atoms/` | Velocity-Verlet MD with frozen atoms (water + nitrobenzene) | examples/molecular_dynamics_freeze_atoms/h2o_md_basic.py; examples/molecular_dynamics_freeze_atoms/md_water_freeze_oxygen.ipynb; examples/molecular_dynamics_freeze_atoms/nitrobenzene_frozen_br.py |

## QED-SAPT0 examples (`qed-sapt0/`)

All three scripts print a formatted component table and wall-clock timing, run
quietly (`quiet=True`, `debug=False`), and are run from the repository root
with an environment that has Psi4 and `cqed-scf` installed:

```bash
python examples/qed-sapt0/<script>.py
```

| Script | System | SCF / SAPT integrals | Cavity | Compares to Psi4? |
|---|---|---|---|---|
| `sapt0_water_methylamine_density_fitting_vs_native_psi4.py` | water ... methylamine, jun-cc-pVDZ | `scf_type: df` / `integral_backend="df"` | off (λ = 0) | yes |
| `cqed_sapt0_he_dimer_full_eri.py` | He ... He, 6-31G | `scf_type: pk` / `integral_backend="full_eri"` | on | no |
| `cqed_sapt0_benzene_dimer_density_fitting.py` | benzene ... benzene, jun-cc-pVDZ | `scf_type: df` / `integral_backend="df"` | on | no |

### Cavity-free comparison with native Psi4 SAPT0

`sapt0_water_methylamine_density_fitting_vs_native_psi4.py` checks that
QED-SAPT0 reduces to ordinary SAPT0 when the cavity coupling is zero. Both
codes use density fitting for the SCF, and Psi4's SAPT0 is density-fitted
internally, so cqed-scf is run with `density_fitting=True` and
`integral_backend="df"` (the cavity code path is still exercised with
`include_cavity_terms=True`).

It prints:

- a table of the six components from cqed-scf and Psi4 side by side, with the
  difference in µEh and in percent, and the sum of the six components;
- Psi4's total SAPT0 energy on its own line. It is *not* expected to equal the
  sum of the six components, because Psi4's total includes the delta-HF
  correction and the full (non-S²) exch10, while cqed-scf's exchange is the S²
  form and its total has no delta-HF term;
- the largest component difference, and a timing table for cqed-scf versus
  Psi4 with their ratio.

Sample output (4 Psi4 threads):

```
Component                cqed-scf (Eh)       Psi4 (Eh)   Diff (uEh)   Diff (%)
------------------------------------------------------------------------------
Electrostatics           -0.0086628887   -0.0086628887        0.000     0.0000
Exchange (S^2)            0.0031362133    0.0031362133       -0.000    -0.0000
Induction                -0.0018142655   -0.0018121968       -2.069    -0.1142
Exchange-induction        0.0008362630    0.0008356945        0.569     0.0680
Dispersion               -0.0013246014   -0.0013246014        0.000     0.0000
Exchange-dispersion       0.0001917810    0.0001917810        0.000     0.0000
------------------------------------------------------------------------------
Sum of six components    -0.0076374983   -0.0076359981       -1.500    -0.0196
Psi4 total SAPT0*                        -0.0080654267

cqed-scf QED-SAPT0 (DF):               1.58 s
Psi4 native SAPT0 (DF):                1.74 s
Ratio cqed-scf / Psi4:                 0.91
```

Electrostatics, exchange, dispersion and exchange-dispersion agree to about
1e-10 Eh. The induction and exchange-induction terms differ by about 2 and
0.6 µEh (roughly 0.1%), and the sum of the six components agrees to 1.5 µEh.
Timings are single-run, end-to-end wall-clock times; Psi4's thread count is set
with `psi4.set_num_threads`, while the NumPy/BLAS threads cqed-scf uses are
controlled separately (for example with `OMP_NUM_THREADS`).

### Helium dimer in a cavity (dense integrals)

`cqed_sapt0_he_dimer_full_eri.py` is the simplest cavity example. With two
atoms and a small basis, exact integrals (`scf_type: pk`,
`integral_backend="full_eri"`) are cheap, so no density fitting is used. It
prints the six components in Eh and kcal/mol with the total, and the wall time.

### Benzene dimer in a cavity (density fitting)

`cqed_sapt0_benzene_dimer_density_fitting.py` uses density fitting for both the
SCF (`scf_type: df`) and the SAPT components (`density_fitting=True`,
`integral_backend="df"`); these are independent switches and both should be on
for large systems. Each monomer is solved in the full dimer basis, and a dense
N⁴ two-electron tensor for a basis this size would need tens of gigabytes; the
script prints the basis-function count and the size of that tensor. Besides the
six components, it partitions the dispersion into standard `[c,c]`, cross
`[c,d]` and cavity `[d,d]` terms with their percentage shares. The cross term is
required: standard + cavity alone does not sum to the total dispersion.

### Monomer reference frame

The cavity examples pass `monomer_reference_frame="monomer_com"` to
`sapt0_components`. CQED orbital energies are not translation invariant, so
solving each monomer's reference with that monomer's own centre of mass at the
origin keeps the cavity-mediated dispersion independent of where the dimer sits
in space. This matters most for distance scans (see
[`qed-sapt0/scans/scan_readme.md`](qed-sapt0/scans/scan_readme.md)). It has no
effect in the λ = 0 comparison above.

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
