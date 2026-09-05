# Development notes about QED-SAPT Driver

1. class QEDSAPT0Driver constructure can take a geometry string for the full dimer with the '--' delimiter and handle it appropriately to get dimer and monomer quantities. 

2. Method prepare_geometries(self) will use the psi4 core method .extract_subsets(i,j) to extract monomer A and monomer B geometries using appropriate ghost atoms from the dimer geometry.  This method returns a psi4 geometry object, but we also capture a geometry string using the psi4.core method .create_psi4_strong_from_molecule.

3. Method prepare_monomers(self) will take the three strings (dimer, monomer A, monomer B) and run scf calculations and collect results.  It stores many useful non-mints quantities (ndocc, nvirt, C, Co, Cv, E_nuc, eps) for dimer and monomers for later use.  These quantities are exposed from the property methods of the monomer class, and they originate from scf calculations on the dimer and monomers.

4. Methods build_orbitals, build_sizes, and build_slices follow the organization of monomer A and B quantities into dictionaries from the helper_SAPT in psi4numpy

5. Method build_integrals is where the base integral quantities are / will be built (partially complete).  It calls build_orbitals, build_sizes, and build_slices before it starts building integral quantities.

6. Method run() should call prepare_geometries, prepare_monomers, build_integrals, and then evaluate SAPT terms

7. Methods .v() is implemented and tested for He dimer, methods .s(), .eps(), .potential(), and .vt() still require implementation and testing.  They should be called by .build_integrals or by .run()

8. After implementation from scratch, we will want to try to hook into as much psi4 capability as possible for a more performant code, i.e. avoiding full 2-ERI builds in favor of JK builds that use density fitting, etc.  This file https://github.com/psi4/psi4/blob/master/psi4/driver/procrouting/sapt/sapt_jk_terms.py looks to be quite useful, as the first function is build_sapt_jk_cach and takes wfn objects for monomer a and b. I have pinned a gemini chat about how to use the helper functions in this file!  The dispersion terms also require MP2-like evaluation, and the file https://github.com/psi4/psi4/blob/master/psi4/driver/procrouting/sapt/sapt_mp2_terms.py has a helpful pattern for this that we should hook into!

9. Added example calling sapt_jk_terms to compute first-order terms (E_{elest} and E_{exch}) in examples/he_dimer/he_dimer_sapt_jk_test.py, this approach seems promising!

## Water-methylamine QED-SAPT0 JK/DSE discrepancy notes

Reference confirmation:

- The hard-coded dense reference values in `examples/water_methylamine/water_methylamine_qed_sapt_jk.py` were independently reproduced by running the reference driver (which matches Kenny's adaptation of Konrad's notebook) in `examples/water_methylamine/water_methylamine_qed_sapt.py`, which uses `cqed_scf/sapt/qed_sapt0.py::QEDSAPT0Driver` with the same geometry, lambda vector `[0.0, 0.0, 0.1]`, omega `0.1`, `jun-cc-pVDZ`, and `include_cavity_terms=True`.
- The reproduced values are:
  - Electrostatics: `-0.0086737093`
  - Exchange: `0.0030805286`
  - Dispersion: `-0.0027151107`
  - Exchange-Dispersion: `0.0001966590`
  - Induction: `-0.0017887767`
  - Exchange-Induction: `0.0008167150`
  - Total: `-0.0090836941`

Cavity-modified monomer data flow - inspected by hand by Jay and confirmed by Codex:

- `cqed_scf/scf.py::CQEDSCF.run()` updates the returned Psi4 wavefunction with the final CQED-SCF state in `_update_wfn_with_cqed()`, including `Ca`, `Cb`, `Da`, `Db`, `epsilon_a`, and `epsilon_b`.
- `cqed_scf/sapt/monomer.py::SAPTMonomer.from_cqed_scf()` stores that same updated wavefunction in `scf_results["wfn"]`.
- `cqed_scf/sapt/qed_sapt_jk.py::build_sapt_jk_cache()` then reads orbitals and orbital energies from the wavefunction via `Ca_subset("AO", "OCC")`, `Ca_subset("AO", "VIR")`, `epsilon_a_subset("AO", "OCC")`, and `epsilon_a_subset("AO", "VIR")`.
- Spot checks showed zero numerical difference between the stored monomer coefficient/epsilon arrays and the wavefunction subsets. The JK path is therefore being fed the cavity-relaxed density, orbital coefficients, and orbital energies.

Exchange Convention and CPHF Sign Conventions raised by GPT 5.5. Jay confirmed Exchange convention, Kenny should implement and test.  No confirmation of CPHF sign convention yet, Kenny should implement and test. 

1. Exchange comparison key in `examples/water_methylamine/water_methylamine_qed_sapt_jk.py`
   - The reference `qed_sapt0.py::compute_Exch100()` that follows Kenny's / Kondrad's reference implements the S^2 exchange expression, 
   - `qed_sapt_jk.py::exchange()` returns both `Exch10(S^2)` and `Exch10`.
   - The water-methylamine JK comparison currently reports `exch["Exch10"]`, which is the S^inf-style value from the Psi4 JK helper path. That is the source of the apparent exchange discrepancy.

**TO DO FOR KENNY** For comparison against the current adapt `water_methylamine_qed_sapt_jk.py` to use `exch["Exch10(S^2)"]` instead of `exch["Exch10"]`.  Re-run comparisons and report back on the agreement between the exchange term.


2. Coupled induction response sign/convention in `cqed_scf/sapt/qed_sapt_jk.py` and `cqed_scf/sapt/dse_jk.py`
   - Direct/uncoupled induction agrees between dense and JK/DSE to numerical noise, so the induction RHS/electrostatic potential construction is OK.
   - The discrepancy appears in coupled response:
     - Dense total `Ind20,r`: `-0.0017887766852`
     - JK/DSE total `Ind20,r`: `-0.0017859880344`
   - `dse_jk.py::DSECPHF.hx_array()` matches the dense cavity Hessian action in the algebra/convention used by `qed_sapt0.py::chf()`.
   - Psi4 `wfn.cphf_Hx()` appears to use the opposite sign convention relative to the dense matrix solve. In `qed_sapt_jk.py::_sapt_cpscf_solve()`, the DSE response action is currently added with `xA.axpy(1.0, dse_cphf_A.hx_matrix(...))` and similarly for B.
   - This sign/convention needs to be reconciled. A local sign-flip experiment for the DSE Hx term largely corrected induction, which supports this as the right area to fix, but it should be implemented with a clear convention rather than an ad hoc patch.

3. Exchange-induction coupled-response mismatch
   - Uncoupled exchange-induction agrees between reference and JK/DSE:
     - Referemce `Exch-Ind20,u`: `0.0006174109081502`
     - JK/DSE `Exch-Ind20,u`: `0.0006174109081214`
   - Coupled exchange-induction differs:
     - Reference total `Exch-Ind20,r`: `0.0008167149799`
     - JK/DSE total `Exch-Ind20,r`: `0.0008137244180`
   - There is already a standard-only coupled exchange-induction difference of about `2.6e-6 Eh`, before adding DSE terms, so this is not purely a cavity/DSE issue.
   - Relevant reference code: `qed_sapt0.py::compute_Eexchind200()` and `qed_sapt0.py::chf()`.
   - Relevant JK code: `qed_sapt_jk.py::induction()`, especially the EX_A/EX_B potential construction and `_sapt_cpscf_solve()` response amplitudes.
   - Recommended next diagnostic: compare dense `compute_Eexchind200()` using the exact dense CPHF amplitudes against the JK EX_A/EX_B contractions using the same amplitudes. This will separate EX potential construction errors from response-amplitude convention errors.

## 2026-07-26 coherent-state DSE terms in SAPT-JK

### Problem summary

The production SAPT-JK path included the factorizable intermolecular DSE two-electron term through `DSEJK`, but the coherent-state fluctuation product also has one-electron and scalar pieces. Before this patch, the water-methylamine diagnostics applied those missing pieces by mutating the example-local cache after `build_sapt_jk_cache()`, so direct library callers could still build an incomplete intermolecular cavity operator.

Electrostatics depends directly on the one-electron potentials and scalar interaction term. Exchange and exchange-induction use the same dressed one-electron potentials inside overlap-dressed contractions. Uncoupled induction uses the electrostatic RHS matrices `w_B_MOA` and `w_A_MOB`, so it also requires the complete DSE operator.

### Operator convention

The dense reference constructs the intermolecular DSE contribution to `vt` as the coherent-state electronic fluctuation product

```text
d_A d_B - <d_A>_el d_B - <d_B>_el d_A + <d_A>_el <d_B>_el
```

where `d_A` and `d_B` are the dipole-projected AO DSE matrices for the two ghosted monomer bases, and `<d_A>_el`, `<d_B>_el` are the electronic coherent-state expectation values from the monomer CQED-SCF references. In the dense source this maps to:

- `I_dimer_cavity = d_A[:, :, None, None] * d_B[None, None, :, :]`.
- `V_A_cavity = -d_exp_el_A * d_B`, the potential generated by monomer A and represented in the shared dimer AO basis.
- `V_B_cavity = -d_exp_el_B * d_A`, the potential generated by monomer B and represented in the shared dimer AO basis.
- `vt_nuc_rep_cavity = d_exp_el_A * d_exp_el_B / (N_A N_B)`, which becomes the interaction-level scalar `d_exp_el_A * d_exp_el_B` in the JK electrostatics formula because that formula contracts with the closed-shell electron-count factors explicitly.

The JK implementation now keeps the ERI-like DSE term matrix-free in `DSEJK`, stores `V_A_standard`, `V_A_cavity`, `V_B_standard`, `V_B_cavity`, `dse_constant`, and adds `dse_constant` only to `nuclear_repulsion_energy`. The ordinary intermolecular nuclear repulsion remains separately available as `nuclear_repulsion_energy_standard`.

### Files changed

#### `src/cqed_scf/sapt/qed_sapt_jk.py`

- Lines 46-100, `_dse_cavity_terms`: added the coherent-state DSE one-electron potential and scalar helper, including the dense-reference monomer-label convention and shape validation.
- Lines 103-118, `build_sapt_jk_cache`: added optional DSE inputs for dipole matrices, electronic dipole expectations, cavity enablement, and the standard interaction nuclear repulsion.
- Lines 163-185, `build_sapt_jk_cache`: split standard and cavity one-electron potentials in the cache, then formed `V_A` and `V_B` from those pieces without mutating wavefunction-owned matrices.
- Lines 223-228, `build_sapt_jk_cache`: stored the standard nuclear interaction separately and added only the coherent-state scalar to the effective interaction energy used by component formulas.

#### `examples/water_methylamine/water_methylamine_qed_sapt_dense_vs_jk.py`

- Lines 72-79, module constants: added the uncoupled induction diagnostic row order.
- Lines 119-230, `_dense_components`, `_dense_uncoupled_components`, `_dense_uncoupled_exch_ind_components`: added dense-reference directional uncoupled induction and exchange-induction diagnostics.
- Lines 254-270, `_compute_jk_components`: switched from example-local cache mutation to passing the DSE one-electron/scalar inputs into `build_sapt_jk_cache()`, and disabled `DSEJK` when explicit cavity terms are disabled.
- Lines 276-290, `_compute_jk_components`: collected directional and total `Ind20,u` and `Exch-Ind20,u` values from the JK path.
- Lines 367-377, `_print_case`: printed the new uncoupled diagnostic table and retained the JK-only `Exch10` S^inf value.

#### `examples/water_methylamine/water_methylamine_qed_sapt_jk.py`

- Lines 123-136, `_compute_jk_components`: switched the companion example to pass DSE one-electron/scalar inputs into `build_sapt_jk_cache()` and disabled `DSEJK` when explicit cavity terms are disabled.

#### `tests/test_dse_jk_scaffold.py`

- Line 5, imports: imported `qed_sapt_jk` for the new helper regression tests.
- Lines 74-110, `test_dse_cavity_terms_match_dense_vt_convention` and `test_dse_cavity_terms_disable_cleanly_and_validate_shapes`: added focused tests for signs, monomer-label mapping, scalar value, disabled behavior, and shape validation.

#### `SAPT_NOTE.md`

- Lines 73-213, this section: documented the root cause, operator convention, files changed, commands, numerical diagnostics, remaining discrepancies, and reviewer guidance.

### Tests and diagnostics

Baseline command run before edits:

```bash
cd /Users/jfoley19/Code/cqed-scf
/Users/jfoley19/miniforge3/envs/p4dev/bin/python examples/water_methylamine/water_methylamine_qed_sapt_dense_vs_jk.py
```

Baseline dense-versus-JK values from the pre-patch diagnostic:

| Component | Dense / Eh | JK or mixed / Eh | Delta / Eh |
|---|---:|---:|---:|
| Electrostatics | -0.0086737093 | -0.0086737093 | -2.742251e-14 |
| Exchange, S^2 | 0.0030805286 | 0.0030805286 | 4.452647e-12 |
| Induction, coupled | -0.0017887767 | -0.0017889757 | -1.990136e-07 |
| Exchange-Induction, coupled | 0.0008167150 | 0.0008140487 | -2.666322e-06 |
| Total mixed QED-SAPT0 | -0.0090836941 | -0.0090865594 | -2.865331e-06 |

The baseline script did not print directional `Ind20,u` or `Exch-Ind20,u`; those rows were added in this patch.

Final command:

```bash
cd /Users/jfoley19/Code/cqed-scf
/Users/jfoley19/miniforge3/envs/p4dev/bin/python examples/water_methylamine/water_methylamine_qed_sapt_dense_vs_jk.py
```

Historical note: the following "Final dense-versus-JK" table was final for the
coherent-state one-electron/scalar DSE patch only. It predates the later
coupled-response fix documented in the next section, so the coupled
`remaining discrepancy` rows below are intentionally preserved as pre-fix
evidence rather than the current status.

Final dense-versus-JK values for the default cavity case before the
coupled-response matrix-free standard-action fix:

| Component | Dense / Eh | JK or mixed / Eh | Delta / Eh | Status |
|---|---:|---:|---:|---|
| Electrostatics | -0.0086737093 | -0.0086737093 | -2.742251e-14 | agrees |
| Exchange, S^2 | 0.0030805286 | 0.0030805286 | 4.452647e-12 | agrees |
| Ind20,u (A<-B) | -0.0003488641 | -0.0003488641 | -1.071550e-14 | agrees |
| Ind20,u (A->B) | -0.0010404588 | -0.0010404588 | 2.329994e-14 | agrees |
| Ind20,u total | -0.0013893229 | -0.0013893229 | 1.258433e-14 | agrees |
| Exch-Ind20,u (A<-B) | 0.0000500646 | 0.0000500646 | -2.778458e-15 | agrees |
| Exch-Ind20,u (A->B) | 0.0005673463 | 0.0005673463 | -2.595211e-14 | agrees |
| Exch-Ind20,u total | 0.0006174109 | 0.0006174109 | -2.873060e-14 | agrees |
| Induction, coupled | -0.0017887767 | -0.0017889757 | -1.990136e-07 | remaining discrepancy |
| Exchange-Induction, coupled | 0.0008167150 | 0.0008140487 | -2.666322e-06 | remaining discrepancy |
| Total mixed QED-SAPT0 | -0.0090836941 | -0.0090865594 | -2.865331e-06 | affected by coupled rows |

The final JK-only `Exch10` S^inf value was `0.0030886178 Eh`; the dense comparison uses `Exch10(S^2)` because that is the dense `compute_Exch100()` convention.

Zero-coupling command:

```bash
cd /Users/jfoley19/Code/cqed-scf
/Users/jfoley19/miniforge3/envs/p4dev/bin/python examples/water_methylamine/water_methylamine_qed_sapt_dense_vs_jk.py --lambda-vector 0 0 0
```

Zero-coupling results: electrostatics delta `-6.705747e-14 Eh`, exchange S^2 delta `4.492990e-12 Eh`, `Ind20,u` total delta `1.273352e-14 Eh`, `Exch-Ind20,u` total delta `-2.826179e-14 Eh`, coupled `Ind20,r` delta `1.500668e-13 Eh`, coupled `Exch-Ind20,r` delta `-6.971244e-13 Eh`, total delta `3.878874e-12 Eh`.

No-cavity command:

```bash
cd /Users/jfoley19/Code/cqed-scf
/Users/jfoley19/miniforge3/envs/p4dev/bin/python examples/water_methylamine/water_methylamine_qed_sapt_dense_vs_jk.py --no-cavity-terms
```

No-cavity results with nonzero CQED monomer references: electrostatics delta `2.919887e-14 Eh`, exchange S^2 delta `4.452644e-12 Eh`, `Ind20,u` total delta `1.265221e-14 Eh`, `Exch-Ind20,u` total delta `-2.899525e-14 Eh`, coupled `Ind20,r` delta `-6.610575e-08 Eh`, coupled `Exch-Ind20,r` delta `-2.644664e-06 Eh`.

Focused tests and companion diagnostic:

```bash
cd /Users/jfoley19/Code/cqed-scf
/Users/jfoley19/miniforge3/envs/p4dev/bin/python -m pytest tests/test_dse_jk_scaffold.py
/Users/jfoley19/miniforge3/envs/p4dev/bin/python -m pytest tests/test_qedsapt0_driver.py
/Users/jfoley19/miniforge3/envs/p4dev/bin/python examples/water_methylamine/water_methylamine_qed_sapt_jk.py
```

Results: `tests/test_dse_jk_scaffold.py` passed `10 passed in 2.23s`; `tests/test_qedsapt0_driver.py` passed `11 passed in 35.06s`; the companion JK example ran and reproduced electrostatics/exchange tightly while retaining the known coupled induction/exchange-induction discrepancies.

### Remaining discrepancies

The immediate uncoupled target is resolved: default-cavity `Ind20,u` and `Exch-Ind20,u` agree to about `1e-14 Eh`. The remaining default-cavity differences are coupled-response quantities: `Ind20,r` differs by `-1.990136e-07 Eh`, and `Exch-Ind20,r` differs by `-2.666322e-06 Eh`. Per the task boundary, `DSECPHF.hx_array()`, the CPHF Hessian sign, and the coupled response solver were not changed.

Dispersion and exchange-dispersion are still carried from the dense reference in the JK diagnostics; this patch does not implement JK dispersion.

### Human-review guidance

The most important algebraic choices to review are the monomer-label mapping in the one-electron DSE terms, the use of electronic rather than total dipole expectations in `V_A_cavity`, `V_B_cavity`, and `dse_constant`, the placement of the scalar only in the interaction-level nuclear term, and the decision to leave the matrix-free `DSEJK` two-electron path unchanged.

Also review that `DSEJK` is disabled when `include_cavity_terms=False`, and that the dense-vs-JK comparison uses `Exch10(S^2)` rather than the JK helper's S^inf-style `Exch10` for the dense exchange convention.

## 2026-07-26 coupled-response CPHF matrix-free standard action

### Problem summary

The remaining coupled induction and exchange-induction discrepancies were not
caused by the coherent-state DSE implementation. Diagnostic comparison of the
explicit dense response matrix against the matrix-free Hessian action showed
that the cavity contribution, `DSECPHF.hx_array()`, reproduced the dense DSE
Hessian to numerical roundoff. The remaining discrepancy originated entirely
from the ordinary, lambda-zero Psi4 `wfn.cphf_Hx()` action when used with
cavity-relaxed monomer references.

To isolate the source of the error, the response implementation was decomposed
into independent standard and cavity Hessian actions. The dense
occupied-virtual Hessian from `QEDSAPT0Driver.chf()` was compared against
matrix-free Hessian-vector products using deterministic test amplitudes. These
tests demonstrated:

- the DSE Hessian action is numerically identical to the dense cavity Hessian;
- the ordinary Psi4 `cphf_Hx()` action is not identical to the dense standard
  Hessian in the cavity-relaxed reference;
- therefore the coupled-response discrepancy was unrelated to the
  coherent-state fluctuation operator itself.

### Implementation

The SAPT-JK response solver now evaluates the standard electronic response
using an explicit matrix-free Coulomb/exchange Hessian constructed from the
cached occupied and virtual orbitals together with the orbital-energy
denominators. This implementation reproduces the dense standard Hessian action
while avoiding construction of the four-index response matrix.

The dense driver now also exposes diagnostic helpers for the occupied-virtual
RHS, explicit dense response matrix, and dense Hessian action. The JK induction
path accepts an optional diagnostics dictionary that exposes:

- `w_B_MOA` and `w_A_MOB`;
- uncoupled and coupled response amplitudes;
- `EX_A` and `EX_B`;
- ordinary Psi4 `cphf_Hx`, matrix-free standard, DSE-only, and combined
  Hessian actions;
- final absolute and relative CPHF residual norms.

### Sign convention

An important sign convention was clarified. The dense implementation solves

```text
A_dense t = -w
```

whereas `solvers.cg_solver` solves

```text
H x = rhs
```

Consequently, the matrix-free operator supplied to the iterative solver must
represent

```text
H = -A_dense_standard - A_dense_cavity
```

which is implemented internally as

```text
H = matrix_free_standard - dse
```

because the DSE Hessian routine returns the dense cavity Hessian action, i.e.
the opposite sign from the solver's combined operator convention.

### Operator diagnostics

Default finite-cavity water-methylamine operator comparisons showed:

| Direction | Matrix-free standard + dense standard | Psi4 `cphf_Hx` + dense standard | DSE - dense cavity | Total + dense total |
|---|---:|---:|---:|---:|
| A<-B max/Frobenius | `6.51e-11 / 1.06e-09` | `8.07e-02 / 5.91e-01` | `2.65e-16 / 1.03e-15` | `6.51e-11 / 1.06e-09` |
| A->B max/Frobenius | `5.10e-11 / 1.03e-09` | `9.21e-02 / 8.09e-01` | `7.44e-16 / 4.87e-15` | `5.10e-11 / 1.03e-09` |

Direct reconstructed-matrix solves agree with the dense amplitudes, and the
Psi4 CG amplitudes agree with the reconstructed matrix solutions. Final
relative residual norms were approximately `4.8e-9` for both response
directions.

### Final resolved numerical status

With this change, the coupled-response implementation reproduces the dense
reference to numerical precision.

Default water-methylamine finite-cavity differences are:

| Component | Delta / Eh |
|---|---:|
| `Ind20,r` | `4.8e-14` |
| `Exch-Ind20,r` | `-1.14e-12` |
| Total SAPT difference | `3.34e-12` |

Directional coupled-response differences are:

| Component | Delta / Eh |
|---|---:|
| `Ind20,r (A<-B)` | `4.87e-14` |
| `Ind20,r (A->B)` | `-6.96e-16` |
| `Exch-Ind20,r (A<-B)` | `-7.13e-14` |
| `Exch-Ind20,r (A->B)` | `-1.06e-12` |

The zero-coupling calculation remains at roundoff agreement, and the
nonzero-reference/no-explicit-cavity stress case is also resolved:
`Ind20,r = 1.52e-13 Eh`, `Exch-Ind20,r = -6.65e-13 Eh`, and total
`= 3.97e-12 Eh`.

These residuals are consistent with floating-point roundoff and confirm that
the dense and matrix-free implementations are mathematically equivalent for
both the standard electronic response and the coherent-state DSE response.

### Commands and tests

Commands run after the fix:

```bash
cd /Users/jfoley19/Code/cqed-scf
/Users/jfoley19/miniforge3/envs/p4dev/bin/python examples/water_methylamine/water_methylamine_qed_sapt_dense_vs_jk.py
/Users/jfoley19/miniforge3/envs/p4dev/bin/python examples/water_methylamine/water_methylamine_qed_sapt_dense_vs_jk.py --lambda-vector 0 0 0
/Users/jfoley19/miniforge3/envs/p4dev/bin/python examples/water_methylamine/water_methylamine_qed_sapt_dense_vs_jk.py --no-cavity-terms
/Users/jfoley19/miniforge3/envs/p4dev/bin/python -m pytest tests/test_dse_jk_scaffold.py
/Users/jfoley19/miniforge3/envs/p4dev/bin/python -m pytest tests/test_qedsapt0_driver.py
/Users/jfoley19/miniforge3/envs/p4dev/bin/python examples/water_methylamine/water_methylamine_qed_sapt_jk.py
```

Results:

- `tests/test_dse_jk_scaffold.py`: `10 passed in 2.01s`;
- `tests/test_qedsapt0_driver.py`: `11 passed in 35.14s`;
- companion JK example total delta: `-7.385354e-11 Eh`.

## 2026-09-05 notes from the QED-CIS response work (Tiers 0-3, 5)

Written at the end of the response-theory session, for whoever picks up the SAPT
driver next. Nothing here changes SAPT behaviour except item 1, which is a
confirmed defect that was fixed and tested.

### 1. Confirmed defect: `canonical_orbital_energies` was aliased (FIXED)

`np.asarray` on a Psi4 `Vector`/`Matrix` returns a **view** of Psi4's buffer, not
a copy. Verified directly:

```python
v = psi4.core.Vector(3); a = np.asarray(v)
v.nph[0][:] = [1., 2., 3.]
print(a)                              # -> [1. 2. 3.]
print(np.shares_memory(a, v.nph[0]))  # -> True
```

`CQEDSCF.run()` captured `eps_canonical = np.asarray(self.wfn.epsilon_a())` early
and then `_update_wfn_with_cqed()` wrote the **CQED** orbital energies into that
same buffer at the end. So `results["canonical_orbital_energies"]` had been
silently identical to `results["orbital_energies"]`.

**SAPT consequence.** `qed_sapt0.py::compute_Edisp200(canonical_denom=True)`
selects `eps_canonical` for the dispersion denominator. Both branches of that
switch were dividing by the same array, so the two settings returned bitwise
identical energies. This is exactly the null result observed when comparing
canonical vs CQED-HF orbital energies in the denominator — the experiment was
uninformative, not wrong.

**Why the suite did not catch it.** The only test exercising
`canonical_denom=True` ran at `lambda_vector = [0, 0, 0]`, where CQED-SCF reduces
to RHF and the two orbital-energy sets *genuinely* coincide. The branch was
covered; the condition that makes the branch matter was not. Worth generalising:
covering a branch only in the regime where it is degenerate with its alternative
is not coverage.

Fixed in `scf.py` (explicit copies for `epsilon_a()` and `Ca()`), with four
regression tests added to `tests/test_qedsapt0_driver.py`:

- `test_canonical_and_cqed_orbital_energies_are_independent_arrays` — asserts
  `not np.shares_memory(...)`, i.e. catches the *mechanism*, not a symptom;
- `test_canonical_orbital_energies_are_the_cavity_free_ones` — anchors
  `eps_canonical(lambda != 0) == eps(lambda = 0)`, so a wrong-but-different array
  cannot pass;
- `test_edisp200_denominator_choice_matters_at_finite_coupling` plus its
  `lambda = 0` control.

Full audit of the remaining `np.asarray` calls on Psi4 objects is in
`docs/development/psi4_array_aliasing.md`. Only the two `scf.py` sites had the
dangerous pattern. In `qed_sapt0.py`, `S_dimer`, `V_A`, `V_B` and
`I_dimer_standard` are views but nothing writes back to them, and
`I_dimer_standard` is a view **by design** (copying the full ERI tensor would be
costly; line 279 correctly does `I_dimer = I_dimer_standard.copy()` before adding
the cavity term).

### 2. Open physics question: which denominator is right

Now that the switch is live, the canonical-vs-CQED comparison can actually be
run. My reading is that the **CQED orbital energies are the consistent
production choice** — the perturbation series is defined relative to the CQED
monomer Hamiltonians, and the numerator already carries cavity-dressed
integrals, so canonical denominators mix two reference definitions.
`canonical_denom=True` reads more like a diagnostic than a setting. Not my call
to make.

### 3. The long-range dispersion plateau is structural, not a bug

`qed_sapt0.py` line 287:

```python
I_dimer_cavity = self.d_A[:, :, None, None] * self.d_B[None, None, :, :]
```

A bare outer product — no Coulomb kernel, therefore **no R-dependence at any
separation**. Through `v('abrs')` this becomes `d^A_{ar} d^B_{bs}` in the
`Disp200` numerator, and since the denominator also tends to a constant at large
R, the cavity part of the dispersion tends to a finite non-zero constant rather
than decaying. That is the single-mode long-wavelength approximation behaving as
specified: both monomers couple to the same mode with the same `lambda`, so the
cavity mediates an interaction with no distance decay.

**A sharp, cheap test if this needs settling.** The `R -> inf` limit of the
cavity part of `Disp200` is computable in closed form from *isolated monomer*
transition dipoles and excitation energies — no dimer calculation. If the
computed curve plateaus at that predicted value, the long-range behaviour is the
model's real content rather than a numerical artifact. Much stronger evidence
than "the curve looks flat".

**One convention note.** In the coherent-state basis the DSE is
`(lambda . (mu - <mu>))^2 / 2`, so the A-B cross term is
`(lambda . dmu_A)(lambda . dmu_B)` with coefficient 1 (the 1/2 cancels against
the cross term appearing twice) — which matches the code. For *dispersion*
specifically the `<mu>` shift is harmless: subtracting a constant times the
identity changes only diagonal elements, so the off-diagonal transition dipoles
`d^A_{ar}` are untouched. It does affect the first-order and induction terms.

### 4. `sapt/dse_jk.py` should be promoted to a shared module

There are now two consumers of the same physics. The unifying statement, worth
putting in the shared module's docstring:

> **The two-electron DSE is an ERI with a separable kernel:**
> `(pq|rs) -> d_pq d_rs`. Every Slater-Condon result carries over verbatim under
> that substitution.

That is why `DSEJK` is exactly the J/K of that kernel
(`J = d Tr(dD)`, `K = d D d^T`), and why the response module's DSE block is
exactly `2(ia|jb) - (ij|ab)` under the same substitution, giving
`2 d_ia d_jb - d_ij d_ab`. Response evaluates it directly in the MO blocks
(cheaper: no AO transform, and no J/K build at all), while SAPT needs the AO
form — but they are the same operator and the derivation should live in one
place.

### 5. The package is internally consistent on treating the DSE as two-electron

Worth recording as a deliberate position, because the literature is not uniform.
Yang *et al.*, JCP **155**, 064107 (2021) evaluate the DSE as the square of a
one-body expectation over a single determinant, giving a **rank-one, mean-field**
`Delta'_ai,bj = sum_alpha lambda_ai lambda_bj` with no exchange-like counterpart.
This package treats the DSE as a genuine two-electron operator throughout — SCF,
SAPT and response alike — which is where the exchange-like terms
(`-N[D]` in the Fock, `K_DSE` in `DSEJK`, `-d_ij d_ab` in the response block)
come from. The two are different levels of theory, not a rescaling of one
another. See `docs/qed_cis_formalism.tex`, "Relationship to linear-response
QED-TDDFT".

### 6. Reusable infrastructure now available from the response work

- **`src/cqed_scf/davidson.py`** — a general real-symmetric block Davidson-Liu
  solver with **no cavity physics in it**: batched matvec interface, diagonal
  preconditioner, thick restart, double Gram-Schmidt with a linear-dependence
  drop. Directly usable for iterative CPHF/CPKS or any dispersion solver that
  wants to stop forming matrices. Two non-obvious details it already gets right:
  preconditioned corrections must be **normalised before** the independence test
  (otherwise residuals plateau ~1e-6 while eigenvalues look converged), and
  restarts must be **thick** (collapsing to exactly `nroots` thrashes).
- **Generalized-JK builder pattern** — `response.py::QEDCIS._build_generalized_jk`
  plus `JKERIEngine` show how to queue independent `C_left`/`C_right` for the
  one-sided response density `D = C_o X C_v^T`, and how to batch many densities
  into a single `compute()`. `scf.py::_build_JK` is `C_left`-only and cannot do
  this; SAPT's CPHF work may want the same pattern.
- **`Psi4HxERIEngine`** — routes the two-electron action through
  `wfn.twoel_Hx_full`, which is how a Kohn-Sham XC kernel gets included without
  reconstructing factors.

### 7. Psi4 `*_Hx` routines each have their own sign and combination convention

A trap that cost time and would cost it again. Psi4's
`scf_products.py::_combine_A` has the docstring
`A X = [(Ea - Ei) + 2J - K] X` and returns **`-A X`**; its caller negates the
result afterwards. Transcribing the method verbatim gives a global sign error.

SAPT already uses `wfn.cphf_Hx`, which is a *different* routine with a
*different* combination (`4J - K - K^T`, per
`docs/development/dse_jk_cphf_design.md`). The lesson generalises: **pin every
Psi4 `Hx`-family call against an independently constructed reference before
trusting it**, and read the caller, not just the callee. A function's docstring
describes its meaning, not necessarily its return value.

### 8. Testing patterns worth reusing

Three that earned their keep this session, all applicable to SAPT:

1. **The column test.** Apply a matrix-free action to every unit vector,
   assemble the implied matrix, and compare *elementwise* against an explicit
   build. This validates every sign, factor and index at once and is far more
   diagnostic than comparing converged energies. It found nothing only because
   it was run first.
2. **Pin conventions with non-symmetric probes.** A symmetric test matrix cannot
   distinguish `M` from `M^T`, so a transpose-convention error passes and
   surfaces much later as an asymmetric operator.
3. **Assert that the naive approach fails.** Where a helper exists because an
   obvious heuristic is wrong, a test that the heuristic *does* get it wrong
   documents why the helper exists — and turns into a signal if the situation
   ever changes.

### 9. SCF contract changes that SAPT gets for free

From Tier 0, all in `scf.py`:

- Orbital canonicalization fix: the loop previously broke *before*
  re-diagonalizing, so `results["orbital_energies"]` were eigenvalues of the
  previous DIIS-extrapolated Fock rather than of the returned `results["F"]`.
  They are now consistent, which matters anywhere `eps` and `F` are used
  together.
- Results dict gained `omega`, `lambda_vector`, `fock_mo`, and `max_fock_ov`
  (the canonicality diagnostic, `max|F_ia|`, which must vanish for the CQED
  Brillouin condition to hold).
- `HARTREE_TO_EV` added to `utils.py`.
