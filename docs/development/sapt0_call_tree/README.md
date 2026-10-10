# QED-SAPT0 call tree: dense (`full_eri`) vs density-fitted (`df`)

Entry point: `calc.sapt0_components(mol, integral_backend=..., include_cavity_terms=True,
monomer_reference_frame="monomer_com")`, as called by `run_worker()` in
`sapt_demonstration_data/Result_3_Computational_Performance/water_methylamine_benchmark_resources.py`.

These trees come from **runtime tracing** (`sys.setprofile`) of that exact call for
water–methylamine/cc-pVDZ, RHF, λ = (0, 0, 0.05). They are not a static guess. The
`(psi4)` leaves show where control passes into Psi4. Line numbers are function `def` lines
at commit `423429b`.

* `trace_full_eri_cc-pVDZ.txt` and `trace_df_cc-pVDZ.txt`: complete traces, including config,
  output, and psi4 boundary calls, with call counts.
* `trace_sapt.py <df|full_eri> <basis> <out.json>` then `render.py <out.json>`: regenerate
  the traces (use the `p4dev` env).

Below is a condensed tree. Config validation, printing, and trivial accessors are pruned.
`[D]` means dense only and `[DF]` means density-fitted only. Lines without a tag run on both paths.

```
CQEDCalculator.sapt0_components                         calculator.py:516
├─ CQEDCalculator.sapt0                                 calculator.py:497
│  └─ QEDSAPT0Driver.__post_init__                      sapt/qed_sapt0.py:117
├─ output.quiet_context / psi4_silent                   output.py:52 / :174
└─ QEDSAPT0Driver.run_components                        sapt/qed_sapt0.py:1646
   ├─ QEDSAPT0Driver.run                                sapt/qed_sapt0.py:1652
   │  │
   │  ├─ prepare_monomers                               sapt/qed_sapt0.py:295
   │  │  ├─ prepare_geometries                          sapt/qed_sapt0.py:196
   │  │  │  ├─ _populate_ghosted_molecules              sapt/qed_sapt0.py:182   (psi4 Molecule.extract_subsets)
   │  │  │  └─ _reference_frame_string ×2               sapt/qed_sapt0.py:207
   │  │  │     └─ _real_atom_center_of_mass             sapt/qed_sapt0.py:233
   │  │  ├─ _populate_dimer_nuclear_terms               sapt/qed_sapt0.py:331
   │  │  ├─ SAPTMonomer.from_cqed_scf ×2  (A, B)        sapt/monomer.py:118
   │  │  │  ├─ CQEDConfig.copy_with                     references.py:279
   │  │  │  ├─ CQEDSCF.__init__                         scf.py:69
   │  │  │  ├─ CQEDSCF.run                              scf.py:127
   │  │  │  │  ├─ _prepare_options                      scf.py:489   [DF] forces scf_type=df ; [D] scf_type=pk
   │  │  │  │  ├─ psi4.energy  (guess / wfn)            (psi4)       PK vs DF integrals
   │  │  │  │  ├─ MintsHelper.ao_kinetic/potential/overlap/dipole/quadrupole   (psi4)
   │  │  │  │  ├─ _build_jk                             scf.py:507   psi4.core.JK.build → PK JK [D] or DF JK [DF]
   │  │  │  │  ├─ DIISSubspace.__init__                 scf.py:16
   │  │  │  │  ├─ SCF loop ×~11 per monomer:
   │  │  │  │  │  ├─ _build_JK                          scf.py:541   (psi4 JK.compute)
   │  │  │  │  │  ├─ DIISSubspace.add / extrapolate     scf.py:21 / :29
   │  │  │  │  │  └─ _canonicalize                      scf.py:578
   │  │  │  │  └─ _update_wfn_with_cqed                 scf.py:607
   │  │  │  └─ SAPTMonomer.from_scf_results             sapt/monomer.py:93
   │  │  ├─ _check_monomer_reference_frame ×2           sapt/qed_sapt0.py:241
   │  │  └─ _populate_monomer_attributes                sapt/qed_sapt0.py:344
   │  │     ├─ SAPTMonomer.{C,Co,Cv,eps,d_ao,...}       sapt/monomer.py:29–89
   │  │     └─ _rebase_dipole_data_to_dimer_frame       sapt/qed_sapt0.py:439
   │  │        └─ _dimer_frame_mints → _dimer_frame_basisset   sapt/qed_sapt0.py:263 / :281
   │  │
   │  ├─ build_integrals                                sapt/qed_sapt0.py:512
   │  │  ├─ build_orbitals / build_slices / build_sizes sapt/qed_sapt0.py:480 / :492 / :501
   │  │  ├─ _dimer_frame_mints ×3, MintsHelper.ao_overlap → S_AB
   │  │  ├─ [D]  _build_dense_eri_tensors               sapt/qed_sapt0.py:642
   │  │  │        ├─ MintsHelper.ao_eri()  (nbf^4)      (psi4)
   │  │  │        ├─ I_cavity = d_A ⊗ d_B  (nbf^4), I_dimer = I_std + I_cav  (nbf^4)
   │  │  │        └─ _read_only ×3                      sapt/qed_sapt0.py:66
   │  │  ├─ [DF] _resolve_df_aux_basis                  sapt/qed_sapt0.py:595
   │  │  ├─ [DF] _pauli_fierz_df("scf")  (JKFIT)        sapt/qed_sapt0.py:609
   │  │  │        └─ PauliFierzDF.from_driver           sapt/dse_df.py:145
   │  │  │           ├─ _dimer_frame_basisset / _dimer_frame_molecule, psi4 BasisSet.build (aux)
   │  │  │           ├─ build_df_ao_tensor             sapt/dse_df.py:59
   │  │  │           │  ├─ MintsHelper.ao_eri(aux,0,p,p) → (P|pq);  ao_eri(aux,0,aux,0) → metric
   │  │  │           │  └─ Matrix.power(-1/2);  contract PQ,Qpq→Ppq
   │  │  │           └─ PauliFierzDF.__init__          sapt/dse_df.py:104   (append dipole row; half-transform to MO union space)
   │  │  │              └─ _validate_ao_tensor         sapt/dse_df.py:237
   │  │  └─ MintsHelper.ao_potential ×2 → V_A, V_B, cavity-shifted, MO-transformed
   │  │
   │  ├─ compute_Elst100                                sapt/qed_sapt0.py:1277   vt ×1
   │  ├─ compute_Exch100                                sapt/qed_sapt0.py:1280   vt ×4, s
   │  ├─ compute_Edisp200                               sapt/qed_sapt0.py:1388
   │  │  ├─ _dispersion_numerator → v('abrs', df_role="corr")    sapt/qed_sapt0.py:1317
   │  │  │     [DF] first use builds a 2nd tensor: _pauli_fierz_df("corr") (RIFIT) → from_driver → build_df_ao_tensor
   │  │  └─ _store_dispersion_amplitudes → _dispersion_denominator → eps   :1329 / :1303 / :839
   │  ├─ compute_Eexchdisp200                           sapt/qed_sapt0.py:1447   _require_dispersion_amplitudes, vt ×9, s ×32
   │  ├─ compute_Eind200                                sapt/qed_sapt0.py:1510
   │  │  └─ chf ×2  (A, B)                              sapt/qed_sapt0.py:1129
   │  │     ├─ v('saba' | 'rbab')                       (shared frame)
   │  │     ├─ v(voov, frame=X), v(vovo, frame=X)       → v(shared) − _cavity_v(None) + _cavity_v(X)
   │  │     │     └─ _cavity_v → _cavity_mo_pair        sapt/qed_sapt0.py:815 / :802
   │  │     └─ np.linalg.solve  (dense ov×ov CPHF)
   │  ├─ compute_Eexchind200                            sapt/qed_sapt0.py:1516   vt ×12, s ×36
   │  └─ qed_sapt_jk.print_sapt_summary                 sapt/qed_sapt_jk.py:41  (runs; output suppressed by quiet)
   │
   └─ _build_results                                    sapt/qed_sapt0.py:1605
      └─ dispersion_energy_partition                    sapt/qed_sapt0.py:1404
         ├─ _dispersion_numerator("standard"), ("cavity")   → 2 more v('abrs') builds
         ├─ _dispersion_denominator
         └─ _energy ×3                                  sapt/qed_sapt0.py:1425
```

## The shared integral kernel, where the two backends differ

All SAPT component code calls the same accessors. The backend switch happens inside them:

```
QEDSAPT0Driver.vt(string)                               sapt/qed_sapt0.py:930
└─ vt_parts                                             sapt/qed_sapt0.py:894
   ├─ v(string, context, df_role, frame)                sapt/qed_sapt0.py:747
   │  ├─ [D]  _eri_for_context → _validate_operator_context   :693 / :687
   │  │        4 × oe.contract("pA,pqrs->Aqrs", ...)    O(nbf^5) quarter transforms of I_dimer
   │  └─ [DF] _pauli_fierz_df(df_role) (cached)         :609
   │           └─ PauliFierzDF.v                        sapt/dse_df.py:222
   │              ├─ PauliFierzDF.b ×2 → _aux_slice     sapt/dse_df.py:201 / :189   (views into one B_mo tensor)
   │              └─ oe.contract("PAC,PBD->ABCD")
   ├─ s ×2                                              sapt/qed_sapt0.py:823
   ├─ potential ×2 → _potential_for_context             sapt/qed_sapt0.py:875 / :710
   └─ _vt_nuc_rep_for_context                           sapt/qed_sapt0.py:728
└─ _sum_vt_parts                                        sapt/qed_sapt0.py:740
```

## Backend differences at a glance

| Stage | Dense `full_eri` | Density-fitted `df` |
|---|---|---|
| Monomer SCF integrals (`CQEDSCF._prepare_options`, `_build_jk`) | `scf_type="pk"` (the script's `--dense-scf-type`) | `scf_type="df"` forced via `density_fitting=True` |
| `build_integrals` | `_build_dense_eri_tensors`: `ao_eri()` plus two more nbf⁴ arrays (cavity, total) | `_pauli_fierz_df("scf")` → `PauliFierzDF.from_driver` → `build_df_ao_tensor` (JKFIT) |
| Dispersion fitting basis | n/a | Lazily builds a 2nd `PauliFierzDF` with role `"corr"` (RIFIT) on the first `_dispersion_numerator` call |
| `v()` | `_eri_for_context` + 4 quarter transforms | `PauliFierzDF.v` → `b()` ×2 + one contraction |
| Four-index `v()` builds per run | 35 (26 via `vt`, 6 in `chf`, 1 in `Edisp200`, 2 in `dispersion_energy_partition`) | same |
| Unique files on path | `calculator.py`, `references.py`, `scf.py`, `output.py`, `sapt/monomer.py`, `sapt/qed_sapt0.py`, `sapt/qed_sapt_jk.py` | same + `sapt/dse_df.py` |

`sapt/dse_jk.py` (`PauliFierzJK`) and most of `sapt/qed_sapt_jk.py` are **not** used by
`sapt0_components`. Only `print_sapt_summary` is imported from the latter.

## Observations relevant to the benchmark timings

1. **`_build_results` → `dispersion_energy_partition` falls inside the timed region.** It is a
   diagnostic, but it rebuilds `v('abrs')` twice more (standard and cavity contexts). It runs
   three extra o²v² denominator contractions. For dense, each rebuild is a full
   O(nbf⁵) transform. For DF, the cavity rebuild is cheap (a one-row aux slice).
2. **DF builds two 3-index tensors** (JKFIT for first-order and induction terms, RIFIT for
   dispersion). Each tensor calls `ao_eri` twice and does a metric `power(-1/2)`.
3. The **monomer SCFs differ too** (PK vs DF JK). So the dense-vs-DF wall-time ratio
   includes the SCF stage as well as the SAPT integral backend. `--dense-scf-type df` isolates
   the SAPT backend, as the script's help text notes.
4. With `monomer_reference_frame="monomer_com"`, `chf` uses the `frame=` path of `v()`.
   Each Hessian block costs one shared `v()` plus two cheap `_cavity_v` rank-one terms.
5. `print_sapt_summary` and the `output.*` helpers execute even with `quiet=True`. The cost
   is negligible.
