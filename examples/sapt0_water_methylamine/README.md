# Standard SAPT0 scan: water–methylamine adaptive curve

A portable psi4 SAPT0 scan over the 27 geometries of the completed `WaterMeNH2`
adaptive curve scan.

`run_sapt0_scan.py` needs only **psi4** and the Python standard library. It does
not import `qcage` or `cqed-scf` and does not need the S66x8 dataset — the
geometries live in `watermenh2_adaptive_scan.xyz`. Copy those two files anywhere
psi4 is installed and run.

```bash
python run_sapt0_scan.py --dry-run                     # inspect geometries; no psi4 needed
python run_sapt0_scan.py --basis 6-31G --scales 1.00   # quick smoke test
python run_sapt0_scan.py --threads 8 --memory "16 GB"  # full 27-point scan
python run_sapt0_scan.py --resume                      # pick up after an interruption
```

## Files

| File | Needs qcage? | Purpose |
|---|---|---|
| `run_sapt0_scan.py` | no | The scan. psi4 + stdlib only. |
| `watermenh2_adaptive_scan.xyz` | no | 27 frames, 10 atoms each, Angstrom. |
| `adaptive_grid.json` | no | Grid provenance (regions, warnings, builder params). |
| `make_geometries.py` | yes | One-off generator that produced the two files above. |

## Where the geometries come from

The scan points are the 23-point adaptive grid from the recorded `WaterMeNH2`
curve scan, plus the four canonical pilot anchors (scale factors 0.90, 1.00,
1.10, 1.50) — 27 unique points, center-of-mass separation 2.99 → 7.31 Å.

The stored surface row
(`surface_row_store/WaterMeNH2__wb97x-d__jun-cc-pVDZ__cavity_free__cavity_free__20260827T180027538240_75dd200f.json`)
does not keep the grid itself, but it keeps `R_e_fine_angstrom` and
`delta_R_e_pilot_fine_angstrom`, whose sum is the pilot Morse minimum the grid
was centered on:

```
r_reference_angstrom = 3.3242935631440167
```

Every node `qcage.analysis.grid.build_adaptive_scan_grid` emits — the Chebyshev
local window, the uniform sub-stencil, the geometric repulsive wall, and the
r⁻⁶-spaced tail — is a function of that number alone, so replaying the builder
with a pilot curve that fits back to it reproduces the original grid.
`make_geometries.py` asserts both the exact R_min round-trip and the point
count against the row's `n_points_fit: 23` / `n_points_used_total: 27`.

Geometries themselves are the usual qcage construction: monomer A (water) fixed,
monomer B (methylamine) rigidly translated along the 0.90 → 1.00 anchor
displacement. The 0.90 and 1.00 frames reproduce
`src/qcage/data/structures/2715_03WaterMeNH2090.xyz` and `2717_…100.xyz` to
9 decimals.

| scale | R (Å) | region | | scale | R (Å) | region |
|---|---|---|---|---|---|---|
| 0.799672 | 2.9919 | local | | 1.095100 | 3.5736 | local |
| 0.822246 | 3.0361 | local | | 1.100000 | 3.5833 | pilot |
| 0.850486 | 3.0916 | wall | | 1.207645 | 3.7962 | local |
| 0.867407 | 3.1248 | local_stencil | | 1.302891 | 3.9850 | local |
| 0.886443 | 3.1623 | local | | 1.366460 | 4.1111 | local |
| 0.900000 | 3.1889 | pilot | | 1.388769 | 4.1554 | local |
| 0.901222 | 3.1913 | local_stencil | | 1.450917 | 4.2788 | tail |
| 0.935006 | 3.2578 | local_stencil | | 1.500000 | 4.3763 | pilot |
| 0.968759 | 3.3243 | local_stencil | | 1.528777 | 4.4335 | tail |
| 0.982304 | 3.3510 | local | | 1.631850 | 4.6386 | tail |
| 1.000000 | 3.3859 | pilot | | 1.781418 | 4.9365 | tail |
| 1.002485 | 3.3908 | local_stencil | | 2.042990 | 5.4582 | tail |
| 1.036185 | 3.4573 | local_stencil | | 2.970695 | 7.3134 | tail |
| 1.069859 | 3.5238 | local_stencil | | | | |

## Protocol

`psi4.energy("sapt0")` with `scf_type df`, `e_convergence`/`d_convergence` 1e-10,
`guess sad`, default basis `jun-cc-pVDZ` (qcage's `DEFAULT_SAPT_BASIS`).

**`freeze_core` is off by default**, deliberately. qcage's own SAPT0 numbers come
from `cqed-scf`'s `CQEDCalculator.sapt0_components` at λ = 0, which never freezes
core; all-electron psi4 SAPT0 is therefore directly comparable term by term. Pass
`--freeze-core` for the conventional frozen-core literature protocol instead.

The dimer is written as a two-fragment psi4 molecule with `no_reorient`, `no_com`,
`units angstrom`, `symmetry c1` — the same layout
`qcage.geometry.psi4_strings.make_psi4_fragmented_geometry` produces, so psi4 sees
exactly the coordinates in the file. psi4's SAPT is dimer-centered-basis
(counterpoise-corrected) by construction.

### Fitting basis sets

psi4 does not ship `jun-cc-pVDZ-JKFIT`/`-RI` and resolves calendar basis sets
(`jun-`, `jul-`, `may-`) to the `aug-cc-pVDZ` fitting sets. That is the intended
behavior; `--df-basis-scf` / `--df-basis-sapt` are available in case a particular
psi4 build errors instead of falling back.

## Output

`sapt0_scan.csv`, one row per geometry, plus a per-point psi4 output file in
`psi4_outputs/`. Component column names mirror the keys qcage's
`compute/interaction_energy.py` returns, so the two sources diff directly:

`elst10`, `exch10`, `exch10_s2`, `ind20`, `exch_ind20`, `disp20`, `exch_disp20`,
`delta_hf`, `total_hartree`, `total_kcal_mol` — alongside the grouped psi4
quantities `elst`, `exch`, `ind`, `disp`, `sapt_hf_total`, and
`index` / `scale_factor` / `com_distance_angstrom` / `region` / `wall_time_s` /
`status`.

`delta_hf` is recovered as `SAPT IND ENERGY − (IND20,r + EXCH-IND20,r)`, since
psi4 folds δ<sub>HF</sub> into the induction group. Variables absent from a given
psi4 version are recorded as empty rather than aborting the run.

Because 27 SAPT0 jobs is a long run, the CSV is flushed after every point, a
failed point is recorded in `status` and does not stop the scan, and `--resume`
skips scale factors already logged as `ok`. Resume appends, so a point that
failed and later succeeded appears twice — the `ok` row is the later one.

## Expected results

The minimum should land near scale ≈ 0.97–1.00 (R ≈ 3.32–3.39 Å) with
`total_kcal_mol` ≈ −6 to −8, bracketing the stored wb97x-d/jun-cc-pVDZ CP value
of −7.49 kcal/mol at R_e = 3.334 Å. The outermost point (R = 7.31 Å) should be
roughly −0.03 kcal/mol.

## Regenerating

```bash
PYTHONPATH=src python3 examples/sapt0_water_methylamine/make_geometries.py
```

Run from the repo root; it rewrites `watermenh2_adaptive_scan.xyz` and
`adaptive_grid.json` and re-checks the grid assertions.
