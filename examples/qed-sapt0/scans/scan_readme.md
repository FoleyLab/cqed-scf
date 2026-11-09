# QED-SAPT0 scan: water–methylamine, density-fitted vs dense

A canonical-style example of the performant QED-SAPT0 path, on the same system
as the plain-psi4 scan in `../sapt0_water_methylamine/`.

```bash
python qed_sapt0_scan.py
```

Roughly **80 seconds** — five separations, each run twice (once per integral
backend), plus a λ = 0 control. No arguments, no output files; the five dimer
geometries are hard-coded so the script is self-contained.

## What it shows

| Section | Point |
|---|---|
| DF vs dense per point | `integral_backend="df"` reproduces `"full_eri"` to DF error at ~4.5x the speed |
| Dispersion partition | The cavity-mediated part of `Disp20` does not decay with separation |
| λ = 0 control | QED-SAPT0 reduces to ordinary SAPT0, comparable to the sibling folder |

## Geometries

Five frames from `../sapt0_water_methylamine/watermenh2_adaptive_scan.xyz`,
keyed by center-of-mass separation:

| R / Å | scale | region |
|---|---|---|
| 3.0916 | 0.850486 | repulsive wall |
| 3.3243 | 0.968759 | R_e of the stored curve |
| 3.3859 | 1.000000 | reference anchor |
| 4.3763 | 1.500000 | mid-range |
| 7.3134 | 2.970695 | tail |

These are *not* the geometry pinned in
`tests/test_qedsapt0_driver.py::test_qedsapt0_driver_water_methylamine_qed_sapt_example`,
which sits between two scan frames — so the numbers here will not match those
pinned values, and that is expected.

## Results

`λ = (0, 0, 0.1)`, `ω = 0.1 Eh`, jun-cc-pVDZ, `scf_type pk` (so the *only*
difference between the two backends is the SAPT integral path):

| R / Å | E(df) / Eh | E(full_eri) / Eh | Δ / Eh | t_df | t_dense |
|---|---|---|---|---|---|
| 3.0916 | −0.007200850779 | −0.007199017932 | −1.8e−06 | 2.9 s | 12.8 s |
| 3.3243 | −0.010570887506 | −0.010570164378 | −7.2e−07 | 2.8 s | 12.6 s |
| 3.3859 | −0.010832881246 | −0.010832314660 | −5.7e−07 | 2.8 s | 12.5 s |
| 4.3763 | −0.006153717106 | −0.006154291840 | +5.7e−07 | 2.5 s | 12.3 s |
| 7.3134 | −0.001532237547 | −0.001532136801 | −1.0e−07 | 2.1 s | 11.8 s |

Agreement is ordinary density-fitting error, largest at the repulsive wall where
the components themselves are largest.

### The dispersion partition is the point

`Disp20` is *quadratic* in the two-electron numerator, so `standard + cavity`
does not sum to the total — there is a cross term. The three-way split does sum,
exactly:

| R / Å | standard | cross | cavity | total |
|---|---|---|---|---|
| 3.0916 | −0.0080882533 | −0.0010148577 | −0.0009745400 | −0.0100776510 |
| 3.3243 | −0.0045776638 | −0.0007952455 | −0.0009626448 | −0.0063355541 |
| 3.3859 | −0.0039501096 | −0.0007475941 | −0.0009596470 | −0.0056573507 |
| 4.3763 | −0.0004921684 | −0.0003227988 | −0.0009226941 | −0.0017376613 |
| 7.3134 | −0.0000126371 | −0.0000673893 | −0.0009007390 | −0.0009807654 |

Over 3.09 → 7.31 Å the `standard` column falls by a factor of 640 while `cavity`
changes by 7.6%. The cavity kernel is a bare product of transition dipoles with
no Coulomb operator, so it carries no distance decay — both monomers couple to
the same cavity mode. See `docs/qed_sapt0_formalism.tex`, "Long-range behaviour".

This is also why the script passes `monomer_reference_frame="monomer_com"`.
CQED-SCF orbital energies are not translation invariant, so under the default
`"dimer"` frame the dispersion tail depends on where the coordinate origin sits
— precisely the regime a distance scan probes.

## λ = 0 control

At R = 3.3859 Å with λ = 0, every component matches psi4's own `sapt0` to
density-fitting error:

| component | example (λ = 0) | psi4 `sapt0` | Δ / Eh |
|---|---|---|---|
| Electrostatics | −0.018562594781 | −0.018565254455 | 2.7e−06 |
| Exchange (S²) | 0.015837377592 | 0.015838348918 | −9.7e−07 |
| Dispersion | −0.003994331751 | −0.003994149762 | −1.8e−07 |
| Exchange-dispersion | 0.000919916818 | 0.000919912518 | 4.3e−09 |
| Induction | −0.007694063305 | −0.007693175810 | −8.9e−07 |
| Exchange-induction | 0.004536726286 | 0.004536857211 | −1.3e−07 |
| **sum of six** | **−0.008956969141** | **−0.008957461379** | **4.9e−07** |

**QED-SAPT0 does not compute δ<sub>HF</sub>**, so its total is not directly
comparable to psi4's `SAPT0 TOTAL ENERGY`. Here psi4 gives δ<sub>HF</sub> =
−0.002243551841 Eh, so psi4's total is −6.915 kcal/mol against this example's
−5.621 kcal/mol for the six explicit terms. Compare term by term, not total to
total.

Note also that exchange must be compared in the S² convention
(`SAPT EXCH10(S^2) ENERGY`); psi4 also reports an S^∞ value.
