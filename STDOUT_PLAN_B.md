# STDOUT_PLAN_B — Tentative content/format migration for scf, calculator, SAPT

This is the **second stage**, to be written only after Stage A
(STDOUT_PLAN_A.md) is implemented and reviewed. Stage A establishes the
shared `output.py` writing layer and the `quiet` flag. Stage B redesigns
*what* and *how* each module prints to match standard quantum chemistry
conventions (Psi4-style banners, `@`-prefixed final values, units, aligned
tables).

These are **tentative** plans. They describe the target output for each
module so implementation can proceed stepwise, with Stage A's `output.py`
helpers (`banner`, `property_`, `table`, `energies`) as the only printing
mechanism used. No new plumbing or flag work is anticipated here unless gaps
appear during implementation.

Cross-cutting conventions applied everywhere:
- Use only `output.py` emit helpers; no raw `print()` or
  `psi4.core.print_out()` at call sites.
- Final/results scalar values printed as Psi4 `@`-prefixed property lines.
- Fixed-width numeric columns; energies in Eh with ~12-16 significant
  digits; deltas/norms in scientific notation.
- Progress/verbose details gated on `debug`; the headline result printed at
  normal verbosity.
- Respect `quiet` automatically (enforced by `output.py`).

---

## Part B1 — `scf.py` (`CQEDSCF.run`)

### Current issues
- Header is plain prose ("Starting CQED-SCF calculation...",
  `scf.py:115-122`).
- The converged SCF energy is printed **only** under `debug` — normal runs
  show no headline result.
- Per-iteration table exists only in `debug` (`scf.py:274-290`) and lacks a
  wall-clock column and units header.
- No calculation summary header (method, basis, charge, multiplicity,
  electrons, orbitals, cavity parameters).

### Target output

**Header block** (normal verbosity), Psi4-style. Pull metadata from the
Psi4 wavefunction (`self.wfn`) and geometry once available:

```
        CQED-SCF Calculation
        --------------------

  Method              = RHF
  Basis               = cc-pVDZ
  Charge              = 0
  Multiplicity        = 1
  Number of electrons = 18
  Orbitals (AO / MO)  = 76 / 76

  Cavity parameters
  -----------------
  lambda              = (0.000000, 0.000000, 0.100000)
  omega               = 0.100000 Eh
  Density fitting     = False
```

(The exact field set is provisional; derive what is cheaply available from
`self.wfn`/`self.mol`.)

**SCF iteration table** (normal verbosity):

```
  ==> SCF Iterations <==

   Iter     Energy (Eh)          dE (Eh)      dRMS (a.u.)   Time (s)
  -------------------------------------------------------------------
     1    -76.044712345678    0.000e+00     6.42e-01     0.003
     ...
    12    -76.059081234567    1.000e-12     8.110100000e-09  0.002
  -------------------------------------------------------------------
  SCF Converged in 12 iterations.
```

- `E` from the computed SCF value (both rhf and rks branches).
- `dE = E - Eold`; `dRMS` already computed in `scf.py:238`.
- Add per-iteration wall time (`time` around each `run` cycle).
- Mark a tolerance-satisfying iteration with `*` (like Psi4) and add the
  `SCF Converged...` line.

**Final result** (normal verbosity):

```
  ==> CQED-RHF Total Energy <==

    @ CQED-SCF Total Energy   =    -76.059081234567 Eh
    @ SCF Iterations          =    12
```

Keep the `E_H`, `E_J`, `E_K`, `E_Exc`, `E_N`, `E_wK` component breakdown
(`scf.py:282-290`) **debug-only**, reformatted as an aligned
`properties`/`table` block.

**Debug-only verbose:** the `_build_jk`/`_build_vbase` debug prints
(`scf.py:190-195`) stay `debug`-gated and route through `output`.

---

## Part B2 — `calculator.py` (`CQEDCalculator`)

### Current issues
- Ad-hoc prose strings ("Running CQED-SCF energy calculation...",
  `calculator.py:179-181`) rather than a banner.
- Dispersion/total energies and gradient norms printed as bare `print` with
  no `@` marker and inconsistent units/precision (`calculator.py:232-242`,
  `279-300`).
- Nothing written to Psi4 globals for downstream tooling.

### Target shape

**`energy(geometry)`**

Wrap the run region in `with output.quiet_context():` (or rely on the
entry-point `output.set_quiet`). Replace `print("\nRunning...")` with a
banner, then emit final results:

```
  ==> CQED-SCF Energy <==

    @ Energy (CQED-SCF)       =    -76.059081234567 Eh
    @ Energy (Dispersion, D4) =     -0.001234567890 Eh
    @ Energy (Total)          =    -76.060315802457 Eh
```

- Only include the dispersion line when post-SCF dispersion applies; with
  the `@`-property presentation the "No dispersion correction applied."
  line is redundant and omitted.
- Keep the `debug` block (`calculator.py:239-242`) as a verbose
  `@`-property breakdown (`E_QED`, `E_disp`, `E_tot`) gated on debug.

**`energy_and_gradient()`**

Add a gradient section after the energy lines:

```
  ==> CQED Gradient <==

    @ E (CQED-SCF)         =    -76.123081234567 Eh
    @ E (Dispersion)       =     -0.001234567890 Eh
    @ E (Total)            =    -76.124315802457 Eh
    @ |g (Total)|          =     5.234567e-03 Eh/Bohr
    @ |g (Dispersion)|     =     1.234567e-05 Eh/Bohr
```

(dispersion lines only when applicable.)

Also populate Psi4 globals so other drivers can consume the values:
`psi4.core.set_variable("CURRENT ENERGY", energy_total)` and a
`"CURRENT GRADIENT"` (plus per-component names) — a Stage B nicety kept
orthogonal to stdout.

`energy_and_projected_gradient` reuses `energy_and_gradient` and needs no
new printing (projection diagnostics remain debug-gated in `drivers.py`).

---

## Part B3 — SAPT modules

Two modules with different mechanisms:
- `sapt/qed_sapt_jk.py` — an existing working SAPT/DFT-implementation
  (JK-based) using `psi4.core.print_out()` and `print_sapt_var`.
- `sapt/qed_sapt0.py` — a diagnostics scaffold using `print()`.

Both migrate onto `output.py`.

### `sapt/qed_sapt_jk.py`

**Current issues**
- Uses `psi4.core.print_out(...)` directly (written through Psi4's stream,
  not `output.py`), so the Python-side `quiet` flag does not silence it.
- Banners are ad hoc strings (`:123`, `:268`, `:289`, `:404`, `:506`,
  `:683`) and include a typo "Electostatics" (`:268`).
- Values emitted via `print_sapt_var(...)` with inconsistent/only-one unit;
  no multi-unit table, no `@` final lines, no overall summary table.
- The CPHF/CPKS solver prints its own header + iteration table
  (`:960-1005`).

**Target form:**

Route every `core.print_out(...)` through `output.echo(...)`/`banner(...)`;
replace `print_sapt_var` calls with `output.property_()` lines or a shared
`output.sapt_component()` helper.

Per-component output:

```
  ==> E10 Electrostatics <==

    @ Elst10, r              =      -12.345678901234 Eh
```

Then a composite SAPT summary table at normal verbosity listing all
components, following the standard practice of giving results in Eh and
alternative units (kcal/mol, kJ/mol, cm^-1):

```
  ==> SAPT0 Interaction Energies <==

  Component                        (kcal/mol)       (kJ/mol)    (cm^-1)
  --------------------------------------------------------------------
  Electrostatics                 -12.345678     ...      ...
  Exchange                       +04.567890     ...
  Induction, r                    ...
  Exchange-Induction, r           ...
  Dispersion                      ...
  Exchange-Dispersion             ...
  --------------------------------------------------------------------
  Total                           ...
  --------------------------------------------------------------------

  @ SAPT0 Total                    -0.123456789012 Eh
```

- Add nuclear-repulsion breakdown to the electrostatics section.
- Fix the "Electostatics" typo.
- Convert the `do_print` / `print_output` parameters throughout
  (`build_sapt_jk_cache`, `electrostatics`, `exchange`, `induction`,
  `_sapt_cpscf_solve`) so they no longer select a stream; instead they defer
  to the `output` verbosity level. Keep the signatures (`do_print=True`
  default) for back-compat but have them effectively consult `output`.

### `sapt/qed_sapt0.py`

**Current issues:**
- Debug-only prints (`:139-141`) and diagnostic tables (`:590-656`) use raw
  `print`/f-strings with manual alignment.

**Target form:**
- Route all of it through `output.table()` / `output.property_()`.
- Keep the cavity diagnostics debug-gated
  (`diagnostic_summary(print_output=config.debug)` in `:658-659`) but
  reformat with consistent widths.
- Emit a component interaction-energy table when components are available,
  consistent with `qed_sapt_jk.py` (Eh + kcal/mol + kJ/mol + cm^-1).

### `sapt/dse_jk.py`
- `print_header` / `core.print_out` line (`:211`) → route through `output`
  so it is silenced in quiet mode.

---

## Recommended order of implementation for Stage B

Implement in this sequence so each module is fully migrated before the next:

1. **`scf.py`** — use `banner`/`table`; add iteration wall-time column; emit
   the normal-verbosity converged-energy `@` line; keep component
   decomposition debug-only; route all output through `output`.
2. **`calculator.py`** — replace prose strings with banner + `@` property
   lines; add gradient-norm property lines; route through `output`; add the
   optional Psi4-global writes; verify both `energy` and
   `energy_and_gradient`.
3. **`sapt/qed_sapt_jk.py`** — move to `output`; fix typo; emit per-component
   `@` lines and the multi-unit final table; wire the solver header to
   `output`.
4. **`sapt/qed_sapt0.py` + `sapt/dse_jk.py`** — reformat diagnostics with
   `output.table`; add component summary table; suppress adapter header in
   quiet mode.

Each step is followed by running `pytest` and a tiny smoke example.

---

## Stage B acceptance / review criteria

- Normal (`quiet=False`) output is informative, well-formatted, and greppable
  (`@`-prefixed final values everywhere).
- Verbose detail is `debug`-gated.
- Quiet mode emits as little as possible across all modules, including Psi4
  engine output (via Stage A's `psi4_silent`).
- No behavioral/numerical changes — only printing.
- `pytest` and the examples pass and match reference outputs (`examples/*`).

---

## Suggested shared additions to `output.py` (decided during Stage B)

```python
def sapt_component(label, value, units=("Eh", "kcal/mol", "kJ/mol", "cm^-1")) -> None:
    """Emit a SAPT component line with the requested unit set."""
    # placeholder — decide exact units/precision in Stage B
```

And a `debug_output()` helper that emits only when `not quiet` and `debug`
are both true, so every module can gate verbose lines consistently without
inline `if` blocks.

(These are provisional and will be finalized during Stage A review.)