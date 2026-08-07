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
primitive used. No new plumbing or flag work is expected here unless gaps
appear during implementation.

---

## Stage A — implementation summary (what is now in place) [REVIEWED]

Stage A is implemented and green. Everything below is **already live on
`main`**; Stage B only has to move content onto it.

### Delivered infrastructure
- **`src/cqed_scf/output.py`** — one choke point for stdout:
  - `Verbosity` (`QUIET=0`, `NORMAL=1`, `VERBOSE=2`) and a module global
    `_verbosity`.
  - `set_quiet(flag)`, `quiet_context(q=True)`, `is_quiet()`, `is_verbose()`.
  - Emit helpers `echo`, `banner` (`\n  ==> {title} <== \n`), `property_`
    (`@ {label:<46s} {value:{fmt}} {unit}`), `table`, `energies`. All return
    `None` immediately when quiet.
- **Psi4 engine suppression**: `psi4_silent()` context manager (see API note
  below for the reconciled revert mechanism) plus a platform-aware
  `_null_sink()` (`/dev/null` on POSIX, temp file on Windows).
- **Config/calculator plumbing**:
  - `CQEDConfig.quiet: bool = False` (+ `from_legacy_kwargs(quiet=...)`).
  - `CQEDCalculator.quiet` property/setter mirroring `debug`.
  - The long-lived entry points `energy`, `energy_and_gradient`, and
    `energy_and_projected_gradient` wrap their bodies in
    `output.quiet_context(self.config.quiet)`. This is the **single choke
    point** that silences every downstream engine and SAPT call, so
    sub-objects (`CQEDSCF`, `SAPTMonomer`, drivers, etc.) do **not** need to
    carry their own `quiet` flag to be silenced.
- **Routed call sites** (preserving exact prior text): `scf.py` header/debug
  prints, `calculator.py` all prints, `qed_sapt0.py` debug + diagnostics,
  `monomer.py` debug prints, `dse_jk.py:211` (gated on `output.is_quiet()`),
  and `qed_sapt_jk.py` `core.print_out(...)` (each routine gates its
  `do_print` with `do_print and not output.is_quiet()`).
- **Tests**: `tests/test_output.py` (verbosity gating, rollback, formatting,
  config flag, debug/quiet orthogonality). Full suite (including
  previously-`slow` cases) and the canonical examples pass.

### Verified behavior
- Smoke RHF energy: `quiet=False` output unchanged vs pre-Stage-A;
  `quiet=True` is fully silent (no CQED-SCF headers, no Psi4 engine output)
  and the energy is unchanged.
- Canonical examples now each demonstrate a verbosity mode intentionally:
  `cqed_rhf_energy.py` (silent / `quiet=True`),
  `cqed_sapt0_components.py` (verbose / `debug=True, quiet=False`),
  gradient examples (`cqed_dft_energy_gradient.py`,
  `cqed_rhf_energy_gradient.py`, `cqed_dft_projected_gradient.py`) (
  normal / `quiet=False`, with a comment explaining the gradient caveat).

---

### Reconciliation — Stage A findings that revise a few assumptions in this
plan (these are corrections, not new scope)

1. **`property_` emits no `=` separator.** The stage implementation prints
   `@ {label:<46s} {value:{fmt}} {unit}` — a space, not `"="`, separates the
   label from the value. Several Stage B target blocks below show
   `=    -76.059... Eh`. Those examples must be updated to match the actual
   format (or, if the `=` is wanted, `output.property_` needs an optional
   `eq=` flag — not recommended; a space is already a valid greppable Psi4
   convention). **Action: edit the affected target blocks in this doc.**
2. **Psi4 revert API.** Installed Psi4 is **1.10** and exposes
   `core.get_output_file()` (returns a `str`, e.g. `"stdout"` or a basename)
   and `core.set_output_file(prior)` — there is **no**
   `core.restore_output_file()`. `psi4_silent()` therefore reverts via
   `set_output_file(prior_string)`, with a version-tolerant no-op fallback.
   Do not rely on `restore_output_file` in Stage B.
3. **`debug`-gated lines already honour `quiet`.** Stage A routes every
   debug line through `output.echo`, so `if self.debug: output.echo(...)`
   already prints only when `debug and not quiet`. A separate
   `debug_output()` helper is therefore **not needed** in Stage B. If the
   team later wants `Verbosity.VERBOSE` (so `debug` feels higher-priority
   than `NORMAL`), do it by mapping `debug` to a `VERBOSE` verbosity with a
   `set_verbose()`; the call sites (already `output.*`) will not change.
4. **Quiet wrap is at the entry points.** Because `calculator` entry points
   set/reset `quiet_context`, Stage B per-module "thread quiet into
   constructors" notes are unnecessary for silencing. Constructors only need
   `quiet` if a module can be driven directly (e.g. a stand-alone
   `CQEDGradient.compute` call or a SAPT driver used outside
   `CQEDCalculator`), where the entry-wrap is not in effect. Decide per
   module; do not blanket-thread.
5. **Out-of-scope raw `print()` remain in other modules.** Stage A's DoD
   covered `scf.py`, `calculator.py`, and the SAPT modules. Unrouted raw
   `print()` still exist in `gradients.py`, `drivers.py` (MD), `uscf.py`,
   and `ugradients.py`. Those are **not** silentable by `quiet` today. They
   should be swept in a later stage (Stage C), not Stage B, unless they fall
   out of the SAPT/gradient work there.
6. `qed_sapt_jk.py` and `qed_sapt0.py` print statements were already
   routed/gated in Stage A; the "Current issues" paragraphs below still
   describe their pre-Stage-A state. Re-read them as "already routed
   through `output.echo`; now reformat".

Cross-cutting conventions applied everywhere (unchanged):
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

    @ CQED-SCF Total Energy         -76.059081234567 Eh
    @ SCF Iterations               12
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

The whole method body is already wrapped in
`output.quiet_context(self.config.quiet)` (Stage A), so no additional
wrapping is needed here. Replace `output.echo("\nRunning...")` with a banner
and emit final results:

```
  ==> CQED-SCF Energy <==

    @ Energy (CQED-SCF)              -76.059081234567 Eh
    @ Energy (Dispersion, D4)        -0.001234567890 Eh
    @ Energy (Total)                -76.060315802457 Eh
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

    @ E (CQED-SCF)                   -76.123081234567 Eh
    @ E (Dispersion)                 -0.001234567890 Eh
    @ E (Total)                     -76.124315802457 Eh
    @ |g (Total)|                     5.234567e-03 Eh/Bohr
    @ |g (Dispersion)|                1.234567e-05 Eh/Bohr
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
- `sapt/qed_sapt0.py` — a diagnostics scaffold whose prints are routed through
  `output.echo` (Stage A); Stage B reformats them with `table`/`property_`.

Both migrate onto `output.py`.

### `sapt/qed_sapt_jk.py`

**Current issues (after Stage A)**
- Still uses `psi4.core.print_out(...)` directly (written through Psi4's
  stream, not `output.py`). Stage A gates each routine's `do_print` with
  `do_print and not output.is_quiet()`, so quiet mode silences them, but the
  text still lands on Psi4's stream rather than through `output.*`.
- Banners are ad hoc strings (`:123`, `:268`, `:289`, `:404`, `:506`,
  `:683`) and include a typo "Electostatics" (`:268`).
- Values emitted via `print_sapt_var(...)` with inconsistent/only-one unit;
  no multi-unit table, no `@` final lines, no overall summary table.
- The CPHF/CPKS solver prints its own header + iteration table
  (`:960-1005`); Stage A gated those on `not output.is_quiet()`.

**Target form:**

Route every `core.print_out(...)` through `output.echo(...)`/`banner(...)`;
replace `print_sapt_var` calls with `output.property_()` lines or a shared
`output.sapt_component()` helper. Once the migration is complete, the
`do_print = do_print and not output.is_quiet()` Stage A gate lines become
redundant and can be dropped (keep the `do_print` signature default for
back-compat, but let `output`'s verbosity be the single authority).

Per-component output:

```
  ==> E10 Electrostatics <==

    @ Elst10, r                        -12.345678901234 Eh
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
- The `do_print` / `print_output` parameters no longer need to select a
  stream: Stage A already makes them consult `output.is_quiet()`. Stage B
  simply converts the `core.print_out(...)` bodies to `output.*` and removes
  the now-redundant gate lines. Keep signatures (`do_print=True` default)
  for back-compat.

### `sapt/qed_sapt0.py`

**Current issues (after Stage A):**
- The debug dimer prints (`:139-141`) and diagnostic tables (`:590-656`)
  already route through `output.echo` (Stage A); they still use manual
  f-string alignment instead of `output.table()`/`output.property_()`.

**Target form:**
- Reformat the already-`output`-routed diagnostics with `output.table()` /
  `output.property_()` (they are `output.echo` today, so this is a pure
  formatting swap).
- Keep the cavity diagnostics debug-gated
  (`diagnostic_summary(print_output=config.debug)` in `:658-659`).
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
3. **`sapt/qed_sapt_jk.py`** — move `core.print_out` to `output`; fix typo;
   emit per-component `@` lines and the multi-unit final table; wire the
   solver header to `output`; drop the now-redundant Stage A
   `do_print and not output.is_quiet()` gates.
4. **`sapt/qed_sapt0.py` + `sapt/dse_jk.py`** — swap the existing
   `output.echo` diagnostics to `output.table`; add component summary table;
   suppress adapter header in quiet mode (already gated; verify).

Each step is followed by running `pytest` and a tiny smoke example.

---

## Stage B acceptance / review criteria

- Normal (`quiet=False`) output is informative, well-formatted, and greppable
  (`@`-prefixed final values everywhere).
- Verbose detail is `debug`-gated.
- Quiet mode emits as little as possible across all modules, including Psi4
  engine output (via Stage A's `psi4_silent`).
- Only stdout text/format changes — no numerical or behavioral change, and
  the resulting text matches the target blocks above.
- `pytest` and the examples pass and match reference outputs (`examples/*`).
- `gradients.py` / `drivers.py` / `uscf.py` prints remain unchanged here
  (tracked under Stage C); they stay non-quiet-capable by design until Stage C.

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

**Post-A review note:** the `debug_output()` helper is **not needed**. Stage A
routes every debug line through `output.echo`, so `if self.debug:
output.echo(...)` already prints only when `debug and not quiet`. If the team
wants `debug` to map to a distinct `Verbosity.VERBOSE` level later (so it
outranks `NORMAL`), do that by mapping `debug` to verbosity via a
`set_verbose()`; call sites will not need to change. Defer that unless there
is a concrete need.

---

## Future work — Stage C (out of Stage B scope)

Stage A's DoD covered `scf.py`, `calculator.py`, and the SAPT modules.
Unrouted raw `print()` still exist and are **not silentable** by `quiet`
today:

- `gradients.py` — per-term timings and gradient/debug matrices
  (these affect `CQEDGradient` output when driven directly).
- `drivers.py` — MD observer/step timing prose.
- `uscf.py`, `ugradients.py` — unrestricted driver prints.

Route these through `output.*` and gate on `config.quiet` in Stage C
(discrete, low-risk follow-up). Until then the canonical gradient examples
should stay `quiet=False` because `quiet=True` would only partially silence
them.