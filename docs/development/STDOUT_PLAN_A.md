# STDOUT_PLAN_A — Shared output layer + quiet flag

> **Status: IMPLEMENTED and reviewed on `main`.** This document is the
> historical specification for Stage A. Everything it describes is live; see
> STDOUT_PLAN_B.md for the implementation summary and Stage B plans.

You are working in `/Users/jfoley19/Code/cqed-scf`

Conda env: `p4dev` at `/Users/jfoley19/miniforge3/envs/p4dev`

Branch: `main`

This is the **first stage** of a multi-stage effort to make CQED-SCF stdout
printing more consistent with standard quantum chemistry packages (Psi4,
Gaussian, NWChem). Stage A establishes the *infrastructure* only: a single
output module that all code paths use, plus a `quiet` flag on the
calculation configuration to silence stdout (including whatever Psi4 emits).

**Scope:** Stage A does **not** restructure the content of any specific
calculation (SCF/calculator/SAPT). It introduces the shared writing layer
and the toggle that everything else will later be migrated onto. Actual
content/format redesign of each module is deferred to STDOUT_PLAN_B.md.

## Goals

1. Give the package a single, module-level output mechanism so that a
   silence flag can control *everything* regardless of whether individual
   call sites currently use `print()` or `psi4.core.print_out()`.
2. Add a `quiet` flag to `CQEDConfig` (and hence to `CQEDCalculator` and
   every engine that consumes a config) and honor it across the whole run,
   including Psi4's own engine output.
3. Provide the formatting primitives (banner, `@`-property line, table,
   energy-with-units) that Stage B will use, so that Stage B is pure
   content/formatting work with no plumbing changes.
4. Preserve existing behavior by default (`quiet=False`).
5. Keep `debug` and `quiet` orthogonal.

## What Stage A is NOT

- No redesign of `scf.py`, `calculator.py`, or the SAPT modules' printed
  content. Those are Stage B.
- No new user-facing behavior beyond the ability to silence output.

---

## 1. New module: `src/cqed_scf/output.py`

A dedicated module that owns stdout writing. It holds a single module-level
flag and a small set of emit helper functions so that call sites never write
to a stream directly.

### Module-level API

```python
class Verbosity:
    # simple int threshold
    QUIET   = 0
    NORMAL  = 1
    VERBOSE = 2  # reserved for future use / == debug
```

State (module globals):
- `_verbosity: int = Verbosity.NORMAL` — global verbosity threshold.

Functions:
- `set_quiet(flag: bool)` → sets `_verbosity = QUIET if flag else NORMAL`.
- `quiet_context(q: bool = True)` → context manager that sets the flag on
  entry and restores it on exit (for wrapping `psi4.energy()` calls, etc.).
- `is_quiet()` → bool.
- `is_verbose()` → bool (for future use / parity with `debug`).

### Emit helpers

All emit helpers return immediately if verbosity is below the level they
target. In Stage A only two levels are used: `QUIET` (nothing emitted) and
`NORMAL` (everything emitted). `VERBOSE` is reserved so `debug` can be
mapped to it later without touching call sites.

- `echo(message="" )` — print a single line (normal level). Closest
  replacement for bare `print(...)`.
- `banner(title: str)` — prints a Psi4-style emphasized header:
  ```
  @-----------------
  @ Psi4-style banner text
  @-----------------
  ```
  Actually: Psi4 uses
  ```
  
    ==> Title <== 
  
  ```
  Implement as:
  ```python
  echo(f"\n  ==> {title} <== \n")
  ```
- `property_(label: str, value: float, unit: str = "Eh", fmt: str = "18.12f")`
  → prints a Psi4-`@`-convention final-result line:
  ```
  @ {label:<46s} {value:{fmt}} {unit}
  ```
  e.g. `@ Total Energy (SCF)                    =  -76.059081234567 Eh`.
  (The `@` prefix is the standard marker used by Psi4 for parseable final
  quantities and is easy to `grep`.)
- `table(headers: list[str], rows: Iterable[list], widths: list[int])`
  — aligned columnar output with a header separator, used later for SCF
  iteration and SAPT component tables.
- `energies(list of (label, value, unit))` — a convenience that emits
  several `@` property lines with consistent column widths.

All of them consult `_verbosity` first and return `None` immediately when
quiet.

### Recommendation on stream independence

Stage A keeps the helpers writing to `sys.stdout` via `print()` *through
the helper* (single choke point). This is deliberately simpler than plugging
into Psi4's `core.print_.out` __printer__, but it means a later, optional
redirection to an injected stream/file requires changing only `output.py`.

---

## 2. Psi4 engine output suppression

`psi4.energy()` and the JK/VBase/gradient internals write through Psi4's
own C++ output manager via `psi4.core.print_out()`. A Python-side silence
flag on `output.py` cannot suppress that. Stage A therefore adds a small
Psi4-aware helper in `output.py`:

- `silence_psi4()` — set Psi4's current output to a null sink:
  ```python
  import psi4
  psi4.core.set_output_file('/dev/null')   # platform-appropriate null sink
  ```
  On Windows use a temp file; on POSIX `/dev/null`.
- `restore_psi4()` — return output to the prior destination (typically
  `psout`/stdout). Psi4's `core.set_output_file` sets the new current
  stream, so callers should remember the prior stream or rely on
  `psi4.core.set_output_file` being re-called with the user's intended file
  or `sys.stdout`.

To make this robust and reversible, `output.py` exposes:

```python
@contextmanager
def psi4_silent():
    """Suppress Psi4's own engine output for the duration of the block."""
    ...
    prior = psi4.core.get_output_file()   # if available in installed psi4
    psi4.core.set_output_file(NULL_SINK)
    try:
        yield
    finally:
        psi4.core.restore_output_file(prior)
```

> Psi4 API note: the exact revert API (`core.get_output_file` /
> `core.restore_output_file`) varies by Psi4 version. Stage A should
> introspect available methods at import time and provide `psi4_silent()`
> as either a true suppressor or a documented no-op fallback. This is the
> one Psi4-version-sensitive piece; confirm against the installed Psi4
> before merging.

`quiet_context()` should compose `echo`-silencing and `psi4_silent()` so a
single `with output.quiet_context():` suppresses *both* our `print` calls
and Psi4's engine output. This is the primary entry point callers will use
in Stage B.

## 3. Flag on `CQEDConfig` + `CQEDCalculator`

### `references.py`

- Add field `quiet: bool = False` to `CQEDConfig` (after `debug`). It goes
  through the same `dataclass` machinery; no coercion needed beyond `bool`.
- Add to `from_legacy_kwargs(...)` a `quiet: bool = False` parameter.

### `calculator.py`

- Add `quiet: bool = False` to `CQEDCalculator.__init__` signature.
- Add property + setter for `quiet` mirroring the existing `debug`
  property/setter pattern (constructor assigns into `self.config`).
- Expose `_quiet` and gate the existing non-debug `print()` calls so
  `quiet=True` suppresses them already in Stage A. To keep single-write,
  this is done by routing through `output.echo()` and honoring the flag
  (see below).
- Thread `config.quiet` into the sub-objects constructed later:
  `CQEDSCF`, `CQEDGradient`, `SAPTMonomer`, `QEDSAPT0Driver` (wired in
  Stage B; signature added now where trivial).

### `scf.py`

- Accept `quiet: bool = False` (Plumb from config; default `False` for
  back-compat). Store as `self.quiet`. For Stage A, replace the top-level
  `print()` calls (`scf.py:115-122`) with `output.echo(...)`, which are now
  automatically suppressed when the module flag is quiet. See Stage B for
  full redesign.

### Initial seeding of the global flag

Because many functions dispatch directly (not through the calculator), the
simplest Stage A wiring is:

- `CQEDConfig.__post_init__` does **not** mutate the module global (a config
  is an immutable description).
- The *drivers* long-lived entry points (`CQEDCalculator.energy`,
  `energy_and_gradient`, `energy_and_projected_gradient`) call
  `output.set_quiet(self.quiet)` at top, and reset it on exit
  (`finally/output.quiet()`), so every downstream engine and SAPT call is
  silenced without each constructor needing its own flag.

This keeps Stage A plumbing minimal and guarantees wrap. Stage B will
discuss conversion details per module.

---

## 4. Dropping old-style prints (Stage A migration list)

Within Stage A, update these current call sites to route keyword through
`output.echo()` (preserving exact text) so the flag immediately takes
effect, before any content redesign in Stage B:

- `scf.py:115-122` header prints.
- `calculator.py:179-181` "Running CQED-SCF..." and Functional prints.
- `calculator.py:232-233, 279-287` dispersion/total/no-disp lines.
- `calculator.py:240-242, 296-300` `if self.config.debug:` lines → keep
  debug-gated but route through `output.echo`.
- `sapt/qed_sapt0.py:139-141` debug-only dimer prints.
- `sapt/monomer.py:144-159` debug-only monomer prints.
- `sapt/qed_sapt_jk.py:*` `core.print_out(...)` calls and the
  `_sapt_cpscf_solve` solver header → Stage A can *additionally* wrap these
  in `output.psi4_silent()` or gate their `do_print` on `not quiet`. Since
  these are large, decide in Stage A whether to convert now or defer to B; a
  pragmatic first pass is to gate the existing `print()`/`print_out()` with
  the console flag and leave formatting in B.
- `sapt/dse_jk.py:211` print_header state line.

`core.set_variable(...)` assignments are **not** stdout output and are
outside the scope of the flag.

---

## 5. Backward compatibility

- `quiet=False` default → behavior identical to today (same text, same
  volume). Only the *mechanism* changes (all text routed through
  `output.py`), which is invisible unless someone introspects `sys.stdout`
  vs a future injected stream.
- `debug` and `quiet` are independent: `quiet=True, debug=True` suppresses
  the informational/debug lines; `quiet=False, debug=True` keeps them.
- Psi4's own engine output suppression only engages when `quiet=True`
  diverts to via `psi4_silent()`.

---

## 6. Verification (Stage A)

Basic gating tests that don't require a full QED-SAPT run:

- `python -c "from output import banner, echo, property_; echo('hi'); set_quiet(True); echo(...)"` → nothing printed when quiet.
- Unit tests (`tests/`) for `CQEDConfig(quiet=True/False)`, `set_quiet`
  reset rollback, and `table()`/`property_` formatting.
- Manual smoke: run a tiny RHF `self.energy(geometry)` with `quiet=False`
  (unchanged) and `quiet=True` (believed silent, including no Psi4 headers,
  or documented Psi4-version fallback).
- `pytest` still collects and passes existing tests.

---

## 7. Deliverables for Stage A

- New `src/cqed_scf/output.py` with the helpers above.
- `CQEDConfig.quiet` + `from_legacy_kwargs(quiet=...)`.
- `CQEDCalculator` `quiet` property/setter + set at entry points.
- Route the Stage A list call sites through `output.*`.
- `psi4_silent()` context (with version-tolerant fallback).
- Tests for the flag/formatting.

### Definition of Done

- All Python `print()` in the three modules + monomer/dse_jk go through
  `output.py`.
- `quiet=True` suppresses our output *and* Psi4 engine output on the
  installed Psi4 (else a documented no-op fallback).
- `quiet=False` output is equivalent in content to today's output.
- pytest green.