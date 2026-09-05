# Psi4 arrays are views: a confirmed defect and an audit

**Status:** defect confirmed and fixed (Tier 0, `docs/development/QED_RESPONSE_PLAN.md`).
**Affected:** `results["canonical_orbital_energies"]`, and through it
`QEDSAPT0Driver.compute_Edisp200(canonical_denom=True)`.

## The mechanism

`np.asarray` on a Psi4 `Vector` or `Matrix` returns a **view** of Psi4's buffer,
not a copy. Verified directly:

```python
import numpy as np, psi4
v = psi4.core.Vector(3)
a = np.asarray(v)
v.nph[0][:] = [1., 2., 3.]
print(a)                             # -> [1. 2. 3.]
print(np.shares_memory(a, v.nph[0])) # -> True
```

`CQEDSCF.run()` captured the canonical RHF orbital energies early:

```python
eps_canonical = np.asarray(self.wfn.epsilon_a())   # a VIEW
```

and then, at the end of the same method, wrote the converged CQED orbital
energies into that same buffer:

```python
wfn.epsilon_a().nph[0][:] = eps                    # overwrites the view
```

So `results["canonical_orbital_energies"]` and `results["orbital_energies"]`
were the same memory. Every consumer asking for canonical (cavity-free) orbital
energies silently received the CQED ones.

## How it surfaced

A student comparing QED-SAPT0 dispersion with canonical versus CQED-HF orbital
energies in the denominator reported **exactly identical** energies. That is the
signature: two physically different choices cannot agree bitwise.

The switch is `qed_sapt0.py::compute_Edisp200`:

```python
if canonical_denom:
    self.eps_rsab = 1 / (-self.eps_canonical('r', ...) ...)
else:
    self.eps_rsab = 1 / (-self.eps('r', ...) ...)
```

Both branches were reading the same array.

## Why the test suite did not catch it

The only test exercising `canonical_denom=True` ran at `lambda_vector = [0, 0, 0]`,
where CQED-SCF reduces to RHF and the two orbital-energy sets *genuinely*
coincide. The test was blind to the defect by construction. Coverage of a branch
is not coverage of the condition that makes the branch matter.

## The fix

`src/cqed_scf/scf.py`, both sites made explicit copies:

```python
eps_canonical = np.array(self.wfn.epsilon_a(), copy=True)
C             = np.array(self.wfn.Ca(), copy=True)
```

Regression tests added:

- `tests/test_scf_canonical_orbitals.py::test_canonical_orbital_energies_are_not_aliased_to_the_cqed_ones`
- `tests/test_qedsapt0_driver.py::test_canonical_and_cqed_orbital_energies_are_independent_arrays`
  (asserts `not np.shares_memory(...)` -- catches the mechanism, not just a symptom)
- `tests/test_qedsapt0_driver.py::test_canonical_orbital_energies_are_the_cavity_free_ones`
  (anchors `eps_canonical(lambda != 0) == eps(lambda = 0)`, so a wrong-but-different
  array cannot pass)
- `tests/test_qedsapt0_driver.py::test_edisp200_denominator_choice_matters_at_finite_coupling`
  plus its `lambda = 0` control

## Audit of the remaining `np.asarray` calls on Psi4 objects

The dangerous pattern is *view taken, then the same buffer written to*. Only the
two fixed sites had it.

| site | verdict |
|---|---|
| `scf.py` `epsilon_a()`, `Ca()` | **was the defect** -- fixed, now copies |
| `scf.py` `ao_dipole()`, `ao_quadrupole()` | safe: stacking a *list* of matrices builds a new array |
| `scf.py` `ao_kinetic/potential/overlap`, `jk.J()`, Vxc | views, but nothing writes back to those buffers |
| `qed_sapt0.py` `S_dimer`, `V_A`, `V_B` | views, never written to |
| `qed_sapt0.py` `I_dimer_standard` | view *by design* -- copying the full ERI tensor would be costly, and line 279 correctly does `I_dimer = I_dimer_standard.copy()` before adding the cavity term |
| `dse_jk.py`, `qed_sapt_jk.py` | safe: inputs are already NumPy |
| `gradients.py:71` `np.asarray(psi4.core.scfgrad(wfn))` | views a temporary Matrix. pybind11 keep-alive should hold it; an `(natom, 3)` `.copy()` is free and would remove the question |

## Rule for this codebase

**Copy at the boundary when the Psi4 object outlives the array, or when anything
may write to it later.** `np.asarray` is appropriate only for large read-only
tensors (the dimer ERIs) where the copy has a real cost and no writer exists.

A view aliasing bug does not raise -- it produces plausible numbers and silently
collapses two code paths into one. The cheap structural guard is
`assert not np.shares_memory(a, b)` in a test, which fails on the mechanism
rather than waiting for a physical consequence to look wrong.
