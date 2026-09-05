# QED-SAPT0 dispersion: audit and implementation plan

**Status:** Phases 0-3 implemented and tested (2026-09-05);
Phases 4-6 outstanding.
**Open defect surfaced by Phase 3:** dispersion is not translation invariant --
see "Finding: the dispersion denominators are origin dependent" below.
Formalism companion: `docs/qed_sapt0_formalism.tex`.
**Scope:** (1) audit the existing QED-SAPT infrastructure, (2) implement the
remaining MP2-like terms, `Disp20` and `Exch-Disp20`, in the performant
JK/DSE path.
**Companion notes:** `docs/SAPT_NOTE.md`,
`docs/development/psi4_array_aliasing.md`, `docs/development/dse_jk_cphf_design.md`,
`docs/qed_cis_formalism.tex`.

---

## 0. Executive summary

Two findings shape the whole plan.

**Finding 1 — `sapt_mp2_terms.py` is not the analogue of `sapt_jk_terms.py`,
and the interface strategy that worked for JK does not transfer.**

`sapt_jk_terms.py` is NumPy-level Python that consumes a `psi4.core.JK`
object. That is exactly why `PauliFierzJK` works: it duck-types `JK`, and the
Coulomb/exchange algebra above it never needs to know the DSE exists.

`sapt_mp2_terms.py` has no comparable seam. All three of its routines are thin
wrappers over compiled C++:

| routine | delegates to | injectable? |
|---|---|---|
| `df_mp2_sapt_dispersion` | `core.sapt(dimer_wfn, wfn_A, wfn_B)` | no — whole SAPT0 in C++ |
| `df_mp2_fisapt_dispersion` | `core.FISAPT(wfn).disp(matrix_cache, vector_cache, …)` | partially — see below |
| `df_fdds_dispersion` | `core.FDDS_Dispersion` | no, and it is a different theory (coupled FDDS, not MP2 `Disp20`) |

`FISAPT::disp` is the closest: it accepts a matrix cache
(`S, D_A, P_A, V_A, J_A, K_A, D_B, P_B, V_B, J_B, K_B, K_O`) and a vector cache
of orbital energies — **keys our `build_sapt_jk_cache()` already produces**. So
the DSE-dressed one-electron and J/K matrices could be handed to it. But the
amplitude denominators, the `(ar|bs)` numerator, and every ERI-driven
contraction inside are built from DF integrals **constructed internally over
the primary/auxiliary basis pair**. There is no hook for the DSE kernel. Passing
dressed `V`/`J`/`K` while the two-electron core stays cavity-free would produce
a *partially* dressed dispersion — worse than not doing it, because it would
look plausible.

**Finding 2 — the DSE two-electron term is exactly a rank-one density-fitting
factorization, and this makes the problem much easier than it looks.**

The unifying statement already recorded in `SAPT_NOTE.md` §4 is that the
two-electron DSE is an ERI with a separable kernel, `(pq|rs) -> d_pq d_rs`.
Read that in DF language and it says something stronger:

```
standard:  (pq|rs) ~= sum_Q  B^Q_pq B^Q_rs        (Q = 1 .. naux)
cavity:    (pq|rs)_DSE = d_pq d_rs                 (one term, exact)
total:     (pq|rs) ~= sum_Q' B^Q'_pq B^Q'_rs       (Q' = 1 .. naux+1)
           with B^(naux+1) := d
```

**The DSE is one extra auxiliary function.** Append one row to the DF
three-index tensor and every downstream MP2-like formula — amplitudes,
`Disp20`, the whole `Exch-Disp20` `h`/`q` algebra — is unchanged, sign
conventions included. No parallel cavity code path, no new signs to reconcile,
and `standard` / `cavity` / `total` partitions fall out of slicing the auxiliary
index.

This requires the kernel to be symmetric under `(pq) <-> (rs)`, i.e.
`d_A == d_B`. **Verified numerically** (water/methylamine, STO-3G,
`lambda = [0,0,0.1]`): `max|d_A - d_B| = 0.0` exactly, and `d` is symmetric.
This is structural rather than accidental — `d_ao = lambda . mu_ao`
(`scf.py:156`) is a property of the *shared dimer AO basis and frame*, and both
monomers are ghosted calculations in that same basis. It held with and without
`no_com`/`no_reorient`. It must still be asserted in code, because the entire
design rests on it.

**Recommended path:** build a `PauliFierzDF` tensor — the MP2-layer sibling of
`PauliFierzJK` — and implement `Disp20`/`Exch-Disp20` in NumPy on top of it.
Use **conventional** `psi4.energy('sapt0')` as the `lambda = 0` validation
oracle, where the theories coincide exactly.

**Not FISAPT.** `FISAPT` is *functional-group* SAPT, a different partitioning;
`df_mp2_fisapt_dispersion` is the SAPT(DFT) dispersion path that happens to
reuse its kernel. The oracle must be ordinary SAPT0, whose relevant Psi4
variables are `SAPT DISP20 ENERGY`, `SAPT EXCH-DISP20 ENERGY`,
`SAPT ELST10,R ENERGY` and `SAPT EXCH10(S^2) ENERGY` (the `(S^2)` form is the
one matching our dense exchange convention).

### 0.1 Decisions taken (Jay, 2026-09-05)

**D1 — all dispersion physics lives in cqed-scf; psi4's SAPT MP2 machinery is
not used, not even additively.** The standard and DSE MP2-like terms *are*
additively separable in principle, so one could evaluate the canonical terms on
the psi4 side and add our DSE terms. Rejected: psi4 would apply its own orbital
energies in the denominators, and the CQED denominators are easy to mis-apply
across that boundary. The workaround — updating the `wfn` object with CQED
orbitals — exists and is already done by `_update_wfn_with_cqed()`, but it makes
a silent convention mismatch the default failure mode rather than a loud one.
A second reason to keep one implementation: psi4 would also bring its own DF
fitting basis and its own exchange convention (`Exch10` vs `Exch10(S^2)`, the
mismatch already recorded in `SAPT_NOTE.md`). One code path, one set of
conventions.

Psi4 retains exactly one role: conventional `psi4.energy('sapt0')` as a
`lambda = 0` oracle, where the theories coincide exactly and psi4 is an
*independent* implementation.

**D2 — CQED orbital energies are the production denominator.**
`canonical_denom=True` stays, but is documented and tested as a **diagnostic**,
not a setting. This closes the open question formerly in section 4.

---

## 1. Audit of the existing QED-SAPT infrastructure

### 1.1 Current state map

| layer | dense reference (`qed_sapt0.py`) | JK/DSE path (`qed_sapt_jk.py`) |
|---|---|---|
| `Elst10` | `compute_Elst100()` | `electrostatics()` — agrees to ~1e-14 |
| `Exch10` | `compute_Exch100()` (S^2) | `exchange()` (S^2 and S^inf) — agrees to ~4e-12 |
| `Ind20,r` | `compute_Eind200()` + `chf()` | `induction()` + matrix-free CPHF — agrees to ~5e-14 |
| `Exch-Ind20,r` | `compute_Eexchind200()` | `induction()` — agrees to ~1e-12 |
| **`Disp20`** | **`compute_Edisp200()` (dense N^4)** | **missing** |
| **`Exch-Disp20`** | **`compute_Eexchdisp200()` (dense N^4)** | **missing** |

`grep -c disp src/cqed_scf/sapt/qed_sapt_jk.py` returns `0`. The diagnostics in
`examples/water_methylamine/water_methylamine_qed_sapt_dense_vs_jk.py:331-335`
carry the two dispersion rows from the dense driver, with a comment saying so.
That is honest bookkeeping, but it means **the JK path cannot run a dimer the
dense path cannot afford** — the dense `N^4` build gates every calculation.

### 1.2 Audit findings

Ordered by consequence. Items A–C are the ones I would fix before or during the
dispersion work; D–F are correctness/hygiene items surfaced by the audit.

---

**A. The dense path allocates up to three full `N^4` AO tensors, and one of
them is a materialized rank-one outer product.**

`qed_sapt0.py:278-295`:

```python
I_dimer_standard = np.asarray(shared_mints.ao_eri())   # view, N^4
I_dimer = I_dimer_standard.copy()                      # +N^4
I_dimer_cavity = np.zeros_like(I_dimer)                # +N^4, even when cavity is OFF
if self.include_cavity_terms:
    I_dimer_cavity = self.d_A[:,:,None,None] * self.d_B[None,None,:,:]
    I_dimer += I_dimer_cavity
```

Measured for the water/methylamine reference geometry:

| basis | nbf | naux (RIFIT) | one `N^4` | three `N^4` | DF `naux x nbf^2` |
|---|---:|---:|---:|---:|---:|
| jun-cc-pVDZ | 89 | 293 | 0.50 GB | 1.51 GB | 18 MB |
| aug-cc-pVDZ | 132 | 377 | 2.43 GB | 7.29 GB | 52 MB |
| aug-cc-pVTZ | 299 | 640 | 63.9 GB | 192 GB | 458 MB |

Two separate defects here:

1. `np.zeros_like(I_dimer)` allocates a full `N^4` array **unconditionally**,
   including when `include_cavity_terms=False`, where it is then only ever
   used as a zero. Cheap fix: a lazily-zero sentinel, or restructure
   `_eri_for_context` to return `None` and have callers skip.
2. The cavity tensor is **rank one**. Materializing `d ⊗ d` into `N^4` storage
   costs `nbf^4` doubles to represent `nbf^2` numbers. Every `v()` call on it
   (`qed_sapt0.py:388-396`) then does a four-step `O(N^5)` transform of a
   rank-one object, where two `O(N^3)` matrix products would do:
   `v_cav(s0 s1 s2 s3) = (C_{s0}^T d C_{s2}) ⊗ (C_{s1}^T d C_{s3})`.

This is the single largest performance item in the dense path and it is fixable
independently of the DF work.

---

**B. `compute_Eexchdisp200()` silently consumes state left behind by
`compute_Edisp200()`.**

`self.t_rsab` is written at `qed_sapt0.py:867` and read at `:929` with no
guard. Consequences:

- Calling `compute_Eexchdisp200()` without a prior `compute_Edisp200()` raises
  `AttributeError`, not a useful message.
- The `canonical_denom` flag leaks across the call boundary. `run()`
  (`:1077-1078`) uses the default, but `tests/test_qedsapt0_driver.py:115-116`
  does `compute_Edisp200(canonical_denom=True)` then `compute_Eexchdisp200()` —
  so the pinned `expected_Eexchdisp200` at `:143` is a **canonical-denominator**
  exchange-dispersion value. That test runs at `lambda = [0,0,0]` where the two
  orbital-energy sets genuinely coincide, so the number is correct today. But
  this is precisely the failure mode `psi4_array_aliasing.md` documents: a
  branch exercised only in the regime where it is degenerate with its
  alternative. The moment someone reuses that pattern at finite coupling, the
  coupling becomes silent and wrong.

Fix: make the amplitudes an explicit return value / parameter rather than
instance state, or at minimum record the denominator choice alongside
`t_rsab` and assert consistency.

Related: `compute_Edisp200` also parks `self.v_rsab` and `self.eps_rsab` on the
instance. Each is `n_o^A n_o^B n_v^A n_v^B`, so three of them are held live for
the lifetime of the driver with no consumer after the contraction.

---

**C. `d_A == d_B` is load-bearing and unasserted.**

Verified `0.0` difference (see §0). The rank-one DF augmentation, and arguably
the correctness of `V_A_cavity = -d_exp_el_A * d_B` / `V_B_cavity =
-d_exp_el_B * d_A` (`qed_sapt0.py:307-308`, note the *crossed* labels), depend
on it. Add `assert np.allclose(d_A, d_B)` at the point of use, plus a test that
states *why* — a mismatch would mean the two monomer bases or dipole origins
have diverged, which breaks more than dispersion.

---

**D. The `canonical_denom` physics question is now answerable and should be
settled before dispersion is reimplemented.**

`SAPT_NOTE.md` §2 (2026-09-05) records that the aliasing fix makes the
canonical-vs-CQED denominator comparison meaningful for the first time, and
argues the CQED orbital energies are the consistent production choice. Whatever
is decided, the new implementation should carry **one** default, with
`canonical_denom` clearly marked as a diagnostic. Reimplementing an unresolved
switch into two code paths doubles the surface.

---

**E. The long-range dispersion plateau should be pinned by a test, not left as
a known-oddity note.**

`SAPT_NOTE.md` §3 explains that `I_dimer_cavity` is a bare outer product with
no Coulomb kernel, so the cavity part of `Disp20` tends to a finite non-zero
constant as `R -> inf`. It also proposes the sharp test: the `R -> inf` limit
is computable in closed form from *isolated-monomer* transition dipoles and
excitation energies. That test belongs in the suite as an anchor for the new
implementation — it validates the numerator algebra at a point where no dimer
calculation is needed, and it converts "the curve looks flat" into a number.

---

**F. `components.py` is entirely dead stubs.**

All five functions raise `NotImplementedError`; nothing imports them. The real
implementations live in `qed_sapt0.py` (dense) and `qed_sapt_jk.py` (JK).
`compute_disp20` and `compute_qed_dse_cross` in particular describe an
architecture that was not the one built. Either delete the module or make it
the thin dispatch layer its docstring imagines. Leaving it is a trap for the
next reader, who will reasonably assume `compute_disp20` is where dispersion
goes.

Similarly, `QEDSAPT0Driver.compute_components()` (`:320-327`) raises
`NotImplementedError` while `_build_results()` (`:1027`) does the real work.

---

## 2. Implementation plan

Six phases. Each has a concrete acceptance criterion; phases 1–3 are
prerequisites that also stand on their own.

### Phase 0 — Audit fixes and a validation oracle

*Deliverables*

- Fix **A.1** (unconditional `N^4` zero allocation) and **B** (implicit
  `t_rsab` coupling). Both are small and both make later diffs readable.
- Add the `d_A == d_B` assertion and its explanatory test (**C**).
- Stand up a **`lambda = 0` oracle**: run conventional `psi4.energy('sapt0')`
  on the same dimer and confirm the dense `Disp20`/`Exch-Disp20` reproduce psi4
  within DF error. At `lambda = 0` the theories coincide exactly, so this pins
  the dense reference against an independent implementation *before* we start
  replacing it.

*Acceptance:* existing suite green; new oracle test passes; peak RSS for the
`include_cavity_terms=False` jun-cc-pVDZ case drops by one `N^4`.

**DONE (2026-09-05).** Delivered in `qed_sapt0.py` and `tests/test_dse_df.py`:

- audit item A.1: with the cavity disabled neither the `N^4` zero tensor nor
  the `N^4` copy is allocated (`I_dimer_cavity = None`, `I_dimer is
  I_dimer_standard`), so that case drops from **three** `N^4` tensors to one.
  `v(..., context='cavity')` short-circuits to correctly shaped exact zeros.
  All three stored tensors are marked non-writeable, so the sharing cannot
  become the aliasing defect of `psi4_array_aliasing.md`.
- audit item B: amplitudes moved out of implicit instance state into
  `dispersion_amplitudes()`, with the denominator convention recorded as
  `t_rsab_canonical_denom`. `compute_Eexchdisp200()` now raises a directed
  error when amplitudes are absent and rebuilds rather than silently reusing
  amplitudes from the other convention. The unused `v_rsab`/`eps_rsab`
  instance state is gone.
- audit item C: `d_A == d_B` asserted in `_populate_monomer_attributes()` and
  again in `PauliFierzDF.from_driver()`, both with a message naming the likely
  cause.
- oracle: `test_lambda_zero_dispersion_matches_psi4_sapt0` pins the dense
  `Disp20`/`Exch-Disp20` against conventional `psi4.energy('sapt0')` to
  `3.9e-8` and `2.5e-8` Eh, and `Elst10`/`Exch10(S^2)` to `1e-5`.

### Phase 1 — `PauliFierzDF`: the augmented three-index tensor

New module, `src/cqed_scf/sapt/dse_df.py`, sibling to `dse_jk.py`.

```
B_std^Q_pq   Q = 1..naux    from  (Q|pq) and J^{-1/2}
B_dse^Q_pq   Q = naux+1     = d_pq          (exact, one row)
```

Construction is already verified to work from Python:

```python
aux  = psi4.core.BasisSet.build(mol, "DF_BASIS_SAPT", "", "RIFIT", basis)
zero = psi4.core.BasisSet.zero_ao_basis_set()
Ppq    = np.squeeze(mints.ao_eri(aux, zero, prim, prim))   # note BraKet order
metric = np.squeeze(mints.ao_eri(aux, zero, aux,  zero))
M = psi4.core.Matrix.from_array(metric); M.power(-0.5, 1e-12)
B = np.einsum("PQ,Qpq->Ppq", np.asarray(M), Ppq)
```

(The `(aux, zero, prim, prim)` ordering matters — `(zero, aux, prim, prim)`
raises `Bad BraKet type in Libint2TwoElectronInt`. Round-trip check on water /
cc-pVDZ: `max|DF - exact| = 2.5e-2`, `rms = 2.6e-4`, i.e. ordinary DF error.)

For production, prefer `core.DFHelper` (`add_space` / `add_transformation` /
`get_tensor`) over the explicit `MintsHelper` route — it is threaded, handles
disk/core, and gives MO-basis tensors directly. Keep the explicit route as the
reference implementation the DFHelper path is tested against.

#### The right primitive is 2-character, not 4-character

`v(s0 s1 s2 s3)` pairs orbital positions **(0,2)** and **(1,3)** — the A-side
pair and the B-side pair. In DF form that is exactly:

```
v(s0 s1 s2 s3)[A,B,C,D] = sum_Q  b^Q_(s0 s2)[A,C] * b^Q_(s1 s3)[B,D]
```

So the natural new method takes a **two-character** string and sits alongside
the driver's existing two-character vocabulary — `s(pair)`,
`potential(pair, side)`:

```python
def b(self, pair: str, context: str = "total") -> np.ndarray:
    """DF three-index MO pair block, shape (naux', n_x, n_y).

    context: 'standard' -> Q < naux ; 'cavity' -> Q == naux ; 'total' -> all
    """
```

and `v()` collapses to a derived one-liner:

```python
def v(self, string, context="total"):
    left  = self.b(string[0] + string[2], context)
    right = self.b(string[1] + string[3], context)
    return oe.contract("PAC,PBD->ABCD", left, right, optimize="optimal")
```

**`vt()` needs no change at all.** It already composes `v()`, `s()`,
`potential()` and the constant (`qed_sapt0.py:469-506`), and the one-electron
and constant pieces are `O(N^3)` and already context-partitioned. Only `v()` is
re-plumbed.

*Pair set.* The 17 four-character strings in `qed_sapt0.py` require 10 distinct
pairs — `aa ab ar as ra ba bb br bs sb` — including **cross-monomer** pairs
(`ar`, `bs` are same-monomer; `ab`, `as`, `br`, `sb` are not). Tests also use
`rs`. Support all 16 combinations of `{a,r,b,s}^2`; see the storage note below
for why that is free.

*Storage.* Do **not** cache 16 separate pair blocks. Half-transform the
augmented AO tensor once into the union MO space
`C_all = [Co_A | Cv_A | Co_B | Cv_B]` (`nbf x 2*nmo`), giving a single
`(naux+1, 2*nmo, 2*nmo)` array in which **every pair block is a slice**. For
aug-cc-pVTZ water/methylamine that is `640 x 598^2 x 8 = 1.8 GB` once, versus
tens of GB if the large blocks (`rs`, `sr`) are cached separately — and pair
lookup then costs no arithmetic at all.

*Context slicing.* With the augmented tensor stored once, all three contexts
are views into one array: `standard = B[:naux]`, `cavity = B[naux:]`,
`total = B`. This mirrors `_eri_for_context` exactly. **Caution
(`psi4_array_aliasing.md`):** these are views, so `b()` must be documented
read-only and its consumers must not write in place, or the three contexts
silently couple — the same class of defect as the `eps_canonical` aliasing bug.
Note also that `b(pair, 'cavity')` needs no DF machinery at all: it is just
`(C_x^T d C_y)[None, :, :]`, an `O(N^3)` product.

*Verified.* The factorization was checked against the dense driver for water/He
at cc-pVDZ, `lambda = [0,0,0.1]`, over all 11 strings including cross-monomer
pairs:

| context | max abs error vs dense `v()` |
|---|---|
| standard | `7e-6` to `6e-5` (ordinary DF error on individual MO integrals) |
| **cavity** | **`3.3e-17` — exact, as predicted** |

*Acceptance (the column test, per `SAPT_NOTE.md` §8.1):* for a small dimer,
assemble `v(string)` from `PauliFierzDF` for **every** 4-character string the
dense code uses and compare **elementwise** against `QEDSAPT0Driver.v(string,
context=…)`. Standard blocks agree to DF error; **cavity blocks agree to
machine precision**, because the DSE row is exact. Use a non-symmetric probe so
a transpose convention error cannot pass (§8.2).

**DONE (2026-09-05).** `src/cqed_scf/sapt/dse_df.py` provides
`build_df_ao_tensor()` and `PauliFierzDF` (exported from `cqed_scf.sapt`), with
the `b(pair, context)` primitive and derived `v(string, context)`. Verified over
all 17 driver strings plus `arbs`, on water/He at cc-pVDZ, `lambda = [0,0,0.1]`:

| context | max deviation from dense `v()` |
|---|---|
| standard | `7e-6` … `6e-5` (DF error) |
| **cavity** | **`3.3e-17` — exact** |

`standard + cavity == total` holds to `1e-13` for every string by auxiliary-index
partition. 66 tests in `tests/test_dse_df.py`; full suite 218 passed.

Deviation from the sketch above: `PauliFierzDF` is a standalone object built via
`from_driver()` rather than a driver attribute — wiring it into the driver is
Phase 2, and keeping it separable means Phase 1 could be validated against the
dense path rather than through it.

### Phase 2 — DF-backed `v()` in the dense driver

Give `QEDSAPT0Driver` an `integral_backend="df"` mode — the parameter already
exists in the signature and in `components.py`'s docstrings, so this is
honouring an existing contract rather than inventing one. `v()` routes to
`PauliFierzDF`; `s()`, `potential()` and the `vt()` one-electron/constant pieces
are unchanged (they are `O(N^3)` and already partitioned).

*Why before dispersion:* it exercises the new tensor against **all** existing
pinned component values — `Elst10`, `Exch10`, `Ind20`, `Exch-Ind20` — which is
far more diagnostic than validating it only through two new numbers.

*Acceptance:* every existing dense test passes in `df` mode within DF error
(loosen `atol` to ~1e-6 Eh for standard-dominated terms; keep 1e-9 for the
`lambda = 0` cavity-off controls). Peak memory for aug-cc-pVDZ drops from
~7 GB to <100 MB.

**DONE (2026-09-05).** `integral_backend="df"` implemented. In that mode
`build_integrals()` builds no `N^4` AO tensor at all (`I_dimer` and friends are
`None`, and `_eri_for_context()` raises rather than returning something that
would silently become zeros). `v()` dispatches to `PauliFierzDF`; `s()`,
`potential()` and the `vt()` dressings are untouched, as predicted.

Water/methylamine reference, jun-cc-pVDZ, `lambda = [0,0,0.1]`:

| component | `full_eri` | `df` | delta / Eh | delta / kcal mol^-1 |
|---|---:|---:|---:|---:|
| Elst10 | -0.0086737093 | -0.0086743054 | -6.0e-7 | -3.7e-4 |
| Exch10 | 0.0030805286 | 0.0030819372 | 1.4e-6 | 8.8e-4 |
| Disp20 | -0.0027151107 | -0.0027146006 | 5.1e-7 | 3.2e-4 |
| Exch-Disp20 | 0.0001966590 | 0.0001952441 | -1.4e-6 | -8.9e-4 |
| Ind20,r | -0.0017887767 | -0.0017887293 | 4.7e-8 | 3.0e-5 |
| Exch-Ind20,r | 0.0008167150 | 0.0008166473 | -6.8e-8 | -4.2e-5 |
| **total** | **-0.0090836941** | **-0.0090838066** | **-1.1e-7** | **-7.1e-5** |

Cost for that case: **11.7 s -> 1.1 s** and **2380 MB -> 254 MB** peak
(`nbf = 89`, `naux_scf = 398`, `naux_corr = 293`).

#### Finding: the fitting basis must be chosen by role

The first `df` implementation used one RIFIT tensor for every term and gave an
`Elst10` error of `3.2e-5 Eh` -- two orders of magnitude worse than the other
components. The cause is not a bug in the factorization but a misuse of the
fitting basis. RIFIT is optimized for correlation energies; `Elst10` is a small
residual of large cancelling terms (two-electron Coulomb against nuclear
attraction and nuclear repulsion), and the fitting error in the two-electron
part cancels against nothing. Psi4's own SAPT makes exactly this split, fitting
electrostatics/exchange/induction with JKFIT (`DF_BASIS_ELST`, `DF_BASIS_SCF`)
and only the MP2-like dispersion with RIFIT (`DF_BASIS_SAPT`).

Measured on water/He at cc-pVDZ:

| component | RIFIT error | JKFIT error | ratio |
|---|---:|---:|---:|
| Elst10 | 3.2e-5 | 4.0e-7 | 79x |
| Exch-Ind20,r | 1.1e-7 | 2.0e-9 | 53x |
| Ind20,r | 7.1e-8 | 3.8e-9 | 19x |
| Exch10 | 4.8e-6 | 2.2e-6 | 2.2x |
| Exch-Disp20 | 1.7e-8 | 5.3e-9 | 3.3x |
| Disp20 | 2.0e-8 | 2.2e-8 | 0.9x |

The driver therefore carries **two** fitting roles, `scf` (JKFIT, default) and
`corr` (RIFIT), selected by a `df_role` argument threaded through `v()`,
`vt_parts()` and `vt()`. The dispersion and exchange-dispersion routines pass
`df_role="corr"`; everything else uses `scf`. The `corr` tensor is built lazily,
so a caller wanting only first-order terms never pays for it.

Two tests pin this behaviourally rather than by inspection:
`test_dispersion_uses_the_correlation_fitting_basis` (changing the correlation
basis must move the dispersion terms and leave the Coulomb-like terms bitwise
unchanged) and
`test_correlation_fitting_basis_is_the_wrong_choice_for_electrostatics`, which
asserts that the naive single-RIFIT approach *does* fail, per `SAPT_NOTE.md`
§8.3.

#### Incidental fix

`qed_sapt0.py` called `psi4.core.clean()` on ten argument-validation error
paths without importing `psi4`, so those paths raised `NameError` instead of
the intended exception. The import is now present.

*Tests:* 81 in `tests/test_dse_df.py`; full suite **233 passed**.

### Phase 3 — `Disp20`

`Disp20 = 4 sum t_rsab v_abrs`, with
`v_abrs = sum_Q' b^Q'_ar b^Q'_bs` and `v_rsab` likewise. Straight DF-MP2 shape;
no new algebra. Amplitudes become an explicit return value (Phase 0 item B).

*Acceptance:*
1. matches dense `compute_Edisp200()` within DF error on water/methylamine at
   `lambda = [0,0,0.1]`;
2. at `lambda = 0`, matches psi4's `Disp20` (Phase 0 oracle);
3. `standard + cavity == total` by auxiliary-index partition, to machine
   precision;
4. the `R -> inf` closed-form plateau test (audit item E).

**DONE (2026-09-05),** with acceptance criterion 4 returning a negative result
that is more valuable than the positive one would have been.

Delivered:

- `_dispersion_numerator()` builds only `v('abrs')`. `v('rsab')` is not built
  at all: it is exactly `v('abrs').transpose(2,3,0,1)`, because the three-index
  tensor is symmetric in its orbital pair so `b_(ra) = b_(ar)^T` (and the dense
  backend satisfies the same identity through the eightfold ERI symmetry).
  Verified to `1e-16` in both backends and all three contexts. This halves the
  four-index transform work in `compute_Edisp200` for both backends.
- `compute_Edisp200(context=...)` and `dispersion_amplitudes(context=...)`,
  with the context recorded as `t_rsab_context` and checked by the audit-item-B
  guard alongside `t_rsab_canonical_denom`.
- `dispersion_energy_partition()`: because `Disp20` is *quadratic* in the
  numerator, `standard + cavity` does **not** sum to the total. The correct
  partition is three-way,

  `Disp20 = Disp20[std,std] + 2 Disp20[std,cav] + Disp20[cav,cav]`,

  which sums exactly (`1e-20`). On water/He at 3.4 Ang the cross term
  (`+4.77e-5`) is larger in magnitude than either part it connects
  (`-3.86e-5`, `-2.97e-5`), so this is not a pedantic distinction. A test
  asserts the cross term is non-zero, so the two-way "simplification" cannot be
  introduced later.
- The cavity dispersion is verified against its closed form
  `4 sum (d^A_ar)^2 (d^B_bs)^2 / (e_a + e_b - e_r - e_s)`, agreeing to `3e-21`.
  Both sides are fitting-error free, so this validates the cavity numerator
  algebra with no dimer two-electron integral involved.

### Finding: the dispersion denominators are origin dependent

Audit item E asked for the `R -> inf` plateau test. Running it shows the claim
in `SAPT_NOTE.md` §3 is **half right**, and the other half is a live defect.

**Right:** the cavity dispersion *numerator* has no `R`-dependence. Measured on
water/He, `||v_cav||` is constant to seven significant figures
(`6.148879e-03`) from 8 to 50 Ang, while `||v_std||` decays as expected. The
kernel `d (x) d` carries no Coulomb operator, exactly as
`docs/qed_sapt0_formalism.tex` derives.

**Wrong:** the cavity dispersion *energy* does not plateau. Against the
isolated-monomer closed-form prediction of `-2.9087e-05 Eh`:

| R / Ang | Disp20[cav,cav] / Eh | fraction of prediction |
|---:|---:|---:|
| 5 | -2.4766e-05 | 0.85 |
| 12 | -1.4532e-05 | 0.50 |
| 20 | -7.7209e-06 | 0.27 |
| 50 | -1.5982e-06 | 0.055 |

With the numerator constant, the drift is entirely in the denominators.

**Root cause: CQED-SCF orbital energies are not translation invariant.** For a
single He atom translated along `z` (physically the same system):

| quantity | `lambda = 0` | `lambda = 0.1` |
|---|---|---|
| total CQED-SCF energy | invariant to `1e-15` | **invariant to `4e-14`** |
| HOMO-LUMO gap | invariant to `1e-15` | **grows as `lambda^2 z^2`** |

The gap shift is `89.259 Eh` at `z = 50 Ang` against `lambda^2 z^2 = 89.276`,
i.e. the origin-dependent part of the one-electron DSE (quadrupole) term, and
it moves occupied and virtual orbitals in *opposite* directions. That last
detail points at the exchange-like `N[D]` term, `N = sum d_pr d_qs D_rs`
(`scf.py:219`), which is built from the **bare** dipole matrix rather than the
coherent-state fluctuation `d - <d>`; under a translation `d -> d + Z*lambda*S`
this gains a `Z^2 lambda^2 (S D S)` piece that acts only within the occupied
space. That is a hypothesis consistent with all the evidence, not a confirmed
diagnosis.

**Consequence for QED-SAPT0.** Rigidly translating an entire dimer by 20 Ang
(not a physical change):

| component | `z = 0` | `z = +20 Ang` | invariant? |
|---|---:|---:|---|
| Elst10 | -0.000027062993 | -0.000027062993 | yes |
| Exch10 | 0.000083884825 | 0.000083884824 | yes |
| Ind20,r | -0.000008346771 | -0.000008346771 | yes |
| Exch-Ind20,r | 0.000002763449 | 0.000002763449 | yes |
| **Disp20** | **-0.000020668081** | **-0.000002792226** | **no, factor 7.4** |
| **Exch-Disp20** | **0.000000700884** | **0.000000100875** | **no, factor 7.0** |

Only the terms that use **bare monomer orbital-energy differences** are
affected. Induction is invariant despite also involving orbital energies,
presumably because its coupled response includes the compensating DSE Hessian
term; dispersion divides by the bare difference with nothing to cancel it.

**This predates the DF work**: every measurement above used the dense
`full_eri` backend, and at `lambda = 0` all six components are invariant. It
also means the pinned water/methylamine dispersion values depend on that
geometry's placement relative to the origin.

Captured as `test_dispersion_energy_is_translation_invariant`, marked
`xfail(strict=True)` so it fails loudly if the behaviour changes -- including
if someone fixes it.

**Not fixed here.** The remedy is a physics decision rather than a refactor.
The follow-up investigation below narrows it to two concrete options.

### Follow-up: mechanism confirmed, and two candidate fixes

*Prompted by Jay's proposal to compute the monomer references with each
monomer's centre of mass at the origin. Reproduce with
`docs/development/probe_translation_invariance.py`.*

#### Mechanism (confirmed analytically and numerically)

Under a rigid translation by `T` along the polarization direction, the AO
dipole matrix transforms as `d -> d + T*lambda*S`. Two Fock terms respond:

- `Q_PF = -1/2 lambda^2 Q` (`scf.py:159`) picks up a term proportional to `S`,
  shifting **every** orbital by `+1/2 lambda^2 T^2`.
- `N[D] = sum_rs d_pr d_qs D_rs` (`scf.py:219`) picks up
  `T^2 lambda^2 (S D S)`, which is supported **only on the occupied space**, so
  `-N` shifts **occupied** orbitals by `-lambda^2 T^2`.

Net: occupied `-1/2 lambda^2 T^2`, virtual `+1/2 lambda^2 T^2`, gap
`+lambda^2 T^2`. Measured for water at `T = 20 Ang`, `lambda = 0.1`: occupied
shift `-7.142242` against `-1/2 lambda^2 T^2 = -7.142130`; virtual shifts
`+7.15` to `+7.61`. The density is invariant throughout (`7.8e-12`), so the
occupied and virtual **subspaces** are correct -- it is the canonical rotation
*within* each subspace, and the eigenvalues, that move.

#### Not only the orbital energies: the coefficients move too

Comparing `C_a^T S C_b` between frames (which must be `+/- 1` on the diagonal
and zero off it if the orbitals agree):

| quantity | `lambda = 0` | `lambda = 0.1` |
|---|---|---|
| total CQED-SCF energy | invariant `1e-15` | invariant `4e-14` |
| density `D` | invariant `1e-13` | invariant `7.8e-12` |
| coefficients `C` | invariant `1.7e-13` | **off-diagonal `1.00`** |
| orbital energies | invariant `2.4e-13` | **shift `7.6`** |

This matters for the proposed fix: MP2-like dispersion with diagonal
denominators is only valid in the canonical basis, so coefficients from one
frame cannot be combined with orbital energies from another.

#### Option 1: intrinsic per-monomer frame (Jay's proposal)

Running each monomer's CQED-SCF with its own centre of mass at the origin makes
every monomer quantity independent of where the dimer sits, so translation
invariance is restored **by construction** -- no test needed. The MO
coefficients transfer correctly into the dimer frame, because a translation
does not change coefficients relative to translated basis functions.

Two constraints:

- **Do not rebuild `d_A`/`d_B` in the per-monomer frames.** Monomers A and B
  would use *different* frames, so their dipole matrices would differ by
  `(z_A - z_B) lambda S`. That breaks `d_A == d_B`, and with it the rank-one
  auxiliary-row factorization of Phases 1-2, as well as the interaction
  operator itself. All interaction quantities (`S_dimer`, `d`, the ERIs) must
  stay in one shared dimer frame.
- **The centre choice is physical, not a gauge choice.** The residual origin
  dependence enters through `<z^2>`, so centre of mass, centre of nuclear
  charge and centroid of electronic charge give *different* answers. Option 1
  picks a convention rather than removing one.

On the dimer frame: Jay's reading is right, and it can be confirmed from the
code. `shared_mints` is used only for `ao_overlap()` and `ao_eri()`, both
translation invariant, and the interaction DSE operator is the *fluctuation*
product `(d_A - <d>_A)(d_B - <d>_B)`, whose origin dependence cancels -- which
is precisely why `Elst10`, `Exch10`, `Ind20,r` and `Exch-Ind20,r` are already
invariant. Centering the dimer is harmless but unnecessary.

#### Option 2: build the DSE Fock terms from the fluctuation operator

Frame-free, and **verified to work**. Define

```text
c  = <d>_el / N_el                (shifts by T*lambda under translation)
d~ = d - c S                      (manifestly translation invariant)
```

and build *both* DSE Fock terms from `d~`: use `d~` in `N[D]`, and correct the
one-electron term to `H = H0 + Q_PF - (c d - 1/2 c^2 S)`, which is
`-1/2 (d~^2)_AO` expanded.

Prototype result (water, cc-pVDZ, `lambda = 0.1`, translation `0 -> 20 Ang`):

| Fock construction | max abs `dD` | max abs `d eps` | HOMO-LUMO gap |
|---|---:|---:|---|
| bare `d` (current `scf.py`) | 3.2e-10 | **7.61** | 0.69545907 -> 15.27376202 |
| fluctuation `d~`, correct sign | 1.8e-13 | **2.6e-13** | 0.69367500 -> 0.69367500 |
| fluctuation `d~`, wrong sign | 4.9e-1 | 14.9 | converges elsewhere |

Two caveats:

- **This changes the answer at the origin too** (`gap 0.69545907 ->
  0.69367500`). It is a different definition of the DSE Fock operator, not a
  gauge fix, so every pinned reference value would move.
- The prototype validates the *Fock* only. Whether the total-energy expression
  stays consistent under the same substitution was not checked, and the sign
  had to be pinned empirically -- the wrong sign converges to a different
  solution, which is exactly the trap `SAPT_NOTE.md` §7 warns about.

#### Option 3: strong-coupling QED-HF

Riso, Haugland, Ronca and Koch, *Nat. Commun.* **13**, 1368 (2022) treat origin
dependence of the QED-HF reference as a structural defect and remove it by
making the coherent-state transformation orbital-dependent and optimising it
with the orbitals. That fixes the orbitals and orbital energies themselves, so
every downstream quantity is fixed at once rather than term by term. Most
principled, most invasive: it replaces the reference, so all pinned energies
would need re-establishing.

#### Status

**Option 1 is implemented, tested and opt-in** (below). Options 2 and 3 remain
under consideration; all three are compared in
`docs/qed_sapt0_formalism.tex` §"Origin dependence of the reference".

### Option 1 implemented: `monomer_reference_frame`

New driver field, default `"dimer"` (historical behaviour, bitwise unchanged).
Setting `"monomer_com"` translates each ghosted monomer so its own real-atom
centre of mass is at the origin before solving its CQED-SCF reference.

The implementation turns on one distinction, which is the real content:

> A block whose four orbital indices all belong to **one** monomer is an
> **internal** quantity and must be built in that monomer's intrinsic frame.
> Every block mixing the monomers is an **interaction** quantity and must be
> built in the shared dimer frame.

Both halves are load-bearing, and getting the second one wrong is the
interesting part of this work:

- *Interaction side.* All interaction integrals (`ao_overlap`, `ao_eri`,
  `ao_dipole`, `ao_potential`) are rebuilt in the dimer frame from
  `BasisSet.build()` on the dimer-frame ghosted molecules -- verified **bitwise
  identical** to the monomer wavefunction's own MintsHelper in the default
  case, so there is one code path. `d_A`, `d_B`, `d_exp_el`, `d_nuc` and
  `d_exp` are likewise rebased. This keeps `d_A == d_B` exactly, without which
  the rank-one auxiliary-row factorization of Phases 1-2 would break.
- *Internal side.* The CPHF matrix is a monomer's **own** orbital Hessian. Its
  diagonal carries intrinsic-frame orbital energies, while its DSE
  exchange-like term `-d_bb' d_ss'` uses the occ-occ and virt-virt blocks of
  `d`, which are **not** translation invariant (the occ-virt blocks are, since
  `S_bs = 0`). Induction is origin independent only because those two pieces
  cancel. A first version of this implementation supplied intrinsic-frame
  orbital energies with a dimer-frame Hessian, which **broke induction**:
  `Ind20,r` moved by `1.1e-5 Eh` and changed sign under a 20 Ang translation.
  With `v(..., frame=...)` routing the Hessian blocks to the matching frame,
  induction returns *exactly* to its dimer-frame value.

Also delivered incidentally: the cavity contribution to `v()` is now formed
from the rank-one MO product rather than a materialized AO tensor, which is
audit item **A.2** from the original audit.

#### Results

Change under a rigid 20 Ang translation of the whole dimer:

| component | `dimer` frame | `monomer_com` frame |
|---|---:|---:|
| Elst10 | 4.7e-14 | 4.5e-14 |
| Exch10 | 5.6e-14 | 8.0e-17 |
| **Disp20** | **1.8e-5** | **1.0e-18** |
| **Exch-Disp20** | **6.0e-7** | **4.6e-18** |
| Ind20,r | 7.1e-15 | 2.0e-17 |
| Exch-Ind20,r | 8.9e-15 | 1.8e-17 |

`Disp20[cav,cav]` now plateaus at `-2.909131200e-05 Eh`, constant to eleven
significant figures from `R = 8` to `50 Ang`, against the isolated-monomer
closed-form prediction `-2.9087e-05 Eh` (the residual is the ghosted basis).
**Audit item E is finally answered in the affirmative.**

Only the dispersion terms change value; `Elst10`, `Exch10`, `Ind20,r` and
`Exch-Ind20,r` are reproduced exactly, which is the sharpest available check
that the frame split is right.

#### Caveats

- **Convention, not a cure.** The residual dependence enters through `<z^2>`,
  so centre of mass, centre of nuclear charge and centroid of electronic charge
  give different answers. Centre of mass is a *mass*-weighted choice in a
  problem with no mass dependence.
- **The JK path (`qed_sapt_jk.py`) is not covered.** It consumes
  `monomer.wfn`, which under `monomer_com` sits in the intrinsic frame. Do not
  combine `monomer_reference_frame="monomer_com"` with `build_sapt_jk_cache()`
  until that path receives the same internal/interaction split.

*Tests:* 116 passed + 2 xfailed in `tests/test_dse_df.py`; full suite
**268 passed, 2 xfailed**.

*Tests:* 98 passed + 2 xfailed in `tests/test_dse_df.py`; full suite
**250 passed, 2 xfailed**.

### Phase 4 — `Exch-Disp20`

The heavy one. The dense `compute_Eexchdisp200()` (`:871-930`) needs nine `vt`
blocks — `abar abra absb abbs abas abrb abab abrs absr` — each an
`o_A o_B v_A v_B`-shaped object built from `v()` plus `S ⊗ V` dressings. With
`v()` DF-factorized, none of them requires an AO `N^4` tensor.

Two sub-decisions to make with numbers in hand:

- **Materialize or factorize.** The `h`/`q` intermediates are
  `o_A o_B v_A v_B`, which for aug-cc-pVTZ water/methylamine is large but not
  prohibitive. Materializing keeps the algebra a line-by-line transcription of
  the validated dense code — strongly preferred for the first pass. Optimize
  only against a profile.
- **Transcribe, do not re-derive.** The dense expression is already validated
  against Kenny's/Konrad's reference. Port it literally; the only change is how
  `v()` is evaluated.

*Acceptance:* matches dense `compute_Eexchdisp200()` within DF error; matches
psi4's `Exch-Disp20,u` at `lambda = 0`; `standard + cavity == total`.

### Phase 5 — Wire into the JK path and retire the carried rows

Add `dispersion(cache, …)` to `qed_sapt_jk.py`, fed from the same
`build_sapt_jk_cache()` (which already holds `Cocc`/`Cvir`/`eps`/`S`/`V_A`/
`V_B`/`dse_constant`). Delete the carried-from-dense rows in
`water_methylamine_qed_sapt_dense_vs_jk.py:331-335` and the disclaimer at
`:526`. Extend `print_sapt_summary` accordingly.

*Acceptance:* the dense-vs-JK diagnostic prints all six components from the JK
path, with the `--lambda-vector 0 0 0` and `--no-cavity-terms` stress cases
still at roundoff for the four already-converged terms and within DF error for
the two new ones. Update `docs/SAPT_NOTE.md` with the final table, in the
established format.

### Phase 6 — Consolidation

- Promote the separable-kernel statement into a shared module docstring
  (`SAPT_NOTE.md` §4 already asks for this): `dse_jk.py` and `dse_df.py` are
  the J/K and DF faces of the same operator, `(pq|rs) -> d_pq d_rs`.
- Resolve `components.py` (audit item F) — delete or make real.
- Record the `sapt_mp2_terms.py` finding in `SAPT_NOTE.md` so the next reader
  does not re-investigate that route.

---

## 3. Risks

| risk | mitigation |
|---|---|
| DF error masks a real bug in the cavity terms | cavity blocks are **exact** under the augmentation — hold them to machine precision, and only the standard part to DF error. This is a sharper test than DF work usually gets. |
| `d_A != d_B` in some geometry/frame we have not tried | assert at point of use (audit item C); the assertion, not a silent wrong answer, is the failure mode. |
| `Exch-Disp20` intermediates blow up memory for large dimers | measure first; the dense code's `o_A o_B v_A v_B` intermediates are already the same shape, so this is not a regression. |
| Psi4 `Hx`/DF sign and ordering conventions | `SAPT_NOTE.md` §7: pin every psi4 call against an independently constructed reference. The `(aux, zero, prim, prim)` BraKet ordering above is one instance already found. |
| Re-deriving validated algebra introduces new errors | Phase 4 is a literal transcription of dense code that is already pinned against an external reference. |

## 4. Resolved questions

**`canonical_denom`** (audit item D) — resolved, see §0.1 D2. CQED orbital
energies are the production denominator; the canonical option is retained as a
diagnostic. The amplitude routine therefore keeps the flag, but the flag must
be recorded alongside the amplitudes it produced (audit item B) so the choice
cannot leak silently into `Exch-Disp20`.

**Whether to use psi4's SAPT MP2 machinery** — resolved, see §0.1 D1. No.
Conventional `psi4.energy('sapt0')` is retained only as a `lambda = 0` oracle;
`FISAPT` is functional-group SAPT and is not used at all.
