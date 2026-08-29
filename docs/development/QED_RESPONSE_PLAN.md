# QED-CIS / QED-TDHF / QED-LR-TDDFT Implementation Plan

Target module: `src/cqed_scf/response.py` (+ a new `src/cqed_scf/davidson.py`)
Oracle: `RESPONSE_REFERENCE/CS_CQED_CIS.py` + `helper_CS_CQED_CIS.py` (QED-CIS-1, MgH+/cc-pVDZ)

---

## 0. What the source inspection established

### 0.1 `scf.py` and `helper_CQED_RHF.py` are the same theory, algebraically simplified

This matters because the oracle's CIS equations are written in terms of the *reference's* Fock
operator and orbital energies. They transfer to `cqed_scf` only if the two Fock operators are
identical. They are.

| | reference `helper_CQED_RHF.py` | package `scf.py` |
|---|---|---|
| core | `H = H0 + Q_PF + d_PF` | `H = H0 + Q_PF` |
| Fock | `F = H + 2J - K + 2M - N` (M, N folded into `I`) | `F = H + 2J - K - N` |
| energy | `Tr[(F+H)D] + Enuc + d_c` | `Tr[(F+H)D] + Enuc` |

with, writing `d = λ·μ_el` (AO), `x = λ·⟨μ_el⟩ = 2 Tr(dD)`, `n = λ·μ_nuc`:

```
M_pq   = d_pq Tr(dD) = d_pq (x/2)      →  2M   = x·d
d_PF   = (n - x - n) d = -x·d          →  d_PF = -x·d
                                          2M + d_PF = 0     (Fock identity)

Tr[2M D]    = +x²/2
Tr[2 d_PF D] = -x²
d_c          = ½n² - n(x+n) + ½(x+n)² = +x²/2
                                          sum = 0            (energy identity)
```

**Conclusion.** Identical Fock matrix → identical orbitals `C`, identical `eps`, identical total
energy. `scf.py` is a clean coherent-state formulation with the mean-field DSE (`2M`) and the
one-electron coherent-state shift (`d_PF`) analytically cancelled, leaving only the exchange-like
DSE term `-N`. Nothing needs to change in the SCF physics; **Tier 0 asserts this numerically** so
the assumption never silently rots.

### 0.2 One real hygiene bug blocks tight agreement

`CQEDSCF.run()` breaks out of the SCF loop *before* re-diagonalizing:

```
build F(D)  →  check convergence  →  break        # eps is from the PREVIOUS iteration
            →  DIIS extrapolate F →  diagonalize  # ...of an extrapolated F
```

So `results["orbital_energies"]` are eigenvalues of the previous, DIIS-extrapolated Fock — not of
the returned `results["F"]`. Every CIS working equation assumes canonical CQED-RHF orbitals
(`F_ij = δ_ij ε_i`, and in particular `F_ia = 0`, which is what makes the `⟨Φ0|H|Φ_i^a⟩` block
vanish). The error is O(d_convergence), but it is exactly the kind of thing that caps agreement
with the oracle at 1e-7 and costs a day to find. Fix it in Tier 0 with a final canonicalizing
diagonalization plus an `|F_ia|` diagnostic.

### 0.3 The reference's basis ordering is the main structural problem

```python
ias = 2 * (i * nvirt + a) + s + 2      # photon index fastest, Φ0 states bolted on at 0 and 1
```

Photon-minor interleaving has three costs:

1. **No exploitable block structure.** The electronic CIS matrix — which is *identical* in every
   Fock state — is scattered across alternating rows and columns. You cannot slice it, cannot
   reuse it, cannot hand it to a sigma routine.
2. **It forces the six-deep Python loop** over `i,a,s,j,b,t`. That loop is the reason the
   reference is unusable beyond a diatomic in a double-zeta basis.
3. **It hard-codes two Fock states.** Going to `N_ph > 1` is an index rewrite, not a parameter.

### 0.4 Two orbital bases — the A block cannot assume canonical orbitals

`helper_CS_CQED_CIS.py` builds `H1e` as `eps_v[a]·δ − eps_o[i]·δ`, which is only valid because the
CQED-RHF Fock matrix is diagonal in its own orbitals. The more general
[`qed-ci` / `helper_PFCI.py`](https://github.com/mapol-chem/qed-ci/blob/5dd08a2a2354/src/helper_PFCI.py)
drops that assumption: it builds every element by Slater-Condon rules over a determinant basis
(`calcApDMatrixElement` → `calcMatrixElementIdentialDet` / `DiffIn1` / `DiffIn2`), so canonical HF
orbitals and CQED-HF orbitals are both **admissible** — meaning computable, *not* equivalent.
**Truncated CI is not orbital invariant**, and this is a physical feature of the method, not a
defect to be tested away. See §0.5.

The generalization is cheap. Let `F` be the Pauli-Fierz Fock matrix expressed in whichever MO basis
is chosen, blocked `F_oo`, `F_ov`, `F_vv`. Then:

```
A_ia,jb              = F_ab δ_ij − F_ij δ_ab + 2(ia|jb) − (ij|ab) + 2 d_ia d_jb − d_ij d_ab
⟨Φ_0,n|H|Φ_i^a,n⟩    = √2 · F_ia                       ← no longer zero
⟨Φ_0,n|H|Φ_0,n⟩      = (E_ref − E_CQED-RHF) + nω
```

Canonical CQED-HF orbitals are the special case `F_oo = diag(ε_o)`, `F_vv = diag(ε_v)`, `F_ov = 0`,
which collapses this back to the oracle's equations exactly. Two consequences:

- **In the sigma build, `(ε_a − ε_i)X_ia` becomes `X F_vv − F_oo X`** — one extra pair of small
  matrix multiplies, still O(n_o²n_v + n_o n_v²). Supporting both bases is nearly free.
- **The Brillouin block is live.** The `⟨Φ_0|H|Φ_i^a⟩` cells that are structural zeros in the
  CQED-HF layout carry `√2 F_ia` in the canonical-HF layout. The block-tridiagonal *photon*
  structure is untouched; only that one row/column per photon block changes. Keep the
  `max|F_ia|` diagnostic from §0.2 as an assertion in the CQED-HF path and as a reported quantity
  in the canonical path.

### 0.5 The four basis combinations, and why the spectra legitimately differ

Two switches are in play, and `qed-ci` currently ties them together:

```python
if self.canonical_mos or self.photon_number_basis:
    self.canonical_mos = True
    self.photon_number_basis = True
    self.coherent_state_basis = False
```

The pairing has real parity behind it: CQED-HF is *solved* under a coherent-state transformation,
so pairing it with a CS-transformed CI Hamiltonian is self-consistent, and canonical MOs (no CS
transformation) pair with an untransformed CI Hamiltonian. The displacement enters through
`d_exp`: CS uses `d_exp = ⟨d_el⟩` with `d_c = ω d_exp²/2`; the number basis uses `d_exp = d_nuc`
with `d_c = −ω d_nuc²/2`. **Exposing all four combinations is the right target** (and is on the
`qed-ci` refactor list too), with the paired ones as defaults.

**Why the four do not agree — two independent sources, with different limiting behavior:**

| source | mechanism | vanishes when |
|---|---|---|
| **electronic** | CIS is invariant under occ-occ and virt-virt rotations but **not** under occ-virt rotations. CQED-HF differs from canonical HF by exactly such a rotation — that rotation *is* the cavity relaxation. So `{Φ_0} ∪ singles` spans a genuinely different space in each basis. | the CI space is closed under the rotation — i.e. FCI, or CAS within its active space. **Never for CIS.** |
| **photonic** | The CS displacement operator mixes photon number states through an infinite expansion, so truncating at `N_ph` in the CS basis is not the same truncation as at `N_ph` in the number basis. | `N_ph → ∞`. **Converges.** |

This is why the four combinations are diagnostically valuable rather than redundant: **each mixed
pair isolates one source.**

- *(CQED-HF, CS)* vs *(CQED-HF, number)* — photon truncation only. The difference **must decay
  monotonically with `N_ph`**. A real convergence test with a real signal.
- *(CQED-HF, CS)* vs *(canonical HF, CS)* — orbital rotation only. The difference **persists at
  every `N_ph`**. It quantifies the cavity orbital relaxation that singles cannot recover, which is
  a publishable quantity, not an error.
- *(CQED-HF, CS)* vs *(canonical HF, number)* — both at once. This is the `qed-ci` pair, and it is
  the one combination from which you can learn the least.

Note there is **no a priori ordering** between the two orbital bases: each is a variational upper
bound within its own space, and the spaces are not nested. Write that down so nobody later "fixes"
a result that comes out the wrong way.

### 0.6 Reusable machinery already in the tree

`src/cqed_scf/sapt/dse_jk.py` already contains `DSEJK` (separable `J_DSE`/`K_DSE` from an AO
density, no four-index tensor), `PauliFierzJK` (adapter summing native ERI JK + DSE JK), and
`DSECPHF` (matrix-free DSE Hessian action), with conventions documented in
`docs/development/dse_jk_cphf_design.md` and tested in `tests/test_dse_jk_scaffold.py`.
This is exactly the DSE sigma machinery response theory needs. **Promote it from `sapt/` to a
shared home rather than reimplementing it.**

---

## 1. Proposed basis ordering: photon-major, reference-leading

```
index(n, p) = n · (1 + n_ov) + p
    p = 0                    →  |Φ_0, n⟩
    p = 1 + i·n_virt + a     →  |Φ_i^a, n⟩          n = 0 … N_ph
```

The Hamiltonian becomes **block-tridiagonal in photon number**, with only **two unique blocks**
for the whole method:

```
        n=0                n=1                n=2
    ┌───────────────┬───────────────┬───────────────┐
n=0 │  A_el + 0ω    │   √1 · Ĝ      │      0        │
    ├───────────────┼───────────────┼───────────────┤
n=1 │  √1 · Ĝ†      │  A_el + 1ω    │   √2 · Ĝ      │
    ├───────────────┼───────────────┼───────────────┤
n=2 │      0        │  √2 · Ĝ†      │  A_el + 2ω    │
    └───────────────┴───────────────┴───────────────┘
```

`A_el` (electronic, photon-independent) and `Ĝ` (one-body coupling, photon-independent) are built
once. Every photon block reuses them; the `√(n+1)` prefactors are scalars. **QED-CIS-N costs the
same code as QED-CIS-1.** Within each block the reference determinant leads, so the electronic
CIS sub-block is a contiguous `(n_ov × n_ov)` slice — sliceable, testable against plain CIS, and
directly consumable by a sigma routine.

### Amplitude container

Store a trial vector as `X` of shape `(N_ph+1, 1 + n_occ·n_virt)`, with helper views
`X.c[n] → scalar` (photonic reference amplitude) and `X.t[n] → (n_occ, n_virt)`. Davidson still
operates on the flattened view; the sigma routine operates on the structured one. No index
arithmetic escapes the container.

---

## 2. Working equations in the new layout

`d` = `λ·μ_el` in the chosen MO basis, blocked as `d_oo`, `d_ov`, `d_vv`. `g = √(ω/2)`.

**Energy zero — a subtlety worth pinning down.** The oracle shifts by `E_CQED-RHF`
(`Hp[0,0] = 0`), so its eigenvalues are energies *relative to the CQED-RHF reference*, not
excitation energies. The ground root is **not** zero once `λ ≠ 0`: the bilinear term couples
`|Φ_0,0⟩` to `|Φ_i^a,1⟩` and relaxes the ground state below the reference. Cross-checking the two
oracles on MgH⁺/cc-pVDZ at 2.2 Å, `ω = 4.75 eV`, `λ = (0,0,0.0125)`:

```
qed-ci totals      E_g  = -199.86358254419457
                   E_LP = -199.69776087489558      →  E_LP − E_g = 0.16582167   (true ω_LP)
helper eigenvalue  [1]  =    0.16557084            →  E_LP − E_CQED-RHF
                                                      implies E_CQED-RHF = -199.86333171
                                                      and    E_g − E_CQED-RHF = -2.5083e-4
```

So `CS_CQED_CIS.py`'s printed "|X,0⟩ → |LP⟩ energy" is `E_LP − E_CQED-RHF`, low by 2.5e-4 Eh
against the true excitation energy. **Define `ω_k = E_k − E_0` explicitly in the API** and report
both quantities, or this discrepancy will resurface as a phantom bug when the two oracles are
compared. `E_CQED-RHF = -199.86333171` is also a free Tier-0 check on `CQEDSCF` itself.

| block | element | note |
|---|---|---|
| `⟨Φ_0,n\|H\|Φ_0,n⟩` | `(E_ref − E_CQED-RHF) + nω` | photon ladder |
| `⟨Φ_0,n\|H\|Φ_0,n±1⟩` | `0` | coherent-state basis removes `⟨d⟩` |
| `⟨Φ_0,n\|H\|Φ_i^a,n⟩` | `√2 · F_ia` | vanishes **only** for CQED-HF orbitals (§0.4) |
| `⟨Φ_i^a,n\|H\|Φ_j^b,n⟩` | `F_ab δ_ij − F_ij δ_ab + 2(ia\|jb) − (ij\|ab) + 2 d_ia d_jb − d_ij d_ab + nω` | `A_el + nω` |
| `⟨Φ_0,n+1\|H\|Φ_i^a,n⟩` | `−√((n+1)ω) · d_ia` | `= −√(n+1)·√2·g·d_ia` |
| `⟨Φ_i^a,n+1\|H\|Φ_j^b,n⟩` | `√(n+1) · g · (δ_ab d_ij − δ_ij d_ab)` | `= √(n+1)·g·Ĝ` |

Sign convention: the bilinear coupling operator is `−g (λ·μ_el − ⟨λ·μ_el⟩)(b + b†)`. Both
off-diagonal forms follow from it, and both reduce to the reference's `Hep` at `n = 0`.

### Sigma (matrix-free), per photon block

```
σ[n]  =  (A_el + nω)·X[n]  +  √(n+1)·g·Ĝ·X[n+1]  +  √n·g·Ĝ†·X[n−1]
```

**Electronic block `A_el·X` — three pieces:**

1. *Fock* — `X F_vv − F_oo X`, plus the Brillouin terms `σ.t[n] += √2 F_ov · X.c[n]` and
   `σ.c[n] += √2 ⟨F_ov, X.t[n]⟩`. With canonical CQED-HF orbitals this degenerates to
   `(ε_a − ε_i) X_ia` and the Brillouin terms vanish; with canonical HF orbitals it costs two
   small matrix multiplies. **Write it in the general form from the start** — the specialization is
   a runtime branch, not a separate code path.
2. *ERI* — AO route, one JK build per photon block:
   ```
   D^X = C_o X C_v^T                       (C_left = C_o, C_right = C_v X^T)
   σ  += C_o^T (2 J[D^X] − K[D^X]) C_v
   ```
3. *DSE* — **no JK build required.** The dipole operator is separable, so
   ```
   σ  += 2 d_ov · ⟨d_ov, X⟩  −  d_oo X d_vv
   ```
   Cost `O(n_o²n_v + n_o n_v²)`. (Equivalent to `2 J_DSE − K_DSE` in MO; route it through
   `DSEJK` only if uniformity is worth the AO transform, which for CIS it is not.)

**Coupling — all `O(n_o²n_v + n_o n_v²)` or less:**

```
σ.t[n+1] += √(n+1)·g·(d_oo X.t[n] − X.t[n] d_vv)          # and the adjoint into σ.t[n]
σ.t[n+1] += −√((n+1)ω)·d_ov · X.c[n]
σ.c[n]   += −√((n+1)ω)·⟨d_ov, X.t[n+1]⟩
```

**Cost claim.** One Davidson iteration = `(N_ph+1) × ` one JK build. Everything cavity-specific is
`O(N³)` or cheaper. Compare the reference: `O((2 n_ov)²)` complex storage, an `O(n_ov²)` Python
loop to fill it, and an `O(n_ov³)` full diagonalization.

### Davidson design notes

- **Hermitian only**, as requested — symmetric Davidson-Liu, block, real arithmetic.
- **Preconditioner**: `H_diag[n, ia] = (ε_a − ε_i) + nω + 2d_ia² − d_ii d_vv[a,a]`;
  `H_diag[n, 0] = nω`.
- **Guess must include photonic seeds.** Take the union of (k lowest electronic diagonals per
  photon block) and (all `|Φ_0, n⟩` for `n ≥ 1`). Without the latter, roots with dominant photon
  character are missed or converge glacially — this is the single most common failure mode in
  polaritonic Davidson implementations.
- **Degeneracy**: at `λ = 0` every root is `(N_ph+1)`-fold degenerate. Use a block size at least
  as large as the expected degeneracy, double Gram-Schmidt, and a tight linear-dependence drop
  threshold (~1e-6).
- **Tight thresholds**: LP/UP splitting is `∝ λ` and can be ~1e-4 Eh. Converge residuals to
  1e-6–1e-8, not the 1e-4 typical of TDDFT defaults.

---

## 3. Tiers

### Tier 0 — Oracle harness and SCF hygiene *(no new physics)*

**Deliverables**
- `tests/data/qed_cis_reference.json` — **two oracles, not one.**
  - `CS_CQED_CIS.py` (CQED-HF orbitals, coherent-state basis), relative to `E_CQED-RHF`.
  - `qed-ci` / `helper_PFCI.py` totals, which pin the absolute scale and cover the canonical-HF
    path. Known values for MgH⁺/cc-pVDZ at 2.2 Å:

    | case | root | energy (Eh) |
    |---|---|---|
    | λ=0, ω=0 | g | −199.8639591041915 |
    | λ=0, ω=0 | 1st singlet (root 4) | −199.6901102832973 |
    | λ=(0,0,0.0125), ω=4.75 eV | g | −199.86358254419457 |
    | " | LP (root 2) | −199.69776087489558 |
    | " | UP (root 5) | −199.68066502792058 |

  These are all *(CQED-HF orbitals, coherent-state basis)*. **No trustworthy reference values exist
  for the canonical-HF path** — the `qed-ci` canonical-MO test asserts nothing (§ Tier 1) — so ours
  will be self-generated and locked in once the λ=0 agreement test of Tier 1 passes.

  Plus one fast case (LiH or H2/STO-3G) for CI-speed tests.
- `tests/test_scf_matches_cqed_rhf_reference.py` — assert `CQEDSCF` total energy, `eps`, and `|C|`
  match `helper_CQED_RHF.cqed_rhf`; assert the `2M + d_PF = 0` and `Tr[2M D] + Tr[2 d_PF D] + d_c = 0`
  identities of §0.1 numerically.
- **`scf.py` fix**: final canonicalizing diagonalization of the converged, un-extrapolated `F`;
  return `eps`/`C` consistent with it; add a debug `max|F_ia|` diagnostic.
- **Results-dict contract**: add `omega` and `lambda_vector` (currently absent — response cannot
  reconstruct the Hamiltonian from the dict alone). Optionally cache `d_mo`.

**Exit criterion** — `CQEDSCF` reproduces `cqed_rhf` to 1e-10 Eh and `eps` to 1e-9; existing SCF
regressions still pass.

---

### Tier 1 — Dense QED-CIS-1 in the new layout *(correctness anchor)*

**Deliverables**
- `QEDCIS` in `response.py`: `build_dense_hamiltonian()` + `numpy.linalg.eigh`.
- Fully vectorized construction — `np.einsum`/`opt_einsum` over MO integrals from
  `mints.mo_eri(Co, Cv, Co, Cv)` and `mints.mo_eri(Co, Co, Cv, Cv)`. Zero Python loops over `i,a`.
- `n_photon` parameter honored from the start (block-tridiagonal assembly).

**Purpose**: readable, obviously-correct, and the internal oracle every later tier is tested
against. All sign and factor conventions get nailed here, once.

**Tests**
- Reproduces the stored oracle eigenvalues to 1e-8.
- **Permutation-similarity test**: build the reference's interleaved matrix and the new
  block-ordered matrix for the same system; assert identical spectra. This proves the reordering
  is a relabeling, not a change of theory.
- **Basis-dependence characterization** (replaces the naive invariance test, which would be wrong
  — see §0.5). Three separate assertions, each with a genuine expected answer:
  1. **λ = 0 → all four combinations agree exactly** (1e-10). At zero coupling CQED-HF reduces to
     RHF, `d = 0`, and the CS displacement is the identity, so every source of difference
     switches off simultaneously. This is the strong invariance test, and it exercises the general
     Fock path and the live Brillouin block without requiring them to be zero.
  2. **Photon-basis convergence** — hold `orbital_basis="cqed_hf"`, sweep `photon_basis` over
     `{coherent_state, number}` at `N_ph = 1, 2, 4, 8`; assert the difference decays monotonically.
  3. **Orbital-basis gap persists** — hold `photon_basis="coherent_state"`, compare
     `cqed_hf` and `canonical_hf` across the same `N_ph` sweep; assert the difference does *not*
     decay, and store the observed value as a documented reference rather than as an error bound.

  `qed-ci`'s `test_mghp_qed_cis_with_cavity_canonical_mo` was written as an anti-test — it was
  *expected* to fail its asserts, and was left in the suite with the asserts stripped and a bare
  `pass`. Do not port it. Its numbers are a record of the expected *disagreement*.
- `λ = 0` → spectrum is the plain-CIS spectrum, each root `(N_ph+1)`-fold degenerate; lowest root
  matches `psi4 tdscf_excitations(..., tda=True)`.
- `ω = 0, λ = 0` → exactly canonical CIS.
- `ignore_dse_terms` and `ignore_coupling` switches reproduce the corresponding `qed-ci` limits.

**Exit criterion** — bit-comparable to the oracle; the dense builder becomes `_reference_hamiltonian`
for later tiers, marked as O(N⁴)-memory and test-only.

---

### Tier 2 — Matrix-free sigma + Davidson *(the performance tier)*

**Deliverables**
- `QEDCISSigma`: JK-backed ERI action + analytic MO-block DSE action + coupling action, per §2.
  Uses `psi4.core.JK` with `C_left`/`C_right` (generalized densities) — note `scf.py`'s existing
  `_build_JK` is `C_left`-only and cannot be reused as-is; factor out a small shared JK builder.
- `davidson.py`: block Davidson-Liu with the preconditioner, photonic seeding, root following by
  overlap, subspace collapse (max ~10–20× `nroots`).
- Result object (`QEDCISResults` dataclass, mirroring `QEDSAPT0Results`) with per-root:
  excitation energy, photon number `⟨b†b⟩ = Σ_n n |X_n|²`, electronic/photonic weight, transition
  dipole from the ground polariton, oscillator strength. This is what makes LP/UP assignment
  possible and is not optional for usability.
- `output.py`-styled results table.

**Tests**
- **Column test**: apply the sigma routine to every unit vector, assemble the implied matrix,
  compare element-wise to the Tier-1 dense matrix. This is the decisive test — if it passes, every
  sign, factor, and index is right.
- Davidson roots == dense `eigh` roots (lowest `k`) to 1e-9 for a small system.
- Degenerate `λ = 0` case converges to the right multiplicities.
- Cost regression: JK-build count per iteration == `N_ph + 1`.

**Exit criterion** — Davidson matches dense for MgH+/cc-pVDZ and runs a system where the dense
build is infeasible (e.g. water dimer / aug-cc-pVDZ, 5 roots).

---

### Tier 3 — Generalization

- **3.1 QED-CIS-N** — arbitrary `N_ph`. Essentially free given the layout; the work is the
  convergence study (roots vs `N_ph`) and documenting when `N_ph = 1` is sufficient.
- **3.2 QED-TDA-DFT** — KS reference. `−K` becomes `−x_alpha·K + f_xc` contribution via
  `Vpot.compute_Vx`; reuse `scf.py`'s `_build_vbase`/`Vpot`. DSE and coupling blocks are unchanged.
  Range-separated: add the `wK` term following `scf.py`'s existing pattern.
- **3.3 Density fitting** — automatic through `psi4.core.JK` when `scf_type = df`. The DSE terms
  are already factorized and DF-free. Test against `tests/test_df_water_regression.py` conventions.

---

### Tier 4 — Full response: QED-TDHF and QED-LR-TDDFT *(DEFERRED — out of current scope)*

> Parked pending the Shao *et al.* prism equations (*J. Chem. Phys.* **155**, 064107 (2021);
> gradients in **156**, 124104 (2022)). Recorded here so Tiers 0-3 are built without foreclosing
> it; **do not start this tier**. The reference implementation is
> [`cc-ats/qed-tddft`](https://github.com/cc-ats/qed-tddft), whose prism factorizes as
> cavity model `{PF, RWA, Rabi, JC}` × electronic response `{TDA, RPA}` — i.e. two orthogonal
> switches on one sigma builder, which is the same architecture Tiers 2-3 already establish.

The excitation manifold gains de-excitation partners: `{a†_a a_i} ⊗ photon` plus `b†`, paired with
`{a†_i a_a} ⊗ photon` and `b`.

```
[ A   B ] [X]        [ 1   0 ] [X]
[ B   A ] [Y]  =  ω  [ 0  −1 ] [Y]
```

**Decisions to settle before coding this tier** (flagged, not pre-empted):
- Which formulation — Fock-state-extended RPA (photon states as additional excitation operators)
  vs. the Flick/Rubio QED-TDDFT eigenvalue problem. They differ in how the photon de-excitation
  sector is treated; the answer determines the `B`-block photon elements.
- Hermitian reduction: with real orbitals and `A − B` positive definite, solve
  `(A−B)^½ (A+B) (A−B)^½ Z = ω² Z`. The pure-photon sector contributes `A − B = ω > 0`, so
  positive-definiteness should survive, but verify it holds with coupling on before relying on it.

**Reuse**: the Tier-2 sigma machinery carries over directly; `σ_A` and `σ_B` differ only in the
JK contraction pattern (`B` uses `2(ia|bj) − (ib|aj)`, i.e. `K` on the transposed density) and the
corresponding DSE pattern. Both stay `O(N³)` on the cavity side.

**Tests** — TDA limit (`B = 0`) reproduces Tier-2 exactly; `λ = 0` reproduces `psi4
tdscf_excitations(tda=False)`; Thomas-Reiche-Kuhn / oscillator-strength sum rule as a physics check.

---

### Tier 5 — Integration, API, docs

- `CQEDCalculator.cis(geometry, nroots=..., n_photon=...)` and `.tddft(...)`, replacing the current
  placeholder factories; result dataclasses; `__init__.py` exports.
- **Option surface**, modelled on `qed-ci`'s `parseCavityOptions` since it is already
  battle-tested: `orbital_basis` (`cqed_hf` | `canonical_hf`), `photon_basis`
  (`coherent_state` | `number`), `n_photon`, `nroots`, `ignore_coupling`, `ignore_dse_terms`,
  `full_diagonalization`, and Davidson controls (`indim`, `maxdim`, `threshold`, `maxiter`).
  Every one of these is a switch on one sigma builder, not a separate method.
- Promote `sapt/dse_jk.py` to a shared `dse_jk.py` (or `integrals/`) now that two consumers exist.
- `docs/development/qed_response_design.md` recording conventions the same way
  `dse_jk_cphf_design.md` does — sign convention, ordering, sigma contract, amplitude layout.
- `examples/` driver reproducing the MgH+ LP/UP splitting figure.
- Deferred: triplets (they decouple from the bilinear term since `d_ia^T = 0`, but the `Ĝ` block
  still acts — worth a short note), complex ω / lossy cavities (explicitly out of scope per the
  Hermitian-only requirement; keep the solver interface non-Hermitian-ready but do not implement).

---

## 4. Open decisions

1. **Amplitude container** — structured `(N_ph+1, 1+n_ov)` array with named views, or a flat vector
   plus index helpers? Recommendation: structured, since the sigma routine is written per photon
   block and Davidson only ever needs a flat view.
2. **`|Φ_0, n⟩` treatment** — part of the CIS manifold (as above), or a separate rank-1 sector
   coupled to it? Affects Davidson seeding and the Tier-4 `B` block. Recommendation: keep it in the
   manifold; the leading-position convention keeps the electronic sub-block contiguous anyway.
3. **Where DSE lives** — MO-block form (fastest for CIS) vs. routing through `DSEJK` (uniform with
   SAPT/CPHF). Recommendation: MO-block in the sigma, `DSEJK` retained for the AO-side consumers.
4. **Complex ω** — confirm it is dropped. The reference supports lossy cavities via `np.linalg.eig`;
   the Hermitian-only requirement removes it. Keep the interface open, do not implement.
