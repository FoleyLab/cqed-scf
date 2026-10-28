Work in `/Users/jfoley19/Code/cqed-scf`.

First read the repository’s `CLAUDE.md` and all applicable `AGENTS.md` files. Inspect the current working tree before editing and preserve existing user changes, especially the recent `<S^2>` implementation in `src/cqed_scf/uscf.py`.

Goal
====

Complete these two related improvements to the unrestricted CQED-SCF implementation:

1. Validate `<S^2>` and spin contamination against Psi4 UHF references.
2. Replace the explicit four-index AO ERI contractions in `uscf.py` with Psi4’s JK machinery and implement UKS, including pure, hybrid, and range-separated functionals through unrestricted `VBase`.

Proceed tier by tier. Run the relevant tests at the end of every tier. Do not move past a failed gate without diagnosing and fixing it. Keep changes focused on unrestricted SCF; unrestricted gradients and response theory remain out of scope.

Important existing resources
============================

Inspect and reuse:

- `src/cqed_scf/uscf.py`
- `src/cqed_scf/scf.py`
- `src/cqed_scf/references.py`
- `src/cqed_scf/calculator.py`
- `tests/test_uhf_matches_psi4_reference.py`
- `tests/test_qed_uhf_matches_hilbert_reference.py`
- `tests/test_unrestricted_plumbing.py`
- `tests/test_scf_matches_cqed_rhf_reference.py`
- `examples/canonical/Unrestricted_Test_Examples/`
- `examples/canonical/Unrestricted_Test_Examples/psi4_reference_energies.json`
- `examples/canonical/Unrestricted_Test_Examples/hilbert_reference_energies.json`
- `examples/canonical/Unrestricted_Test_Examples/run_psi4_uhf_uks.py`
- `examples/canonical/Unrestricted_Test_Examples/run_hilbert_qed_uhf_uks.py`
- `examples/canonical/Unrestricted_Test_Examples/systems.py`
- `tests/UNRESTRICTED_REFERENCE_ENERGIES.md`

Use the canonical geometries, orientations, basis, grids, and convergence settings from `Unrestricted_Test_Examples`. Do not create a second inconsistent reference set.

Tier 0 — Baseline and test-environment audit
============================================

1. Inspect `git status` and the current diff.
2. Identify the supported command/environment for running Psi4 tests from the repository documentation.
3. Run the existing unrestricted unit/regression tests before editing.
4. Record existing failures, including stale plumbing tests that still expect unrestricted energy calculations to raise `NotImplementedError`.
5. Do not weaken tolerances merely to accommodate an implementation problem.

Gate:

- The baseline state is understood.
- The Psi4 test environment is identified.
- Existing failures are distinguished from regressions introduced during this task.

Tier 1 — Spin diagnostics and Psi4 validation
=============================================

Extract the spin calculation into a focused helper, either private to `uscf.py` or in a suitably scoped shared module.

For occupied alpha and beta coefficient matrices,

    S_ab = Ca_occ.conj().T @ S @ Cb_occ

use

    M_s = 0.5 * (N_alpha - N_beta)

    <S^2> = M_s^2
            + 0.5 * (N_alpha + N_beta)
            - sum_ij |S_ab[i,j]|^2

and

    S_target = 0.5 * abs(N_alpha - N_beta)

    spin_contamination =
        <S^2> - S_target * (S_target + 1)

Requirements:

- Use the conjugate transpose.
- Use the real part of `np.vdot`.
- Return real Python floats.
- Keep both `s_squared` and `spin_contamination` in the results dictionary.
- Calculate them from the final converged CQED orbitals, not the original Psi4 orbitals.

Extend the Psi4 reference-generation record to store explicit spin contamination:

    spin_contamination =
        s_squared - s_squared_exact

Regenerate the JSON only through the existing reference generator and only if the appropriate Psi4 environment is available.

Update `tests/test_uhf_matches_psi4_reference.py` to use the canonical systems and settings. Compare energy, `<S^2>`, and spin contamination for:

- OH doublet
- O2 triplet
- NH triplet

Use absolute tolerances appropriate to the tight canonical convergence settings; target approximately `1e-8` or better for spin diagnostics.

Add inexpensive helper-level tests for:

- A restricted singlet determinant: `<S^2> = 0`.
- A pure doublet determinant: `<S^2> = 0.75`.
- Complex-valued mock orbitals.
- The internal identity relating contamination and the target spin.

Gate:

- Canonical UHF `<S^2>` and contamination agree with the Psi4 references.
- Existing UHF energies still agree.
- Spin helper tests pass.

Tier 2 — Explicit unrestricted QED cancellation gate
====================================================

Before simplifying the QED terms, add explicit numerical tests proving the cancellation in the unrestricted implementation.

Define

    Dt = Da + Db
    x = Tr(d Dt)

The explicit one-electron shift is

    H_shift = -x d

and the DSE Coulomb matrices are

    J_DSE_a = d Tr(d Da)
    J_DSE_b = d Tr(d Db)

Therefore test the Fock identity

    H_shift + J_DSE_a + J_DSE_b = 0

to near machine precision.

Test the energy identity using the current unrestricted half-trace expression:

    E_core_shift = Tr(H_shift Dt) = -x^2

    E_J_DSE =
        0.5 * Tr[(J_DSE_a + J_DSE_b) Da]
      + 0.5 * Tr[(J_DSE_a + J_DSE_b) Db]
      = 0.5 * x^2

    E_scalar = 0.5 * x^2

    E_core_shift + E_J_DSE + E_scalar = 0

Also compare, on exactly the same density and matrices,

    F_sigma_explicit =
        H - x d + J_total - K_sigma
        + J_DSE_a + J_DSE_b - N_sigma

with

    F_sigma_simplified =
        H + J_total - K_sigma - N_sigma

where

    N_sigma = d @ D_sigma @ d

Require both alpha and beta Focks to agree to approximately `1e-12`.

Use nontrivial polar systems so the cancellation is not a zero-dipole tautology:

- OH doublet with `z_only`
- NH triplet with `z_only` or `general`

O2 alone is not sufficient because its dipole expectation is nearly zero.

Gate:

- Fock cancellation passes for both spin channels.
- Energy cancellation passes term by term.
- Explicit and simplified Focks and energies agree on the same densities.
- Nonzero individual terms are demonstrated before their sum is tested.

Tier 3 — Migrate QED-UHF from dense ERIs to Psi4 JK
===================================================

Remove the production allocation of

    I = np.asarray(mints.ao_eri())

and all electronic `einsum` J/K builds.

Correct configuration selection first:

- Derive the unrestricted method from `config.reference`.
- Use `config.base_scf_functional`, not `config.functional`.
- UHF must remain UHF even if the general config carries a functional for another purpose.

Build one Psi4 JK object. During each iteration queue both occupied coefficient matrices in this order:

    Ca_occ
    Cb_occ

Retrieve spin-resolved matrices:

    Ja, Jb
    Ka, Kb
    wKa, wKb

when requested, and define

    Jtot = Ja + Jb

The simplified QED-UHF Focks are

    Fa = H + Jtot - Ka - Na
    Fb = H + Jtot - Kb - Nb

with

    H = T + V + Q_PF
    Na = d @ Da @ d
    Nb = d @ Db @ d

The corresponding energy is

    E =
        Tr[H (Da + Db)]
      + 0.5 Tr[Jtot (Da + Db)]
      - 0.5 Tr[Ka Da]
      - 0.5 Tr[Kb Db]
      - 0.5 Tr[Na Da]
      - 0.5 Tr[Nb Db]
      + E_nuc

Requirements:

- Preserve the shared-coefficient unrestricted DIIS behavior unless a test demonstrates it is wrong.
- On convergence, canonicalize the final unextrapolated alpha and beta Focks.
- Return orbital energies belonging to the final CQED Focks.
- Keep original Psi4 orbital energies under explicitly named canonical/reference keys if they remain useful.
- Do not call `ao_eri()` in the production UHF path.
- Allow Psi4’s selected JK backend to handle PK versus DF.

Add a small-basis JK parity test comparing Psi4 JK matrices against dense contractions before deleting dense contractions from production:

- `Ja`
- `Jb`
- `Ka`
- `Kb`

Add a no-dense-ERI guard that makes `ao_eri()` raise and confirms UHF still runs.

Regression gates:

- Lambda-zero UHF matches Psi4.
- Nonzero QED-UHF matches the existing Hilbert references for all canonical coupling vectors.
- `<S^2>` and spin contamination remain unchanged within tight tolerances.
- PK works.
- DF runs without allocating four-index AO ERIs.

Tier 4 — Add unrestricted `VBase` and UKS
=========================================

Build an unrestricted potential using:

    psi4.core.VBase.build(
        wfn.basisset(),
        wfn.functional(),
        "UV",
    )

For each iteration provide both densities:

    Vpot.set_D([Da, Db])
    Vpot.compute_V([Vxc_a, Vxc_b])

Read the total XC energy from:

    Vpot.quadrature_values()["FUNCTIONAL"]

Configure JK according to the functional:

- Always build Coulomb.
- Build `K` only when exact short-range/global exchange is needed.
- Build `wK` only for long-range-corrected functionals.
- Set the functional’s omega when needed.
- Use the functional’s reported `x_alpha()` and `x_beta()` coefficients. Do not assume every range-separated functional has `x_beta = 1 - x_alpha` without verification.

The UKS Focks are

    F_sigma =
        H
      + Jtot
      + Vxc_sigma
      - N_sigma
      - x_alpha * K_sigma
      - x_beta * wK_sigma

for whichever K terms the functional requires.

The UKS energy is

    E =
        Tr[H Dt]
      + 0.5 Tr[Jtot Dt]
      + Exc
      + E_nuc
      - 0.5 x_alpha * sum_sigma Tr[K_sigma D_sigma]
      - 0.5 x_beta  * sum_sigma Tr[wK_sigma D_sigma]
      - 0.5 * sum_sigma Tr[N_sigma D_sigma]

Test three functional branches:

- PBE: pure functional, no exact exchange
- PBE0: global hybrid
- wB97X or the repository’s established range-separated functional: K and wK behavior

Regression gates:

- Lambda-zero UKS/PBE0 agrees with Psi4 energy, `<S^2>`, and contamination.
- Nonzero QED-UKS/PBE0 agrees with the existing Hilbert reference data for OH, O2, and NH over all canonical coupling vectors.
- Pure, hybrid, and range-separated functional branches run and agree with Psi4 at zero coupling.
- PK and DF UKS paths run without `ao_eri()`.

Tier 5 — Integration and cleanup
================================

1. Update stale unrestricted plumbing tests:
   - UHF and UKS energy dispatch should now succeed.
   - Unrestricted gradients and response theory should continue to raise clear `NotImplementedError` exceptions.
2. Remove obsolete tutorial placeholders, commented-out implementation fragments, duplicate assignments, and unconditional debug `print` statements from `uscf.py`.
3. Use the package output layer consistently.
4. Ensure the unrestricted results contract is explicit and does not alias restricted-only keys such as `density`, `coefficients`, or `ndocc` to one spin channel.
5. Add useful unrestricted result keys where appropriate:
   - `Ca`, `Cb`
   - `Da`, `Db`
   - `Coa`, `Cob`
   - `Fa`, `Fb`
   - final alpha/beta orbital energies
   - canonical/reference alpha/beta orbital energies
   - `nalpha`, `nbeta`
   - `s_squared`
   - `spin_contamination`
   - method/reference/functional metadata
   - JK/DF metadata
6. Finalize Psi4 JK/VBase objects appropriately if required by the supported Psi4 API.

Final validation
================

Run:

- Focused spin tests
- Cancellation identity tests
- UHF/Psi4 tests
- QED-UHF/Hilbert tests
- UKS/Psi4 tests
- QED-UKS/Hilbert tests
- Unrestricted plumbing tests
- Relevant restricted SCF tests to ensure shared-helper changes did not regress RHF/RKS
- The broader test suite if practical

Report:

1. Files changed.
2. The final UHF and UKS equations implemented.
3. Numerical cancellation residuals for the Fock and energy identities.
4. Maximum deviations from Psi4 and Hilbert for energy and `<S^2>`.
5. PK and DF coverage.
6. Any tests not run and the concrete reason.
7. Any remaining unrestricted limitations.

Do not implement unrestricted gradients, CIS, response theory, or QED-CI as part of this task.
