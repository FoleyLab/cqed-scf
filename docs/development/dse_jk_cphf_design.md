# DSE JK and CPHF Conventions

This note records the conventions used to connect `DSEJK`, `PauliFierzJK`,
and the local SAPT JK helper routines.

## Matrix Shapes

- `Cocc` is AO/SO-by-occupied, shape `(nbf, nocc)`.
- `Cvir` is AO/SO-by-virtual, shape `(nbf, nvir)`.
- `C_left_add(C_L)` and `C_right_add(C_R)` queue coefficient matrices with
  matching row dimension and matching column count: `(nbf, nvec)`.
- A queued generalized density is `D = C_L @ C_R.T`, shape `(nbf, nbf)`.
  It may be nonsymmetric.
- Psi4 `wfn.cphf_Hx([X])` accepts and returns CPHF trial matrices shaped
  `(nocc, nvir)`.

## CPHF Orientation

The iterative SAPT helper stores CPHF trial vectors as occupied-by-virtual
matrices. The dense `QEDSAPT0Driver.chf()` routine solves a flattened
occupied-by-virtual linear system, then transposes the converged amplitudes
to virtual-by-occupied for later dense SAPT contractions.

For the matrix-free Hessian action, use the Psi4 convention throughout:
`X[o, v]` has shape `(nocc, nvir)`, and the returned `Hx[o, v]` has the same
shape.

## AO Response Density

To reproduce the RHF occupied-virtual Hessian block used by the dense SAPT
oracle, form the one-sided AO response density

```text
D_ov = Cocc @ X @ Cvir.T
```

Do not symmetrize this density before building J/K. The transpose-side
contribution enters explicitly through the exchange transpose term.

## DSE Hessian Action

For the separable dipole-dipole operator,

```text
J_DSE[D]_pq = d_pq sum_rs d_rs D_rs
K_DSE[D]_pq = sum_rs d_pr d_qs D_rs = d @ D @ d.T
```

The RHF DSE two-electron Hessian contribution in Psi4's `(nocc, nvir)`
orientation is

```text
D_ov = Cocc @ X @ Cvir.T
J, K = DSEJK.jk_from_density(D_ov)
G_DSE = 4 J - K - K.T
Hx_DSE = Cocc.T @ G_DSE @ Cvir
```

This is equivalent to the dense orbital block

```text
A_ovOV = 4 d_ov d_OV - d_oo d_vv - d_oV d_Ov
```

contracted with `X_OV`. The production path never constructs `A_ovOV`.

## Call Flow

- Python SAPT matrix builds use `cache["jk"]`, a `PauliFierzJK` composite
  when DSE is supplied.
- Psi4 wavefunction internals use `cache["native_jk"]`, obtained from
  `PauliFierzJK.native_jk()`.
- `_sapt_cpscf_solve()` sets only the native JK on each wavefunction, calls
  `wfn.cphf_Hx()` for the native ERI action, then adds
  `DSECPHF.hx_matrix()` for each active monomer.
- `QEDSAPT0Driver.chf()` remains the dense numerical oracle: compare the
  dense cavity block above against the matrix-free `DSECPHF` action for small
  systems or synthetic coefficient/dipole matrices.
