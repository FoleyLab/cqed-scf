# Unrestricted (QED-)SCF reference data

Reference energies for the forthcoming unrestricted engine in
`src/cqed_scf/uscf.py`, which is currently an architecture placeholder.

Two independent oracles are used:

| target method | oracle |
| --- | --- |
| UHF, UKS/PBE0 | Psi4 |
| QED-UHF, QED-UKS/PBE0 | Hilbert `polaritonic_scf` (`POLARITONIC_UHF` / `POLARITONIC_UKS`) |

All numbers are in `REFERENCE_ENERGIES.md`, with the machine-readable versions
in `psi4_reference_energies.json` and `hilbert_reference_energies.json`.

## Files

| file | role |
| --- | --- |
| `systems.py` | Shared geometries, basis, convergence settings, and cavity coupling vectors. Both drivers import this so the two oracles see bit-identical inputs. |
| `run_psi4_uhf_uks.py` | Generates `psi4_reference_energies.json` (UHF, UKS/PBE0). |
| `run_hilbert_qed_uhf_uks.py` | Generates `hilbert_reference_energies.json` (QED-UHF, QED-UKS/PBE0 over four coupling vectors). |
| `validate_protocol_restricted.py` | Proves the Hilbert option set reproduces `cqed_scf`'s own CQED-RHF and CQED-RKS energies, so the QED-U* numbers are apples-to-apples. |
| `make_reference_table.py` | Cross-checks the two JSON files and writes `REFERENCE_ENERGIES.md`. |

## Regenerating everything

Requires an environment with Psi4, the built Hilbert plugin, and `cqed_scf`
installed (`p4dev` on the development machine):

```bash
python validate_protocol_restricted.py && python run_psi4_uhf_uks.py && python run_hilbert_qed_uhf_uks.py && python make_reference_table.py
```

`run_hilbert_qed_uhf_uks.py` and `validate_protocol_restricted.py` prepend
`/Users/jfoley19/Code` to `sys.path` so that `import hilbert` picks up the built
plugin; change that path if the checkout lives elsewhere.

## Conventions worth knowing before you use these numbers

**Coupling strength.** `cqed_scf` is parameterized by the coupling vector
`lambda`. Hilbert's option is `CAVITY_COUPLING_STRENGTH` = `g`, and it rebuilds
`lambda_i = g_i * sqrt(2 * omega)` internally
(`hilbert/src/polaritonic_scf/hf.cc`). The drivers here pass
`g = lambda / sqrt(2 * omega)`.

**Frequency.** In the coherent-state basis the CQED-SCF *ground-state* energy
does not depend on `omega`; it enters only through the `lambda` <-> `g` map.
Confirmed numerically: Hilbert returns the identical energy for
`N_PHOTON_STATES` = 1 and 2. `omega = 0.1` a.u. is fixed here for
reproducibility.

**Orientation.** Every geometry uses `no_reorient`, `no_com`, and `symmetry c1`.
The dipole self-energy couples to the molecule's orientation relative to the
cavity polarization vector, so letting Psi4 reorient or recenter would silently
change the QED energies. Do not remove those keywords.

**Integrals.** `scf_type = pk`, so these are true reference values with no
density-fitting error. The DFT grid is fixed at 99 radial x 590 spherical.

## Test systems

| key | description | multiplicity | why it is here |
| --- | --- | --- | --- |
| `oh_radical` | Hydroxide radical, OH | 2 | The common doublet case for all four methods. Strongly polar, so it has a large first-order cavity response. |
| `o2_triplet` | Molecular oxygen, O2 | 3 | Triplet, *non-polar*. With no permanent dipole the cavity contribution comes entirely from the two-electron dipole self-energy and quadrupole terms, which isolates those pieces of the Fock build. |
| `nh_triplet` | Imidogen, NH | 3 | Triplet *and* polar. Complements O2: exercises the permanent-dipole cavity terms in a high-spin reference, the regime where a bug in the beta-spin Fock build would otherwise hide. |

O2 was the suggested triplet and is included as such; NH is added because a
non-polar triplet alone leaves the permanent-dipole terms untested at high spin.

## Suggested order of tests for `uscf.py`

1. **`lambda = 0`.** QED-UHF and QED-UKS must reproduce the Psi4 UHF / UKS
   energies exactly. Both oracles agree here to better than `5e-13` Eh, so this
   is a hard equality test, not a loose one.
2. **`z_only`.** Polarization along the molecular axis; the largest cavity shift.
3. **`x_only`.** Polarization perpendicular to the axis; catches sign and index
   errors in the transverse dipole self-energy components.
4. **`general`.** Non-axial `lambda`; the only case that exercises every
   off-diagonal `lambda_i * lambda_j` quadrupole and dipole cross term.

Running `o2_triplet` before `nh_triplet` separates "the two-electron DSE is
wrong" from "the permanent-dipole term is wrong": O2 fails only for the former.
