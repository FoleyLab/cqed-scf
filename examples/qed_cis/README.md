# QED-CIS examples

Polaritonic excited states from the `cqed_scf` response module.

| script | what it shows |
|---|---|
| `mghp_polaritons.py` | LP/UP splitting of MgH+ in a resonant cavity, checked against the independent `qed-ci` determinant-basis reference |
| `mghp_rabi_scan.py` | Rabi splitting versus coupling strength; the `split/lambda` column should be roughly constant in the linear regime |
| `water_qed_tda_dft.py` | the Kohn-Sham path (QED-TDA-DFT) with a B3LYP reference |

## API

```python
from cqed_scf import CQEDCalculator

calc = CQEDCalculator(
    lambda_vector=[0.0, 0.0, 0.0125],
    psi4_options={"basis": "cc-pVDZ", "scf_type": "pk", "save_jk": True},
    omega=0.17456,          # Eh
)

results = calc.cis(geometry, nroots=8, n_photon=1)
```

`calc.tddft(...)` is the same computation named for a Kohn-Sham reference, and
requires one. `calc.response(...)` returns the driver without solving it, for
access to the Hamiltonian or the sigma action.

`QEDCISResults` carries `eigenvalues` (relative to the SCF reference),
`total_energies`, `excitation_energies`, `photon_numbers`, `reference_weights`,
`singles_weights`, `transition_dipoles` and `oscillator_strengths`.

## Two things that catch people out

**Eigenvalues are not excitation energies.** They are referenced to the CQED-SCF
energy, and the QED-CIS ground state relaxes *below* that reference once the
coupling is on — the bilinear term couples `|Phi_0,0>` to `|Phi_i^a,1>`. Use
`excitation_energies` (`E_k - E_0`), not the raw eigenvalues. For MgH+ at
lambda = 0.0125 the difference is 2.5e-4 Eh.

**Do not track roots by index across a scan.** Changing the coupling, or the
number of photon states, reorders the spectrum and inserts new states into it.
Use `results.polariton_indices()`, which both example scans do.

**Do not identify polaritons by `photon_numbers` either.** That is `sum_n n w_n`,
which counts *every* amplitude in the n-photon block — including `|Phi_i^a, n>`,
an electronic excitation *carrying* a photon. Such a photon-dressed state sits
near (bare transition + omega) with a photon number close to 1 while mixing with
nothing, and will outrank a genuine polariton. On MgH+ at resonance it does
exactly that. A polariton is the mixture of the bare photon `|Phi_0, n>` with an
electronic excitation, so the identifying quantity is the weight on the photonic
*reference* state, `reference_weights[:, n]` — which is what
`polariton_indices()` uses. A genuine LP/UP pair shares that weight, so it should
sum to roughly one across the two.

## Notes

`save_jk: True` is required in the Psi4 options if you also want to compare
against `psi4.driver.procrouting.response.scf_response.tdscf_excitations`, which
reads the JK object off the wavefunction.

The default solver is matrix-free Davidson; pass `solver="dense"` to build the
Hamiltonian explicitly, which is practical for small systems and is what the
test suite uses as its oracle.
