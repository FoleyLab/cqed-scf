# Unrestricted (QED-)SCF reference energies

Reference data for the forthcoming `cqed_scf.uscf` implementation.

| setting | value |
| --- | --- |
| basis | `6-311++G**` |
| functional (UKS / QED-UKS) | `PBE0` |
| `scf_type` | `pk` (no density fitting) |
| `e_convergence` / `d_convergence` | `1e-11` / `1e-10` |
| DFT grid | 99 radial x 590 spherical |
| cavity frequency omega | `0.1` a.u. |
| photon states | 1 (coherent-state basis) |
| UHF / UKS oracle | Psi4 1.10 |
| QED-UHF / QED-UKS oracle | Hilbert `polaritonic_scf` (Psi4 1.10) |

Coupling convention: `cqed_scf` takes `lambda`; Hilbert takes
`CAVITY_COUPLING_STRENGTH` = `lambda / sqrt(2 * omega)`.

## Cavity coupling vectors

| label | lambda (a.u.) | purpose |
| --- | --- | --- |
| `zero` | (0.00, 0.00, 0.00) | regression anchor: must reproduce the cavity-free UHF / UKS energy |
| `z_only` | (0.00, 0.00, 0.05) | polarization parallel to the molecular axis |
| `x_only` | (0.05, 0.00, 0.00) | polarization perpendicular to the molecular axis |
| `general` | (0.02, 0.03, 0.05) | non-axial, exercises every off-diagonal lambda_i*lambda_j term |

## Test systems

| key | description | charge | multiplicity |
| --- | --- | --- | --- |
| `oh_radical` | Hydroxide radical (OH), doublet, polar | 0 | 2 |
| `o2_triplet` | Molecular oxygen (O2), triplet ground state, non-polar | 0 | 3 |
| `nh_triplet` | Imidogen (NH), triplet ground state, polar | 0 | 3 |

### `oh_radical` geometry

```
0 2
O     0.000000     0.000000     0.000000
H     0.000000     0.000000     0.969300
no_reorient
no_com
units angstrom
symmetry c1
```

### `o2_triplet` geometry

```
0 3
O     0.000000     0.000000     0.000000
O     0.000000     0.000000     1.207520
no_reorient
no_com
units angstrom
symmetry c1
```

### `nh_triplet` geometry

```
0 3
N     0.000000     0.000000     0.000000
H     0.000000     0.000000     1.036200
no_reorient
no_com
units angstrom
symmetry c1
```

## Psi4 reference: UHF and UKS

| system | method | energy / Eh | &lt;S^2&gt; | exact &lt;S^2&gt; |
| --- | --- | ---: | ---: | ---: |
| `oh_radical` | UHF | `-75.414526952824` | 0.755764 | 0.7500 |
| `oh_radical` | UKS/PBE0 | `-75.679146442729` | 0.752311 | 0.7500 |
| `o2_triplet` | UHF | `-149.659551073945` | 2.044794 | 2.0000 |
| `o2_triplet` | UKS/PBE0 | `-150.214394044296` | 2.009371 | 2.0000 |
| `nh_triplet` | UHF | `-54.978041641531` | 2.015796 | 2.0000 |
| `nh_triplet` | UKS/PBE0 | `-55.175076229479` | 2.005949 | 2.0000 |

## Hilbert reference: QED-UHF and QED-UKS

| system | method | lambda | energy / Eh | dE vs. cavity-free / Eh |
| --- | --- | --- | ---: | ---: |
| `oh_radical` | QED-UHF | `zero` | `-75.414526952824` | +0.000000000 |
| `oh_radical` | QED-UHF | `z_only` | `-75.409799160801` | +0.004727792 |
| `oh_radical` | QED-UHF | `x_only` | `-75.410652176375` | +0.003874776 |
| `oh_radical` | QED-UHF | `general` | `-75.407789826615` | +0.006737126 |
| `oh_radical` | QED-UKS/PBE0 | `zero` | `-75.679146442729` | +0.000000000 |
| `oh_radical` | QED-UKS/PBE0 | `z_only` | `-75.674236062459` | +0.004910380 |
| `oh_radical` | QED-UKS/PBE0 | `x_only` | `-75.675168529603` | +0.003977913 |
| `oh_radical` | QED-UKS/PBE0 | `general` | `-75.672108337110` | +0.007038106 |
| `o2_triplet` | QED-UHF | `zero` | `-149.659551073944` | +0.000000000 |
| `o2_triplet` | QED-UHF | `z_only` | `-149.650195876160` | +0.009355198 |
| `o2_triplet` | QED-UHF | `x_only` | `-149.653447107661` | +0.006103966 |
| `o2_triplet` | QED-UHF | `general` | `-149.647034901405` | +0.012516173 |
| `o2_triplet` | QED-UKS/PBE0 | `zero` | `-150.214394044296` | +0.000000000 |
| `o2_triplet` | QED-UKS/PBE0 | `z_only` | `-150.204805831192` | +0.009588213 |
| `o2_triplet` | QED-UKS/PBE0 | `x_only` | `-150.208313048987` | +0.006080995 |
| `o2_triplet` | QED-UKS/PBE0 | `general` | `-150.201657411416` | +0.012736633 |
| `nh_triplet` | QED-UHF | `zero` | `-54.978041641531` | +0.000000000 |
| `nh_triplet` | QED-UHF | `z_only` | `-54.972868426137` | +0.005173215 |
| `nh_triplet` | QED-UHF | `x_only` | `-54.973719375981` | +0.004322266 |
| `nh_triplet` | QED-UHF | `general` | `-54.970631182269` | +0.007410459 |
| `nh_triplet` | QED-UKS/PBE0 | `zero` | `-55.175076229479` | +0.000000000 |
| `nh_triplet` | QED-UKS/PBE0 | `z_only` | `-55.169693150469` | +0.005383079 |
| `nh_triplet` | QED-UKS/PBE0 | `x_only` | `-55.170646909629` | +0.004429320 |
| `nh_triplet` | QED-UKS/PBE0 | `general` | `-55.167402510340` | +0.007673719 |

## lambda -> 0 consistency check

Hilbert's QED-UHF / QED-UKS must collapse onto Psi4's UHF / UKS when the
coupling is switched off. This is the first test `uscf.py` should pass.

| system | method | Psi4 / Eh | Hilbert (lambda=0) / Eh | difference / Eh |
| --- | --- | ---: | ---: | ---: |
| `oh_radical` | UHF | `-75.414526952824` | `-75.414526952824` | -2.84e-13 |
| `oh_radical` | UKS | `-75.679146442729` | `-75.679146442729` | -7.11e-14 |
| `o2_triplet` | UHF | `-149.659551073945` | `-149.659551073944` | 4.83e-13 |
| `o2_triplet` | UKS | `-150.214394044296` | `-150.214394044296` | 4.55e-13 |
| `nh_triplet` | UHF | `-54.978041641531` | `-54.978041641531` | 2.84e-14 |
| `nh_triplet` | UKS | `-55.175076229479` | `-55.175076229479` | -2.84e-14 |

All 6 cases agree to better than 1e-10 Eh.

