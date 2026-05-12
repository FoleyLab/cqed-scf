"""
H2 Dimer Potential Energy Curve
Methods: wB97X-D, CCSD, FCI
Basis:   cc-pVDZ
- Counterpoise (CP) correction for BSSE on all methods
- Each method referenced to its own asymptote
- Monomer geometry fixed at experimental r(H-H) = 0.74 Ang
"""

import numpy as np
import matplotlib.pyplot as plt
import psi4

# ── Output / memory ────────────────────────────────────────────────────────────
psi4.core.set_output_file("h2_dimer.out", False)
psi4.set_memory("4 GB")

# ── Global options ─────────────────────────────────────────────────────────────
psi4.set_options({
    "basis":                "cc-pVDZ",
    "e_convergence":        1e-10,
    "d_convergence":        1e-8,
    # CCSD / FCI
    "freeze_core":          False,       # explicit: correlate all electrons
    "cc_type":              "conv",      # conventional integrals for CCSD/FCI
    # DFT grid
    "dft_radial_points":    99,
    "dft_spherical_points": 302,
    "dft_pruning_scheme":   "none",
})

# ── Geometry builder ───────────────────────────────────────────────────────────
R_HH = 0.74   # intramolecular H-H bond length (Angstrom)

def make_dimer(d):
    """
    Dimer with ghost-atom syntax for counterpoise.
    Monomer A: real H2 at z=0; Monomer B: real H2 at z=d.
    @X = ghost atom of element X.
    """
    h = R_HH / 2
    return f"""
    0 1
    H   0.000000  0.000000  {-h}
    H   0.000000  0.000000  { h}
    @H  0.000000  0.000000  {d - h}
    @H  0.000000  0.000000  {d + h}
    --
    0 1
    @H  0.000000  0.000000  {-h}
    @H  0.000000  0.000000  { h}
    H   0.000000  0.000000  {d - h}
    H   0.000000  0.000000  {d + h}
    symmetry c1
    no_reorient
    nocom
    """

def make_monomer():
    """Isolated monomer in its own (monomer) basis — for the asymptote."""
    h = R_HH / 2
    return f"""
    0 1
    H  0.000000  0.000000  {-h}
    H  0.000000  0.000000  { h}
    symmetry c1
    no_reorient
    nocom
    """

# ── Scan parameters ────────────────────────────────────────────────────────────
d_vals = np.linspace(2.2, 6.0, 41)   # intermolecular separation (Ang)

# ── Run calculations ───────────────────────────────────────────────────────────
methods = {
    "wB97X-D": {"scf_type": "df"},    # DF is fine and fast for DFT
    "CCSD":    {"scf_type": "pk"},    # conventional for correlated
    "FCI":     {"scf_type": "pk"},
}

# Storage: interaction energies (Eh) at each distance, per method
E_int = {m: [] for m in methods}

for d in d_vals:
    print(f"\n{'='*60}")
    print(f"  d = {d:.3f} Ang")
    print(f"{'='*60}")

    geom_str = make_dimer(d)

    for method, opts in methods.items():
        psi4.set_options(opts)
        mol = psi4.geometry(geom_str)

        # Counterpoise-corrected interaction energy via built-in CP driver.
        # psi4.energy with bsse_type='cp' automatically:
        #   1. Computes dimer energy
        #   2. Computes each monomer in the FULL dimer basis (with ghost atoms)
        #   3. Returns E_dimer - E_A(dimer basis) - E_B(dimer basis)
        E_cp = psi4.energy(method, bsse_type="cp", molecule=mol)
        E_int[method].append(E_cp)
        print(f"  {method:10s}  E_int(CP) = {E_cp:+.8f} Eh")

# Convert to arrays (Eh → kcal/mol)
EH2KCAL = 627.5094740631
d_vals = np.array(d_vals)
E_int_kcal = {m: np.array(v) * EH2KCAL for m, v in E_int.items()}

# ── Plot ───────────────────────────────────────────────────────────────────────
styles = {
    "wB97X-D": dict(ls="-",  color="tab:blue",   label="wB97X-D / cc-pVDZ (CP)"),
    "CCSD":    dict(ls="--", color="tab:orange",  label="CCSD / cc-pVDZ (CP)"),
    "FCI":     dict(ls="",   color="tab:green",   marker="*", label="FCI / cc-pVDZ (CP)"),
}

fig, ax = plt.subplots(figsize=(7, 5))
for method, style in styles.items():
    ax.plot(d_vals, E_int_kcal[method], **style)

ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("Intermolecular separation (Å)", fontsize=12)
ax.set_ylabel("Interaction energy (kcal/mol)", fontsize=12)
ax.set_title("H₂ Dimer PEC — CP-corrected", fontsize=13)
ax.set_xlim(2.2, 6.0)
ax.legend(fontsize=10)
ax.grid(alpha=0.4)
fig.tight_layout()
plt.savefig("h2_dimer_pec.png", dpi=150)
plt.show()

# ── Print well-depth summary ───────────────────────────────────────────────────
print("\n── Well depth summary ──────────────────────────────────")
for method in methods:
    E = E_int_kcal[method]
    idx_min = np.argmin(E)
    print(f"  {method:10s}  D_e = {E[idx_min]:.4f} kcal/mol  at d = {d_vals[idx_min]:.2f} Ang")
