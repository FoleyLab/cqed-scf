"""
Basis set superposition error in the benzene dimer slip scan at the DFT level
(density-fitted wB97X-D / jun-cc-pVDZ, Psi4).

Three interaction energies are compared at every slip R_AB:

  1. counterpoise (CP):
         E_int = E_AB(R_AB) - E_A(R_A Gh(B)) - E_B(Gh(A) R_B)
  2. no counterpoise (noCP):
         E_int = E_AB(R_AB) - E_A(R_A) - E_B(R_B)
  3. supermolecular with a distant reference:
         E_int = E_AB(R_AB) - E_AB(R_AB = 25 Ang)

The dimer energy is computed once per slip.  The bare monomers do not depend on
the slip (A is fixed, B is rigidly translated), so they are computed once.  Only
the ghosted monomers change with R_AB.  At 25 Ang the diffuse functions do not
overlap, so approach 3 should nearly reproduce approach 2; the difference
E_AB(25) - (E_A + E_B) is printed as a consistency check on the monomer
references (long-range tail plus DFT grid noise).

Geometry construction
---------------------
The base dimer is a parallel-displaced pair whose ring normal is tilted ~6 degrees
from the lab z axis.  Both monomers are rotated into a stack frame centered on the
centroid of monomer A:

    z' : normal of monomer A's ring plane, pointing toward monomer B
    x' : in-plane direction of the base dimer's own slip offset (the slip coordinate)
    y' : z' cross x'

Monomer B is placed in a true stack (centroid directly above A's) at the base
interplane distance and rigidly translated along x' by the slip.  This is the same
construction as qed_sapt0_df_benzene_slip_scan.py.

Usage:
    python dft_bsse_benzene_slip_scan.py [--max-slip 8.0] [--npoints 81]
"""

import argparse
import json
import time

import numpy as np
import psi4

EH_TO_KCAL = 627.5094740631
FAR_SLIP = 25.0           # Ang, non-interacting reference separation along x'
RESULTS_FILE = "dft_bsse_scan_results.json"


# ---------------------------------------------------------
# Base dimer geometry
# ---------------------------------------------------------

base_dimer = """
0 1
C    0.712645325   1.120995701   0.060540783
H    1.357841649   1.986399167   0.127737169
C    1.258235731  -0.159251901   0.124233518
H    2.324954277  -0.287099878   0.246743035
C    0.426884963  -1.274526662   0.042650428
H    0.850444649  -2.268432680   0.094749952
C   -0.949577845  -1.110074058  -0.100313595
H   -1.594455696  -1.976273703  -0.163713480
C   -1.495525640   0.171050561  -0.161546018
H   -2.563782791   0.299221149  -0.273703115
C   -0.663827601   1.286642887  -0.083401433
H   -1.086900697   2.281000204  -0.132886135
--
0 1
C    1.949488213   1.110116294   3.600833554
H    2.594333337   1.976300037   3.664584272
C    2.495441339  -0.171034805   3.662353287
H    3.563648223  -0.299181855   3.775097506
C    1.663791859  -1.286634905   3.583702858
H    2.086873566  -2.280987340   3.633356521
C    0.287390298  -1.120999876   3.438986110
H   -0.357712298  -1.986410120   3.371353478
C   -0.258169573   0.159232871   3.375020282
H   -1.324797607   0.287062084   3.251841521
C    0.573107377   1.274513633   3.457091226
H    0.149582495   2.268418667   3.404722296
"""


def parse_monomers(text):
    lines = text.strip().splitlines()
    sep = lines.index("--")
    def block(chunk):
        rows = [l.split() for l in chunk]
        return [r[0] for r in rows], np.array([[float(v) for v in r[1:]] for r in rows])
    sym_a, xyz_a = block(lines[1:sep])
    sym_b, xyz_b = block(lines[sep + 2:])
    return sym_a, xyz_a, sym_b, xyz_b


def ring_normal(xyz):
    return np.linalg.svd(xyz - xyz.mean(axis=0))[2][2]


parser = argparse.ArgumentParser(description="Benzene dimer DFT BSSE slip scan")
parser.add_argument("--max-slip", type=float, default=8.0, help="largest slip in Ang")
parser.add_argument("--npoints", type=int, default=81, help="number of slip points from 0")
args = parser.parse_args()

sym_a, xyz_a, sym_b, xyz_b = parse_monomers(base_dimer)

cen_a, cen_b = xyz_a.mean(axis=0), xyz_b.mean(axis=0)
z_axis = ring_normal(xyz_a)
if (cen_b - cen_a) @ z_axis < 0:
    z_axis = -z_axis
offset = cen_b - cen_a
interplane = offset @ z_axis
inplane = offset - interplane * z_axis
x_axis = inplane / np.linalg.norm(inplane)
y_axis = np.cross(z_axis, x_axis)
frame = np.vstack([x_axis, y_axis, z_axis])   # rows: new axes in lab coordinates

# Monomers in the stack frame, origin at A's centroid.  B sits directly above A.
xyz_a_f = (xyz_a - cen_a) @ frame.T
xyz_b_f = (xyz_b - cen_b) @ frame.T + np.array([0.0, 0.0, interplane])

print(f"interplane distance : {interplane:.4f} Ang")
print(f"base slip offset    : {np.linalg.norm(inplane):.4f} Ang")


def slipped_dimer(slip):
    """Dimer string with monomer B rigidly translated by ``slip`` Ang along x'."""
    shifted = xyz_b_f + np.array([slip, 0.0, 0.0])
    fmt = lambda syms, xyz: "\n".join(
        f"{s:2s} {x:16.9f} {y:16.9f} {z:16.9f}" for s, (x, y, z) in zip(syms, xyz)
    )
    return "\n".join([
        "0 1", fmt(sym_a, xyz_a_f), "--", "0 1", fmt(sym_b, shifted),
        "units angstrom", "symmetry c1", "no_reorient", "no_com", "",
    ])


slips = [round(float(s), 6) for s in np.linspace(0.0, args.max_slip, args.npoints)]


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("8 GB")
psi4.core.set_output_file("dft_bsse_scan.out", False)

psi4.set_options({
    "basis": "jun-cc-pVDZ",
    "scf_type": "df",
    "reference": "rks",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
})

FUNCTIONAL = "wb97x-d"


def energy(molecule):
    psi4.core.clean()
    return psi4.energy(FUNCTIONAL, molecule=molecule)


# ---------------------------------------------------------
# Reference energies: independent of the slip
# ---------------------------------------------------------

ref_dimer = psi4.geometry(slipped_dimer(0.0))
t0 = time.time()
E_A = energy(ref_dimer.extract_subsets(1))
E_B = energy(ref_dimer.extract_subsets(2))
print(f"E_A(R_A) = {E_A:.10f} Eh   E_B(R_B) = {E_B:.10f} Eh   ({time.time() - t0:.0f} s)")

t0 = time.time()
E_far = energy(psi4.geometry(slipped_dimer(FAR_SLIP)))
print(f"E_AB(R_AB = {FAR_SLIP:g} Ang) = {E_far:.10f} Eh   ({time.time() - t0:.0f} s)")
print(f"E_AB(far) - (E_A + E_B) = {EH_TO_KCAL * (E_far - E_A - E_B): .6f} kcal/mol  (should be ~0)\n")


# ---------------------------------------------------------
# Scan
# ---------------------------------------------------------

rows = []


def save():
    with open(RESULTS_FILE, "w") as fh:
        json.dump(
            {
                "meta": {"functional": FUNCTIONAL, "basis": "jun-cc-pVDZ",
                         "far_slip": FAR_SLIP, "E_A": E_A, "E_B": E_B, "E_far": E_far},
                "rows": rows,
            },
            fh, indent=2,
        )


print(f"{'slip / Ang':>10}  {'CP':>11}  {'noCP':>11}  {'far-ref':>11}  {'BSSE':>9}  {'t / s':>7}"
      "     (kcal/mol)")
for slip in slips:
    start = time.time()
    dimer = psi4.geometry(slipped_dimer(slip))
    E_AB = energy(dimer)
    E_A_ghB = energy(dimer.extract_subsets(1, 2))
    E_B_ghA = energy(dimer.extract_subsets(2, 1))

    e_cp = E_AB - E_A_ghB - E_B_ghA
    e_nocp = E_AB - E_A - E_B
    e_far = E_AB - E_far
    rows.append({
        "slip": slip, "E_AB": E_AB, "E_A_ghB": E_A_ghB, "E_B_ghA": E_B_ghA,
        "int_cp": e_cp, "int_nocp": e_nocp, "int_far": e_far,
    })
    save()
    print(f"{slip:10.3f}  {EH_TO_KCAL * e_cp: 11.4f}  {EH_TO_KCAL * e_nocp: 11.4f}  "
          f"{EH_TO_KCAL * e_far: 11.4f}  {EH_TO_KCAL * (e_nocp - e_cp): 9.4f}  "
          f"{time.time() - start:7.1f}", flush=True)

print(f"\nwrote {RESULTS_FILE}")


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # results are saved; plot later
    print(f"skipping plot ({exc})")
else:
    x = [r["slip"] for r in rows]
    kcal = lambda key: [EH_TO_KCAL * r[key] for r in rows]
    bsse = [EH_TO_KCAL * (r["int_nocp"] - r["int_cp"]) for r in rows]

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.plot(x, kcal("int_cp"), "o-", ms=3, color="tab:blue", label="counterpoise")
    ax.plot(x, kcal("int_nocp"), "s-", ms=3, color="tab:red", label="no counterpoise")
    ax.plot(x, kcal("int_far"), "^--", ms=3, color="0.4", label=f"E(R) - E({FAR_SLIP:g} Å)")
    ax.set_xlabel("slip / Å")
    ax.set_ylabel("E$_{int}$ / kcal mol$^{-1}$")
    ax.set_title("Benzene dimer, wB97X-D / jun-cc-pVDZ (DF)")
    ax.grid(alpha=0.3)
    ax.legend()

    axb.plot(x, bsse, "o-", ms=3, color="tab:purple")
    axb.set_xlabel("slip / Å")
    axb.set_ylabel("BSSE = E$_{noCP}$ - E$_{CP}$ / kcal mol$^{-1}$")
    axb.set_title("Basis set superposition error")
    axb.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("dft_bsse_scan.png", dpi=200)
    print("wrote dft_bsse_scan.png")
