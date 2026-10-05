"""
QED-SAPT0 (density-fitted) benzene dimer slip scan for three cavity polarizations.

Geometry construction
---------------------
The base dimer is a parallel-displaced pair whose ring normal is tilted ~6 degrees
from the lab z axis.  Both monomers are rotated into a stack frame centered on the
centroid of monomer A:

    z' : normal of monomer A's ring plane, pointing toward monomer B
         (the inter-monomer / stacking coordinate)
    x' : in-plane direction of the base dimer's own slip offset (the slip coordinate)
    y' : z' cross x' (in-plane, perpendicular to the slip)

Monomer B is then placed in a true stack (centroid directly above A's centroid at
the base interplane distance, rings parallel -- B is an exact translate of A in the
base geometry) and rigidly translated along x' by the slip distance.  Because the
frame axes are the Cartesian axes, lambda along lab x/y/z means slip/perp/stack.

no_reorient / no_com keep Psi4 from re-centering each frame, which would silently
change the cavity coupling direction.
"""

import argparse
import json
import time

import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig


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


parser = argparse.ArgumentParser(description="Benzene dimer QED-SAPT0 slip scan")
parser.add_argument("--components", action="store_true",
                    help="plot each SAPT component instead of only the total")
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
print(f"ring-normal angle   : {np.degrees(np.arccos(abs(ring_normal(xyz_a) @ ring_normal(xyz_b)))):.3f} deg")


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


slips = np.linspace(0, 8,81) 
#slips = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
geometries = {s: slipped_dimer(s) for s in slips}


# ---------------------------------------------------------
# Psi4 / CQED configuration
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "jun-cc-pVDZ",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

LAMBDA = 0.05
polarizations = {
    "none (lambda=0)": np.zeros(3),
    "x (slip)": np.array([LAMBDA, 0.0, 0.0]),
    "y (perp)": np.array([0.0, LAMBDA, 0.0]),
    "z (stack)": np.array([0.0, 0.0, LAMBDA]),
}

base_config = CQEDConfig(
    lambda_vector=np.zeros(3),
    omega=0.1,
    psi4_options=psi4_options,
    reference="rhf",
    functional=None,
    density_fitting=True,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=False,
    quiet=True,
)


def run_point(calc, dimer):
    psi4.core.clean()
    start = time.time()
    components = calc.sapt0_components(
        psi4.geometry(dimer),
        integral_backend="df",
        include_cavity_terms=True,
        monomer_reference_frame="monomer_com",
    )
    return components, time.time() - start


# ---------------------------------------------------------
# Run the scan
# ---------------------------------------------------------

results = {}
print("\nQED-SAPT0 benzene dimer slip scan")
print("=================================")
print(f"omega = {base_config.omega} Eh   lambda = {LAMBDA}   basis = jun-cc-pVDZ\n")

for label, lam in polarizations.items():
    calc = CQEDCalculator(config=base_config.copy_with(lambda_vector=lam))
    print(f"--- polarization: {label} ---")
    print(f"{'slip / Ang':>10}  {'total / Eh':>16}  {'elst':>13}  {'exch':>13}  "
          f"{'ind':>13}  {'disp':>13}  {'t / s':>7}")
    for slip, dimer in geometries.items():
        comp, t = run_point(calc, dimer)
        results[(label, slip)] = comp
        print(f"{slip:10.2f}  {comp.total: 16.10f}  {comp.elst10: 13.8f}  "
              f"{comp.exch10: 13.8f}  {comp.ind20 + comp.exch_ind20: 13.8f}  "
              f"{comp.disp20 + comp.exch_disp20: 13.8f}  {t:7.1f}", flush=True)
    print()


# ---------------------------------------------------------
# Cavity-induced shift of the total interaction energy
# ---------------------------------------------------------

RESULTS_FILE = "slip_scan_results.json"
COMPONENT_KEYS = ("elst10", "exch10", "ind20", "exch_ind20", "disp20", "exch_disp20", "total")
with open(RESULTS_FILE, "w") as fh:
    json.dump(
        {
            "meta": {"lambda": LAMBDA, "omega": base_config.omega, "basis": psi4_options["basis"]},
            "results": {
                label: {
                    str(slip): {k: float(getattr(results[(label, slip)], k)) for k in COMPONENT_KEYS}
                    for slip in geometries
                }
                for label in polarizations
            },
        },
        fh,
        indent=2,
    )
print(f"wrote {RESULTS_FILE}")

try:
    import plot_benzene_slip_scan

    plot_benzene_slip_scan.plot(RESULTS_FILE, components=args.components)
except ImportError as exc:  # matplotlib missing; results are saved, plot later
    print(f"skipping plot ({exc}); run plot_benzene_slip_scan.py on {RESULTS_FILE}")

print("Total QED-SAPT0 minus lambda = 0 (mEh)")
print("======================================")
labels = [l for l in polarizations if not l.startswith("none")]
print(f"{'slip / Ang':>10}  " + "  ".join(f"{l:>12}" for l in labels))
for slip in geometries:
    ref = results[("none (lambda=0)", slip)].total
    print(f"{slip:10.2f}  " + "  ".join(
        f"{1e3 * (results[(l, slip)].total - ref):12.5f}" for l in labels))
