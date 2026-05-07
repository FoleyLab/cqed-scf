import os
import csv
import numpy as np
import psi4

# If needed:
# from oo_cqed_rhf import CQEDRHFCalculator
import psi4
import numpy as np
from cqed_rhf.calculator import CQEDRHFCalculator
# ─────────────────────────────────────────────
# Psi4 basic config
# ─────────────────────────────────────────────
psi4.set_memory("10 GB")
psi4.set_num_threads(2)
psi4.set_output_file("psi4_env_check.out", False)
psi4.core.clean()
psi4.core.clean_options()


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------

psi4_options = {
    "basis": "6-311G*",
    "reference": "rks",
    "scf_type": "df",
    "e_convergence": 1e-9,
    "d_convergence": 1e-9,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none",
}

lambda_vector = np.array([0.078, 0.055, 0.027])
omega = 0.06615
functional = "wb97x-d"

# 3D raster scan settings, in Angstrom
x_values = np.linspace(-3.0, 3.0, 10)
y_values = np.linspace(-3.0, 3.0, 10)
z_values = np.linspace(2.0, 3.5, 10)

output_csv = "nitrobenzene_brplus_3d_scan.csv"

# Optional: skip high-energy close contacts by enforcing a minimum Br-heavy-atom distance
minimum_br_heavy_atom_distance = 1.8  # Angstrom; set to None to disable


# ------------------------------------------------------------
# Nitrobenzene geometry
# ------------------------------------------------------------

base_atoms = [
    ("C", -1.468931365599,  1.178697647085, -1.610476189598),
    ("C", -0.081420645599,  1.196870987085, -1.586337729598),
    ("C",  0.594038404401, -0.009641582915, -1.487579279598),
    ("C", -2.153305305599, -0.029816812915, -1.536826409598),
    ("H", -2.016475245599,  2.111128687085, -1.687252519598),
    ("H",  0.478758204401,  2.120622867085, -1.641899559598),
    ("H", -3.237744725599, -0.037851282915, -1.556289569598),
    ("N",  2.070250494401,  0.001301307085, -1.461192949598),
    ("O",  2.638548954401, -1.067946332915, -1.375316239598),
    ("O",  2.625214324401,  1.078933737085, -1.526960829598),
    ("C", -1.454272445599, -1.228165032915, -1.438287069598),
    ("H", -1.990524005599, -2.168537162915, -1.380997049598),
    ("C", -0.066668565599, -1.225978212915, -1.412581159598),
    ("H",  0.504648244401, -2.141351302915, -1.336153489598),
]

# The six aromatic carbons are atom indices 0, 1, 2, 3, 10, 12.
ring_carbon_indices = [0, 1, 2, 3, 10, 12]


# ------------------------------------------------------------
# Geometry utilities
# ------------------------------------------------------------

def get_coords(atoms):
    return np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)


def compute_ring_frame(atoms, ring_indices):
    """
    Returns ring_center, e1, e2, normal.

    e1 and e2 span the best-fit ring plane.
    normal is the ring-plane normal.
    """
    coords = get_coords(atoms)
    ring_coords = coords[ring_indices]

    center = ring_coords.mean(axis=0)
    centered = ring_coords - center

    # PCA/SVD best-fit plane.
    # Last right-singular vector is the normal to the plane.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    e1 = vh[0]
    e2 = vh[1]
    normal = vh[2]

    # Make the normal point roughly toward positive Cartesian z for consistency.
    if normal[2] < 0:
        normal *= -1.0
        e2 *= -1.0

    return center, e1, e2, normal


def make_geom_string(atoms, br_position, charge=1, multiplicity=1):
    lines = [f"{charge} {multiplicity}"]

    for symbol, x, y, z in atoms:
        lines.append(f"{symbol:2s} {x:18.12f} {y:18.12f} {z:18.12f}")

    bx, by, bz = br_position
    lines.append(f"Br {bx:18.12f} {by:18.12f} {bz:18.12f}")

    lines.append("units angstrom")
    lines.append("no_com")
    lines.append("no_reorient")
    lines.append("symmetry c1")

    return "\n".join(lines)


def min_distance_to_heavy_atoms(atoms, br_position):
    coords = get_coords(atoms)
    symbols = [a[0] for a in atoms]

    heavy_coords = np.array(
        [coords[i] for i, s in enumerate(symbols) if s.upper() != "H"],
        dtype=float,
    )

    distances = np.linalg.norm(heavy_coords - br_position[None, :], axis=1)
    return float(distances.min())


# ------------------------------------------------------------
# Calculator wrapper
# ------------------------------------------------------------

def make_calculator(initial_geom):
    """
    Supports both possible CQEDRHFCalculator APIs:

    1. Newer style:
       CQEDRHFCalculator(lambda_vector=..., psi4_options=..., omega=...,
                         charge=..., multiplicity=..., density_fitting=...,
                         functional=..., debug=...)

    2. Uploaded oo_cqed_rhf.py style:
       CQEDRHFCalculator(lambda_vector, molecule_string, psi4_options, omega)
    """
    try:
        return CQEDRHFCalculator(
            lambda_vector=lambda_vector,
            psi4_options=psi4_options,
            omega=omega,
            charge=1,
            multiplicity=1,
            density_fitting=True,
            functional=functional,
            debug=False,
        )
    except TypeError:
        return CQEDRHFCalculator(
            lambda_vector=lambda_vector,
            molecule_string=initial_geom,
            psi4_options=psi4_options,
            omega=omega,
        )


def qed_energy(calc, geom_str):
    """
    Supports either calc.energy(geom_str) or the uploaded style where
    calc.calc_cqed_rhf_energy() updates calc.cqed_rhf_energy.
    """
    if hasattr(calc, "energy"):
        return float(calc.energy(geom_str))

    calc.molecule_string = geom_str
    calc.calc_cqed_rhf_energy()
    return float(calc.cqed_rhf_energy)


# ------------------------------------------------------------
# Restart helpers
# ------------------------------------------------------------

def load_completed_points(filename):
    completed = set()

    if not os.path.exists(filename):
        return completed

    with open(filename, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            completed.add(
                (
                    round(float(row["x_scan_A"]), 8),
                    round(float(row["y_scan_A"]), 8),
                    round(float(row["z_scan_A"]), 8),
                )
            )

    return completed


def initialize_csv(filename):
    if os.path.exists(filename):
        return

    with open(filename, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_scan_A",
                "y_scan_A",
                "z_scan_A",
                "Br_x_A",
                "Br_y_A",
                "Br_z_A",
                "min_Br_heavy_atom_distance_A",
                "energy_Ha",
                "relative_energy_kcal_mol",
                "status",
            ]
        )


def append_result(filename, row):
    with open(filename, "a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


# ------------------------------------------------------------
# Main scan
# ------------------------------------------------------------

psi4.set_options(psi4_options)

ring_center, e1, e2, normal = compute_ring_frame(base_atoms, ring_carbon_indices)

print("Ring center:", ring_center)
print("Ring e1:    ", e1)
print("Ring e2:    ", e2)
print("Ring normal:", normal)

initial_br_position = ring_center + x_values[0] * e1 + y_values[0] * e2 + z_values[0] * normal
initial_geom = make_geom_string(base_atoms, initial_br_position)

calc = make_calculator(initial_geom)

initialize_csv(output_csv)
completed = load_completed_points(output_csv)

hartree_to_kcal_mol = 627.509474

raw_results = []

# If restarting, read previous energies so relative energies can be updated later externally.
if os.path.exists(output_csv):
    with open(output_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["status"] == "ok":
                raw_results.append(float(row["energy_Ha"]))

current_min_energy = min(raw_results) if raw_results else None

n_total = len(x_values) * len(y_values) * len(z_values)
n_done = len(completed)
print(f"Total scan points: {n_total}")
print(f"Already completed: {n_done}")

for z_scan in z_values:
    for x_scan in x_values:
        for y_scan in y_values:

            key = (round(float(x_scan), 8), round(float(y_scan), 8), round(float(z_scan), 8))
            if key in completed:
                continue

            br_position = ring_center + x_scan * e1 + y_scan * e2 + z_scan * normal
            min_dist = min_distance_to_heavy_atoms(base_atoms, br_position)

            if minimum_br_heavy_atom_distance is not None:
                if min_dist < minimum_br_heavy_atom_distance:
                    append_result(
                        output_csv,
                        [
                            x_scan,
                            y_scan,
                            z_scan,
                            br_position[0],
                            br_position[1],
                            br_position[2],
                            min_dist,
                            "",
                            "",
                            "skipped_close_contact",
                        ],
                    )
                    print(
                        f"Skipped close contact: "
                        f"x={x_scan:.3f}, y={y_scan:.3f}, z={z_scan:.3f}, "
                        f"min_dist={min_dist:.3f} A"
                    )
                    continue

            geom_str = make_geom_string(base_atoms, br_position)

            try:
                energy = qed_energy(calc, geom_str)

                if current_min_energy is None or energy < current_min_energy:
                    current_min_energy = energy

                rel_kcal = (energy - current_min_energy) * hartree_to_kcal_mol

                append_result(
                    output_csv,
                    [
                        x_scan,
                        y_scan,
                        z_scan,
                        br_position[0],
                        br_position[1],
                        br_position[2],
                        min_dist,
                        energy,
                        rel_kcal,
                        "ok",
                    ],
                )

                print(
                    f"OK x={x_scan:8.3f} y={y_scan:8.3f} z={z_scan:8.3f} "
                    f"E={energy:18.10f} Ha  "
                    f"rel_to_current_min={rel_kcal:12.6f} kcal/mol"
                )

            except Exception as exc:
                append_result(
                    output_csv,
                    [
                        x_scan,
                        y_scan,
                        z_scan,
                        br_position[0],
                        br_position[1],
                        br_position[2],
                        min_dist,
                        "",
                        "",
                        f"failed: {repr(exc)}",
                    ],
                )

                print(
                    f"FAILED x={x_scan:.3f}, y={y_scan:.3f}, z={z_scan:.3f}: {repr(exc)}"
                )

            # Helps Psi4 release memory between many single-point calculations.
            psi4.core.clean()

print(f"Scan complete. Results written to {output_csv}")
