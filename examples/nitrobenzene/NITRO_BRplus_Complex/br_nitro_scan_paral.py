import os
import csv
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psi4

import psi4
import numpy as np

import sys
sys.path.insert(0, "/Users/jfoley19/Code/cqed-rhf")
from cqed_rhf.calculator import CQEDRHFCalculator


# ------------------------------------------------------------
# User controls
# ------------------------------------------------------------

N_WORKERS = 6
PSI4_THREADS_PER_WORKER = 1
PSI4_MEMORY_PER_WORKER = "1500 MB"

output_csv = "nitrobenzene_brplus_3d_scan_parallel.csv"

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

x_values = np.linspace(-3.0, 3.0, 4)
y_values = np.linspace(-3.0, 3.0, 4)
z_values = np.linspace(2.0, 3.5, 6)

minimum_br_heavy_atom_distance = 1.8


# ------------------------------------------------------------
# Geometry
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

ring_carbon_indices = [0, 1, 2, 3, 10, 12]


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------

def get_coords(atoms):
    return np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)


def compute_ring_frame(atoms, ring_indices):
    coords = get_coords(atoms)
    ring_coords = coords[ring_indices]

    center = ring_coords.mean(axis=0)
    centered = ring_coords - center

    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    e1 = vh[0]
    e2 = vh[1]
    normal = vh[2]

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
# CSV restart helpers
# ------------------------------------------------------------

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
                "status",
                "worker_pid",
            ]
        )


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


def append_result(filename, row):
    with open(filename, "a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


# ------------------------------------------------------------
# Worker initializer and worker function
# ------------------------------------------------------------

def init_worker():
    """
    Runs once inside each worker process.
    Keeps each single-point calculation single-threaded.
    """
    os.environ["OMP_NUM_THREADS"] = str(PSI4_THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(PSI4_THREADS_PER_WORKER)
    os.environ["OPENBLAS_NUM_THREADS"] = str(PSI4_THREADS_PER_WORKER)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(PSI4_THREADS_PER_WORKER)
    os.environ["NUMEXPR_NUM_THREADS"] = str(PSI4_THREADS_PER_WORKER)

    psi4.core.be_quiet()
    psi4.set_num_threads(PSI4_THREADS_PER_WORKER)
    psi4.set_memory(PSI4_MEMORY_PER_WORKER)
    psi4.set_options(psi4_options)


def run_single_point(task):
    """
    One independent scan point.

    Returns a row to be written by the parent process.
    The parent process does all CSV writing to avoid file-write races.
    """
    x_scan, y_scan, z_scan, br_position, min_dist = task

    pid = os.getpid()

    try:
        geom_str = make_geom_string(base_atoms, br_position)

        calc = CQEDRHFCalculator(
            lambda_vector=lambda_vector,
            psi4_options=psi4_options,
            omega=omega,
            charge=1,
            multiplicity=1,
            density_fitting=True,
            functional=functional,
            debug=False,
        )

        energy = float(calc.energy(geom_str))

        psi4.core.clean()

        return [
            x_scan,
            y_scan,
            z_scan,
            br_position[0],
            br_position[1],
            br_position[2],
            min_dist,
            energy,
            "ok",
            pid,
        ]

    except Exception:
        err = traceback.format_exc().replace("\n", " | ")
        try:
            psi4.core.clean()
        except Exception:
            pass

        return [
            x_scan,
            y_scan,
            z_scan,
            br_position[0],
            br_position[1],
            br_position[2],
            min_dist,
            "",
            f"failed: {err}",
            pid,
        ]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ring_center, e1, e2, normal = compute_ring_frame(base_atoms, ring_carbon_indices)

    print("Ring center:", ring_center)
    print("Ring e1:    ", e1)
    print("Ring e2:    ", e2)
    print("Ring normal:", normal)

    initialize_csv(output_csv)
    completed = load_completed_points(output_csv)

    tasks = []

    for z_scan in z_values:
        for x_scan in x_values:
            for y_scan in y_values:
                key = (
                    round(float(x_scan), 8),
                    round(float(y_scan), 8),
                    round(float(z_scan), 8),
                )

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
                                "skipped_close_contact",
                                "parent",
                            ],
                        )
                        continue

                tasks.append((x_scan, y_scan, z_scan, br_position, min_dist))

    print(f"Pending single points: {len(tasks)}")
    print(f"Using {N_WORKERS} worker processes")
    print(f"Psi4 threads per worker: {PSI4_THREADS_PER_WORKER}")
    print(f"Psi4 memory per worker: {PSI4_MEMORY_PER_WORKER}")

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=init_worker,
    ) as executor:

        futures = [executor.submit(run_single_point, task) for task in tasks]

        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            append_result(output_csv, row)

            x_scan, y_scan, z_scan = row[0], row[1], row[2]
            status = row[8]

            if status == "ok":
                print(
                    f"[{i:6d}/{len(tasks):6d}] "
                    f"OK x={x_scan:7.3f} y={y_scan:7.3f} z={z_scan:7.3f} "
                    f"E={float(row[7]):18.10f} Ha"
                )
            else:
                print(
                    f"[{i:6d}/{len(tasks):6d}] "
                    f"{status[:80]} at x={x_scan:.3f}, y={y_scan:.3f}, z={z_scan:.3f}"
                )

    print(f"Done. Results written to {output_csv}")


if __name__ == "__main__":
    # On macOS this guard is essential for multiprocessing.
    mp.set_start_method("spawn", force=True)
    main()
