#!/usr/bin/env python3
"""
QED-wB97X-D benzene-water interaction-energy scan.

Input coordinate format:

# Geometry 1: d = 0.90
C ...
...
H ...

# Geometry 2: d = 0.95
...

Assumptions for each geometry block:
    atoms 1-12  = benzene
    atoms 13-15 = water

The script computes, for each geometry:
    E_dimer
    E_benzene_raw
    E_water_raw
    E_benzene_CP   using water ghost basis functions
    E_water_CP     using benzene ghost basis functions
    raw and CP-corrected interaction energies

Run:
    python benzene_water_qed_wb97xd_scan.py --coords benzene_water_geoms.txt

Optional parallel run:
    python benzene_water_qed_wb97xd_scan.py --coords benzene_water_geoms.txt --workers 4
"""

import argparse
import csv
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psi4

# -----------------------------------------------------------------------------
# Import your updated QED-DFT calculator
# -----------------------------------------------------------------------------
# Edit this path/module name if needed.
# Example if running from /Users/jfoley19/Code/cqed-rhf/examples/...:
sys.path.insert(0, "/Users/jfoley19/Code/cqed-rhf")

try:
    from cqed_rhf.calculator import CQEDRHFCalculator
except ImportError:
    # Customize this fallback for your local repo layout if necessary.
    sys.path.insert(0, os.path.expanduser("~/Code/cqed-rhf"))
    from cqed_rhf.calculator import CQEDRHFCalculator


# -----------------------------------------------------------------------------
# QED-DFT settings
# -----------------------------------------------------------------------------

PSI4_OPTIONS = {
    "basis": "6-311G*",
    "reference": "rks",
    "scf_type": "df",
    "e_convergence": 1e-9,
    "d_convergence": 1e-9,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none",
}

LAMBDA_VECTOR = np.array([0.078, 0.055, 0.027])
OMEGA = 0.06615
FUNCTIONAL = "wb97x-d"
HARTREE_TO_KCAL_MOL = 627.509474


# -----------------------------------------------------------------------------
# Parsing and geometry construction
# -----------------------------------------------------------------------------

HEADER_RE = re.compile(
    r"^\s*#\s*Geometry\s+(?P<index>\d+)\s*:\s*d\s*=\s*(?P<distance>[-+]?\d*\.?\d+)\s*$",
    re.IGNORECASE,
)


def parse_coordinate_file(filename):
    """Parse the pasted coordinate format into geometry dictionaries."""
    geometries = []
    current = None

    with open(filename, "r") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            header_match = HEADER_RE.match(line)
            if header_match:
                if current is not None:
                    validate_geometry(current)
                    geometries.append(current)

                current = {
                    "index": int(header_match.group("index")),
                    "distance_A": float(header_match.group("distance")),
                    "atoms": [],
                }
                continue

            if current is None:
                raise ValueError(
                    f"Found atom line before first geometry header at line {line_number}: {line}"
                )

            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Could not parse line {line_number}: {line}")

            symbol = parts[0]
            x, y, z = map(float, parts[1:])
            current["atoms"].append((symbol, x, y, z))

    if current is not None:
        validate_geometry(current)
        geometries.append(current)

    if not geometries:
        raise ValueError("No geometry blocks were found.")

    return geometries


def validate_geometry(geom):
    atoms = geom["atoms"]
    if len(atoms) != 15:
        raise ValueError(
            f"Geometry {geom['index']} at d={geom['distance_A']} has {len(atoms)} atoms; expected 15."
        )

    benzene = atoms[:12]
    water = atoms[12:]

    benzene_formula = "".join(atom[0] for atom in benzene)
    water_formula = "".join(atom[0] for atom in water)

    if sum(1 for atom in benzene if atom[0].upper() == "C") != 6:
        raise ValueError(f"Geometry {geom['index']} first 12 atoms do not contain 6 carbons.")

    if sum(1 for atom in benzene if atom[0].upper() == "H") != 6:
        raise ValueError(f"Geometry {geom['index']} first 12 atoms do not contain 6 hydrogens.")

    if water_formula.upper() != "OHH":
        raise ValueError(
            f"Geometry {geom['index']} atoms 13-15 are {water_formula}; expected OHH."
        )


def atom_line(atom, ghost=False):
    symbol, x, y, z = atom
    if ghost:
        symbol = f"Gh({symbol})"
    return f"{symbol:6s} {x:18.10f} {y:18.10f} {z:18.10f}"


def make_geom(atoms, charge=0, multiplicity=1):
    lines = [f"{charge} {multiplicity}"]
    lines.extend(atom_line(atom, ghost=False) for atom in atoms)
    lines.extend([
        "units angstrom",
        "no_com",
        "no_reorient",
        "symmetry c1",
    ])
    return "\n".join(lines)


def make_cp_benzene_geom(benzene_atoms, water_atoms):
    lines = ["0 1"]
    lines.extend(atom_line(atom, ghost=False) for atom in benzene_atoms)
    lines.extend(atom_line(atom, ghost=True) for atom in water_atoms)
    lines.extend([
        "units angstrom",
        "no_com",
        "no_reorient",
        "symmetry c1",
    ])
    return "\n".join(lines)


def make_cp_water_geom(benzene_atoms, water_atoms):
    lines = ["0 1"]
    lines.extend(atom_line(atom, ghost=True) for atom in benzene_atoms)
    lines.extend(atom_line(atom, ghost=False) for atom in water_atoms)
    lines.extend([
        "units angstrom",
        "no_com",
        "no_reorient",
        "symmetry c1",
    ])
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# QED energy wrapper
# -----------------------------------------------------------------------------


def configure_psi4(memory, threads, quiet=True):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

    if quiet:
        psi4.core.be_quiet()

    psi4.set_memory(memory)
    psi4.set_num_threads(threads)
    psi4.set_options(PSI4_OPTIONS)


def qed_energy(geom_str):
    calc = CQEDRHFCalculator(
        lambda_vector=LAMBDA_VECTOR,
        psi4_options=PSI4_OPTIONS,
        omega=OMEGA,
        charge=0,
        multiplicity=1,
        density_fitting=True,
        functional=FUNCTIONAL,
        debug=False,
    )
    energy = float(calc.energy(geom_str))
    psi4.core.clean()
    return energy


def compute_one_geometry(geom):
    index = geom["index"]
    distance_A = geom["distance_A"]
    atoms = geom["atoms"]

    benzene_atoms = atoms[:12]
    water_atoms = atoms[12:]

    dimer_geom = make_geom(benzene_atoms + water_atoms)
    benzene_raw_geom = make_geom(benzene_atoms)
    water_raw_geom = make_geom(water_atoms)
    benzene_cp_geom = make_cp_benzene_geom(benzene_atoms, water_atoms)
    water_cp_geom = make_cp_water_geom(benzene_atoms, water_atoms)

    E_dimer = qed_energy(dimer_geom)
    E_benzene_raw = qed_energy(benzene_raw_geom)
    E_water_raw = qed_energy(water_raw_geom)
    E_benzene_cp = qed_energy(benzene_cp_geom)
    E_water_cp = qed_energy(water_cp_geom)

    E_int_raw = E_dimer - E_benzene_raw - E_water_raw
    E_int_cp = E_dimer - E_benzene_cp - E_water_cp

    return {
        "geometry_index": index,
        "distance_A": distance_A,
        "E_dimer_Ha": E_dimer,
        "E_benzene_raw_Ha": E_benzene_raw,
        "E_water_raw_Ha": E_water_raw,
        "E_benzene_CP_Ha": E_benzene_cp,
        "E_water_CP_Ha": E_water_cp,
        "E_int_raw_Ha": E_int_raw,
        "E_int_raw_kcal_mol": E_int_raw * HARTREE_TO_KCAL_MOL,
        "E_int_CP_Ha": E_int_cp,
        "E_int_CP_kcal_mol": E_int_cp * HARTREE_TO_KCAL_MOL,
        "status": "ok",
    }


def worker_init(memory, threads):
    configure_psi4(memory=memory, threads=threads, quiet=True)
    scratch = f"/tmp/psi4_benzene_water_worker_{os.getpid()}"
    os.makedirs(scratch, exist_ok=True)
    psi4.core.IOManager.shared_object().set_default_path(scratch)


def worker_compute(geom):
    try:
        return compute_one_geometry(geom)
    except Exception as exc:
        return {
            "geometry_index": geom.get("index"),
            "distance_A": geom.get("distance_A"),
            "E_dimer_Ha": "",
            "E_benzene_raw_Ha": "",
            "E_water_raw_Ha": "",
            "E_benzene_CP_Ha": "",
            "E_water_CP_Ha": "",
            "E_int_raw_Ha": "",
            "E_int_raw_kcal_mol": "",
            "E_int_CP_Ha": "",
            "E_int_CP_kcal_mol": "",
            "status": f"failed: {type(exc).__name__}: {exc}; traceback: {traceback.format_exc().replace(chr(10), ' | ')}",
        }


# -----------------------------------------------------------------------------
# CSV handling
# -----------------------------------------------------------------------------

FIELDNAMES = [
    "geometry_index",
    "distance_A",
    "E_dimer_Ha",
    "E_benzene_raw_Ha",
    "E_water_raw_Ha",
    "E_benzene_CP_Ha",
    "E_water_CP_Ha",
    "E_int_raw_Ha",
    "E_int_raw_kcal_mol",
    "E_int_CP_Ha",
    "E_int_CP_kcal_mol",
    "status",
]


def load_finished_indices(output_csv):
    finished = set()
    if not os.path.exists(output_csv):
        return finished

    with open(output_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "ok":
                finished.add(int(row["geometry_index"]))
    return finished


def append_row(output_csv, row):
    exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords", required=True, help="Coordinate text file.")
    parser.add_argument("--output", default="benzene_water_qed_wb97xd_interaction_energies.csv")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--memory-per-worker", default="1500 MB")
    parser.add_argument("--rerun-failed", action="store_true", help="Failed rows are always rerun; this flag is kept for clarity.")
    args = parser.parse_args()

    geometries = parse_coordinate_file(args.coords)
    finished = load_finished_indices(args.output)
    pending = [geom for geom in geometries if geom["index"] not in finished]

    print(f"Parsed {len(geometries)} geometries from {args.coords}")
    print(f"Already finished: {len(finished)}")
    print(f"Pending: {len(pending)}")
    print(f"Workers: {args.workers}")

    if args.workers == 1:
        configure_psi4(memory=args.memory_per_worker, threads=args.threads_per_worker, quiet=False)
        for geom in pending:
            row = worker_compute(geom)
            append_row(args.output, row)
            print(
                f"Geometry {row['geometry_index']:3d}  d={float(row['distance_A']):6.2f} A  "
                f"status={row['status']}  "
                f"Eint_CP={row['E_int_CP_kcal_mol']}"
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=worker_init,
            initargs=(args.memory_per_worker, args.threads_per_worker),
        ) as executor:
            futures = [executor.submit(worker_compute, geom) for geom in pending]
            for future in as_completed(futures):
                row = future.result()
                append_row(args.output, row)
                print(
                    f"Geometry {row['geometry_index']:3d}  d={float(row['distance_A']):6.2f} A  "
                    f"status={row['status']}  "
                    f"Eint_CP={row['E_int_CP_kcal_mol']}"
                )

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
