#!/usr/bin/env python3
"""
Serial QED-wB97X-D benzene-water interaction-energy scan.

This script hard-codes the benzene-water geometries.  For each geometry it computes:

    E_dimer
    E_benzene_raw
    E_water_raw
    E_benzene_CP
    E_water_CP

and writes raw and counterpoise-corrected interaction energies to CSV.

Assumptions:
  - Each hard-coded geometry has 15 atoms.
  - Atoms 1-12 are benzene.
  - Atoms 13-15 are water.
  - The system is neutral singlet.
"""

import csv
import os
import sys
import traceback

import numpy as np
import psi4

# -----------------------------------------------------------------------------
# Import your updated CQEDCalculator.
# Edit this block if your local module path/name is different.
# -----------------------------------------------------------------------------
from cqed_scf.calculator import CQEDCalculator


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

OUTPUT_CSV = "benzene_water_qed_wb97xd_aug_cc_pVTZ_no_cavity.csv"

psi4_options = {
    "basis": "aug-cc-pVTZ",
    "reference": "rks",
    "scf_type": "df",
    "e_convergence": 1e-9,
    "d_convergence": 1e-9,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_pruning_scheme": "none",
}

lambda_vector = np.array([0.0, 0.0, 0.0]) 
omega = 0.06615
functional = "wb97x-d"

hartree_to_kcal_mol = 627.509474

# Set Psi4 to serial mode for this driver.
psi4.core.be_quiet()
psi4.set_num_threads(1)
psi4.set_memory("4000 MB")
psi4.set_options(psi4_options)


# -----------------------------------------------------------------------------
# Hard-coded coordinate series
# -----------------------------------------------------------------------------

GEOMETRIES = [
    {
        "label": "Geometry 1",
        "distance_A": 0.90,
        "atoms": [
            'C      0.780147171000    -0.609914733000    -1.207556891000',
            'H      0.896191595000    -1.137639594000    -2.144144625000',
            'C      0.477942753000     0.750993631000    -1.207895407000',
            'H      0.356964231000     1.278167803000    -2.144054074000',
            'C      0.327289279000     1.431867868000    -0.000000000000',
            'H      0.091465028000     2.487139215000     0.000000000000',
            'C      0.477942754000     0.750993631000     1.207895407000',
            'H      0.356964231000     1.278167803000     2.144054074000',
            'C      0.780147171000    -0.609914733000     1.207556891000',
            'H      0.896191595000    -1.137639594000     2.144144625000',
            'C      0.931648311000    -1.289981342000     0.000000000000',
            'H      1.168485730000    -2.345213690000    -0.000000000000',
            'O     -2.499730195000    -0.241863092000     0.000000000000',
            'H     -2.334926200000    -1.186584620000     0.000000000000',
            'H     -1.612429252000     0.129727233000     0.000000000000',
        ],
    },
    {
        "label": "Geometry 2",
        "distance_A": 0.95,
        "atoms": [
            'C      0.780147171000    -0.609914733000    -1.207556891000',
            'H      0.896191595000    -1.137639594000    -2.144144625000',
            'C      0.477942753000     0.750993631000    -1.207895407000',
            'H      0.356964231000     1.278167803000    -2.144054074000',
            'C      0.327289279000     1.431867868000    -0.000000000000',
            'H      0.091465028000     2.487139215000     0.000000000000',
            'C      0.477942754000     0.750993631000     1.207895407000',
            'H      0.356964231000     1.278167803000     2.144054074000',
            'C      0.780147171000    -0.609914733000     1.207556891000',
            'H      0.896191595000    -1.137639594000     2.144144625000',
            'C      0.931648311000    -1.289981342000     0.000000000000',
            'H      1.168485730000    -2.345213690000    -0.000000000000',
            'O     -2.644077504000    -0.258065567000     0.000000000000',
            'H     -2.479273509000    -1.202787095000     0.000000000000',
            'H     -1.756776561000     0.113524758000     0.000000000000',
        ],
    },
    {
        "label": "Geometry 3",
        "distance_A": 1.00,
        "atoms": [
            'C      0.780147171000    -0.609914733000    -1.207556891000',
            'H      0.896191595000    -1.137639594000    -2.144144625000',
            'C      0.477942753000     0.750993631000    -1.207895407000',
            'H      0.356964231000     1.278167803000    -2.144054074000',
            'C      0.327289279000     1.431867868000    -0.000000000000',
            'H      0.091465028000     2.487139215000     0.000000000000',
            'C      0.477942754000     0.750993631000     1.207895407000',
            'H      0.356964231000     1.278167803000     2.144054074000',
            'C      0.780147171000    -0.609914733000     1.207556891000',
            'H      0.896191595000    -1.137639594000     2.144144625000',
            'C      0.931648311000    -1.289981342000     0.000000000000',
            'H      1.168485730000    -2.345213690000    -0.000000000000',
            'O     -2.786535929000    -0.274056021000    -0.000000000000',
            'H     -2.621731934000    -1.218777549000    -0.000000000000',
            'H     -1.899234986000     0.097534304000    -0.000000000000',
        ],
    },
    {
        "label": "Geometry 4",
        "distance_A": 1.05,
        "atoms": [
            'C      0.780147171000    -0.609914733000    -1.207556891000',
            'H      0.896191595000    -1.137639594000    -2.144144625000',
            'C      0.477942753000     0.750993631000    -1.207895407000',
            'H      0.356964231000     1.278167803000    -2.144054074000',
            'C      0.327289279000     1.431867868000    -0.000000000000',
            'H      0.091465028000     2.487139215000     0.000000000000',
            'C      0.477942754000     0.750993631000     1.207895407000',
            'H      0.356964231000     1.278167803000     2.144054074000',
            'C      0.780147171000    -0.609914733000     1.207556891000',
            'H      0.896191595000    -1.137639594000     2.144144625000',
            'C      0.931648311000    -1.289981342000     0.000000000000',
            'H      1.168485730000    -2.345213690000    -0.000000000000',
            'O     -2.927711248000    -0.289902451000    -0.000000000000',
            'H     -2.762907253000    -1.234623979000    -0.000000000000',
            'H     -2.040410305000     0.081687874000    -0.000000000000',
        ],
    },
    {
        "label": "Geometry 5",
        "distance_A": 1.10,
        "atoms": [
            'C      0.780147171000    -0.609914733000    -1.207556891000',
            'H      0.896191595000    -1.137639594000    -2.144144625000',
            'C      0.477942753000     0.750993631000    -1.207895407000',
            'H      0.356964231000     1.278167803000    -2.144054074000',
            'C      0.327289279000     1.431867868000    -0.000000000000',
            'H      0.091465028000     2.487139215000     0.000000000000',
            'C      0.477942754000     0.750993631000     1.207895407000',
            'H      0.356964231000     1.278167803000     2.144054074000',
            'C      0.780147171000    -0.609914733000     1.207556891000',
            'H      0.896191595000    -1.137639594000     2.144144625000',
            'C      0.931648311000    -1.289981342000     0.000000000000',
            'H      1.168485730000    -2.345213690000    -0.000000000000',
            'O     -3.067544023000    -0.305598185000    -0.000000000000',
            'H     -2.902740028000    -1.250319713000    -0.000000000000',
            'H     -2.180243080000     0.065992140000    -0.000000000000',
        ],
    },
    {
        "label": "Geometry 6",
        "distance_A": 1.25,
        "atoms": [
            'C      0.780147171000    -0.609914733000    -1.207556891000',
            'H      0.896191595000    -1.137639594000    -2.144144625000',
            'C      0.477942753000     0.750993631000    -1.207895407000',
            'H      0.356964231000     1.278167803000    -2.144054074000',
            'C      0.327289279000     1.431867868000    -0.000000000000',
            'H      0.091465028000     2.487139215000     0.000000000000',
            'C      0.477942754000     0.750993631000     1.207895407000',
            'H      0.356964231000     1.278167803000     2.144054074000',
            'C      0.780147171000    -0.609914733000     1.207556891000',
            'H      0.896191595000    -1.137639594000     2.144144625000',
            'C      0.931648311000    -1.289981342000     0.000000000000',
            'H      1.168485730000    -2.345213690000    -0.000000000000',
            'O     -3.481290900000    -0.352039808000    -0.000000000000',
            'H     -3.316486905000    -1.296761336000    -0.000000000000',
            'H     -2.593989957000     0.019550517000    -0.000000000000',
        ],
    },
    {
        "label": "Geometry 7",
        "distance_A": 1.50,
        "atoms": [
            'C      0.780147171000    -0.609914733000    -1.207556891000',
            'H      0.896191595000    -1.137639594000    -2.144144625000',
            'C      0.477942753000     0.750993631000    -1.207895407000',
            'H      0.356964231000     1.278167803000    -2.144054074000',
            'C      0.327289279000     1.431867868000    -0.000000000000',
            'H      0.091465028000     2.487139215000     0.000000000000',
            'C      0.477942754000     0.750993631000     1.207895407000',
            'H      0.356964231000     1.278167803000     2.144054074000',
            'C      0.780147171000    -0.609914733000     1.207556891000',
            'H      0.896191595000    -1.137639594000     2.144144625000',
            'C      0.931648311000    -1.289981342000     0.000000000000',
            'H      1.168485730000    -2.345213690000    -0.000000000000',
            'O     -4.158058556000    -0.428004583000    -0.000000000000',
            'H     -3.993254561000    -1.372726111000    -0.000000000000',
            'H     -3.270757613000    -0.056414258000    -0.000000000000',
        ],
    },
    {
        "label": "Geometry 8",
        "distance_A": 2.00,
        "atoms": [
            'C      0.780147171000    -0.609914733000    -1.207556891000',
            'H      0.896191595000    -1.137639594000    -2.144144625000',
            'C      0.477942753000     0.750993631000    -1.207895407000',
            'H      0.356964231000     1.278167803000    -2.144054074000',
            'C      0.327289279000     1.431867868000    -0.000000000000',
            'H      0.091465028000     2.487139215000     0.000000000000',
            'C      0.477942754000     0.750993631000     1.207895407000',
            'H      0.356964231000     1.278167803000     2.144054074000',
            'C      0.780147171000    -0.609914733000     1.207556891000',
            'H      0.896191595000    -1.137639594000     2.144144625000',
            'C      0.931648311000    -1.289981342000     0.000000000000',
            'H      1.168485730000    -2.345213690000    -0.000000000000',
            'O     -5.486606793000    -0.577129423000    -0.000000000000',
            'H     -5.321802798000    -1.521850951000    -0.000000000000',
            'H     -4.599305850000    -0.205539098000    -0.000000000000',
        ],
    },
]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def make_calc(charge=0, multiplicity=1):
    return CQEDCalculator(
        lambda_vector=lambda_vector,
        psi4_options=psi4_options,
        omega=omega,
        charge=charge,
        multiplicity=multiplicity,
        density_fitting=True,
        functional=functional,
        debug=False,
    )


def qed_energy(geom_str, charge=0, multiplicity=1):
    calc = make_calc(charge=charge, multiplicity=multiplicity)
    energy = float(calc.energy(geom_str))
    psi4.core.clean()
    return energy


def make_geom(atom_lines, charge=0, multiplicity=1):
    return (
        f"{charge} {multiplicity}\n"
        + "\n".join(atom_lines)
        + "\nunits angstrom\n"
        + "no_com\n"
        + "no_reorient\n"
        + "symmetry c1\n"
    )


def ghost_line(atom_line):
    parts = atom_line.split()
    symbol = parts[0]
    coords = parts[1:]
    return f"Gh({symbol}) " + " ".join(coords)


def split_benzene_water(atom_lines):
    if len(atom_lines) != 15:
        raise ValueError(f"Expected 15 atoms, got {len(atom_lines)}")
    benzene = atom_lines[:12]
    water = atom_lines[12:15]
    return benzene, water


def compute_one_geometry(entry):
    label = entry["label"]
    distance_A = entry["distance_A"]
    atom_lines = entry["atoms"]

    benzene, water = split_benzene_water(atom_lines)

    dimer_lines = benzene + water

    benzene_raw_lines = benzene
    water_raw_lines = water

    benzene_cp_lines = benzene + [ghost_line(line) for line in water]
    water_cp_lines = [ghost_line(line) for line in benzene] + water

    dimer_geom = make_geom(dimer_lines, charge=0, multiplicity=1)
    benzene_raw_geom = make_geom(benzene_raw_lines, charge=0, multiplicity=1)
    water_raw_geom = make_geom(water_raw_lines, charge=0, multiplicity=1)
    benzene_cp_geom = make_geom(benzene_cp_lines, charge=0, multiplicity=1)
    water_cp_geom = make_geom(water_cp_lines, charge=0, multiplicity=1)

    print(f"\n=== {label}: d = {distance_A:.2f} A ===")

    print("  Dimer energy...")
    E_dimer = qed_energy(dimer_geom)

    print("  Benzene raw monomer energy...")
    E_benzene_raw = qed_energy(benzene_raw_geom)

    print("  Water raw monomer energy...")
    E_water_raw = qed_energy(water_raw_geom)

    print("  Benzene CP monomer energy...")
    E_benzene_cp = qed_energy(benzene_cp_geom)

    print("  Water CP monomer energy...")
    E_water_cp = qed_energy(water_cp_geom)

    E_int_raw = E_dimer - E_benzene_raw - E_water_raw
    E_int_cp = E_dimer - E_benzene_cp - E_water_cp

    result = {
        "label": label,
        "distance_A": distance_A,
        "E_dimer_Ha": E_dimer,
        "E_benzene_raw_Ha": E_benzene_raw,
        "E_water_raw_Ha": E_water_raw,
        "E_benzene_CP_Ha": E_benzene_cp,
        "E_water_CP_Ha": E_water_cp,
        "E_int_raw_Ha": E_int_raw,
        "E_int_raw_kcal_mol": E_int_raw * hartree_to_kcal_mol,
        "E_int_CP_Ha": E_int_cp,
        "E_int_CP_kcal_mol": E_int_cp * hartree_to_kcal_mol,
        "status": "ok",
    }

    print(
        f"  E_int_raw = {result['E_int_raw_kcal_mol']: .8f} kcal/mol; "
        f"E_int_CP = {result['E_int_CP_kcal_mol']: .8f} kcal/mol"
    )

    return result


def write_header_if_needed(filename, fieldnames):
    if os.path.exists(filename):
        return
    with open(filename, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_result(filename, fieldnames, result):
    with open(filename, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(result)


def load_completed_ok_distances(filename):
    completed = set()
    if not os.path.exists(filename):
        return completed

    with open(filename, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "ok":
                completed.add(round(float(row["distance_A"]), 8))
    return completed


def main():
    fieldnames = [
        "label",
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

    write_header_if_needed(OUTPUT_CSV, fieldnames)
    completed = load_completed_ok_distances(OUTPUT_CSV)

    print(f"Writing results to {OUTPUT_CSV}")
    print(f"Number of hard-coded geometries: {len(GEOMETRIES)}")

    for entry in GEOMETRIES:
        distance_key = round(float(entry["distance_A"]), 8)
        if distance_key in completed:
            print(f"Skipping {entry['label']} at d={entry['distance_A']:.2f} A; already completed.")
            continue

        try:
            result = compute_one_geometry(entry)
        except Exception as exc:
            print(f"FAILED {entry['label']} at d={entry['distance_A']:.2f} A")
            traceback.print_exc()
            result = {
                "label": entry["label"],
                "distance_A": entry["distance_A"],
                "E_dimer_Ha": "",
                "E_benzene_raw_Ha": "",
                "E_water_raw_Ha": "",
                "E_benzene_CP_Ha": "",
                "E_water_CP_Ha": "",
                "E_int_raw_Ha": "",
                "E_int_raw_kcal_mol": "",
                "E_int_CP_Ha": "",
                "E_int_CP_kcal_mol": "",
                "status": f"failed: {type(exc).__name__}: {exc}",
            }

        append_result(OUTPUT_CSV, fieldnames, result)

    print("\nDone.")


if __name__ == "__main__":
    main()
