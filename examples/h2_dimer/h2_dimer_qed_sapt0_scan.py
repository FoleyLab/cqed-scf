"""QED-SAPT0 scan for an H2 dimer in a z-polarized cavity.

This example reproduces a useful diagnostic scan:

1. A short scan over ``np.linspace(3, 10, 10)`` can make the QED-SAPT0
   interaction look like it is becoming steadily more attractive.
2. A longer scan from 3 to 20 Angstrom shows whether the curve turns back up
   and reports the minimum sampled point.

The script writes CSV files with the QED-SAPT0 component energies in Hartree
and kcal/mol.  It also prints a compact summary of the minimum found on each
scan grid.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import psi4

from cqed_scf import CQEDConfig
from cqed_scf.sapt import QEDSAPT0Driver


HARTREE_TO_KCAL_MOL = 627.5094740631

LAMBDA_VECTOR = np.array([0.0, 0.0, 0.05])
DEFAULT_OMEGA = 0.1

PSI4_GLOBAL_OPTIONS = {
    "basis": "aug-cc-pvdz",
    "e_convergence": 1e-8,
    "d_convergence": 1e-8,
    "scf_type": "pk",
}

SHORT_DISTANCES = np.linspace(3.0, 10.0, 10)
LONG_DISTANCES = np.linspace(3.0, 20.0, 18)

COMPONENTS = (
    ("elst10", "Electrostatics"),
    ("exch10", "Exchange"),
    ("disp20", "Dispersion"),
    ("exch_disp20", "Exchange-Dispersion"),
    ("ind20r", "Induction"),
    ("exch_ind20r", "Exchange-Induction"),
    ("total", "Total QED-SAPT0"),
)


def make_h2_dimer_geometry(R: float) -> str:
    """Return the two-fragment H2 dimer geometry at separation R in Angstrom."""

    return f"""
0 1
H  -0.370000000  0.000000000  0.000000000
H   0.370000000  0.000000000  0.000000000
--
0 1
H  -0.370000000  0.000000000  {R:12.8f}
H   0.370000000  0.000000000  {R:12.8f}
symmetry c1
no_reorient
nocom
"""


def make_config(omega: float) -> CQEDConfig:
    return CQEDConfig(
        lambda_vector=LAMBDA_VECTOR,
        omega=omega,
        psi4_options=PSI4_GLOBAL_OPTIONS,
        reference="rhf",
        functional=None,
        density_fitting=False,
        charge=0,
        multiplicity=1,
        dispersion_policy="none",
        debug=False,
    )


def run_qed_sapt0_at_distance(R: float, config: CQEDConfig) -> dict[str, float]:
    psi4.core.clean()
    psi4.core.clean_options()

    dimer_geometry = psi4.geometry(make_h2_dimer_geometry(R))
    driver = QEDSAPT0Driver(
        dimer_geometry=dimer_geometry,
        config=config,
        integral_backend="full_eri",
        include_cavity_terms=True,
    )
    total = driver.run()

    return {
        "distance_angstrom": R,
        "elst10_hartree": driver.Eelst100,
        "exch10_hartree": driver.Eexch100,
        "disp20_hartree": driver.Edisp200,
        "exch_disp20_hartree": driver.Eexchdisp200,
        "ind20r_hartree": driver.Eind200,
        "exch_ind20r_hartree": driver.Eexchind200,
        "total_hartree": total,
    }


def with_kcal_columns(row: dict[str, float]) -> dict[str, float]:
    converted = dict(row)
    for key, value in row.items():
        if key.endswith("_hartree"):
            converted[key.replace("_hartree", "_kcal_mol")] = value * HARTREE_TO_KCAL_MOL
    return converted


def run_scan(label: str, distances: np.ndarray, config: CQEDConfig) -> list[dict[str, float]]:
    rows = []
    print(f"\nRunning {label} scan with {len(distances)} points")
    print("=" * 72)
    for R in distances:
        row = with_kcal_columns(run_qed_sapt0_at_distance(float(R), config))
        rows.append(row)
        print(
            f"R = {R:6.2f} Ang  "
            f"Total = {row['total_hartree']: .10f} Ha  "
            f"({row['total_kcal_mol']: .4f} kcal/mol)"
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["distance_angstrom"]
    for component, _ in COMPONENTS:
        fieldnames.append(f"{component}_hartree")
    for component, _ in COMPONENTS:
        fieldnames.append(f"{component}_kcal_mol")

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_scan(label: str, rows: list[dict[str, float]]) -> None:
    distances = np.array([row["distance_angstrom"] for row in rows])
    totals = np.array([row["total_hartree"] for row in rows])
    idx_min = int(np.argmin(totals))
    is_strictly_decreasing = bool(np.all(np.diff(totals) < 0.0))

    trend = "strictly decreasing" if is_strictly_decreasing else "turns around on this grid"
    print(f"\n{label} summary")
    print("-" * 72)
    print(f"Trend: {trend}")
    print(
        "Minimum sampled point: "
        f"R = {distances[idx_min]:.2f} Ang, "
        f"E = {totals[idx_min]:.10f} Ha, "
        f"{totals[idx_min] * HARTREE_TO_KCAL_MOL:.4f} kcal/mol"
    )


def plot_scans(output_path: Path, scans: dict[str, list[dict[str, float]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed; skipping plot generation.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, rows in scans.items():
        distances = [row["distance_angstrom"] for row in rows]
        totals = [row["total_kcal_mol"] for row in rows]
        ax.plot(distances, totals, marker="o", label=label)

    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Intermolecular separation R (Angstrom)")
    ax.set_ylabel("QED-SAPT0 total energy (kcal/mol)")
    ax.set_title("H2 dimer QED-SAPT0 scan")
    ax.grid(alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan",
        choices=("short", "long", "both"),
        default="both",
        help="Which scan grid to run. Default: both.",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=DEFAULT_OMEGA,
        help=f"Cavity frequency used in CQEDConfig. Default: {DEFAULT_OMEGA}.",
    )
    parser.add_argument(
        "--memory",
        default="4 GB",
        help="Psi4 memory setting. Default: 4 GB.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for CSV and plot output. Default: this example folder.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the PNG plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    psi4.core.be_quiet()
    psi4.set_memory(args.memory)

    config = make_config(args.omega)
    scans_to_run = {}
    if args.scan in ("short", "both"):
        scans_to_run["short_3_to_10"] = SHORT_DISTANCES
    if args.scan in ("long", "both"):
        scans_to_run["long_3_to_20"] = LONG_DISTANCES

    completed_scans = {}
    for label, distances in scans_to_run.items():
        rows = run_scan(label, distances, config)
        output_path = args.output_dir / f"h2_dimer_qed_sapt0_{label}.csv"
        write_csv(output_path, rows)
        summarize_scan(label, rows)
        print(f"Saved data: {output_path}")
        completed_scans[label] = rows

    if not args.no_plot:
        plot_scans(args.output_dir / "h2_dimer_qed_sapt0_scan.png", completed_scans)


if __name__ == "__main__":
    main()
