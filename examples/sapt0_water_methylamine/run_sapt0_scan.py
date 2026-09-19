"""Standard psi4 SAPT0 scan over the water--methylamine adaptive curve geometries.

Portable by design: this script needs only **psi4** and the Python standard
library.  It does not import ``qcage`` or ``cqed-scf``, and it does not need the
S66x8 dataset -- every geometry comes from the companion file
``watermenh2_adaptive_scan.xyz``.  Copy the two files anywhere psi4 is
installed and run.

Geometries
----------
27 points: the 23-point adaptive grid from the completed WaterMeNH2 curve scan
(Chebyshev local window + uniform sub-stencil + repulsive wall + r^-6 tail,
centered on R_min = 3.3242935631440167 A) plus the four canonical pilot anchors
at scale factors 0.90 / 1.00 / 1.10 / 1.50.  Center-of-mass separation runs from
2.99 to 7.31 A.  ``make_geometries.py`` in this directory regenerates the .xyz
and records the grid provenance in ``adaptive_grid.json``.

Protocol
--------
Plain psi4 SAPT0, ``freeze_core false`` -- deliberately all-electron, so the
components line up term-for-term with qcage's ``sapt0_components`` path
(cqed-scf QED-SAPT0 at lambda = 0), which never freezes core.  Pass
``--freeze-core`` for the conventional frozen-core literature protocol instead.

Note on fitting bases: psi4 does not ship ``jun-cc-pVDZ-JKFIT``/``-RI`` and
resolves calendar basis sets (jun-, jul-, may-) to the ``aug-cc-pVDZ`` fitting
sets.  That is the intended behavior; ``--df-basis-scf`` / ``--df-basis-sapt``
are there in case a particular psi4 build errors instead of falling back.

Usage
-----
    python run_sapt0_scan.py --dry-run                     # no psi4 needed
    python run_sapt0_scan.py --basis 6-31G --scales 1.00   # quick smoke test
    python run_sapt0_scan.py --threads 8 --memory "16 GB"  # full scan
    python run_sapt0_scan.py --resume                      # pick up where it stopped
"""

import argparse
import csv
import os
import time

HARTREE_TO_KCAL_MOL = 627.509474

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XYZ = os.path.join(HERE, "watermenh2_adaptive_scan.xyz")

#: psi4 SAPT0 variables to harvest.  Names have drifted slightly between psi4
#: versions, so every read is tolerant (missing -> None) rather than fatal.
COMPONENT_VARIABLES = {
    "elst10": "SAPT ELST10,R ENERGY",
    "exch10": "SAPT EXCH10 ENERGY",
    "exch10_s2": "SAPT EXCH10(S^2) ENERGY",
    "ind20": "SAPT IND20,R ENERGY",
    "exch_ind20": "SAPT EXCH-IND20,R ENERGY",
    "disp20": "SAPT DISP20 ENERGY",
    "exch_disp20": "SAPT EXCH-DISP20 ENERGY",
}

GROUPED_VARIABLES = {
    "elst": "SAPT ELST ENERGY",
    "exch": "SAPT EXCH ENERGY",
    "ind": "SAPT IND ENERGY",
    "disp": "SAPT DISP ENERGY",
    "sapt_hf_total": "SAPT HF TOTAL ENERGY",
    "total_hartree": "SAPT TOTAL ENERGY",
}

CSV_FIELDS = (
    ["index", "scale_factor", "com_distance_angstrom", "region"]
    + list(COMPONENT_VARIABLES)
    + list(GROUPED_VARIABLES)
    + ["delta_hf", "total_kcal_mol", "wall_time_s", "status"]
)


# --------------------------------------------------------------------------
# Geometry file
# --------------------------------------------------------------------------


def parse_xyz_frames(path):
    """Parse a multi-frame extended XYZ file.

    The comment line holds ``key=value`` metadata (scale factor, COM
    separation, fragment slices, per-fragment charge/multiplicity).  Returns a
    list of ``{"metadata": {...}, "atoms": [(symbol, x, y, z), ...]}``.
    """
    with open(path) as handle:
        lines = handle.read().splitlines()

    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        n_atoms = int(lines[i].strip())
        comment = lines[i + 1]

        metadata = {}
        for token in comment.split():
            if "=" in token:
                key, value = token.split("=", 1)
                metadata[key] = value

        atoms = []
        for line in lines[i + 2 : i + 2 + n_atoms]:
            symbol, x, y, z = line.split()
            atoms.append((symbol, float(x), float(y), float(z)))

        if len(atoms) != n_atoms:
            raise ValueError(
                f"Frame {len(frames)} in {path} declares {n_atoms} atoms but has "
                f"{len(atoms)}."
            )

        frames.append({"metadata": metadata, "atoms": atoms})
        i += 2 + n_atoms

    if not frames:
        raise ValueError(f"No frames parsed from {path}.")
    return frames


def _slice(metadata, key):
    start, stop = metadata[key].split(":")
    return int(start), int(stop)


def build_psi4_dimer_string(frame):
    """Build a two-fragment psi4 geometry string for one frame.

    Layout matches what qcage generates: dimer charge/multiplicity, fragment A,
    ``--``, fragment B charge/multiplicity, fragment B, then ``no_reorient``,
    ``no_com``, ``units angstrom``, ``symmetry c1``.  ``no_reorient``/``no_com``
    keep psi4 from moving the dimer, so the geometry psi4 sees is exactly the
    one in the file.
    """
    metadata = frame["metadata"]
    a_start, a_stop = _slice(metadata, "frag_a")
    b_start, b_stop = _slice(metadata, "frag_b")

    def atom_lines(start, stop):
        return [
            f"{symbol:2s} {x:16.9f} {y:16.9f} {z:16.9f}"
            for symbol, x, y, z in frame["atoms"][start:stop]
        ]

    lines = [f'{metadata.get("charge", 0)} {metadata.get("mult", 1)}']
    lines += atom_lines(a_start, a_stop)
    lines.append("--")
    lines.append(f'{metadata.get("charge_b", 0)} {metadata.get("mult_b", 1)}')
    lines += atom_lines(b_start, b_stop)
    lines += ["no_reorient", "no_com", "units angstrom", "symmetry c1"]

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# psi4 driver
# --------------------------------------------------------------------------


def configure_psi4(psi4, args):
    """Apply the process-wide psi4 settings (memory, threads, scratch)."""
    psi4.set_memory(args.memory)
    psi4.set_num_threads(args.threads)
    if args.scratch:
        os.makedirs(args.scratch, exist_ok=True)
        os.environ["PSI_SCRATCH"] = args.scratch


def sapt0_options(args):
    options = {
        "basis": args.basis,
        "scf_type": "df",
        "freeze_core": "true" if args.freeze_core else "false",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
        "guess": "sad",
    }
    if args.df_basis_scf:
        options["df_basis_scf"] = args.df_basis_scf
    if args.df_basis_sapt:
        options["df_basis_sapt"] = args.df_basis_sapt
    return options


def harvest(psi4):
    """Read the SAPT0 variables psi4 set, tolerating absent ones."""

    def read(name):
        try:
            return float(psi4.variable(name))
        except Exception:
            return None

    values = {key: read(name) for key, name in COMPONENT_VARIABLES.items()}
    values.update({key: read(name) for key, name in GROUPED_VARIABLES.items()})

    # psi4 folds delta_HF into the induction group; recover it by subtracting
    # the second-order induction terms back out.
    ind = values.get("ind")
    ind20, exch_ind20 = values.get("ind20"), values.get("exch_ind20")
    if None not in (ind, ind20, exch_ind20):
        values["delta_hf"] = ind - (ind20 + exch_ind20)
    else:
        values["delta_hf"] = None

    total = values.get("total_hartree")
    values["total_kcal_mol"] = None if total is None else total * HARTREE_TO_KCAL_MOL
    return values


def run_point(psi4, frame, index, args, output_dir):
    """Run SAPT0 on one frame and return a CSV row dict."""
    metadata = frame["metadata"]
    scale = float(metadata["scale_factor"])

    row = {
        "index": index,
        "scale_factor": f"{scale:.6f}",
        "com_distance_angstrom": f'{float(metadata["com_distance_angstrom"]):.6f}',
        "region": metadata.get("region", ""),
    }

    psi4.core.set_output_file(
        os.path.join(output_dir, f"psi4_s{scale:.6f}.out"), False
    )

    started = time.time()
    try:
        psi4.core.clean_variables()
        molecule = psi4.geometry(build_psi4_dimer_string(frame))
        psi4.set_options(sapt0_options(args))
        psi4.energy("sapt0", molecule=molecule)
        row.update(harvest(psi4))
        row["status"] = "ok"
    except Exception as exc:  # one bad point should not end a 27-point scan
        row["status"] = f"error: {type(exc).__name__}: {exc}".replace("\n", " ")[:500]
    finally:
        psi4.core.clean()
        row["wall_time_s"] = f"{time.time() - started:.1f}"

    return row


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def completed_scales(path):
    """Scale factors already recorded as successful in an existing CSV."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as handle:
        return {
            row["scale_factor"]
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--xyz", default=DEFAULT_XYZ, help="Multi-frame geometry file.")
    parser.add_argument("--basis", default="jun-cc-pVDZ", help="Orbital basis set.")
    parser.add_argument(
        "--freeze-core",
        dest="freeze_core",
        action="store_true",
        help="Frozen-core SAPT0 (literature protocol). Default is all-electron, "
        "matching qcage's cqed-scf SAPT0 path.",
    )
    parser.add_argument(
        "--no-freeze-core", dest="freeze_core", action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(freeze_core=False)
    parser.add_argument("--df-basis-scf", default=None, help="Override DF_BASIS_SCF.")
    parser.add_argument("--df-basis-sapt", default=None, help="Override DF_BASIS_SAPT.")
    parser.add_argument("--memory", default="4 GB", help="psi4 memory.")
    parser.add_argument("--threads", type=int, default=1, help="psi4 threads.")
    parser.add_argument("--scratch", default=None, help="PSI_SCRATCH directory.")
    parser.add_argument("--out", default="sapt0_scan.csv", help="Output CSV path.")
    parser.add_argument(
        "--output-dir",
        default="psi4_outputs",
        help="Directory for per-point psi4 output files.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=None,
        help="Run only these scale factors (matched to 1e-6). Useful for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scale factors already recorded as 'ok' in the output CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the geometries and psi4 input blocks without importing psi4.",
    )
    return parser.parse_args()


def select_frames(frames, args):
    if args.scales is None:
        return frames
    wanted = [float(s) for s in args.scales]
    selected = [
        frame
        for frame in frames
        if any(
            abs(float(frame["metadata"]["scale_factor"]) - s) < 1e-6 for s in wanted
        )
    ]
    if not selected:
        raise SystemExit(f"No frames matched --scales {args.scales}")
    return selected


def dry_run(frames):
    print(f"{'#':>3} {'scale':>10} {'R (A)':>9}  region")
    for index, frame in enumerate(frames):
        metadata = frame["metadata"]
        print(
            f"{index:3d} "
            f'{float(metadata["scale_factor"]):10.6f} '
            f'{float(metadata["com_distance_angstrom"]):9.4f}  '
            f'{metadata.get("region", "")}'
        )

    print()
    for index, frame in enumerate(frames):
        print(f'--- frame {index}: scale {frame["metadata"]["scale_factor"]} ---')
        print(build_psi4_dimer_string(frame))


def main():
    args = parse_args()
    frames = select_frames(parse_xyz_frames(args.xyz), args)

    if args.dry_run:
        dry_run(frames)
        return

    import psi4  # imported late so --dry-run works without psi4 installed

    configure_psi4(psi4, args)
    os.makedirs(args.output_dir, exist_ok=True)

    done = completed_scales(args.out) if args.resume else set()
    write_header = not (args.resume and os.path.exists(args.out))

    print(f"SAPT0 / {args.basis}   freeze_core={'true' if args.freeze_core else 'false'}")
    print(f"{len(frames)} geometries from {args.xyz}")
    if done:
        print(f"Resuming: skipping {len(done)} already-completed points")
    print()

    mode = "a" if (args.resume and os.path.exists(args.out)) else "w"
    with open(args.out, mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
            handle.flush()

        for index, frame in enumerate(frames):
            scale = f'{float(frame["metadata"]["scale_factor"]):.6f}'
            if scale in done:
                print(f"[{index + 1:2d}/{len(frames)}] scale {scale}  skipped (resume)")
                continue

            print(f"[{index + 1:2d}/{len(frames)}] scale {scale}  running ...", flush=True)
            row = run_point(psi4, frame, index, args, args.output_dir)

            # Flush every row so a killed run leaves usable partial results.
            writer.writerow(row)
            handle.flush()

            if row["status"] == "ok":
                print(
                    f"           E_int = {row['total_kcal_mol']:.4f} kcal/mol "
                    f"({row['wall_time_s']} s)"
                )
            else:
                print(f"           {row['status']}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
