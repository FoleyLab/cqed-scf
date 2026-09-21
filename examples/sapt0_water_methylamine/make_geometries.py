"""Regenerate the WaterMeNH2 adaptive-curve-scan geometries as a portable .xyz.

This is a **one-off generator** kept for provenance.  It imports ``qcage`` and
is not needed to run the SAPT0 scan -- ``run_sapt0_scan.py`` reads only the
``.xyz`` file this writes.

How the grid is recovered
-------------------------
The completed adaptive curve scan for WaterMeNH2 is recorded in
``surface_row_store/WaterMeNH2__wb97x-d__jun-cc-pVDZ__cavity_free__cavity_free__*.json``.
That row does not store the grid itself, but it does store ``R_e_fine_angstrom``
and ``delta_R_e_pilot_fine_angstrom``, and

    r_reference_angstrom = R_e_fine_angstrom + delta_R_e_pilot_fine_angstrom

is exactly the pilot Morse minimum the grid was centered on (see
``qcage.result_schema._region1_block``).  Every node position that
:func:`qcage.analysis.grid.build_adaptive_scan_grid` produces -- the Chebyshev
local window, the uniform sub-stencil, the geometric wall, and the r^-6 tail --
is a function of that one number alone.  So replaying the builder with a
synthetic pilot curve that fits back to that R_min reproduces the original
grid.  The point count is the cross-check: the row records ``n_points_fit: 23``
(the grid) and ``n_points_used_total: 27`` (grid + the four pilot anchors).

Usage
-----
    PYTHONPATH=src python3 examples/sapt0_water_methylamine/make_geometries.py
"""

import argparse
import glob
import json
import os

from qcage.analysis.grid import build_adaptive_scan_grid
from qcage.analysis.morse import morse_energy
from qcage.constants import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL_MOL
from qcage.geometry.com import com_distance_for_scale
from qcage.geometry.scaling import load_scaled_geometry
from qcage.workflows.curve_scan import DEFAULT_PILOT_SCALES

COMPLEX_NAME = "WaterMeNH2"

#: The grid center recorded by the completed adaptive run, to full precision.
#: Asserted below against what the replayed builder reports.
EXPECTED_R_REFERENCE_ANGSTROM = 3.3242935631440167

#: ``n_points_fit`` from the stored surface row -- the size of the main grid.
EXPECTED_N_GRID = 23

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
SURFACE_ROW_GLOB = os.path.join(
    REPO_ROOT, "surface_row_store", f"{COMPLEX_NAME}__wb97x-d__jun-cc-pVDZ__*.json"
)


def load_surface_row():
    """Return the stored adaptive-curve surface row for this complex."""
    matches = sorted(glob.glob(SURFACE_ROW_GLOB))
    if not matches:
        raise SystemExit(f"No stored surface row matching {SURFACE_ROW_GLOB}")
    if len(matches) > 1:
        raise SystemExit(
            f"Expected one stored surface row, found {len(matches)}:\n  "
            + "\n  ".join(matches)
        )
    with open(matches[0]) as handle:
        return matches[0], json.load(handle)["surface_row"]


def synthesize_pilot_curve(r_reference_angstrom, d_e_kcal_mol, a_inv_bohr):
    """Build a pilot curve that fits back to the recorded grid center.

    :func:`build_adaptive_scan_grid` takes the shape ``extract_curve`` returns
    and immediately Morse-fits it.  Sampling an analytic Morse at the canonical
    pilot separations round-trips through that fit exactly, which is what lets
    this reproduce the original node positions rather than approximate them.
    """
    d_e_hartree = d_e_kcal_mol / HARTREE_TO_KCAL_MOL
    r_e_bohr = r_reference_angstrom / BOHR_TO_ANGSTROM

    distances_bohr = []
    energies_hartree = []
    for scale in DEFAULT_PILOT_SCALES:
        r_bohr = com_distance_for_scale(COMPLEX_NAME, scale) / BOHR_TO_ANGSTROM
        distances_bohr.append(r_bohr)
        energies_hartree.append(
            float(morse_energy(r_bohr, d_e_hartree, a_inv_bohr, r_e_bohr, 0.0))
        )

    return {
        "com_distances_bohr": distances_bohr,
        "energies_hartree": energies_hartree,
    }


def region_labels(grid):
    """Map each grid scale factor to its region name(s)."""
    labels = {}
    for name, indices in grid["regions"].items():
        for index in indices:
            labels.setdefault(grid["target_scales"][index], []).append(name)
    return labels


def build_frames(grid):
    """Merge the grid with the pilot anchors and build one frame per scale."""
    labels = region_labels(grid)
    pilot = {round(float(s), 6) for s in DEFAULT_PILOT_SCALES}
    scales = sorted(set(grid["target_scales"]) | pilot)

    frames = []
    for scale in scales:
        record = load_scaled_geometry(COMPLEX_NAME, scale)
        metadata = record["metadata"]
        monomer_a, monomer_b = metadata["monomers"]

        names = list(labels.get(scale, []))
        if scale in pilot:
            names.append("pilot")

        frames.append(
            {
                "scale_factor": scale,
                "com_distance_angstrom": com_distance_for_scale(COMPLEX_NAME, scale),
                "region": "+".join(names) or "unassigned",
                "atoms": record["atoms"],
                "monomer_a": monomer_a,
                "monomer_b": monomer_b,
                "metadata": metadata,
            }
        )
    return frames


def _contiguous_slice(indices, what):
    """Return ``(start, stop)`` for a contiguous index list, else fail loudly.

    The .xyz comment line encodes each fragment as a half-open slice, which is
    only faithful if the fragments are contiguous blocks of atoms.  They are for
    every S66 dimer, but a silent mis-split would produce plausible-looking
    nonsense, so check rather than assume.
    """
    expected = list(range(indices[0], indices[0] + len(indices)))
    if list(indices) != expected:
        raise SystemExit(f"{what} atom indices are not contiguous: {indices}")
    return indices[0], indices[0] + len(indices)


def write_xyz(path, frames):
    """Write the frames as a multi-frame extended XYZ file."""
    with open(path, "w") as handle:
        for frame in frames:
            a_start, a_stop = _contiguous_slice(
                frame["monomer_a"]["atom_indices"], "Monomer A"
            )
            b_start, b_stop = _contiguous_slice(
                frame["monomer_b"]["atom_indices"], "Monomer B"
            )
            metadata = frame["metadata"]

            comment = " ".join(
                [
                    f'complex={COMPLEX_NAME}',
                    f'scale_factor={frame["scale_factor"]:.6f}',
                    f'com_distance_angstrom={frame["com_distance_angstrom"]:.6f}',
                    f'region={frame["region"]}',
                    f"frag_a={a_start}:{a_stop}",
                    f"frag_b={b_start}:{b_stop}",
                    f'name_a={frame["monomer_a"]["name"]}',
                    f'name_b={frame["monomer_b"]["name"]}',
                    f'charge={metadata.get("charge", 0)}',
                    f'mult={metadata.get("multiplicity", 1)}',
                    f'charge_a={frame["monomer_a"].get("charge", 0)}',
                    f'mult_a={frame["monomer_a"].get("multiplicity", 1)}',
                    f'charge_b={frame["monomer_b"].get("charge", 0)}',
                    f'mult_b={frame["monomer_b"].get("multiplicity", 1)}',
                ]
            )

            handle.write(f'{len(frame["atoms"])}\n')
            handle.write(f"{comment}\n")
            for atom in frame["atoms"]:
                handle.write(
                    f'{atom["symbol"]:2s} '
                    f'{atom["x"]:16.9f} '
                    f'{atom["y"]:16.9f} '
                    f'{atom["z"]:16.9f}\n'
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        default=HERE,
        help="Directory to write watermenh2_adaptive_scan.xyz and adaptive_grid.json into.",
    )
    args = parser.parse_args()

    row_path, row = load_surface_row()
    r_reference = row["R_e_fine_angstrom"] + row["delta_R_e_pilot_fine_angstrom"]
    print(f"Surface row : {os.path.relpath(row_path, REPO_ROOT)}")
    print(f"Grid center : {r_reference!r} A")

    if abs(r_reference - EXPECTED_R_REFERENCE_ANGSTROM) > 1e-12:
        raise SystemExit(
            f"Recovered grid center {r_reference!r} differs from the expected "
            f"{EXPECTED_R_REFERENCE_ANGSTROM!r}."
        )

    pilot_curve = synthesize_pilot_curve(
        r_reference, row["D_e_kcal_mol"], row["a_inv_bohr"]
    )
    grid = build_adaptive_scan_grid(COMPLEX_NAME, pilot_curve)

    if abs(grid["r_reference_angstrom"] - EXPECTED_R_REFERENCE_ANGSTROM) > 1e-12:
        raise SystemExit(
            f"Replayed grid centered on {grid['r_reference_angstrom']!r}, not the "
            f"recorded {EXPECTED_R_REFERENCE_ANGSTROM!r}; the pilot curve did not "
            f"round-trip through fit_morse."
        )
    if len(grid["target_scales"]) != EXPECTED_N_GRID:
        raise SystemExit(
            f"Replayed grid has {len(grid['target_scales'])} points, but the stored "
            f"row records n_points_fit={EXPECTED_N_GRID}."
        )

    for warning in grid["warnings"]:
        print(f"  grid warning: {warning}")

    frames = build_frames(grid)
    print(f"Points      : {len(grid['target_scales'])} grid + pilot -> {len(frames)} unique")

    os.makedirs(args.out_dir, exist_ok=True)
    xyz_path = os.path.join(args.out_dir, "watermenh2_adaptive_scan.xyz")
    json_path = os.path.join(args.out_dir, "adaptive_grid.json")

    write_xyz(xyz_path, frames)
    with open(json_path, "w") as handle:
        json.dump(
            {
                "complex_name": COMPLEX_NAME,
                "source_surface_row": os.path.basename(row_path),
                "r_reference_angstrom": r_reference,
                "pilot_scales": list(DEFAULT_PILOT_SCALES),
                "grid": grid,
                "all_scales": [frame["scale_factor"] for frame in frames],
            },
            handle,
            indent=2,
        )

    print(f"Wrote       : {xyz_path}")
    print(f"Wrote       : {json_path}")
    print()
    print(f"{'scale':>10} {'R (A)':>9}  region")
    for frame in frames:
        print(
            f'{frame["scale_factor"]:10.6f} '
            f'{frame["com_distance_angstrom"]:9.4f}  '
            f'{frame["region"]}'
        )


if __name__ == "__main__":
    main()
