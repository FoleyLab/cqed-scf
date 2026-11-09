"""Cross-check the two oracles and emit the reference-energy table.

Reads ``psi4_reference_energies.json`` and ``hilbert_reference_energies.json``,
asserts that Hilbert's QED-UHF / QED-UKS energies collapse onto Psi4's UHF / UKS
energies at ``lambda = 0``, and writes ``REFERENCE_ENERGIES.md``.

Usage
-----
    python make_reference_table.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PSI4_JSON = os.path.join(HERE, "psi4_reference_energies.json")
HILBERT_JSON = os.path.join(HERE, "hilbert_reference_energies.json")
MARKDOWN = os.path.join(HERE, "REFERENCE_ENERGIES.md")

# Tolerance for the lambda -> 0 consistency check.  The two codes run the same
# Psi4 integrals and the same SCF convergence thresholds, so they should agree
# far more tightly than this.
ZERO_LAMBDA_TOL = 1.0e-10

METHOD_PAIRS = (("uhf", "qed_uhf", "UHF"), ("uks", "qed_uks", "UKS"))


def check_zero_lambda(psi4_data, hilbert_data):
    """Verify QED-U{HF,KS}(lambda=0) == U{HF,KS}; returns the per-case deltas."""
    rows = []
    failures = []
    for system in psi4_data["systems"]:
        for psi4_key, hilbert_key, label in METHOD_PAIRS:
            e_psi4 = psi4_data["systems"][system]["methods"][psi4_key]["energy"]
            e_hilbert = (
                hilbert_data["systems"][system]["methods"][hilbert_key]["zero"]["energy"]
            )
            delta = e_hilbert - e_psi4
            rows.append((system, label, e_psi4, e_hilbert, delta))
            if abs(delta) > ZERO_LAMBDA_TOL:
                failures.append((system, label, delta))
    return rows, failures


def main():
    with open(PSI4_JSON) as handle:
        psi4_data = json.load(handle)
    with open(HILBERT_JSON) as handle:
        hilbert_data = json.load(handle)

    rows, failures = check_zero_lambda(psi4_data, hilbert_data)

    basis = psi4_data["basis"]
    functional = psi4_data["functional"]
    omega = hilbert_data["omega"]
    lambdas = hilbert_data["lambda_vectors"]

    lines = []
    add = lines.append

    add("# Unrestricted (QED-)SCF reference energies")
    add("")
    add("Reference data for the forthcoming `cqed_scf.uscf` implementation.")
    add("")
    add("| setting | value |")
    add("| --- | --- |")
    add(f"| basis | `{basis}` |")
    add(f"| functional (UKS / QED-UKS) | `{functional}` |")
    add("| `scf_type` | `pk` (no density fitting) |")
    add("| `e_convergence` / `d_convergence` | `1e-11` / `1e-10` |")
    add("| DFT grid | 99 radial x 590 spherical |")
    add(f"| cavity frequency omega | `{omega}` a.u. |")
    add("| photon states | 1 (coherent-state basis) |")
    add(f"| UHF / UKS oracle | Psi4 {psi4_data['psi4_version']} |")
    add(f"| QED-UHF / QED-UKS oracle | Hilbert `polaritonic_scf` (Psi4 {hilbert_data['psi4_version']}) |")
    add("")
    add("Coupling convention: `cqed_scf` takes `lambda`; Hilbert takes")
    add("`CAVITY_COUPLING_STRENGTH` = `lambda / sqrt(2 * omega)`.")
    add("")

    add("## Cavity coupling vectors")
    add("")
    add("| label | lambda (a.u.) | purpose |")
    add("| --- | --- | --- |")
    purposes = {
        "zero": "regression anchor: must reproduce the cavity-free UHF / UKS energy",
        "z_only": "polarization parallel to the molecular axis",
        "x_only": "polarization perpendicular to the molecular axis",
        "general": "non-axial, exercises every off-diagonal lambda_i*lambda_j term",
    }
    for label, vec in lambdas.items():
        vec_str = "(" + ", ".join(f"{v:.2f}" for v in vec) + ")"
        add(f"| `{label}` | {vec_str} | {purposes.get(label, '')} |")
    add("")

    add("## Test systems")
    add("")
    add("| key | description | charge | multiplicity |")
    add("| --- | --- | --- | --- |")
    for name, entry in psi4_data["systems"].items():
        add(
            f"| `{name}` | {entry['description']} | {entry['charge']} "
            f"| {entry['multiplicity']} |"
        )
    add("")
    for name, entry in psi4_data["systems"].items():
        add(f"### `{name}` geometry")
        add("")
        add("```")
        add(entry["geometry"].strip())
        add("```")
        add("")

    add("## Psi4 reference: UHF and UKS")
    add("")
    add("| system | method | energy / Eh | &lt;S^2&gt; | exact &lt;S^2&gt; |")
    add("| --- | --- | ---: | ---: | ---: |")
    for name, entry in psi4_data["systems"].items():
        for key, data in entry["methods"].items():
            label = "UHF" if key == "uhf" else f"UKS/{functional}"
            add(
                f"| `{name}` | {label} | `{data['energy']:.12f}` "
                f"| {data['s_squared']:.6f} | {data['s_squared_exact']:.4f} |"
            )
    add("")

    add("## Hilbert reference: QED-UHF and QED-UKS")
    add("")
    add("| system | method | lambda | energy / Eh | dE vs. cavity-free / Eh |")
    add("| --- | --- | --- | ---: | ---: |")
    for name, entry in hilbert_data["systems"].items():
        for method, per_lambda in entry["methods"].items():
            label = "QED-UHF" if method == "qed_uhf" else f"QED-UKS/{functional}"
            e_zero = per_lambda["zero"]["energy"]
            for lam_label, data in per_lambda.items():
                shift = data["energy"] - e_zero
                add(
                    f"| `{name}` | {label} | `{lam_label}` "
                    f"| `{data['energy']:.12f}` | {shift:+.9f} |"
                )
    add("")

    add("## lambda -> 0 consistency check")
    add("")
    add("Hilbert's QED-UHF / QED-UKS must collapse onto Psi4's UHF / UKS when the")
    add("coupling is switched off. This is the first test `uscf.py` should pass.")
    add("")
    add("| system | method | Psi4 / Eh | Hilbert (lambda=0) / Eh | difference / Eh |")
    add("| --- | --- | ---: | ---: | ---: |")
    for system, label, e_psi4, e_hilbert, delta in rows:
        add(
            f"| `{system}` | {label} | `{e_psi4:.12f}` | `{e_hilbert:.12f}` "
            f"| {delta:.2e} |"
        )
    add("")
    if failures:
        add(f"**{len(failures)} case(s) exceeded the {ZERO_LAMBDA_TOL:.0e} Eh tolerance.**")
    else:
        add(
            f"All {len(rows)} cases agree to better than {ZERO_LAMBDA_TOL:.0e} Eh."
        )
    add("")

    with open(MARKDOWN, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    max_delta = max(abs(row[4]) for row in rows)
    print(f"lambda -> 0 consistency: max |difference| = {max_delta:.3e} Eh "
          f"over {len(rows)} cases")
    for system, label, delta in failures:
        print(f"  FAIL {system} {label}: {delta:.3e} Eh")
    print(f"wrote {MARKDOWN}")


if __name__ == "__main__":
    main()
