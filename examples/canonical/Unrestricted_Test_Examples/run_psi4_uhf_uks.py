"""Generate UHF and UKS/PBE0 reference energies with Psi4.

Psi4 is the oracle for the *cavity-free* unrestricted references that
``cqed_scf.uscf`` must reproduce in the lambda -> 0 limit.

Usage
-----
    python run_psi4_uhf_uks.py

Writes ``psi4_reference_energies.json`` next to this script and prints a
summary table.
"""

from __future__ import annotations

import json
import os

import numpy as np
import psi4

from systems import (
    BASIS,
    FUNCTIONAL,
    SYSTEMS,
    base_psi4_options,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(HERE, "psi4_reference_energies.json")


def spin_squared(wfn):
    """<S^2> for an unrestricted determinant.

    <S^2> = S_z(S_z + 1) + N_beta - sum_ij |<phi_i^alpha | phi_j^beta>|^2

    Psi4 does not expose this as a QCVariable for a plain SCF run, so we build
    it from the alpha/beta MO coefficients and the AO overlap directly.
    """
    S = np.asarray(wfn.S())
    Ca = np.asarray(wfn.Ca_subset("AO", "OCC"))
    Cb = np.asarray(wfn.Cb_subset("AO", "OCC"))
    na, nb = wfn.nalpha(), wfn.nbeta()
    sz = 0.5 * (na - nb)
    overlap = Ca.T @ S @ Cb
    return sz * (sz + 1.0) + nb - np.sum(overlap ** 2)


def run_one(name, spec, method):
    """Run a single UHF or UKS calculation and pull out the reference data.

    ``method`` is ``"uhf"`` or ``"uks"``.
    """
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()

    mol = psi4.geometry(spec["geometry"])
    psi4.set_options(base_psi4_options("uhf" if method == "uhf" else "uks"))

    # For UHF the Psi4 method name is "scf"; for UKS it is the functional name.
    psi4_method = "scf" if method == "uhf" else FUNCTIONAL
    energy, wfn = psi4.energy(psi4_method, molecule=mol, return_wfn=True)

    return {
        "energy": float(energy),
        "nalpha": int(wfn.nalpha()),
        "nbeta": int(wfn.nbeta()),
        "nbf": int(wfn.basisset().nbf()),
        "s_squared": float(spin_squared(wfn)),
        "s_squared_exact": float(0.5 * (wfn.nalpha() - wfn.nbeta())
                                 * (0.5 * (wfn.nalpha() - wfn.nbeta()) + 1.0)),
        "nuclear_repulsion": float(mol.nuclear_repulsion_energy()),
        "scf_dipole_au": [float(x) for x in np.asarray(psi4.variable("SCF DIPOLE"))],
    }


def main():
    psi4.set_memory("8 GB")
    psi4.core.set_output_file(os.path.join(HERE, "psi4_reference.out"), False)

    results = {
        "oracle": "psi4",
        "psi4_version": psi4.__version__,
        "basis": BASIS,
        "functional": FUNCTIONAL,
        "systems": {},
    }

    for name, spec in SYSTEMS.items():
        results["systems"][name] = {
            "description": spec["description"],
            "charge": spec["charge"],
            "multiplicity": spec["multiplicity"],
            "geometry": spec["geometry"],
            "methods": {},
        }
        for method in ("uhf", "uks"):
            label = "UHF" if method == "uhf" else f"UKS/{FUNCTIONAL}"
            print(f"[psi4] {name:<12s} {label}", flush=True)
            results["systems"][name]["methods"][method] = run_one(name, spec, method)

    with open(OUTFILE, "w") as handle:
        json.dump(results, handle, indent=2)

    print()
    print(f"Psi4 reference energies  ({BASIS}, scf_type=pk)")
    print("=" * 78)
    print(f"{'system':<14s}{'method':<14s}{'energy / Eh':>22s}{'<S^2>':>12s}")
    print("-" * 78)
    for name, entry in results["systems"].items():
        for method, data in entry["methods"].items():
            label = "UHF" if method == "uhf" else f"UKS/{FUNCTIONAL}"
            print(
                f"{name:<14s}{label:<14s}{data['energy']:>22.12f}"
                f"{data['s_squared']:>12.6f}"
            )
    print("=" * 78)
    print(f"\nwrote {OUTFILE}")


if __name__ == "__main__":
    main()
