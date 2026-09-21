"""Generate QED-UHF and QED-UKS/PBE0 reference energies with Hilbert.

Hilbert's ``polaritonic_scf`` module (``POLARITONIC_UHF`` / ``POLARITONIC_UKS``)
is the oracle for the coherent-state CQED unrestricted references that
``cqed_scf.uscf`` must reproduce.

Coupling convention
-------------------
``cqed_scf`` is parameterized by the coupling vector ``lambda``.  Hilbert's
option is ``CAVITY_COUPLING_STRENGTH`` = g, and internally it rebuilds
``lambda_i = g_i * sqrt(2 * omega)`` (``src/polaritonic_scf/hf.cc``).  The
conversion lives in ``systems.lambda_to_hilbert_coupling``.

Validation of this protocol
---------------------------
For the closed-shell case the settings below reproduce ``cqed_scf``'s existing
CQED-RHF and CQED-RKS/PBE0 energies to 1e-12 Eh, and at ``lambda = 0`` the
QED-UHF / QED-UKS energies here reproduce the Psi4 UHF / UKS energies from
``run_psi4_uhf_uks.py`` exactly.

Usage
-----
    python run_hilbert_qed_uhf_uks.py

Writes ``hilbert_reference_energies.json`` next to this script and prints a
summary table.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# Hilbert is imported from the directory *above* the package checkout, which is
# how the plugin registers its procedures with the Psi4 driver.
sys.path.insert(0, "/Users/jfoley19/Code")

import psi4  # noqa: E402
import hilbert  # noqa: E402,F401  (import registers the qed-* procedures)

from systems import (  # noqa: E402
    BASIS,
    FUNCTIONAL,
    LAMBDA_VECTORS,
    OMEGA,
    SYSTEMS,
    base_psi4_options,
    lambda_to_hilbert_coupling,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(HERE, "hilbert_reference_energies.json")


def spin_squared(wfn):
    """<S^2> for an unrestricted determinant; see run_psi4_uhf_uks.py."""
    S = np.asarray(wfn.S())
    Ca = np.asarray(wfn.Ca_subset("AO", "OCC"))
    Cb = np.asarray(wfn.Cb_subset("AO", "OCC"))
    na, nb = wfn.nalpha(), wfn.nbeta()
    sz = 0.5 * (na - nb)
    overlap = Ca.T @ S @ Cb
    return sz * (sz + 1.0) + nb - np.sum(overlap ** 2)


def run_one(spec, method, lambda_vector):
    """Run a single QED-UHF or QED-UKS calculation.

    ``method`` is ``"qed_uhf"`` or ``"qed_uks"``.
    """
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()

    mol = psi4.geometry(spec["geometry"])

    reference = "uhf" if method == "qed_uhf" else "uks"
    options = base_psi4_options(reference)
    options.update({
        "hilbert__cavity_frequency": OMEGA,
        "hilbert__cavity_coupling_strength": lambda_to_hilbert_coupling(lambda_vector),
        # 1 photon state => pure coherent-state CQED-SCF ground state, which is
        # what cqed_scf implements.  (Verified: N_PHOTON_STATES=2 gives the
        # identical SCF energy, since the coherent-state basis decouples the
        # photon sector from the ground-state energy.)
        "hilbert__n_photon_states": 1,
        "hilbert__use_coherent_state_basis": True,
        "hilbert__use_quadrupole_integrals": True,
        "hilbert__maxiter": 500,
    })
    if method == "qed_uks":
        options["hilbert__qed_dft_functional"] = FUNCTIONAL
    psi4.set_options(options)

    driver_name = "qed-scf" if method == "qed_uhf" else "qed-dft"
    energy, wfn = psi4.energy(driver_name, molecule=mol, return_wfn=True)

    record = {
        "energy": float(energy),
        "lambda_vector": [float(x) for x in np.asarray(lambda_vector)],
        "omega": OMEGA,
        "hilbert_coupling_strength": lambda_to_hilbert_coupling(lambda_vector),
        "nuclear_repulsion": float(mol.nuclear_repulsion_energy()),
    }
    try:
        record["s_squared"] = float(spin_squared(wfn))
    except Exception:  # the plugin wavefunction may not expose Ca/Cb subsets
        record["s_squared"] = None
    return record


def main():
    psi4.set_memory("8 GB")
    psi4.core.set_output_file(os.path.join(HERE, "hilbert_reference.out"), False)

    results = {
        "oracle": "hilbert (polaritonic_scf)",
        "psi4_version": psi4.__version__,
        "basis": BASIS,
        "functional": FUNCTIONAL,
        "omega": OMEGA,
        "n_photon_states": 1,
        "coupling_convention": "lambda = g * sqrt(2 * omega); we pass g = lambda / sqrt(2 * omega)",
        "lambda_vectors": {k: [float(x) for x in v] for k, v in LAMBDA_VECTORS.items()},
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
        for method in ("qed_uhf", "qed_uks"):
            results["systems"][name]["methods"][method] = {}
            for lam_label, lam in LAMBDA_VECTORS.items():
                label = "QED-UHF" if method == "qed_uhf" else f"QED-UKS/{FUNCTIONAL}"
                print(f"[hilbert] {name:<12s} {label:<16s} lambda={lam_label}", flush=True)
                results["systems"][name]["methods"][method][lam_label] = run_one(
                    spec, method, lam
                )

    with open(OUTFILE, "w") as handle:
        json.dump(results, handle, indent=2)

    print()
    print(f"Hilbert QED reference energies  ({BASIS}, scf_type=pk, omega={OMEGA})")
    print("=" * 92)
    print(
        f"{'system':<14s}{'method':<18s}{'lambda':<10s}"
        f"{'(lx, ly, lz)':<22s}{'energy / Eh':>22s}"
    )
    print("-" * 92)
    for name, entry in results["systems"].items():
        for method, per_lambda in entry["methods"].items():
            label = "QED-UHF" if method == "qed_uhf" else f"QED-UKS/{FUNCTIONAL}"
            for lam_label, data in per_lambda.items():
                lam = data["lambda_vector"]
                lam_str = f"({lam[0]:.2f}, {lam[1]:.2f}, {lam[2]:.2f})"
                print(
                    f"{name:<14s}{label:<18s}{lam_label:<10s}"
                    f"{lam_str:<22s}{data['energy']:>22.12f}"
                )
    print("=" * 92)
    print(f"\nwrote {OUTFILE}")


if __name__ == "__main__":
    main()
