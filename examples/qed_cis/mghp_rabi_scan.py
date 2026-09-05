"""Rabi splitting versus coupling strength.

Scans the cavity coupling and tracks the lower/upper polariton pair.  In the
two-level limit the splitting is linear in |lambda|, so the ratio
splitting / |lambda| should be close to constant across the scan -- a cheap way
to see the model behaving as expected before trusting anything more elaborate.

Run:  python examples/qed_cis/mghp_rabi_scan.py
"""

import numpy as np
import psi4

from cqed_scf import CQEDCalculator

psi4.set_memory("2 GB")
psi4.core.set_output_file("mghp_rabi_scan.out", False)

MGHP = """
Mg
H 1 2.2
symmetry c1
1 1
"""

PSI4_OPTIONS = {
    "basis": "cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "save_jk": True,
}

HARTREE_TO_EV = psi4.constants.Hartree_energy_in_eV
OMEGA = 4.75 / HARTREE_TO_EV
COUPLINGS = [0.0025, 0.0050, 0.0075, 0.0100, 0.0125, 0.0150]


def polariton_pair(results):
    """The LP/UP pair, by weight on the photonic reference state.

    Selecting on total photon number instead would pick up photon-dressed
    electronic states |Phi_i^a,1>, which carry a photon without mixing.
    """

    return results.polariton_indices()


def main():
    print("=" * 78)
    print("  Rabi splitting vs coupling:  MgH+ / cc-pVDZ,  hbar*omega = 4.75 eV")
    print("=" * 78)
    print(f"\n  {'lambda_z':>9s} {'LP (eV)':>10s} {'UP (eV)':>10s} "
          f"{'split (eV)':>11s} {'split/lambda':>13s} {'<b+b> LP':>10s} {'<b+b> UP':>10s}")
    print("  " + "-" * 76)

    for coupling in COUPLINGS:
        calculator = CQEDCalculator(
            lambda_vector=np.array([0.0, 0.0, coupling]),
            psi4_options=PSI4_OPTIONS,
            omega=OMEGA,
            quiet=True,
        )
        results = calculator.cis(MGHP, nroots=8, n_photon=1, print_results=False)

        lower, upper = polariton_pair(results)
        lp = results.excitation_energies[lower] * HARTREE_TO_EV
        up = results.excitation_energies[upper] * HARTREE_TO_EV
        splitting = up - lp

        print(f"  {coupling:9.4f} {lp:10.4f} {up:10.4f} {splitting:11.4f} "
              f"{splitting / coupling:13.2f} "
              f"{results.photon_numbers[lower]:10.4f} "
              f"{results.photon_numbers[upper]:10.4f}")

    print(
        "\n  A roughly constant split/lambda column is the signature of linear\n"
        "  (two-level) Rabi splitting.  Departures at larger coupling are real:\n"
        "  the dipole self-energy grows as lambda^2 and more electronic states\n"
        "  mix in."
    )


if __name__ == "__main__":
    main()
