"""QED-CIS polaritons of MgH+ in a resonant cavity.

MgH+ at 2.2 Angstrom has its X -> A transition near 4.75 eV.  Tuning the cavity
to that frequency splits it into a lower and an upper polariton, and this is the
system the method was originally demonstrated on.

Run:  python examples/qed_cis/mghp_polaritons.py
"""

import numpy as np
import psi4

from cqed_scf import CQEDCalculator

psi4.set_memory("2 GB")
psi4.core.set_output_file("mghp_polaritons.out", False)

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

# Reference values from the qed-ci determinant-basis implementation
REFERENCE = {
    "ground": -199.86358254419457,
    "lower_polariton": -199.69776087489558,
    "upper_polariton": -199.68066502792058,
}


def main():
    print("=" * 78)
    print("  QED-CIS: MgH+ / cc-pVDZ at 2.2 A,  hbar*omega = 4.75 eV")
    print("=" * 78)

    # ---- cavity-free reference, for the bare electronic transition ----------
    field_free = CQEDCalculator(
        lambda_vector=np.zeros(3),
        psi4_options=PSI4_OPTIONS,
        omega=OMEGA,
        quiet=True,
    )
    bare = field_free.cis(MGHP, nroots=6, print_results=False)
    bare_excitation = bare.excitation_energies[bare.excitation_energies > 1e-8][0]
    print(f"\n  Bare X -> A transition : {bare_excitation * HARTREE_TO_EV:8.4f} eV")

    # ---- coupled to the cavity ---------------------------------------------
    coupled = CQEDCalculator(
        lambda_vector=np.array([0.0, 0.0, 0.0125]),
        psi4_options=PSI4_OPTIONS,
        omega=OMEGA,
        quiet=True,
    )
    results = coupled.cis(MGHP, nroots=8, n_photon=1)

    # ---- identify the polaritons -------------------------------------------
    # By weight on the photonic REFERENCE state |Phi_0,1>, not by total photon
    # number.  <b+b> counts every amplitude in the one-photon block, including
    # |Phi_i^a,1> -- an electronic excitation carrying a photon.  That dressed
    # state sits near (bare transition + omega) with <b+b> close to 1 while
    # mixing with nothing, and outranks the genuine lower polariton.
    lower, upper = results.polariton_indices()

    rabi = results.excitation_energies[upper] - results.excitation_energies[lower]

    shared = results.reference_weights[[lower, upper], 1].sum()

    print("\n  Polariton assignment (by weight on the photonic reference |Phi_0,1>)")
    print("  " + "-" * 70)
    print(f"  {'':6s} {'E (Eh)':>18s} {'w (eV)':>10s} {'|c_1|^2':>9s} {'<b+b>':>9s}")
    for label, root in (("LP", lower), ("UP", upper)):
        print(
            f"  {label:6s} {results.total_energies[root]:18.10f} "
            f"{results.excitation_energies[root] * HARTREE_TO_EV:10.4f} "
            f"{results.reference_weights[root, 1]:9.4f} "
            f"{results.photon_numbers[root]:9.4f}"
        )
    print(f"\n  Rabi splitting  : {rabi:.8f} Eh = {rabi * HARTREE_TO_EV:.4f} eV")
    print(f"  Shared |c_1|^2  : {shared:.4f}   (a genuine pair shares the photon; expect ~1)")

    # ---- compare against the independent determinant-basis reference --------
    print("\n  Against the qed-ci determinant-basis reference")
    print("  " + "-" * 62)
    for label, key, root in (
        ("ground", "ground", 0),
        ("LP", "lower_polariton", lower),
        ("UP", "upper_polariton", upper),
    ):
        ours = results.total_energies[root]
        delta = ours - REFERENCE[key]
        print(f"  {label:8s} ours {ours:18.10f}   ref {REFERENCE[key]:18.10f}   "
              f"diff {delta: .2e}")

    print(
        "\n  Note: the QED-CIS ground state relaxes "
        f"{results.eigenvalues[0]:.3e} Eh below the CQED-SCF reference, so\n"
        "  excitation energies must be taken as E_k - E_0 rather than as raw\n"
        "  eigenvalues."
    )


if __name__ == "__main__":
    main()
