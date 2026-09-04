"""QED-CIS on a QED-Kohn-Sham reference: water in a cavity, B3LYP.

Demonstrates the Kohn-Sham path.  The two-electron block carries an
exchange-correlation kernel, routed through Psi4's own TDA machinery, so the
same driver serves Hartree-Fock and Kohn-Sham references.

NOT QED-TDDFT, and not its Tamm-Dancoff approximation.  This is a configuration
interaction in a composite electron-photon Fock basis: it carries
|Phi_i^a, n>=1> configurations that the product ansatz of linear-response
QED-TDDFT cannot represent, treats the dipole self-energy as a genuine
two-electron operator rather than at mean-field level, and yields a correlated
ground state rather than excitations from an unrelaxed reference.  See
docs/qed_cis_formalism.tex, "Relationship to linear-response QED-TDDFT".

Run:  python examples/qed_cis/water_qed_cis_ks.py
"""

import numpy as np
import psi4

from cqed_scf import CQEDCalculator

psi4.set_memory("2 GB")
psi4.core.set_output_file("water_qed_tda_dft.out", False)

WATER = """
0 1
O   0.000000000000   0.000000000000  -0.068516219320
H   0.000000000000  -0.790689573744   0.543701060715
H   0.000000000000   0.790689573744   0.543701060715
no_reorient
no_com
symmetry c1
"""

PSI4_OPTIONS = {
    "basis": "6-31G",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "save_jk": True,
    "dft_spherical_points": 590,
}

HARTREE_TO_EV = psi4.constants.Hartree_energy_in_eV
OMEGA = 0.35  # Eh, in the region of water's lowest singlet excitations


def build(coupling):
    return CQEDCalculator(
        lambda_vector=np.array([0.0, 0.0, coupling]),
        psi4_options=PSI4_OPTIONS,
        omega=OMEGA,
        functional="B3LYP",
        density_fitting=True,
        quiet=True,
    )


def main():
    print("=" * 78)
    print("  QED-CIS / CQED-RKS(B3LYP) / 6-31G,  omega = 0.35 Eh")
    print("=" * 78)

    # cis() handles both reference types; the calculator's functional decides.
    # There is no tddft() entry point -- that name is reserved for the genuine
    # linear-response theory, which this is not.
    field_free = build(0.0).cis(WATER, nroots=6, print_results=False)
    coupled = build(0.05).cis(WATER, nroots=6, n_photon=1)

    print("\n  Cavity-induced shifts of the lowest excitations")
    print("  " + "-" * 62)
    print(f"  {'root':>5s} {'free (eV)':>11s} {'cavity (eV)':>12s} "
          f"{'shift (meV)':>12s} {'<b+b>':>9s}")

    free = field_free.excitation_energies
    cav = coupled.excitation_energies
    for root in range(1, min(6, free.size, cav.size)):
        shift = (cav[root] - free[root]) * HARTREE_TO_EV * 1000.0
        print(f"  {root:5d} {free[root] * HARTREE_TO_EV:11.4f} "
              f"{cav[root] * HARTREE_TO_EV:12.4f} {shift:12.3f} "
              f"{coupled.photon_numbers[root]:9.4f}")

    print(
        "\n  Roots are compared by index here only because the two calculations\n"
        "  share a photon-state count and the coupling is weak enough not to\n"
        "  reorder them.  For a coupling scan, track states by photon character\n"
        "  or overlap instead -- see mghp_rabi_scan.py."
    )


if __name__ == "__main__":
    main()
