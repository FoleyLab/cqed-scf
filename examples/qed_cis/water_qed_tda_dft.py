"""QED-TDA-DFT: water in a cavity with a B3LYP reference.

Demonstrates the Kohn-Sham path.  The two-electron block carries an
exchange-correlation kernel, routed through Psi4's own TDA machinery, so the
same driver serves Hartree-Fock and Kohn-Sham references.

Run:  python examples/qed_cis/water_qed_tda_dft.py
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
    print("  QED-TDA-DFT: water / B3LYP / 6-31G,  omega = 0.35 Eh")
    print("=" * 78)

    # tddft() is cis() with a Kohn-Sham reference, named for what it is
    field_free = build(0.0).tddft(WATER, nroots=6, print_results=False)
    coupled = build(0.05).tddft(WATER, nroots=6, n_photon=1)

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
