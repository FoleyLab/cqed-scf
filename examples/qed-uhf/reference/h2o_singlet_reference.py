"""
UHF single-point energy using psi4
"""

import psi4



# Define water from psi4numpy tutorial example
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4.set_options({
    "basis": "cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
})


# ---------------------------------------------------------
# Compute and print energy
# ---------------------------------------------------------
energy = psi4.energy("scf", molecule=mol)

print(F"Final UHF Energy is {energy:16.12e}")

