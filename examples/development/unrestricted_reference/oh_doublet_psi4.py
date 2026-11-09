import psi4
"""
CQED-UHF single-point energy using CQEDConfig and CQEDCalculator.
"""

import psi4




# Define water from psi4numpy tutorial example
mol_str = """
0 2
O
H 1 0.9697
symmetry c1
"""

# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "6-311+G*",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
    "reference" : "uhf",
}

mol = psi4.geometry(mol_str)
psi4.set_options(psi4_options)

en, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)

#s_squared = wfn.variable("SCF TOTAL SPIN POPULATION") 
wfn_dict = wfn.to_dict()
print(list(wfn_dict.keys()))
#print(F"Psi4 got s_squared as {s_squared}")

