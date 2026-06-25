import psi4
import numpy as np

# Make sure we import the C++ core and the target driver module
from psi4 import core
from psi4.driver.procrouting.sapt import sapt_jk_terms

# 1. Configuration & Geometry Setup
psi4.set_memory('2 GB')
psi4.set_options({
'basis': "6-31G",
'scf_type': "df"
})

# Dimer geometry block (He -- He at 2.0 Angstroms)
dimer_geo = """
0 1
He 0.0 0.0 0.0
--
0 1
He 0.0 0.0 2.0
symmetry c1
units angstrom
no_reorient
no_com
"""

# Extract the individual fragments as distinct Psi4 Molecule objects
# keeping the Dimer Basis Set (DCBS) layout context intact.
dimer = psi4.geometry(dimer_geo)
monomer_A = dimer.extract_subsets(1, 2) # Fragment 1 (with ghost atoms on 2)
monomer_B = dimer.extract_subsets(2, 1) # Fragment 2 (with ghost atoms on 1)

print("==> Running Reference Monomer RHF Calculations (DCBS) <==\n")
e_scf_A, wfn_A = psi4.energy("scf", molecule=monomer_A, return_wfn=True)
e_scf_B, wfn_B = psi4.energy("scf", molecule=monomer_B, return_wfn=True)

# 2. Re-constructing the core JK builder required by sapt_jk_terms
# The driver needs an active JK object initialized with the full dimer basis.
full_basis = wfn_A.basisset()
# Build the corresponding auxiliary fit basis for DF
aux_basis = core.BasisSet.build(dimer, "DF_BASIS_SCF", "", "JKFIT", "cc-pvdz")

jk = core.JK.build_JK(full_basis, aux_basis)
jk.set_memory(int(5e8)) # Allocate memory slot for JK operations
jk.initialize()

print("\n==> Triggering sapt_jk_terms internal machinery <==\n")
# Step A: Build the cache (This wraps densities, eigenvalues, and custom J/K operators)
cache = sapt_jk_terms.build_sapt_jk_cache(wfn_A, wfn_B, jk, do_print=False)

# Step B: Call first-order electrostatics and exchange explicit routines
elst_results = sapt_jk_terms.electrostatics(cache, do_print=False)
exch_results = sapt_jk_terms.exchange(cache, jk, do_print=False)

# Safely shut down the JK tracking memory block
jk.C_clear()

# 3. Perform a Native standard SAPT0 run for side-by-side verification
print("\n==> Running Benchmark Native SAPT0 Calculation <==\n")
# Reset global molecule selector context to full dimer
psi4.activate(dimer)
e_sapt = psi4.energy("sapt0")

# Extract the exact benchmark variables recorded by Psi4's driver
native_elst = psi4.variable("SAPT ELST ENERGY")
native_exch = psi4.variable("SAPT EXCH ENERGY")

# 4. Final Comparison & Verification
print("\n" + "="*50)
print(f"{'SAPT0 Component Comparison':^50}")
print("="*50)
print(f"Manual Electrostatics (Elst10,r): {elst_results['Elst10,r']:16.8f} a.u.")
print(f"Native SAPT0 Electrostatics:     {native_elst:16.8f} a.u.")
print(f"Discrepancy:                     {abs(elst_results['Elst10,r'] - native_elst):16.2e} a.u.")
print("-"*50)
print(f"Manual Exchange (Exch10):         {exch_results['Exch10']:16.8f} a.u.")
print(f"Native SAPT0 Exchange:           {native_exch:16.8f} a.u.")
print(f"Discrepancy:                     {abs(exch_results['Exch10'] - native_exch):16.2e} a.u.")
print("="*50)
