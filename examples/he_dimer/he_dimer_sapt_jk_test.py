import psi4
import numpy as np

# Make sure we import the C++ core and the target driver module
from psi4 import core
from cqed_scf.sapt import qed_sapt_jk
from cqed_scf.sapt.dse_jk import DSEJK, PauliFierzJK, DSECPHF

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

# 2. Re-constructing the core JK builder required by local qed_sapt_jk
# The driver needs an active JK object initialized with the full dimer basis.
full_basis = wfn_A.basisset()
# Build the corresponding auxiliary fit basis for DF
aux_basis = core.BasisSet.build(dimer, "DF_BASIS_SCF", "", "JKFIT", "cc-pvdz")

jk = core.JK.build_JK(full_basis, aux_basis)
jk.set_memory(int(5e8)) # Allocate memory slot for JK operations
jk.initialize()

# Cavity/DSE scaffold: lambda_vector is zero for now, and DSEJK currently
# returns zero J/K contributions. Once DSEJK.jk_from_density is implemented,
# this example becomes the PF-SAPT0 test driver.
lambda_vector = np.array([0.0, 0.0, 0.0])

mints = core.MintsHelper(full_basis)
mu_x, mu_y, mu_z = [np.asarray(mu) for mu in mints.ao_dipole()]
d_ao = (
    lambda_vector[0] * mu_x
    + lambda_vector[1] * mu_y
    + lambda_vector[2] * mu_z
)

dse_jk = DSEJK(
    d_ao=d_ao,
    j_scale=1.0,
    k_scale=1.0,
    enabled=True,
    metadata={"description": "No-op DSEJK scaffold for future PF-SAPT0 work"},
)

pf_jk = PauliFierzJK(jk, dse_jk=dse_jk)

print("\n==> Triggering local qed_sapt_jk internal machinery <==\n")
# Step A: Build the cache (This wraps densities, eigenvalues, and custom J/K operators)
cache = qed_sapt_jk.build_sapt_jk_cache(
    wfn_A,
    wfn_B,
    pf_jk,
    do_print=False,
    dse_jk=dse_jk,
)

# Step B: Call first-order electrostatics and exchange explicit routines
elst_results = qed_sapt_jk.electrostatics(cache, do_print=False)
exch_results = qed_sapt_jk.exchange(cache, pf_jk, do_print=False)

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
