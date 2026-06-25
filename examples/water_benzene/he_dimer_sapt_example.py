"""
Example input script for future QED-SAPT0 workflow.

System: benzene--water
Purpose:
    Demonstrate how the new cqed_scf architecture will organize a QED-SAPT0
    calculation before the SAPT physics is implemented.

Important:
    - Benzene is monomer A.
    - Water is monomer B.
    - Monomer calculations should be performed in the dimer basis.
    - Therefore, monomer A is computed with water atoms as ghosts.
    - Monomer B is computed with benzene atoms as ghosts.

Expected future flow:
    CQEDConfig
        -> QEDSAPT0Driver
            -> SAPTMonomer.from_cqed_scf(...)
                -> CQEDSCF.run(...)
            -> sapt.components.compute_elst10(...)
            -> sapt.components.compute_exch10(...)
            -> sapt.components.compute_ind20(...)
            -> sapt.components.compute_disp20(...)
            -> QEDSAPT0Results
"""

import array

import numpy as np
import psi4

from cqed_scf import CQEDConfig
from cqed_scf.sapt import QEDSAPT0Driver, SAPTMonomer


# ---------------------------------------------------------
# Geometry data
# ---------------------------------------------------------

dimer = """
He 0.0000000000 0.0000000000 0.0000000000
--
He 0.0000000000 0.0000000000 2.0000000000
symmetry c1
units angstrom
no_reorient
no_com
"""


# ---------------------------------------------------------
# Psi4 / CQED options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "6-31g",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

lambda_vector = np.array([0.0, 0.0, 0.00])
omega = 0.07349864501573


# ---------------------------------------------------------
# Central configuration object
# ---------------------------------------------------------

config = CQEDConfig(
    lambda_vector=lambda_vector,
    omega=omega,
    psi4_options=psi4_options,
    reference="rhf",
    functional=None,
    density_fitting=False,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=True,
)

print("CQED-SAPT0 configuration")
print("========================")
print(f"Reference        : {config.reference}")
print(f"Functional       : {config.functional}")
print(f"Base functional  : {config.base_scf_functional}")
print(f"Density fitting  : {config.density_fitting}")
print(f"Lambda vector    : {config.lambda_vector}")
print()


# ---------------------------------------------------------
# Optional: build SAPTMonomer objects explicitly
# ---------------------------------------------------------
#
# In the future, SAPTMonomer.from_cqed_scf(...) should:
#
#   1. Receive monomer_a_geometry or monomer_b_geometry.
#   2. Call CQEDSCF using the shared CQEDConfig.
#   3. Store the resulting SCF dictionary.
#   4. Expose convenience properties:
#        monomer.C
#        monomer.eps
#        monomer.D
#        monomer.mints
#        monomer.wfn
#        monomer.d_ao
#        monomer.ndocc
#
# These monomers are then passed into QEDSAPT0Driver or into
# lower-level component routines.

try:
    print(" Doing monomer A calculation")
    monomer_a = SAPTMonomer.from_cqed_scf(
        label="helium_a",
        geometry=monomer_a_string,
        charge=0,
        multiplicity=1,
        config=config,
    )
    print(F"Result is monomer_a.energy_scf: {monomer_a.energy_scf}")

    print(" Doing monomer B calculation")
    monomer_b = SAPTMonomer.from_cqed_scf(
        label="helium_b",
        geometry=monomer_b_string,
        charge=0,
        multiplicity=1,
        config=config,
    )
    print(F"Result is monomer_b.energy_scf: {monomer_b.energy_scf}")
    print("Doing dimer calculation")
    dimer = SAPTMonomer.from_cqed_scf(
        label="dimer",
        geometry=dimer_string,
        charge=0,
        multiplicity=1,
        config=config,
    )
    print(F"Result is dimer.energy_scf: {dimer.energy_scf}")

except NotImplementedError:
    print("SAPTMonomer.from_cqed_scf is scaffolded but not implemented yet.")
    monomer_a = None
    monomer_b = None


# ---------------------------------------------------------
# QED-SAPT0 driver
# ---------------------------------------------------------
#
# Intended future workflow:
#
#   driver = QEDSAPT0Driver(...)
#
#   driver.prepare_geometries()
#       - parse dimer geometry
#       - identify monomer A and B atoms
#       - build dimer-basis monomer geometries using ghost atoms
#
#   driver.build_monomers()
#       - construct SAPTMonomer objects
#       - each SAPTMonomer delegates its SCF step to CQEDSCF
#
#   driver.build_integrals()
#       - first implementation: full two-electron integrals
#       - future implementation: density-fitted integral backend
#
#   driver.compute_components()
#       - compute_elst10(...)
#       - compute_exch10(...)
#       - compute_ind20(...)
#       - compute_disp20(...)
#       - compute_qed_dse_cross(...)
#
#   results = driver.run()
#       - returns QEDSAPT0Results

driver = QEDSAPT0Driver(
    dimer_geometry=dimer_geometry,
    dimer=dimer,
    monomer_a=monomer_a,
    monomer_b=monomer_b,
    config=config,
    # These are intentionally explicit for future development.
    # First implementation will use full ERIs.
    integral_backend="full_eri",
)

monomers = driver.prepare_monomers()
driver.build_integrals(monomers)
print("Printing the orbitals object")
print(driver.orbitals)

V = driver.v("arbs")
print("Successfully built integral intermediates. Example V(arbs) shape:", V.shape)
print(V)

#try:
#    monomers = driver.prepare_monomers()
#    driver.build_integrals(monomers)
#    V = driver.v("arbs")
#    print("Successfully built integral intermediates. Example V(arbs) shape:", V.shape)
#    print(V)
#    #results = driver.run()

    #print("\nQED-SAPT0 results")
    #print("=================")
    #print(results.summary())

#except NotImplementedError:
#    print("\nQEDSAPT0Driver is scaffolded but SAPT physics is not implemented yet.")
#    print("This script successfully demonstrates the intended architecture/data flow.")
