# Development notes about QED-SAPT Driver

1. class QEDSAPT0Driver constructure can take a geometry string for the full dimer with the '--' delimiter and handle it appropriately to get dimer and monomer quantities. 

2. Method prepare_geometries(self) will use the psi4 core method .extract_subsets(i,j) to extract monomer A and monomer B geometries using appropriate ghost atoms from the dimer geometry.  This method returns a psi4 geometry object, but we also capture a geometry string using the psi4.core method .create_psi4_strong_from_molecule.

3. Method prepare_monomers(self) will take the three strings (dimer, monomer A, monomer B) and run scf calculations and collect results.  It stores many useful non-mints quantities (ndocc, nvirt, C, Co, Cv, E_nuc, eps) for dimer and monomers for later use.  These quantities are exposed from the property methods of the monomer class, and they originate from scf calculations on the dimer and monomers.

4. Methods build_orbitals, build_sizes, and build_slices follow the organization of monomer A and B quantities into dictionaries from the helper_SAPT in psi4numpy

5. Method build_integrals is where the base integral quantities are / will be built (partially complete).  It calls build_orbitals, build_sizes, and build_slices before it starts building integral quantities.

6. Method run() should call prepare_geometries, prepare_monomers, build_integrals, and then evaluate SAPT terms

7. Methods .v() is implemented and tested for He dimer, methods .s(), .eps(), .potential(), and .vt() still require implementation and testing.  They should be called by .build_integrals or by .run()

8. After implementation from scratch, we will want to try to hook into as much psi4 capability as possible for a more performant code, i.e. avoiding full 2-ERI builds in favor of JK builds that use density fitting, etc.  This file https://github.com/psi4/psi4/blob/master/psi4/driver/procrouting/sapt/sapt_jk_terms.py looks to be quite useful, as the first function is build_sapt_jk_cach and takes wfn objects for monomer a and b. I have pinned a gemini chat about how to use the helper functions in this file! 

9. Added example calling sapt_jk_terms to compute first-order terms (E_{elest} and E_{exch}) in examples/he_dimer/he_dimer_sapt_jk_test.py, this approach seems promising!
