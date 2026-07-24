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

## Water-methylamine QED-SAPT0 JK/DSE discrepancy notes

Reference confirmation:

- The hard-coded dense reference values in `examples/water_methylamine/water_methylamine_qed_sapt_jk.py` were independently reproduced by running the dense reference driver in `examples/water_methylamine/water_methylamine_qed_sapt.py`, which uses `cqed_scf/sapt/qed_sapt0.py::QEDSAPT0Driver` with the same geometry, lambda vector `[0.0, 0.0, 0.1]`, omega `0.1`, `jun-cc-pVDZ`, and `include_cavity_terms=True`.
- The reproduced values are:
  - Electrostatics: `-0.0086737093`
  - Exchange: `0.0030805286`
  - Dispersion: `-0.0027151107`
  - Exchange-Dispersion: `0.0001966590`
  - Induction: `-0.0017887767`
  - Exchange-Induction: `0.0008167150`
  - Total: `-0.0090836941`

Cavity-modified monomer data flow:

- `cqed_scf/scf.py::CQEDSCF.run()` updates the returned Psi4 wavefunction with the final CQED-SCF state in `_update_wfn_with_cqed()`, including `Ca`, `Cb`, `Da`, `Db`, `epsilon_a`, and `epsilon_b`.
- `cqed_scf/sapt/monomer.py::SAPTMonomer.from_cqed_scf()` stores that same updated wavefunction in `scf_results["wfn"]`.
- `cqed_scf/sapt/qed_sapt_jk.py::build_sapt_jk_cache()` then reads orbitals and orbital energies from the wavefunction via `Ca_subset("AO", "OCC")`, `Ca_subset("AO", "VIR")`, `epsilon_a_subset("AO", "OCC")`, and `epsilon_a_subset("AO", "VIR")`.
- Spot checks showed zero numerical difference between the stored monomer coefficient/epsilon arrays and the wavefunction subsets. The JK path is therefore being fed the cavity-relaxed density, orbital coefficients, and orbital energies.

Needed fixes / follow-up:

1. Exchange comparison key in `examples/water_methylamine/water_methylamine_qed_sapt_jk.py`
   - The dense reference `qed_sapt0.py::compute_Exch100()` implements the S^2 exchange expression.
   - `qed_sapt_jk.py::exchange()` returns both `Exch10(S^2)` and `Exch10`.
   - The water-methylamine JK comparison currently reports `exch["Exch10"]`, which is the S^inf-style value from the Psi4 JK helper path. That is the source of the apparent exchange discrepancy.
   - For comparison against the current dense reference, use `exch["Exch10(S^2)"]`.
   - Diagnostic values:
     - Dense exchange: `0.0030805285602`
     - JK `Exch10(S^2)`: `0.0030805285647`
     - JK `Exch10`: `0.0030886178404`

2. Coupled induction response sign/convention in `cqed_scf/sapt/qed_sapt_jk.py` and `cqed_scf/sapt/dse_jk.py`
   - Direct/uncoupled induction agrees between dense and JK/DSE to numerical noise, so the induction RHS/electrostatic potential construction is OK.
   - The discrepancy appears in coupled response:
     - Dense total `Ind20,r`: `-0.0017887766852`
     - JK/DSE total `Ind20,r`: `-0.0017859880344`
   - `dse_jk.py::DSECPHF.hx_array()` matches the dense cavity Hessian action in the algebra/convention used by `qed_sapt0.py::chf()`.
   - Psi4 `wfn.cphf_Hx()` appears to use the opposite sign convention relative to the dense matrix solve. In `qed_sapt_jk.py::_sapt_cpscf_solve()`, the DSE response action is currently added with `xA.axpy(1.0, dse_cphf_A.hx_matrix(...))` and similarly for B.
   - This sign/convention needs to be reconciled. A local sign-flip experiment for the DSE Hx term largely corrected induction, which supports this as the right area to fix, but it should be implemented with a clear convention rather than an ad hoc patch.

3. Exchange-induction coupled-response mismatch
   - Uncoupled exchange-induction agrees between dense and JK/DSE:
     - Dense `Exch-Ind20,u`: `0.0006174109081502`
     - JK/DSE `Exch-Ind20,u`: `0.0006174109081214`
   - Coupled exchange-induction differs:
     - Dense total `Exch-Ind20,r`: `0.0008167149799`
     - JK/DSE total `Exch-Ind20,r`: `0.0008137244180`
   - There is already a standard-only coupled exchange-induction difference of about `2.6e-6 Eh`, before adding DSE terms, so this is not purely a cavity/DSE issue.
   - Relevant dense code: `qed_sapt0.py::compute_Eexchind200()` and `qed_sapt0.py::chf()`.
   - Relevant JK code: `qed_sapt_jk.py::induction()`, especially the EX_A/EX_B potential construction and `_sapt_cpscf_solve()` response amplitudes.
   - Recommended next diagnostic: compare dense `compute_Eexchind200()` using the exact dense CPHF amplitudes against the JK EX_A/EX_B contractions using the same amplitudes. This will separate EX potential construction errors from response-amplitude convention errors.
