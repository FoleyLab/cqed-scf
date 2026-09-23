"""Unrestricted CQED-SCF engine (UHF / UKS references).

The restricted engine in :mod:`cqed_scf.scf` handles closed-shell RHF and RKS.
This module is its open-shell counterpart, reached automatically by
:class:`~cqed_scf.calculator.CQEDCalculator` whenever the configured reference
is ``"uhf"`` or ``"uks"``.

The surrounding plumbing is complete: :class:`CQEDConfig` accepts and validates
unrestricted references, the calculator checks the geometry's charge and
multiplicity against the config and dispatches here, and gradients and response
theory refuse unrestricted references up front.  What is missing is the physics
below.

Reference energies to implement against live in
``examples/canonical/Unrestricted_Test_Examples/`` (Psi4 for UHF/UKS, Hilbert
``polaritonic_scf`` for QED-UHF/QED-UKS, cross-validated to <5e-13 Eh at
lambda = 0).

Implementation notes
--------------------
Start from :class:`cqed_scf.scf.CQEDSCF` and split each spin channel:

* Read the functional from ``config.base_scf_functional``, **not**
  ``config.functional``.  The former returns ``None`` for ``"uhf"`` and strips
  known dispersion suffixes; the latter would try to build a superfunctional for
  a Hartree-Fock reference.
* The restricted code carries an explicit factor of 2 throughout because its
  ``D`` is the alpha density alone.  With separate ``Da``/``Db`` those factors
  disappear and ``Dt = Da + Db`` takes their place -- in the Coulomb term, in the
  dipole expectation value (``mu_el`` near ``scf.py:358``), and in the dipole self-energy.
* :meth:`~cqed_scf.scf.CQEDSCF._build_JK` adds a single ``C_left_add(Cocc)``,
  relying on ``J_alpha == J_beta``.  Unrestricted needs both occupied blocks
  pushed through, with ``J`` summed over spins and ``K`` kept spin-resolved.
* :meth:`~cqed_scf.scf.CQEDSCF._build_vbase` builds a ``"RV"`` potential.  UKS
  needs ``"UV"``, with ``set_D([Da, Db])`` and ``compute_V([Va, Vb])``.
* The DIIS subspace in ``scf.py`` stores flat error/Fock arrays.  Either keep two
  subspaces or stack the alpha and beta errors into one vector.

Return contract
---------------
:meth:`run` must return ``(energy, results)``.  ``CQEDCalculator._run_scf``
returns that tuple unchanged, and :meth:`CQEDCalculator.energy` immediately reads
``results["energy_psi4"]``, so that key is mandatory.

``results`` should carry the keys the restricted engine produces (see the dict
built at the end of ``cqed_scf/scf.py``) so that shared consumers keep working,
with these spin-resolved additions and substitutions:

===========================  ==================================================
key                          meaning
===========================  ==================================================
``energy_scf``               total CQED-SCF energy
``energy_psi4``              the underlying Psi4 UHF/UKS energy (**required**)
``Ca``, ``Cb``               alpha/beta MO coefficients
``Da``, ``Db``               alpha/beta AO densities (each *unscaled*)
``orbital_energies_a/_b``    alpha/beta orbital energies
``nalpha``, ``nbeta``        occupied counts per spin
``s_squared``                <S^2> for the converged determinant
===========================  ==================================================

Keep ``density``, ``coefficients``, ``orbital_energies``, and ``ndocc`` out of
the unrestricted dict rather than aliasing them to the alpha channel.  Several
consumers (``cqed_scf.gradients``, ``cqed_scf.response``) read those keys and
silently assume the restricted factor-of-2 convention; a missing key raises,
while a plausible-looking alias would produce a wrong number.
"""

from __future__ import annotations

from typing import Any

from .references import CQEDConfig
import time

import psi4
import numpy as np
import opt_einsum as oe
from . import output

# ==> Define function to diagonalize F -- directly from psi4numpy <==
def diag_F(A, F, norb):
    F_p = A.dot(F).dot(A)
    e, C_p = np.linalg.eigh(F_p)
    C = A.dot(C_p)
    C_occ = C[:, :norb]
    D = np.einsum('pi,qi->pq', C_occ, C_occ, optimize=True)
    return (C, D)

# ==> Build DIIS Extrapolation Function - directly from psi4numpy <==
def diis_xtrap(F_list, DIIS_RESID):
    # Build B matrix
    B_dim = len(F_list) + 1
    B = np.empty((B_dim, B_dim))
    B[-1, :] = -1
    B[:, -1] = -1
    B[-1, -1] = 0
    for i in range(len(F_list)):
        for j in range(len(F_list)):
            B[i, j] = np.einsum('ij,ij->', DIIS_RESID[i], DIIS_RESID[j], optimize=True)

    # Build RHS of Pulay equation 
    rhs = np.zeros((B_dim))
    rhs[-1] = -1
      
    # Solve Pulay equation for c_i's with NumPy
    coeff = np.linalg.solve(B, rhs)
      
    # Build DIIS Fock matrix
    F_DIIS = np.zeros_like(F_list[0])
    for x in range(coeff.shape[0] - 1):
        F_DIIS += coeff[x] * F_list[x]
    
    return F_DIIS

class CQEDUSCF:
    """Unrestricted CQED-SCF driver.

    Intended workflow:
    1. Build separate alpha/beta reference wavefunctions from Psi4 UHF/UKS.
    2. Construct CQED one-electron and dipole self-energy contributions.
    3. Iterate alpha/beta Fock builds to convergence.
    4. Return ``(energy, results)`` following the contract in the module
       docstring.
    """

    def __init__(self, geometry: Any, config: CQEDConfig, method: str | None = None):

        # geometry will specify the geometry string
        self.geometry = geometry
        # config will have the psi4 options and the CQED options
        self.config = config
        functional = config.functional
        self.functional = functional
        self.density_fitting = config.density_fitting
        self.psi4_options = config.psi4_options


        # infer method if not explicitly provided
        if method is None:
            self.method = "uks" if functional is not None else "uhf"
        else:
            self.method = method.lower()



        if self.method not in ("uhf", "uks", "hybrid"):
            raise ValueError("method must be 'uhf', 'uks', or 'hybrid'")

        if self.method in ("uks", "hybrid") and self.functional is None:
            raise ValueError("functional must be provided for method='uks' or 'hybrid'")

        self.is_dft = self.method in ("uks", "hybrid")
        

    def run(self):
        """Run unrestricted CQED-SCF.

        Returns
        -------
        tuple[float, dict]
            The CQED-SCF energy and the results dictionary described in the
            module docstring.
        """

        output.banner("Unrestricted Open-Shell CQED-SCF Calculation")
        self._prepare_options()

        self.mol = psi4.geometry(self.geometry)
        psi4.set_options(self.config.psi4_options)

        # run psi4 to get the reference energy and wavefunction
        ref_method = self._reference_method_string()
        E_psi4, self.wfn = psi4.energy(ref_method, return_wfn=True)

        # get instance of mints object, call it `mints`
        # Hint: psi4.core.MintsHelper(...) takes the basis set of the Psi4 wavefunction, self.wfn.basisset()
        #<-- code goes here to get mints object -->
        mints = psi4.core.MintsHelper(self.wfn.basisset())

        # get overlap matrix, call it `S`
        # Hint: mints.ao_overlap() returns a psi4 Matrix; store S as a NumPy array with np.array(..., copy=True)
        # because the DIIS error vector below needs S as a NumPy array
        #<-- code goes here to get overlap matrix -->
        S = np.array(mints.ao_overlap(), copy=True)
        

        # get basic information about the system
        # number of basis functions
        nbf = self.wfn.nso()
        # number of alpha and beta electrons
        nalpha = self.wfn.nalpha()
        nbeta = self.wfn.nbeta()
        # number of double occupied orbitals (for unrestricted, this is the minimum of nalpha and nbeta)
        ndocc = min(nalpha, nbeta)
        # number of singly occupied orbitals
        nsocc = abs(nalpha - nbeta)

        # print this basic information to the output
        #output.print_basic_info(nbf, nalpha, nbeta, ndocc, nsocc)
        print(f"Number of basis functions: {nbf}")
        print(f"Number of alpha electrons: {nalpha}")
        print(f"Number of beta electrons: {nbeta}")
        print(f"Number of double occupied orbitals: {ndocc}")
        print(f"Number of singly occupied orbitals: {nsocc}")

        # get the canonical Psi4 orbital energies for alpha and beta, call them `self.eps_a` and `self.eps_b`
        # (these are reported in the results dictionary; the SCF guess below comes from the core Hamiltonian,
        # not from the Psi4 orbitals)
        # Hint: self.wfn.epsilon_a() and self.wfn.epsilon_b(), copied into NumPy arrays
        #<-- code goes here to get orbital energies -->
        self.eps_a = np.array(self.wfn.epsilon_a(), copy=True)
        self.eps_b = np.array(self.wfn.epsilon_b(), copy=True)


        # Memory check for ERI tensor
        # ==> Set Basic Psi4 Options <==
        # Memory specification
        psi4.set_memory(int(5e8))
        numpy_memory = 2
        I_size = (nbf**4) * 8.e-9
        print('\nSize of the ERI tensor will be {:4.2f} GB.'.format(I_size))
        if I_size > numpy_memory:
            psi4.core.clean()
            raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory \
                            limit of %4.2f GB." % (I_size, numpy_memory))

        # Build ERI Tensor, call it `I`
        # Hint: np.asarray(mints.ao_eri()) gives the 4-index array (pq|rs) in chemist's notation
        #<-- code goes here to build ERI tensor -->
        I = np.asarray(mints.ao_eri())


        # Build core Hamiltonian
        # Recall for QED-UHF, H_0 = T + V + Q_PF - <d> d
        # because we need <d> we need guess C_x first
        # So build guess from canonical core Hamiltonian and orthogonalization matrix

        #<-- code to build kinetic energy matrix T -->
        # Hint: mints.ao_kinetic()

        #<-- code to build nuclear attraction matrix V -->
        # Hint: mints.ao_potential()

        #<-- code to build canonical core Hamiltonian H_canonical = T + V -->

        # Construct AO orthogonalization matrix `A` = S^(-1/2)
        # Hint: get a *fresh* overlap matrix with A = mints.ao_overlap() -- do NOT reuse S!
        # Then call A.power(-0.5, 1.e-16); this modifies A in place and returns nothing,
        # so reusing S here would silently overwrite S with S^(-1/2) and break the DIIS error vector below.
        # A is a psi4 core Matrix, so convert it to a NumPy array with np.asarray(A)
        #<-- code goes here to build AO orthogonalization matrix -->

        # get guess coefficients and density from Core Hamiltonian
        # Hint - use diag_F function defined above to get guess coefficients and density
        # using the canonical core Hamiltonian and orthogonalization matrix
        # Cx, Dx = diag_F(A, H_canonical, nx) where x is a or b and nx is nalpha or nbeta
        #<-- code goes here to get guess coefficients and density `Ca`, `Da`, `Cb`, `Db` -->


        #<-- code to build dipole matrix `d_ao` -->
        # Hint: mu = [np.asarray(x) for x in mints.ao_dipole()] gives [mu_x, mu_y, mu_z]
        # recall d_ao = sum(lambda_i * mu_i for i in range(3)), with lambda from self.config.lambda_vector

        #<-- code to compute dipole expectation value <d> -->
        # recall <d>_a = Tr(Da d) and <d>_b = Tr(Db d) and <d> = <d>_a + <d>_b
        # Note: <d> here is the ELECTRONIC dipole only -- do not add the nuclear dipole.
        # In the coherent-state basis the nuclear contribution cancels, and it is the electronic <d>
        # that makes the -<d> d term in H_0 cancel the J_dse terms in the Fock matrix below.

        #<-- code to build quadrupole matrices Q and `Q_PF` -->
        # Hint: Q = [np.asarray(x) for x in mints.ao_quadrupole()] gives the 6 unique components
        # q = [Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz] and Q_PF = -0.5 * sum(lambda_i * lambda_j * Q_ij for i,j in range(3))
        # Index map (i,j) -> position in the list: (0,0)->0, (0,1)->1, (0,2)->2, (1,1)->3, (1,2)->4, (2,2)->5
        # Q_ij = Q_ji, so each off-diagonal term appears twice in the double sum:
        # Q_PF = -0.5*(l_x^2 Q_xx + l_y^2 Q_yy + l_z^2 Q_zz) - (l_x l_y Q_xy + l_x l_z Q_xz + l_y l_z Q_yz)

        #<-- code goes here to build the QED-UHF core Hamiltonian `H_0` = H_canonical + Q_PF - <d> d_ao -->

        # get nuclear repulsion energy, call it `E_nuc`
        # Hint: self.mol.nuclear_repulsion_energy()
        #<-- code goes here to get nuclear repulsion energy -->

        # pre-iteration values
        SCF_E = 0.0
        SCF_E_old = 0.0

        # Trial and residual lists for DIIS
        F_list_a = []
        F_list_b = []
        R_list_a = []
        R_list_b = []

        # get convergence criteria from psi4 options
        e_conv = self.psi4_options.get("e_convergence", 1.0e-10)
        d_conv = self.psi4_options.get("d_convergence", 1.0e-8)
        max_iter = 100

        for it in range(1, max_iter + 1):
            # build Ja and Jb matrices using the ERI tensor and the alpha and beta densities
            # recall definition Jx_{pq} = sum_{rs} (pq|rs) D_x^{rs}
            #<-- code goes here to build `Ja` and `Jb` matrices -->

            # build `J_dse_a` and `J_dse_b` matrices using the dipole matrix and the dipole expectation value
            # recall definition J_dse_x_{pq} = sum_{rs} d_pq d_rs D_x^{rs} = <d>_x d_pq
            #<-- code goes here to build `J_dse_a` and `J_dse_b` matrices -->

            # build `Ka` and `Kb` matrices
            # recall definition Kx_{pq} = sum_{rs} (pr|qs) D_x^{rs}
            #<-- code goes here to build `Ka` and `Kb` matrices -->

            # build `K_dse_a` and `K_dse_b` matrices using the dipole matrix and the dipole expectation value
            # recall definition K_dse_x_{pq} = sum_{rs} d_pr d_qs D_x^{rs}
            # (in matrix form this is just d_ao @ D_x @ d_ao)
            #<-- code goes here to build `K_dse_a` and `K_dse_b` matrices -->

            # build Fock matrices for alpha and beta
            # Recall F_x = H_0 + J_a + J_b - K_x + J_dse_a + J_dse_b - K_dse_x
            # where x is alpha or beta
            #<-- code goes here to build `Fa` and `Fb` matrices -->


            # DIIS error vectors (orthogonalized FDS - SDF)
            diis_r_a = A.dot(Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)).dot(A)
            diis_r_b = A.dot(Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)).dot(A)

            # Append trial and residual vectors to lists
            F_list_a.append(Fa)
            F_list_b.append(Fb)
            R_list_a.append(diis_r_a)
            R_list_b.append(diis_r_b)

            # Compute QED-UHF energy, call it `SCF_E`
            # Recall E_scf = 0.5 * (Tr((Da + Db) H_0) + Tr(Da Fa) + Tr(Db Fb)) + 0.5 * <d>^2 + E_nuc
            # The constant 0.5 * <d>^2 is needed so that all <d>-dependent terms cancel in the energy
            # (it vanishes when lambda = 0, so the cavity-free tests cannot catch it if it is missing!)
            #<-- code goes here to compute QED-UHF energy -->

            dE = SCF_E - SCF_E_old
            dRMS = 0.5 * (np.mean(diis_r_a**2) + np.mean(diis_r_b**2)) ** 0.5
            print(f"Iteration {it}: SCF Energy = {SCF_E:.12f} dE = {dE:.6e} dRMS = {dRMS:.6e}")

            if (abs(dE) < e_conv) and (dRMS < d_conv):
                print(f"SCF converged in {it} iterations.")
                break

            SCF_E_old = SCF_E

            # DIIS Extrapolation
            if it >= 8:
                Fa = diis_xtrap(F_list_a, R_list_a)
                Fb = diis_xtrap(F_list_b, R_list_b)

            # compute new orbital guess
            Ca, Da = diag_F(A, Fa, nalpha)
            Cb, Db = diag_F(A, Fb, nbeta)

            # Update <d>_a and <d>_b expectation values and QED Core Hamiltonian
            #<-- code goes here to update <d>_a and <d>_b expectation values and QED Core Hamiltonian -->

            # max iterations check
            if it == max_iter:
                psi4.core.clean()
                raise Exception(f"SCF did not converge in {max_iter} iterations.")

        # results dictionary to return
        results = {
          "energy_scf": SCF_E,
          "energy_psi4": E_psi4,
          "Ca": Ca,
          "Cb": Cb,
          "Da": Da,
          "Db": Db,
          "orbital_energies_a": self.eps_a,
          "orbital_energies_b": self.eps_b,
          "nalpha": nalpha,
          "nbeta": nbeta,
          "s_squared": nalpha-nbeta,
        }

        return SCF_E, results


        




    def _prepare_options(self):
        opts = dict(self.config.psi4_options)

        if self.method == "uhf":
            opts["reference"] = "uhf"
        else:
            opts["reference"] = "uks"

        if self.density_fitting:
            opts["scf_type"] = "df"

        psi4.set_options(opts)

    def _reference_method_string(self):
        if self.method == "uhf":
            return "scf"
        return self.functional
    
CQEDUHFSCF = CQEDUSCF
