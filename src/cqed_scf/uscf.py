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
  dipole expectation value at ``scf.py:341``, and in the dipole self-energy.
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

        # get instance of mints object
        mints = psi4.core.MintsHelper(self.wfn.basisset())

        # get overlap matrix
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

        # get the canonical orbital energies and coefficients for alpha and beta
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

        # Build ERI Tensor
        I = np.asarray(mints.ao_eri())

        # Build core Hamiltonian
        T = np.asarray(mints.ao_kinetic())
        V = np.asarray(mints.ao_potential())
        H = T + V

        # Construct AO orthogonalization matrix A
        A = mints.ao_overlap()
        A.power(-0.5, 1.e-16)
        A = np.asarray(A)

        # get guess coefficients and density frojm Core Hamiltonian
        Ca, Da = diag_F(A, H, nalpha)
        Cb, Db = diag_F(A, H, nbeta)

        # get nuclear repulsion energy
        E_nuc = self.mol.nuclear_repulsion_energy()

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

        for it in range(1, 501):
            # build Ja and Jb matrices
            Ja = oe.contract("pqrs,rs->pq", I, Da, optimize="optimal")
            Jb = oe.contract("pqrs,rs->pq", I, Db, optimize="optimal")

            # build Ka and Kb matrices
            Ka = oe.contract("prqs,rs->pq", I, Da, optimize="optimal")
            Kb = oe.contract("prqs,rs->pq", I, Db, optimize="optimal")

            # build Fock matrices for alpha and beta
            Fa = H + Ja + Jb - Ka
            Fb = H + Ja + Jb - Kb

            # DIIS extrapolation
            diis_r_a = A.dot(Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)).dot(A)
            diis_r_b = A.dot(Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)).dot(A)

            # Append trial and residual vectors to lists
            F_list_a.append(Fa)
            F_list_b.append(Fb)
            R_list_a.append(diis_r_a)
            R_list_b.append(diis_r_b)

            # Compute UHF energy
            SCF_E = oe.contract("pq,pq->", (Da + Db), H, optimize="optimal")
            SCF_E += oe.contract("pq,pq->", Da, Fa, optimize="optimal")
            SCF_E += oe.contract("pq,pq->", Db, Fb, optimize="optimal")
            SCF_E *= 0.5
            SCF_E += E_nuc

            dE = SCF_E - SCF_E_old
            dRMS = 0.5 * (np.mean(diis_r_a**2) + np.mean(diis_r_b**2)) ** 0.5
            print(f"Iteration {it}: SCF Energy = {SCF_E:.12f} dE = {dE:.6e} dRMS = {dRMS:.6e}")

            if (abs(dE) < e_conv) and (dRMS < d_conv):
                print(f"SCF converged in {it} iterations.")
                break

            SCF_E_old = SCF_E

            # DIIS Extrapolation
            if it >= 2:
                Fa = diis_xtrap(F_list_a, R_list_a)
                Fb = diis_xtrap(F_list_b, R_list_b)

            # compute new orbital guess
            Ca, Da = diag_F(A, Fa, nalpha)
            Cb, Db = diag_F(A, Fb, nbeta)

            # max iterations check
            if it == 500:
                psi4.core.clean()
                raise Exception("SCF did not converge in 500 iterations.")

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
