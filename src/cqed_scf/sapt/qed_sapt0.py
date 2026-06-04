"""Architecture scaffold for QED-SAPT0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple
import opt_einsum as oe
import numpy as np

from ..references import CQEDConfig

from .monomer import SAPTMonomer
from .results import QEDSAPT0Results


@dataclass
class QEDSAPT0Driver:
    """Future QED-SAPT0 driver.

    The driver is responsible for data orchestration, not component physics:

    1. Prepare dimer and monomer geometries.
    2. Run or attach monomer CQED-SCF references.
    3. Build full two-electron integral intermediates.
    4. Call SAPT component functions.
    5. Return a :class:`QEDSAPT0Results` object.
    """

    dimer_geometry: Any
    config: CQEDConfig
    dimer : Optional[SAPTMonomer] = None
    monomer_a: Optional[SAPTMonomer] = None
    monomer_b: Optional[SAPTMonomer] = None
    monomer_definitions: Optional[Sequence[Any]] = None
    monomer_indices: Optional[Tuple[Sequence[int], Sequence[int]]] = None
    integral_backend: str = "full_eri"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault("integral_backend", self.integral_backend)

    def prepare_geometries(self) -> Tuple[str, str, str]:
        """Build dimer and ghosted monomer geometry strings from a Psi4 dimer."""

        monomer_A_geometry = self.dimer_geometry.extract_subsets(1, 2)
        monomer_B_geometry = self.dimer_geometry.extract_subsets(2, 1)

        dimer_string = self.dimer_geometry.create_psi4_string_from_molecule()
        monomer_A_string = monomer_A_geometry.create_psi4_string_from_molecule()
        monomer_B_string = monomer_B_geometry.create_psi4_string_from_molecule()

        return dimer_string, monomer_A_string, monomer_B_string

    def prepare_monomers(self) -> Tuple[SAPTMonomer, SAPTMonomer, SAPTMonomer]:
        """Prepare or retrieve monomer references."""

        if self.dimer is not None and self.monomer_A is not None and self.monomer_B is not None:
            return self.dimer, self.monomer_A, self.monomer_B

        dimer_string, monomer_A_string, monomer_B_string = self.prepare_geometries()

        self.dimer = SAPTMonomer.from_cqed_scf(
            label="dimer",
            geometry=dimer_string,
            config=self.config,
        )
        self.monomer_A = SAPTMonomer.from_cqed_scf(
            label="monomer_A",
            geometry=monomer_A_string,
            config=self.config,
        )
        self.monomer_B = SAPTMonomer.from_cqed_scf(
            label="monomer_B",
            geometry=monomer_B_string,
            config=self.config,
        )
        # store basic quantities as self attributes
        # scf energies for use in the induction component and for reporting
        self.E_scf_dimer = self.dimer.energy_scf
        self.E_scf_A = self.monomer_A.energy_scf
        self.E_scf_B = self.monomer_B.energy_scf
               

        # size of orbital subspaces for use in component functions and for reporting
        self.ndocc_dimer = self.dimer.ndocc
        self.nvirt_dimer = self.dimer.nvirt
        self.ndocc_A = self.monomer_A.ndocc
        self.nvirt_A = self.monomer_A.nvirt
        self.ndocc_B = self.monomer_B.ndocc
        self.nvirt_B = self.monomer_B.nvirt
        
        # currently assumes closed shell, nsocc = 0
        self.nsocc_dimer = 0
        self.nsocc_A = 0
        self.nsocc_B = 0

        # orbital coefficients
        self.C_dimer = self.dimer.C
        self.C_A = self.monomer_A.C
        self.C_B = self.monomer_B.C
        self.Co_A = self.monomer_A.Co
        self.Cv_A = self.monomer_A.Cv
        self.Co_B = self.monomer_B.Co
        self.Cv_B = self.monomer_B.Cv

        # orbtial energies
        self.eps_dimer = self.dimer.eps
        self.eps_A = self.monomer_A.eps
        self.eps_B = self.monomer_B.eps

        # nuclear repulsion energy for reporting and use in the electrostatics component
        self.E_nuc_dimer = self.dimer.nuc_rep
        self.E_nuc_A = self.monomer_A.nuc_rep
        self.E_nuc_B = self.monomer_B.nuc_rep
        self.nuc_rep = self.E_nuc_dimer - self.E_nuc_A - self.E_nuc_B
        self.vt_nuc_rep = self.nuc_rep / ((2 * self.ndocc_A + self.nsocc_A) * (2 * self.ndocc_B + self.nsocc_B))

        # lambda-scaled expectation values of the electronic dipole operator, <d_el>_dimer, <d_el>_A, and <d_el>_B
        self.d_exp_dimer = self.dimer.d_exp_el 
        self.d_exp_A = self.monomer_A.d_exp_el
        self.d_exp_B = self.monomer_B.d_exp_el

        # lambda-scaled dipole integrals in ao basis, d_dimer, d_A, d_B
        self.d_dimer = self.dimer.d_ao
        self.d_A = self.monomer_A.d_ao
        self.d_B = self.monomer_B.d_ao 
        assert np.allclose(self.d_A, self.d_B)
        assert np.allclose(self.d_A, self.d_dimer)
        

        return self.dimer, self.monomer_A, self.monomer_B

    def build_orbitals(self) -> Any:
        """Build orbital intermediates needed for QED-SAPT0 components.
           Takes a tuple of monomer instances as input:
           (dimer, monomer_a, monomer_b)

           As an example, if you want to access the MO coefficients of monomer_a:
           C_monomer_a = monomers[1].C

           Occupied orbitals for monomer a
           Co_monomer_a = monomers[1].Co

           Virtual orbitals for monomer b
           Cv_monomer_b = monomers[2].Cv

        """
        # organize orbitals into a dictionary for convenient access in component functions, using the same string labels as the original SAPT0 implementation where possible (a, b, r, s)
        self.orbitals = {'a' : self.Co_A,
                         'r': self.Cv_A,
                         'b': self.Co_B,
                         's': self.Cv_B
                         }
        

        
    def build_slices(self) -> Any:
        """Build slice objects for occupied and virtual orbital subspaces of each monomer.
           Takes a tuple of monomer instances as input:
           (monomer_A, monomer_B)

           As an example, if you want to access the occupied slice for monomer_a:
           occ_slice_a = slice(0, monomers[0].ndocc)
        """
        self.slices = {'a' : slice(0, self.ndocc_A),
                       'r': slice(self.ndocc_A, None),
                       'b': slice(0, self.ndocc_B),
                       's': slice(self.ndocc_B, None)
                       }
        
    def build_sizes(self) -> Any:
        """Build integers for number of occupied and virtual orbitals of each monomer.
           Takes a tuple of monomer instances as input:
           (monomer_a, monomer_b)

           As an example, if you want to access the number of occupied orbitals for monomer_a:
           nocc_a = monomers[0].ndocc
        """
        self.sizes = {'a' : self.ndocc_A,
                      'r': self.nvirt_A,
                      'b': self.ndocc_A,
                      's': self.nvirt_A
                      }

        

    def build_integrals(self, monomers: Tuple[SAPTMonomer, SAPTMonomer, SAPTMonomer]) -> Any:
        """Build all integral intermediates needed for QED-SAPT0 components.
           Takes a tuple of monomer instances as input:
           (dimer, monomer_a, monomer_b)

           As an example, if you want to access the mints of the dimer:
           dimer_mints = monomers[0].mints

           mints of monomer_a: monomers[1].mints

        
        
        """
        # build orbitals using monomer A and monomer B SCF results, which may be None if the monomer SCF references were not run with orbital storage enabled
        self.build_orbitals()

        # build slices for occupied and virtual orbital subspaces of each monomer
        self.build_slices()

        # build sizes for number of occupied and virtual orbitals of each monomer
        self.build_sizes()

        dimer_mints = monomers[0].mints
        monomer_A_mints = monomers[1].mints
        monomer_B_mints = monomers[2].mints

        # overlap of dimer in AO basis
        self.S_dimer = np.asarray(dimer_mints.ao_overlap())

        # overlap transformed on bra with monomer A and ket with monomer B
        self.S_AB = oe.contract("uI,vJ,uv->IJ", self.C_A, self.C_B, self.S_dimer)

        # 1. Get the ERI array directly (try to avoid copying if ao_eri() allows)
        self.I_dimer = np.asarray(dimer_mints.ao_eri())

        # 2. Reshape self.d_A and self.d_B to broadcast into a 4D shape (pqrs)
        #    d_A (p, q) -> (p, q, 1, 1)
        #    d_B (r, s) -> (1, 1, r, s)
        # This adds the outer product directly to self.I_dimer in-place, swapping axes on the fly.
        self.I_dimer += self.d_A[:, :, np.newaxis, np.newaxis] * self.d_B[np.newaxis, np.newaxis, :, :]

        # 3. Swap axes in-place (creates a view, zero memory overhead)
        #    Note: If a contiguous array is strictly required by downstream code, 
        #    append .copy() here, but it will double the memory momentarily.
        self.I_dimer = self.I_dimer.swapaxes(1, 2)


        # build the one-electron potential integrals for monomer A and monomer B
        self.V_A = np.asarray(monomer_A_mints.ao_potential())
        self.V_A -= 0.5 * self.d_exp_B * self.d_A

        self.V_B = np.asarray(monomer_B_mints.ao_potential())
        self.V_B -= 0.5 * self.d_exp_A * self.d_B

        # potential integrals
        self.V_A_BB = oe.contract("uI,vJ,uv->IJ", self.C_B, self.C_B, self.V_A, optimize="optimal")
        self.V_A_AB = oe.contract("uI,vJ,uv->IJ", self.C_A, self.C_B, self.V_A, optimize="optimal")
        self.V_B_AA = oe.contract("uI,vJ,uv->IJ", self.C_A, self.C_A, self.V_B, optimize="optimal")
        self.V_B_AB = oe.contract("uI,vJ,uv->IJ", self.C_A, self.C_B, self.V_B, optimize="optimal")




    def compute_components(self, monomers, integrals) -> QEDSAPT0Results:
        """Call future component functions and collect a result object."""

        raise NotImplementedError(
            "QED-SAPT0 component physics is not implemented yet. "
            "Future code will call compute_elst10, compute_exch10, "
            "compute_ind20, compute_disp20, and compute_qed_dse_cross."
        )
    
    def v(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception("v: string %s does not have length 4" % string)
        
        # ERI's from mints are in chemist's notation (pq|rs), but we want to access them in physicist's notation (pr|qs)
        # so we need to swap the middle two indices
        V = oe.contract("pA,pqrs->Aqrs", self.orbitals[string[0]], self.I_dimer, optimize="optimal")
        V = oe.contract("qB,Aqrs->ABrs", self.orbitals[string[1]], V, optimize="optimal")
        V = oe.contract("rC,ABrs->ABCs", self.orbitals[string[2]], V, optimize="optimal")
        V = oe.contract("sD,ABCs->ABCD", self.orbitals[string[3]], V, optimize="optimal")
        return V
    
    def s(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception("s: string %s does not have length 2" % string)
        
        for alpha in 'ijab':
            if (alpha in string) and (self.sizes[alpha] == 0):
                return np.array([0]).reshape(1,1)
            
        s1 = string[0]
        s2 = string[1]

        # compute on the fly
        return (self.orbitals[s1].T).dot(self.S_dimer).dot(self.orbitals[s2])
    
    def eps(self, string, dim=1):
        if len(string) != 1:
            psi4.core.clean()
            raise Exception("eps: string %s does not have length 1" % string)
        
        shape = (-1,) + tuple([1] * (dim - 1))

        if (string=='i') or (string=='a') or (string=='r'):
            return self.eps_A[self.slices[string]].reshape(shape)
        
        elif (string=='j') or (string=='b') or (string=='s'):
            return self.eps_B[self.slices[string]].reshape(shape)
        
        else:
            psi4.core.clean()
            raise Exception("eps: string %s does not have valid monomer label" % string)
    

    def potential(self, string, side):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception("potential: string %s does not have length 2" % string)
        
        s1 = string[0]
        s2 = string[1]

        if side == 'A':
            return (self.orbitals[s1].T).dot(self.V_A).dot(self.orbitals[s2])
        
        elif side == 'B':
            return (self.orbitals[s1].T).dot(self.V_B).dot(self.orbitals[s2])
        
        else:
            psi4.core.clean()
            raise Exception("potential: side %s is not A or B" % side)
        

    def vt(self, string):
        if len(string)!=4:
            psi4.core.clean()
            raise Exception('Compute tilde{v}: string %s does not have 4 elements' % string)
        
        for alpha in 'ijab':
            if (alpha in string) and (self.sizes[alpha] == 0):
                return np.array([0]).reshape(1,1,1,1)
            
        # grab left and right strings
        s_left = string[0] + string[2]
        s_right = string[1] + string[3]

        # ERI term
        V = self.v(string)

        # potential A
        S_A = self.s(s_left)
        V_A = self.potential(s_right, 'A') / (2 * self.ndocc_A + self.nsocc_A)
        V += oe.contract("ik,jl->ijkl", S_A, V_A)

        # potential B
        S_B = self.s(s_right)
        V_B = self.potential(s_left, 'B') / (2 * self.ndocc_B + self.nsocc_B)
        V += np.einsum('ik,jl->ijkl', V_B, S_B)

        # nuclear
        V += np.einsum("ik,jl->ijkl", S_A, S_B) * self.vt_nuc_rep

        return V
        
    def Elst100(self):
        return 4 * oe.contract('abab', self.vt('abab'), optimize="optimal")
    
    def Exch100(self):
        vt_abba = self.vt('abba')
        vt_abaa = self.vt('abaa')
        vt_abbb = self.vt('abbb')
        vt_abab = self.vt('abab')
        s_ab = self.s('ab')

        Exch100 = oe.contract("abba", vt_abba, optimize="optimal")

        _tmp = 2 * vt_abaa - vt_abaa.swapaxes(2,3)
        Exch100 += oe.contract('Ab,abaA', s_ab, _tmp, optimize="optimal")

        _tmp = 2 * vt_abbb - vt_abbb.swapaxes(2,3)
        Exch100 += oe.contract('Ba,abBb', s_ab.T, _tmp, optimize="optimal")

        Exch100 -= 2 * oe.contract('Ab,BA,abaB', s_ab, s_ab.T, vt_abab, optimize="optimal")
        Exch100 -= 2 * oe.contract('AB,Ba,abAb', s_ab, s_ab.T, vt_abab, optimize="optimal")
        Exch100 += oe.contract('Ab,Ba, abAB', s_ab, s_ab.T, vt_abab, optimize="optimal")

        Exch100 *= -2

        return Exch100
    
    def Edisp200(self):

        v_abrs = self.v('abrs')
        v_rsab = self.v('rsab')
        eps_rsab = 1 / (self.eps('r', dim=4) - self.eps('s', dim=3) + self.eps('a', dim=2) + self.eps('b'))

        Disp200 = 4 * oe.contract('rsab,rsab,abrs->', eps_rsab, v_rsab, v_abrs, optimize="optimal")
        return Disp200
    

    def run(self) -> QEDSAPT0Results:
        """Run the future QED-SAPT0 workflow."""

        monomers = self.prepare_monomers()
        integrals = self.build_integrals(monomers)

        return self.v("arbs")
        #return self.compute_components(monomers, integrals)
