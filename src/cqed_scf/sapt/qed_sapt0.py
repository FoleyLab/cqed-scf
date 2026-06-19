"""Architecture scaffold for QED-SAPT0."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple
import warnings
import opt_einsum as oe
import numpy as np

from ..references import CQEDConfig

from .monomer import SAPTMonomer
from .results import QEDSAPT0Results


_VT_PART_KEYS = ("eri", "potential_A", "potential_B", "nuclear")
_OPERATOR_CONTEXTS = ("standard", "total", "cavity")


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
    monomer_A: Optional[SAPTMonomer] = None
    monomer_B: Optional[SAPTMonomer] = None
    monomer_definitions: Optional[Sequence[Any]] = None
    monomer_indices: Optional[Tuple[Sequence[int], Sequence[int]]] = None
    integral_backend: str = "full_eri"
    include_cavity_terms: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    monomer_a: InitVar[Optional[SAPTMonomer]] = None
    monomer_b: InitVar[Optional[SAPTMonomer]] = None
    dimer: InitVar[Optional[SAPTMonomer]] = None

    def __post_init__(
        self,
        monomer_a: Optional[SAPTMonomer],
        monomer_b: Optional[SAPTMonomer],
        dimer: Optional[SAPTMonomer],
    ) -> None:
        if monomer_a is not None:
            if self.monomer_A is not None:
                raise ValueError("Specify only one of monomer_A or monomer_a.")
            warnings.warn(
                "monomer_a is deprecated; use monomer_A.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.monomer_A = monomer_a
        if monomer_b is not None:
            if self.monomer_B is not None:
                raise ValueError("Specify only one of monomer_B or monomer_b.")
            warnings.warn(
                "monomer_b is deprecated; use monomer_B.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.monomer_B = monomer_b
        if dimer is not None:
            warnings.warn(
                "Passing dimer as a SAPTMonomer is deprecated and ignored; "
                "QEDSAPT0Driver now uses dimer_geometry for dimer nuclear and AO-basis data.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.integral_backend == "no_cavity":
            warnings.warn(
                'integral_backend="no_cavity" is deprecated; use '
                "include_cavity_terms=False with an ordinary integral backend.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.integral_backend = "full_eri"
            self.include_cavity_terms = False

        self.metadata.setdefault("integral_backend", self.integral_backend)
        self.metadata.setdefault("include_cavity_terms", self.include_cavity_terms)

    def prepare_geometries(self) -> Tuple[str, str, str]:
        """Build dimer and ghosted monomer geometry strings from a Psi4 dimer."""

        monomer_A_geometry = self.dimer_geometry.extract_subsets(1, 2)
        monomer_B_geometry = self.dimer_geometry.extract_subsets(2, 1)

        dimer_string = self.dimer_geometry.create_psi4_string_from_molecule()
        monomer_A_string = monomer_A_geometry.create_psi4_string_from_molecule()
        monomer_B_string = monomer_B_geometry.create_psi4_string_from_molecule()

        return dimer_string, monomer_A_string, monomer_B_string

    def prepare_monomers(self) -> Tuple[SAPTMonomer, SAPTMonomer]:
        """Prepare or retrieve monomer references."""

        if self.monomer_A is not None and self.monomer_B is not None:
            self._populate_dimer_nuclear_terms()
            self._populate_monomer_attributes()
            return self.monomer_A, self.monomer_B

        _, monomer_A_string, monomer_B_string = self.prepare_geometries()
        self._populate_dimer_nuclear_terms()

        if self.monomer_A is None:
            self.monomer_A = SAPTMonomer.from_cqed_scf(
                label="monomer_A",
                geometry=monomer_A_string,
                config=self.config,
            )
        if self.monomer_B is None:
            self.monomer_B = SAPTMonomer.from_cqed_scf(
                label="monomer_B",
                geometry=monomer_B_string,
                config=self.config,
            )

        self._populate_monomer_attributes()
        return self.monomer_A, self.monomer_B

    def _populate_dimer_nuclear_terms(self) -> None:
        """Populate dimer quantities that do not require a dimer SCF reference."""

        dimer_mu_nuc = self.dimer_geometry.nuclear_dipole()
        self.dimer_mu_nuc = np.array(
            [dimer_mu_nuc[0], dimer_mu_nuc[1], dimer_mu_nuc[2]]
        )
        self.E_nuc_dimer = self.dimer_geometry.nuclear_repulsion_energy()
        if self.config.debug:
            print("Dimer nuclear dipole moment")
            print(self.dimer_mu_nuc)
            print(f"Dimer nuclear repulsion energy {self.E_nuc_dimer}")

    def _populate_monomer_attributes(self) -> None:
        """Cache monomer reference data used by the SAPT component formulas."""

        if self.monomer_A is None or self.monomer_B is None:
            raise RuntimeError("prepare_monomers requires both monomer references.")

        self.E_scf_A = self.monomer_A.energy_scf
        self.E_scf_B = self.monomer_B.energy_scf

        self.ndocc_A = self.monomer_A.ndocc
        self.nvirt_A = self.monomer_A.nvirt
        self.ndocc_B = self.monomer_B.ndocc
        self.nvirt_B = self.monomer_B.nvirt

        # currently assumes closed shell, nsocc = 0
        self.nsocc_A = 0
        self.nsocc_B = 0

        self.C_A = self.monomer_A.C
        self.C_B = self.monomer_B.C
        self.Co_A = self.monomer_A.Co
        self.Cv_A = self.monomer_A.Cv
        self.Co_B = self.monomer_B.Co
        self.Cv_B = self.monomer_B.Cv

        self.eps_A = self.monomer_A.eps
        self.eps_B = self.monomer_B.eps

        self.E_nuc_A = self.monomer_A.nuc_rep
        self.E_nuc_B = self.monomer_B.nuc_rep
        self.nuc_rep = self.E_nuc_dimer - self.E_nuc_A - self.E_nuc_B

        self.d_exp_el_A = self.monomer_A.d_exp_el
        self.d_exp_el_B = self.monomer_B.d_exp_el
        self.d_exp_A = self.monomer_A.d_exp
        self.d_exp_B = self.monomer_B.d_exp
        self.d_nuc_A = self.monomer_A.d_nuc
        self.d_nuc_B = self.monomer_B.d_nuc

        # there is a term <d_A> * <d_B>
        self.sum_d_exp_A_d_exp_B = +(self.d_exp_A * self.d_exp_B)

        assert np.isclose(self.d_exp_A, (self.d_exp_el_A + self.d_nuc_A))
        assert np.isclose(self.d_exp_B, (self.d_exp_el_B + self.d_nuc_B))

        self.d_A = self.monomer_A.d_ao
        self.d_B = self.monomer_B.d_ao

        electron_count_A = 2 * self.ndocc_A + self.nsocc_A
        electron_count_B = 2 * self.ndocc_B + self.nsocc_B
        self.vt_nuc_rep_standard = self.nuc_rep / (electron_count_A * electron_count_B)
        self.vt_nuc_rep_cavity = (
            (self.d_exp_A * self.d_exp_B) / (electron_count_A * electron_count_B)
            if self.include_cavity_terms
            else 0.0
        )
        self.vt_nuc_rep = self.vt_nuc_rep_standard + self.vt_nuc_rep_cavity

    def build_orbitals(self) -> Any:
        """Build orbital intermediates needed for QED-SAPT0 components.
        """
        # organize orbitals into a dictionary for convenient access in component functions, using the same string labels as the original SAPT0 implementation where possible (a, b, r, s)
        self.orbitals = {'a' : self.Co_A,
                         'r': self.Cv_A,
                         'b': self.Co_B,
                         's': self.Cv_B
                         }
        

        
    def build_slices(self) -> Any:
        """Build slice objects for occupied and virtual orbital subspaces of each monomer.
        """
        self.slices = {'a' : slice(0, self.ndocc_A),
                       'r': slice(self.ndocc_A, None),
                       'b': slice(0, self.ndocc_B),
                       's': slice(self.ndocc_B, None)
                       }
        
    def build_sizes(self) -> Any:
        """Build integers for number of occupied and virtual orbitals of each monomer.
        """
        self.sizes = {'a' : self.ndocc_A,
                      'r': self.nvirt_A,
                      'b': self.ndocc_B,
                      's': self.nvirt_B
                      }

        

    def build_integrals(self, monomers: Optional[Tuple[SAPTMonomer, SAPTMonomer]] = None) -> Any:
        """Build all integral intermediates needed for QED-SAPT0 components."""
        if monomers is not None:
            warnings.warn(
                "Passing monomers to build_integrals is deprecated; the driver "
                "uses its prepared monomer_A and monomer_B references.",
                DeprecationWarning,
                stacklevel=2,
            )
            if len(monomers) != 2:
                raise ValueError("build_integrals expects only (monomer_A, monomer_B).")
            self.monomer_A, self.monomer_B = monomers
            self._populate_dimer_nuclear_terms()
            self._populate_monomer_attributes()

        if self.monomer_A is None or self.monomer_B is None:
            self.prepare_monomers()

        # build orbitals using monomer A and monomer B SCF results, which may be None if the monomer SCF references were not run with orbital storage enabled
        self.build_orbitals()

        # build slices for occupied and virtual orbital subspaces of each monomer
        self.build_slices()

        # build sizes for number of occupied and virtual orbitals of each monomer
        self.build_sizes()

        # Monomer A and B are ghosted calculations in the dimer basis, so either
        # MintsHelper can define the shared AO integral environment.
        shared_mints = self.monomer_A.mints
        monomer_A_mints = self.monomer_A.mints
        monomer_B_mints = self.monomer_B.mints

        # overlap of dimer in AO basis
        self.S_dimer = np.asarray(shared_mints.ao_overlap())

        # overlap transformed on bra with monomer A and ket with monomer B
        self.S_AB = oe.contract("uI,vJ,uv->IJ", self.C_A, self.C_B, self.S_dimer)

        # 1. Get the ERI array directly (try to avoid copying if ao_eri() allows)
        I_dimer_standard = np.asarray(shared_mints.ao_eri())
        I_dimer = I_dimer_standard.copy()

        # 2. Reshape self.d_A and self.d_B to broadcast into a 4D shape (pqrs)
        #    d_A (p, q) -> (p, q, 1, 1)
        #    d_B (r, s) -> (1, 1, r, s)
        # This adds the outer product directly to self.I_dimer in-place, swapping axes on the fly.
        I_dimer_cavity = np.zeros_like(I_dimer)
        if self.include_cavity_terms:
            I_dimer_cavity = self.d_A[:, :, np.newaxis, np.newaxis] * self.d_B[np.newaxis, np.newaxis, :, :]
            I_dimer += I_dimer_cavity

        # 3. Swap axes in-place (creates a view, zero memory overhead)
        #    Note: If a contiguous array is strictly required by downstream code, 
        #    append .copy() here, but it will double the memory momentarily.
        self.I_dimer_standard = I_dimer_standard.swapaxes(1, 2)
        self.I_dimer_cavity = I_dimer_cavity.swapaxes(1, 2)
        self.I_dimer = I_dimer.swapaxes(1, 2)


        # build the one-electron potential integrals for monomer A and monomer B
        # the V_A and V_B terms are scaled by 1 / N_A and 1 / N_B in the v_tilde build
        self.V_A = np.asarray(monomer_A_mints.ao_potential())
        self.V_B = np.asarray(monomer_B_mints.ao_potential())

        self.V_A_standard = self.V_A.copy()
        self.V_B_standard = self.V_B.copy()
        self.V_A_cavity = np.zeros_like(self.V_A)
        self.V_B_cavity = np.zeros_like(self.V_B)
        if self.include_cavity_terms: 
            self.V_A_cavity = -self.d_exp_B * self.d_A
            self.V_B_cavity = -self.d_exp_A * self.d_B
            self.V_A += self.V_A_cavity
            self.V_B += self.V_B_cavity

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
    
    def _validate_operator_context(self, context: str) -> str:
        if context not in _OPERATOR_CONTEXTS:
            allowed = ", ".join(_OPERATOR_CONTEXTS)
            raise ValueError(f"operator context must be one of {allowed}; got {context!r}")
        return context

    def _eri_for_context(self, context: str):
        context = self._validate_operator_context(context)
        if context == "standard":
            return self.I_dimer_standard
        if context == "cavity":
            return self.I_dimer_cavity
        return self.I_dimer

    def _potential_for_context(self, side: str, context: str):
        context = self._validate_operator_context(context)
        if side == "A":
            if context == "standard":
                return self.V_A_standard
            if context == "cavity":
                return self.V_A_cavity
            return self.V_A
        if side == "B":
            if context == "standard":
                return self.V_B_standard
            if context == "cavity":
                return self.V_B_cavity
            return self.V_B

        psi4.core.clean()
        raise Exception("potential: side %s is not A or B" % side)

    def _vt_nuc_rep_for_context(self, context: str):
        context = self._validate_operator_context(context)
        if context == "standard":
            return self.vt_nuc_rep_standard
        if context == "cavity":
            return self.vt_nuc_rep_cavity
        return self.vt_nuc_rep

    def _zero_vt_parts(self):
        zero = np.array([0]).reshape(1, 1, 1, 1)
        return {key: zero.copy() for key in _VT_PART_KEYS}

    def _sum_vt_parts(self, parts):
        total = parts["eri"].copy()
        total += parts["potential_A"]
        total += parts["potential_B"]
        total += parts["nuclear"]
        return total

    def v(self, string, context: str = "total"):
        """
        Builds two-electron integrals dressed with monomerA - monomerB dipole integrals
        transformed with appropriate MO coefficients
        """
        if len(string) != 4:
            psi4.core.clean()
            raise Exception("v: string %s does not have length 4" % string)
        I_dimer = self._eri_for_context(context)
        
        # ERI's from mints are in chemist's notation (pq|rs), but we want to access them in physicist's notation (pr|qs)
        # so we need to swap the middle two indices
        V = oe.contract("pA,pqrs->Aqrs", self.orbitals[string[0]], I_dimer, optimize="optimal")
        V = oe.contract("qB,Aqrs->ABrs", self.orbitals[string[1]], V, optimize="optimal")
        V = oe.contract("rC,ABrs->ABCs", self.orbitals[string[2]], V, optimize="optimal")
        V = oe.contract("sD,ABCs->ABCD", self.orbitals[string[3]], V, optimize="optimal")
        return V
    
    def s(self, string):
        # Grap appropriate overlap integrals 
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
    

    def potential(self, string, side, context: str = "total"):
        """
        Grab one-electron potential integrals for monomer X dressed with dipole integrals for monomer X scaled by expectation value of
        <d_Y>
        """
        if len(string) != 2:
            psi4.core.clean()
            raise Exception("potential: string %s does not have length 2" % string)
        
        s1 = string[0]
        s2 = string[1]
        potential = self._potential_for_context(side, context)

        return (self.orbitals[s1].T).dot(potential).dot(self.orbitals[s2])
        

    def vt_parts(self, string, context: str = "total"):
        if len(string)!=4:
            psi4.core.clean()
            raise Exception('Compute tilde{v}: string %s does not have 4 elements' % string)
        
        for alpha in 'ijab':
            if (alpha in string) and (self.sizes[alpha] == 0):
                return self._zero_vt_parts()
            
        # grab left and right strings
        s_left = string[0] + string[2]
        s_right = string[1] + string[3]

        # ERI term
        eri = self.v(string, context=context)

        # potential A
        S_A = self.s(s_left)
        V_A = self.potential(s_right, 'A', context=context) / (2 * self.ndocc_A + self.nsocc_A)
        potential_A = oe.contract("ik,jl->ijkl", S_A, V_A)

        # potential B
        S_B = self.s(s_right)
        V_B = self.potential(s_left, 'B', context=context) / (2 * self.ndocc_B + self.nsocc_B)
        potential_B = np.einsum('ik,jl->ijkl', V_B, S_B)

        # nuclear - scaling by 1/N_A and 1/N_B already happened in prepare_monomers
        nuclear = np.einsum("ik,jl->ijkl", S_A, S_B) * self._vt_nuc_rep_for_context(context)

        return {
            "eri": eri,
            "potential_A": potential_A,
            "potential_B": potential_B,
            "nuclear": nuclear,
        }

    def vt(self, string, context: str = "total"):
        return self._sum_vt_parts(self.vt_parts(string, context=context))

    def vt_partitions(self, string):
        standard = self.vt_parts(string, context="standard")
        total = self.vt_parts(string, context="total")
        cavity = {key: total[key] - standard[key] for key in _VT_PART_KEYS}

        partitions = {
            "standard": dict(standard),
            "total": dict(total),
            "cavity": cavity,
        }
        for context in _OPERATOR_CONTEXTS:
            partitions[context]["total"] = self._sum_vt_parts(partitions[context])

        return partitions

    def _contract_vt_array(self, string, array, contraction):
        if callable(contraction):
            return float(contraction(array))
        if contraction is None or contraction in {"einsum", "sapt"}:
            return float(np.einsum(f"{string}->", array))
        if contraction == "sum":
            return float(np.sum(array))
        return float(np.einsum(contraction, array))

    def contract_vt_parts(self, string, contraction=None, prefactor: float = 1.0):
        partitions = self.vt_partitions(string)
        return {
            context: {
                key: prefactor * self._contract_vt_array(string, value, contraction)
                for key, value in parts.items()
            }
            for context, parts in partitions.items()
        }

    def diagnostic_summary(self, print_output: Optional[bool] = None):
        print_output = self.config.debug if print_output is None else print_output
        summary = {
            "Elst100": self.contract_vt_parts("abab", prefactor=4.0),
        }

        if print_output:
            print("QED-SAPT0 operator diagnostics")
            print("Component: Elst100")
            print(f"{'context':<10} {'piece':<14} {'value / Eh':>18}")
            print("-" * 44)
            for context in ("standard", "cavity", "total"):
                for piece in ("eri", "potential_A", "potential_B", "nuclear", "total"):
                    print(f"{context:<10} {piece:<14} {summary['Elst100'][context][piece]:18.10f}")

        return summary

    def print_diagnostics(self):
        return self.diagnostic_summary(print_output=True)
    
    def chf(self, monomer, ind=False):
        if monomer not in ['A', 'B']:
            psi4.core.clean()
            raise Exception("chf: monomer %s is not A or B" % monomer)
        
        if monomer == 'A':
            w_n = 2 * oe.contract('saba->bs', self.v('saba'), optimize="optimal")
            w_n += self.V_A_BB[self.slices['b'], self.slices['s']]
            eps_ov = (self.eps('b', dim=2) - self.eps('s'))

            # set terms
            v_term1 = 'sbbs'
            v_term2 = 'sbsb'
            no, nv = self.ndocc_B, self.nvirt_B

        if monomer == 'B':
            w_n = 2 * oe.contract('rbab->ar', self.v('rbab'), optimize="optimal")
            w_n += self.V_B_AA[self.slices['a'], self.slices['r']]
            eps_ov = (self.eps('a', dim=2) - self.eps('r'))
            v_term1 = 'raar'
            v_term2 = 'rara'
            no, nv = self.ndocc_A, self.nvirt_A

        # form A matrix (LHS)
        voov = self.v(v_term1)
        v_vOov = 2 * voov - self.v(v_term2).swapaxes(2,3)
        v_ooaa = voov.swapaxes(1,3)
        v_vVoO = 2 * v_ooaa- v_ooaa.swapaxes(2,3)
        # A_ovOV = np.einsum('vOoV->ovOV', v_vOoV + v_vVoO.swapaxes(1, 3))
        #A_ovOV = oe.contract('vOov->ovOV', v_vOov + v_vVoO.swapaxes(1,3), optimize="optimal")
        A_ovOV = oe.contract('vOoV->ovOV',  v_vOov + v_vVoO.swapaxes(1, 3),optimize="optimal")
        # copy back to C contibous
        nov = nv * no 
        A_ovOV = A_ovOV.reshape(nov, nov).copy(order='C')
        A_ovOV[np.diag_indices_from(A_ovOV)] -= eps_ov.ravel()

        # call DGESV, need flat ov array 
        B_ov = -1 * w_n.ravel()
        t = np.linalg.solve(A_ovOV, B_ov)
        t = t.reshape(no, nv).T

        if ind:
            e20_ind = 2 * oe.contract('vo,ov->', t, w_n, optimize="optimal")
            return t, e20_ind
        
        else:
            return t

    def compute_Elst100(self):
        return 4 * oe.contract('abab->', self.vt('abab'), optimize="optimal")
    
    def compute_Exch100(self):
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
    
    def compute_Edisp200(self):

        v_abrs = self.v('abrs')
        self.v_rsab = self.v('rsab')
        self.eps_rsab = 1 / (-self.eps('r', dim=4) - self.eps('s', dim=3) + self.eps('a', dim=2) + self.eps('b'))
        self.t_rsab = oe.contract("rsab,rsab->rsab", self.v_rsab, self.eps_rsab, optimize="optimal")
        Disp200 = 4 * oe.contract('rsab,abrs->', self.t_rsab, v_abrs, optimize="optimal")
        return Disp200
    
    def compute_Eexchdisp200(self):

        vt_abar = self.vt('abar')
        vt_abra = self.vt('abra')
        vt_absb = self.vt('absb')
        vt_abbs = self.vt('abbs')

        _tmp = 2 * vt_abar - vt_abra.swapaxes(2,3)
        h_abrs = oe.contract('as,AbAr->abrs', self.s('as'), _tmp, optimize="optimal")

        _tmp = 2 * vt_abra - vt_abar.swapaxes(2,3)
        h_abrs += oe.contract('As,abrA->abrs', self.s('as'), _tmp, optimize="optimal")

        _tmp = 2 * vt_absb - vt_abbs.swapaxes(2,3)
        h_abrs += oe.contract('br,aBsB->abrs', self.s('br'), _tmp, optimize="optimal")

        _tmp = 2 * vt_abbs - vt_absb.swapaxes(2,3)
        h_abrs += oe.contract('Br,abBs->abrs', self.s('br'), _tmp, optimize="optimal")

        # build q_abrs
        vt_abas = self.vt('abas')
        # q_abrs =      np.einsum('br,AB,aBAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas, optimize=True)
        q_abrs = oe.contract('br,AB,aBAs->abrs', self.s('br'), self.s('ab'), vt_abas, optimize="optimal")
        # q_abrs -= 2 * np.einsum('Br,AB,abAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas, optimize=True)
        q_abrs -= 2 * oe.contract('Br,AB,abAs->abrs', self.s('br'), self.s('ab'), vt_abas, optimize="optimal")
        # q_abrs -= 2 * np.einsum('br,aB,ABAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas, optimize=True)
        q_abrs -= 2 * oe.contract('br,aB,ABAs->abrs', self.s('br'), self.s('ab'), vt_abas, optimize="optimal")
        # q_abrs += 4 * np.einsum('Br,aB,AbAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas, optimize=True)
        q_abrs += 4 * oe.contract('Br,aB,AbAs->abrs', self.s('br'), self.s('ab'), vt_abas, optimize="optimal")

        vt_abrb = self.vt('abrb')
        #q_abrs -= 2 * np.einsum('as,bA,ABrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb, optimize=True)
        q_abrs -= 2 * oe.contract('as,bA,ABrB->abrs', self.s('as'), self.s('ba'), vt_abrb, optimize="optimal")
        #q_abrs += 4 * np.einsum('As,bA,aBrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb, optimize=True)
        q_abrs += 4 * oe.contract('As,bA,aBrB->abrs', self.s('as'), self.s('ba'), vt_abrb, optimize="optimal")
        #q_abrs +=     np.einsum('as,BA,AbrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb, optimize=True)
        q_abrs +=    oe.contract('as,BA,AbrB->abrs', self.s('as'), self.s('ba'), vt_abrb, optimize="optimal")
        #q_abrs -= 2 * np.einsum('As,BA,abrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb, optimize=True)
        q_abrs -= 2 * oe.contract('As,BA,abrB->abrs', self.s('as'), self.s('ba'), vt_abrb, optimize="optimal")

        vt_abab = self.vt('abab')
        #q_abrs +=     np.einsum('Br,As,abAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab, optimize=True)
        q_abrs +=     oe.contract('Br,As,abAB->abrs', self.s('br'), self.s('as'), vt_abab, optimize="optimal")
        #q_abrs -= 2 * np.einsum('br,As,aBAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab, optimize=True)
        q_abrs -= 2 * oe.contract('br,As,aBAB->abrs', self.s('br'), self.s('as'), vt_abab, optimize="optimal")
        #q_abrs -= 2 * np.einsum('Br,as,AbAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab, optimize=True)
        q_abrs -= 2 * oe.contract('Br,as,AbAB->abrs', self.s('br'), self.s('as'), vt_abab, optimize="optimal")

        vt_abrs = self.vt('abrs')
        #q_abrs +=     np.einsum('bA,aB,ABrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs, optimize=True)
        q_abrs +=     oe.contract('bA,aB,ABrs->abrs', self.s('ba'), self.s('ab'), vt_abrs, optimize="optimal")
        #q_abrs -= 2 * np.einsum('bA,AB,aBrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs, optimize=True)
        q_abrs -= 2 * oe.contract('bA,AB,aBrs->abrs', self.s('ba'), self.s('ab'), vt_abrs, optimize="optimal")
        #q_abrs -= 2 * np.einsum('BA,aB,Abrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs, optimize=True)
        q_abrs -= 2 * oe.contract('BA,aB,Abrs->abrs', self.s('ba'), self.s('ab'), vt_abrs, optimize="optimal")

        # sum all terms and contract with t_rsab
        xd_absr = self.vt('absr') + h_abrs.swapaxes(2,3) + q_abrs.swapaxes(2,3)
        Eexchdisp200 = -2 * oe.contract('absr,rsab->', xd_absr, self.t_rsab, optimize="optimal")
        return Eexchdisp200
    
    def compute_Eind200(self):
        self.CPHF_ra, Ind200_ba = self.chf('B', ind=True)
        self.CPHF_sb, Ind200_ab = self.chf('A', ind=True)

        return Ind200_ba + Ind200_ab
    
    def compute_Eexchind200(self):
        # A <- B
        vt_abra = self.vt('abra')
        vt_abar = self.vt('abar')

        #ExchInd20_ab  =     np.einsum('ra,abbr', CPHF_ra, sapt.vt('abbr'), optimize=True)
        ExchInd20_ab = oe.contract('ra,abbr', self.CPHF_ra, self.vt('abbr'), optimize="optimal")
        #ExchInd20_ab += 2 * np.einsum('rA,Ab,abar', CPHF_ra, sapt.s('ab'), vt_abar, optimize=True)
        ExchInd20_ab += 2 * oe.contract('rA,Ab,abar', self.CPHF_ra, self.s('ab'), vt_abar, optimize="optimal")
        #ExchInd20_ab += 2 * np.einsum('ra,Ab,abrA', CPHF_ra, sapt.s('ab'), vt_abra, optimize=True)
        ExchInd20_ab += 2 * oe.contract('ra,Ab,abrA', self.CPHF_ra, self.s('ab'), vt_abra, optimize="optimal")
        #ExchInd20_ab -=     np.einsum('rA,Ab,abra', CPHF_ra, sapt.s('ab'), vt_abra, optimize=True)
        ExchInd20_ab -=     oe.contract('rA,Ab,abra', self.CPHF_ra, self.s('ab'), vt_abra, optimize="optimal")

        vt_abbb = self.vt('abbb')
        vt_abab = self.vt('abab')
        #ExchInd20_ab -=     np.einsum('ra,Ab,abAr', CPHF_ra, sapt.s('ab'), vt_abar, optimize=True)
        ExchInd20_ab -=     oe.contract('ra,Ab,abAr', self.CPHF_ra, self.s('ab'), vt_abar, optimize="optimal")
        #ExchInd20_ab += 2 * np.einsum('ra,Br,abBb', CPHF_ra, sapt.s('br'), vt_abbb, optimize=True)
        ExchInd20_ab += 2 * oe.contract('ra,Br,abBb', self.CPHF_ra, self.s('br'), vt_abbb, optimize="optimal")
        #ExchInd20_ab -=     np.einsum('ra,Br,abbB', CPHF_ra, sapt.s('br'), vt_abbb, optimize=True)
        ExchInd20_ab -=     oe.contract('ra,Br,abbB', self.CPHF_ra, self.s('br'), vt_abbb, optimize="optimal")
        #ExchInd20_ab -= 2 * np.einsum('rA,Ab,Br,abaB', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab, optimize=True)
        ExchInd20_ab -= 2 * oe.contract('rA,Ab,Br,abaB', self.CPHF_ra, self.s('ab'), self.s('br'), vt_abab, optimize="optimal")

        vt_abrb = self.vt('abrb')
        #ExchInd20_ab -= 2 * np.einsum('ra,Ab,BA,abrB', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb, optimize=True)
        ExchInd20_ab -= 2 * oe.contract('ra,Ab,BA,abrB', self.CPHF_ra, self.s('ab'), self.s('ba'), vt_abrb, optimize="optimal")
        #ExchInd20_ab -= 2 * np.einsum('ra,AB,Br,abAb', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab, optimize=True)
        ExchInd20_ab -= 2 * oe.contract('ra,AB,Br,abAb', self.CPHF_ra, self.s('ab'), self.s('br'), vt_abab, optimize="optimal")
        #ExchInd20_ab -= 2 * np.einsum('rA,AB,Ba,abrb', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb, optimize=True)
        ExchInd20_ab -= 2 * oe.contract('rA,AB,Ba,abrb', self.CPHF_ra, self.s('ab'), self.s('ba'), vt_abrb, optimize="optimal")

        #ExchInd20_ab +=     np.einsum('ra,Ab,Br,abAB', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab, optimize=True)
        ExchInd20_ab +=     oe.contract('ra,Ab,Br,abAB', self.CPHF_ra, self.s('ab'), self.s('br'), vt_abab, optimize="optimal")
        #ExchInd20_ab +=     np.einsum('rA,Ab,Ba,abrB', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb, optimize=True)
        ExchInd20_ab +=     oe.contract('rA,Ab,Ba,abrB', self.CPHF_ra, self.s('ab'), self.s('ba'), vt_abrb, optimize="optimal")

        ExchInd20_ab *= -2

        # B <- A
        vt_abbs = self.vt('abbs')
        vt_absb = self.vt('absb')
        #ExchInd20_ba  =     np.einsum('sb,absa', CPHF_sb, sapt.vt('absa'), optimize=True)
        ExchInd20_ba  =     oe.contract('sb,absa', self.CPHF_sb, self.vt('absa'), optimize="optimal")
        #ExchInd20_ba += 2 * np.einsum('sB,Ba,absb', CPHF_sb, sapt.s('ba'), vt_absb, optimize=True)
        ExchInd20_ba += 2 * oe.contract('sB,Ba,absb', self.CPHF_sb, self.s('ba'), vt_absb, optimize="optimal")
        #ExchInd20_ba += 2 * np.einsum('sb,Ba,abBs', CPHF_sb, sapt.s('ba'), vt_abbs, optimize=True)
        ExchInd20_ba += 2 * oe.contract('sb,Ba,abBs', self.CPHF_sb, self.s('ba'), vt_abbs, optimize="optimal")
        #ExchInd20_ba -=     np.einsum('sB,Ba,abbs', CPHF_sb, sapt.s('ba'), vt_abbs, optimize=True)
        ExchInd20_ba -=     oe.contract('sB,Ba,abbs', self.CPHF_sb, self.s('ba'), vt_abbs, optimize="optimal")

        #vt_abaa = sapt.vt('abaa')
        #vt_abab = sapt.vt('abab')
        vt_abaa = self.vt('abaa')
        vt_abab = self.vt('abab')

        #ExchInd20_ba -=     np.einsum('sb,Ba,absB', CPHF_sb, sapt.s('ba'), vt_absb, optimize=True)
        ExchInd20_ba -=     oe.contract('sb,Ba,absB', self.CPHF_sb, self.s('ba'), vt_absb, optimize="optimal")
        #ExchInd20_ba += 2 * np.einsum('sb,As,abaA', CPHF_sb, sapt.s('as'), vt_abaa, optimize=True)
        ExchInd20_ba += 2 * oe.contract('sb,As,abaA', self.CPHF_sb, self.s('as'), vt_abaa, optimize="optimal")
        #ExchInd20_ba -=     np.einsum('sb,As,abAa', CPHF_sb, sapt.s('as'), vt_abaa, optimize=True)
        ExchInd20_ba -=     oe.contract('sb,As,abAa', self.CPHF_sb, self.s('as'), vt_abaa, optimize="optimal")
        #ExchInd20_ba -= 2 * np.einsum('sB,Ba,As,abAb', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab, optimize=True)
        ExchInd20_ba -= 2 * oe.contract('sB,Ba,As,abAb', self.CPHF_sb, self.s('ba'), self.s('as'), vt_abab, optimize="optimal")

        #vt_abas = sapt.vt('abas')
        vt_abas = self.vt('abas')
        #ExchInd20_ba -= 2 * np.einsum('sb,Ba,AB,abAs', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas, optimize=True)
        ExchInd20_ba -= 2 * oe.contract('sb,Ba,AB,abAs', self.CPHF_sb, self.s('ba'), self.s('ab'), vt_abas, optimize="optimal")
        #ExchInd20_ba -= 2 * np.einsum('sb,BA,As,abaB', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab, optimize=True)
        ExchInd20_ba -= 2 * oe.contract('sb,BA,As,abaB', self.CPHF_sb, self.s('ba'), self.s('as'), vt_abab, optimize="optimal")
        #ExchInd20_ba -= 2 * np.einsum('sB,BA,Ab,abas', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas, optimize=True)
        ExchInd20_ba -= 2 * oe.contract('sB,BA,Ab,abas', self.CPHF_sb, self.s('ba'), self.s('ab'), vt_abas, optimize="optimal")

        #ExchInd20_ba +=     np.einsum('sb,Ba,As,abAB', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab, optimize=True)
        ExchInd20_ba +=     oe.contract('sb,Ba,As,abAB', self.CPHF_sb, self.s('ba'), self.s('as'), vt_abab, optimize="optimal")
        #ExchInd20_ba +=     np.einsum('sB,Ba,Ab,abAs', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas, optimize=True)
        ExchInd20_ba +=     oe.contract('sB,Ba,Ab,abAs', self.CPHF_sb, self.s('ba'), self.s('ab'), vt_abas, optimize="optimal")

        ExchInd20_ba *= -2
        ExchInd200 = ExchInd20_ab + ExchInd20_ba
        return ExchInd200






    def run(self) -> QEDSAPT0Results:
        """Run the future QED-SAPT0 workflow."""

        monomers = self.prepare_monomers()
        integrals = self.build_integrals()

        self.Eelst100 = self.compute_Elst100()
        self.Eexch100 = self.compute_Exch100()
        self.Edisp200 = self.compute_Edisp200()
        self.Eexchdisp200 = self.compute_Eexchdisp200()
        self.Eind200 = self.compute_Eind200()
        self.Eexchind200 = self.compute_Eexchind200()
        
        self.E_SAPT0 = self.Eelst100 + self.Eexch100 + self.Edisp200 + self.Eexchdisp200 + self.Eind200 + self.Eexchind200
        return self.E_SAPT0
