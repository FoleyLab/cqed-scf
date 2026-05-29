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

        monomer_a_geometry = self.dimer_geometry.extract_subsets(1, 2)
        monomer_b_geometry = self.dimer_geometry.extract_subsets(2, 1)

        dimer_string = self.dimer_geometry.create_psi4_string_from_molecule()
        monomer_a_string = monomer_a_geometry.create_psi4_string_from_molecule()
        monomer_b_string = monomer_b_geometry.create_psi4_string_from_molecule()

        return dimer_string, monomer_a_string, monomer_b_string

    def prepare_monomers(self) -> Tuple[SAPTMonomer, SAPTMonomer, SAPTMonomer]:
        """Prepare or retrieve monomer references."""

        if self.dimer is not None and self.monomer_a is not None and self.monomer_b is not None:
            return self.dimer, self.monomer_a, self.monomer_b

        dimer_string, monomer_a_string, monomer_b_string = self.prepare_geometries()

        self.dimer = SAPTMonomer.from_cqed_scf(
            label="dimer",
            geometry=dimer_string,
            config=self.config,
        )
        self.monomer_a = SAPTMonomer.from_cqed_scf(
            label="monomer_a",
            geometry=monomer_a_string,
            config=self.config,
        )
        self.monomer_b = SAPTMonomer.from_cqed_scf(
            label="monomer_b",
            geometry=monomer_b_string,
            config=self.config,
        )

        return self.dimer, self.monomer_a, self.monomer_b
    
    def build_orbitals(self, monomers: Tuple[SAPTMonomer, SAPTMonomer, SAPTMonomer]) -> Any:
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
        self.orbitals = {'a' : monomers[1].Co,
                         'r': monomers[1].Cv,
                         'b': monomers[2].Co,
                         's': monomers[2].Cv
                         }
        
    def build_slices(self, monomers: Tuple[SAPTMonomer, SAPTMonomer, SAPTMonomer]) -> Any:
        """Build slice objects for occupied and virtual orbital subspaces of each monomer.
           Takes a tuple of monomer instances as input:
           (monomer_a, monomer_b)

           As an example, if you want to access the occupied slice for monomer_a:
           occ_slice_a = slice(0, monomers[0].ndocc)
        """
        self.slices = {'a' : slice(0, monomers[1].ndocc),
                       'r': slice(monomers[1].ndocc, None),
                       'b': slice(0, monomers[2].ndocc),
                       's': slice(monomers[2].ndocc, None)
                       }
        
    def build_sizes(self, monomers: Tuple[SAPTMonomer, SAPTMonomer]) -> Any:
        """Build integers for number of occupied and virtual orbitals of each monomer.
           Takes a tuple of monomer instances as input:
           (monomer_a, monomer_b)

           As an example, if you want to access the number of occupied orbitals for monomer_a:
           nocc_a = monomers[0].ndocc
        """
        self.sizes = {'a' : monomers[1].ndocc,
                      'r': monomers[1].nvirt,
                      'b': monomers[2].ndocc,
                      's': monomers[2].nvirt
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
        self.build_orbitals(monomers)

        # build slices for occupied and virtual orbital subspaces of each monomer
        self.build_slices(monomers)

        # build sizes for number of occupied and virtual orbitals of each monomer
        self.build_sizes(monomers)

        dimer_mints = monomers[0].mints
        # build the full two-electron integral tensor in physicist's notation (pr|qs) for the dimer
        self.I_dimer = np.asarray(dimer_mints.ao_eri()).swapaxes(1, 2)
        # self.v(dimer_mints, "pqrs")  # example of how to access dimer ERIs in physicist's notation

        # get stored integrals from both monomers scf results, which may be None
        # if the monomer SCF references were not run with integral storage enabled


        #raise NotImplementedError(
        #    "QED-SAPT0 integral construction is not implemented yet. "
        #    "The first backend should build full two-electron integrals."
        #)

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
        V = oe.contract("pA,pqrs->Aqrs", self.orbitals[string[0]], self.I_dimer)
        V = oe.contract("qB,Aqrs->ABrs", self.orbitals[string[1]], V)
        V = oe.contract("rC,ABrs->ABCs", self.orbitals[string[2]], V)
        V = oe.contract("sD,ABCs->ABCD", self.orbitals[string[3]], V)
        return V

    def run(self) -> QEDSAPT0Results:
        """Run the future QED-SAPT0 workflow."""

        monomers = self.prepare_monomers()
        integrals = self.build_integrals(monomers)
        return self.v("arbs")
        #return self.compute_components(monomers, integrals)
