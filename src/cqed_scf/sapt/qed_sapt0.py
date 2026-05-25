"""Architecture scaffold for QED-SAPT0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

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
    monomer_a: Optional[SAPTMonomer] = None
    monomer_b: Optional[SAPTMonomer] = None
    monomer_definitions: Optional[Sequence[Any]] = None
    monomer_indices: Optional[Tuple[Sequence[int], Sequence[int]]] = None
    integral_backend: str = "full_eri"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault("integral_backend", self.integral_backend)

    def prepare_monomers(self) -> Tuple[SAPTMonomer, SAPTMonomer]:
        """Prepare or retrieve monomer references."""

        if self.monomer_a is not None and self.monomer_b is not None:
            return self.monomer_a, self.monomer_b

        raise NotImplementedError(
            "Automatic monomer extraction from dimer geometries is not implemented yet. "
            "Pass monomer_a and monomer_b SAPTMonomer objects, or implement geometry "
            "partitioning from monomer_definitions/monomer_indices."
        )

    def build_integrals(self, monomers: Tuple[SAPTMonomer, SAPTMonomer]):
        """Build future full-ERI SAPT integral intermediates."""

        raise NotImplementedError(
            "QED-SAPT0 integral construction is not implemented yet. "
            "The first backend should build full two-electron integrals."
        )

    def compute_components(self, monomers, integrals) -> QEDSAPT0Results:
        """Call future component functions and collect a result object."""

        raise NotImplementedError(
            "QED-SAPT0 component physics is not implemented yet. "
            "Future code will call compute_elst10, compute_exch10, "
            "compute_ind20, compute_disp20, and compute_qed_dse_cross."
        )

    def run(self) -> QEDSAPT0Results:
        """Run the future QED-SAPT0 workflow."""

        monomers = self.prepare_monomers()
        integrals = self.build_integrals(monomers)
        return self.compute_components(monomers, integrals)
