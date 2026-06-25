"""Result containers for future QED-SAPT0 calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class QEDSAPT0Results:
    """QED-SAPT0 component energies.

    Numerical fields default to ``0.0`` so a partially assembled scaffold can
    still summarize its current component state.  Real component evaluation is
    intentionally not implemented yet.
    """

    elst10: Optional[float] = 0.0
    exch10: Optional[float] = 0.0
    ind20: Optional[float] = 0.0
    exch_ind20: Optional[float] = 0.0
    disp20: Optional[float] = 0.0
    exch_disp20: Optional[float] = 0.0
    delta_hf: Optional[float] = 0.0
    qed_dse_cross: Optional[float] = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """Total energy from all non-``None`` components."""

        return sum(
            value
            for value in (
                self.elst10,
                self.exch10,
                self.ind20,
                self.exch_ind20,
                self.disp20,
                self.exch_disp20,
                self.delta_hf,
                self.qed_dse_cross,
            )
            if value is not None
        )

    def components(self) -> Dict[str, Optional[float]]:
        """Return the component energies as a plain dictionary."""

        return {
            "elst10": self.elst10,
            "exch10": self.exch10,
            "ind20": self.ind20,
            "exch_ind20": self.exch_ind20,
            "disp20": self.disp20,
            "exch_disp20": self.exch_disp20,
            "delta_hf": self.delta_hf,
            "qed_dse_cross": self.qed_dse_cross,
        }

    def summary(self) -> str:
        """Return a compact text summary of component and total energies."""

        lines = ["QED-SAPT0 component summary"]
        for name, value in self.components().items():
            text = "None" if value is None else f"{value:.12f}"
            lines.append(f"{name:>14s}: {text}")
        lines.append(f"{'total':>14s}: {self.total:.12f}")
        return "\n".join(lines)
