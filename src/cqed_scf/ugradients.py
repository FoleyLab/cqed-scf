"""Future unrestricted CQED nuclear gradients."""

from __future__ import annotations

from typing import Any, Mapping

from .references import CQEDConfig


class CQEDUGradient:
    """Placeholder unrestricted gradient driver.

    The future implementation will consume spin-resolved SCF result
    dictionaries from :class:`cqed_scf.uscf.CQEDUSCF`, build alpha/beta
    one-particle density responses as needed, and return gradient components in
    the same style as :class:`cqed_scf.gradients.CQEDGradient`.
    """

    def __init__(self, config: CQEDConfig, canonical: str = "psi4", debug: bool = False):
        self.config = config
        self.canonical = canonical
        self.debug = debug

    def compute(self, scf_results: Mapping[str, Any]):
        """Compute future unrestricted CQED gradients."""

        raise NotImplementedError(
            "CQEDUGradient is an architecture placeholder. "
            "Unrestricted CQED gradient physics has not been implemented yet."
        )


CQEDUHFGradient = CQEDUGradient
