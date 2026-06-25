"""Shared response-theory scaffolding for CQED calculations.

Response drivers will consume converged SCF result dictionaries produced by
``cqed_scf.scf`` and, later, ``cqed_scf.uscf``.  The restricted SCF dictionary
already exposes the core objects needed here:

- ``coefficients``
- ``orbital_energies``
- ``density``
- ``mints``
- ``wfn``
- ``d_ao``
- ``d_exp``
- ``ndocc``
- ``functional`` and ``reference``/``method`` metadata

No response physics is implemented in this module yet.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .references import CQEDConfig


class CQEDResponse:
    """Base class for future CQED CPHF/CPKS/TDDFT response solvers.

    Intended workflow:
    1. Read occupied and virtual orbital blocks from SCF coefficients and
       orbital energies.
    2. Build property-specific right-hand-side vectors.
    3. Provide matrix-free Hessian-vector products ``sigma = A[X]``.
    4. Solve linear systems ``A X = b`` with full ERIs first.
    5. Add density-fitting backends later behind the same interface.
    6. Supply reusable response objects to SAPT induction and LR-TDDFT.
    """

    required_scf_keys = (
        "coefficients",
        "orbital_energies",
        "density",
        "mints",
        "wfn",
        "d_ao",
        "d_exp",
        "ndocc",
        "functional",
        "reference",
    )

    def __init__(
        self,
        config: CQEDConfig,
        scf_results: Optional[Mapping[str, Any]] = None,
        integral_backend: str = "full_eri",
    ):
        self.config = config
        self.scf_results = scf_results
        self.integral_backend = integral_backend

    def validate_scf_results(self) -> None:
        """Validate that the SCF result dictionary has the expected contract."""

        if self.scf_results is None:
            raise ValueError("scf_results must be supplied before building response objects")

        missing = [key for key in self.required_scf_keys if key not in self.scf_results]
        if missing:
            raise KeyError(f"SCF results missing required response keys: {missing}")

    def build_orbital_blocks(self):
        """Build occupied/virtual orbital blocks for response equations."""

        raise NotImplementedError(
            "CQEDResponse.build_orbital_blocks is not implemented yet. "
            "Future code will build occupied/virtual blocks from SCF results."
        )

    def build_rhs(self, operator: Any):
        """Build a property right-hand side for ``A X = b``."""

        raise NotImplementedError(
            "CQEDResponse.build_rhs is not implemented yet. "
            "Future code will build response right-hand sides from AO operators."
        )

    def sigma(self, amplitudes: Any):
        """Apply the matrix-free response Hessian to trial amplitudes."""

        raise NotImplementedError(
            "CQEDResponse.sigma is not implemented yet. "
            "Future code will evaluate sigma = A[X], using full ERIs first."
        )

    def solve(self, rhs: Any):
        """Solve future CPHF/CPKS linear response equations."""

        raise NotImplementedError(
            "CQEDResponse.solve is not implemented yet. "
            "Future code will solve A X = b for SAPT induction and properties."
        )


class CQEDCPHF(CQEDResponse):
    """Future restricted/unrestricted CPHF response driver."""


class CQEDCPKS(CQEDResponse):
    """Future restricted/unrestricted CPKS response driver."""


class CQEDTDDFT(CQEDResponse):
    """Future LR-TDDFT driver built on shared CQED response machinery."""

    def excitation_energies(self, nroots: Optional[int] = None):
        """Compute future CQED LR-TDDFT excitation energies."""

        raise NotImplementedError(
            "CQEDTDDFT.excitation_energies is not implemented yet. "
            "Future code will use the shared response Hessian and SCF metadata."
        )
