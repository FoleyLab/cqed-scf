"""Future unrestricted CQED-SCF engine.

The current production implementation supports restricted closed-shell
references through :mod:`cqed_scf.scf`.  This module reserves the unrestricted
entry point and documents the data contract expected by response theory and
QED-SAPT0 once UHF/UKS support is added.
"""

from __future__ import annotations

from typing import Any

from .references import CQEDConfig


class CQEDUSCF:
    """Placeholder unrestricted CQED-SCF driver.

    Intended workflow:
    1. Build separate alpha/beta reference wavefunctions from Psi4 UHF/UKS.
    2. Construct CQED one-electron and dipole self-energy contributions.
    3. Iterate alpha/beta Fock builds to convergence.
    4. Return an SCF result dictionary compatible with restricted results, with
       spin-resolved extensions for ``coefficients``, ``orbital_energies``, and
       ``density``.
    """

    def __init__(self, geometry: Any, config: CQEDConfig):
        self.geometry = geometry
        self.config = config

    def run(self):
        """Run future unrestricted CQED-SCF."""

        raise NotImplementedError(
            "CQEDUSCF is an architecture placeholder. "
            "Unrestricted CQED-SCF physics has not been implemented yet."
        )


CQEDUHFSCF = CQEDUSCF
