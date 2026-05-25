"""Monomer reference containers for future QED-SAPT0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..references import CQEDConfig


@dataclass
class SAPTMonomer:
    """Container for a monomer CQED-SCF reference.

    The first QED-SAPT0 implementation will use monomer SCF references and
    full two-electron integrals.  This object keeps the monomer geometry,
    reference settings, and SCF result dictionary together so component
    functions can consume a consistent interface.
    """

    label: str
    geometry: Any
    charge: int
    multiplicity: int
    config: CQEDConfig
    scf_results: Mapping[str, Any]

    @property
    def C(self):
        return self.scf_results.get("coefficients")

    @property
    def eps(self):
        return self.scf_results.get("orbital_energies")

    @property
    def D(self):
        return self.scf_results.get("density")

    @property
    def mints(self):
        return self.scf_results.get("mints")

    @property
    def wfn(self):
        return self.scf_results.get("wfn")

    @property
    def d_ao(self):
        return self.scf_results.get("d_ao")

    @property
    def d_exp(self):
        return self.scf_results.get("d_exp")

    @property
    def ndocc(self):
        return self.scf_results.get("ndocc")
    
    @property
    def energy_scf(self):
        return self.scf_results.get("energy_scf")

    @classmethod
    def from_scf_results(
        cls,
        label: str,
        geometry: Any,
        scf_results: Mapping[str, Any],
        config: CQEDConfig,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
    ) -> "SAPTMonomer":
        """Wrap an existing CQED-SCF result dictionary as a SAPT monomer."""

        monomer_config = config.copy_with(
            charge=config.charge if charge is None else charge,
            multiplicity=config.multiplicity if multiplicity is None else multiplicity,
        )
        return cls(
            label=label,
            geometry=geometry,
            charge=monomer_config.charge,
            multiplicity=monomer_config.multiplicity,
            config=monomer_config,
            scf_results=scf_results,
        )

    @classmethod
    def from_cqed_scf(
        cls,
        label: str,
        geometry: Any,
        config: CQEDConfig,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
    ) -> "SAPTMonomer":
        """Run a monomer CQED-SCF calculation and wrap its results.

        This is safe for the current restricted RHF/RKS engine.  Unrestricted
        monomers route to future ``CQEDUSCF`` and currently raise
        ``NotImplementedError``.
        """

        monomer_config = config.copy_with(
            charge=config.charge if charge is None else charge,
            multiplicity=config.multiplicity if multiplicity is None else multiplicity,
        )

        if monomer_config.is_unrestricted:
            from ..uscf import CQEDUSCF

            _, scf_results = CQEDUSCF(geometry=geometry, config=monomer_config).run()
        else:
            from ..scf import CQEDSCF

            scf = CQEDSCF(
                geometry=geometry,
                lambda_vector=monomer_config.lambda_vector,
                psi4_options=monomer_config.psi4_options,
                omega=monomer_config.omega,
                density_fitting=monomer_config.density_fitting,
                method=monomer_config.scf_method,
                functional=monomer_config.base_scf_functional,
                debug=monomer_config.debug,
            )
            _, scf_results = scf.run()

        return cls.from_scf_results(
            label=label,
            geometry=geometry,
            scf_results=scf_results,
            config=monomer_config,
            charge=monomer_config.charge,
            multiplicity=monomer_config.multiplicity,
        )
