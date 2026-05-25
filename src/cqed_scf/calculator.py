"""User-facing CQED calculation facade.

The calculator owns compatibility with the historical package API and routes
work to lower-level engines.  Restricted CQED-SCF and gradients remain the only
implemented physics paths; unrestricted references, response theory, and
QED-SAPT0 are exposed here as explicit future hooks.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .references import CQEDConfig


class CQEDCalculator:
    """Facade and dispatcher for CQED calculations.

    The constructor accepts either a :class:`CQEDConfig` or the original keyword
    arguments used by earlier versions of the package.  Restricted RHF/RKS work
    is delegated to :class:`cqed_scf.scf.CQEDSCF`; unrestricted requests are
    routed to placeholder drivers that document the intended future data flow.
    """

    def __init__(
        self,
        lambda_vector: Any = None,
        psi4_options: Optional[Mapping[str, Any]] = None,
        omega: float = 0.1,
        charge: int = 0,
        multiplicity: int = 1,
        density_fitting: bool = False,
        functional: Optional[str] = None,
        debug: bool = False,
        reference: Optional[str] = None,
        dispersion_policy: str = "post_scf",
        config: Optional[CQEDConfig] = None,
    ):
        if isinstance(lambda_vector, CQEDConfig):
            if config is not None:
                raise ValueError(
                    "Pass either config=... or a CQEDConfig positional argument, not both."
                )
            config = lambda_vector

        if config is not None:
            if not isinstance(config, CQEDConfig):
                raise TypeError("config must be a CQEDConfig instance")
            self.config = config
        else:
            if lambda_vector is None:
                raise TypeError("lambda_vector is required when config is not provided")
            self.config = CQEDConfig.from_legacy_kwargs(
                lambda_vector=lambda_vector,
                psi4_options=psi4_options,
                omega=omega,
                charge=charge,
                multiplicity=multiplicity,
                density_fitting=density_fitting,
                functional=functional,
                reference=reference,
                dispersion_policy=dispersion_policy,
                debug=debug,
            )

        self.geometry = None
        self._refresh_compatibility_attrs()

    # -------------------------
    # Backward-compatible attributes
    # -------------------------

    def _refresh_compatibility_attrs(self) -> None:
        self._has_dispersion = self.config.apply_post_scf_dispersion
        self._base_functional = self.config.base_scf_functional
        self._dispersion_map = {"wb97x-d": "wb97x"}

    @property
    def lambda_vector(self):
        return self.config.lambda_vector

    @lambda_vector.setter
    def lambda_vector(self, value):
        self.config = self.config.copy_with(lambda_vector=value)
        self._refresh_compatibility_attrs()

    @property
    def psi4_options(self):
        return self.config.psi4_options

    @psi4_options.setter
    def psi4_options(self, value):
        self.config = self.config.copy_with(psi4_options=dict(value or {}))
        self._refresh_compatibility_attrs()

    @property
    def omega(self):
        return self.config.omega

    @omega.setter
    def omega(self, value):
        self.config = self.config.copy_with(omega=value)
        self._refresh_compatibility_attrs()

    @property
    def density_fitting(self):
        return self.config.density_fitting

    @density_fitting.setter
    def density_fitting(self, value):
        self.config = self.config.copy_with(density_fitting=bool(value))
        self._refresh_compatibility_attrs()

    @property
    def functional(self):
        return self.config.functional

    @functional.setter
    def functional(self, value):
        self.config = self.config.copy_with(functional=value)
        self._refresh_compatibility_attrs()

    @property
    def debug(self):
        return self.config.debug

    @debug.setter
    def debug(self, value):
        self.config = self.config.copy_with(debug=bool(value))
        self._refresh_compatibility_attrs()

    @property
    def charge(self):
        return self.config.charge

    @charge.setter
    def charge(self, value):
        self.config = self.config.copy_with(charge=int(value))
        self._refresh_compatibility_attrs()

    @property
    def multiplicity(self):
        return self.config.multiplicity

    @multiplicity.setter
    def multiplicity(self, value):
        self.config = self.config.copy_with(multiplicity=int(value))
        self._refresh_compatibility_attrs()

    # -------------------------
    # internal helpers
    # -------------------------

    def _make_restricted_scf(self, geometry):
        from .scf import CQEDSCF

        if self.config.multiplicity != 1:
            raise NotImplementedError(
                "Restricted CQED-SCF currently supports only multiplicity=1. "
                "Use reference='uhf' or reference='uks' once CQEDUSCF is implemented."
            )

        return CQEDSCF(
            geometry=geometry,
            lambda_vector=self.config.lambda_vector,
            psi4_options=self.config.psi4_options,
            omega=self.config.omega,
            density_fitting=self.config.density_fitting,
            method=self.config.scf_method,
            functional=self.config.base_scf_functional,
            debug=self.config.debug,
        )

    def _make_unrestricted_scf(self, geometry):
        from .uscf import CQEDUSCF

        return CQEDUSCF(geometry=geometry, config=self.config)

    def _run_scf(self, geometry):
        if self.config.is_unrestricted:
            scf = self._make_unrestricted_scf(geometry)
            return scf.run()

        scf = self._make_restricted_scf(geometry)
        print("\nRunning CQED-SCF energy calculation...\n")
        print(f"Functional: {self.config.base_scf_functional}")
        return scf.run()

    def _compute_dispersion_energy(self, geometry, energy_no_disp):
        import psi4

        if not self.config.apply_post_scf_dispersion:
            return 0.0

        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_options(self.config.psi4_options)

        mol = psi4.geometry(geometry)
        energy_with_dispersion = psi4.energy(self.config.functional, molecule=mol)

        return energy_with_dispersion - energy_no_disp

    def _compute_dispersion_gradient(self, geometry, energy_no_disp, grad_no_disp):
        import numpy as np
        import psi4

        if not self.config.apply_post_scf_dispersion:
            mol = psi4.geometry(geometry)
            return 0.0, np.zeros((mol.natom(), 3))

        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_options(self.config.psi4_options)

        mol = psi4.geometry(geometry)

        grad_w = psi4.gradient(self.config.functional, molecule=mol)
        energy_w = psi4.core.variable("CURRENT ENERGY")

        disp_gradient = grad_w.np - grad_no_disp
        dispersion_energy = energy_w - energy_no_disp

        return dispersion_energy, disp_gradient

    # -------------------------
    # public API
    # -------------------------

    def energy(self, geometry):
        self.geometry = geometry

        energy_qed, results = self._run_scf(geometry)
        energy_psi4_base = results["energy_psi4"]

        if self.config.apply_post_scf_dispersion:
            energy_disp = self._compute_dispersion_energy(geometry, energy_psi4_base)
            print(f"Dispersion correction energy: {energy_disp:.12f} Eh")
            print(f"Total energy (CQED + dispersion): {energy_qed + energy_disp:.12f} Eh")
        else:
            energy_disp = 0.0

        energy_total = energy_qed + energy_disp

        if self.config.debug:
            print(f"E_QED  = {energy_qed: .12f}")
            print(f"E_disp = {energy_disp: .12f}")
            print(f"E_tot  = {energy_total: .12f}")

        return energy_total

    def energy_and_gradient(self, geometry, canonical="psi4"):
        import numpy as np
        import psi4

        self.geometry = geometry

        energy_qed, data = self._run_scf(geometry)
        energy_psi4_base = data["energy_psi4"]

        if self.config.is_unrestricted:
            from .ugradients import CQEDUGradient

            grad_engine = CQEDUGradient(self.config)
            grad_engine.compute(data)

        from .gradients import CQEDGradient

        grad_engine = CQEDGradient(
            self.config.lambda_vector,
            canonical=canonical,
            debug=self.config.debug,
        )

        grad_results = grad_engine.compute(data)
        canonical_grad = grad_results["canonical_grad"]
        grad_qed = grad_results["total_grad"]

        if self.config.apply_post_scf_dispersion:
            energy_disp, grad_disp = self._compute_dispersion_gradient(
                geometry,
                energy_psi4_base,
                canonical_grad,
            )
            print(f"Dispersion correction energy: {energy_disp:.12f} Eh")
            print(f"Total energy (CQED + dispersion): {energy_qed + energy_disp:.12f} Eh")
            print(
                "Dispersion correction gradient norm: {:.6e} Eh/Bohr".format(
                    np.linalg.norm(grad_disp)
                )
            )
        else:
            print("No dispersion correction applied.")
            energy_disp = 0.0
            grad_disp = np.zeros_like(grad_qed)

        energy_total = energy_qed + energy_disp
        grad_total = grad_qed + grad_disp

        g = (self.config.omega / 2) ** 0.5 * data["d_exp"]

        if self.config.debug:
            print(f"E_QED      = {energy_qed: .12f}")
            print(f"E_disp     = {energy_disp: .12f}")
            print(f"E_total    = {energy_total: .12f}")
            print(f"|grad_disp|= {np.linalg.norm(grad_disp): .6e}")

        psi4.core.clean()
        psi4.core.clean_options()

        return energy_total, grad_total, g

    def response(self, scf_results=None, **kwargs):
        """Create a future CQED response-theory driver."""

        from .response import CQEDResponse

        return CQEDResponse(config=self.config, scf_results=scf_results, **kwargs)

    def tddft(self, scf_results=None, **kwargs):
        """Create a future LR-TDDFT driver."""

        from .response import CQEDTDDFT

        return CQEDTDDFT(config=self.config, scf_results=scf_results, **kwargs)

    def sapt0(self, dimer_geometry, **kwargs):
        """Create a future QED-SAPT0 driver for a dimer calculation."""

        from .sapt import QEDSAPT0Driver

        return QEDSAPT0Driver(dimer_geometry=dimer_geometry, config=self.config, **kwargs)


# backward-compatible alias
CQEDRHFCalculator = CQEDCalculator
