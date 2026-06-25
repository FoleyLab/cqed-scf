"""Reference and calculation configuration helpers for CQED workflows.

This module intentionally contains no Psi4 imports.  The goal is to keep the
user-facing calculator, SCF engines, response drivers, and SAPT scaffolding
sharing one lightweight description of a calculation without tying that
description to any particular physics implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional


_REFERENCE_ALIASES = {
    "hf": "rhf",
    "rhf": "rhf",
    "restricted_hf": "rhf",
    "restricted-hf": "rhf",
    "rks": "rks",
    "ks": "rks",
    "dft": "rks",
    "restricted_ks": "rks",
    "restricted-ks": "rks",
    "uhf": "uhf",
    "unrestricted_hf": "uhf",
    "unrestricted-hf": "uhf",
    "uks": "uks",
    "unrestricted_ks": "uks",
    "unrestricted-ks": "uks",
}

_DISPERSION_STRIP_MAP = {
    "wb97x-d": "wb97x",
}

_DISPERSION_POLICIES = {"post_scf", "none", "scf"}


def normalize_reference_name(reference: Optional[str], functional: Optional[str] = None) -> str:
    """Return a canonical reference name from user-facing aliases.

    If no reference is supplied, the historical calculator behavior is used:
    calculations with a functional default to RKS and calculations without one
    default to RHF.
    """

    if reference is None:
        return "rks" if functional is not None else "rhf"

    key = reference.strip().lower()
    try:
        return _REFERENCE_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted({"rhf", "rks", "uhf", "uks"}))
        raise ValueError(f"reference must be one of {allowed}; got {reference!r}") from exc


@dataclass
class CQEDConfig:
    """Central configuration for CQED-SCF and future response/SAPT drivers.

    Parameters mirror the original :class:`cqed_scf.calculator.CQEDCalculator`
    keyword arguments while adding explicit reference metadata for future
    unrestricted and response-theory code paths.
    """

    lambda_vector: tuple[float, float, float]
    omega: float = 0.1
    psi4_options: dict[str, Any] = field(default_factory=dict)
    reference: Optional[str] = None
    functional: Optional[str] = None
    density_fitting: bool = False
    charge: int = 0
    multiplicity: int = 1
    dispersion_policy: str = "post_scf"
    debug: bool = False

    def __post_init__(self) -> None:
        self.lambda_vector = self._coerce_lambda_vector(self.lambda_vector)
        self.psi4_options = dict(self.psi4_options or {})
        self.reference = normalize_reference_name(self.reference, self.functional)
        self.dispersion_policy = self.dispersion_policy.strip().lower()
        if self.dispersion_policy not in _DISPERSION_POLICIES:
            allowed = ", ".join(sorted(_DISPERSION_POLICIES))
            raise ValueError(
                f"dispersion_policy must be one of {allowed}; "
                f"got {self.dispersion_policy!r}"
            )
        self._validate_functional()

    @classmethod
    def from_legacy_kwargs(
        cls,
        *,
        lambda_vector: Any,
        psi4_options: Optional[Mapping[str, Any]],
        omega: float = 0.1,
        charge: int = 0,
        multiplicity: int = 1,
        density_fitting: bool = False,
        functional: Optional[str] = None,
        reference: Optional[str] = None,
        dispersion_policy: str = "post_scf",
        debug: bool = False,
    ) -> "CQEDConfig":
        """Build a config from the historical calculator keyword arguments."""

        return cls(
            lambda_vector=lambda_vector,
            omega=omega,
            psi4_options=dict(psi4_options or {}),
            reference=reference,
            functional=functional,
            density_fitting=density_fitting,
            charge=charge,
            multiplicity=multiplicity,
            dispersion_policy=dispersion_policy,
            debug=debug,
        )

    @staticmethod
    def _coerce_lambda_vector(lambda_vector: Any) -> tuple[float, float, float]:
        try:
            values = tuple(float(x) for x in lambda_vector)
        except TypeError as exc:
            raise TypeError("lambda_vector must be an iterable with three numbers") from exc

        if len(values) != 3:
            raise ValueError("lambda_vector must contain exactly three components")
        return values

    def _validate_functional(self) -> None:
        if self.is_ks and self.functional is None:
            raise ValueError(f"functional must be provided for {self.reference.upper()}")

    @property
    def is_restricted(self) -> bool:
        """Whether the reference is restricted."""

        return self.reference in {"rhf", "rks"}

    @property
    def is_unrestricted(self) -> bool:
        """Whether the reference is unrestricted."""

        return self.reference in {"uhf", "uks"}

    @property
    def is_hf(self) -> bool:
        """Whether the reference uses a Hartree-Fock exchange-correlation model."""

        return self.reference in {"rhf", "uhf"}

    @property
    def is_ks(self) -> bool:
        """Whether the reference uses a Kohn-Sham exchange-correlation model."""

        return self.reference in {"rks", "uks"}

    @property
    def has_strippable_dispersion(self) -> bool:
        """Whether the named functional has a known nonlocal dispersion suffix."""

        return (
            self.is_ks
            and self.functional is not None
            and self.functional.lower() in _DISPERSION_STRIP_MAP
        )

    @property
    def apply_post_scf_dispersion(self) -> bool:
        """Whether the facade should add a post-SCF dispersion correction."""

        return self.dispersion_policy == "post_scf" and self.has_strippable_dispersion

    @property
    def base_scf_functional(self) -> Optional[str]:
        """Functional to pass to the base SCF engine.

        For ``dispersion_policy="post_scf"`` or ``"none"``, known dispersion
        variants are stripped before CQED-SCF so a separate dispersion
        correction can be added or intentionally omitted.  For
        ``dispersion_policy="scf"``, the functional is passed through unchanged.
        HF references return ``None`` because the restricted SCF engine expects
        no functional for RHF.
        """

        if self.is_hf:
            return None
        if self.functional is None:
            return None
        if self.dispersion_policy == "scf":
            return self.functional
        return _DISPERSION_STRIP_MAP.get(self.functional.lower(), self.functional)

    @property
    def scf_method(self) -> str:
        """Method string expected by the current restricted CQED-SCF engine."""

        if self.is_unrestricted:
            return self.reference
        return "rks" if self.is_ks else "rhf"

    def copy_with(self, **updates: Any) -> "CQEDConfig":
        """Return a validated copy with selected fields replaced."""

        return replace(self, **updates)
