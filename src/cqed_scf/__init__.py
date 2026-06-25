"""Public package exports for cqed_scf."""

from .version import __version__
from .calculator import CQEDCalculator, CQEDRHFCalculator
from .references import CQEDConfig
from .sapt import QEDSAPT0Driver, QEDSAPT0Results


def __getattr__(name):
    """Lazily import Psi4-dependent engines only when requested."""

    if name in {"CQEDSCF", "CQEDRHFSCF"}:
        from .scf import CQEDRHFSCF, CQEDSCF

        return CQEDSCF if name == "CQEDSCF" else CQEDRHFSCF

    if name in {"CQEDGradient", "CQEDRHFGradient"}:
        from .gradients import CQEDGradient, CQEDRHFGradient

        return CQEDGradient if name == "CQEDGradient" else CQEDRHFGradient

    raise AttributeError(f"module 'cqed_scf' has no attribute {name!r}")


__all__ = [
    "__version__",
    "CQEDConfig",
    "CQEDSCF",
    "CQEDGradient",
    "CQEDCalculator",
    "CQEDRHFGradient",
    "CQEDRHFCalculator",
    "QEDSAPT0Driver",
    "QEDSAPT0Results",
]
