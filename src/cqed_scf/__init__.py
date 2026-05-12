from .version import __version__
from .scf import CQEDSCF
from .gradients import CQEDGradient
from .calculator import CQEDCalculator

# backward-compatible aliases
CQEDRHFGradient = CQEDGradient
CQEDRHFCalculator = CQEDCalculator

__all__ = ["CQEDSCF", "CQEDGradient", "CQEDCalculator", "CQEDRHFGradient", "CQEDRHFCalculator"]
