"""Central stdout layer for CQED-SCF.

Stage A introduces a single module-level output mechanism plus a `quiet`
toggle so that every code path (our own ``print`` calls and Psi4's engine
output) can be silenced from one place.  Call sites should never write to a
stream directly; they route through the emit helpers here so a later
redirection to an injected stream/file requires changing only this module.

This module also owns the Psi4-engine-output suppression: Python-side
``print`` calls cannot reach Psi4's C++ output manager, so a small
Psi4-aware helper is provided for that.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import Iterable, Sequence


class Verbosity:
    """Simple integer verbosity thresholds."""

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2  # reserved for future use / parity with ``debug``


_verbosity: int = Verbosity.NORMAL


def set_quiet(flag: bool) -> None:
    """Globally raise or lower the output threshold (``quiet=True`` silences)."""

    global _verbosity
    _verbosity = Verbosity.QUIET if flag else Verbosity.NORMAL


def is_quiet() -> bool:
    """Whether global output above the quiet threshold should be emitted."""

    return _verbosity < Verbosity.NORMAL


def is_verbose() -> bool:
    """Whether verbose (debug-level) output is enabled."""

    return _verbosity >= Verbosity.VERBOSE


@contextmanager
def quiet_context(q: bool = True):
    """Temporarily silence both this module's emit helpers and Psi4's engine.

    On exit the previous module verbosity (and Psi's prior output file) are
    restored, so this is a safe way to wrap e.g. ``psi4.energy()`` calls.
    When ``q`` is false the block is a no-op.
    """
    global _verbosity
    prior = _verbosity
    set_quiet(q)
    try:
        with psi4_silent():
            yield
    finally:
        _verbosity = prior


# -------------------------
# Emit helpers (normal level)
# -------------------------


def echo(message: str = "") -> None:
    """Print a single line at normal verbosity.  Closest to bare ``print``."""

    if is_quiet():
        return None
    print(message)
    return None


def banner(title: str) -> None:
    """Print a Psi4-style emphasized header."""

    if is_quiet():
        return None
    echo(f"\n  ==> {title} <== \n")
    return None


def property_(
    label: str,
    value: float,
    unit: str = "Eh",
    fmt: str = "18.12f",
) -> None:
    """Print a Psi4-``@``-convention final-result line.

    The ``@`` marker is the standard parseable prefix Psi4 uses for final
    quantities and is easy to grep.
    """

    if is_quiet():
        return None
    echo(f"@ {label:<46s} {value:{fmt}} {unit}")
    return None


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence],
    widths: Sequence[int],
) -> None:
    """Print aligned columnar output with a header separator."""

    if is_quiet():
        return None

    def _format(cell, width):
        return f"{str(cell):<{width}}"

    lines = [
        "  ".join(_format(h, w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append("  ".join(_format(c, w) for c, w in zip(row, widths)))
    print("\n".join(lines))
    return None


def energies(entries: Iterable[Sequence]) -> None:
    """Print several ``@`` property lines with consistent column widths."""

    if is_quiet():
        return None
    for label, value, unit in entries:
        property_(label, value, unit)
    return None


# -------------------------
# Psi4 engine output suppression
# -------------------------


def _null_sink() -> str:
    """Return a platform-appropriate null sink path."""

    if os.name == "nt":
        handle, path = tempfile.mkstemp(prefix="cqed_scf_null_", suffix=".out")
        os.close(handle)
        return path
    return "/dev/null"


_PSI4_AVAILABLE = False
try:
    import psi4

    _PSI4_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    _PSI4_AVAILABLE = False


@contextmanager
def psi4_silent():
    """Suppress Psi4's own engine output for the duration of the block.

    Introspects the installed Psi4's available API.  If ``get_output_file``
    / ``set_output_file`` are not available this degrades to a no-op so the
    surrounding calculation still runs.
    """
    if not _PSI4_AVAILABLE:
        yield
        return

    prior = None
    try:
        prior = psi4.core.get_output_file()
    except Exception:
        prior = None

    try:
        psi4.core.set_output_file(_null_sink())
    except Exception:
        prior = None  # could not divert; leave output on prior stream
    try:
        yield
    finally:
        if isinstance(prior, str):
            try:
                psi4.core.set_output_file(prior)
            except Exception:
                pass