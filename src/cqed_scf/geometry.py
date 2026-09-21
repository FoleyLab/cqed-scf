"""Consistency checks between a geometry specification and a :class:`CQEDConfig`.

Psi4 geometry strings carry their own charge and multiplicity on a leading
``"0 2"``-style line, and :class:`~cqed_scf.references.CQEDConfig` carries them
as fields.  Nothing used to compare the two, so they could disagree silently:
a doublet geometry could be run against a config that believed it was a closed
shell, or a cation could be run with ``charge=0``.

This module makes that disagreement an error.

Reading charge and multiplicity
-------------------------------
The source of truth is the *parsed molecule*, not the literal leading line.
Those are not the same thing: a geometry with no charge/multiplicity line at all
is not necessarily neutral-singlet, because Psi4 fills the multiplicity in from
the electron count.  ``O ... H ...`` with no leading line parses as charge 0,
multiplicity **2**, not ``0 1``.

Parsing goes through :meth:`psi4.core.Molecule.from_string`, which does not
activate the molecule as a global side effect the way :func:`psi4.geometry`
does, and which does not swallow construction failures.
"""

from __future__ import annotations

from typing import Any

from .references import CQEDConfig


def read_charge_and_multiplicity(geometry: Any) -> tuple[int, int]:
    """Return ``(charge, multiplicity)`` for a geometry string or Psi4 molecule.

    Handles Cartesian and Z-matrix input, and geometries that omit the
    charge/multiplicity line entirely.
    """

    if hasattr(geometry, "molecular_charge") and hasattr(geometry, "multiplicity"):
        return int(geometry.molecular_charge()), int(geometry.multiplicity())

    import psi4

    molecule = psi4.core.Molecule.from_string(geometry)
    return int(molecule.molecular_charge()), int(molecule.multiplicity())


def validate_geometry_against_config(geometry: Any, config: CQEDConfig) -> tuple[int, int]:
    """Raise unless ``geometry`` and ``config`` agree on charge and multiplicity.

    Returns the parsed ``(charge, multiplicity)`` on success so callers do not
    have to parse the geometry a second time.
    """

    try:
        charge, multiplicity = read_charge_and_multiplicity(geometry)
    except Exception as exc:
        raise ValueError(
            "could not determine the charge and multiplicity of the geometry "
            f"(CQEDConfig has charge={config.charge}, "
            f"multiplicity={config.multiplicity}): {exc}"
        ) from exc

    mismatches = []
    if charge != config.charge:
        mismatches.append(
            f"charge: CQEDConfig has {config.charge} but the geometry specifies "
            f"{charge}"
        )
    if multiplicity != config.multiplicity:
        mismatches.append(
            f"multiplicity: CQEDConfig has {config.multiplicity} but the "
            f"geometry specifies {multiplicity}"
        )

    if mismatches:
        detail = "; ".join(mismatches)
        raise ValueError(
            f"geometry and CQEDConfig disagree -- {detail}. Update the leading "
            "charge/multiplicity line of the geometry or the corresponding "
            "CQEDConfig field so the two match. Note that a geometry with no "
            "such line is not automatically '0 1': Psi4 infers the "
            "multiplicity from the electron count."
        )

    return charge, multiplicity
