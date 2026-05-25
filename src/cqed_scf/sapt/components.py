"""Placeholder QED-SAPT0 component builders.

The initial implementation target is a full two-electron-integral backend.
Function signatures leave room for a future ``integral_backend="df"`` mode or
an explicit integral container without committing to density fitting now.
"""

from __future__ import annotations

from typing import Any


def compute_elst10(
    monomer_a: Any,
    monomer_b: Any,
    integrals: Any = None,
    integral_backend: str = "full_eri",
) -> float:
    """Compute future first-order electrostatics.

    Inputs will be two :class:`SAPTMonomer` objects plus full intermonomer AO/MO
    integrals.  The output will be the ``E_elst^(10)`` component in Hartree.
    """

    raise NotImplementedError(
        "compute_elst10 is not implemented yet. "
        "The first implementation should use full two-electron integrals."
    )


def compute_exch10(
    monomer_a: Any,
    monomer_b: Any,
    integrals: Any = None,
    integral_backend: str = "full_eri",
) -> float:
    """Compute future first-order exchange.

    Inputs will be monomer orbitals, occupations, overlap blocks, and full ERI
    tensors.  The output will be the ``E_exch^(10)`` component in Hartree.
    """

    raise NotImplementedError(
        "compute_exch10 is not implemented yet. "
        "The first implementation should use full two-electron integrals."
    )


def compute_ind20(
    monomer_a: Any,
    monomer_b: Any,
    response: Any = None,
    integrals: Any = None,
    integral_backend: str = "full_eri",
) -> float:
    """Compute future second-order induction.

    Inputs will include monomer response objects from ``cqed_scf.response`` and
    full ERI interaction blocks.  Density fitting can be added later by passing
    a different integral backend or integral container.
    """

    raise NotImplementedError(
        "compute_ind20 is not implemented yet. "
        "It will consume CQED response solvers and full ERI interaction blocks."
    )


def compute_disp20(
    monomer_a: Any,
    monomer_b: Any,
    response: Any = None,
    integrals: Any = None,
    integral_backend: str = "full_eri",
) -> float:
    """Compute future second-order dispersion.

    Inputs will include monomer excitation/response information and full ERI
    coupling blocks.  The initial implementation should be full-ERI based.
    """

    raise NotImplementedError(
        "compute_disp20 is not implemented yet. "
        "It will use monomer response/excitation data with full ERI couplings."
    )


def compute_qed_dse_cross(
    monomer_a: Any,
    monomer_b: Any,
    integrals: Any = None,
    integral_backend: str = "full_eri",
) -> float:
    """Compute future QED dipole self-energy cross terms.

    Inputs will include monomer densities, dipole-projected AO operators
    ``d_ao``, and any needed full-integral intermediates.  The output will be
    the QED DSE cross contribution in Hartree.
    """

    raise NotImplementedError(
        "compute_qed_dse_cross is not implemented yet. "
        "It will use monomer CQED-SCF densities and dipole-projected operators."
    )
