"""Density-fitted two-electron integrals for QED-SAPT0.

The unifying statement, shared with :mod:`cqed_scf.sapt.dse_jk`:

    **The two-electron dipole self-energy is an ERI with a separable kernel:**
    ``(pq|rs) -> d_pq d_rs``.  Every Slater-Condon result carries over verbatim
    under that substitution.

Read that in density-fitting language and it says something stronger.  Ordinary
density fitting writes the Coulomb kernel as a sum over auxiliary functions,

    ``(pq|rs) ~= sum_Q B^Q_pq B^Q_rs``,      ``Q = 1 .. naux``

while the coherent-state DSE kernel is a *single* separable product.  So the
combined Pauli-Fierz two-electron integral is an ordinary DF expansion with one
extra auxiliary row appended:

    ``(pq|rs)_PF ~= sum_Q' B^Q'_pq B^Q'_rs``,   ``Q' = 1 .. naux + 1``

with ``B^(naux+1) := d``.  The DSE row is *exact* -- it carries no fitting
error, unlike the ``naux`` Coulomb rows.

Two consequences make this the whole design:

1. Every downstream MP2-like formula is unchanged.  Amplitudes, ``Disp20`` and
   the ``Exch-Disp20`` algebra all see one tensor with one more auxiliary
   index, so there is no parallel cavity code path and no new sign convention
   to reconcile.
2. The ``standard`` / ``cavity`` / ``total`` partitions are slices of the
   auxiliary index, matching the operator-context vocabulary already used by
   :class:`cqed_scf.sapt.qed_sapt0.QEDSAPT0Driver`.

This requires the kernel to be symmetric under ``(pq) <-> (rs)``, i.e.
``d_A == d_B``.  That holds structurally because ``d_ao = lambda . mu_ao`` is a
property of the shared dimer AO basis and frame rather than of either monomer,
and both monomers are ghosted calculations in that basis.  It is asserted, not
assumed -- see :meth:`PauliFierzDF.from_driver`.

See ``docs/qed_sapt0_formalism.tex`` for the derivation and
``docs/development/QED_SAPT_DISPERSION_PLAN.md`` for how this fits the driver.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import opt_einsum as oe
import psi4
from psi4 import core


_OPERATOR_CONTEXTS = ("standard", "total", "cavity")

# Orbital-space labels shared with QEDSAPT0Driver: occupied/virtual of A and B.
_ORBITAL_LABELS = ("a", "r", "b", "s")


def build_df_ao_tensor(basisset, aux_basisset):
    """Build the symmetric AO three-index tensor ``B^Q_pq``.

    Returns ``(naux, nbf, nbf)`` with ``sum_Q B^Q_pq B^Q_rs ~= (pq|rs)``, i.e.
    the metric's inverse square root is already folded in.

    The ``(aux, zero, primary, primary)`` argument order matters:
    ``(zero, aux, primary, primary)`` raises ``Bad BraKet type in
    Libint2TwoElectronInt``.  This is one instance of the general rule in
    ``docs/SAPT_NOTE.md``: pin every Psi4 integral call against an
    independently constructed reference rather than trusting the signature.
    """
    zero = core.BasisSet.zero_ao_basis_set()
    mints = core.MintsHelper(basisset)

    Ppq = np.squeeze(mints.ao_eri(aux_basisset, zero, basisset, basisset))
    metric = np.squeeze(mints.ao_eri(aux_basisset, zero, aux_basisset, zero))

    metric_matrix = core.Matrix.from_array(metric)
    metric_matrix.power(-0.5, 1.0e-12)

    return oe.contract("PQ,Qpq->Ppq", np.asarray(metric_matrix), Ppq, optimize="optimal")


class PauliFierzDF:
    """Density-fitted Pauli-Fierz two-electron integrals in the SAPT MO spaces.

    This is the MP2-layer sibling of :class:`cqed_scf.sapt.dse_jk.PauliFierzJK`:
    where ``PauliFierzJK`` duck-types a ``psi4.core.JK`` so the Coulomb/exchange
    algebra never learns the DSE exists, ``PauliFierzDF`` supplies a three-index
    tensor so the MP2-like algebra never learns it either.

    The primitive is :meth:`b`, which takes a **two-character** orbital pair --
    the same vocabulary as ``QEDSAPT0Driver.s()`` and ``.potential()``.  The
    four-index :meth:`v` is derived from it, because ``v(s0 s1 s2 s3)`` pairs
    orbital positions ``(0, 2)`` and ``(1, 3)``:

        ``v(s0 s1 s2 s3)[A,B,C,D] = sum_Q b(s0 s2)[Q,A,C] * b(s1 s3)[Q,B,D]``

    Storage note: the AO tensor is half-transformed once into the union MO space
    ``[Co_A | Cv_A | Co_B | Cv_B]``, so every pair block is a *slice* of one
    array rather than a separately cached tensor.  All 16 pairs are therefore
    available at the cost of the largest one.
    """

    def __init__(self, B_ao, d_ao, orbitals: Mapping[str, np.ndarray], include_cavity_terms: bool = True):
        self._validate_ao_tensor(B_ao, d_ao)

        self.include_cavity_terms = bool(include_cavity_terms)
        self.naux = int(B_ao.shape[0])
        self.nbf = int(B_ao.shape[1])

        missing = [label for label in _ORBITAL_LABELS if label not in orbitals]
        if missing:
            raise ValueError(f"orbitals is missing required labels: {', '.join(missing)}")

        # Column offsets of each orbital space inside the union MO basis.
        self._orbitals = {label: np.asarray(orbitals[label]) for label in _ORBITAL_LABELS}
        self._slices: Dict[str, slice] = {}
        offset = 0
        blocks = []
        for label in _ORBITAL_LABELS:
            C = self._orbitals[label]
            if C.ndim != 2 or C.shape[0] != self.nbf:
                raise ValueError(
                    f"orbital block {label!r} has shape {C.shape}; expected ({self.nbf}, n)"
                )
            width = C.shape[1]
            self._slices[label] = slice(offset, offset + width)
            blocks.append(C)
            offset += width
        C_all = np.hstack(blocks)

        # The augmented auxiliary index: naux Coulomb rows, then one exact DSE
        # row.  When the cavity is disabled the extra row is simply absent.
        if self.include_cavity_terms:
            B_aug = np.concatenate([B_ao, np.asarray(d_ao)[None, :, :]], axis=0)
        else:
            B_aug = B_ao

        # One half-transform into the union MO space; pair blocks are slices.
        self._B_mo = oe.contract("Ppq,pA,qC->PAC", B_aug, C_all, C_all, optimize="optimal")
        self._B_mo.flags.writeable = False

    # -- construction ----------------------------------------------------

    @classmethod
    def from_driver(cls, driver, aux_basis_name: Optional[str] = None, fitting_role: str = "RIFIT"):
        """Build from a prepared :class:`QEDSAPT0Driver`.

        ``driver.build_integrals()`` (or at least ``prepare_monomers()`` plus
        ``build_orbitals()``) must have run.
        """
        if getattr(driver, "orbitals", None) is None:
            raise RuntimeError("PauliFierzDF.from_driver requires driver.build_orbitals() to have run.")

        # Always the shared dimer frame: with intrinsic-frame monomer references
        # the wavefunction's own basis sits somewhere else entirely, and the
        # interaction integrals must not follow it there.
        basisset = driver._dimer_frame_basisset("A")
        molecule = driver._dimer_frame_molecule("A")

        if aux_basis_name is None:
            aux_basis_name = core.get_global_option("BASIS")

        aux = core.BasisSet.build(molecule, "DF_BASIS_SAPT", "", fitting_role, aux_basis_name)

        d_A = np.asarray(driver.d_A)
        d_B = np.asarray(driver.d_B)
        if not np.allclose(d_A, d_B, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "PauliFierzDF requires d_A == d_B so that the cavity kernel is "
                "symmetric and representable as a single auxiliary row "
                f"(max|d_A - d_B| = {np.abs(d_A - d_B).max():.3e})."
            )

        B_ao = build_df_ao_tensor(basisset, aux)
        return cls(
            B_ao=B_ao,
            d_ao=d_A,
            orbitals=driver.orbitals,
            include_cavity_terms=bool(driver.include_cavity_terms),
        )

    # -- primitives ------------------------------------------------------

    @property
    def cavity_is_active(self) -> bool:
        return self.include_cavity_terms

    def _aux_slice(self, context: str) -> slice:
        if context not in _OPERATOR_CONTEXTS:
            allowed = ", ".join(_OPERATOR_CONTEXTS)
            raise ValueError(f"operator context must be one of {allowed}; got {context!r}")
        if context == "standard":
            return slice(0, self.naux)
        if context == "cavity":
            # Empty slice when the cavity is disabled: contractions then yield
            # exact zeros without a special case.
            return slice(self.naux, self.naux + (1 if self.include_cavity_terms else 0))
        return slice(0, self.naux + (1 if self.include_cavity_terms else 0))

    def b(self, pair: str, context: str = "total") -> np.ndarray:
        """Three-index MO pair block ``b^Q_xy``, shape ``(naux', n_x, n_y)``.

        ``pair`` is two orbital labels drawn from ``a`` (occ A), ``r`` (vir A),
        ``b`` (occ B), ``s`` (vir B).  Cross-monomer pairs such as ``ab``,
        ``as``, ``br`` are valid and are needed by exchange-dispersion.

        The returned array is a **read-only view** into one shared tensor.  Do
        not write to it: the three operator contexts are overlapping slices of
        that tensor, so an in-place update would silently couple them.
        """
        if len(pair) != 2:
            raise ValueError(f"b: pair {pair!r} does not have length 2")
        for label in pair:
            if label not in self._slices:
                raise ValueError(
                    f"b: {label!r} is not a valid orbital label; expected one of {_ORBITAL_LABELS}"
                )

        return self._B_mo[self._aux_slice(context), self._slices[pair[0]], self._slices[pair[1]]]

    def v(self, string: str, context: str = "total") -> np.ndarray:
        """Four-index integral block, matching ``QEDSAPT0Driver.v`` exactly.

        ``v(s0 s1 s2 s3) = sum_Q b(s0 s2)[Q] (x) b(s1 s3)[Q]``: the A-side pair
        is positions ``(0, 2)`` and the B-side pair is positions ``(1, 3)``.
        """
        if len(string) != 4:
            raise ValueError(f"v: string {string!r} does not have length 4")

        left = self.b(string[0] + string[2], context=context)
        right = self.b(string[1] + string[3], context=context)
        return oe.contract("PAC,PBD->ABCD", left, right, optimize="optimal")

    # -- validation ------------------------------------------------------

    @staticmethod
    def _validate_ao_tensor(B_ao, d_ao):
        B_ao = np.asarray(B_ao)
        if B_ao.ndim != 3 or B_ao.shape[1] != B_ao.shape[2]:
            raise ValueError(f"B_ao must have shape (naux, nbf, nbf); got {B_ao.shape}")
        d = np.asarray(d_ao)
        if d.shape != B_ao.shape[1:]:
            raise ValueError(f"d_ao shape {d.shape} does not match B_ao AO dimensions {B_ao.shape[1:]}")
