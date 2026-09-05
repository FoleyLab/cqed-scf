"""Architecture scaffold for QED-SAPT0."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple
import warnings
import opt_einsum as oe
import numpy as np
# Used by _resolve_df_aux_basis and by the psi4.core.clean() calls on the
# argument-validation error paths below, which previously raised NameError
# instead of the intended exception.
import psi4

from ..references import CQEDConfig
from .. import output

from .monomer import SAPTMonomer
from .results import QEDSAPT0Results
from .dse_df import PauliFierzDF


_VT_PART_KEYS = ("eri", "potential_A", "potential_B", "constant")
_OPERATOR_CONTEXTS = ("standard", "total", "cavity")

#: ``full_eri`` builds the dense nbf**4 AO tensor; ``df`` factorizes the
#: Pauli-Fierz two-electron integral as an augmented density-fitting expansion
#: in which the dipole self-energy is a single exact auxiliary row.  See
#: docs/qed_sapt0_formalism.tex.
_INTEGRAL_BACKENDS = ("full_eri", "df")

#: Fitting-basis roles, following Psi4's own SAPT practice.  Psi4 fits the
#: Coulomb-like terms (electrostatics, exchange, induction) with a JKFIT set
#: (DF_BASIS_ELST / DF_BASIS_SCF) and the MP2-like dispersion terms with a
#: RIFIT set (DF_BASIS_SAPT).  The distinction is not cosmetic: RIFIT is
#: optimized for correlation energies, and using it for electrostatics costs
#: roughly two orders of magnitude of accuracy, because Elst10 is a small
#: residual of large cancelling terms and the fitting error does not cancel
#: with anything.  Measured on water/He at cc-pVDZ, Elst10 error was 3.2e-5 Eh
#: with RIFIT versus 4.0e-7 Eh with JKFIT.
_DF_ROLES = ("scf", "corr")

#: Frame in which each monomer's CQED-SCF reference is solved.
#:
#: ``dimer`` (default, historical) solves each ghosted monomer where it sits in
#: the dimer.  ``monomer_com`` first translates the ghosted monomer so that its
#: own real-atom centre of mass is at the coordinate origin.
#:
#: This matters because CQED-SCF orbitals and orbital energies are *not*
#: translation invariant, even though the total energy and the density are: the
#: one-electron DSE term shifts every orbital by +lambda^2 T^2 / 2 while the
#: exchange-like N[D] term shifts only the occupied ones by -lambda^2 T^2.  A
#: monomer sitting a distance T from the origin therefore acquires a spurious
#: lambda^2 T^2 widening of its orbital-energy gaps, which propagates into the
#: dispersion denominators.  Solving each reference in an intrinsic frame
#: removes the dependence on where the dimer happens to sit.
#:
#: Only the *reference* moves.  Every interaction quantity -- overlap, ERIs,
#: dipole matrices, nuclear attraction -- is built in the shared dimer frame,
#: because monomers A and B use different intrinsic frames and the interaction
#: operator must live in one.  In particular this keeps ``d_A == d_B``, on which
#: the density-fitted backend depends.
_MONOMER_REFERENCE_FRAMES = ("dimer", "monomer_com")


def _read_only(array):
    """Mark a large shared integral view read-only.

    The AO integral tensors are views (of Psi4 buffers, or of one another when
    the cavity is disabled and total == standard).  Nothing should write to
    them, and a stray in-place update would silently couple the operator
    contexts -- the failure mode documented in
    docs/development/psi4_array_aliasing.md.  Making the views non-writeable
    turns that into an immediate error instead of a plausible wrong number.
    """
    array.flags.writeable = False
    return array


@dataclass
class QEDSAPT0Driver:
    """Future QED-SAPT0 driver.

    The driver is responsible for data orchestration, not component physics:

    1. Prepare dimer and monomer geometries.
    2. Run or attach monomer CQED-SCF references.
    3. Build full two-electron integral intermediates.
    4. Call SAPT component functions.
    5. Return a :class:`QEDSAPT0Results` object.
    """

    dimer_geometry: Any
    config: CQEDConfig
    monomer_A: Optional[SAPTMonomer] = None
    monomer_B: Optional[SAPTMonomer] = None
    monomer_definitions: Optional[Sequence[Any]] = None
    monomer_indices: Optional[Tuple[Sequence[int], Sequence[int]]] = None
    integral_backend: str = "full_eri"
    include_cavity_terms: bool = True
    df_aux_basis: Optional[str] = None
    df_scf_fitting_role: str = "JKFIT"
    df_corr_fitting_role: str = "RIFIT"
    monomer_reference_frame: str = "dimer"
    metadata: Dict[str, Any] = field(default_factory=dict)
    monomer_a: InitVar[Optional[SAPTMonomer]] = None
    monomer_b: InitVar[Optional[SAPTMonomer]] = None
    dimer: InitVar[Optional[SAPTMonomer]] = None

    def __post_init__(
        self,
        monomer_a: Optional[SAPTMonomer],
        monomer_b: Optional[SAPTMonomer],
        dimer: Optional[SAPTMonomer],
    ) -> None:
        if monomer_a is not None:
            if self.monomer_A is not None:
                raise ValueError("Specify only one of monomer_A or monomer_a.")
            warnings.warn(
                "monomer_a is deprecated; use monomer_A.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.monomer_A = monomer_a
        if monomer_b is not None:
            if self.monomer_B is not None:
                raise ValueError("Specify only one of monomer_B or monomer_b.")
            warnings.warn(
                "monomer_b is deprecated; use monomer_B.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.monomer_B = monomer_b
        if dimer is not None:
            warnings.warn(
                "Passing dimer as a SAPTMonomer is deprecated and ignored; "
                "QEDSAPT0Driver now uses dimer_geometry for dimer nuclear and AO-basis data.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.integral_backend == "no_cavity":
            warnings.warn(
                'integral_backend="no_cavity" is deprecated; use '
                "include_cavity_terms=False with an ordinary integral backend.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.integral_backend = "full_eri"
            self.include_cavity_terms = False

        if self.integral_backend not in _INTEGRAL_BACKENDS:
            allowed = ", ".join(repr(name) for name in _INTEGRAL_BACKENDS)
            raise ValueError(
                f"integral_backend must be one of {allowed}; got {self.integral_backend!r}"
            )

        if self.monomer_reference_frame not in _MONOMER_REFERENCE_FRAMES:
            allowed = ", ".join(repr(name) for name in _MONOMER_REFERENCE_FRAMES)
            raise ValueError(
                "monomer_reference_frame must be one of "
                f"{allowed}; got {self.monomer_reference_frame!r}"
            )

        self._pf_df: Dict[str, PauliFierzDF] = {}
        self._ghosted_molecules: Dict[str, Any] = {}
        self._frame_mints: Dict[str, Any] = {}

        self.metadata.setdefault("integral_backend", self.integral_backend)
        self.metadata.setdefault("include_cavity_terms", self.include_cavity_terms)
        self.metadata.setdefault("monomer_reference_frame", self.monomer_reference_frame)
        if self.integral_backend == "df":
            self.metadata.setdefault("df_scf_fitting_role", self.df_scf_fitting_role)
            self.metadata.setdefault("df_corr_fitting_role", self.df_corr_fitting_role)

    def prepare_geometries(self) -> Tuple[str, str, str]:
        """Build dimer and ghosted monomer geometry strings from a Psi4 dimer."""

        monomer_A_geometry = self.dimer_geometry.extract_subsets(1, 2)
        monomer_B_geometry = self.dimer_geometry.extract_subsets(2, 1)

        # Keep the dimer-frame ghosted molecules: every interaction integral is
        # built in this single shared frame regardless of where the monomer
        # references are solved.
        self._ghosted_molecules = {"A": monomer_A_geometry, "B": monomer_B_geometry}
        self._frame_mints = {}

        dimer_string = self.dimer_geometry.create_psi4_string_from_molecule()
        monomer_A_string = self._reference_frame_string(monomer_A_geometry, 1)
        monomer_B_string = self._reference_frame_string(monomer_B_geometry, 2)

        return dimer_string, monomer_A_string, monomer_B_string

    def _reference_frame_string(self, ghosted_molecule, subset_index: int) -> str:
        """Geometry string in which this monomer's CQED-SCF reference is solved."""

        if self.monomer_reference_frame == "dimer":
            return ghosted_molecule.create_psi4_string_from_molecule()

        # Translate so the monomer's own real-atom centre of mass is at the
        # origin.  Ghost centres move with it, so the ghosted basis and every
        # internal distance are unchanged -- only the origin-dependent dipole
        # and quadrupole operators see the shift.
        real_molecule = self.dimer_geometry.extract_subsets(subset_index)
        com = real_molecule.center_of_mass()
        shifted = ghosted_molecule.clone()
        shifted.translate(psi4.core.Vector3(-com[0], -com[1], -com[2]))
        return shifted.create_psi4_string_from_molecule()

    def _basis_name(self) -> str:
        basis = (self.config.psi4_options or {}).get("basis")
        if basis is None:
            basis = psi4.core.get_global_option("BASIS")
        return str(basis)

    def _dimer_frame_mints(self, side: str):
        """MintsHelper for one ghosted monomer, always in the shared dimer frame.

        When the references are solved in the dimer frame this is exactly the
        monomer's own MintsHelper (verified bitwise identical), so the default
        path is unchanged.
        """
        if side not in ("A", "B"):
            raise ValueError(f"side must be 'A' or 'B'; got {side!r}")

        if self.monomer_reference_frame == "dimer":
            return (self.monomer_A if side == "A" else self.monomer_B).mints

        if side not in self._frame_mints:
            basisset = self._dimer_frame_basisset(side)
            self._frame_mints[side] = psi4.core.MintsHelper(basisset)
        return self._frame_mints[side]

    def _dimer_frame_basisset(self, side: str = "A"):
        """Orbital basis for one ghosted monomer, in the shared dimer frame."""
        if self.monomer_reference_frame == "dimer":
            return (self.monomer_A if side == "A" else self.monomer_B).wfn.basisset()
        return psi4.core.BasisSet.build(
            self._ghosted_molecules[side], "BASIS", self._basis_name()
        )

    def _dimer_frame_molecule(self, side: str = "A"):
        """Ghosted monomer molecule in the shared dimer frame."""
        if self.monomer_reference_frame == "dimer":
            return (self.monomer_A if side == "A" else self.monomer_B).wfn.molecule()
        return self._ghosted_molecules[side]

    def prepare_monomers(self) -> Tuple[SAPTMonomer, SAPTMonomer]:
        """Prepare or retrieve monomer references."""

        if self.monomer_A is not None and self.monomer_B is not None:
            self._populate_dimer_nuclear_terms()
            self._populate_monomer_attributes()
            return self.monomer_A, self.monomer_B

        _, monomer_A_string, monomer_B_string = self.prepare_geometries()
        self._populate_dimer_nuclear_terms()

        if self.monomer_A is None:
            self.monomer_A = SAPTMonomer.from_cqed_scf(
                label="monomer_A",
                geometry=monomer_A_string,
                config=self.config,
            )
        if self.monomer_B is None:
            self.monomer_B = SAPTMonomer.from_cqed_scf(
                label="monomer_B",
                geometry=monomer_B_string,
                config=self.config,
            )

        self._populate_monomer_attributes()
        return self.monomer_A, self.monomer_B

    def _populate_dimer_nuclear_terms(self) -> None:
        """Populate dimer quantities that do not require a dimer SCF reference."""

        dimer_mu_nuc = self.dimer_geometry.nuclear_dipole()
        self.dimer_mu_nuc = np.array(
            [dimer_mu_nuc[0], dimer_mu_nuc[1], dimer_mu_nuc[2]]
        )
        self.E_nuc_dimer = self.dimer_geometry.nuclear_repulsion_energy()
        if self.config.debug:
            output.echo("Dimer nuclear dipole moment")
            output.echo(f"  {self.dimer_mu_nuc[0]:16.10f} {self.dimer_mu_nuc[1]:16.10f} {self.dimer_mu_nuc[2]:16.10f}")
            output.echo(f"Dimer nuclear repulsion energy {self.E_nuc_dimer:16.12f}")

    def _populate_monomer_attributes(self) -> None:
        """Cache monomer reference data used by the SAPT component formulas."""

        if self.monomer_A is None or self.monomer_B is None:
            raise RuntimeError("prepare_monomers requires both monomer references.")

        self.E_scf_A = self.monomer_A.energy_scf
        self.E_scf_B = self.monomer_B.energy_scf

        self.ndocc_A = self.monomer_A.ndocc
        self.nvirt_A = self.monomer_A.nvirt
        self.ndocc_B = self.monomer_B.ndocc
        self.nvirt_B = self.monomer_B.nvirt

        # currently assumes closed shell, nsocc = 0
        self.nsocc_A = 0
        self.nsocc_B = 0

        self.C_A = self.monomer_A.C
        self.C_B = self.monomer_B.C
        self.Co_A = self.monomer_A.Co
        self.Cv_A = self.monomer_A.Cv
        self.Co_B = self.monomer_B.Co
        self.Cv_B = self.monomer_B.Cv

        self.eps_A = self.monomer_A.eps
        self.eps_B = self.monomer_B.eps

        self.eps_canonical_A = self.monomer_A.eps_canonical
        self.eps_canonical_B = self.monomer_B.eps_canonical

        self.E_nuc_A = self.monomer_A.nuc_rep
        self.E_nuc_B = self.monomer_B.nuc_rep
        self.nuc_rep = self.E_nuc_dimer - self.E_nuc_A - self.E_nuc_B

        self.d_exp_el_A = self.monomer_A.d_exp_el
        self.d_exp_el_B = self.monomer_B.d_exp_el
        self.d_exp_A = self.monomer_A.d_exp
        self.d_exp_B = self.monomer_B.d_exp
        self.d_nuc_A = self.monomer_A.d_nuc
        self.d_nuc_B = self.monomer_B.d_nuc
        self.d_A = self.monomer_A.d_ao
        self.d_B = self.monomer_B.d_ao

        # When the references were solved in their own intrinsic frames, every
        # dipole-derived quantity above belongs to the wrong frame.  Rebuild all
        # of them in the shared dimer frame, from the (frame-independent)
        # monomer densities.
        if self.monomer_reference_frame != "dimer":
            self._rebase_dipole_data_to_dimer_frame()

        # Each monomer's own CQED-SCF was solved with this dipole matrix, so it
        # is the one its *internal* orbital Hessian must use.  Under the default
        # frame these are the shared dimer-frame matrices and every correction
        # below vanishes identically.
        self._d_intrinsic = {"A": self.monomer_A.d_ao, "B": self.monomer_B.d_ao}

        # Constant term in the coherent-state fluctuation product,
        # + <d_A><d_B>.
        self.d_exp_cross_AB = self.d_exp_A * self.d_exp_B

        assert np.isclose(self.d_exp_A, (self.d_exp_el_A + self.d_nuc_A))
        assert np.isclose(self.d_exp_B, (self.d_exp_el_B + self.d_nuc_B))

        # d_ao = lambda . mu_ao is a property of the shared dimer AO basis and
        # frame, not of the monomer, so the two ghosted monomer calculations must
        # produce the same matrix.  This is load-bearing: it makes the cavity
        # two-electron kernel d_A (x) d_B symmetric under (pq) <-> (rs), which is
        # what lets the density-fitted backend represent the whole dipole
        # self-energy as a single extra auxiliary row (see dse_df.PauliFierzDF
        # and docs/qed_sapt0_formalism.tex).  A mismatch would mean the monomer
        # bases or dipole origins have diverged, which breaks far more than
        # dispersion, so fail loudly here rather than silently downstream.
        if not np.allclose(self.d_A, self.d_B, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "monomer A and B dipole-projected AO matrices differ "
                f"(max|d_A - d_B| = {np.abs(self.d_A - self.d_B).max():.3e}). "
                "They must be identical in the shared dimer basis; check that "
                "both monomers were built from the same dimer geometry and frame."
            )

        electron_count_A = 2 * self.ndocc_A + self.nsocc_A
        electron_count_B = 2 * self.ndocc_B + self.nsocc_B
        self.vt_nuc_rep_standard = self.nuc_rep / (electron_count_A * electron_count_B)
        self.vt_nuc_rep_cavity = (
            (self.d_exp_el_A * self.d_exp_el_B) / (electron_count_A * electron_count_B)
            if self.include_cavity_terms 
            else 0.0
        )
        self.vt_nuc_rep = self.vt_nuc_rep_standard + self.vt_nuc_rep_cavity 

    def _rebase_dipole_data_to_dimer_frame(self) -> None:
        """Recompute every dipole-derived quantity in the shared dimer frame.

        Only ``C`` and ``eps`` are taken from the intrinsic-frame reference.
        The interaction operator must live in one common frame: monomers A and
        B use *different* intrinsic frames, so dipole matrices taken from those
        would differ by ``(z_A - z_B) lambda S``, which would both misstate the
        interaction and break the ``d_A == d_B`` identity the density-fitted
        backend relies on.

        The MO coefficients transfer without modification: a rigid translation
        moves the basis functions with the molecule, so a coefficient vector
        describes the correspondingly translated orbital.
        """
        lambda_vector = np.asarray(self.config.lambda_vector)

        for side, monomer in (("A", self.monomer_A), ("B", self.monomer_B)):
            mints = self._dimer_frame_mints(side)
            molecule = self._ghosted_molecules[side]

            mu_ao = np.asarray(mints.ao_dipole())
            d_ao = sum(lambda_vector[i] * mu_ao[i] for i in range(3))

            # Densities are frame independent given the coefficients.
            Co = monomer.Co
            D = oe.contract("pi,qi->pq", Co, Co, optimize="optimal")

            mu_el = np.array(
                [2.0 * oe.contract("pq,pq->", mu_ao[i], D, optimize="optimal") for i in range(3)]
            )
            nuclear_dipole = molecule.nuclear_dipole()
            mu_nuc = np.array([nuclear_dipole[0], nuclear_dipole[1], nuclear_dipole[2]])

            d_exp_el = float(np.dot(lambda_vector, mu_el))
            d_nuc = float(np.dot(lambda_vector, mu_nuc))

            setattr(self, f"d_{side}", d_ao)
            setattr(self, f"d_exp_el_{side}", d_exp_el)
            setattr(self, f"d_nuc_{side}", d_nuc)
            setattr(self, f"d_exp_{side}", d_exp_el + d_nuc)

    def build_orbitals(self) -> Any:
        """Build orbital intermediates needed for QED-SAPT0 components.
        """
        # organize orbitals into a dictionary for convenient access in component functions, using the same string labels as the original SAPT0 implementation where possible (a, b, r, s)
        self.orbitals = {'a' : self.Co_A,
                         'r': self.Cv_A,
                         'b': self.Co_B,
                         's': self.Cv_B
                         }
        

        
    def build_slices(self) -> Any:
        """Build slice objects for occupied and virtual orbital subspaces of each monomer.
        """
        self.slices = {'a' : slice(0, self.ndocc_A),
                       'r': slice(self.ndocc_A, None),
                       'b': slice(0, self.ndocc_B),
                       's': slice(self.ndocc_B, None)
                       }
        
    def build_sizes(self) -> Any:
        """Build integers for number of occupied and virtual orbitals of each monomer.
        """
        self.sizes = {'a' : self.ndocc_A,
                      'r': self.nvirt_A,
                      'b': self.ndocc_B,
                      's': self.nvirt_B
                      }

        

    def build_integrals(self, monomers: Optional[Tuple[SAPTMonomer, SAPTMonomer]] = None) -> Any:
        """Build all integral intermediates needed for QED-SAPT0 components."""
        if monomers is not None:
            warnings.warn(
                "Passing monomers to build_integrals is deprecated; the driver "
                "uses its prepared monomer_A and monomer_B references.",
                DeprecationWarning,
                stacklevel=2,
            )
            if len(monomers) != 2:
                raise ValueError("build_integrals expects only (monomer_A, monomer_B).")
            self.monomer_A, self.monomer_B = monomers
            self._populate_dimer_nuclear_terms()
            self._populate_monomer_attributes()

        if self.monomer_A is None or self.monomer_B is None:
            self.prepare_monomers()

        # build orbitals using monomer A and monomer B SCF results, which may be None if the monomer SCF references were not run with orbital storage enabled
        self.build_orbitals()

        # build slices for occupied and virtual orbital subspaces of each monomer
        self.build_slices()

        # build sizes for number of occupied and virtual orbitals of each monomer
        self.build_sizes()

        # Monomer A and B are ghosted calculations in the same dimer basis, so
        # either MintsHelper can define the shared AO integral environment --
        # provided it is the *dimer-frame* one.  With intrinsic-frame references
        # the monomers' own MintsHelpers sit in different frames and must not be
        # used for interaction integrals.
        shared_mints = self._dimer_frame_mints("A")
        monomer_A_mints = self._dimer_frame_mints("A")
        monomer_B_mints = self._dimer_frame_mints("B")

        # overlap of dimer in AO basis
        self.S_dimer = np.asarray(shared_mints.ao_overlap())

        # overlap transformed on bra with monomer A and ket with monomer B
        self.S_AB = oe.contract("uI,vJ,uv->IJ", self.C_A, self.C_B, self.S_dimer)

        if self.integral_backend == "df":
            # Density-fitted backend: the Pauli-Fierz two-electron integral is
            # an augmented DF expansion whose last auxiliary row is exactly the
            # dipole matrix (docs/qed_sapt0_formalism.tex).  No nbf**4 AO tensor
            # is ever formed, so the dense attributes are left unset and
            # _eri_for_context() refuses rather than returning something
            # plausible.
            self.I_dimer_standard = None
            self.I_dimer_cavity = None
            self.I_dimer = None
            self._pf_df = {}
            self.metadata["df_aux_basis"] = self._resolve_df_aux_basis()
            # Build the Coulomb-fitted tensor eagerly so a bad fitting-basis
            # request fails inside build_integrals rather than mid-component.
            # The correlation-fitted tensor is built on first use, so a caller
            # that only wants first-order terms never pays for it.
            self.metadata["df_naux_scf"] = self._pauli_fierz_df("scf").naux
        else:
            self._build_dense_eri_tensors(shared_mints)

        # build the one-electron potential integrals for monomer A and monomer B
        # the V_A and V_B terms are scaled by 1 / N_A and 1 / N_B in the v_tilde build
        self.V_A = np.asarray(monomer_A_mints.ao_potential())
        self.V_B = np.asarray(monomer_B_mints.ao_potential())

        self.V_A_standard = self.V_A.copy()
        self.V_B_standard = self.V_B.copy()
        self.V_A_cavity = np.zeros_like(self.V_A)
        self.V_B_cavity = np.zeros_like(self.V_B)
        if self.include_cavity_terms: 
            self.V_A_cavity = -self.d_exp_el_A * self.d_B
            self.V_B_cavity = -self.d_exp_el_B * self.d_A
            self.V_A += self.V_A_cavity
            self.V_B += self.V_B_cavity

        # potential integrals
        self.V_A_BB = oe.contract("uI,vJ,uv->IJ", self.C_B, self.C_B, self.V_A, optimize="optimal")
        self.V_A_AB = oe.contract("uI,vJ,uv->IJ", self.C_A, self.C_B, self.V_A, optimize="optimal")
        self.V_B_AA = oe.contract("uI,vJ,uv->IJ", self.C_A, self.C_A, self.V_B, optimize="optimal")
        self.V_B_AB = oe.contract("uI,vJ,uv->IJ", self.C_A, self.C_B, self.V_B, optimize="optimal")

    def _resolve_df_aux_basis(self) -> str:
        """Pick the fitting-basis key for the DF backend.

        Explicit ``df_aux_basis`` wins; otherwise follow the orbital basis the
        monomer references were actually run with, and fall back to Psi4's
        current global option only if the config does not say.
        """
        if self.df_aux_basis is not None:
            return self.df_aux_basis
        basis = (self.config.psi4_options or {}).get("basis")
        if basis is not None:
            return str(basis)
        return psi4.core.get_global_option("BASIS")

    def _pauli_fierz_df(self, df_role: str = "scf") -> PauliFierzDF:
        """Return the density-fitted integral tensor for a fitting role.

        ``scf`` fits with a JKFIT set and serves the Coulomb-like terms;
        ``corr`` fits with a RIFIT set and serves the MP2-like dispersion
        terms.  See :data:`_DF_ROLES` for why the distinction matters.
        """
        if df_role not in _DF_ROLES:
            allowed = ", ".join(repr(name) for name in _DF_ROLES)
            raise ValueError(f"df_role must be one of {allowed}; got {df_role!r}")
        if self.integral_backend != "df":
            raise RuntimeError(
                'density-fitted tensors are only built for integral_backend="df"; '
                f"this driver uses {self.integral_backend!r}."
            )
        if getattr(self, "orbitals", None) is None:
            raise RuntimeError(
                "The density-fitted backend is not built. Call build_integrals() "
                'before requesting integrals with integral_backend="df".'
            )

        if df_role not in self._pf_df:
            fitting_role = (
                self.df_scf_fitting_role if df_role == "scf" else self.df_corr_fitting_role
            )
            self._pf_df[df_role] = PauliFierzDF.from_driver(
                self,
                aux_basis_name=self._resolve_df_aux_basis(),
                fitting_role=fitting_role,
            )
            self.metadata[f"df_naux_{df_role}"] = self._pf_df[df_role].naux
        return self._pf_df[df_role]

    def _build_dense_eri_tensors(self, shared_mints) -> None:
        """Build the dense nbf**4 AO tensors used by the ``full_eri`` backend."""

        # 1. Get the ERI array directly (try to avoid copying if ao_eri() allows)
        I_dimer_standard = np.asarray(shared_mints.ao_eri())

        # 2. The coherent-state fluctuation product contributes a separable
        #    rank-one kernel, (pq|rs) -> d_A[p, q] * d_B[r, s].  Broadcasting
        #    d_A (p, q) -> (p, q, 1, 1) against d_B (r, s) -> (1, 1, r, s)
        #    materializes it in the same chemists' layout as the ordinary ERIs.
        #
        #    When the cavity is disabled there is no cavity tensor and the total
        #    equals the standard tensor, so neither the N^4 zero array nor the
        #    N^4 copy is allocated.  ``None`` marks the inactive contribution;
        #    v() and the diagnostics handle it explicitly rather than paying
        #    nbf**4 doubles to represent zero.
        if self.include_cavity_terms:
            I_dimer_cavity = self.d_A[:, :, np.newaxis, np.newaxis] * self.d_B[np.newaxis, np.newaxis, :, :]
            I_dimer = I_dimer_standard + I_dimer_cavity
        else:
            I_dimer_cavity = None
            I_dimer = I_dimer_standard

        # 3. Swap axes (creates a view, zero memory overhead) so that
        #    I_dimer[p, q, r, s] holds the chemists' integral (p r | q s).
        self.I_dimer_standard = _read_only(I_dimer_standard.swapaxes(1, 2))
        self.I_dimer_cavity = (
            None if I_dimer_cavity is None else _read_only(I_dimer_cavity.swapaxes(1, 2))
        )
        # NOTE: with the cavity disabled this is deliberately the *same* view as
        # I_dimer_standard.  Both are marked read-only so the sharing cannot turn
        # into the silent aliasing defect documented in
        # docs/development/psi4_array_aliasing.md.
        self.I_dimer = _read_only(I_dimer.swapaxes(1, 2))


    def compute_components(self, monomers, integrals) -> QEDSAPT0Results:
        """Call future component functions and collect a result object."""

        raise NotImplementedError(
            "QED-SAPT0 component physics is not implemented yet. "
            "Future code will call compute_elst10, compute_exch10, "
            "compute_ind20, compute_disp20, and compute_qed_dse_cross."
        )
    
    def _validate_operator_context(self, context: str) -> str:
        if context not in _OPERATOR_CONTEXTS:
            allowed = ", ".join(_OPERATOR_CONTEXTS)
            raise ValueError(f"operator context must be one of {allowed}; got {context!r}")
        return context

    def _eri_for_context(self, context: str):
        context = self._validate_operator_context(context)
        if self.integral_backend == "df":
            # There is no dense AO tensor to return.  Refuse loudly: returning
            # None here would make v() hand back zeros, which is exactly the
            # kind of plausible-but-wrong result this codebase has been bitten
            # by before (docs/development/psi4_array_aliasing.md).
            raise RuntimeError(
                'the "df" integral backend has no dense AO ERI tensor; '
                "use v() or PauliFierzDF.b() instead of _eri_for_context()."
            )
        if context == "standard":
            return self.I_dimer_standard
        if context == "cavity":
            return self.I_dimer_cavity
        return self.I_dimer

    def _potential_for_context(self, side: str, context: str):
        context = self._validate_operator_context(context)
        if side == "A":
            if context == "standard":
                return self.V_A_standard
            if context == "cavity":
                return self.V_A_cavity
            return self.V_A
        if side == "B":
            if context == "standard":
                return self.V_B_standard
            if context == "cavity":
                return self.V_B_cavity
            return self.V_B

        psi4.core.clean()
        raise Exception("potential: side %s is not A or B" % side)

    def _vt_nuc_rep_for_context(self, context: str):
        context = self._validate_operator_context(context)
        if context == "standard":
            return self.vt_nuc_rep_standard
        if context == "cavity":
            return self.vt_nuc_rep_cavity
        return self.vt_nuc_rep

    def _zero_vt_parts(self):
        zero = np.array([0]).reshape(1, 1, 1, 1)
        return {key: zero.copy() for key in _VT_PART_KEYS}

    def _sum_vt_parts(self, parts):
        total = parts["eri"].copy()
        total += parts["potential_A"]
        total += parts["potential_B"]
        total += parts["constant"]
        return total

    def v(self, string, context: str = "total", df_role: str = "scf", frame: Optional[str] = None):
        """
        Builds two-electron integrals dressed with monomerA - monomerB dipole integrals
        transformed with appropriate MO coefficients

        ``df_role`` selects the fitting basis under the ``df`` backend: ``scf``
        (JKFIT) for the Coulomb-like terms, ``corr`` (RIFIT) for the MP2-like
        dispersion terms.  It is ignored by the exact ``full_eri`` backend.

        ``frame`` selects the frame of the cavity (dipole self-energy)
        contribution.  Leave it ``None`` for interaction blocks; pass the
        monomer label for a block whose four indices all belong to one monomer,
        so that its cavity operator matches the reference its orbital energies
        came from.  Under ``monomer_reference_frame="dimer"`` this is a no-op.
        """
        if frame is not None:
            if frame not in ("A", "B"):
                raise ValueError(f"frame must be 'A', 'B' or None; got {frame!r}")
            # When the references were solved in the dimer frame the intrinsic
            # and shared operators are the same matrix, so short-circuit rather
            # than subtracting and re-adding it: (a - x) + x is not bitwise a.
            if (
                self.monomer_reference_frame == "dimer"
                or context == "standard"
                or not self.include_cavity_terms
            ):
                if context == "cavity":
                    return self._cavity_v(string, frame)
                return self.v(string, context=context, df_role=df_role)
            if context == "cavity":
                return self._cavity_v(string, frame)
            shared = self.v(string, context=context, df_role=df_role)
            return shared - self._cavity_v(string, None) + self._cavity_v(string, frame)

        if len(string) != 4:
            psi4.core.clean()
            raise Exception("v: string %s does not have length 4" % string)
        if self.integral_backend == "df":
            return self._pauli_fierz_df(df_role).v(string, context=context)

        I_dimer = self._eri_for_context(context)

        # An inactive cavity contributes exactly zero.  Return that directly
        # rather than transforming an nbf**4 array of zeros.
        if I_dimer is None:
            return np.zeros(tuple(self.orbitals[label].shape[1] for label in string))

        # ERI's from mints are in chemist's notation (pq|rs), but we want to access them in physicist's notation (pr|qs)
        # so we need to swap the middle two indices
        V = oe.contract("pA,pqrs->Aqrs", self.orbitals[string[0]], I_dimer, optimize="optimal")
        V = oe.contract("qB,Aqrs->ABrs", self.orbitals[string[1]], V, optimize="optimal")
        V = oe.contract("rC,ABrs->ABCs", self.orbitals[string[2]], V, optimize="optimal")
        V = oe.contract("sD,ABCs->ABCD", self.orbitals[string[3]], V, optimize="optimal")
        return V
    
    def _cavity_mo_pair(self, pair: str, frame: Optional[str] = None):
        """``C_x^T d C_y`` for the rank-one cavity kernel.

        ``frame=None`` uses the shared dimer-frame dipole matrix, which is what
        every *interaction* quantity needs.  ``frame="A"`` / ``"B"`` uses that
        monomer's own intrinsic-frame matrix, which is what its *internal*
        orbital Hessian needs -- the CPHF denominators come from that monomer's
        orbital energies, and mixing them with a differently-framed dipole
        matrix breaks the cancellation that makes induction origin independent.
        """
        d_ao = self.d_A if frame is None else self._d_intrinsic[frame]
        return self.orbitals[pair[0]].T @ d_ao @ self.orbitals[pair[1]]

    def _cavity_v(self, string: str, frame: Optional[str] = None):
        """Rank-one cavity block, never materialized as an AO tensor."""
        if not self.include_cavity_terms:
            return np.zeros(tuple(self.orbitals[label].shape[1] for label in string))
        left = self._cavity_mo_pair(string[0] + string[2], frame)
        right = self._cavity_mo_pair(string[1] + string[3], frame)
        return oe.contract("AC,BD->ABCD", left, right, optimize="optimal")

    def s(self, string):
        # Grap appropriate overlap integrals 
        if len(string) != 2:
            psi4.core.clean()
            raise Exception("s: string %s does not have length 2" % string)
        
        for alpha in 'ijab':
            if (alpha in string) and (self.sizes[alpha] == 0):
                return np.array([0]).reshape(1,1)
            
        s1 = string[0]
        s2 = string[1]

        # compute on the fly
        return (self.orbitals[s1].T).dot(self.S_dimer).dot(self.orbitals[s2])
    
    def eps(self, string, dim=1):
        if len(string) != 1:
            psi4.core.clean()
            raise Exception("eps: string %s does not have length 1" % string)
        
        shape = (-1,) + tuple([1] * (dim - 1))

        if (string=='i') or (string=='a') or (string=='r'):
            return self.eps_A[self.slices[string]].reshape(shape)
        
        elif (string=='j') or (string=='b') or (string=='s'):
            return self.eps_B[self.slices[string]].reshape(shape)
        
        else:
            psi4.core.clean()
            raise Exception("eps: string %s does not have valid monomer label" % string)
        
    def eps_canonical(self, string, dim=1):
        if len(string) != 1:
            psi4.core.clean()
            raise Exception("eps: string %s does not have length 1" % string)
        
        shape = (-1,) + tuple([1] * (dim - 1))

        if (string=='i') or (string=='a') or (string=='r'):
            return self.eps_canonical_A[self.slices[string]].reshape(shape)
        
        elif (string=='j') or (string=='b') or (string=='s'):
            return self.eps_canonical_B[self.slices[string]].reshape(shape)
        
        else:
            psi4.core.clean()
            raise Exception("eps: string %s does not have valid monomer label" % string)

    

    def potential(self, string, side, context: str = "total"):
        """
        Grab one-electron potential integrals for monomer X.

        In the cavity context these are dressed by dipole integrals scaled by
        electronic coherent-state expectation values from the fluctuation
        residual.
        """
        if len(string) != 2:
            psi4.core.clean()
            raise Exception("potential: string %s does not have length 2" % string)
        
        s1 = string[0]
        s2 = string[1]
        potential = self._potential_for_context(side, context)

        return (self.orbitals[s1].T).dot(potential).dot(self.orbitals[s2])
        

    def vt_parts(self, string, context: str = "total", df_role: str = "scf"):
        if len(string)!=4:
            psi4.core.clean()
            raise Exception('Compute tilde{v}: string %s does not have 4 elements' % string)
        
        for alpha in 'ijab':
            if (alpha in string) and (self.sizes[alpha] == 0):
                return self._zero_vt_parts()
            
        # grab left and right strings
        s_left = string[0] + string[2]
        s_right = string[1] + string[3]

        # ERI term
        eri = self.v(string, context=context, df_role=df_role)

        # potential A
        S_A = self.s(s_left)
        V_A = self.potential(s_right, 'A', context=context) / (2 * self.ndocc_A + self.nsocc_A)
        potential_A = oe.contract("ik,jl->ijkl", S_A, V_A)

        # potential B
        S_B = self.s(s_right)
        V_B = self.potential(s_left, 'B', context=context) / (2 * self.ndocc_B + self.nsocc_B)
        potential_B = np.einsum('ik,jl->ijkl', V_B, S_B)

        # constant - scaling by 1/N_A and 1/N_B already happened in prepare_monomers
        constant = np.einsum("ik,jl->ijkl", S_A, S_B) * self._vt_nuc_rep_for_context(context)

        return {
            "eri": eri,
            "potential_A": potential_A,
            "potential_B": potential_B,
            "constant": constant,
        }

    def vt(self, string, context: str = "total", df_role: str = "scf"):
        return self._sum_vt_parts(
            self.vt_parts(string, context=context, df_role=df_role)
        )

    def vt_partitions(self, string):
        standard = self.vt_parts(string, context="standard")
        total = self.vt_parts(string, context="total")
        cavity = {key: total[key] - standard[key] for key in _VT_PART_KEYS}

        partitions = {
            "standard": dict(standard),
            "total": dict(total),
            "cavity": cavity,
        }
        for context in _OPERATOR_CONTEXTS:
            partitions[context]["total"] = self._sum_vt_parts(partitions[context])

        return partitions

    def _contract_vt_array(self, string, array, contraction):
        if callable(contraction):
            return float(contraction(array))
        if contraction is None or contraction in {"einsum", "sapt"}:
            return float(np.einsum(f"{string}->", array))
        if contraction == "sum":
            return float(np.sum(array))
        return float(np.einsum(contraction, array))

    def contract_vt_parts(self, string, contraction=None, prefactor: float = 1.0):
        partitions = self.vt_partitions(string)
        return {
            context: {
                key: prefactor * self._contract_vt_array(string, value, contraction)
                for key, value in parts.items()
            }
            for context, parts in partitions.items()
        }

    def _diagnostic_scalar_summary(self):
        scalars = {
            "include_cavity_terms": self.include_cavity_terms,
            "integral_backend": self.integral_backend,
            "d_exp_A": self.d_exp_A,
            "d_exp_el_A": self.d_exp_el_A,
            "d_nuc_A": self.d_nuc_A,
            "d_exp_A_residual": self.d_exp_A - self.d_exp_el_A - self.d_nuc_A,
            "d_exp_B": self.d_exp_B,
            "d_exp_el_B": self.d_exp_el_B,
            "d_nuc_B": self.d_nuc_B,
            "d_exp_B_residual": self.d_exp_B - self.d_exp_el_B - self.d_nuc_B,
            "d_exp_A_times_d_exp_B": self.d_exp_A * self.d_exp_B,
            "d_exp_el_A_times_d_exp_el_B": self.d_exp_el_A * self.d_exp_el_B,
            "d_nuc_A_times_d_nuc_B": self.d_nuc_A * self.d_nuc_B,
        }

        if hasattr(self, "I_dimer_cavity"):
            scalars["I_dimer_cavity_norm"] = self._cavity_eri_norm()
        if hasattr(self, "V_A_cavity"):
            scalars["V_A_cavity_norm"] = np.linalg.norm(self.V_A_cavity)
        if hasattr(self, "V_B_cavity"):
            scalars["V_B_cavity_norm"] = np.linalg.norm(self.V_B_cavity)
        if hasattr(self, "vt_nuc_rep_cavity"):
            scalars["vt_nuc_rep_cavity"] = self.vt_nuc_rep_cavity

        return scalars

    def _cavity_eri_norm(self) -> float:
        """Frobenius norm of the cavity two-electron tensor.

        The cavity kernel is the rank-one outer product d_A (x) d_B, so its
        Frobenius norm is ||d_A|| ||d_B|| exactly.  Using that identity keeps
        the diagnostic meaningful under the "df" backend, where the tensor is
        deliberately never materialized, and avoids touching an nbf**4 array
        under "full_eri".
        """
        if not self.include_cavity_terms:
            return 0.0
        if getattr(self, "d_A", None) is None or getattr(self, "d_B", None) is None:
            return 0.0
        return float(np.linalg.norm(self.d_A) * np.linalg.norm(self.d_B))

    def _diagnostic_vt_summary(self):
        prefactor = 4.0
        elst100 = self.contract_vt_parts("abab", prefactor=prefactor)
        check_compute = float(self.compute_Elst100() - elst100["total"]["total"])
        elst100.update(
            {
                "label": "Elst100",
                "prefactor": prefactor,
                "checks": {
                    "standard_plus_cavity_minus_total": float(
                        elst100["standard"]["total"]
                        + elst100["cavity"]["total"]
                        - elst100["total"]["total"]
                    ),
                    "cavity_total_abs": float(abs(elst100["cavity"]["total"])),
                    "compute_Elst100_minus_diagnostic_total": check_compute,
                },
            }
        )
        return {"abab": elst100}

    def _print_scalar_diagnostics(self, scalars):
        def _row(label, value):
            if isinstance(value, (bool, str)):
                return [label, str(value)]
            return [label, f"{float(value):18.10e}"]

        output.echo("QED-SAPT0 cavity diagnostics")
        output.echo()
        output.echo("Scalar dipole checks")
        output.echo("--------------------")

        scalar_keys = (
            ("d_exp_A", "d_exp_A"),
            ("d_exp_el_A", "d_exp_el_A"),
            ("d_nuc_A", "d_nuc_A"),
            ("d_exp_A_residual", "d_exp_A_residual"),
            ("d_exp_B", "d_exp_B"),
            ("d_exp_el_B", "d_exp_el_B"),
            ("d_nuc_B", "d_nuc_B"),
            ("d_exp_B_residual", "d_exp_B_residual"),
            ("d_exp_A_times_d_exp_B", "d_exp_A * d_exp_B"),
            ("d_exp_el_A_times_d_exp_el_B", "d_exp_el_A * d_exp_el_B"),
            ("d_nuc_A_times_d_nuc_B", "d_nuc_A * d_nuc_B"),
        )
        output.table(
            ["Label", "Value"],
            [_row(key, scalars[key]) for key, label in scalar_keys],
            [36, 22],
        )

        norm_keys = (
            ("I_dimer_cavity_norm", "||I_dimer_cavity||"),
            ("V_A_cavity_norm", "||V_A_cavity||"),
            ("V_B_cavity_norm", "||V_B_cavity||"),
            ("vt_nuc_rep_cavity", "vt_nuc_rep_cavity"),
        )
        if any(key in scalars for key, _ in norm_keys):
            output.echo()
            output.echo("Operator norms")
            output.echo("--------------")
            output.table(
                ["Label", "Value"],
                [
                    _row(label, scalars[key])
                    for key, label in norm_keys
                    if key in scalars
                ],
                [36, 22],
            )

    def _print_vt_diagnostics(self, vt_summary):
        vt_abab = vt_summary["abab"]
        output.echo()
        output.echo("QED-SAPT0 operator diagnostics")
        output.echo(
            f"Component: {vt_abab['label']}, tensor: vt(\"abab\"), "
            f"prefactor: {vt_abab['prefactor']:.1f}"
        )

        vt_rows = []
        for context in ("standard", "cavity", "total"):
            for piece in ("eri", "potential_A", "potential_B", "constant", "total"):
                vt_rows.append(
                    [context, piece, f"{vt_abab[context][piece]:18.10f}"]
                )
        output.table(["context", "piece", "value / Eh"], vt_rows, [10, 14, 22])

        output.echo()
        output.echo("Checks")
        output.echo("------")
        checks = vt_abab["checks"]
        check_rows = [
            ["standard + cavity - total", f"{checks['standard_plus_cavity_minus_total']:18.10e}"],
            ["abs(cavity total)", f"{checks['cavity_total_abs']:18.10e}"],
            ["compute_Elst100 - diagnostic", f"{checks['compute_Elst100_minus_diagnostic_total']:18.10e}"],
        ]
        output.table(["Check", "Value"], check_rows, [32, 22])

    def diagnostic_summary(self, print_output: Optional[bool] = None):
        print_output = self.config.debug if print_output is None else print_output
        vt_summary = self._diagnostic_vt_summary()
        summary = {
            "scalars": self._diagnostic_scalar_summary(),
            "vt": vt_summary,
            "Elst100": vt_summary["abab"],
        }

        if print_output:
            self._print_scalar_diagnostics(summary["scalars"])
            self._print_vt_diagnostics(summary["vt"])

        return summary

    def print_diagnostics(self):
        return self.diagnostic_summary(print_output=True)
    
    def chf(self, monomer, ind=False):
        if monomer not in ['A', 'B']:
            psi4.core.clean()
            raise Exception("chf: monomer %s is not A or B" % monomer)
        
        if monomer == 'A':
            w_n = 2 * oe.contract('saba->bs', self.v('saba'), optimize="optimal")
            w_n += self.V_A_BB[self.slices['b'], self.slices['s']]
            eps_ov = (self.eps('b', dim=2) - self.eps('s'))

            # set terms
            v_term1 = 'sbbs'
            v_term2 = 'sbsb'
            no, nv = self.ndocc_B, self.nvirt_B
            # The matrix below is monomer B's own orbital Hessian, so its cavity
            # operator belongs in B's reference frame.
            hessian_frame = 'B'

        if monomer == 'B':
            w_n = 2 * oe.contract('rbab->ar', self.v('rbab'), optimize="optimal")
            w_n += self.V_B_AA[self.slices['a'], self.slices['r']]
            eps_ov = (self.eps('a', dim=2) - self.eps('r'))
            v_term1 = 'raar'
            v_term2 = 'rara'
            no, nv = self.ndocc_A, self.nvirt_A
            hessian_frame = 'A'

        # form A matrix (LHS)
        voov = self.v(v_term1, frame=hessian_frame)
        v_vOov = 2 * voov - self.v(v_term2, frame=hessian_frame).swapaxes(2,3)
        v_ooaa = voov.swapaxes(1,3)
        v_vVoO = 2 * v_ooaa- v_ooaa.swapaxes(2,3)
        # A_ovOV = np.einsum('vOoV->ovOV', v_vOoV + v_vVoO.swapaxes(1, 3))
        #A_ovOV = oe.contract('vOov->ovOV', v_vOov + v_vVoO.swapaxes(1,3), optimize="optimal")
        A_ovOV = oe.contract('vOoV->ovOV',  v_vOov + v_vVoO.swapaxes(1, 3),optimize="optimal")
        # copy back to C contibous
        nov = nv * no 
        A_ovOV = A_ovOV.reshape(nov, nov).copy(order='C')
        A_ovOV[np.diag_indices_from(A_ovOV)] -= eps_ov.ravel()

        # call DGESV, need flat ov array 
        B_ov = -1 * w_n.ravel()
        t = np.linalg.solve(A_ovOV, B_ov)
        t = t.reshape(no, nv).T

        if ind:
            e20_ind = 2 * oe.contract('vo,ov->', t, w_n, optimize="optimal")
            return t, e20_ind
        
        else:
            return t

    def diagnostic_chf_rhs(self, monomer, context: str = "total"):
        """Return the dense CPHF RHS in the occupied-virtual convention.

        ``monomer`` follows :meth:`chf`: ``"B"`` means monomer A responds to
        the field from B and the returned shape is ``(nocc_A, nvir_A)``;
        ``"A"`` means monomer B responds to the field from A.
        """
        self._validate_operator_context(context)
        if monomer not in ['A', 'B']:
            psi4.core.clean()
            raise Exception("diagnostic_chf_rhs: monomer %s is not A or B" % monomer)

        if monomer == 'A':
            w_n = 2 * oe.contract('saba->bs', self.v('saba', context=context), optimize="optimal")
            w_n += self.potential("bs", "A", context=context)
        else:
            w_n = 2 * oe.contract('rbab->ar', self.v('rbab', context=context), optimize="optimal")
            w_n += self.potential("ar", "B", context=context)

        return np.ascontiguousarray(w_n)

    def diagnostic_chf_matrix(
        self,
        monomer,
        context: str = "total",
        include_orbital_diagonal: bool = True,
    ):
        """Return the dense occupied-virtual CPHF matrix used by :meth:`chf`.

        The matrix is returned in the dense solver's flattened ``(occ, vir)``
        ordering.  With ``include_orbital_diagonal=False`` only the two-electron
        response block for the selected operator context is returned.
        """
        context = self._validate_operator_context(context)
        if monomer not in ['A', 'B']:
            psi4.core.clean()
            raise Exception("diagnostic_chf_matrix: monomer %s is not A or B" % monomer)

        if monomer == 'A':
            eps_ov = (self.eps('b', dim=2) - self.eps('s'))
            v_term1 = 'sbbs'
            v_term2 = 'sbsb'
            no, nv = self.ndocc_B, self.nvirt_B
        else:
            eps_ov = (self.eps('a', dim=2) - self.eps('r'))
            v_term1 = 'raar'
            v_term2 = 'rara'
            no, nv = self.ndocc_A, self.nvirt_A

        voov = self.v(v_term1, context=context)
        v_vOov = 2 * voov - self.v(v_term2, context=context).swapaxes(2, 3)
        v_ooaa = voov.swapaxes(1, 3)
        v_vVoO = 2 * v_ooaa - v_ooaa.swapaxes(2, 3)
        A_ovOV = oe.contract(
            'vOoV->ovOV',
            v_vOov + v_vVoO.swapaxes(1, 3),
            optimize="optimal",
        )
        A_ovOV = A_ovOV.reshape(no * nv, no * nv).copy(order='C')

        if include_orbital_diagonal:
            A_ovOV[np.diag_indices_from(A_ovOV)] -= eps_ov.ravel()

        return A_ovOV

    def diagnostic_chf_hessian_action(
        self,
        monomer,
        amplitude,
        context: str = "total",
        include_orbital_diagonal: bool = True,
        psi4_convention: bool = True,
    ):
        """Apply the dense CPHF matrix to a trial occupied-virtual amplitude.

        By default this accepts and returns Psi4-style ``(nocc, nvir)`` arrays.
        Set ``psi4_convention=False`` to use the dense energy-expression
        convention ``(nvir, nocc)``.
        """
        A_ovOV = self.diagnostic_chf_matrix(
            monomer,
            context=context,
            include_orbital_diagonal=include_orbital_diagonal,
        )
        trial = np.asarray(amplitude, dtype=float)
        if psi4_convention:
            flat = trial.ravel()
            no, nv = trial.shape
            action = A_ovOV @ flat
            return np.ascontiguousarray(action.reshape(no, nv))

        flat = trial.T.ravel()
        nv, no = trial.shape
        action = A_ovOV @ flat
        return np.ascontiguousarray(action.reshape(no, nv).T)

    def compute_Elst100(self):
        return 4 * oe.contract('abab->', self.vt('abab'), optimize="optimal")
    
    def compute_Exch100(self):
        vt_abba = self.vt('abba')
        vt_abaa = self.vt('abaa')
        vt_abbb = self.vt('abbb')
        vt_abab = self.vt('abab')
        s_ab = self.s('ab')

        Exch100 = oe.contract("abba", vt_abba, optimize="optimal")

        _tmp = 2 * vt_abaa - vt_abaa.swapaxes(2,3)
        Exch100 += oe.contract('Ab,abaA', s_ab, _tmp, optimize="optimal")

        _tmp = 2 * vt_abbb - vt_abbb.swapaxes(2,3)
        Exch100 += oe.contract('Ba,abBb', s_ab.T, _tmp, optimize="optimal")

        Exch100 -= 2 * oe.contract('Ab,BA,abaB', s_ab, s_ab.T, vt_abab, optimize="optimal")
        Exch100 -= 2 * oe.contract('AB,Ba,abAb', s_ab, s_ab.T, vt_abab, optimize="optimal")
        Exch100 += oe.contract('Ab,Ba, abAB', s_ab, s_ab.T, vt_abab, optimize="optimal")

        Exch100 *= -2

        return Exch100
    
    def _dispersion_denominator(self, canonical_denom=False):
        """``1 / (eps_a + eps_b - eps_r - eps_s)``, indexed ``[r, s, a, b]``.

        The CQED orbital energies are the production choice: the perturbation
        series is defined relative to the CQED monomer Hamiltonians, and the
        numerator already carries cavity-dressed integrals, so canonical
        denominators would mix two reference definitions.
        ``canonical_denom=True`` is retained as a *diagnostic*.
        """
        eps = self.eps_canonical if canonical_denom else self.eps
        return 1 / (
            -eps('r', dim=4) - eps('s', dim=3) + eps('a', dim=2) + eps('b')
        )

    def _dispersion_numerator(self, context="total"):
        """``v(abrs)``, the only two-electron block dispersion needs.

        ``v('rsab')`` is not built separately: it is exactly
        ``v('abrs').transpose(2, 3, 0, 1)``, because the AO three-index tensor
        is symmetric in its orbital pair and so ``b_(ra) = b_(ar)^T`` (the same
        identity follows from the eightfold ERI symmetry for the dense
        backend).  Verified to machine precision in both backends and all three
        operator contexts; see ``test_disp20_rsab_block_is_the_abrs_transpose``.
        """
        return self.v('abrs', context=context, df_role="corr")

    def _store_dispersion_amplitudes(self, v_abrs, canonical_denom, context):
        self.t_rsab = oe.contract(
            "rsab,rsab->rsab",
            v_abrs.transpose(2, 3, 0, 1),
            self._dispersion_denominator(canonical_denom),
            optimize="optimal",
        )
        self.t_rsab_canonical_denom = bool(canonical_denom)
        self.t_rsab_context = context
        return self.t_rsab

    def dispersion_amplitudes(self, canonical_denom=False, context="total"):
        """Build the second-order dispersion amplitudes ``t_rsab``.

        Both the denominator convention and the operator context are recorded
        on the driver (``t_rsab_canonical_denom``, ``t_rsab_context``) so that
        ``compute_Eexchdisp200`` cannot silently consume amplitudes built under
        a different one.
        """
        return self._store_dispersion_amplitudes(
            self._dispersion_numerator(context), canonical_denom, context
        )

    def _require_dispersion_amplitudes(self, canonical_denom=None, context="total"):
        """Return dispersion amplitudes, building them only when unambiguous.

        ``canonical_denom=None`` means "reuse whatever ``compute_Edisp200``
        built".  That is the ordinary ``run()`` ordering, but it is only safe
        because the amplitudes must already exist -- reusing stale amplitudes
        built under a different denominator convention or operator context is
        exactly the silent coupling this guard exists to prevent.
        """
        cached = getattr(self, "t_rsab", None)

        if canonical_denom is None:
            if cached is None:
                raise RuntimeError(
                    "Exchange-dispersion requires the Disp20 amplitudes, which "
                    "compute_Edisp200() builds. Call compute_Edisp200() (or "
                    "dispersion_amplitudes()) first, or pass canonical_denom "
                    "explicitly to build them here."
                )
            if self.t_rsab_context != context:
                raise RuntimeError(
                    "The cached Disp20 amplitudes were built in the "
                    f"{self.t_rsab_context!r} operator context, but "
                    f"{context!r} was requested. Rebuild them explicitly."
                )
            return cached

        if (
            cached is not None
            and self.t_rsab_canonical_denom == bool(canonical_denom)
            and self.t_rsab_context == context
        ):
            return cached

        return self.dispersion_amplitudes(canonical_denom=canonical_denom, context=context)

    def compute_Edisp200(self, canonical_denom=False, context="total"):
        """Second-order dispersion.

        ``context`` is a diagnostic: ``"standard"`` and ``"cavity"`` evaluate
        the expression with only that part of the two-electron operator in
        *both* factors of the numerator.  Because the energy is quadratic in
        the numerator, those two do **not** sum to the total --- there is a
        cross term.  Use :meth:`dispersion_energy_partition` for a partition
        that does sum.
        """
        v_abrs = self._dispersion_numerator(context)
        t_rsab = self._store_dispersion_amplitudes(v_abrs, canonical_denom, context)

        Disp200 = 4 * oe.contract('rsab,abrs->', t_rsab, v_abrs, optimize="optimal")
        return Disp200

    def dispersion_energy_partition(self, canonical_denom=False):
        """Split ``Disp20`` into standard, cavity-mediated, and cross parts.

        With ``v = v_std + v_cav`` the energy is quadratic in the numerator, so

            ``Disp20 = Disp20[std,std] + 2 Disp20[std,cav] + Disp20[cav,cav]``

        and the naive two-way split does *not* add up.  The three-way split
        here does, exactly.  The ``cavity`` term is the purely cavity-mediated
        dispersion, which carries no Coulomb kernel and therefore does not
        decay with intermonomer separation (see
        ``docs/qed_sapt0_formalism.tex``); ``cross`` is the interference
        between the two mechanisms.

        This is a diagnostic and deliberately does not disturb the cached
        production amplitudes.
        """
        v_std = self._dispersion_numerator("standard")
        v_cav = self._dispersion_numerator("cavity")
        denominator = self._dispersion_denominator(canonical_denom)

        def _energy(left, right):
            return 4 * float(
                oe.contract(
                    "abrs,rsab,rsab->",
                    left,
                    right.transpose(2, 3, 0, 1),
                    denominator,
                    optimize="optimal",
                )
            )

        standard = _energy(v_std, v_std)
        cavity = _energy(v_cav, v_cav)
        cross = 2 * _energy(v_std, v_cav)

        return {
            "standard": standard,
            "cavity": cavity,
            "cross": cross,
            "total": standard + cross + cavity,
        }
    
    def compute_Eexchdisp200(self, canonical_denom=None):

        t_rsab = self._require_dispersion_amplitudes(canonical_denom)

        vt_abar = self.vt('abar', df_role="corr")
        vt_abra = self.vt('abra', df_role="corr")
        vt_absb = self.vt('absb', df_role="corr")
        vt_abbs = self.vt('abbs', df_role="corr")

        _tmp = 2 * vt_abar - vt_abra.swapaxes(2,3)
        h_abrs = oe.contract('as,AbAr->abrs', self.s('as'), _tmp, optimize="optimal")

        _tmp = 2 * vt_abra - vt_abar.swapaxes(2,3)
        h_abrs += oe.contract('As,abrA->abrs', self.s('as'), _tmp, optimize="optimal")

        _tmp = 2 * vt_absb - vt_abbs.swapaxes(2,3)
        h_abrs += oe.contract('br,aBsB->abrs', self.s('br'), _tmp, optimize="optimal")

        _tmp = 2 * vt_abbs - vt_absb.swapaxes(2,3)
        h_abrs += oe.contract('Br,abBs->abrs', self.s('br'), _tmp, optimize="optimal")

        # build q_abrs
        vt_abas = self.vt('abas', df_role="corr")
        # q_abrs =      np.einsum('br,AB,aBAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas, optimize=True)
        q_abrs = oe.contract('br,AB,aBAs->abrs', self.s('br'), self.s('ab'), vt_abas, optimize="optimal")
        # q_abrs -= 2 * np.einsum('Br,AB,abAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas, optimize=True)
        q_abrs -= 2 * oe.contract('Br,AB,abAs->abrs', self.s('br'), self.s('ab'), vt_abas, optimize="optimal")
        # q_abrs -= 2 * np.einsum('br,aB,ABAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas, optimize=True)
        q_abrs -= 2 * oe.contract('br,aB,ABAs->abrs', self.s('br'), self.s('ab'), vt_abas, optimize="optimal")
        # q_abrs += 4 * np.einsum('Br,aB,AbAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas, optimize=True)
        q_abrs += 4 * oe.contract('Br,aB,AbAs->abrs', self.s('br'), self.s('ab'), vt_abas, optimize="optimal")

        vt_abrb = self.vt('abrb', df_role="corr")
        #q_abrs -= 2 * np.einsum('as,bA,ABrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb, optimize=True)
        q_abrs -= 2 * oe.contract('as,bA,ABrB->abrs', self.s('as'), self.s('ba'), vt_abrb, optimize="optimal")
        #q_abrs += 4 * np.einsum('As,bA,aBrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb, optimize=True)
        q_abrs += 4 * oe.contract('As,bA,aBrB->abrs', self.s('as'), self.s('ba'), vt_abrb, optimize="optimal")
        #q_abrs +=     np.einsum('as,BA,AbrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb, optimize=True)
        q_abrs +=    oe.contract('as,BA,AbrB->abrs', self.s('as'), self.s('ba'), vt_abrb, optimize="optimal")
        #q_abrs -= 2 * np.einsum('As,BA,abrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb, optimize=True)
        q_abrs -= 2 * oe.contract('As,BA,abrB->abrs', self.s('as'), self.s('ba'), vt_abrb, optimize="optimal")

        vt_abab = self.vt('abab', df_role="corr")
        #q_abrs +=     np.einsum('Br,As,abAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab, optimize=True)
        q_abrs +=     oe.contract('Br,As,abAB->abrs', self.s('br'), self.s('as'), vt_abab, optimize="optimal")
        #q_abrs -= 2 * np.einsum('br,As,aBAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab, optimize=True)
        q_abrs -= 2 * oe.contract('br,As,aBAB->abrs', self.s('br'), self.s('as'), vt_abab, optimize="optimal")
        #q_abrs -= 2 * np.einsum('Br,as,AbAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab, optimize=True)
        q_abrs -= 2 * oe.contract('Br,as,AbAB->abrs', self.s('br'), self.s('as'), vt_abab, optimize="optimal")

        vt_abrs = self.vt('abrs', df_role="corr")
        #q_abrs +=     np.einsum('bA,aB,ABrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs, optimize=True)
        q_abrs +=     oe.contract('bA,aB,ABrs->abrs', self.s('ba'), self.s('ab'), vt_abrs, optimize="optimal")
        #q_abrs -= 2 * np.einsum('bA,AB,aBrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs, optimize=True)
        q_abrs -= 2 * oe.contract('bA,AB,aBrs->abrs', self.s('ba'), self.s('ab'), vt_abrs, optimize="optimal")
        #q_abrs -= 2 * np.einsum('BA,aB,Abrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs, optimize=True)
        q_abrs -= 2 * oe.contract('BA,aB,Abrs->abrs', self.s('ba'), self.s('ab'), vt_abrs, optimize="optimal")

        # sum all terms and contract with t_rsab
        xd_absr = self.vt('absr', df_role="corr") + h_abrs.swapaxes(2,3) + q_abrs.swapaxes(2,3)
        Eexchdisp200 = -2 * oe.contract('absr,rsab->', xd_absr, t_rsab, optimize="optimal")
        return Eexchdisp200
    
    def compute_Eind200(self):
        self.CPHF_ra, Ind200_ba = self.chf('B', ind=True)
        self.CPHF_sb, Ind200_ab = self.chf('A', ind=True)

        return Ind200_ba + Ind200_ab
    
    def compute_Eexchind200(self):
        # A <- B
        vt_abra = self.vt('abra')
        vt_abar = self.vt('abar')

        #ExchInd20_ab  =     np.einsum('ra,abbr', CPHF_ra, sapt.vt('abbr'), optimize=True)
        ExchInd20_ab = oe.contract('ra,abbr', self.CPHF_ra, self.vt('abbr'), optimize="optimal")
        #ExchInd20_ab += 2 * np.einsum('rA,Ab,abar', CPHF_ra, sapt.s('ab'), vt_abar, optimize=True)
        ExchInd20_ab += 2 * oe.contract('rA,Ab,abar', self.CPHF_ra, self.s('ab'), vt_abar, optimize="optimal")
        #ExchInd20_ab += 2 * np.einsum('ra,Ab,abrA', CPHF_ra, sapt.s('ab'), vt_abra, optimize=True)
        ExchInd20_ab += 2 * oe.contract('ra,Ab,abrA', self.CPHF_ra, self.s('ab'), vt_abra, optimize="optimal")
        #ExchInd20_ab -=     np.einsum('rA,Ab,abra', CPHF_ra, sapt.s('ab'), vt_abra, optimize=True)
        ExchInd20_ab -=     oe.contract('rA,Ab,abra', self.CPHF_ra, self.s('ab'), vt_abra, optimize="optimal")

        vt_abbb = self.vt('abbb')
        vt_abab = self.vt('abab')
        #ExchInd20_ab -=     np.einsum('ra,Ab,abAr', CPHF_ra, sapt.s('ab'), vt_abar, optimize=True)
        ExchInd20_ab -=     oe.contract('ra,Ab,abAr', self.CPHF_ra, self.s('ab'), vt_abar, optimize="optimal")
        #ExchInd20_ab += 2 * np.einsum('ra,Br,abBb', CPHF_ra, sapt.s('br'), vt_abbb, optimize=True)
        ExchInd20_ab += 2 * oe.contract('ra,Br,abBb', self.CPHF_ra, self.s('br'), vt_abbb, optimize="optimal")
        #ExchInd20_ab -=     np.einsum('ra,Br,abbB', CPHF_ra, sapt.s('br'), vt_abbb, optimize=True)
        ExchInd20_ab -=     oe.contract('ra,Br,abbB', self.CPHF_ra, self.s('br'), vt_abbb, optimize="optimal")
        #ExchInd20_ab -= 2 * np.einsum('rA,Ab,Br,abaB', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab, optimize=True)
        ExchInd20_ab -= 2 * oe.contract('rA,Ab,Br,abaB', self.CPHF_ra, self.s('ab'), self.s('br'), vt_abab, optimize="optimal")

        vt_abrb = self.vt('abrb')
        #ExchInd20_ab -= 2 * np.einsum('ra,Ab,BA,abrB', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb, optimize=True)
        ExchInd20_ab -= 2 * oe.contract('ra,Ab,BA,abrB', self.CPHF_ra, self.s('ab'), self.s('ba'), vt_abrb, optimize="optimal")
        #ExchInd20_ab -= 2 * np.einsum('ra,AB,Br,abAb', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab, optimize=True)
        ExchInd20_ab -= 2 * oe.contract('ra,AB,Br,abAb', self.CPHF_ra, self.s('ab'), self.s('br'), vt_abab, optimize="optimal")
        #ExchInd20_ab -= 2 * np.einsum('rA,AB,Ba,abrb', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb, optimize=True)
        ExchInd20_ab -= 2 * oe.contract('rA,AB,Ba,abrb', self.CPHF_ra, self.s('ab'), self.s('ba'), vt_abrb, optimize="optimal")

        #ExchInd20_ab +=     np.einsum('ra,Ab,Br,abAB', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab, optimize=True)
        ExchInd20_ab +=     oe.contract('ra,Ab,Br,abAB', self.CPHF_ra, self.s('ab'), self.s('br'), vt_abab, optimize="optimal")
        #ExchInd20_ab +=     np.einsum('rA,Ab,Ba,abrB', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb, optimize=True)
        ExchInd20_ab +=     oe.contract('rA,Ab,Ba,abrB', self.CPHF_ra, self.s('ab'), self.s('ba'), vt_abrb, optimize="optimal")

        ExchInd20_ab *= -2

        # B <- A
        vt_abbs = self.vt('abbs')
        vt_absb = self.vt('absb')
        #ExchInd20_ba  =     np.einsum('sb,absa', CPHF_sb, sapt.vt('absa'), optimize=True)
        ExchInd20_ba  =     oe.contract('sb,absa', self.CPHF_sb, self.vt('absa'), optimize="optimal")
        #ExchInd20_ba += 2 * np.einsum('sB,Ba,absb', CPHF_sb, sapt.s('ba'), vt_absb, optimize=True)
        ExchInd20_ba += 2 * oe.contract('sB,Ba,absb', self.CPHF_sb, self.s('ba'), vt_absb, optimize="optimal")
        #ExchInd20_ba += 2 * np.einsum('sb,Ba,abBs', CPHF_sb, sapt.s('ba'), vt_abbs, optimize=True)
        ExchInd20_ba += 2 * oe.contract('sb,Ba,abBs', self.CPHF_sb, self.s('ba'), vt_abbs, optimize="optimal")
        #ExchInd20_ba -=     np.einsum('sB,Ba,abbs', CPHF_sb, sapt.s('ba'), vt_abbs, optimize=True)
        ExchInd20_ba -=     oe.contract('sB,Ba,abbs', self.CPHF_sb, self.s('ba'), vt_abbs, optimize="optimal")

        #vt_abaa = sapt.vt('abaa')
        #vt_abab = sapt.vt('abab')
        vt_abaa = self.vt('abaa')
        vt_abab = self.vt('abab')

        #ExchInd20_ba -=     np.einsum('sb,Ba,absB', CPHF_sb, sapt.s('ba'), vt_absb, optimize=True)
        ExchInd20_ba -=     oe.contract('sb,Ba,absB', self.CPHF_sb, self.s('ba'), vt_absb, optimize="optimal")
        #ExchInd20_ba += 2 * np.einsum('sb,As,abaA', CPHF_sb, sapt.s('as'), vt_abaa, optimize=True)
        ExchInd20_ba += 2 * oe.contract('sb,As,abaA', self.CPHF_sb, self.s('as'), vt_abaa, optimize="optimal")
        #ExchInd20_ba -=     np.einsum('sb,As,abAa', CPHF_sb, sapt.s('as'), vt_abaa, optimize=True)
        ExchInd20_ba -=     oe.contract('sb,As,abAa', self.CPHF_sb, self.s('as'), vt_abaa, optimize="optimal")
        #ExchInd20_ba -= 2 * np.einsum('sB,Ba,As,abAb', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab, optimize=True)
        ExchInd20_ba -= 2 * oe.contract('sB,Ba,As,abAb', self.CPHF_sb, self.s('ba'), self.s('as'), vt_abab, optimize="optimal")

        #vt_abas = sapt.vt('abas')
        vt_abas = self.vt('abas')
        #ExchInd20_ba -= 2 * np.einsum('sb,Ba,AB,abAs', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas, optimize=True)
        ExchInd20_ba -= 2 * oe.contract('sb,Ba,AB,abAs', self.CPHF_sb, self.s('ba'), self.s('ab'), vt_abas, optimize="optimal")
        #ExchInd20_ba -= 2 * np.einsum('sb,BA,As,abaB', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab, optimize=True)
        ExchInd20_ba -= 2 * oe.contract('sb,BA,As,abaB', self.CPHF_sb, self.s('ba'), self.s('as'), vt_abab, optimize="optimal")
        #ExchInd20_ba -= 2 * np.einsum('sB,BA,Ab,abas', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas, optimize=True)
        ExchInd20_ba -= 2 * oe.contract('sB,BA,Ab,abas', self.CPHF_sb, self.s('ba'), self.s('ab'), vt_abas, optimize="optimal")

        #ExchInd20_ba +=     np.einsum('sb,Ba,As,abAB', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab, optimize=True)
        ExchInd20_ba +=     oe.contract('sb,Ba,As,abAB', self.CPHF_sb, self.s('ba'), self.s('as'), vt_abab, optimize="optimal")
        #ExchInd20_ba +=     np.einsum('sB,Ba,Ab,abAs', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas, optimize=True)
        ExchInd20_ba +=     oe.contract('sB,Ba,Ab,abAs', self.CPHF_sb, self.s('ba'), self.s('ab'), vt_abas, optimize="optimal")

        ExchInd20_ba *= -2
        ExchInd200 = ExchInd20_ab + ExchInd20_ba
        return ExchInd200






    def _build_results(self) -> QEDSAPT0Results:
        """Package computed SAPT0 component attributes into a result object."""

        required = (
            "Eelst100",
            "Eexch100",
            "Edisp200",
            "Eexchdisp200",
            "Eind200",
            "Eexchind200",
            "E_SAPT0",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                "QED-SAPT0 components are not available yet; call run() first. "
                f"Missing: {', '.join(missing)}"
            )

        return QEDSAPT0Results(
            elst10=float(self.Eelst100),
            exch10=float(self.Eexch100),
            disp20=float(self.Edisp200),
            exch_disp20=float(self.Eexchdisp200),
            ind20=float(self.Eind200),
            exch_ind20=float(self.Eexchind200),
            metadata={
                **self.metadata,
                "total": float(self.E_SAPT0),
            },
        )

    def results(self) -> QEDSAPT0Results:
        """Return component results from the most recent QED-SAPT0 run."""

        return self._build_results()

    def run_components(self) -> QEDSAPT0Results:
        """Run QED-SAPT0 and return named component energies."""

        self.run()
        return self._build_results()

    def run(self) -> float:
        """Run QED-SAPT0 and return the total interaction energy."""

        monomers = self.prepare_monomers()
        integrals = self.build_integrals()

        self.Eelst100 = self.compute_Elst100()
        self.Eexch100 = self.compute_Exch100()
        self.Edisp200 = self.compute_Edisp200()
        self.Eexchdisp200 = self.compute_Eexchdisp200()
        self.Eind200 = self.compute_Eind200()
        self.Eexchind200 = self.compute_Eexchind200()
        
        self.E_SAPT0 = self.Eelst100 + self.Eexch100 + self.Edisp200 + self.Eexchdisp200 + self.Eind200 + self.Eexchind200

        from .qed_sapt_jk import print_sapt_summary
        print_sapt_summary(
            [
                ("Electrostatics", float(self.Eelst100)),
                ("Exchange", float(self.Eexch100)),
                ("Induction", float(self.Eind200)),
                ("Exchange-Induction", float(self.Eexchind200)),
                ("Dispersion", float(self.Edisp200)),
                ("Exchange-Dispersion", float(self.Eexchdisp200)),
            ],
            total=float(self.E_SAPT0),
        )
        return self.E_SAPT0
