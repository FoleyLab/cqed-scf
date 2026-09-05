"""Shared response-theory scaffolding for CQED calculations.

Response drivers will consume converged SCF result dictionaries produced by
``cqed_scf.scf`` and, later, ``cqed_scf.uscf``.  The restricted SCF dictionary
already exposes the core objects needed here:

- ``coefficients``
- ``orbital_energies``
- ``density``
- ``mints``
- ``wfn``
- ``d_ao``
- ``d_exp``
- ``ndocc``
- ``functional`` and ``reference``/``method`` metadata

No response physics is implemented in this module yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import opt_einsum as oe

from .references import CQEDConfig


class CQEDResponse:
    """Base class for future CQED CPHF/CPKS/TDDFT response solvers.

    Intended workflow:
    1. Read occupied and virtual orbital blocks from SCF coefficients and
       orbital energies.
    2. Build property-specific right-hand-side vectors.
    3. Provide matrix-free Hessian-vector products ``sigma = A[X]``.
    4. Solve linear systems ``A X = b`` with full ERIs first.
    5. Add density-fitting backends later behind the same interface.
    6. Supply reusable response objects to SAPT induction and LR-TDDFT.
    """

    required_scf_keys = (
        "coefficients",
        "orbital_energies",
        "density",
        "mints",
        "wfn",
        "d_ao",
        "d_exp",
        "ndocc",
        "functional",
        "reference",
    )

    def __init__(
        self,
        config: CQEDConfig,
        scf_results: Optional[Mapping[str, Any]] = None,
        integral_backend: str = "full_eri",
    ):
        self.config = config
        self.scf_results = scf_results
        self.integral_backend = integral_backend

    def validate_scf_results(self) -> None:
        """Validate that the SCF result dictionary has the expected contract."""

        if self.scf_results is None:
            raise ValueError("scf_results must be supplied before building response objects")

        missing = [key for key in self.required_scf_keys if key not in self.scf_results]
        if missing:
            raise KeyError(f"SCF results missing required response keys: {missing}")

    def build_orbital_blocks(self):
        """Build occupied/virtual orbital blocks for response equations."""

        raise NotImplementedError(
            "CQEDResponse.build_orbital_blocks is not implemented yet. "
            "Future code will build occupied/virtual blocks from SCF results."
        )

    def build_rhs(self, operator: Any):
        """Build a property right-hand side for ``A X = b``."""

        raise NotImplementedError(
            "CQEDResponse.build_rhs is not implemented yet. "
            "Future code will build response right-hand sides from AO operators."
        )

    def sigma(self, amplitudes: Any):
        """Apply the matrix-free response Hessian to trial amplitudes."""

        raise NotImplementedError(
            "CQEDResponse.sigma is not implemented yet. "
            "Future code will evaluate sigma = A[X], using full ERIs first."
        )

    def solve(self, rhs: Any):
        """Solve future CPHF/CPKS linear response equations."""

        raise NotImplementedError(
            "CQEDResponse.solve is not implemented yet. "
            "Future code will solve A X = b for SAPT induction and properties."
        )


class CQEDCPHF(CQEDResponse):
    """Future restricted/unrestricted CPHF response driver."""


class CQEDCPKS(CQEDResponse):
    """Future restricted/unrestricted CPKS response driver."""


class CQEDTDDFT(CQEDResponse):
    """Future LR-TDDFT driver built on shared CQED response machinery."""

    def excitation_energies(self, nroots: Optional[int] = None):
        """Compute future CQED LR-TDDFT excitation energies."""

        raise NotImplementedError(
            "CQEDTDDFT.excitation_energies is not implemented yet. "
            "Future code will use the shared response Hessian and SCF metadata."
        )


# ===========================================================================
# QED-CIS
# ===========================================================================
#
# Basis ordering (see docs/development/QED_RESPONSE_PLAN.md, section 1):
#
#     index(n, p) = n * (1 + n_ov) + p
#         p = 0                  ->  |Phi_0, n>
#         p = 1 + i*n_virt + a   ->  |Phi_i^a, n>          n = 0 .. N_ph
#
# Photon-major with the reference determinant leading each block.  The
# Hamiltonian is then block-tridiagonal in photon number with only two unique
# blocks for the whole method -- the electronic block A_el and the one-body
# coupling G -- both photon independent:
#
#          n=0            n=1            n=2
#      +--------------+--------------+--------------+
#  n=0 |  A_el + 0w   |   sqrt(1) G  |      0       |
#  n=1 |  sqrt(1) G   |  A_el + 1w   |   sqrt(2) G  |
#  n=2 |      0       |   sqrt(2) G  |  A_el + 2w   |
#      +--------------+--------------+--------------+
#
# so QED-CIS-N costs the same code as QED-CIS-1.
#
# The electronic block is written in the GENERAL Fock form
#
#     A_ia,jb = F_ab d_ij - F_ij d_ab + 2(ia|jb) - (ij|ab)
#               + 2 d_ia d_jb - d_ij d_ab
#     <Phi_0,n|H|Phi_i^a,n> = sqrt(2) F_ia
#
# rather than assuming eps_a - eps_i.  With canonical CQED-HF orbitals F_oo and
# F_vv are diagonal and F_ov = 0, so this reduces to the oracle's equations
# exactly; with canonical HF orbitals it stays correct, at the cost of a live
# Brillouin block.  Truncated CI is NOT orbital invariant, so the two bases give
# genuinely different spectra -- that is physics, not error.  See plan 0.4/0.5.


@dataclass
class QEDCISResults:
    """Eigenpairs and polaritonic character of a QED-CIS calculation."""

    eigenvalues: np.ndarray          # relative to E_CQED-SCF
    eigenvectors: np.ndarray         # columns, in the packed basis above
    excitation_energies: np.ndarray  # E_k - E_0, the physical omega_k
    photon_numbers: np.ndarray       # <b^dag b> per root
    reference_weights: np.ndarray    # (nroot, N_ph+1): weight on |Phi_0, n>
    singles_weights: np.ndarray      # (nroot, N_ph+1): weight on |Phi_i^a, n>
    n_photon: int
    omega: float
    scf_energy: float
    transition_dipoles: Optional[np.ndarray] = None   # (nroot, 3), from the ground polariton
    oscillator_strengths: Optional[np.ndarray] = None  # (nroot,)
    davidson: Any = None                               # DavidsonResult, when used

    @property
    def total_energies(self) -> np.ndarray:
        """Absolute energies, comparable to qed-ci totals."""

        return self.scf_energy + self.eigenvalues

    def polariton_indices(self, count: int = 2, photon_state: int = 1):
        """Roots carrying |Phi_0, n> character, ordered by energy.

        A polariton is the mixture of the *bare photon* |Phi_0, n> with an
        electronic excitation |Phi_i^a, 0>, so the quantity that identifies one
        is the weight on the photonic reference state.

        Do NOT select on ``photon_numbers`` instead.  That is
        ``sum_n n w_n``, which counts every amplitude in the n-photon block --
        including |Phi_i^a, n>, an electronic excitation *carrying* a photon.
        Such a photon-dressed state has photon number near 1 while mixing with
        nothing, and will outrank a genuine polariton.  On MgH+ at resonance it
        does exactly that: the dressed state at (bare transition + omega) has
        <b+b> = 0.91 and displaces the lower polariton at 0.48.

        Returns
        -------
        list[int]
            ``count`` root indices, sorted by increasing energy.  For a single
            bright transition the default of two gives the LP/UP pair, whose
            photonic reference weights should sum to close to one.
        """

        weights = self.reference_weights[:, photon_state]
        count = min(int(count), weights.size)
        selected = np.argsort(weights)[::-1][:count]
        return sorted((int(index) for index in selected),
                      key=lambda index: self.eigenvalues[index])


class QEDCIS(CQEDResponse):
    """Dense QED-CIS-N in the photon-major basis.

    This is the correctness anchor for the matrix-free solver: it is O(N^4) in
    memory and builds the full Hamiltonian, so it is for small systems and for
    testing.  Everything it computes should later be reproducible by applying a
    sigma routine to unit vectors.
    """

    def __init__(
        self,
        config: Optional[CQEDConfig] = None,
        scf_results: Optional[Mapping[str, Any]] = None,
        n_photon: int = 1,
        integral_backend: str = "full_eri",
    ):
        super().__init__(
            config=config,
            scf_results=scf_results,
            integral_backend=integral_backend,
        )
        if int(n_photon) < 0:
            raise ValueError("n_photon must be non-negative")
        self.n_photon = int(n_photon)

        # <Phi_0|H|Phi_0> - E_CQED-SCF.  Exactly zero for CQED-HF orbitals,
        # which is the only basis wired up so far.
        self.reference_shift = 0.0

        # Set here as well as in build_orbital_blocks so that a driver
        # constructed with orbital blocks assigned by hand (as the synthetic
        # tests do, bypassing build_orbital_blocks) is fully formed.
        self.is_ks = False
        self.ovov = None
        self.oovv = None

        self._prepared = False

    # -------------------------
    # setup
    # -------------------------

    def build_orbital_blocks(self) -> None:
        """Pull the occupied/virtual Fock, dipole, and ERI blocks from the SCF."""

        import psi4

        self.validate_scf_results()
        res = self.scf_results

        self.C = np.asarray(res["coefficients"])
        self.ndocc = int(res["ndocc"])
        self.nmo = int(res["nmo"])
        self.nvirt = self.nmo - self.ndocc
        self.n_ov = self.ndocc * self.nvirt

        if self.nvirt < 1:
            raise ValueError("QED-CIS requires at least one virtual orbital")

        self.omega = float(res["omega"]) if "omega" in res else float(self.config.omega)
        self.scf_energy = float(res.get("energy_scf", 0.0))

        no, nv = self.ndocc, self.nvirt

        # Fock in the MO basis.  scf.py caches this; recompute if absent so the
        # class also works against an older results dict.
        F_mo = res.get("fock_mo")
        if F_mo is None:
            F_mo = self.C.T @ np.asarray(res["F"]) @ self.C
        F_mo = np.asarray(F_mo)
        self.F_oo = F_mo[:no, :no]
        self.F_ov = F_mo[:no, no:]
        self.F_vv = F_mo[no:, no:]

        # lambda . mu_el in the MO basis
        d_mo = self.C.T @ np.asarray(res["d_ao"]) @ self.C
        self.d_oo = d_mo[:no, :no]
        self.d_ov = d_mo[:no, no:]
        self.d_vv = d_mo[no:, no:]

        # Explicit MO ERIs are deferred: the transformation is O(N^4) and the
        # matrix-free path never needs it.  _ensure_mo_eri() materialises them
        # for the dense builder and the dense integral engine.
        self.ovov = None
        self.oovv = None

        # Kohn-Sham references route the two-electron action through Psi4 so
        # that the XC kernel and hybrid scaling follow Psi4's own convention.
        self.is_ks = res.get("functional") is not None

        # cartesian electronic dipole in the MO basis, for transition properties
        mints = res["mints"]
        self.mu_mo = np.array(
            [self.C.T @ np.asarray(component) @ self.C for component in mints.ao_dipole()]
        )

        self._prepared = True

    # -------------------------
    # the two unique blocks
    # -------------------------

    def _ensure_mo_eri(self) -> None:
        """Materialise (ia|jb) and (ij|ab) on first use.

        Only the explicit-Hamiltonian path and DenseERIEngine need these; the
        Davidson path goes through J/K builds and never forms them.
        """

        if self.ovov is not None and self.oovv is not None:
            return

        import psi4

        no = self.ndocc
        mints = self.scf_results["mints"]
        Co = psi4.core.Matrix.from_array(np.ascontiguousarray(self.C[:, :no]))
        Cv = psi4.core.Matrix.from_array(np.ascontiguousarray(self.C[:, no:]))
        self.ovov = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))  # (ia|jb)
        self.oovv = np.asarray(mints.mo_eri(Co, Co, Cv, Cv))  # (ij|ab)

    def _two_electron_block(self) -> np.ndarray:
        """The (n_ov x n_ov) two-electron block of A.

        For Hartree-Fock this is 2(ia|jb) - (ij|ab) from explicit integrals.
        For Kohn-Sham it also carries the XC kernel and the hybrid exchange
        scaling, neither of which is available as a simple MO integral, so it is
        assembled by applying the integral engine to unit vectors -- one batched
        pass, and exactly the action the Davidson path uses.
        """

        if not self.is_ks:
            # An explicit MO-integral form always exists for a non-KS reference,
            # and using it avoids constructing an integral engine at all -- which
            # keeps the dense path importable without Psi4 for synthetic tests.
            self._ensure_mo_eri()
            block = 2.0 * self.ovov - self.oovv.transpose(0, 2, 1, 3)
            return block.reshape(self.n_ov, self.n_ov)

        identity = np.eye(self.n_ov).reshape(self.n_ov, self.ndocc, self.nvirt)
        return self.eri_engine.ov_sigma(identity).reshape(self.n_ov, self.n_ov).T

    def build_electronic_block(self) -> np.ndarray:
        """A_el, the photon-independent electronic block, shape (n_ov, n_ov)."""

        if not self._prepared:
            self.build_orbital_blocks()

        no, nv = self.ndocc, self.nvirt
        eye_o, eye_v = np.eye(no), np.eye(nv)

        A = oe.contract("ab,ij->iajb", self.F_vv, eye_o, optimize="optimal")
        A -= oe.contract("ij,ab->iajb", self.F_oo, eye_v, optimize="optimal")

        # dipole self-energy: rank-one direct term minus a separable exchange term
        A += 2.0 * oe.contract("ia,jb->iajb", self.d_ov, self.d_ov, optimize="optimal")
        A -= oe.contract("ij,ab->iajb", self.d_oo, self.d_vv, optimize="optimal")

        return A.reshape(self.n_ov, self.n_ov) + self._two_electron_block()

    def build_coupling_block(self) -> np.ndarray:
        """G, the photon-independent bilinear coupling, shape (n_ov, n_ov).

        G_ia,jb = delta_ab d_ij - delta_ij d_ab
        """

        if not self._prepared:
            self.build_orbital_blocks()

        no, nv = self.ndocc, self.nvirt
        G = oe.contract("ij,ab->iajb", self.d_oo, np.eye(nv), optimize="optimal")
        G -= oe.contract("ij,ab->iajb", np.eye(no), self.d_vv, optimize="optimal")
        return G.reshape(self.n_ov, self.n_ov)

    # -------------------------
    # assembly
    # -------------------------

    @property
    def block_size(self) -> int:
        """States per photon block: the reference plus all singles.

        Prepares the orbital blocks on demand, like every other entry point on
        this class, so that a driver obtained from ``CQEDCalculator.response()``
        can report its own size before anything is solved.
        """

        if not self._prepared:
            self.build_orbital_blocks()
        return 1 + self.n_ov

    @property
    def dimension(self) -> int:
        """Total basis dimension, ``(N_ph + 1) * (1 + n_ov)``."""

        if not self._prepared:
            self.build_orbital_blocks()
        return (self.n_photon + 1) * self.block_size

    def build_dense_hamiltonian(self) -> np.ndarray:
        """Assemble the full block-tridiagonal Hamiltonian.

        Energies are relative to E_CQED-SCF.  Note that the lowest eigenvalue is
        NOT zero once lambda != 0: the bilinear term couples |Phi_0,0> to
        |Phi_i^a,1> and relaxes the ground state below the reference.  Physical
        excitation energies are E_k - E_0, not the eigenvalues themselves.
        """

        if not self._prepared:
            self.build_orbital_blocks()

        A = self.build_electronic_block()
        G = self.build_coupling_block()

        nph, nov, block = self.n_photon, self.n_ov, self.block_size
        omega, g = self.omega, np.sqrt(self.omega / 2.0)

        H = np.zeros((self.dimension, self.dimension))
        eye_ov = np.eye(nov)
        F_ov_flat = np.sqrt(2.0) * self.F_ov.ravel()

        # diagonal photon blocks
        for n in range(nph + 1):
            s = n * block
            H[s, s] = n * omega + self.reference_shift
            H[s + 1 : s + block, s + 1 : s + block] = A + (n * omega) * eye_ov
            # Brillouin block: zero for canonical CQED-HF orbitals, live otherwise
            H[s, s + 1 : s + block] = F_ov_flat
            H[s + 1 : s + block, s] = F_ov_flat

        # off-diagonal photon blocks (Delta n = +-1)
        for n in range(nph):
            s, t = n * block, (n + 1) * block
            ladder = np.sqrt(n + 1.0)
            ref_coupling = -np.sqrt((n + 1.0) * omega) * self.d_ov.ravel()

            # <Phi_0, n+1| H |Phi_i^a, n>
            H[t, s + 1 : s + block] = ref_coupling
            H[s + 1 : s + block, t] = ref_coupling
            # <Phi_0, n| H |Phi_i^a, n+1>
            H[s, t + 1 : t + block] = ref_coupling
            H[t + 1 : t + block, s] = ref_coupling
            # <Phi_i^a, n+1| H |Phi_j^b, n>;  <Phi_0,n|H|Phi_0,n+-1> stays zero
            H[t + 1 : t + block, s + 1 : s + block] = ladder * g * G
            H[s + 1 : s + block, t + 1 : t + block] = ladder * g * G.T

        return H

    # -------------------------
    # solve
    # -------------------------

    def _make_results(self, eigenvalues, eigenvectors, davidson=None) -> QEDCISResults:
        """Package eigenpairs with their polaritonic character."""

        nph, block = self.n_photon, self.block_size
        packed = eigenvectors.T.reshape(-1, nph + 1, block)

        reference_weights = packed[:, :, 0] ** 2
        singles_weights = np.sum(packed[:, :, 1:] ** 2, axis=2)
        block_weights = reference_weights + singles_weights
        photon_numbers = block_weights @ np.arange(nph + 1, dtype=float)

        dipoles, strengths = self._transition_properties(packed, eigenvalues)

        return QEDCISResults(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            excitation_energies=eigenvalues - eigenvalues[0],
            photon_numbers=photon_numbers,
            reference_weights=reference_weights,
            singles_weights=singles_weights,
            n_photon=self.n_photon,
            omega=self.omega,
            scf_energy=self.scf_energy,
            transition_dipoles=dipoles,
            oscillator_strengths=strengths,
            davidson=davidson,
        )

    def _transition_properties(self, packed, eigenvalues):
        """Transition dipoles from the ground polariton, and oscillator strengths.

        The electronic dipole is diagonal in photon number, so

            <Psi_0| mu |Psi_k> = sum_n [ sqrt(2) ( c_n^0 <mu_ov, t_n^k>
                                                 + c_n^k <mu_ov, t_n^0> )
                                       + <t_n^0, t_n^k mu_vv - mu_oo t_n^k> ]

        The singles-singles term matters here in a way it does not for ordinary
        CIS: the polaritonic ground state carries genuine singles amplitudes
        because the bilinear coupling mixes |Phi_0,0> with |Phi_i^a,1>.  Any
        constant (reference-expectation) part of the dipole drops out for k != 0
        by orthogonality of the eigenvectors.
        """

        mu_mo = getattr(self, "mu_mo", None)
        if mu_mo is None:
            return None, None

        no, nv = self.ndocc, self.nvirt
        mu_ov = mu_mo[:, :no, no:]
        mu_oo = mu_mo[:, :no, :no]
        mu_vv = mu_mo[:, no:, no:]

        c = packed[:, :, 0]                                        # (nroot, nph+1)
        t = packed[:, :, 1:].reshape(-1, self.n_photon + 1, no, nv)
        c0, t0 = c[0], t[0]

        ref_term = np.sqrt(2.0) * (
            oe.contract("n,xia,knia->kx", c0, mu_ov, t, optimize="optimal")
            + oe.contract("kn,xia,nia->kx", c, mu_ov, t0, optimize="optimal")
        )
        singles_term = oe.contract(
            "nia,knib,xab->kx", t0, t, mu_vv, optimize="optimal"
        ) - oe.contract("nia,knja,xij->kx", t0, t, mu_oo, optimize="optimal")

        dipoles = ref_term + singles_term
        omegas = eigenvalues - eigenvalues[0]
        strengths = (2.0 / 3.0) * omegas * np.sum(dipoles**2, axis=1)
        return dipoles, strengths

    def kernel(
        self,
        nroots: Optional[int] = None,
        solver: str = "dense",
        tol: float = 1e-8,
        strict: bool = True,
        **davidson_kwargs,
    ) -> QEDCISResults:
        """Solve for the lowest roots.  Hermitian only, by design.

        ``solver="dense"`` builds the full Hamiltonian and calls ``eigh`` -- the
        correctness anchor, O(N^4) in memory.  ``solver="davidson"`` never forms
        the Hamiltonian and costs ``(N_ph + 1)`` JK builds per iteration.
        """

        solver = solver.lower()
        if solver == "dense":
            H = self.build_dense_hamiltonian()
            asymmetry = np.max(np.abs(H - H.T))
            if asymmetry > 1e-12:
                raise RuntimeError(
                    f"QED-CIS Hamiltonian is not symmetric (max|H - H^T| = {asymmetry:.3e})"
                )
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            if nroots is not None:
                nroots = min(int(nroots), eigenvalues.size)
                eigenvalues, eigenvectors = eigenvalues[:nroots], eigenvectors[:, :nroots]
            return self._make_results(eigenvalues, eigenvectors)

        if solver != "davidson":
            raise ValueError(f"solver must be 'dense' or 'davidson'; got {solver!r}")

        from .davidson import davidson_solve

        if not self._prepared:
            self.build_orbital_blocks()
        if nroots is None:
            nroots = min(5, self.dimension)
        nroots = min(int(nroots), self.dimension)

        result = davidson_solve(
            self._matvec,
            self.hamiltonian_diagonal(),
            self.initial_guess(nroots),
            nroots,
            tol=tol,
            **davidson_kwargs,
        )
        if strict and not result.converged:
            raise RuntimeError(
                "QED-CIS Davidson did not converge; max residual "
                f"{np.max(result.residual_norms):.3e} after {result.n_iterations} iterations"
            )
        return self._make_results(result.eigenvalues, result.eigenvectors, davidson=result)

    def excitation_energies(self, nroots: Optional[int] = None, **kwargs) -> np.ndarray:
        return self.kernel(nroots=nroots, **kwargs).excitation_energies

    # -------------------------
    # matrix-free machinery
    # -------------------------

    @property
    def eri_engine(self):
        """Lazily built ERI action.  Dense for small runs and tests, JK otherwise."""

        engine = getattr(self, "_eri_engine", None)
        if engine is None:
            if not self._prepared:
                self.build_orbital_blocks()
            backend = self.integral_backend
            if backend in ("full_eri", "df", None):
                # auto: Kohn-Sham needs the XC kernel, which only Psi4 supplies
                backend = "psi4_hx" if self.is_ks else "jk"

            if backend == "dense_eri":
                self._ensure_mo_eri()
                engine = DenseERIEngine(self.ovov, self.oovv)
            elif backend == "psi4_hx":
                engine = Psi4HxERIEngine(
                    self.scf_results["wfn"],
                    self.C[:, : self.ndocc],
                    self.C[:, self.ndocc :],
                )
            else:
                engine = JKERIEngine(
                    self._build_generalized_jk(),
                    self.C[:, : self.ndocc],
                    self.C[:, self.ndocc :],
                )
            self._eri_engine = engine
        return engine

    def _build_generalized_jk(self):
        """A JK object accepting independent C_left/C_right (nonsymmetric densities).

        scf.py's ``_build_JK`` queues only ``C_left``, so it builds ``D = C C^T``
        and cannot form the one-sided response density ``D = C_o X C_v^T``.
        """

        import psi4

        jk = psi4.core.JK.build(self.scf_results["wfn"].basisset())
        jk.set_memory(int(5e8))
        for setter, value in (("set_do_J", True), ("set_do_K", True), ("set_do_wK", False)):
            if hasattr(jk, setter):
                getattr(jk, setter)(value)
        jk.initialize()
        return jk

    def sigma(self, X):
        """Apply the QED-CIS Hamiltonian without building it.

        ``X`` is ``(nvec, N_ph+1, 1+n_ov)`` (a single ``(N_ph+1, 1+n_ov)`` is also
        accepted).  Cost is ``(N_ph+1)`` JK builds for the whole batch; every
        cavity-specific term is O(N^3) or cheaper.
        """

        if not self._prepared:
            self.build_orbital_blocks()

        X = np.asarray(X, dtype=float)
        single = X.ndim == 2
        if single:
            X = X[None]

        nvec = X.shape[0]
        nph, no, nv = self.n_photon, self.ndocc, self.nvirt
        omega, g = self.omega, np.sqrt(self.omega / 2.0)

        c = X[:, :, 0]                                          # (nvec, nph+1)
        t = X[:, :, 1:].reshape(nvec, nph + 1, no, nv)
        sc = np.zeros_like(c)
        st = np.zeros_like(t)

        # --- electronic block: identical in every photon block ---
        # one batched ERI action for all vectors and all photon blocks at once
        st += self.eri_engine.ov_sigma(
            t.reshape(nvec * (nph + 1), no, nv)
        ).reshape(nvec, nph + 1, no, nv)

        st += oe.contract("pnib,ab->pnia", t, self.F_vv, optimize="optimal")
        st -= oe.contract("ij,pnja->pnia", self.F_oo, t, optimize="optimal")

        # dipole self-energy: rank-one direct term, separable exchange term.
        # No JK build -- the dipole operator is separable.
        overlap = oe.contract("ia,pnia->pn", self.d_ov, t, optimize="optimal")
        st += 2.0 * oe.contract("pn,ia->pnia", overlap, self.d_ov, optimize="optimal")
        st -= oe.contract(
            "ij,pnjb,ab->pnia", self.d_oo, t, self.d_vv, optimize="optimal"
        )

        ladder = np.arange(nph + 1, dtype=float)
        st += omega * ladder[None, :, None, None] * t
        sc += omega * ladder[None, :] * c + self.reference_shift * c

        # Brillouin block: zero for canonical CQED-HF orbitals, live otherwise
        root2 = np.sqrt(2.0)
        st += root2 * oe.contract("pn,ia->pnia", c, self.F_ov, optimize="optimal")
        sc += root2 * oe.contract("ia,pnia->pn", self.F_ov, t, optimize="optimal")

        # --- bilinear coupling, Delta n = +-1 ---
        for n in range(nph):
            rung = np.sqrt(n + 1.0)
            ref = -np.sqrt((n + 1.0) * omega)

            sc[:, n + 1] += ref * oe.contract("ia,pia->p", self.d_ov, t[:, n])
            sc[:, n] += ref * oe.contract("ia,pia->p", self.d_ov, t[:, n + 1])
            st[:, n + 1] += ref * c[:, n, None, None] * self.d_ov
            st[:, n] += ref * c[:, n + 1, None, None] * self.d_ov

            st[:, n + 1] += rung * g * (self.d_oo @ t[:, n] - t[:, n] @ self.d_vv)
            st[:, n] += rung * g * (self.d_oo @ t[:, n + 1] - t[:, n + 1] @ self.d_vv)

        out = np.concatenate(
            [sc[:, :, None], st.reshape(nvec, nph + 1, no * nv)], axis=2
        )
        return out[0] if single else out

    def _matvec(self, V):
        """Davidson interface: (nvec, dim) -> (nvec, dim)."""

        V = np.atleast_2d(V)
        X = V.reshape(-1, self.n_photon + 1, self.block_size)
        return self.sigma(X).reshape(V.shape[0], self.dimension)

    def hamiltonian_diagonal(self) -> np.ndarray:
        """Diagonal of H, for preconditioning and guess selection.

        The two-electron ERI diagonal is included only if the engine can supply
        it cheaply; the preconditioner does not need to be exact.
        """

        if not self._prepared:
            self.build_orbital_blocks()

        diag_ov = np.diag(self.F_vv)[None, :] - np.diag(self.F_oo)[:, None]
        diag_ov = diag_ov + 2.0 * self.d_ov**2
        diag_ov = diag_ov - np.outer(np.diag(self.d_oo), np.diag(self.d_vv))

        eri_diagonal = self.eri_engine.ov_diagonal()
        if eri_diagonal is not None:
            diag_ov = diag_ov + eri_diagonal

        diagonal = np.zeros((self.n_photon + 1, self.block_size))
        for n in range(self.n_photon + 1):
            diagonal[n, 0] = n * self.omega + self.reference_shift
            diagonal[n, 1:] = (diag_ov + n * self.omega).ravel()
        return diagonal.ravel()

    def initial_guess(self, nroots: int, n_guess: Optional[int] = None) -> np.ndarray:
        """Unit-vector guess seeded on low diagonals AND every photonic reference.

        The photonic seeds cost N_ph + 1 extra guess vectors and guarantee the
        photonic sector is represented from the first iteration.  This matters
        most on the production path, where the JK engine cannot supply a cheap
        two-electron diagonal and the preconditioner is orbital-energy-only, so
        |Phi_0, n> can rank far down the diagonal ordering while its root sits
        among the lowest few.

        Measured honestly: in synthetic tests -- including a constructed case
        where an 86%-photon root ranked 32nd of 66 by diagonal and fell outside
        the guess -- Davidson recovered it anyway, in 30 iterations versus 29
        with the seed.  So treat this as cheap insurance, not a necessity; no
        case has yet been found here where it is load-bearing.
        """

        diagonal = self.hamiltonian_diagonal()
        if n_guess is None:
            n_guess = max(2 * nroots, nroots + 4)
        n_guess = min(n_guess, self.dimension)

        indices = list(np.argsort(diagonal)[:n_guess])
        for n in range(self.n_photon + 1):
            photonic = n * self.block_size
            if photonic not in indices:
                indices.append(photonic)

        guess = np.zeros((len(indices), self.dimension))
        for row, index in enumerate(indices):
            guess[row, index] = 1.0
        return guess


class DenseERIEngine:
    """(2 J - K) action from precomputed MO integrals.  For tests and small runs."""

    def __init__(self, ovov, oovv):
        self.ovov = np.asarray(ovov)
        self.oovv = np.asarray(oovv)
        self.n_builds = 0

    def ov_sigma(self, X):
        self.n_builds += 1
        out = 2.0 * oe.contract("iajb,pjb->pia", self.ovov, X, optimize="optimal")
        out -= oe.contract("ijab,pjb->pia", self.oovv, X, optimize="optimal")
        return out

    def ov_diagonal(self):
        return 2.0 * np.einsum("iaia->ia", self.ovov) - np.einsum("iiaa->ia", self.oovv)


class JKERIEngine:
    """(2 J - K) action through Psi4's JK, on one-sided response densities.

    For each trial amplitude ``X`` the AO density is ``D = C_o X C_v^T``, queued
    as ``C_left = C_o`` and ``C_right = C_v X^T``.  The whole batch goes into a
    single ``compute()``, so one Davidson iteration costs ``(N_ph + 1)`` JK
    builds regardless of how many trial vectors are in flight.
    """

    def __init__(self, jk, Co, Cv, k_scale: float = 1.0):
        self.jk = jk
        self.Co = np.ascontiguousarray(Co)
        self.Cv = np.ascontiguousarray(Cv)
        self.k_scale = float(k_scale)  # x_alpha for hybrids, later
        self.n_builds = 0

    def ov_sigma(self, X):
        import psi4

        X = np.asarray(X)
        self.jk.C_clear()
        for amplitude in X:
            self.jk.C_left_add(psi4.core.Matrix.from_array(self.Co))
            self.jk.C_right_add(
                psi4.core.Matrix.from_array(np.ascontiguousarray(self.Cv @ amplitude.T))
            )
        self.jk.compute()
        self.n_builds += 1

        J_list, K_list = self.jk.J(), self.jk.K()
        out = np.empty_like(X)
        for index in range(X.shape[0]):
            G = 2.0 * np.asarray(J_list[index]) - self.k_scale * np.asarray(K_list[index])
            out[index] = self.Co.T @ G @ self.Cv
        return out

    def ov_diagonal(self):
        return None  # not available without a transformation; not needed


class Psi4HxERIEngine:
    """Two-electron action through Psi4's own TDA machinery.

    Required for Kohn-Sham references, where the two-electron block carries an
    exchange-correlation kernel that is not expressible as MO integrals.  Rather
    than reconstruct the factors, this calls the same routine Psi4's own TDSCF
    engine uses and combines the pieces the same way
    (``psi4/driver/procrouting/response/scf_products.py::_combine_A``):

        A X = F X + C_o^T ( 2 J_like - K_like ) C_v

    with ``J_like`` returned by ``twoel_Hx_full`` already carrying the XC kernel
    (for a KS reference) and ``K_like`` already carrying the hybrid exchange
    fraction and any range-separated contribution.  Inheriting that convention
    is safer than reproducing it: the factors live in Psi4 and follow whatever
    functional is in play.  The one-electron (Fock) part is *not* taken from
    Psi4's ``onel_Hx`` -- this class supplies only the two-electron action,
    because QEDCIS needs the general non-canonical Fock form.

    Note that ``twoel_Hx_full`` builds its response density from the
    wavefunction's own orbitals.  ``CQEDSCF`` writes the converged CQED orbitals
    into the wavefunction, so this is consistent for the CQED-orbital path; a
    canonical-HF-orbital run would need the wavefunction's coefficients reset
    first, and is not supported through this engine.
    """

    requires_column_build = True  # no cheap explicit-integral form exists

    def __init__(self, wfn, Co, Cv, singlet: bool = True):
        self.wfn = wfn
        self.Co = np.ascontiguousarray(Co)
        self.Cv = np.ascontiguousarray(Cv)
        self.singlet = bool(singlet)
        self.n_builds = 0

        # Mirror Psi4's TDRSCFEngine.__init__.  An RHF wavefunction may not
        # expose a functional in every Psi4 version, in which case the
        # Hartree-Fock answer (full exchange, Coulomb present) is correct.
        try:
            functional = wfn.functional()
            self.needs_K_like = functional.is_x_hybrid() or functional.is_x_lrc()
            self.needs_J_like = self.singlet or functional.needs_xc()
        except (AttributeError, RuntimeError):
            self.needs_K_like = True
            self.needs_J_like = True

    def ov_sigma(self, X):
        import psi4

        X = np.asarray(X)
        vectors = [
            psi4.core.Matrix.from_array(np.ascontiguousarray(amplitude))
            for amplitude in X
        ]

        if hasattr(self.wfn, "twoel_Hx_full"):
            twoel = self.wfn.twoel_Hx_full(vectors, False, "SO", self.singlet)
        elif hasattr(self.wfn, "twoel_Hx"):
            if not self.singlet:
                raise NotImplementedError(
                    "this Psi4 build exposes only twoel_Hx, which cannot select "
                    "the triplet two-electron combination"
                )
            twoel = self.wfn.twoel_Hx(vectors, False, "SO")
        else:
            raise RuntimeError(
                "the Psi4 wavefunction exposes neither twoel_Hx_full nor "
                "twoel_Hx; a Kohn-Sham QED-CIS reference cannot be built"
            )
        self.n_builds += 1

        if self.needs_K_like:
            J_like, K_like = twoel[0::2], twoel[1::2]
        else:
            J_like, K_like = twoel, None

        out = np.empty_like(X)
        for index in range(X.shape[0]):
            G = np.zeros((self.Co.shape[0], self.Co.shape[0]))
            if self.needs_J_like:
                G += 2.0 * np.asarray(J_like[index])
            if K_like is not None:
                G -= np.asarray(K_like[index])
            # Overall sign: Psi4's _combine_A returns -A X, and its caller
            # (TDRSCFEngine.compute_products) negates the result afterwards --
            #     AX_new = self._combine_A(Fx, Jx, Kx)
            #     for Ax in AX_new:
            #         self.vector_scale(-1.0, Ax)
            # Transcribing _combine_A alone therefore gives the wrong sign.  Do
            # not "simplify" this negation away.
            out[index] = -(self.Co.T @ G @ self.Cv)
        return out

    def ov_diagonal(self):
        return None


def print_qed_cis_results(results: QEDCISResults, n_print: Optional[int] = None,
                          title: str = "QED-CIS Excited States") -> None:
    """Emit a results table through the package output layer.

    Photon number is the column that makes the table interpretable: it is what
    separates a polariton from a dark state, and a pair of polaritons formed
    from one electronic transition should share the photon between them.
    """

    from . import output
    from .utils import HARTREE_TO_EV

    output.banner(title)
    output.echo(f"  Reference energy   = {results.scf_energy:18.12f} Eh")
    output.echo(f"  Ground state       = {results.total_energies[0]:18.12f} Eh")
    output.echo(
        f"  Ground relaxation  = {results.eigenvalues[0]:18.12f} Eh "
        "(below the SCF reference)"
    )
    output.echo(f"  Photon frequency   = {results.omega:18.12f} Eh")
    output.echo(f"  Fock states        = 0 .. {results.n_photon}")
    output.echo()

    count = results.eigenvalues.size if n_print is None else min(n_print, results.eigenvalues.size)
    strengths = results.oscillator_strengths

    headers = ["Root", "E (Eh)", "w (Eh)", "w (eV)", "<b+b>", "f (osc)"]
    widths = [6, 20, 14, 12, 10, 12]
    rows = []
    for root in range(count):
        rows.append([
            str(root),
            f"{results.total_energies[root]:18.10f}",
            f"{results.excitation_energies[root]:12.8f}",
            f"{results.excitation_energies[root] * HARTREE_TO_EV:10.5f}",
            f"{results.photon_numbers[root]:8.4f}",
            "n/a" if strengths is None else f"{strengths[root]:10.6f}",
        ])
    output.table(headers, rows, widths)

    if results.davidson is not None:
        davidson = results.davidson
        output.echo(
            f"  Davidson: {davidson.n_iterations} iterations, "
            f"{davidson.n_matvec} sigma builds, "
            f"max residual {np.max(davidson.residual_norms):.3e}"
        )
