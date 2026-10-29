"""Unrestricted CQED-SCF engine for UHF and UKS references.

The returned result dictionary is deliberately spin resolved.  In particular,
it does not expose the restricted-only keys ``density``, ``coefficients``,
``orbital_energies``, or ``ndocc``: aliasing any of those names to one spin
channel would allow restricted consumers to produce plausible but incorrect
results.

Return contract
---------------
:meth:`run` must return ``(energy, results)``.  ``CQEDCalculator._run_scf``
returns that tuple unchanged, and :meth:`CQEDCalculator.energy` immediately reads
``results["energy_psi4"]``, so that key is mandatory.

The core unrestricted keys are:

===========================  ==================================================
key                          meaning
===========================  ==================================================
``energy_scf``               total CQED-SCF energy
``energy_psi4``              the underlying Psi4 UHF/UKS energy (**required**)
``Ca``, ``Cb``               alpha/beta MO coefficients
``Da``, ``Db``               alpha/beta AO densities (each *unscaled*)
``Coa``, ``Cob``             occupied alpha/beta MO coefficients
``Fa``, ``Fb``               converged alpha/beta AO Fock matrices
``orbital_energies_a/_b``    alpha/beta orbital energies
``canonical_orbital_...``    cavity-free reference orbital energies
``nalpha``, ``nbeta``        occupied counts per spin
``s_squared``                <S^2> for the converged determinant
``spin_contamination``       <S^2> minus the target-spin eigenvalue
``method``, ``reference``     unrestricted reference metadata
``functional``               base functional for UKS, otherwise ``None``
``scf_type``, ``jk_type``     requested SCF and actual JK backend metadata
===========================  ==================================================

"""

from __future__ import annotations

from typing import Any

import numpy as np
import psi4

from . import output
from .references import CQEDConfig


def _spin_diagnostics(Ca_occ, Cb_occ, S, nalpha, nbeta):
    """Return ``(<S^2>, spin contamination)`` for one UHF determinant.

    The occupied-orbital overlap may be complex even though the current SCF
    driver uses real orbitals, so its squared Frobenius norm is deliberately
    evaluated with :func:`numpy.vdot`.
    """

    S_ab = Ca_occ.conj().T @ S @ Cb_occ
    overlap_norm = float(np.real(np.vdot(S_ab, S_ab)))

    m_s = 0.5 * (nalpha - nbeta)
    s_squared = m_s**2 + 0.5 * (nalpha + nbeta) - overlap_norm

    s_target = 0.5 * abs(nalpha - nbeta)
    spin_contamination = s_squared - s_target * (s_target + 1.0)
    return float(np.real(s_squared)), float(np.real(spin_contamination))


def _diagonalize_fock(A, F, nocc):
    """Diagonalize an AO Fock matrix and form its unscaled spin density."""

    F_p = A.dot(F).dot(A)
    e, C_p = np.linalg.eigh(F_p)
    C = A.dot(C_p)
    C_occ = C[:, :nocc]
    D = np.einsum("pi,qi->pq", C_occ, C_occ, optimize=True)
    return C, D


def _diis_extrapolate(F_list_a, F_list_b, R_list_a, R_list_b):
    """Shared-coefficient UHF DIIS extrapolation.

    Extrapolating alpha and beta independently lets the two spin channels pull
    against each other, which stalls open-shell cases with near-degenerate
    singly-occupied orbitals (e.g. the pi hole in OH).  Instead build a single
    Pulay B matrix from the combined error,

        B_ij = <r^a_i | r^a_j> + <r^b_i | r^b_j>,

    and apply the resulting coefficients to both Fock series.
    """
    n = len(F_list_a)
    B = np.empty((n + 1, n + 1))
    B[-1, :] = -1
    B[:, -1] = -1
    B[-1, -1] = 0
    for i in range(n):
        for j in range(n):
            B[i, j] = np.einsum(
                "ij,ij->", R_list_a[i], R_list_a[j], optimize=True
            ) + np.einsum("ij,ij->", R_list_b[i], R_list_b[j], optimize=True)

    # Build RHS of Pulay equation
    rhs = np.zeros((n + 1))
    rhs[-1] = -1

    # Solve Pulay equation for c_i's with NumPy
    coeff = np.linalg.solve(B, rhs)

    # Build DIIS Fock matrices with the shared coefficients
    Fa_DIIS = np.zeros_like(F_list_a[0])
    Fb_DIIS = np.zeros_like(F_list_b[0])
    for x in range(n):
        Fa_DIIS += coeff[x] * F_list_a[x]
        Fb_DIIS += coeff[x] * F_list_b[x]

    return Fa_DIIS, Fb_DIIS


class CQEDUSCF:
    """Unrestricted CQED-SCF driver following the module result contract."""

    def __init__(self, geometry: Any, config: CQEDConfig, method: str | None = None):

        self.geometry = geometry
        self.config = config
        self.functional = config.base_scf_functional
        self.density_fitting = config.density_fitting
        self.psi4_options = dict(config.psi4_options)

        self.method = config.reference
        if self.method not in ("uhf", "uks"):
            raise ValueError("CQEDUSCF requires config.reference='uhf' or 'uks'")
        if method is not None and method.lower() != self.method:
            raise ValueError(
                f"method={method!r} conflicts with config.reference={self.method!r}"
            )
        if self.method == "uks" and self.functional is None:
            raise ValueError(
                "config.base_scf_functional is required for reference='uks'"
            )

        self.is_dft = self.method == "uks"
        self.x_alpha = 1.0
        self.x_beta = 0.0
        self.jk = None
        self.Vpot = None
        self.jk_finalized = True
        self.vpot_finalized = True

    def run(self):
        """Run unrestricted CQED-SCF.

        Returns
        -------
        tuple[float, dict]
            The CQED-SCF energy and the results dictionary described in the
            module docstring.
        """

        if self.config.quiet and not output.is_quiet():
            with output.quiet_context(True):
                return self._run_managed()
        return self._run_managed()

    def _run_managed(self):
        """Run the calculation while guaranteeing builder finalization."""

        output.banner("Unrestricted Open-Shell CQED-SCF Calculation")
        self._prepare_options()
        try:
            return self._run()
        finally:
            self._finalize_builders()

    def _run(self):
        """Implementation body separated so Psi4 builders always finalize."""

        self.mol = psi4.geometry(self.geometry)

        ref_method = self._reference_method_string()
        E_psi4, self.wfn = psi4.energy(ref_method, return_wfn=True)

        mints = psi4.core.MintsHelper(self.wfn.basisset())
        S = np.array(mints.ao_overlap(), copy=True)

        nbf = self.wfn.nso()
        nalpha = self.wfn.nalpha()
        nbeta = self.wfn.nbeta()

        Ca = np.array(self.wfn.Ca(), copy=True)
        Da = np.array(self.wfn.Da(), copy=True)
        Cb = np.array(self.wfn.Cb(), copy=True)
        Db = np.array(self.wfn.Db(), copy=True)

        eps_reference_a = np.array(self.wfn.epsilon_a(), copy=True)
        eps_reference_b = np.array(self.wfn.epsilon_b(), copy=True)

        T = np.asarray(mints.ao_kinetic())
        V = np.asarray(mints.ao_potential())
        H_canonical = T + V

        A = mints.ao_overlap()
        A.power(-0.5, 1.0e-16)
        A = np.asarray(A)

        mu = [np.asarray(x) for x in mints.ao_dipole()]
        d_ao = sum(
            lambda_i * mu_i for lambda_i, mu_i in zip(self.config.lambda_vector, mu)
        )

        Q = [np.asarray(x) for x in mints.ao_quadrupole()]
        l = self.config.lambda_vector
        Q_PF = -0.5 * (l[0] ** 2 * Q[0] + l[1] ** 2 * Q[3] + l[2] ** 2 * Q[5]) - (
            l[0] * l[1] * Q[1] + l[0] * l[2] * Q[2] + l[1] * l[2] * Q[4]
        )

        # Simplified Pauli--Fierz one-electron Hamiltonian.  The coherent-state
        # -<d>d term cancels the DSE Coulomb contribution exactly, so neither
        # belongs in the production unrestricted Fock build.
        H = H_canonical + Q_PF
        E_nuc = self.mol.nuclear_repulsion_energy()

        if self.is_dft:
            self._build_vbase()
        self._build_jk()
        self._print_calculation_metadata(nbf, nalpha, nbeta)

        SCF_E = 0.0
        SCF_E_old = 0.0

        # Trial and residual lists for DIIS
        F_list_a = []
        F_list_b = []
        R_list_a = []
        R_list_b = []

        e_conv = self.psi4_options.get("e_convergence", 1.0e-10)
        d_conv = self.psi4_options.get("d_convergence", 1.0e-8)
        max_iter = int(self.psi4_options.get("maxiter", 100))

        # DIIS controls: keep the subspace small so the B matrix stays well
        # conditioned, and start extrapolating early
        diis_max = 8
        diis_start = 2
        iterations = []

        for it in range(1, max_iter + 1):
            Cocc_a = Ca[:, :nalpha]
            Cocc_b = Cb[:, :nbeta]
            J_a, J_b, K_a, K_b, wK_a, wK_b = self._build_JK(Cocc_a, Cocc_b)
            J = J_a + J_b

            if self.method == "uhf":
                N_a = d_ao @ Da @ d_ao
                N_b = d_ao @ Db @ d_ao
                Fa = H + J - K_a - N_a
                Fb = H + J - K_b - N_b
            else:
                N_a = d_ao @ Da @ d_ao
                N_b = d_ao @ Db @ d_ao
                Exc, Vxc_a, Vxc_b = self._build_xc(Da, Db)
                Fa = H + J + Vxc_a - N_a
                Fb = H + J + Vxc_b - N_b
                if K_a is not None:
                    Fa -= self.x_alpha * K_a
                    Fb -= self.x_alpha * K_b
                if wK_a is not None:
                    Fa -= self.x_beta * wK_a
                    Fb -= self.x_beta * wK_b

            # DIIS error vectors (orthogonalized FDS - SDF)
            diis_r_a = A.dot(Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)).dot(A)
            diis_r_b = A.dot(Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)).dot(A)

            # Append trial and residual vectors to lists
            F_list_a.append(Fa)
            F_list_b.append(Fb)
            R_list_a.append(diis_r_a)
            R_list_b.append(diis_r_b)

            # Drop the oldest vectors once the subspace is full.  An unbounded
            # subspace makes every iteration more expensive and drives the B
            # matrix toward singularity as the error vectors become linearly
            # dependent.
            if len(F_list_a) > diis_max:
                for _list in (F_list_a, F_list_b, R_list_a, R_list_b):
                    del _list[0]

            # Evaluate the spin-resolved energy with the same unextrapolated
            # J/K matrices and densities that produced the Focks above.
            Dt = Da + Db
            if self.method == "uhf":
                SCF_E = (
                    np.trace(H @ Dt)
                    + 0.5 * np.trace(J @ Dt)
                    - 0.5 * np.trace(K_a @ Da)
                    - 0.5 * np.trace(K_b @ Db)
                    - 0.5 * np.trace(N_a @ Da)
                    - 0.5 * np.trace(N_b @ Db)
                    + E_nuc
                )
            else:
                SCF_E = (
                    np.trace(Dt @ H)
                    + 0.5 * np.trace(Dt @ J)
                    + Exc
                    + E_nuc
                    - 0.5 * np.trace(Da @ N_a)
                    - 0.5 * np.trace(Db @ N_b)
                )
                if K_a is not None:
                    SCF_E -= (
                        0.5 * self.x_alpha * (np.trace(Da @ K_a) + np.trace(Db @ K_b))
                    )
                if wK_a is not None:
                    SCF_E -= (
                        0.5 * self.x_beta * (np.trace(Da @ wK_a) + np.trace(Db @ wK_b))
                    )

            dE = SCF_E - SCF_E_old
            dRMS = 0.5 * (np.mean(diis_r_a**2) + np.mean(diis_r_b**2)) ** 0.5
            converged = abs(dE) < e_conv and dRMS < d_conv
            iterations.append((it, SCF_E, dE, dRMS, converged))

            if converged:
                # Canonicalize the converged, unextrapolated CQED Focks.  DIIS
                # is only an iteration accelerator; its extrapolated operator
                # must not define the reported orbitals or orbital energies.
                eps_a, Ca, Da = self._canonicalize(Fa, A, nalpha)
                eps_b, Cb, Db = self._canonicalize(Fb, A, nbeta)
                break

            SCF_E_old = SCF_E

            # DIIS Extrapolation
            if it >= diis_start:
                Fa, Fb = _diis_extrapolate(F_list_a, F_list_b, R_list_a, R_list_b)

            # compute new orbital guess
            Ca, Da = _diagonalize_fock(A, Fa, nalpha)
            Cb, Db = _diagonalize_fock(A, Fb, nbeta)

            if it == max_iter:
                raise RuntimeError(f"SCF did not converge in {max_iter} iterations.")

        self._print_scf_iterations(iterations)

        # Compute diagnostics from the final converged CQED orbitals, not the
        # original Psi4 reference orbitals used as the initial guess.
        s_squared, spin_contamination = _spin_diagnostics(
            Ca[:, :nalpha], Cb[:, :nbeta], S, nalpha, nbeta
        )
        jk_type = self.jk.name()
        effective_scf_type = (
            "df"
            if self.density_fitting
            else str(self.psi4_options.get("scf_type", "pk")).lower()
        )

        results = {
            "energy_scf": SCF_E,
            "energy_psi4": E_psi4,
            "Ca": Ca,
            "Cb": Cb,
            "Da": Da,
            "Db": Db,
            "Coa": Ca[:, :nalpha],
            "Cob": Cb[:, :nbeta],
            "Fa": Fa,
            "Fb": Fb,
            "orbital_energies_a": eps_a,
            "orbital_energies_b": eps_b,
            "canonical_orbital_energies_a": eps_reference_a,
            "canonical_orbital_energies_b": eps_reference_b,
            "reference_orbital_energies_a": eps_reference_a.copy(),
            "reference_orbital_energies_b": eps_reference_b.copy(),
            "nalpha": nalpha,
            "nbeta": nbeta,
            "nmo": nbf,
            "natom": self.mol.natom(),
            "s_squared": s_squared,
            "spin_contamination": spin_contamination,
            "method": self.method,
            "reference": self.method,
            "functional": self.functional,
            "density_fitting": self.density_fitting,
            "scf_type": effective_scf_type,
            "jk_type": jk_type,
            "x_alpha": self.x_alpha,
            "x_beta": self.x_beta,
            "omega": self.config.omega,
            "lambda_vector": np.asarray(self.config.lambda_vector).copy(),
            "mints": mints,
            "wfn": self.wfn,
        }

        self._print_final_results(SCF_E, s_squared, spin_contamination)
        return SCF_E, results

    @staticmethod
    def _canonicalize(F, A, nocc):
        """Return eigenvalues, orbitals, and density for an AO Fock matrix."""

        eps, C_orth = np.linalg.eigh(A @ F @ A)
        C = A @ C_orth
        Cocc = C[:, :nocc]
        D = Cocc @ Cocc.T
        return eps, C, D

    def _build_vbase(self):
        """Initialize an unrestricted numerical XC potential."""

        self.Vpot = psi4.core.VBase.build(
            self.wfn.basisset(), self.wfn.functional(), "UV"
        )
        self.Vpot.initialize()
        self.vpot_finalized = False
        functional = self.Vpot.functional()
        self.x_alpha = float(functional.x_alpha())
        self.x_beta = float(functional.x_beta())

    def _build_jk(self):
        """Initialize Psi4's integral-direct or density-fitted JK builder."""

        functional = self.Vpot.functional() if self.is_dft else None
        self.jk = psi4.core.JK.build(self.wfn.basisset())
        self.jk.set_memory(int(5e8))

        self._do_K = not self.is_dft
        self._do_wK = False
        if hasattr(self.jk, "set_do_J"):
            self.jk.set_do_J(True)
        if self.is_dft:
            # x_alpha multiplies short-range/global exact exchange.  x_beta is
            # meaningful for JK only when the functional actually reports an
            # LRC component (pure functionals may still report x_beta=1).
            self._do_K = self.x_alpha != 0.0
            self._do_wK = functional.is_x_lrc() and self.x_beta != 0.0
            if hasattr(self.jk, "set_do_K"):
                self.jk.set_do_K(self._do_K)
            if hasattr(self.jk, "set_do_wK"):
                self.jk.set_do_wK(self._do_wK)
            if self._do_wK:
                omega = float(functional.x_omega())
                self.jk.set_omega(omega)
        else:
            if hasattr(self.jk, "set_do_K"):
                self.jk.set_do_K(True)
            if hasattr(self.jk, "set_do_wK"):
                self.jk.set_do_wK(False)

        self.jk.initialize()
        self.jk_finalized = False

    def _build_JK(self, Cocc_a, Cocc_b):
        """Build spin-resolved J/K matrices in one Psi4 JK batch."""

        self.jk.C_clear()
        self.jk.C_left_add(psi4.core.Matrix.from_array(Cocc_a))
        self.jk.C_left_add(psi4.core.Matrix.from_array(Cocc_b))
        self.jk.compute()

        J = [np.array(matrix, copy=True) for matrix in self.jk.J()]

        K = [None, None]
        if self._do_K:
            K_p4 = self.jk.K()
            if K_p4 is not None and len(K_p4) == 2:
                K = [np.array(matrix, copy=True) for matrix in K_p4]

        wK = [None, None]
        if self._do_wK and hasattr(self.jk, "wK"):
            wK_p4 = self.jk.wK()
            if wK_p4 is not None and len(wK_p4) == 2:
                wK = [np.array(matrix, copy=True) for matrix in wK_p4]

        return J[0], J[1], K[0], K[1], wK[0], wK[1]

    def _build_xc(self, Da, Db):
        """Return unrestricted XC energy and alpha/beta AO potentials."""

        Da_p4 = psi4.core.Matrix.from_array(Da)
        Db_p4 = psi4.core.Matrix.from_array(Db)
        Va_p4 = psi4.core.Matrix(self.wfn.nso(), self.wfn.nso())
        Vb_p4 = psi4.core.Matrix(self.wfn.nso(), self.wfn.nso())

        self.Vpot.set_D([Da_p4, Db_p4])
        self.Vpot.compute_V([Va_p4, Vb_p4])
        Exc = float(self.Vpot.quadrature_values()["FUNCTIONAL"])
        return Exc, np.array(Va_p4, copy=True), np.array(Vb_p4, copy=True)

    def _finalize_builders(self):
        """Release Psi4 JK and VBase resources when their APIs provide it."""

        if self.jk is not None and not self.jk_finalized:
            finalize = getattr(self.jk, "finalize", None)
            if finalize is not None:
                finalize()
            self.jk_finalized = True

        if self.Vpot is not None and not self.vpot_finalized:
            finalize = getattr(self.Vpot, "finalize", None)
            if finalize is not None:
                finalize()
            self.vpot_finalized = True

    def _print_calculation_metadata(self, nbf, nalpha, nbeta):
        """Emit unrestricted calculation metadata through the output layer."""

        lambda_str = (
            "(" + ", ".join(f"{value:.6f}" for value in self.config.lambda_vector) + ")"
        )
        rows = [
            ("Method", self.method.upper()),
            ("Functional", self.functional or "HF"),
            ("Basis", self.wfn.basisset().name()),
            ("Charge", self.mol.molecular_charge()),
            ("Multiplicity", self.mol.multiplicity()),
            ("Alpha / beta electrons", f"{nalpha} / {nbeta}"),
            ("Orbitals (AO / MO)", f"{nbf} / {nbf}"),
            ("lambda", lambda_str),
            ("omega", f"{self.config.omega:.6f} Eh"),
            ("Density fitting", self.density_fitting),
            ("JK builder", self.jk.name()),
        ]
        output.table(["Setting", "Value"], rows, [26, 28])

    @staticmethod
    def _print_scf_iterations(iterations):
        """Emit the unrestricted SCF iteration table."""

        output.banner("Unrestricted SCF Iterations")
        rows = [
            [
                str(it),
                f"{energy:16.12f}",
                f"{delta_e:14.6e}",
                f"{rms:14.6e}",
                "*" if converged else "",
            ]
            for it, energy, delta_e, rms, converged in iterations
        ]
        output.table(
            ["Iter", "Energy (Eh)", "dE (Eh)", "dRMS (a.u.)", ""],
            rows,
            [6, 20, 16, 16, 2],
        )
        output.echo(f"  SCF Converged in {len(iterations)} iterations.")

    def _print_final_results(self, energy, s_squared, spin_contamination):
        """Emit final energy and spin diagnostics."""

        output.banner(f"CQED-{self.method.upper()} Total Energy")
        output.property_("CQED-SCF Total Energy", energy, "Eh")
        output.property_("<S^2>", s_squared, "")
        output.property_("Spin Contamination", spin_contamination, "")

    def _prepare_options(self):
        opts = dict(self.config.psi4_options)

        if self.method == "uhf":
            opts["reference"] = "uhf"
        else:
            opts["reference"] = "uks"

        if self.density_fitting:
            opts["scf_type"] = "df"

        psi4.set_options(opts)

    def _reference_method_string(self):
        if self.method == "uhf":
            return "scf"
        return self.functional


CQEDUHFSCF = CQEDUSCF
