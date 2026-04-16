import psi4
import numpy as np
import opt_einsum as oe
#from pkg_resources import parse_version

if True: #parse_version(psi4.__version__) >= parse_version("1.3a1"):
    build_superfunctional = psi4.driver.dft.build_superfunctional
else:
    build_superfunctional = psi4.driver.dft_funcs.build_superfunctional


class DIISSubspace:
    def __init__(self, max_dim=6):
        self.max_dim = max_dim
        self.errors = []
        self.focks = []

    def add(self, error, fock):
        self.errors.append(np.array(error, copy=True))
        self.focks.append(np.array(fock, copy=True))

        if len(self.errors) > self.max_dim:
            self.errors.pop(0)
            self.focks.pop(0)

    def extrapolate(self):
        n = len(self.errors)
        if n == 0:
            raise RuntimeError("DIISSubspace.extrapolate called with empty subspace.")
        if n == 1:
            return self.focks[0].copy()

        B = np.empty((n + 1, n + 1))
        B[-1, :] = -1.0
        B[:, -1] = -1.0
        B[-1, -1] = 0.0

        for i in range(n):
            for j in range(n):
                B[i, j] = np.dot(self.errors[i].ravel(), self.errors[j].ravel())

        rhs = np.zeros(n + 1)
        rhs[-1] = -1.0

        coeff = np.linalg.solve(B, rhs)[:-1]

        F = np.zeros_like(self.focks[0])
        for c, f in zip(coeff, self.focks):
            F += c * f
        return F


class CQEDSCF:
    """
    CQED-SCF driver using Psi4 JK builds and Psi4NumPy-style VBase for DFT.

    Supports
    --------
    - RHF
    - pure RKS
    - hybrid RKS
    - density fitting via Psi4 SCF_TYPE = DF
    - CQED quadrupole + dipole self-energy terms
    """

    def __init__(
        self,
        geometry,
        lambda_vector,
        psi4_options,
        omega,
        density_fitting=False,
        method=None,
        functional=None,
        debug=False,
    ):
        self.geometry = geometry
        self.lambda_vector = np.asarray(lambda_vector, dtype=float)
        self.psi4_options = dict(psi4_options)
        self.omega = omega
        self.density_fitting = density_fitting
        self.functional = functional
        self.debug = debug

        # infer method if not explicitly provided
        if method is None:
            self.method = "rks" if functional is not None else "rhf"
        else:
            self.method = method.lower()

        if self.method not in ("rhf", "rks", "hybrid"):
            raise ValueError("method must be 'rhf', 'rks', or 'hybrid'")

        if self.method in ("rks", "hybrid") and self.functional is None:
            raise ValueError("functional must be provided for method='rks' or 'hybrid'")

        self.is_dft = self.method in ("rks", "hybrid")

        # populated in run()
        self.mol = None
        self.wfn = None
        self.mints = None
        self.jk = None
        self.Vpot = None
        self.x_alpha = 1.0
        self.D_prev = None
        self.nbf = None
        self.ndocc = None

    # -------------------------
    # main driver
    # -------------------------

    def run(self):
        print("Starting CQED-SCF calculation...")
        print(f"Method: {self.method.upper()}")
        if self.functional is not None:
            print(f"Functional: {self.functional}")
        if self.density_fitting:
            print("Using density fitting through Psi4 JK.")
        else:
            print("Using conventional JK through Psi4 JK.")

        self._prepare_options()

        self.mol = psi4.geometry(self.geometry)

        # get a Psi4 wavefunction carrying the correct reference/method metadata
        ref_method = self._reference_method_string()
        E_psi4, self.wfn = psi4.energy(ref_method, return_wfn=True)

        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        self.nbf = self.wfn.nmo()
        self.ndocc = self.wfn.nalpha()

        # closed-shell only
        if self.wfn.nalpha() != self.wfn.nbeta():
            raise ValueError("CQEDSCF currently assumes a restricted closed-shell reference.")

        # one-electron integrals
        T = np.asarray(self.mints.ao_kinetic())
        V = np.asarray(self.mints.ao_potential())
        S = np.asarray(self.mints.ao_overlap())

        H0 = T + V

        # dipole and quadrupole AO integrals
        mu_ao = np.asarray(self.mints.ao_dipole())
        d_ao = sum(self.lambda_vector[i] * mu_ao[i] for i in range(3))

        Q = [np.asarray(x) for x in self.mints.ao_quadrupole()]
        Q_PF = (
            -0.5 * self.lambda_vector[0] ** 2 * Q[0]
            -0.5 * self.lambda_vector[1] ** 2 * Q[3]
            -0.5 * self.lambda_vector[2] ** 2 * Q[5]
            -self.lambda_vector[0] * self.lambda_vector[1] * Q[1]
            -self.lambda_vector[0] * self.lambda_vector[2] * Q[2]
            -self.lambda_vector[1] * self.lambda_vector[2] * Q[4]
        )

        H = H0 + Q_PF

        # orthogonalizer
        A = psi4.core.Matrix.from_array(S)
        A.power(-0.5, 1.0e-16)
        A = np.asarray(A)

        Enuc = self.mol.nuclear_repulsion_energy()

        # initial guess from Psi4 reference
        C = np.asarray(self.wfn.Ca())
        Cocc = C[:, :self.ndocc]
        D = oe.contract("pi,qi->pq", Cocc, Cocc, optimize="optimal")

        # reuse previous density for MD if available
        if self.D_prev is not None:
            D = self.D_prev.copy()
            Cocc = self._density_to_Cocc_guess(D, A)

        # build JK and, if needed, VBase
        if self.is_dft:
            self._build_vbase()

        self._build_jk()

        if self.debug and self.is_dft:
            f = self.Vpot.functional()
            print("Functional info:")
            print("  x_alpha =", f.x_alpha())
            print("  x_beta  =", f.x_beta())
            print("  x_omega =", f.x_omega())


        diis = DIISSubspace(max_dim=6)
        Eold = 0.0
        F_old = None

        e_conv = self.psi4_options.get("e_convergence", 1.0e-10)
        d_conv = self.psi4_options.get("d_convergence", 1.0e-8)

        for it in range(1, 501):
            # JK from Psi4
            J, K, wK = self._build_JK(Cocc)

            # N from opt_einsum contraction of d_ao, D, d_ao
            N = oe.contract("pr,qs,rs->pq", d_ao, d_ao, D, optimize="optimal")

            if self.method == "rhf":
                F = H + 2.0 * J - K - N

            elif self.is_dft:
                Exc, Vxc = self._build_xc(D)

                F = H + 2.0 * J + Vxc - N

                if K is not None:
                    F -= self.x_alpha * K

                if wK is not None:
                    beta = 1.0 - self.x_alpha
                    F -= beta * wK



            # mild damping in early iterations
            if F_old is not None and it < 5:
                F = 0.5 * F + 0.5 * F_old
            F_old = F.copy()

            # DIIS error
            err = F @ D @ S - S @ D @ F
            diis.add(err, F)

            diis_e = A @ err @ A
            dRMS = np.mean(diis_e**2) ** 0.5

            # energy
            if self.method == "rhf":
                E = oe.contract("pq,pq->", F + H, D, optimize="optimal") + Enuc
            else:

                E = (
                    2.0 * oe.contract("pq,pq->", H, D)
                    + 2.0 * oe.contract("pq,pq->", J, D)
                    + Exc
                    + Enuc
                    - oe.contract("pq,pq->", N, D)
                )
                # if debug, compute components of energy separately for more insight
                if self.debug:
                    E_H = 2.0 * oe.contract("pq,pq->", H, D)
                    E_J = 2.0 * oe.contract("pq,pq->", J, D)
                    E_N = oe.contract("pq,pq->", N, D)
                if K is not None:
                    E -= self.x_alpha * oe.contract("pq,pq->", K, D)
                    if self.debug:
                        E_K = oe.contract("pq,pq->", K, D)

                if wK is not None:
                    beta = 1.0 - self.x_alpha
                    E -= beta * oe.contract("pq,pq->", wK, D)
                    if self.debug:
                        E_wK = oe.contract("pq,pq->", wK, D)

            if self.debug:
                print(
                    f"CQED Iter {it:3d}: "
                    f"E = {E:18.10f}  "
                    f"dE = {E - Eold: .8e}  "
                    f"dRMS = {dRMS: .8e}"
                )
                # print energy components
                print(f"  E_H = {E_H:18.10f}")
                if self.is_dft:
                    print(f"  E_Exc = {Exc:18.10f}")
                print(f"  E_J = {E_J:18.10f}")
                if K is not None:
                    print(f"  E_K = {-self.x_alpha * E_K:18.10f}")
                if wK is not None:
                    print(f"  E_wK = {-beta * E_wK:18.10f}")
                print(f"  E_N = {-E_N:18.10f}") 

            if abs(E - Eold) < e_conv and dRMS < d_conv:
                break
            Eold = E

            if it > 2:
                F = diis.extrapolate()

            # diagonalize
            Fp = A @ F @ A
            eps, C2 = np.linalg.eigh(Fp)
            C = A @ C2
            Cocc = C[:, :self.ndocc]
            D = oe.contract("pi,qi->pq", Cocc, Cocc, optimize="optimal")

        else:
            raise RuntimeError("Maximum number of SCF cycles exceeded.")

        self.D_prev = D.copy()

        mu_el = np.array(
            [2.0 * oe.contract("pq,pq->", mu_ao[i], D, optimize="optimal") for i in range(3)]
        )
        mu_nuc = np.array(
            [self.mol.nuclear_dipole()[0], self.mol.nuclear_dipole()[1], self.mol.nuclear_dipole()[2]]
        )

        d_exp_el = np.dot(self.lambda_vector, mu_el)
        d_nuc = np.dot(self.lambda_vector, mu_nuc)
        d_exp = d_exp_el + d_nuc

        results = dict(
            energy_scf=E,
            energy_psi4=E_psi4,
            density=D,
            coefficients=C,
            orbital_energies=eps,
            mints=self.mints,
            wfn=self.wfn,
            dipole_el=mu_el,
            dipole_nuc=mu_nuc,
            d_ao=d_ao,
            d_exp=d_exp,
            H0=H0,
            F=F,
            ndocc=self.ndocc,
            natom=self.mol.natom(),
            method=self.method,
            functional=self.functional,
            x_alpha=self.x_alpha,
        )

        return E, results

    # -------------------------
    # internal helpers
    # -------------------------

    def _prepare_options(self):
        opts = dict(self.psi4_options)

        if self.method == "rhf":
            opts["reference"] = "rhf"
        else:
            opts["reference"] = "rks"

        if self.density_fitting:
            opts["scf_type"] = "df"

        psi4.set_options(opts)

    def _reference_method_string(self):
        if self.method == "rhf":
            return "scf"
        return self.functional
    
    def _build_jk(self):
        f = self.Vpot.functional() if self.is_dft else None

        self.jk = psi4.core.JK.build(self.wfn.basisset())
        self.jk.set_memory(int(5e8))

        if hasattr(self.jk, "set_do_J"):
            self.jk.set_do_J(True)

        if self.is_dft and f is not None:
            if hasattr(self.jk, "set_do_K"):
                self.jk.set_do_K(f.is_x_hybrid())
            if hasattr(self.jk, "set_do_wK"):
                self.jk.set_do_wK(f.is_x_lrc())

            omega = f.x_omega()
            if omega != 0.0:
                self.jk.set_omega(omega)
        else:
            if hasattr(self.jk, "set_do_K"):
                self.jk.set_do_K(True)
            if hasattr(self.jk, "set_do_wK"):
                self.jk.set_do_wK(False)

        self.jk.initialize()


    def _build_vbase(self):
        sup = self.wfn.functional()
        self.Vpot = psi4.core.VBase.build(self.wfn.basisset(), sup, "RV")
        self.Vpot.initialize()
        self.x_alpha = self.Vpot.functional().x_alpha()


    def _build_JK(self, Cocc_np):
        Cocc_p4 = psi4.core.Matrix.from_array(Cocc_np)

        self.jk.C_clear()
        self.jk.C_left_add(Cocc_p4)
        self.jk.compute()

        J = np.asarray(self.jk.J()[0])

        # --- K ---
        K = None
        K_list = self.jk.K()
        if K_list is not None and len(K_list) > 0:
            K = np.asarray(K_list[0])

        # --- wK ---
        wK = None
        if hasattr(self.jk, "wK"):
            wK_list = self.jk.wK()
            if wK_list is not None and len(wK_list) > 0:
                wK = np.asarray(wK_list[0])

        return J, K, wK
    

    def _build_xc(self, D_np):
        D_p4 = psi4.core.Matrix.from_array(D_np)
        Vxc_p4 = psi4.core.Matrix(self.nbf, self.nbf)

        self.Vpot.set_D([D_p4])
        self.Vpot.compute_V([Vxc_p4])

        Exc = self.Vpot.quadrature_values()["FUNCTIONAL"]
        Vxc = np.asarray(Vxc_p4)

        return Exc, Vxc

    def _density_to_Cocc_guess(self, D, A):
        """
        Build a Cocc guess from a supplied density by diagonalizing an
        effective Fock-like matrix from the old density. This is only used
        when reusing density between MD steps and is intentionally simple.
        """
        # diagonalize D in orthogonal AO basis to get a compact occupied guess
        Dp = A @ D @ A
        evals, evecs = np.linalg.eigh(Dp)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        Cguess = A @ evecs
        return Cguess[:, :self.ndocc]


# backward compatibility for older imports
CQEDRHFSCF = CQEDSCF
CQEDRHFCalculator = CQEDSCF
