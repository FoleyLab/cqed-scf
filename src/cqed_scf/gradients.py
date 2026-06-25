import numpy as np
import opt_einsum as oe
import psi4
import time 


class CQEDGradient:
    """
    Analytic nuclear gradient for CQED-RHF.

    Parameters
    ----------
    lambda_vector : array-like, shape (3,)
        Light–matter coupling direction.
    canonical : {'psi4', 'exact'}
        How to compute the canonical RHF gradient.
    debug : bool
        If True, print individual gradient contributions.
    """

    def __init__(self, lambda_vector, canonical="psi4", debug=False):
        self.lambda_vector = np.asarray(lambda_vector)
        self.canonical = canonical
        self.debug = debug


    # =========================
    # Public driver
    # =========================

    def compute(self, scf):
        if self.canonical == "psi4":
            t0 = time.time()
            canonical_grad = self._canonical_gradient_psi4(scf)
            print("Psi4 canonical gradient time: {:.4f} s".format(time.time() - t0))
        elif self.canonical == "exact":
            t0 = time.time()
            canonical_grad = self._canonical_gradient_exact(scf)
            print("Exact canonical gradient time: {:.4f} s".format(time.time() - t0))
        else:
            raise ValueError("canonical must be 'psi4' or 'exact'")
        
        t0 = time.time()
        quad_grad = self._quadrupole_gradient(scf)
        print("Quadrupole gradient time: {:.4f} s".format(time.time() - t0))
        t0 = time.time()
        dipole_grad = self._dipole_dipole_gradient(scf)
        print("Dipole–dipole gradient time: {:.4f} s".format(time.time() - t0))

        grad = canonical_grad + quad_grad + dipole_grad

        results = dict(
            canonical_grad=canonical_grad,
            quadrupole_grad=quad_grad,
            dipole_dipole_grad=dipole_grad,
            total_grad=grad,
        )
        return results

    # =========================
    # Canonical RHF gradients
    # =========================

    def _canonical_gradient_psi4(self, scf):
        # wfn = self._update_wfn_with_cqed(scf) #<-- update wfn to reflect CQED-RHF density
        wfn = scf["wfn"]
        if self.debug:
            D_wfn = np.asarray(wfn.Da())
            assert np.allclose(D_wfn, scf["density"], atol=1e-10)

        return np.asarray(psi4.core.scfgrad(wfn))

    def _canonical_gradient_exact(self, scf):
        D = 2.0 * scf["density"]
        C = scf["coefficients"]
        C_p4 = psi4.core.Matrix.from_array(C)

        F = scf["F"]
        F_MO = C.T @ F @ C

        mints = scf["mints"]
        natom = scf["natom"]
        ndocc = scf["ndocc"]

        grad = np.zeros((natom, 3))

        if self.debug:
            S_grad = np.zeros_like(grad)
            T_grad = np.zeros_like(grad)
            V_grad = np.zeros_like(grad)
            J_grad = np.zeros_like(grad)
            K_grad = np.zeros_like(grad)
            nuc_grad = np.zeros_like(grad)

        # --- Overlap (Pulay) ---
        for A in range(natom):
            Sder = mints.mo_oei_deriv1("OVERLAP", A, C_p4, C_p4)
            for cart in range(3):
                Sder_np = Sder[cart].np
                val = -2.0 * np.trace(
                    F_MO[:ndocc, :ndocc] @ Sder_np[:ndocc, :ndocc]
                )
                grad[A, cart] += val
                if self.debug:
                    S_grad[A, cart] += val

        # --- One-electron (T + V) ---
        for A in range(natom):
            Tder = mints.ao_oei_deriv1("KINETIC", A)
            Vder = mints.ao_oei_deriv1("POTENTIAL", A)
            for cart in range(3):
                #tval = np.einsum("uv,uv", D, Tder[cart])
                tval = oe.contract("uv,uv->", D, np.asarray(Tder[cart]), optimize="optimal")
                #vval = np.einsum("uv,uv", D, Vder[cart])
                vval = oe.contract("uv,uv->", D, np.asarray(Vder[cart]), optimize="optimal")
                grad[A, cart] += tval + vval
                if self.debug:
                    T_grad[A, cart] += tval
                    V_grad[A, cart] += vval

        # --- Two-electron (J + K) ---
        nbf = D.shape[0]
        for A in range(natom):
            tei = mints.ao_tei_deriv1(A)
            for cart in range(3):
                eri = np.asarray(tei[cart]).reshape(nbf, nbf, nbf, nbf)
                #J = np.einsum("uvls,ls->uv", eri, D, optimize="optimal")
                J = oe.contract("uvls,ls->uv", eri, D, optimize="optimal")
                #K = -0.5 * np.einsum("ulvs,ls->uv", eri, D, optimize="optimal")
                K = -0.5 * oe.contract("ulvs,ls->uv", eri, D, optimize="optimal")
                #jval = 0.5 * np.einsum("uv,uv", D, J, optimize="optimal")
                #kval = 0.5 * np.einsum("uv,uv", D, K, optimize="optimal")
                jval = 0.5 * oe.contract("uv,uv->", D, J, optimize="optimal")
                kval = 0.5 * oe.contract("uv,uv->", D, K, optimize="optimal")
                grad[A, cart] += jval + kval
                if self.debug:
                    J_grad[A, cart] += jval
                    K_grad[A, cart] += kval

        # --- Nuclear repulsion ---
        mol = scf["wfn"].molecule()
        nuc = np.asarray(mol.nuclear_repulsion_energy_deriv1())
        grad += nuc
        if self.debug:
            nuc_grad += nuc

        if self.debug:
            self._print_canonical_components(
                S_grad, T_grad, V_grad, J_grad, K_grad, nuc_grad
            )

        return grad

    # =========================
    # CQED-specific gradients
    # =========================

    def _quadrupole_gradient(self, scf):
        D = scf["density"]
        mints = scf["mints"]
        natom = scf["natom"]
        lam = self.lambda_vector

        multipole = np.asarray(
            mints.multipole_grad(
                psi4.core.Matrix.from_array(2.0 * D),
                2,
                [0.0, 0.0, 0.0],
            )
        )

        grad = np.zeros((natom, 3))
        for A in range(natom):
            for cart in range(3):
                i = 3 * A + cart
                grad[A, cart] -= 0.5 * lam[0] ** 2 * multipole[i, 3]
                grad[A, cart] -= 0.5 * lam[1] ** 2 * multipole[i, 6]
                grad[A, cart] -= 0.5 * lam[2] ** 2 * multipole[i, 8]
                grad[A, cart] -= lam[0] * lam[1] * multipole[i, 4]
                grad[A, cart] -= lam[0] * lam[2] * multipole[i, 5]
                grad[A, cart] -= lam[1] * lam[2] * multipole[i, 7]

        if self.debug:
            print("\nQuadrupole gradient:\n", grad)

        return grad

    def _dipole_dipole_gradient(self, scf):
        D = scf["density"]
        d_ao = scf["d_ao"]
        mints = scf["mints"]
        natom = scf["natom"]
        lam = self.lambda_vector

        grad = np.zeros((natom, 3))

        for A in range(natom):
            dip = np.asarray(mints.ao_elec_dip_deriv1(A))
            for cart in range(3):
                if cart == 0:
                    dprime = lam[0] * dip[0] + lam[1] * dip[3] + lam[2] * dip[6]
                elif cart == 1:
                    dprime = lam[0] * dip[1] + lam[1] * dip[4] + lam[2] * dip[7]
                else:
                    dprime = lam[0] * dip[2] + lam[1] * dip[5] + lam[2] * dip[8]

                Kmat = -oe.contract("us,lv,ls->uv", dprime, d_ao, D, optimize="optimal")
                grad[A, cart] += 2.0 * oe.contract("uv,uv->", D, Kmat, optimize="optimal")

        if self.debug:
            print("\nDipole–dipole gradient:\n", grad)

        return grad

    # =========================
    # Debug helpers
    # =========================

    @staticmethod
    def _print_canonical_components(S, T, V, J, K, N):
        print("\nCanonical gradient components:")
        print("Overlap (Pulay):\n", S)
        print("Kinetic:\n", T)
        print("Potential:\n", V)
        print("Coulomb (J):\n", J)
        print("Exchange (K):\n", K)
        print("Nuclear repulsion:\n", N)


# backward-compatible alias
CQEDRHFGradient = CQEDGradient
