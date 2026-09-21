"""Same prototype, but with damping + explicit convergence reporting, and both
sign choices for the one-electron correction, so the conclusion is trustworthy."""
import sys; sys.path.insert(0, "src")
import numpy as np, psi4, opt_einsum as oe
psi4.core.be_quiet()
LAM = np.array([0.0, 0.0, 0.1]); BASIS = "cc-pvdz"

def cqed_rhf(z, mode):
    psi4.core.clean()
    psi4.geometry(f"O 0.0 0.0 {z}\nH 0.0 0.757 {z+0.587}\nH 0.0 -0.757 {z+0.587}"
                  "\nsymmetry c1\nunits angstrom\nno_reorient\nno_com")
    psi4.set_options({"basis": BASIS, "scf_type": "pk",
                      "e_convergence": 1e-12, "d_convergence": 1e-12})
    _, wfn = psi4.energy("scf", return_wfn=True)
    mints = psi4.core.MintsHelper(wfn.basisset())
    S = np.asarray(mints.ao_overlap())
    H0 = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    mu = np.asarray(mints.ao_dipole()); d_ao = sum(LAM[i]*mu[i] for i in range(3))
    Q = [np.asarray(x) for x in mints.ao_quadrupole()]
    Q_PF = (-0.5*LAM[0]**2*Q[0] - 0.5*LAM[1]**2*Q[3] - 0.5*LAM[2]**2*Q[5]
            - LAM[0]*LAM[1]*Q[1] - LAM[0]*LAM[2]*Q[2] - LAM[1]*LAM[2]*Q[4])
    ndocc = wfn.nalpha(); nel = 2*ndocc
    A = psi4.core.Matrix.from_array(S); A.power(-0.5, 1e-16); A = np.asarray(A)
    I = np.asarray(mints.ao_eri())
    C = np.array(wfn.Ca(), copy=True); D = C[:, :ndocc] @ C[:, :ndocc].T

    conv = None
    for it in range(2000):
        J = oe.contract("pqrs,rs->pq", I, D, optimize="optimal")
        K = oe.contract("prqs,rs->pq", I, D, optimize="optimal")
        if mode == "bare":
            d_use, H_use = d_ao, H0 + Q_PF
        else:
            c = 2.0*np.einsum("pq,pq->", d_ao, D) / nel
            d_use = d_ao - c*S
            sgn = +1.0 if mode == "fluct+" else -1.0
            H_use = H0 + Q_PF + sgn*(c*d_ao - 0.5*c*c*S)
        N = oe.contract("pr,qs,rs->pq", d_use, d_use, D, optimize="optimal")
        F = H_use + 2.0*J - K - N
        eps, Cp = np.linalg.eigh(A @ F @ A)
        C = A @ Cp
        Dn = C[:, :ndocc] @ C[:, :ndocc].T
        err = np.abs(Dn - D).max()
        D = 0.7*D + 0.3*Dn if it < 200 else Dn      # damped early
        if err < 1e-11:
            conv = it; break
    return eps, D, conv, ndocc

for mode in ("bare", "fluct+", "fluct-"):
    e0, D0, c0, no = cqed_rhf(0.0, mode)
    e1, D1, c1, _ = cqed_rhf(20.0, mode)
    ok = "converged" if (c0 is not None and c1 is not None) else "NOT CONVERGED"
    print(f"{mode:7s} [{ok:13s} it={c0},{c1}]  max|dD|={np.abs(D1-D0).max():8.2e}  "
          f"max|deps|={np.abs(e1-e0).max():9.3e}  "
          f"gap {e0[no]-e0[no-1]:.8f} -> {e1[no]-e1[no-1]:.8f}")
