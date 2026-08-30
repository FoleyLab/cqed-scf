"""Disposable diagnostic: pin down Psi4's twoel_Hx_full convention.

Psi4HxERIEngine assumes

    A_2e X = C_o^T ( 2 J_like - K_like ) C_v

with J_like / K_like returned in the SO basis by
``wfn.twoel_Hx_full(vectors, False, "SO", singlet)``, following
psi4/driver/procrouting/response/scf_products.py::_combine_A.

For a Hartree-Fock reference that must equal the exact MO expression
2(ia|jb) X_jb - (ij|ab) X_jb.  This script builds the exact answer from MO
integrals and reports which combination -- if any -- reproduces it, plus the
shapes actually returned.  Run it and paste the output.

    python docs/development/diagnose_twoel_hx.py

Delete this file once the convention is settled.
"""

import numpy as np
import psi4

psi4.core.be_quiet()

GEOM = """
0 1
O   0.000000000000   0.000000000000  -0.068516219320
H   0.000000000000  -0.790689573744   0.543701060715
H   0.000000000000   0.790689573744   0.543701060715
no_reorient
no_com
symmetry c1
"""

psi4.set_options({
    "basis": "sto-3g", "scf_type": "pk",
    "e_convergence": 1e-10, "d_convergence": 1e-10, "save_jk": True,
})
psi4.geometry(GEOM)
_, wfn = psi4.energy("scf", return_wfn=True)

C = np.asarray(wfn.Ca())
no = wfn.nalpha()
nmo = wfn.nmo()
nv = nmo - no
Co, Cv = np.ascontiguousarray(C[:, :no]), np.ascontiguousarray(C[:, no:])
print(f"nbf={C.shape[0]}  nmo={nmo}  nocc={no}  nvir={nv}")

rng = np.random.default_rng(0)
X = rng.normal(size=(no, nv))

# ---- exact reference from MO integrals -------------------------------------
mints = psi4.core.MintsHelper(wfn.basisset())
Co4, Cv4 = psi4.core.Matrix.from_array(Co), psi4.core.Matrix.from_array(Cv)
ovov = np.asarray(mints.mo_eri(Co4, Cv4, Co4, Cv4))
oovv = np.asarray(mints.mo_eri(Co4, Co4, Cv4, Cv4))
reference = 2.0 * np.einsum("iajb,jb->ia", ovov, X) - np.einsum("ijab,jb->ia", oovv, X)
print(f"\nreference ||2(ia|jb)X - (ij|ab)X|| = {np.linalg.norm(reference):.8f}")

# ---- what does twoel_Hx_full actually return? ------------------------------
vectors = [psi4.core.Matrix.from_array(np.ascontiguousarray(X))]
print("\nwfn has twoel_Hx_full:", hasattr(wfn, "twoel_Hx_full"),
      " twoel_Hx:", hasattr(wfn, "twoel_Hx"))
try:
    functional = wfn.functional()
    print("functional: is_x_hybrid", functional.is_x_hybrid(),
          " is_x_lrc", functional.is_x_lrc(), " needs_xc", functional.needs_xc())
except Exception as exc:
    print("wfn.functional() unavailable:", exc)

for name, args in [
    ("twoel_Hx_full(v, False, 'SO', True)", (vectors, False, "SO", True)),
    ("twoel_Hx_full(v, False, 'MO', True)", (vectors, False, "MO", True)),
    ("twoel_Hx(v, False, 'SO')",            (vectors, False, "SO")),
    ("twoel_Hx(v, False, 'MO')",            (vectors, False, "MO")),
]:
    method = "twoel_Hx_full" if "full" in name else "twoel_Hx"
    if not hasattr(wfn, method):
        continue
    try:
        out = getattr(wfn, method)(*args)
    except Exception as exc:
        print(f"\n{name}: raised {type(exc).__name__}: {exc}")
        continue

    shapes = [np.asarray(o).shape for o in out]
    print(f"\n{name}: returned {len(out)} matrices, shapes {shapes}")

    mats = [np.asarray(o) for o in out]
    def to_ov(G):
        if G.shape == (no, nv):
            return G
        if G.shape == C.shape[:1] * 2 or G.shape == (C.shape[0], C.shape[0]):
            return Co.T @ G @ Cv
        return None

    if len(mats) == 1:
        trials = {"J": mats[0], "2J": 2 * mats[0]}
    else:
        J, K = mats[0], mats[1]
        trials = {
            "2J - K": 2 * J - K,
            "2J - K^T": 2 * J - K.T,
            "J - K": J - K,
            "4J - K - K^T": 4 * J - K - K.T,
            "2J": 2 * J,
            "-K": -K,
        }
    for label, G in trials.items():
        got = to_ov(G)
        if got is None:
            print(f"    {label:14s} unexpected shape {G.shape}")
            continue
        print(f"    {label:14s} max|got - reference| = {np.max(np.abs(got - reference)):.3e}"
              f"   ratio = {np.linalg.norm(got) / np.linalg.norm(reference):.6f}")
