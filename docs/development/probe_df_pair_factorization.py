"""Probe: does v(string) factor into exactly two DF pair blocks?

Claim:  v(s0 s1 s2 s3)[A,B,C,D] = sum_Q b^Q_(s0 s2)[A,C] * b^Q_(s1 s3)[B,D]
with the DSE contributing exactly ONE extra auxiliary row, b^dse_(xy) = Cx^T d Cy.
"""
import numpy as np, psi4, sys
sys.path.insert(0, "src")
from cqed_scf import CQEDConfig
from cqed_scf.sapt import QEDSAPT0Driver

psi4.core.be_quiet()

GEOM = """
0 1
O 0.0 0.0 0.0
H 0.0 0.757 0.587
H 0.0 -0.757 0.587
--
0 1
He 0.0 0.0 3.4
symmetry c1
units angstrom
no_reorient
no_com
"""

BASIS = "cc-pvdz"
cfg = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.1]), omega=0.1,
    psi4_options={"basis": BASIS, "scf_type": "pk",
                  "e_convergence": 1e-10, "d_convergence": 1e-10},
    reference="rhf", functional=None, density_fitting=False,
    charge=0, multiplicity=1, dispersion_policy="none", debug=False,
)

drv = QEDSAPT0Driver(dimer_geometry=psi4.geometry(GEOM), config=cfg,
                     integral_backend="full_eri", include_cavity_terms=True)
drv.prepare_monomers()
drv.build_integrals()

# ---- build the augmented B tensor -------------------------------------
mol  = drv.monomer_A.wfn.molecule()
prim = drv.monomer_A.wfn.basisset()
zero = psi4.core.BasisSet.zero_ao_basis_set()
aux  = psi4.core.BasisSet.build(mol, "DF_BASIS_SAPT", "", "RIFIT", BASIS)
mints = psi4.core.MintsHelper(prim)

Ppq    = np.squeeze(mints.ao_eri(aux, zero, prim, prim))
metric = np.squeeze(mints.ao_eri(aux, zero, aux, zero))
M = psi4.core.Matrix.from_array(metric); M.power(-0.5, 1e-12)
B_std = np.einsum("PQ,Qpq->Ppq", np.asarray(M), Ppq, optimize=True)   # (naux, nbf, nbf)

assert np.allclose(drv.d_A, drv.d_B), "rank-one augmentation requires d_A == d_B"
d = drv.d_A
B_tot = np.concatenate([B_std, d[None, :, :]], axis=0)                # (naux+1, nbf, nbf)
naux = B_std.shape[0]
print(f"nbf={prim.nbf()}  naux={naux}  ->  augmented naux+1={B_tot.shape[0]}")

def bpair(pair, context="total"):
    """The proposed new 2-character primitive: DF three-index MO pair block."""
    Bsel = {"standard": B_std, "cavity": B_tot[naux:], "total": B_tot}[context]
    Cx, Cy = drv.orbitals[pair[0]], drv.orbitals[pair[1]]
    return np.einsum("Ppq,pA,qC->PAC", Bsel, Cx, Cy, optimize=True)

def v_df(string, context="total"):
    """v() becomes a one-line contraction of two pair blocks."""
    left  = bpair(string[0] + string[2], context)
    right = bpair(string[1] + string[3], context)
    return np.einsum("PAC,PBD->ABCD", left, right, optimize=True)

# ---- compare against the dense driver ---------------------------------
strings = ["abrs", "arbs", "abab", "abar", "abra", "absb",
           "abbs", "abas", "abrb", "absr", "rsab"]

print(f"\n{'string':8s} {'pairs':12s} {'shape':18s} "
      f"{'std max|err|':>13s} {'cav max|err|':>13s} {'tot max|err|':>13s}")
print("-" * 84)
worst_cav = 0.0
for s in strings:
    pairs = f"({s[0]}{s[2]})({s[1]}{s[3]})"
    row = []
    for ctx in ("standard", "cavity", "total"):
        ref = drv.v(s, context=ctx)
        got = v_df(s, context=ctx)
        err = np.abs(got - ref).max()
        row.append(err)
        if ctx == "cavity":
            worst_cav = max(worst_cav, err)
    shape = "x".join(str(n) for n in drv.v(s).shape)
    print(f"{s:8s} {pairs:12s} {shape:18s} "
          f"{row[0]:13.3e} {row[1]:13.3e} {row[2]:13.3e}")

print("-" * 84)
print(f"worst cavity error across all strings: {worst_cav:.3e}  "
      f"(should be ~machine precision: the DSE row is EXACT)")
