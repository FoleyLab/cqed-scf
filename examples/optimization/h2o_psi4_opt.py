import psi4

h2o_string = """
O  0.000000000000   0.000000000000  -0.068516219320
H  0.000000000000  -0.790689573744   0.543701060715
H  0.000000000000   0.790689573744   0.543701060715
units angstrom
no_reorient
no_com
symmetry c1
"""

# =========================
# Psi4 options
# =========================

psi4_options = {
    "basis": "cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
    "g_convergence": "GAU_VERYTIGHT"
}

psi4.set_options(psi4_options)
mol = psi4.geometry(h2o_string)
psi4.optimize("wb97x-d")                                 
