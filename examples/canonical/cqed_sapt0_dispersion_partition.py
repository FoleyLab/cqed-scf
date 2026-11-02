"""
QED-SAPT0 dispersion decomposition into Coulomb/DSE contributions.

The dispersion energy is quadratic in the interaction numerator and is split
into Coulomb-Coulomb [c,c], Coulomb-DSE [c,d], and DSE-DSE [d,d] parts.
The partition is computed automatically by ``sapt0_components`` and exposed
through the result metadata.
"""

import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig


# ---------------------------------------------------------
# Dimer geometry
# ---------------------------------------------------------

dimer = """
0 1
He 0.0000000000 0.0000000000 0.0000000000
--
0 1
He 0.0000000000 0.0000000000 2.0000000000
units angstrom
symmetry c1
no_reorient
no_com
"""


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "6-31g",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}


# ---------------------------------------------------------
# Build CQED configuration
# ---------------------------------------------------------

config = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.05]),
    omega=0.1,
    psi4_options=psi4_options,
    reference="rhf",
    functional=None,
    density_fitting=False,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=False,
    quiet=False,
)


# ---------------------------------------------------------
# Run QED-SAPT0 and inspect the dispersion partition
# ---------------------------------------------------------

calc = CQEDCalculator(config=config)
dimer_geometry = psi4.geometry(dimer)

components = calc.sapt0_components(
    dimer_geometry,
    integral_backend="full_eri",
    include_cavity_terms=True,
)

# ``standard``, ``cross``, and ``cavity`` correspond to [c,c], [c,d], and
# [d,d], respectively.  The cross term already includes both cross orderings.
partition = components.metadata["disp20_partition"]
disp20_cc = partition["standard"]
disp20_cd = partition["cross"]
disp20_dd = partition["cavity"]
disp20_partition_total = disp20_cc + disp20_cd + disp20_dd

print("\nQED-SAPT0 dispersion decomposition")
print("===================================")
print(f"Dispersion [c,c] : {disp20_cc: .12f} Eh")
print(f"Dispersion [c,d] : {disp20_cd: .12f} Eh")
print(f"Dispersion [d,d] : {disp20_dd: .12f} Eh")
print("-" * 39)
print(f"Partition total  : {disp20_partition_total: .12f} Eh")
print(f"Dispersion total : {components.disp20: .12f} Eh")

# The three terms are an exact partition of the production Disp20 value.
assert np.isclose(disp20_partition_total, components.disp20)
