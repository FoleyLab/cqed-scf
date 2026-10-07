"""
QED-SAPT0 total energy and components using CQEDConfig and CQEDCalculator.
"""

import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig


# ---------------------------------------------------------
# Dimer geometry
# ---------------------------------------------------------

dimer = """
0 1
O   -0.687464896  -0.111744327  -0.019625472
H   -1.046121544   0.775938208   0.012706845
H    0.274042519   0.025850654  -0.003497262
--
0 1
N    2.787113199   0.125007400   0.008492726
H    3.082477630  -0.427630575  -0.786298137
H    3.097193694  -0.385713691   0.825352219
C    3.446448476   1.433371365  -0.031748912
H    3.135906054   2.015096325   0.832766508
H    4.537757766   1.394076393  -0.040704580
H    3.119736204   1.969288834  -0.919572724
symmetry c1
no_com
no_reorient
"""

# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

dense_options = {
    "basis": "jun-cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

df_options = {
    "basis": "jun-cc-pVDZ",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

# ---------------------------------------------------------
# Build CQED configuration
# ---------------------------------------------------------

dense_config = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.05]),
    omega=0.1,
    psi4_options=dense_options,
    reference="rhf",
    functional=None,
    density_fitting=False,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=True,   # VERBOSE: add SCF/component detail (debug route via output.echo)
    quiet=False,  # keep normal verbosity so the verbose detail is emitted
)


df_config = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.05]),
    omega=0.1,
    psi4_options=df_options,
    reference="rhf",
    functional=None,
    density_fitting=True,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=True,   # VERBOSE: add SCF/component detail (debug route via output.echo)
    quiet=False,  # keep normal verbosity so the verbose detail is emitted
)


# ---------------------------------------------------------
# Run QED-SAPT0 components
# ---------------------------------------------------------

dense_calc = CQEDCalculator(config=dense_config)
df_calc = CQEDCalculator(config=df_config)

dimer_geometry = psi4.geometry(dimer)

dense_components = dense_calc.sapt0_components(
    dimer_geometry,
    integral_backend="full_eri",
    include_cavity_terms=True,
)

df_components = df_calc.sapt0_components(
    dimer_geometry,
    integral_backend="df",
    include_cavity_terms=True,
)


print("\nQED-SAPT0 Dense components")
print("====================")
print(f"Electrostatics       : {dense_components.elst10: .12f} Eh")
print(f"Exchange             : {dense_components.exch10: .12f} Eh")
print(f"Dispersion           : {dense_components.disp20: .12f} Eh")
print(f"Exchange-dispersion  : {dense_components.exch_disp20: .12f} Eh")
print(f"Induction            : {dense_components.ind20: .12f} Eh")
print(f"Exchange-induction   : {dense_components.exch_ind20: .12f} Eh")
print("-" * 48)
print(f"Total QED-SAPT0      : {dense_components.total: .12f} Eh")

print("\nQED-SAPT0 DF components")
print("====================")
print(f"Electrostatics       : {df_components.elst10: .12f} Eh")
print(f"Exchange             : {df_components.exch10: .12f} Eh")
print(f"Dispersion           : {df_components.disp20: .12f} Eh")
print(f"Exchange-dispersion  : {df_components.exch_disp20: .12f} Eh")
print(f"Induction            : {df_components.ind20: .12f} Eh")
print(f"Exchange-induction   : {df_components.exch_ind20: .12f} Eh")
print("-" * 48)
print(f"Total QED-SAPT0      : {df_components.total: .12f} Eh")

print("\nQED-SAPT0 Dense vs DF comparison")
print("===============================")
print(f"Electrostatics DF Error (% of component)      : {(dense_components.elst10 - df_components.elst10) / dense_components.elst10 * 100: .2f} %")
print(f"Exchange DF Error (% of component)            : {(dense_components.exch10 - df_components.exch10) / dense_components.exch10 * 100: .2f} %")
print(f"Dispersion DF Error (% of component)          : {(dense_components.disp20 - df_components.disp20) / dense_components.disp20 * 100: .2f} %")
print(f"Exchange-dispersion DF Error (% of component) : {(dense_components.exch_disp20 - df_components.exch_disp20) / dense_components.exch_disp20 * 100: .2f} %")
print(f"Induction DF Error (% of component)           : {(dense_components.ind20 - df_components.ind20) / dense_components.ind20 * 100: .2f} %")
print(f"Exchange-induction DF Error (% of component)   : {(dense_components.exch_ind20 - df_components.exch_ind20) / dense_components.exch_ind20 * 100: .2f} %")
print("-" * 48)
print(f"Total QED-SAPT0 DF Error (% of total)     : {((dense_components.total - df_components.total) / dense_components.total) * 100: .2f} %")
