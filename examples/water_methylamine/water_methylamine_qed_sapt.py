import numpy as np
import psi4

from cqed_scf import CQEDConfig
from cqed_scf.sapt import QEDSAPT0Driver

psi4.core.be_quiet()  # Suppress Psi4 output for cleaner test output


psi4_options = {
    "basis": "jun-cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}

config = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.1]),
    omega=0.1,
    psi4_options=psi4_options,
    reference="rhf",
    functional=None,
    density_fitting=False,
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=False,
)

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


dimer_geometry = psi4.geometry(dimer)
sapt_driver = QEDSAPT0Driver(
    dimer_geometry=dimer_geometry,
    config=config,
    integral_backend="full_eri",
    include_cavity_terms=True,
)

SAPT0_Energy = sapt_driver.run()


# Define a width for the labels to ensure perfect alignment
w = 22

print(f"{'Electrostatics:':<{w}} {sapt_driver.Eelst100:15.10f} Hartree")
print(f"{'Exchange:':<{w}} {sapt_driver.Eexch100:15.10f} Hartree")
print(f"{'Dispersion:':<{w}} {sapt_driver.Edisp200:15.10f} Hartree")
print(f"{'Exchange-Dispersion:':<{w}} {sapt_driver.Eexchdisp200:15.10f} Hartree")
print(f"{'Induction:':<{w}} {sapt_driver.Eind200:15.10f} Hartree")
print(f"{'Exchange-Induction:':<{w}} {sapt_driver.Eexchind200:15.10f} Hartree")
print("-" * 50)  # Visual separator for the total
print(f"{'Total QED-SAPT0 Energy:':<{w}} {SAPT0_Energy:15.10f} Hartree")

print()

#sapt_driver.print_diagnostics()

#print("d_nuc_A", sapt_driver.d_nuc_A)
#print("d_exp_el_A", sapt_driver.d_exp_el_A)
#print("d_exp_A", sapt_driver.d_exp_A)

#print("d_nuc_B", sapt_driver.d_nuc_B)
#print("d_exp_el_B", sapt_driver.d_exp_el_B)
#print("d_exp_B", sapt_driver.d_exp_B)

#print("d_nuc_A * d_nuc_B", sapt_driver.d_nuc_A * sapt_driver.d_nuc_B)
#print("d_exp_el_A * d_exp_el_B", sapt_driver.d_exp_el_A * sapt_driver.d_exp_el_B)
#print("d_exp_A * d_exp_B", sapt_driver.d_exp_A * sapt_driver.d_exp_B)
