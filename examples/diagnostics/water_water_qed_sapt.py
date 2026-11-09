import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig

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
    O   -0.066999140   0.000000000   1.494354740
    H    0.815734270   0.000000000   1.865866390
    H    0.068855100   0.000000000   0.539142770
    --
    O    0.062547750   0.000000000  -1.422632080
    H   -0.406965400  -0.760178410  -1.771744500
    H   -0.406965400   0.760178410  -1.771744500
    symmetry c1
    no_com
    no_reorient
"""

dimer_geometry = psi4.geometry(dimer)
calc = CQEDCalculator(config=config)
components = calc.sapt0_components(
    dimer_geometry,
    integral_backend="full_eri",
    include_cavity_terms=True,
)


# Define a width for the labels to ensure perfect alignment
w = 22

print(f"{'Electrostatics:':<{w}} {components.elst10:15.10f} Hartree")
print(f"{'Exchange:':<{w}} {components.exch10:15.10f} Hartree")
print(f"{'Dispersion:':<{w}} {components.disp20:15.10f} Hartree")
print(f"{'Exchange-Dispersion:':<{w}} {components.exch_disp20:15.10f} Hartree")
print(f"{'Induction:':<{w}} {components.ind20:15.10f} Hartree")
print(f"{'Exchange-Induction:':<{w}} {components.exch_ind20:15.10f} Hartree")
print("-" * 50)  # Visual separator for the total
print(f"{'Total QED-SAPT0 Energy:':<{w}} {components.total:15.10f} Hartree")

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
