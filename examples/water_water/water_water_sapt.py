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
        lambda_vector=np.array([0.0, 0.0, 0.0]),
        omega=0.0,
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
sapt_driver = QEDSAPT0Driver(
    dimer_geometry=dimer_geometry,
    config=config,
    integral_backend="full_eri",
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
print("-" * 50) # Visual separator for the total
print(f"{'Total SAPT0 Energy:':<{w}} {SAPT0_Energy:15.10f} Hartree")
