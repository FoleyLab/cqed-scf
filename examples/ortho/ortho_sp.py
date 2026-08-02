"""
CQED-DFT water regression using the new CQEDConfig architecture.

This example runs CQED-DFT with lambda = 0 using wB97X-D and compares
the energy and gradient directly against Psi4. Since lambda = 0, the
CQED result should reduce to ordinary Psi4 DFT.

The calculator internally strips the dispersion-corrected functional
for the CQED-SCF step when needed, then adds the dispersion correction
post-SCF according to the config/dispersion policy.
"""

import numpy as np
import psi4

from cqed_scf import CQEDConfig, CQEDCalculator


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

geometry = """
1 1
         C           -1.804928163307     1.957993763262     0.703312273806 
         C           -0.379708783307     1.994122833262     0.698532703806
         C            0.296125016693     0.817793533262     0.710271493806
         C           -2.520286433307     0.755089873262     0.736288843806
         H           -2.344947113307     2.899196893262     0.691895063806
         H            0.158564066693     2.933869823262     0.699142733806
         H           -3.601954283307     0.764862203262     0.746931053806
         N            1.767881836693     0.820900013262     0.771891313806
         O            2.315054046693    -0.296733496738     0.879853723806
         O            2.340645916693     1.923356243262     0.711986073806
         C           -1.829967733307    -0.442167236738     0.756258983806
         H           -2.356763623307    -1.389967436738     0.789740873806
         C           -0.361341153307    -0.491572936738     0.714148383806
         H            0.119338216693    -1.238105076738     1.350400383806
         Br          -0.151212663307    -1.224162306738    -1.170925976194
no_com
no_reorient
symmetry c1
"""


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("4 GB")

psi4_options = {
    "basis": "6-31g",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}


# ---------------------------------------------------------
# Build CQED configuration
# ---------------------------------------------------------

lam_1 = np.array([0.2, 0.1, 0.3])
lam_2 = -1 * lam_1

config_1 = CQEDConfig(
    lambda_vector=lam_1,
    omega=0.1,
    psi4_options=psi4_options,
    reference="rks",
    functional="wb97x",
    density_fitting=True,
    charge=0,
    multiplicity=1,
    debug=True,
)

config_2 = CQEDConfig(
    lambda_vector=lam_2,
    omega=0.1,
    psi4_options=psi4_options,
    reference="rks",
    functional="wb97x",
    density_fitting=True,
    charge=0,
    multiplicity=1,
    debug=True,
)

# ---------------------------------------------------------
# Build calculator from config
# ---------------------------------------------------------

calc_1 = CQEDCalculator(config=config_1)

calc_2 = CQEDCalculator(config=config_2)

# ---------------------------------------------------------
# Run CQED-DFT energy + gradient
# ---------------------------------------------------------

E_1 = calc_1.energy(geometry)
E_2 = calc_2.energy(geometry)


print(f"Energy with lam_1 is {E_1:12.10f}")
print(f"Energy with lam_2 is {E_2:12.10f}")
