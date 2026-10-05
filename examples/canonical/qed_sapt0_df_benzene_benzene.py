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
C    0.712645325   1.120995701   0.060540783
H    1.357841649   1.986399167   0.127737169
C    1.258235731  -0.159251901   0.124233518
H    2.324954277  -0.287099878   0.246743035
C    0.426884963  -1.274526662   0.042650428
H    0.850444649  -2.268432680   0.094749952
C   -0.949577845  -1.110074058  -0.100313595
H   -1.594455696  -1.976273703  -0.163713480
C   -1.495525640   0.171050561  -0.161546018
H   -2.563782791   0.299221149  -0.273703115
C   -0.663827601   1.286642887  -0.083401433
H   -1.086900697   2.281000204  -0.132886135
--
0 1
C    1.949488213   1.110116294   3.600833554
H    2.594333337   1.976300037   3.664584272
C    2.495441339  -0.171034805   3.662353287
H    3.563648223  -0.299181855   3.775097506
C    1.663791859  -1.286634905   3.583702858
H    2.086873566  -2.280987340   3.633356521
C    0.287390298  -1.120999876   3.438986110
H   -0.357712298  -1.986410120   3.371353478
C   -0.258169573   0.159232871   3.375020282
H   -1.324797607   0.287062084   3.251841521
C    0.573107377   1.274513633   3.457091226
H    0.149582495   2.268418667   3.404722296
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
    "basis": "jun-cc-pVDZ",
    "scf_type": "df",
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

calc = CQEDCalculator(config=config)
dimer_geometry = psi4.geometry(dimer)

components = calc.sapt0_components(
    dimer_geometry,
    integral_backend="df",
    include_cavity_terms=True,
    monomer_reference_frame="monomer_com",
)

print("\nQED-SAPT0 components")
print("====================")
print(f"Electrostatics       : {components.elst10: .12f} Eh")
print(f"Exchange             : {components.exch10: .12f} Eh")
print(f"Dispersion           : {components.disp20: .12f} Eh")
print(f"Exchange-dispersion  : {components.exch_disp20: .12f} Eh")
print(f"Induction            : {components.ind20: .12f} Eh")
print(f"Exchange-induction   : {components.exch_ind20: .12f} Eh")
print("-" * 48)
print(f"Total QED-SAPT0      : {components.total: .12f} Eh")
