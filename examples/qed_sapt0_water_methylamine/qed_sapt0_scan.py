"""
QED-SAPT0 water-methylamine scan: density-fitted versus dense two-electron integrals.
"""

import time

import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig


# ---------------------------------------------------------
# Dimer geometries
# ---------------------------------------------------------

# Five points from the adaptive curve in ../sapt0_water_methylamine/, keyed by
# center-of-mass separation in Angstrom: repulsive wall, R_e, the scale-1.00
# anchor, mid-range, and the long-range tail.  Water is fixed; only methylamine
# translates.  no_reorient / no_com are required -- without them Psi4 re-centers
# each frame and silently changes the cavity coupling direction.

geometries = {
    3.0916: """
0 1
O     -0.687464896    -0.111744327    -0.019625472
H     -1.046121544     0.775938208     0.012706845
H      0.274042519     0.025850654    -0.003497262
--
0 1
N      1.983989073     0.093319005     0.004660980
H      2.279353504    -0.459318970    -0.790129883
H      2.294069568    -0.417402086     0.821520473
C      2.643324350     1.401682970    -0.035580658
H      2.332781928     1.983407930     0.828934762
H      3.734633640     1.362387998    -0.044536327
H      2.316612078     1.937600439    -0.923404470
units angstrom
symmetry c1
no_reorient
no_com
""",
    3.3243: """
0 1
O     -0.687464896    -0.111744327    -0.019625472
H     -1.046121544     0.775938208     0.012706845
H      0.274042519     0.025850654    -0.003497262
--
0 1
N      2.221644167     0.102696022     0.005794845
H      2.517008598    -0.449941953    -0.788996018
H      2.531724662    -0.408025069     0.822654338
C      2.880979444     1.411059987    -0.034446793
H      2.570437022     1.992784947     0.830068627
H      3.972288734     1.371765015    -0.043402462
H      2.554267172     1.946977456    -0.922270605
units angstrom
symmetry c1
no_reorient
no_com
""",
    3.3859: """
0 1
O     -0.687464896    -0.111744327    -0.019625472
H     -1.046121544     0.775938208     0.012706845
H      0.274042519     0.025850654    -0.003497262
--
0 1
N      2.284419126     0.105172897     0.006094347
H      2.579783557    -0.447465078    -0.788696516
H      2.594499621    -0.405548194     0.822953840
C      2.943754403     1.413536862    -0.034147291
H      2.633211981     1.995261822     0.830368129
H      4.035063693     1.374241890    -0.043102960
H      2.617042131     1.949454331    -0.921971103
units angstrom
symmetry c1
no_reorient
no_com
""",
    4.3763: """
0 1
O     -0.687464896    -0.111744327    -0.019625472
H     -1.046121544     0.775938208     0.012706845
H      0.274042519     0.025850654    -0.003497262
--
0 1
N      3.289107826     0.144814307     0.010887767
H      3.584472257    -0.407823668    -0.783903096
H      3.599188321    -0.365906784     0.827747260
C      3.948443103     1.453178272    -0.029353871
H      3.637900681     2.034903232     0.835161549
H      5.039752393     1.413883300    -0.038309540
H      3.621730831     1.989095741    -0.917177683
units angstrom
symmetry c1
no_reorient
no_com
""",
    7.3134: """
0 1
O     -0.687464896    -0.111744327    -0.019625472
H     -1.046121544     0.775938208     0.012706845
H      0.274042519     0.025850654    -0.003497262
--
0 1
N      6.244289121     0.261415154     0.024987085
H      6.539653552    -0.291222821    -0.769803778
H      6.554369616    -0.249305937     0.841846578
C      6.903624398     1.569779119    -0.015254553
H      6.593081976     2.151504079     0.849260867
H      7.994933688     1.530484147    -0.024210222
H      6.576912126     2.105696588    -0.903078365
units angstrom
symmetry c1
no_reorient
no_com
""",
}


# ---------------------------------------------------------
# Psi4 options
# ---------------------------------------------------------

psi4.set_memory("8 GB")

psi4_options = {
    "basis": "jun-cc-pVDZ",
    "scf_type": "pk",
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}


# ---------------------------------------------------------
# Build CQED configuration
# ---------------------------------------------------------

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
    quiet=True,   # SILENT: suppress all stdout (CQED-SCF + Psi4 engine output)
)

calc = CQEDCalculator(config=config)


# ---------------------------------------------------------
# Run the scan
# ---------------------------------------------------------

# monomer_reference_frame="monomer_com" solves each monomer reference with its
# own center of mass at the origin.  This matters for a scan: CQED-SCF orbital
# energies are not translation invariant, so with the default "dimer" frame the
# dispersion tail would depend on where the coordinate origin happens to sit.

def run_point(dimer, backend):
    psi4.core.clean()
    start = time.time()
    components = calc.sapt0_components(
        psi4.geometry(dimer),
        integral_backend=backend,
        include_cavity_terms=True,
        monomer_reference_frame="monomer_com",
    )
    return components, time.time() - start


results = {}

print("\nQED-SAPT0 water-methylamine scan")
print("================================")
print(f"lambda = {config.lambda_vector}   omega = {config.omega} Eh   basis = jun-cc-pVDZ")
print()
print(f"{'R / Ang':>9}  {'E(df) / Eh':>18}  {'E(full_eri) / Eh':>18}  "
      f"{'delta / Eh':>12}  {'t_df':>7}  {'t_dense':>8}")
print("-" * 82)

for R, dimer in geometries.items():
    df, t_df = run_point(dimer, "df")
    dense, t_dense = run_point(dimer, "full_eri")
    results[R] = (df, dense)
    print(f"{R:9.4f}  {df.total: 18.12f}  {dense.total: 18.12f}  "
          f"{df.total - dense.total: 12.3e}  {t_df:6.1f}s  {t_dense:7.1f}s")


# ---------------------------------------------------------
# Dispersion partition versus separation
# ---------------------------------------------------------

# Disp20 is quadratic in the two-electron numerator, so "standard + cavity" does
# not sum to the total -- there is a cross term.  The cavity kernel carries no
# Coulomb operator, so that column does not decay with separation.

print("\nDispersion partition (dense)")
print("============================")
print(f"{'R / Ang':>9}  {'standard':>15}  {'cross':>15}  {'cavity':>15}  {'total':>15}")
print("-" * 82)

for R, (_, dense) in results.items():
    part = dense.metadata["disp20_partition"]
    print(f"{R:9.4f}  {part['standard']: 15.10f}  {part['cross']: 15.10f}  "
          f"{part['cavity']: 15.10f}  {part['total']: 15.10f}")


# ---------------------------------------------------------
# Control: lambda = 0 reduces to ordinary SAPT0
# ---------------------------------------------------------

reference_R = 3.3859

calc_zero = CQEDCalculator(config=config.copy_with(lambda_vector=np.zeros(3)))
psi4.core.clean()
components = calc_zero.sapt0_components(
    psi4.geometry(geometries[reference_R]),
    integral_backend="df",
    include_cavity_terms=True,
    monomer_reference_frame="monomer_com",
)

print(f"\nQED-SAPT0 components at lambda = 0, R = {reference_R} Ang")
print("=======================================================")
print(f"Electrostatics       : {components.elst10: .12f} Eh")
print(f"Exchange             : {components.exch10: .12f} Eh")
print(f"Dispersion           : {components.disp20: .12f} Eh")
print(f"Exchange-dispersion  : {components.exch_disp20: .12f} Eh")
print(f"Induction            : {components.ind20: .12f} Eh")
print(f"Exchange-induction   : {components.exch_ind20: .12f} Eh")
print("-" * 48)
print(f"Total QED-SAPT0      : {components.total: .12f} Eh")
