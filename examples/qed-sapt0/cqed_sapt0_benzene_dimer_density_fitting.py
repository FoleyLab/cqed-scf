"""
QED-SAPT0 for a parallel-displaced benzene dimer in a cavity (cqed-scf only).

Psi4 has no native cavity SAPT0, so this example shows cqed-scf on its own.

Why density fitting for both the SCF and the SAPT components?
-------------------------------------------------------------
Every monomer is solved in the full dimer basis (the partner is ghosted), and
the SAPT components need two-electron integrals in that basis.  With
jun-cc-pVDZ the benzene dimer has a few hundred basis functions (the script
prints the exact number), and a dense ``full_eri`` tensor grows as N^4 -- tens
of gigabytes here, which is impractical.  Density fitting replaces it with
three-index tensors that scale as N^2 * N_aux and are far cheaper in both
memory and time, at the cost of a small, well-controlled fitting error.

Two independent switches are involved, and both should be on for large systems:

* ``scf_type: df`` in ``psi4_options`` -- density-fitted CQED-SCF.
* ``density_fitting=True`` in ``CQEDConfig`` together with
  ``integral_backend="df"`` -- density-fitted SAPT0 components.

``monomer_reference_frame="monomer_com"`` solves each monomer's CQED-SCF
reference with that monomer's own centre of mass at the origin.  CQED orbital
energies are not translation invariant, so this keeps the cavity-mediated
dispersion independent of where the dimer sits in space.
"""

import time

import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig

EH_TO_KCAL = 627.5094740631

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
    "scf_type": "df",  # density-fitted SCF (see module docstring)
    "e_convergence": 1e-10,
    "d_convergence": 1e-8,
}

# ---------------------------------------------------------
# Build CQED configuration
# ---------------------------------------------------------

lambda_vector = np.array([0.0, 0.0, 0.05])
omega = 0.1

config = CQEDConfig(
    lambda_vector=lambda_vector,
    omega=omega,
    psi4_options=psi4_options,
    reference="rhf",
    functional=None,
    density_fitting=True,  # density-fitted SAPT0 components (independent of scf_type)
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=False,
    quiet=True,
)

# ---------------------------------------------------------
# Size of the problem, to show why density fitting is used
# ---------------------------------------------------------

dimer_geometry = psi4.geometry(dimer)
nbf = psi4.core.BasisSet.build(dimer_geometry, "ORBITAL", psi4_options["basis"]).nbf()
dense_eri_gb = nbf**4 * 8 / 1e9

# ---------------------------------------------------------
# Run QED-SAPT0 and get standard components
# ---------------------------------------------------------

calc = CQEDCalculator(config=config)

t0 = time.perf_counter()
components = calc.sapt0_components(
    dimer_geometry,
    integral_backend="df",
    include_cavity_terms=True,
    monomer_reference_frame="monomer_com",
)
elapsed = time.perf_counter() - t0

# ---------------------------------------------------------
# Report components
# ---------------------------------------------------------

ROWS = [
    ("Electrostatics", components.elst10),
    ("Exchange", components.exch10),
    ("Induction", components.ind20),
    ("Exchange-induction", components.exch_ind20),
    ("Dispersion", components.disp20),
    ("Exchange-dispersion", components.exch_disp20),
]

width = 58
print()
print("QED-SAPT0 components: benzene ... benzene (parallel displaced)")
print(f"basis={psi4_options['basis']} ({nbf} basis functions), "
      f"scf_type={psi4_options['scf_type']}, backend=df, frame=monomer_com")
print(f"lambda = {lambda_vector.tolist()}, omega = {omega} Eh")
print(f"A dense N^4 ERI tensor would need ~{dense_eri_gb:.0f} GB")
print("=" * width)
print(f"{'Component':<24}{'Energy (Eh)':>18}{'(kcal/mol)':>16}")
print("-" * width)
for label, value in ROWS:
    print(f"{label:<24}{value:18.12f}{value * EH_TO_KCAL:16.6f}")
print("-" * width)
print(f"{'Total QED-SAPT0':<24}{components.total:18.12f}{components.total * EH_TO_KCAL:16.6f}")
print("=" * width)

# ---------------------------------------------------------
# Further partition dispersion into the standard, mixed, and cavity-only components
# ---------------------------------------------------------

# The standard, cross, and cavity components correspond to [c,c], [c,d], and
# [d,d], respectively.  This is additional metadata of the results object.
# Dispersion is quadratic in the interaction, so the cross term is required:
# standard + cavity alone does NOT sum to the total.
partition = components.metadata["disp20_partition"]
disp20_cc = partition["standard"]
disp20_cd = partition["cross"]
disp20_dd = partition["cavity"]
disp20_partition_total = disp20_cc + disp20_cd + disp20_dd

PARTITION_ROWS = [
    ("Standard [c,c]", disp20_cc),
    ("Cross [c,d]", disp20_cd),
    ("Cavity [d,d]", disp20_dd),
]

width = 70
print("\nDispersion partition")
print("=" * width)
print(f"{'Term':<24}{'Energy (Eh)':>18}{'(kcal/mol)':>16}{'% of total':>12}")
print("-" * width)
for label, value in PARTITION_ROWS:
    pct = 100.0 * value / disp20_partition_total
    print(f"{label:<24}{value:18.12f}{value * EH_TO_KCAL:16.6f}{pct:12.2f}")
print("-" * width)
print(
    f"{'Total dispersion':<24}{disp20_partition_total:18.12f}"
    f"{disp20_partition_total * EH_TO_KCAL:16.6f}{100.0:12.2f}"
)
print("=" * width)

# Verify that the partition components sum to the total dispersion.
assert np.isclose(disp20_partition_total, components.disp20)

print(f"\nWall time: {elapsed:.2f} s (includes the CQED-SCF solves)")
