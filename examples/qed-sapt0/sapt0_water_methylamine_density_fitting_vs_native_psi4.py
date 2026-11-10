"""
Cavity-free QED-SAPT0 (cqed-scf) versus native Psi4 SAPT0, density-fitted.

Both codes are put on the same footing:

* The SCF uses density fitting in both (``scf_type: df`` in ``psi4_options``).
* cqed-scf uses density fitting for the SAPT0 components as well
  (``density_fitting=True`` and ``integral_backend="df"``).  Psi4's native
  SAPT0 is always density-fitted internally.
* The cavity coupling is zero (``lambda_vector = 0``), so QED-SAPT0 should
  reduce to ordinary SAPT0 and agree with Psi4 up to small differences in the
  density-fitting treatment.  The cavity code path is still exercised
  (``include_cavity_terms=True``).

Like-for-like comparison notes
------------------------------
* cqed-scf reports the exchange in the S^2 approximation, so it is compared to
  Psi4's ``SAPT EXCH10(S^2) ENERGY``.
* cqed-scf's induction terms are the response (``,r``) terms and do not
  include the delta-HF correction.  Psi4's total SAPT0 energy does include
  delta-HF (and the full, non-S^2 exch10), so it is shown on its own line and
  is *not* expected to equal the sum of the six components.

Timings are single-run, end-to-end wall-clock times (all SCF solves included).
Psi4 threads are set with ``psi4.set_num_threads``; the NumPy/BLAS threads
used by cqed-scf are controlled separately (e.g. ``OMP_NUM_THREADS``).
"""

import time

import numpy as np
import psi4

from cqed_scf import CQEDCalculator, CQEDConfig

# ---------------------------------------------------------
# Dimer geometry (water ... methylamine)
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
# Options shared by both codes
# ---------------------------------------------------------

psi4.set_memory("4 GB")
psi4.set_num_threads(4)
psi4.core.be_quiet()  # keep the console to the tables below

psi4_options = {
    "basis": "jun-cc-pVDZ",
    "scf_type": "df",  # density-fitted SCF in both codes
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}
psi4.set_options(psi4_options)

# ---------------------------------------------------------
# cqed-scf: QED-SAPT0 with the cavity coupling switched off
# ---------------------------------------------------------

config = CQEDConfig(
    lambda_vector=np.array([0.0, 0.0, 0.0]),
    omega=0.0,
    psi4_options=psi4_options,
    reference="rhf",
    functional=None,
    density_fitting=True,  # density-fitted SAPT0 components
    charge=0,
    multiplicity=1,
    dispersion_policy="none",
    debug=False,
    quiet=True,
)

dimer_geometry = psi4.geometry(dimer)
calc = CQEDCalculator(config=config)

t0 = time.perf_counter()
components = calc.sapt0_components(
    dimer_geometry,
    integral_backend="df",
    include_cavity_terms=True,
)
t_cqed = time.perf_counter() - t0

# ---------------------------------------------------------
# Native Psi4 SAPT0 (density-fitted)
# ---------------------------------------------------------

psi4.activate(dimer_geometry)
t0 = time.perf_counter()
e_sapt0_total = psi4.energy("sapt0")
t_psi4 = time.perf_counter() - t0

def psi4_component(*names):
    """Return the first Psi4 variable that exists among ``names``."""
    for name in names:
        if psi4.core.has_variable(name):
            return psi4.variable(name)
    available = sorted(k for k in psi4.core.variables() if "SAPT" in k)
    raise KeyError(f"None of {names} found; available SAPT variables: {available}")


# Psi4 values matching each cqed-scf component (exchange is the S^2 form).
psi4_elst = psi4_component("SAPT ELST10,R ENERGY", "SAPT ELST ENERGY")
psi4_exch = psi4_component("SAPT EXCH10(S^2) ENERGY")
psi4_ind = psi4_component("SAPT IND20,R ENERGY")
psi4_exch_ind = psi4_component("SAPT EXCH-IND20,R ENERGY")
psi4_disp = psi4_component("SAPT DISP20 ENERGY")
psi4_exch_disp = psi4_component("SAPT EXCH-DISP20 ENERGY")

psi4_sum = psi4_elst + psi4_exch + psi4_ind + psi4_exch_ind + psi4_disp + psi4_exch_disp

# ---------------------------------------------------------
# Component comparison table
# ---------------------------------------------------------


def row(label, cqed, ref):
    """One table line: cqed-scf value, Psi4 value, difference in uEh and in %."""
    diff = cqed - ref
    pct = 100.0 * diff / abs(ref) if ref != 0.0 else float("nan")
    print(f"{label:<22}{cqed:16.10f}{ref:16.10f}{diff * 1e6:13.3f}{pct:11.4f}")


print()
print("Cavity-free QED-SAPT0 (cqed-scf, DF) vs native Psi4 SAPT0 (DF)")
print(f"water ... methylamine, {psi4_options['basis']}, scf_type={psi4_options['scf_type']}")
print("=" * 78)
print(
    f"{'Component':<22}{'cqed-scf (Eh)':>16}{'Psi4 (Eh)':>16}"
    f"{'Diff (uEh)':>13}{'Diff (%)':>11}"
)
print("-" * 78)
row("Electrostatics", components.elst10, psi4_elst)
row("Exchange (S^2)", components.exch10, psi4_exch)
row("Induction", components.ind20, psi4_ind)
row("Exchange-induction", components.exch_ind20, psi4_exch_ind)
row("Dispersion", components.disp20, psi4_disp)
row("Exchange-dispersion", components.exch_disp20, psi4_exch_disp)
print("-" * 78)
row("Sum of six components", components.total, psi4_sum)
print(f"{'Psi4 total SAPT0*':<22}{'':16}{e_sapt0_total:16.10f}")
print("=" * 78)
print("* Psi4's total includes delta-HF and the full (non-S^2) exch10, so it is")
print("  not expected to equal the sum of the six components above.")

max_diff = max(
    abs(components.elst10 - psi4_elst),
    abs(components.exch10 - psi4_exch),
    abs(components.ind20 - psi4_ind),
    abs(components.exch_ind20 - psi4_exch_ind),
    abs(components.disp20 - psi4_disp),
    abs(components.exch_disp20 - psi4_exch_disp),
)
print(f"\nLargest component difference: {max_diff * 1e6:.3f} uEh")

# ---------------------------------------------------------
# Timing
# ---------------------------------------------------------

print("\nTiming (single run, wall clock, all SCF solves included)")
print("=" * 50)
print(f"{'cqed-scf QED-SAPT0 (DF):':<34}{t_cqed:9.2f} s")
print(f"{'Psi4 native SAPT0 (DF):':<34}{t_psi4:9.2f} s")
print(f"{'Ratio cqed-scf / Psi4:':<34}{t_cqed / t_psi4:9.2f}")
print(f"(Psi4 threads: {psi4.get_num_threads()})")
