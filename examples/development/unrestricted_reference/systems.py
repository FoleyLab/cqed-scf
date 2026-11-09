"""Shared definitions for the unrestricted (QED-)SCF reference test set.

Both oracle drivers in this directory import from here so that the Psi4
(UHF / UKS) and Hilbert (QED-UHF / QED-UKS) reference numbers are generated
from *bit-identical* geometries, basis, and convergence settings.

Conventions
-----------
* Geometries are frozen with ``no_reorient`` / ``no_com`` and ``symmetry c1``.
  This matters: the dipole self-energy couples to the *orientation* of the
  molecule relative to the cavity polarization vector, so any reorientation or
  center-of-mass shift by Psi4 would silently change the QED energies.
* ``lambda_vector`` is the coupling used by ``cqed_scf``.  Hilbert instead takes
  ``CAVITY_COUPLING_STRENGTH`` = lambda / sqrt(2 * omega); see
  ``lambda_to_hilbert_coupling`` below.
* In the coherent-state basis the CQED-SCF ground-state energy is independent
  of ``omega`` (verified numerically: Hilbert gives the same energy for
  ``N_PHOTON_STATES`` = 1 and 2).  ``OMEGA`` is fixed here only so that the
  lambda -> g conversion is well defined and reproducible.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Common computational settings
# --------------------------------------------------------------------------

BASIS = "6-311++G**"
FUNCTIONAL = "PBE0"

# Tight, deterministic SCF settings.  scf_type=pk avoids density-fitting error
# entirely, so these numbers are true reference values rather than DF values.
E_CONVERGENCE = 1.0e-11
D_CONVERGENCE = 1.0e-10
SCF_TYPE = "pk"
MAXITER = 500

# Dense, fixed DFT grid so the UKS / QED-UKS numbers are grid-converged and
# reproducible across Psi4 builds.
DFT_SPHERICAL_POINTS = 590
DFT_RADIAL_POINTS = 99

# Cavity frequency (a.u.).  See module docstring: the SCF energy does not
# depend on it, but it defines the lambda <-> g map used by Hilbert.
OMEGA = 0.1


# --------------------------------------------------------------------------
# Cavity coupling vectors
# --------------------------------------------------------------------------
# * "zero"    -- regression check: QED-UHF/QED-UKS must reproduce UHF/UKS exactly.
# * "z_only"  -- polarization along the molecular axis (OH, O2, NH all lie on z),
#                the strongly-coupled parallel case.
# * "x_only"  -- polarization perpendicular to the molecular axis, which probes
#                the transverse components of the dipole self-energy.
# * "general" -- a non-axial vector that exercises every off-diagonal
#                lambda_i * lambda_j quadrupole / dipole cross term.

LAMBDA_VECTORS = {
    "zero": np.array([0.00, 0.00, 0.00]),
    "z_only": np.array([0.00, 0.00, 0.05]),
    "x_only": np.array([0.05, 0.00, 0.00]),
    "general": np.array([0.02, 0.03, 0.05]),
}


def lambda_to_hilbert_coupling(lambda_vector, omega=OMEGA):
    """Convert a ``cqed_scf`` lambda vector to Hilbert's coupling strength.

    Hilbert's ``polaritonic_scf`` code reconstructs lambda internally as
    ``lambda_i = g_i * sqrt(2 * omega)`` (see ``src/polaritonic_scf/hf.cc``),
    so the inverse map is what we hand it.
    """
    return (np.asarray(lambda_vector, dtype=float) / np.sqrt(2.0 * omega)).tolist()


# --------------------------------------------------------------------------
# Test molecules
# --------------------------------------------------------------------------
# Every molecule is aligned along z so that the "z_only" / "x_only" lambda
# vectors have an unambiguous parallel / perpendicular meaning.

SYSTEMS = {
    # Doublet, strongly polar: the primary common test case for all four
    # methods.  Permanent dipole => large first-order cavity response.
    "oh_radical": {
        "charge": 0,
        "multiplicity": 2,
        "description": "Hydroxide radical (OH), doublet, polar",
        "geometry": """
0 2
O     0.000000     0.000000     0.000000
H     0.000000     0.000000     0.969300
no_reorient
no_com
units angstrom
symmetry c1
""",
    },
    # Triplet, non-polar.  No permanent dipole, so the cavity contribution comes
    # entirely from the two-electron dipole self-energy and quadrupole terms --
    # a clean isolation of those pieces of the UHF/UKS Fock build.
    "o2_triplet": {
        "charge": 0,
        "multiplicity": 3,
        "description": "Molecular oxygen (O2), triplet ground state, non-polar",
        "geometry": """
0 3
O     0.000000     0.000000     0.000000
O     0.000000     0.000000     1.207520
no_reorient
no_com
units angstrom
symmetry c1
""",
    },
    # Triplet *and* polar.  Complements O2: exercises the permanent-dipole
    # cavity terms in a high-spin open-shell reference, which is exactly the
    # regime where a bug in the beta-spin Fock build would hide.
    "nh_triplet": {
        "charge": 0,
        "multiplicity": 3,
        "description": "Imidogen (NH), triplet ground state, polar",
        "geometry": """
0 3
N     0.000000     0.000000     0.000000
H     0.000000     0.000000     1.036200
no_reorient
no_com
units angstrom
symmetry c1
""",
    },
}


def base_psi4_options(reference):
    """Common Psi4 options shared by the Psi4 and Hilbert drivers."""
    options = {
        "basis": BASIS,
        "scf_type": SCF_TYPE,
        "reference": reference,
        "e_convergence": E_CONVERGENCE,
        "d_convergence": D_CONVERGENCE,
        "maxiter": MAXITER,
        "guess": "sad",
        "dft_spherical_points": DFT_SPHERICAL_POINTS,
        "dft_radial_points": DFT_RADIAL_POINTS,
    }
    return options
