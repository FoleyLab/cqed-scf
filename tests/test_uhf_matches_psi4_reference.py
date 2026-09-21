import numpy as np
import pytest

from cqed_scf.references import CQEDConfig

### cavity-free singlet uhf reference matches psi4numpy tutorial example
# ==> Set Basic Psi4 Options <==
# Memory specification
psi4.set_memory(int(5e8))
numpy_memory = 2

# Set output file
psi4.core.set_output_file('output.dat', False)

# Define water from psi4numpy tutorial example
mol_str = """
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
"""

# Set options from psi4numpy tutorial example
psi4_options =    {'guess': 'core',
                  'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'reference': 'uhf'}



LAMBDA = np.array([0.0, 0.0, 0.05])

OH_DOUBLET = """
0 2
O     0.000000     0.000000     0.000000
H     0.000000     0.000000     0.969300
no_reorient
no_com
units angstrom
symmetry c1
"""

WATER_SINGLET = """
0 1
O  0.000000000000   0.000000000000  -0.068516219320
H  0.000000000000  -0.790689573744   0.543701060715
H  0.000000000000   0.790689573744   0.543701060715
no_reorient
no_com
units angstrom
symmetry c1
"""

MGH_CATION = """
1 1
Mg
H 1 1.4
symmetry c1
"""

def build_calculator(**overrides):
    from cqed_scf import CQEDCalculator

    options = {"basis": "sto-3g", "scf_type": "pk"}
    return CQEDCalculator(config=make_config(psi4_options=options, **overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(reference="uhf", multiplicity=2),
        dict(reference="uks", functional="pbe0", multiplicity=2),
    ],
    ids=["uhf", "uks"],
)
def test_energy_dispatches_to_uscf(overrides):
    """Validation passes and dispatch lands in CQEDUSCF, not the restricted engine."""
    calc = build_calculator(**overrides)
    with pytest.raises(NotImplementedError) as excinfo:
        calc.energy(OH_DOUBLET)
    message = str(excinfo.value)
    assert "Unrestricted CQED-SCF physics is not implemented" in message
    assert overrides["reference"] in message

