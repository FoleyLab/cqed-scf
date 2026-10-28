import importlib.util
import json
from pathlib import Path

import pytest

from cqed_scf import CQEDConfig
from cqed_scf.uscf import CQEDUSCF


REFERENCE_DIR = (
    Path(__file__).parents[1]
    / "examples"
    / "canonical"
    / "Unrestricted_Test_Examples"
)


def _load_system_definitions():
    spec = importlib.util.spec_from_file_location(
        "unrestricted_reference_systems", REFERENCE_DIR / "systems.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYSTEM_DEFINITIONS = _load_system_definitions()
with (REFERENCE_DIR / "psi4_reference_energies.json").open() as handle:
    PSI4_REFERENCES = json.load(handle)


@pytest.mark.slow
@pytest.mark.parametrize("system_name", ["oh_radical", "o2_triplet", "nh_triplet"])
def test_canonical_uhf_energy_and_spin_match_psi4(system_name):
    system = SYSTEM_DEFINITIONS.SYSTEMS[system_name]
    reference = PSI4_REFERENCES["systems"][system_name]["methods"]["uhf"]
    psi4_options = SYSTEM_DEFINITIONS.base_psi4_options("uhf")
    psi4_options.pop("reference")
    config = CQEDConfig(
        lambda_vector=(0.0, 0.0, 0.0),
        omega=SYSTEM_DEFINITIONS.OMEGA,
        psi4_options=psi4_options,
        reference="uhf",
        density_fitting=False,
        charge=system["charge"],
        multiplicity=system["multiplicity"],
        dispersion_policy="none",
        quiet=True,
    )

    energy, results = CQEDUSCF(system["geometry"], config).run()

    assert energy == pytest.approx(reference["energy"], abs=1.0e-9, rel=0.0)
    assert results["s_squared"] == pytest.approx(
        reference["s_squared"], abs=1.0e-8, rel=0.0
    )
    assert results["spin_contamination"] == pytest.approx(
        reference["spin_contamination"], abs=1.0e-8, rel=0.0
    )
    assert isinstance(results["s_squared"], float)
    assert isinstance(results["spin_contamination"], float)
