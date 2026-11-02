import importlib.util
import json
from pathlib import Path

import numpy as np
import psi4
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
        "unrestricted_uks_reference_systems", REFERENCE_DIR / "systems.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYSTEM_DEFINITIONS = _load_system_definitions()
with (REFERENCE_DIR / "psi4_reference_energies.json").open() as handle:
    PSI4_REFERENCES = json.load(handle)
with (REFERENCE_DIR / "hilbert_reference_energies.json").open() as handle:
    HILBERT_REFERENCES = json.load(handle)


WATER_CATION = """
1 2
O  0.000000  0.000000  0.000000
H  0.000000  0.757000  0.586000
H  0.000000 -0.757000  0.586000
no_reorient
no_com
symmetry c1
units angstrom
"""


def _run_uks(system_name, functional):
    system = SYSTEM_DEFINITIONS.SYSTEMS[system_name]
    psi4_options = SYSTEM_DEFINITIONS.base_psi4_options("uks")
    psi4_options.pop("reference")
    config = CQEDConfig(
        lambda_vector=(0.0, 0.0, 0.0),
        omega=SYSTEM_DEFINITIONS.OMEGA,
        psi4_options=psi4_options,
        reference="uks",
        functional=functional,
        density_fitting=False,
        charge=system["charge"],
        multiplicity=system["multiplicity"],
        dispersion_policy="none",
        quiet=True,
    )
    engine = CQEDUSCF(system["geometry"], config)
    energy, results = engine.run()
    return engine, energy, results


def _run_exchange_branch(functional):
    """Use a nondegenerate doublet for robust pure-DFT branch coverage."""

    options = {
        "basis": "sto-3g",
        "scf_type": "pk",
        "e_convergence": 1.0e-10,
        "d_convergence": 1.0e-8,
        "maxiter": 200,
        "dft_spherical_points": 302,
        "dft_radial_points": 75,
    }
    config = CQEDConfig(
        lambda_vector=(0.0, 0.0, 0.0),
        omega=SYSTEM_DEFINITIONS.OMEGA,
        psi4_options=options,
        reference="uks",
        functional=functional,
        density_fitting=False,
        charge=1,
        multiplicity=2,
        dispersion_policy="none",
        quiet=True,
    )
    engine = CQEDUSCF(WATER_CATION, config)
    energy, results = engine.run()
    return engine, energy, results


@pytest.mark.slow
@pytest.mark.parametrize("system_name", ["oh_radical", "o2_triplet", "nh_triplet"])
def test_canonical_pbe0_uks_matches_psi4(system_name):
    _, energy, results = _run_uks(system_name, SYSTEM_DEFINITIONS.FUNCTIONAL)
    reference = PSI4_REFERENCES["systems"][system_name]["methods"]["uks"]

    assert energy == pytest.approx(reference["energy"], abs=1.0e-9, rel=0.0)
    assert results["s_squared"] == pytest.approx(
        reference["s_squared"], abs=1.0e-8, rel=0.0
    )
    assert results["spin_contamination"] == pytest.approx(
        reference["spin_contamination"], abs=1.0e-8, rel=0.0
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "functional, expected_kind",
    [("PBE", "pure"), ("PBE0", "hybrid"), ("wB97X", "range-separated")],
)
def test_uks_exchange_branches_match_live_psi4(functional, expected_kind):
    engine, energy, results = _run_exchange_branch(functional)

    assert energy == pytest.approx(results["energy_psi4"], abs=1.0e-9, rel=0.0)
    functional_obj = engine.Vpot.functional()
    if expected_kind == "pure":
        assert not functional_obj.is_x_hybrid()
        assert not functional_obj.is_x_lrc()
        assert not engine._do_K
        assert not engine._do_wK
    elif expected_kind == "hybrid":
        assert functional_obj.is_x_hybrid()
        assert not functional_obj.is_x_lrc()
        assert engine.x_alpha > 0.0
        assert engine._do_K
        assert not engine._do_wK
    else:
        assert functional_obj.is_x_hybrid()
        assert functional_obj.is_x_lrc()
        assert engine.x_alpha > 0.0
        assert engine.x_beta > 0.0
        assert functional_obj.x_omega() > 0.0
        assert engine._do_K
        assert engine._do_wK


@pytest.mark.parametrize(
    "scf_type,density_fitting", [("pk", False), ("df", True)], ids=["pk", "df"]
)
def test_uks_runs_when_dense_ao_eri_is_forbidden(
    monkeypatch, scf_type, density_fitting
):
    """Both UKS JK backends must avoid four-index AO ERI construction."""

    def forbidden_ao_eri(*args, **kwargs):
        raise AssertionError("production QED-UKS must not call MintsHelper.ao_eri()")

    system = SYSTEM_DEFINITIONS.SYSTEMS["oh_radical"]
    options = SYSTEM_DEFINITIONS.base_psi4_options("uks")
    options.pop("reference")
    options.update(
        basis="sto-3g",
        scf_type=scf_type,
        dft_spherical_points=110,
        dft_radial_points=50,
    )
    config = CQEDConfig(
        lambda_vector=(0.0, 0.0, 0.0),
        omega=SYSTEM_DEFINITIONS.OMEGA,
        psi4_options=options,
        reference="uks",
        functional="PBE0",
        density_fitting=density_fitting,
        charge=system["charge"],
        multiplicity=system["multiplicity"],
        dispersion_policy="none",
        quiet=True,
    )

    psi4.core.clean()
    psi4.core.clean_options()
    monkeypatch.setattr(psi4.core.MintsHelper, "ao_eri", forbidden_ao_eri)
    energy, results = CQEDUSCF(system["geometry"], config).run()

    assert np.isfinite(energy)
    assert results["energy_scf"] == pytest.approx(energy)


@pytest.mark.slow
@pytest.mark.parametrize("system_name", ["oh_radical", "o2_triplet", "nh_triplet"])
@pytest.mark.parametrize("lambda_name", ["zero", "z_only", "x_only", "general"])
def test_canonical_qed_uks_matches_hilbert(system_name, lambda_name):
    system = SYSTEM_DEFINITIONS.SYSTEMS[system_name]
    psi4_options = SYSTEM_DEFINITIONS.base_psi4_options("uks")
    psi4_options.pop("reference")
    config = CQEDConfig(
        lambda_vector=SYSTEM_DEFINITIONS.LAMBDA_VECTORS[lambda_name],
        omega=SYSTEM_DEFINITIONS.OMEGA,
        psi4_options=psi4_options,
        reference="uks",
        functional=SYSTEM_DEFINITIONS.FUNCTIONAL,
        density_fitting=False,
        charge=system["charge"],
        multiplicity=system["multiplicity"],
        dispersion_policy="none",
        quiet=True,
    )
    energy, results = CQEDUSCF(system["geometry"], config).run()
    reference = HILBERT_REFERENCES["systems"][system_name]["methods"]["qed_uks"][
        lambda_name
    ]

    assert energy == pytest.approx(reference["energy"], abs=1.0e-8, rel=0.0)
    assert results["s_squared"] == pytest.approx(
        reference["s_squared"], abs=1.0e-8, rel=0.0
    )
