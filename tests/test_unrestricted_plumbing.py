"""Plumbing for unrestricted (UHF / UKS) references.

These cover the wiring, not the physics: config validation, the geometry-vs-config
consistency check, dispatch into :class:`cqed_scf.uscf.CQEDUSCF`, and the guards
on features that have no unrestricted implementation yet.

The config tests import from ``cqed_scf.references`` directly, which pulls in no
Psi4, so they stay fast.
"""

import numpy as np
import pytest

from cqed_scf.references import CQEDConfig

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


def make_config(**overrides):
    kwargs = dict(lambda_vector=LAMBDA, omega=0.1, psi4_options={})
    kwargs.update(overrides)
    return CQEDConfig(**kwargs)


# ---------------------------------------------------------------------------
# CQEDConfig: reference and multiplicity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("multiplicity", [1, 2, 3])
def test_uhf_accepts_any_multiplicity(multiplicity):
    config = make_config(reference="uhf", multiplicity=multiplicity)
    assert config.reference == "uhf"
    assert config.is_unrestricted and config.is_hf
    assert config.multiplicity == multiplicity


@pytest.mark.parametrize("multiplicity", [1, 2, 3])
def test_uks_accepts_any_multiplicity(multiplicity):
    config = make_config(reference="uks", functional="pbe0", multiplicity=multiplicity)
    assert config.reference == "uks"
    assert config.is_unrestricted and config.is_ks
    assert config.base_scf_functional == "pbe0"


@pytest.mark.parametrize("alias", ["unrestricted-hf", "UHF", " uhf "])
def test_unrestricted_aliases_normalize(alias):
    assert make_config(reference=alias, multiplicity=2).reference == "uhf"


def test_uhf_reports_no_functional_to_the_engine():
    """base_scf_functional must be None for UHF even if a functional is set."""
    config = make_config(reference="uhf", functional="pbe0", multiplicity=2)
    assert config.base_scf_functional is None


def test_restricted_rejects_open_shell_multiplicity():
    with pytest.raises(ValueError) as excinfo:
        make_config(reference="rhf", multiplicity=3)
    message = str(excinfo.value)
    assert "multiplicity=3" in message
    assert "'rhf'" in message
    assert "'uhf'" in message


def test_inferred_rks_reports_the_inference_and_suggests_uks():
    """reference=None + functional infers 'rks'; the error must say so."""
    with pytest.raises(ValueError) as excinfo:
        make_config(functional="pbe0", multiplicity=3)
    message = str(excinfo.value)
    assert "defaulted" in message
    assert "'rks'" in message
    assert "'uks'" in message


def test_multiplicity_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        make_config(reference="uhf", multiplicity=0)


@pytest.mark.parametrize("bad", [2.5, "2", None])
def test_non_integral_multiplicity_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        make_config(reference="uhf", multiplicity=bad)


def test_non_integral_charge_rejected():
    with pytest.raises((TypeError, ValueError)):
        make_config(reference="rhf", charge=0.5)


def test_reference_in_psi4_options_rejected():
    """It would be silently overridden by the SCF engine, so refuse it."""
    with pytest.raises(ValueError, match="psi4_options"):
        make_config(reference="uhf", multiplicity=2, psi4_options={"reference": "uhf"})


def test_copy_with_revalidates():
    config = make_config(reference="rhf", multiplicity=1)
    with pytest.raises(ValueError):
        config.copy_with(multiplicity=3)


# ---------------------------------------------------------------------------
# geometry <-> config consistency
# ---------------------------------------------------------------------------


def test_geometry_multiplicity_mismatch_raises():
    from cqed_scf.geometry import validate_geometry_against_config

    config = make_config(reference="uhf", multiplicity=1)
    with pytest.raises(ValueError) as excinfo:
        validate_geometry_against_config(OH_DOUBLET, config)
    message = str(excinfo.value)
    assert "multiplicity" in message
    assert "1" in message and "2" in message


def test_geometry_multiplicity_match_passes():
    from cqed_scf.geometry import validate_geometry_against_config

    config = make_config(reference="uhf", multiplicity=2)
    assert validate_geometry_against_config(OH_DOUBLET, config) == (0, 2)


def test_geometry_charge_mismatch_raises():
    from cqed_scf.geometry import validate_geometry_against_config

    config = make_config(reference="rhf", charge=0)
    with pytest.raises(ValueError, match="charge"):
        validate_geometry_against_config(MGH_CATION, config)


def test_zmatrix_charge_is_read_correctly():
    """The heuristic string parser drops Z-matrix lines; Psi4 does not."""
    from cqed_scf.geometry import read_charge_and_multiplicity

    assert read_charge_and_multiplicity(MGH_CATION) == (1, 1)


def test_neutral_singlet_defaults_still_pass():
    """What every pre-existing test relies on: no charge line, closed shell."""
    from cqed_scf.geometry import validate_geometry_against_config

    geometry = "\n".join(
        line for line in WATER_SINGLET.splitlines() if line.strip() != "0 1"
    )
    assert validate_geometry_against_config(geometry, make_config()) == (0, 1)


def test_missing_charge_line_is_not_assumed_neutral_singlet():
    """OH with no leading line parses as 0 2, not 0 1 -- so it must not pass."""
    from cqed_scf.geometry import read_charge_and_multiplicity, validate_geometry_against_config

    geometry = "\n".join(
        line for line in OH_DOUBLET.splitlines() if line.strip() != "0 2"
    )
    assert read_charge_and_multiplicity(geometry) == (0, 2)
    with pytest.raises(ValueError, match="multiplicity"):
        validate_geometry_against_config(geometry, make_config())


# ---------------------------------------------------------------------------
# dispatch and guards
# ---------------------------------------------------------------------------


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


def test_energy_rejects_mismatched_geometry_before_running():
    calc = build_calculator(reference="uhf", multiplicity=3)
    with pytest.raises(ValueError, match="multiplicity"):
        calc.energy(OH_DOUBLET)


@pytest.mark.parametrize("method", ["energy_and_gradient", "energy_and_projected_gradient"])
def test_gradients_refuse_unrestricted(method):
    calc = build_calculator(reference="uhf", multiplicity=2)
    with pytest.raises(NotImplementedError, match="gradients"):
        getattr(calc, method)(OH_DOUBLET)


def test_gradient_guard_fires_before_any_scf():
    """A basis this large would be unmissably slow if an SCF actually ran."""
    calc = build_calculator(reference="uhf", multiplicity=2)
    calc.config.psi4_options["basis"] = "aug-cc-pv5z"
    with pytest.raises(NotImplementedError):
        calc.energy_and_gradient(OH_DOUBLET)


@pytest.mark.parametrize("method", ["cis", "response"])
def test_response_refuses_unrestricted(method):
    calc = build_calculator(reference="uhf", multiplicity=2)
    with pytest.raises(NotImplementedError, match="[Uu]nrestricted"):
        getattr(calc, method)(OH_DOUBLET)


@pytest.mark.parametrize("method", ["cis", "response"])
def test_response_refuses_unrestricted_via_scf_results(method):
    """The scf_results= path skips _run_scf, so it needs its own guard."""
    calc = build_calculator(reference="uhf", multiplicity=2)
    with pytest.raises(NotImplementedError, match="[Uu]nrestricted"):
        getattr(calc, method)(scf_results={"bogus": True})


def test_restricted_engine_refuses_unrestricted_method_by_name():
    from cqed_scf import CQEDSCF

    with pytest.raises(ValueError, match="CQEDUSCF"):
        CQEDSCF(
            geometry=WATER_SINGLET,
            lambda_vector=LAMBDA,
            psi4_options={"basis": "sto-3g"},
            omega=0.1,
            method="uhf",
        )


def test_restricted_engine_refuses_open_shell_geometry():
    """Direct CQEDSCF construction bypasses the calculator; the backstop holds."""
    from cqed_scf import CQEDSCF

    engine = CQEDSCF(
        geometry=OH_DOUBLET,
        lambda_vector=LAMBDA,
        psi4_options={"basis": "sto-3g", "scf_type": "pk"},
        omega=0.1,
        method="rhf",
        quiet=True,
    )
    with pytest.raises(ValueError, match="CQEDUSCF"):
        engine.run()


def test_unrestricted_engines_are_exported():
    import cqed_scf

    assert cqed_scf.CQEDUSCF is cqed_scf.CQEDUHFSCF
    assert cqed_scf.CQEDUGradient is cqed_scf.CQEDUHFGradient
