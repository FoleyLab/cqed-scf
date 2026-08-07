"""Stage A tests for the central output layer and the config quiet flag."""

from cqed_scf import CQEDCalculator, CQEDConfig
from cqed_scf import output


def _reset_verbosity():
    output.set_quiet(False)


def test_echo_emits_when_not_quiet(capsys):
    _reset_verbosity()
    output.echo("hello")
    captured = capsys.readouterr()
    assert captured.out.rstrip() == "hello"


def test_quiet_suppresses_echo(capsys):
    _reset_verbosity()
    output.set_quiet(True)
    output.echo("hidden")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_quiet_context_suppresses_and_resets(capsys):
    _reset_verbosity()
    output.echo("before")
    with output.quiet_context(True):
        output.echo("inside")
    output.echo("after")
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert lines == ["before", "after"]


def test_quiet_context_noop_when_false(capsys):
    _reset_verbosity()
    with output.quiet_context(False):
        output.echo("kept")
    captured = capsys.readouterr()
    assert captured.out.rstrip() == "kept"


def test_set_quiet_toggle_resets_verbosity(capsys):
    _reset_verbosity()
    output.set_quiet(True)
    assert output.is_quiet()
    output.set_quiet(False)
    assert not output.is_quiet()
    output.echo("normal again")
    assert capsys.readouterr().out.rstrip() == "normal again"


def test_banner_and_property_format(capsys):
    _reset_verbosity()
    output.property_("Total Energy (SCF)", -76.0590812345, "Eh")
    out = capsys.readouterr().out
    assert "@ Total Energy (SCF)" in out
    assert "Eh" in out


def test_energies_property_lines(capsys):
    _reset_verbosity()
    output.energies([("Total Energy", -76.0, "Eh"), ("Correction", 0.0, "Eh")])
    out = capsys.readouterr().out
    assert "@ Total Energy" in out
    assert "@ Correction" in out
    assert out.count("\n") == 2


def test_property_suppressed_when_quiet(capsys):
    _reset_verbosity()
    output.set_quiet(True)
    output.property_("Total Energy (SCF)", -76.0, "Eh")
    assert capsys.readouterr().out == ""


def test_table_emits_aligned_columns(capsys):
    _reset_verbosity()
    output.table(
        ["Label", "Value"],
        [["a", 1.0], ["bb", 2.5]],
        widths=[10, 12],
    )
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 4  # header, separator, two rows
    assert "Label" in lines[0]
    assert "-" * 10 in lines[1]


def test_table_suppressed_when_quiet(capsys):
    _reset_verbosity()
    output.set_quiet(True)
    output.table(["A"], [["x"]], widths=[4])
    assert capsys.readouterr().out == ""


def test_config_quiet_field_defaults_false():
    config = CQEDConfig(
        lambda_vector=(0.0, 0.0, 0.0),
        omega=0.1,
        reference="rhf",
        functional=None,
    )
    assert config.quiet is False


def test_config_quiet_true_and_copy_with():
    config = CQEDConfig(
        lambda_vector=(0.0, 0.0, 0.0),
        omega=0.1,
        reference="rhf",
        quiet=True,
    )
    assert config.quiet is True
    assert config.copy_with(quiet=False).quiet is False


def test_from_legacy_kwargs_quiet():
    config = CQEDConfig.from_legacy_kwargs(
        lambda_vector=(0.0, 0.0, 0.0),
        psi4_options={},
        quiet=True,
    )
    assert config.quiet is True


def test_calculator_quiet_property_and_config():
    calc = CQEDCalculator(
        lambda_vector=(0.0, 0.0, 0.0),
        psi4_options={},
        quiet=True,
    )
    assert calc.quiet is True
    assert calc.config.quiet is True
    calc.quiet = False
    assert calc.config.quiet is False
    calc.quiet = True
    assert calc.config.quiet is True


def test_quiet_orthogonal_to_debug():
    config = CQEDConfig(
        lambda_vector=(0.0, 0.0, 0.0),
        debug=True,
        quiet=True,
    )
    assert config.debug is True
    assert config.quiet is True

    quiet_false_config = config.copy_with(quiet=False)
    assert quiet_false_config.debug is True
    assert quiet_false_config.quiet is False


def test_sapt_component_emits_eh_property(capsys):
    _reset_verbosity()
    output.sapt_component("Elst10, r", -0.5)
    out = capsys.readouterr().out
    assert "@ Elst10, r" in out
    assert "-0.500000000000" in out
    assert "Eh" in out


def test_sapt_component_suppressed_when_quiet(capsys):
    _reset_verbosity()
    output.set_quiet(True)
    output.sapt_component("Elst10, r", -0.5)
    assert capsys.readouterr().out == ""