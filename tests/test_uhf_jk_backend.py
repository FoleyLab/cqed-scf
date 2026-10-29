"""Regression gates for the Psi4 JK-backed QED-UHF implementation."""

import numpy as np
import psi4
import pytest

from cqed_scf import CQEDConfig
from cqed_scf.uscf import CQEDUSCF


OH_STO3G = """
0 2
O  0.0  0.0  0.0
H  0.0  0.0  0.9693
no_reorient
no_com
symmetry c1
units angstrom
"""


def _config(*, scf_type="pk", density_fitting=False, functional=None):
    return CQEDConfig(
        lambda_vector=(0.02, 0.01, 0.03),
        omega=0.1,
        psi4_options={
            "basis": "sto-3g",
            "scf_type": scf_type,
            "e_convergence": 1.0e-10,
            "d_convergence": 1.0e-8,
        },
        reference="uhf",
        functional=functional,
        density_fitting=density_fitting,
        charge=0,
        multiplicity=2,
        dispersion_policy="none",
        quiet=True,
    )


def test_spin_resolved_psi4_jk_matches_dense_contractions():
    """Lock the two-entry JK queue ordering to alpha, then beta."""

    psi4.core.clean()
    psi4.core.clean_options()
    engine = CQEDUSCF(OH_STO3G, _config())
    engine._prepare_options()
    molecule = psi4.geometry(OH_STO3G)
    _, engine.wfn = psi4.energy("scf", molecule=molecule, return_wfn=True)
    engine._build_jk()

    Ca_occ = np.asarray(engine.wfn.Ca())[:, : engine.wfn.nalpha()]
    Cb_occ = np.asarray(engine.wfn.Cb())[:, : engine.wfn.nbeta()]
    Da = Ca_occ @ Ca_occ.T
    Db = Cb_occ @ Cb_occ.T
    Ja, Jb, Ka, Kb, _, _ = engine._build_JK(Ca_occ, Cb_occ)

    eri = np.asarray(psi4.core.MintsHelper(engine.wfn.basisset()).ao_eri())
    np.testing.assert_allclose(Ja, np.einsum("pqrs,rs->pq", eri, Da), atol=1e-10)
    np.testing.assert_allclose(Jb, np.einsum("pqrs,rs->pq", eri, Db), atol=1e-10)
    np.testing.assert_allclose(Ka, np.einsum("prqs,rs->pq", eri, Da), atol=1e-10)
    np.testing.assert_allclose(Kb, np.einsum("prqs,rs->pq", eri, Db), atol=1e-10)


@pytest.mark.parametrize(
    "scf_type,density_fitting", [("pk", False), ("df", True)], ids=["pk", "df"]
)
def test_uhf_runs_when_dense_ao_eri_is_forbidden(
    monkeypatch, scf_type, density_fitting
):
    """Neither PK nor DF may allocate a four-index AO ERI in CQEDUSCF."""

    def forbidden_ao_eri(*args, **kwargs):
        raise AssertionError("production QED-UHF must not call MintsHelper.ao_eri()")

    psi4.core.clean()
    psi4.core.clean_options()
    monkeypatch.setattr(psi4.core.MintsHelper, "ao_eri", forbidden_ao_eri)
    engine = CQEDUSCF(
        OH_STO3G,
        _config(scf_type=scf_type, density_fitting=density_fitting),
    )
    energy, results = engine.run()

    assert np.isfinite(energy)
    assert results["energy_scf"] == pytest.approx(energy)


def test_final_orbital_energies_belong_to_cqed_focks():
    psi4.core.clean()
    psi4.core.clean_options()
    _, results = CQEDUSCF(OH_STO3G, _config()).run()

    for spin in ("a", "b"):
        C = results[f"C{spin}"]
        F_mo = C.T @ results[f"F{spin}"] @ C
        eps = results[f"orbital_energies_{spin}"]
        np.testing.assert_allclose(np.diag(F_mo), eps, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(
            F_mo - np.diag(np.diag(F_mo)), 0.0, atol=1e-10, rtol=0.0
        )
        assert not np.shares_memory(
            eps, results[f"canonical_orbital_energies_{spin}"]
        )


def test_uhf_reference_wins_over_unrelated_functional():
    engine = CQEDUSCF(OH_STO3G, _config(functional="pbe0"))

    assert engine.method == "uhf"
    assert engine.functional is None
    assert not engine.is_dft
