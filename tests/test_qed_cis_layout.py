"""Tier 1: the QED-CIS block layout is the oracle's Hamiltonian, relabelled.

These tests drive QEDCIS with synthetic orbital blocks instead of a real SCF, so
they are fast and they isolate the assembly from everything upstream of it.  The
central test transcribes helper_CS_CQED_CIS.py's interleaved six-deep loop
literally and shows the two matrices are related by an explicit permutation --
elementwise, not merely in spectrum.  That check pins down every sign, factor,
ladder coefficient and index in the new layout.

See docs/development/QED_RESPONSE_PLAN.md, sections 1 and 2.
"""

import numpy as np
import pytest

from cqed_scf.response import QEDCIS


NO, NV = 3, 4
NMO = NO + NV
OMEGA = 0.1745


def _synthetic_blocks(seed=11, d_scale=0.05):
    """Random but physically-shaped blocks: symmetric dipole, 8-fold-symmetric ERIs."""

    rng = np.random.default_rng(seed)

    # ERIs from a 3-index factorization, which guarantees chemist-notation symmetry
    B = rng.normal(size=(6, NMO, NMO)) * 0.1
    B = 0.5 * (B + B.transpose(0, 2, 1))
    eri = np.einsum("Ppq,Prs->pqrs", B, B)

    d = rng.normal(size=(NMO, NMO)) * d_scale
    d = 0.5 * (d + d.T)
    eps = np.sort(rng.normal(size=NMO))
    return eri, d, eps


def _build(eri, d, eps, n_photon=1, omega=OMEGA):
    """Instantiate QEDCIS with the orbital blocks set by hand.

    response.py imports psi4 lazily inside build_orbital_blocks, so bypassing
    that method exercises the real assembly without needing an SCF.
    """

    cis = QEDCIS(n_photon=n_photon)
    cis.ndocc, cis.nvirt, cis.nmo = NO, NV, NMO
    cis.n_ov = NO * NV
    cis.omega, cis.scf_energy = omega, 0.0
    cis.F_oo, cis.F_vv = np.diag(eps[:NO]), np.diag(eps[NO:])
    cis.F_ov = np.zeros((NO, NV))  # canonical CQED-HF orbitals
    cis.d_oo, cis.d_ov, cis.d_vv = d[:NO, :NO], d[:NO, NO:], d[NO:, NO:]
    cis.ovov = eri[:NO, NO:, :NO, NO:]
    cis.oovv = eri[:NO, :NO, NO:, NO:]
    cis._prepared = True
    return cis


def _reference_interleaved_hamiltonian(eri, d, eps, omega=OMEGA):
    """Literal transcription of helper_CS_CQED_CIS.py's build, ias = 2*(i*nv+a)+s+2."""

    ovov = eri[:NO, NO:, :NO, NO:]
    oovv = eri[:NO, :NO, NO:, NO:]
    eps_o, eps_v = eps[:NO], eps[NO:]

    dim = NO * NV * 2 + 2
    H = np.zeros((dim, dim))
    H[1, 1] = omega
    sq = np.sqrt(omega / 2.0)

    for i in range(NO):
        for a in range(NV):
            A = a + NO
            ia0 = 2 * (i * NV + a) + 2
            ia1 = 2 * (i * NV + a) + 3
            H[0, ia1] = H[ia1, 0] = -np.sqrt(omega) * d[i, A]
            H[1, ia0] = H[ia0, 1] = -np.sqrt(omega) * d[i, A]

    for i in range(NO):
        for a in range(NV):
            A = a + NO
            for s in range(2):
                ias = 2 * (i * NV + a) + s + 2
                for j in range(NO):
                    for b in range(NV):
                        B = b + NO
                        for t in range(2):
                            jbt = 2 * (j * NV + b) + t + 2
                            H[ias, jbt] += (2.0 * ovov[i, a, j, b] - oovv[i, j, a, b]) * (s == t)
                            H[ias, jbt] += (2.0 * d[i, A] * d[j, B]) * (s == t)
                            H[ias, jbt] -= d[i, j] * d[A, B] * (s == t)
                            H[ias, jbt] += eps_v[a] * (s == t) * (a == b) * (i == j)
                            H[ias, jbt] -= eps_o[i] * (s == t) * (a == b) * (i == j)
                            H[ias, jbt] += (omega * t) * (s == t) * (i == j) * (a == b)
                            H[ias, jbt] += np.sqrt(t + 1) * sq * d[i, j] * (s == t + 1) * (a == b)
                            H[ias, jbt] += np.sqrt(t) * sq * d[i, j] * (s == t - 1) * (a == b)
                            H[ias, jbt] -= np.sqrt(t + 1) * sq * d[A, B] * (s == t + 1) * (i == j)
                            H[ias, jbt] -= np.sqrt(t) * sq * d[A, B] * (s == t - 1) * (i == j)
    return H


def _reference_permutation():
    """Map reference interleaved index -> photon-major index."""

    block = 1 + NO * NV
    perm = np.empty(NO * NV * 2 + 2, dtype=int)
    perm[0] = 0 * block + 0
    perm[1] = 1 * block + 0
    for i in range(NO):
        for a in range(NV):
            for s in range(2):
                perm[2 * (i * NV + a) + s + 2] = s * block + 1 + i * NV + a
    return perm


def test_hamiltonian_is_symmetric():
    eri, d, eps = _synthetic_blocks()
    H = _build(eri, d, eps).build_dense_hamiltonian()
    assert np.max(np.abs(H - H.T)) == 0.0


def test_matches_the_reference_ordering_elementwise():
    """The strong form: same matrix, relabelled -- not just the same spectrum."""

    eri, d, eps = _synthetic_blocks()
    H_ours = _build(eri, d, eps).build_dense_hamiltonian()
    H_ref = _reference_interleaved_hamiltonian(eri, d, eps)
    perm = _reference_permutation()

    assert H_ours.shape == H_ref.shape
    np.testing.assert_allclose(H_ours[np.ix_(perm, perm)], H_ref, atol=1e-13)


def test_matches_the_reference_spectrum():
    eri, d, eps = _synthetic_blocks()
    H_ours = _build(eri, d, eps).build_dense_hamiltonian()
    H_ref = _reference_interleaved_hamiltonian(eri, d, eps)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(H_ours), np.linalg.eigvalsh(H_ref), atol=1e-12
    )


@pytest.mark.parametrize("n_photon", [0, 1, 2, 3])
def test_zero_coupling_replicates_the_electronic_spectrum(n_photon):
    """At lambda = 0 every root is (N_ph+1)-fold, offset by n*omega."""

    eri, d, eps = _synthetic_blocks(d_scale=0.0)
    cis = _build(eri, d, eps, n_photon=n_photon)

    spectrum = np.linalg.eigvalsh(cis.build_dense_hamiltonian())
    cis_only = np.linalg.eigvalsh(cis.build_electronic_block())
    expected = np.sort(
        np.concatenate(
            [np.concatenate([[n * OMEGA], cis_only + n * OMEGA]) for n in range(n_photon + 1)]
        )
    )
    assert spectrum.size == (n_photon + 1) * (1 + NO * NV)
    np.testing.assert_allclose(spectrum, expected, atol=1e-12)


def test_block_tridiagonal_structure():
    """Photon blocks differing by more than one must be exactly zero."""

    eri, d, eps = _synthetic_blocks()
    cis = _build(eri, d, eps, n_photon=3)
    H = cis.build_dense_hamiltonian()
    blk = cis.block_size

    for n in range(4):
        for m in range(4):
            if abs(n - m) > 1:
                sub = H[n * blk : (n + 1) * blk, m * blk : (m + 1) * blk]
                assert np.max(np.abs(sub)) == 0.0, f"block ({n},{m}) is not zero"


def test_reference_states_do_not_couple_across_photon_blocks():
    """<Phi_0,n|H|Phi_0,n+-1> = 0: the coherent-state basis removes <d>."""

    eri, d, eps = _synthetic_blocks()
    cis = _build(eri, d, eps, n_photon=3)
    H = cis.build_dense_hamiltonian()
    blk = cis.block_size

    for n in range(3):
        assert H[n * blk, (n + 1) * blk] == 0.0


def test_brillouin_block_vanishes_for_canonical_cqed_orbitals():
    eri, d, eps = _synthetic_blocks()
    cis = _build(eri, d, eps, n_photon=1)
    H = cis.build_dense_hamiltonian()
    blk = cis.block_size

    for n in range(2):
        s = n * blk
        assert np.max(np.abs(H[s, s + 1 : s + blk])) == 0.0


def test_live_brillouin_block_for_non_canonical_orbitals():
    """With F_ov != 0 the reference/singles coupling appears as sqrt(2) F_ia.

    Truncated CI is not orbital invariant, so this changes the spectrum -- that
    is expected physics, and the point of writing A in the general Fock form.
    """

    eri, d, eps = _synthetic_blocks()
    cis = _build(eri, d, eps, n_photon=1)
    rng = np.random.default_rng(5)
    cis.F_ov = rng.normal(size=(NO, NV)) * 0.01

    H = cis.build_dense_hamiltonian()
    blk = cis.block_size
    np.testing.assert_allclose(
        H[0, 1:blk], np.sqrt(2.0) * cis.F_ov.ravel(), atol=1e-14
    )
    assert np.max(np.abs(H - H.T)) == 0.0


def test_results_bookkeeping():
    eri, d, eps = _synthetic_blocks()
    results = _build(eri, d, eps, n_photon=3).kernel()

    total = results.reference_weights.sum(axis=1) + results.singles_weights.sum(axis=1)
    np.testing.assert_allclose(total, 1.0, atol=1e-12)

    assert results.excitation_energies[0] == 0.0
    assert np.all(np.diff(results.eigenvalues) >= -1e-14)
    assert results.photon_numbers.min() > -1e-12
    assert results.photon_numbers.max() < results.n_photon + 1e-12

    # the ground root relaxes below the CQED-SCF reference once lambda != 0
    assert results.eigenvalues[0] < 0.0
    np.testing.assert_allclose(
        results.total_energies, results.scf_energy + results.eigenvalues, atol=1e-14
    )
