# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import time

import numpy as np

from psi4 import core

from psi4.driver.p4util import solvers
from psi4.driver.p4util.exceptions import *
from psi4.driver.procrouting.sapt.sapt_util import print_sapt_var

from .. import output
from .dse_jk import DSEJK, PauliFierzJK, DSECPHF


def _native_jk(jk):
    """Return the real psi4.core.JK object if jk is a wrapper."""
    if hasattr(jk, "native_jk"):
        return jk.native_jk()
    return jk


def _effective_jk(jk, dse_jk=None):
    """Return a PauliFierzJK wrapper when a DSEJK object is supplied."""
    if isinstance(jk, PauliFierzJK):
        return jk
    if dse_jk is None:
        return jk
    return PauliFierzJK(jk, dse_jk)


def _matrix_jk(cache, jk=None):
    """Return the composite JK object used for Python SAPT matrix builds."""
    return cache.get("jk", jk)


def _dse_cavity_terms(
    shape,
    *,
    dse_jk=None,
    d_ao=None,
    d_ao_A=None,
    d_ao_B=None,
    d_exp_el_A=None,
    d_exp_el_B=None,
    include_cavity_terms=True,
):
    """Return coherent-state one-electron and scalar DSE cache terms."""

    zero = np.zeros(shape)
    V_A_cavity = core.Matrix.from_array(zero)
    V_B_cavity = core.Matrix.from_array(zero)
    dse_constant = 0.0

    if not include_cavity_terms:
        return V_A_cavity, V_B_cavity, dse_constant

    if d_ao is None and dse_jk is not None:
        d_ao = dse_jk.d_ao
    if d_ao_A is None:
        d_ao_A = d_ao
    if d_ao_B is None:
        d_ao_B = d_ao

    missing = (
        d_ao_A is None
        or d_ao_B is None
        or d_exp_el_A is None
        or d_exp_el_B is None
    )
    if missing:
        return V_A_cavity, V_B_cavity, dse_constant

    d_ao_A = np.asarray(d_ao_A, dtype=float)
    d_ao_B = np.asarray(d_ao_B, dtype=float)
    if d_ao_A.shape != shape or d_ao_B.shape != shape:
        raise ValueError(
            "DSE dipole matrices must match the shared AO shape; "
            f"got {d_ao_A.shape} and {d_ao_B.shape}, expected {shape}."
        )

    # From reference implementation: potential from monomer A carries -<d_A>_el d_B,
    # and potential from monomer B carries -<d_B>_el d_A.
    V_A_cavity = core.Matrix.from_array(
        np.ascontiguousarray(-float(d_exp_el_A) * d_ao_B)
    )
    V_B_cavity = core.Matrix.from_array(
        np.ascontiguousarray(-float(d_exp_el_B) * d_ao_A)
    )
    dse_constant = float(d_exp_el_A) * float(d_exp_el_B)
    return V_A_cavity, V_B_cavity, dse_constant


def build_sapt_jk_cache(
    wfn_A,
    wfn_B,
    jk,
    do_print=True,
    dse_jk=None,
    dse_cphf_A=None,
    dse_cphf_B=None,
    d_ao=None,
    d_ao_A=None,
    d_ao_B=None,
    d_exp_el_A=None,
    d_exp_el_B=None,
    include_cavity_terms=True,
    nuclear_repulsion_energy=None,
):
    """
    Constructs the DCBS cache data required to compute ELST/EXCH/IND
    """

    do_print = do_print and not output.is_quiet()

    if do_print:
        core.print_out("\n  ==> Preparing SAPT Data Cache <== \n\n")
    jk_eff = _effective_jk(jk, dse_jk)
    dse_jk_eff = jk_eff.dse_jk if isinstance(jk_eff, PauliFierzJK) else dse_jk
    jk_eff.print_header()

    # empty dictionary
    cache = {}
    # add wfn_A
    cache["wfn_A"] = wfn_A
    # add wfn_B
    cache["wfn_B"] = wfn_B

    # Effective JK interface used by the SAPT routines:
    # ordinary ERI J/K plus DSE J/K when dse_jk is present.
    cache["jk"] = jk_eff

    # Underlying native psi4.core.JK object:
    # ordinary electron-repulsion-integral J/K only.
    cache["native_jk"] = _native_jk(jk_eff)

    # Standalone DSEJK object:
    # provides only the DSE J/K contribution.
    cache["dse_jk"] = dse_jk_eff

    # First grab the orbitals
    cache["Cocc_A"] = wfn_A.Ca_subset("AO", "OCC")
    cache["Cvir_A"] = wfn_A.Ca_subset("AO", "VIR")

    cache["Cocc_B"] = wfn_B.Ca_subset("AO", "OCC")
    cache["Cvir_B"] = wfn_B.Ca_subset("AO", "VIR")

    if dse_jk_eff is not None and dse_jk_eff.is_active():
        # hand only the DSE part of J/K to the DSECPHF objects, which are used to compute the DSE response contributions.
        if dse_cphf_A is None:
            dse_cphf_A = DSECPHF(dse_jk=dse_jk_eff, Cocc=cache["Cocc_A"], Cvir=cache["Cvir_A"])
        if dse_cphf_B is None:
            dse_cphf_B = DSECPHF(dse_jk=dse_jk_eff, Cocc=cache["Cocc_B"], Cvir=cache["Cvir_B"])

    # add instances of DSECPHF to the cache for later use in computing the DSE response contributions to induction.
    cache["dse_cphf_A"] = dse_cphf_A
    cache["dse_cphf_B"] = dse_cphf_B

    # add orbital energies to cache
    cache["eps_occ_A"] = wfn_A.epsilon_a_subset("AO", "OCC")
    cache["eps_vir_A"] = wfn_A.epsilon_a_subset("AO", "VIR")

    cache["eps_occ_B"] = wfn_B.epsilon_a_subset("AO", "OCC")
    cache["eps_vir_B"] = wfn_B.epsilon_a_subset("AO", "VIR")

    # Build the densities from monomer orbitals
    cache["D_A"] = core.doublet(cache["Cocc_A"], cache["Cocc_A"], False, True)
    cache["D_B"] = core.doublet(cache["Cocc_B"], cache["Cocc_B"], False, True)

    cache["P_A"] = core.doublet(cache["Cvir_A"], cache["Cvir_A"], False, True)
    cache["P_B"] = core.doublet(cache["Cvir_B"], cache["Cvir_B"], False, True)

    # Potential ints
    mints = core.MintsHelper(wfn_A.basisset())

    # this is the standard one-electron potential for monomer A
    cache["V_A_standard"] = mints.ao_potential().clone()

    # this is the standard one-electron potential for monomer B
    mints = core.MintsHelper(wfn_B.basisset())
    cache["V_B_standard"] = mints.ao_potential().clone()

    # capture the cavity contributions to the one-electron potentials and the DSE constant
    cache["V_A_cavity"], cache["V_B_cavity"], cache["dse_constant"] = _dse_cavity_terms(
        cache["V_A_standard"].shape,
        dse_jk=dse_jk_eff,
        d_ao=d_ao,
        d_ao_A=d_ao_A,
        d_ao_B=d_ao_B,
        d_exp_el_A=d_exp_el_A,
        d_exp_el_B=d_exp_el_B,
        include_cavity_terms=include_cavity_terms,
    )

    # Add the cavity contributions to the standard one-electron potentials to form the total one-electron potentials for each monomer.
    cache["V_A"] = cache["V_A_standard"].clone()

    # Note the negative sign on the cavity term is handled by _dse_cavity_terms, which returns -<d>_el d for each monomer.
    cache["V_A"].axpy(1.0, cache["V_A_cavity"])

    # repeat for monomer B
    cache["V_B"] = cache["V_B_standard"].clone()
    cache["V_B"].axpy(1.0, cache["V_B_cavity"])

    # Anything else we might need
    cache["S"] = wfn_A.S().clone()

    # J and K matrices
    jk_eff.C_clear()

    # Normal J/K for Monomer A
    jk_eff.C_left_add(wfn_A.Ca_subset("SO", "OCC"))
    jk_eff.C_right_add(wfn_A.Ca_subset("SO", "OCC"))

    # Normal J/K for Monomer B
    jk_eff.C_left_add(wfn_B.Ca_subset("SO", "OCC"))
    jk_eff.C_right_add(wfn_B.Ca_subset("SO", "OCC"))

    # K_O J/K
    C_O_A = core.triplet(cache["D_B"], cache["S"], cache["Cocc_A"], False, False, False)
    jk_eff.C_left_add(C_O_A)
    jk_eff.C_right_add(cache["Cocc_A"])

    jk_eff.compute()

    # Clone them as the JK object will overwrite.
    cache["J_A"] = jk_eff.J()[0].clone()
    cache["K_A"] = jk_eff.K()[0].clone()

    cache["J_B"] = jk_eff.J()[1].clone()
    cache["K_B"] = jk_eff.K()[1].clone()

    cache["J_O"] = jk_eff.J()[2].clone()
    cache["K_O"] = jk_eff.K()[2].clone()
    cache["K_O"].transpose_this()

    monA_nr = wfn_A.molecule().nuclear_repulsion_energy()
    monB_nr = wfn_B.molecule().nuclear_repulsion_energy()
    dimer_nr = wfn_A.molecule().extract_subsets([1, 2]).nuclear_repulsion_energy()

    if nuclear_repulsion_energy is None:
        nuclear_repulsion_energy = dimer_nr - monA_nr - monB_nr

    # capture the nuclear repulsion energy in the cache for later use in computing the electrostatics.

    # this is the standard nuclear repulsion energy for the dimer, without any DSE contributions.
    cache["nuclear_repulsion_energy_standard"] = float(nuclear_repulsion_energy)
    # this is the total nuclear repulsion energy for the dimer, including the DSE constant contribution.
    cache["nuclear_repulsion_energy"] = (
        cache["nuclear_repulsion_energy_standard"] + cache["dse_constant"]
    )

    return cache


def electrostatics(cache, do_print=True):
    """
    Computes the E10 electrostatics from a build_sapt_jk_cache datacache.
    """

    do_print = do_print and not output.is_quiet()

    if do_print:
        core.print_out("\n  ==> E10 Electostatics <== \n\n")

    # ELST
    Elst10 = 4.0 * cache["D_B"].vector_dot(cache["J_A"])
    Elst10 += 2.0 * cache["D_A"].vector_dot(cache["V_B"])
    Elst10 += 2.0 * cache["D_B"].vector_dot(cache["V_A"])
    Elst10 += cache["nuclear_repulsion_energy"]

    if do_print:
        core.print_out(print_sapt_var("Elst10,r ", Elst10, short=True))
        core.print_out("\n")

    return {"Elst10,r": Elst10}


def exchange(cache, jk=None, do_print=True):
    """
    Computes the E10 exchange (S^2 and S^inf) from a build_sapt_jk_cache datacache.
    """

    do_print = do_print and not output.is_quiet()

    if do_print:
        core.print_out("\n  ==> E10 Exchange <== \n\n")
    jk = _matrix_jk(cache, jk)

    # Build potenitals
    h_A = cache["V_A"].clone()
    h_A.axpy(2.0, cache["J_A"])
    h_A.axpy(-1.0, cache["K_A"])

    h_B = cache["V_B"].clone()
    h_B.axpy(2.0, cache["J_B"])
    h_B.axpy(-1.0, cache["K_B"])

    w_A = cache["V_A"].clone()
    w_A.axpy(2.0, cache["J_A"])

    w_B = cache["V_B"].clone()
    w_B.axpy(2.0, cache["J_B"])

    # Build inverse exchange metric
    nocc_A = cache["Cocc_A"].shape[1]
    nocc_B = cache["Cocc_B"].shape[1]
    SAB = core.triplet(cache["Cocc_A"], cache["S"], cache["Cocc_B"], True, False, False)
    num_occ = nocc_A + nocc_B

    Sab = core.Matrix(num_occ, num_occ)
    Sab.np[:nocc_A, nocc_A:] = SAB.np
    Sab.np[nocc_A:, :nocc_A] = SAB.np.T
    Sab.np[np.diag_indices_from(Sab.np)] += 1
    Sab.power(-1.0, 1.e-14)
    Sab.np[np.diag_indices_from(Sab.np)] -= 1.0

    Tmo_AA = core.Matrix.from_array(Sab.np[:nocc_A, :nocc_A])
    Tmo_BB = core.Matrix.from_array(Sab.np[nocc_A:, nocc_A:])
    Tmo_AB = core.Matrix.from_array(Sab.np[:nocc_A, nocc_A:])

    T_A = np.dot(cache["Cocc_A"], Tmo_AA).dot(cache["Cocc_A"].np.T)
    T_B = np.dot(cache["Cocc_B"], Tmo_BB).dot(cache["Cocc_B"].np.T)
    T_AB = np.dot(cache["Cocc_A"], Tmo_AB).dot(cache["Cocc_B"].np.T)

    S = cache["S"]

    D_A = cache["D_A"]
    P_A = cache["P_A"]

    D_B = cache["D_B"]
    P_B = cache["P_B"]

    # Compute the J and K matrices
    jk.C_clear()

    jk.C_left_add(cache["Cocc_A"])
    jk.C_right_add(core.doublet(cache["Cocc_A"], Tmo_AA, False, False))

    jk.C_left_add(cache["Cocc_B"])
    jk.C_right_add(core.doublet(cache["Cocc_A"], Tmo_AB, False, False))

    jk.C_left_add(cache["Cocc_A"])
    jk.C_right_add(core.Matrix.chain_dot(P_B, S, cache["Cocc_A"]))

    jk.compute()

    JT_A, JT_AB, Jij = jk.J()
    KT_A, KT_AB, Kij = jk.K()

    # Start S^2
    Exch_s2 = 0.0

    tmp = core.Matrix.chain_dot(D_A, S, D_B, S, P_A)
    Exch_s2 -= 2.0 * w_B.vector_dot(tmp)

    tmp = core.Matrix.chain_dot(D_B, S, D_A, S, P_B)
    Exch_s2 -= 2.0 * w_A.vector_dot(tmp)

    tmp = core.Matrix.chain_dot(P_A, S, D_B)
    Exch_s2 -= 2.0 * Kij.vector_dot(tmp)

    if do_print:
        core.print_out(print_sapt_var("Exch10(S^2) ", Exch_s2, short=True))
        core.print_out("\n")

    # Start Sinf
    Exch10 = 0.0
    Exch10 -= 2.0 * np.vdot(cache["D_A"], cache["K_B"])
    Exch10 += 2.0 * np.vdot(T_A, h_B.np)
    Exch10 += 2.0 * np.vdot(T_B, h_A.np)
    Exch10 += 2.0 * np.vdot(T_AB, h_A.np + h_B.np)
    Exch10 += 4.0 * np.vdot(T_B, JT_AB.np - 0.5 * KT_AB.np)
    Exch10 += 4.0 * np.vdot(T_A, JT_AB.np - 0.5 * KT_AB.np)
    Exch10 += 4.0 * np.vdot(T_B, JT_A.np - 0.5 * KT_A.np)
    Exch10 += 4.0 * np.vdot(T_AB, JT_AB.np - 0.5 * KT_AB.np.T)

    if do_print:
        core.set_variable("Exch10", Exch10)
        core.print_out(print_sapt_var("Exch10", Exch10, short=True))
        core.print_out("\n")

    return {"Exch10(S^2)": Exch_s2, "Exch10": Exch10}


def induction(
    cache,
    jk=None,
    do_print=True,
    maxiter=12,
    conv=1.e-8,
    do_response=True,
    Sinf=False,
    sapt_jk_B=None,
    diagnostics=None,
):
    """
    Compute Ind20 and Exch-Ind20 quantities from a SAPT cache and JK object.
    """

    do_print = do_print and not output.is_quiet()

    if do_print:
        core.print_out("\n  ==> E20 Induction <== \n\n")
    jk = _matrix_jk(cache, jk)

    # Build Induction and Exchange-Induction potentials
    S = cache["S"]

    D_A = cache["D_A"]
    V_A = cache["V_A"]
    J_A = cache["J_A"]
    K_A = cache["K_A"]

    D_B = cache["D_B"]
    V_B = cache["V_B"]
    J_B = cache["J_B"]
    K_B = cache["K_B"]

    K_O = cache["K_O"]
    J_O = cache["J_O"]

    jk.C_clear()

    jk.C_left_add(core.Matrix.chain_dot(D_B, S, cache["Cocc_A"]))
    jk.C_right_add(cache["Cocc_A"])

    jk.C_left_add(core.Matrix.chain_dot(D_B, S, D_A, S, cache["Cocc_B"]))
    jk.C_right_add(cache["Cocc_B"])

    jk.C_left_add(core.Matrix.chain_dot(D_A, S, D_B, S, cache["Cocc_A"]))
    jk.C_right_add(cache["Cocc_A"])

    jk.compute()

    J_Ot, J_P_B, J_P_A = jk.J()
    K_Ot, K_P_B, K_P_A = jk.K()

    # Exch-Ind Potential A
    EX_A = K_B.clone()
    EX_A.scale(-1.0)
    EX_A.axpy(-2.0, J_O)
    EX_A.axpy(1.0, K_O)
    EX_A.axpy(2.0, J_P_B)

    EX_A.axpy(-1.0, core.Matrix.chain_dot(S, D_B, V_A))
    EX_A.axpy(-2.0, core.Matrix.chain_dot(S, D_B, J_A))
    EX_A.axpy(1.0, core.Matrix.chain_dot(S, D_B, K_A))
    EX_A.axpy(1.0, core.Matrix.chain_dot(S, D_B, S, D_A, V_B))
    EX_A.axpy(2.0, core.Matrix.chain_dot(S, D_B, S, D_A, J_B))
    EX_A.axpy(1.0, core.Matrix.chain_dot(S, D_B, V_A, D_B, S))
    EX_A.axpy(2.0, core.Matrix.chain_dot(S, D_B, J_A, D_B, S))
    EX_A.axpy(-1.0, core.Matrix.chain_dot(S, D_B, K_O, trans=[False, False, True]))

    EX_A.axpy(-1.0, core.Matrix.chain_dot(V_B, D_B, S))
    EX_A.axpy(-2.0, core.Matrix.chain_dot(J_B, D_B, S))
    EX_A.axpy(1.0, core.Matrix.chain_dot(K_B, D_B, S))
    EX_A.axpy(1.0, core.Matrix.chain_dot(V_B, D_A, S, D_B, S))
    EX_A.axpy(2.0, core.Matrix.chain_dot(J_B, D_A, S, D_B, S))
    EX_A.axpy(-1.0, core.Matrix.chain_dot(K_O, D_B, S))

    EX_A = core.Matrix.chain_dot(cache["Cocc_A"], EX_A, cache["Cvir_A"], trans=[True, False, False])

    # Exch-Ind Potential B
    EX_B = K_A.clone()
    EX_B.scale(-1.0)
    EX_B.axpy(-2.0, J_O)
    EX_B.axpy(1.0, K_O.transpose())
    EX_B.axpy(2.0, J_P_A)

    EX_B.axpy(-1.0, core.Matrix.chain_dot(S, D_A, V_B))
    EX_B.axpy(-2.0, core.Matrix.chain_dot(S, D_A, J_B))
    EX_B.axpy(1.0, core.Matrix.chain_dot(S, D_A, K_B))
    EX_B.axpy(1.0, core.Matrix.chain_dot(S, D_A, S, D_B, V_A))
    EX_B.axpy(2.0, core.Matrix.chain_dot(S, D_A, S, D_B, J_A))
    EX_B.axpy(1.0, core.Matrix.chain_dot(S, D_A, V_B, D_A, S))
    EX_B.axpy(2.0, core.Matrix.chain_dot(S, D_A, J_B, D_A, S))
    EX_B.axpy(-1.0, core.Matrix.chain_dot(S, D_A, K_O))

    EX_B.axpy(-1.0, core.Matrix.chain_dot(V_A, D_A, S))
    EX_B.axpy(-2.0, core.Matrix.chain_dot(J_A, D_A, S))
    EX_B.axpy(1.0, core.Matrix.chain_dot(K_A, D_A, S))
    EX_B.axpy(1.0, core.Matrix.chain_dot(V_A, D_B, S, D_A, S))
    EX_B.axpy(2.0, core.Matrix.chain_dot(J_A, D_B, S, D_A, S))
    EX_B.axpy(-1.0, core.Matrix.chain_dot(K_O, D_A, S, trans=[True, False, False]))

    EX_B = core.Matrix.chain_dot(cache["Cocc_B"], EX_B, cache["Cvir_B"], trans=[True, False, False])

    # Build electrostatic potenital
    w_A = cache["V_A"].clone()
    w_A.axpy(2.0, cache["J_A"])

    w_B = cache["V_B"].clone()
    w_B.axpy(2.0, cache["J_B"])

    w_B_MOA = core.triplet(cache["Cocc_A"], w_B, cache["Cvir_A"], True, False, False)
    w_A_MOB = core.triplet(cache["Cocc_B"], w_A, cache["Cvir_B"], True, False, False)

    if diagnostics is not None:
        diagnostics["w_B_MOA"] = w_B_MOA.np.copy()
        diagnostics["w_A_MOB"] = w_A_MOB.np.copy()
        diagnostics["EX_A"] = EX_A.np.copy()
        diagnostics["EX_B"] = EX_B.np.copy()

    # Do uncoupled
    core.print_out("   => Uncoupled Induction <= \n\n")
    unc_x_B_MOA = w_B_MOA.clone()
    unc_x_B_MOA.np[:] /= (cache["eps_occ_A"].np.reshape(-1, 1) - cache["eps_vir_A"].np)
    unc_x_A_MOB = w_A_MOB.clone()
    unc_x_A_MOB.np[:] /= (cache["eps_occ_B"].np.reshape(-1, 1) - cache["eps_vir_B"].np)

    if diagnostics is not None:
        diagnostics["uncoupled_amplitudes"] = {
            "A<-B": unc_x_B_MOA.np.copy(),
            "A->B": unc_x_A_MOB.np.copy(),
        }

    unc_ind_ab = 2.0 * unc_x_B_MOA.vector_dot(w_B_MOA)
    unc_ind_ba = 2.0 * unc_x_A_MOB.vector_dot(w_A_MOB)
    unc_indexch_ab = 2.0 * unc_x_B_MOA.vector_dot(EX_A)
    unc_indexch_ba = 2.0 * unc_x_A_MOB.vector_dot(EX_B)

    ret = {}
    ret["Ind20,u (A<-B)"] = unc_ind_ab
    ret["Ind20,u (A->B)"] = unc_ind_ba
    ret["Ind20,u"] = unc_ind_ab + unc_ind_ba
    ret["Exch-Ind20,u (A<-B)"] = unc_indexch_ab
    ret["Exch-Ind20,u (A->B)"] = unc_indexch_ba
    ret["Exch-Ind20,u"] = unc_indexch_ba + unc_indexch_ab

    plist = [
        "Ind20,u (A<-B)", "Ind20,u (A->B)", "Ind20,u", "Exch-Ind20,u (A<-B)", "Exch-Ind20,u (A->B)", "Exch-Ind20,u"
    ]

    if do_print:
        for name in plist:
            # core.set_variable(name, ret[name])
            core.print_out(print_sapt_var(name, ret[name], short=True))
            core.print_out("\n")

    # Exch-Ind without S^2
    if Sinf:
        nocc_A = cache["Cocc_A"].shape[1]
        nocc_B = cache["Cocc_B"].shape[1]
        SAB = core.triplet(cache["Cocc_A"], cache["S"], cache["Cocc_B"], True, False, False)
        num_occ = nocc_A + nocc_B

        Sab = core.Matrix(num_occ, num_occ)
        Sab.np[:nocc_A, nocc_A:] = SAB.np
        Sab.np[nocc_A:, :nocc_A] = SAB.np.T
        Sab.np[np.diag_indices_from(Sab.np)] += 1
        Sab.power(-1.0, 1.e-14)

        Tmo_AA = core.Matrix.from_array(Sab.np[:nocc_A, :nocc_A])
        Tmo_BB = core.Matrix.from_array(Sab.np[nocc_A:, nocc_A:])
        Tmo_AB = core.Matrix.from_array(Sab.np[:nocc_A, nocc_A:])

        T_A = core.triplet(cache["Cocc_A"], Tmo_AA, cache["Cocc_A"], False, False, True)
        T_B = core.triplet(cache["Cocc_B"], Tmo_BB, cache["Cocc_B"], False, False, True)
        T_AB = core.triplet(cache["Cocc_A"], Tmo_AB, cache["Cocc_B"], False, False, True)

        sT_A = core.Matrix.chain_dot(cache["Cvir_A"],
                                     unc_x_B_MOA,
                                     Tmo_AA,
                                     cache["Cocc_A"],
                                     trans=[False, True, False, True])
        sT_B = core.Matrix.chain_dot(cache["Cvir_B"],
                                     unc_x_A_MOB,
                                     Tmo_BB,
                                     cache["Cocc_B"],
                                     trans=[False, True, False, True])
        sT_AB = core.Matrix.chain_dot(cache["Cvir_A"],
                                      unc_x_B_MOA,
                                      Tmo_AB,
                                      cache["Cocc_B"],
                                      trans=[False, True, False, True])
        sT_BA = core.Matrix.chain_dot(cache["Cvir_B"],
                                      unc_x_A_MOB,
                                      Tmo_AB,
                                      cache["Cocc_A"],
                                      trans=[False, True, True, True])

        jk.C_clear()

        jk.C_left_add(core.Matrix.chain_dot(cache["Cocc_A"], Tmo_AA))
        jk.C_right_add(cache["Cocc_A"])

        jk.C_left_add(core.Matrix.chain_dot(cache["Cocc_B"], Tmo_BB))
        jk.C_right_add(cache["Cocc_B"])

        jk.C_left_add(core.Matrix.chain_dot(cache["Cocc_A"], Tmo_AB))
        jk.C_right_add(cache["Cocc_B"])

        jk.compute()

        J_AA_inf, J_BB_inf, J_AB_inf = jk.J()
        K_AA_inf, K_BB_inf, K_AB_inf = jk.K()

        # A <- B
        EX_AA_inf = V_B.clone()
        EX_AA_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_AB, V_B, trans=[False, True, False]))
        EX_AA_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_B, V_B))
        EX_AA_inf.axpy(2.00, J_AB_inf)
        EX_AA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf, trans=[False, True, False]))
        EX_AA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_AB_inf))
        EX_AA_inf.axpy(2.00, J_BB_inf)
        EX_AA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_BB_inf, trans=[False, True, False]))
        EX_AA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_BB_inf))
        EX_AA_inf.axpy(-1.00, K_AB_inf.transpose())
        EX_AA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf, trans=[False, True, True]))
        EX_AA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_B, K_AB_inf, trans=[False, False, True]))
        EX_AA_inf.axpy(-1.00, K_BB_inf)
        EX_AA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_BB_inf, trans=[False, True, False]))
        EX_AA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_B, K_BB_inf))

        EX_AB_inf = V_A.clone()
        EX_AB_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_AB, V_A, trans=[False, True, False]))
        EX_AB_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_B, V_A))
        EX_AB_inf.axpy(2.00, J_AA_inf)
        EX_AB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AA_inf, trans=[False, True, False]))
        EX_AB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_AA_inf))
        EX_AB_inf.axpy(2.00, J_AB_inf)
        EX_AB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf, trans=[False, True, False]))
        EX_AB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_B, J_AB_inf))
        EX_AB_inf.axpy(-1.00, K_AA_inf)
        EX_AB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_AA_inf, trans=[False, True, False]))
        EX_AB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_B, K_AA_inf))
        EX_AB_inf.axpy(-1.00, K_AB_inf)
        EX_AB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf, trans=[False, True, False]))
        EX_AB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_B, K_AB_inf))

        # B <- A
        EX_BB_inf = V_A.clone()
        EX_BB_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_AB, V_A))
        EX_BB_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_A, V_A))
        EX_BB_inf.axpy(2.00, J_AB_inf)
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf))
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_AB_inf))
        EX_BB_inf.axpy(2.00, J_AA_inf)
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AA_inf))
        EX_BB_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_AA_inf))
        EX_BB_inf.axpy(-1.00, K_AB_inf)
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf))
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_A, K_AB_inf))
        EX_BB_inf.axpy(-1.00, K_AA_inf)
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_AA_inf))
        EX_BB_inf.axpy(1.00, core.Matrix.chain_dot(S, T_A, K_AA_inf))

        EX_BA_inf = V_B.clone()
        EX_BA_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_AB, V_B))
        EX_BA_inf.axpy(-1.00, core.Matrix.chain_dot(S, T_A, V_B))
        EX_BA_inf.axpy(2.00, J_BB_inf)
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_BB_inf))
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_BB_inf))
        EX_BA_inf.axpy(2.00, J_AB_inf)
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_AB, J_AB_inf))
        EX_BA_inf.axpy(-2.00, core.Matrix.chain_dot(S, T_A, J_AB_inf))
        EX_BA_inf.axpy(-1.00, K_BB_inf)
        EX_BA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_BB_inf))
        EX_BA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_A, K_BB_inf))
        EX_BA_inf.axpy(-1.00, K_AB_inf.transpose())
        EX_BA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_AB, K_AB_inf, trans=[False, False, True]))
        EX_BA_inf.axpy(1.00, core.Matrix.chain_dot(S, T_A, K_AB_inf, trans=[False, False, True]))

        unc_ind_ab_total = 2.0 * (sT_A.vector_dot(EX_AA_inf) + sT_AB.vector_dot(EX_AB_inf))
        unc_ind_ba_total = 2.0 * (sT_B.vector_dot(EX_BB_inf) + sT_BA.vector_dot(EX_BA_inf))
        unc_indexch_ab_inf = unc_ind_ab_total - unc_ind_ab
        unc_indexch_ba_inf = unc_ind_ba_total - unc_ind_ba

        ret["Exch-Ind20,u (A<-B) (S^inf)"] = unc_indexch_ab_inf
        ret["Exch-Ind20,u (A->B) (S^inf)"] = unc_indexch_ba_inf
        ret["Exch-Ind20,u (S^inf)"] = unc_indexch_ba_inf + unc_indexch_ab_inf

        if do_print:
            for name in plist[3:]:
                name = name + ' (S^inf)'

                core.print_out(print_sapt_var(name, ret[name], short=True))
                core.print_out("\n")

    # Do coupled
    if do_response:
        core.print_out("\n   => Coupled Induction <= \n\n")

        cphf_r_convergence = core.get_option("SAPT", "CPHF_R_CONVERGENCE")

        x_B_MOA, x_A_MOB = _sapt_cpscf_solve(
            cache,
            jk,
            w_B_MOA,
            w_A_MOB,
            20,
            cphf_r_convergence,
            sapt_jk_B=sapt_jk_B,
            diagnostics=diagnostics,
        )

        if diagnostics is not None:
            diagnostics["coupled_amplitudes"] = {
                "A<-B": x_B_MOA.np.copy(),
                "A->B": x_A_MOB.np.copy(),
            }

        ind_ab = 2.0 * x_B_MOA.vector_dot(w_B_MOA)
        ind_ba = 2.0 * x_A_MOB.vector_dot(w_A_MOB)
        indexch_ab = 2.0 * x_B_MOA.vector_dot(EX_A)
        indexch_ba = 2.0 * x_A_MOB.vector_dot(EX_B)

        ret["Ind20,r (A<-B)"] = ind_ab
        ret["Ind20,r (A->B)"] = ind_ba
        ret["Ind20,r"] = ind_ab + ind_ba
        ret["Exch-Ind20,r (A<-B)"] = indexch_ab
        ret["Exch-Ind20,r (A->B)"] = indexch_ba
        ret["Exch-Ind20,r"] = indexch_ba + indexch_ab

        if do_print:
            core.print_out("\n")
            for name in plist:
                name = name.replace(",u", ",r")

                # core.set_variable(name, ret[name])
                core.print_out(print_sapt_var(name, ret[name], short=True))
                core.print_out("\n")

        # Exch-Ind without S^2
        if Sinf:
            cT_A = core.Matrix.chain_dot(cache["Cvir_A"],
                                         x_B_MOA,
                                         Tmo_AA,
                                         cache["Cocc_A"],
                                         trans=[False, True, False, True])
            cT_B = core.Matrix.chain_dot(cache["Cvir_B"],
                                         x_A_MOB,
                                         Tmo_BB,
                                         cache["Cocc_B"],
                                         trans=[False, True, False, True])
            cT_AB = core.Matrix.chain_dot(cache["Cvir_A"],
                                          x_B_MOA,
                                          Tmo_AB,
                                          cache["Cocc_B"],
                                          trans=[False, True, False, True])
            cT_BA = core.Matrix.chain_dot(cache["Cvir_B"],
                                          x_A_MOB,
                                          Tmo_AB,
                                          cache["Cocc_A"],
                                          trans=[False, True, True, True])

            ind_ab_total = 2.0 * (cT_A.vector_dot(EX_AA_inf) + cT_AB.vector_dot(EX_AB_inf))
            ind_ba_total = 2.0 * (cT_B.vector_dot(EX_BB_inf) + cT_BA.vector_dot(EX_BA_inf))
            indexch_ab_inf = ind_ab_total - ind_ab
            indexch_ba_inf = ind_ba_total - ind_ba

            ret["Exch-Ind20,r (A<-B) (S^inf)"] = indexch_ab_inf
            ret["Exch-Ind20,r (A->B) (S^inf)"] = indexch_ba_inf
            ret["Exch-Ind20,r (S^inf)"] = indexch_ba_inf + indexch_ab_inf

            if do_print:
                for name in plist[3:]:
                    name = name.replace(",u", ",r") + ' (S^inf)'

                    core.print_out(print_sapt_var(name, ret[name], short=True))
                    core.print_out("\n")

    return ret


def _sapt_cpscf_solve(
    cache,
    jk,
    rhsA,
    rhsB,
    maxiter,
    conv,
    sapt_jk_B=None,
    diagnostics=None,
    standard_action="matrix_free",
):
    """
    Solve the SAPT CPHF (or CPKS) equations.
    """

    native = cache.get("native_jk", _native_jk(jk))
    native_A = native
    cache["wfn_A"].set_jk(native)
    if sapt_jk_B:
        native_B = _native_jk(sapt_jk_B)
        cache["wfn_B"].set_jk(native_B)
    else:
        native_B = native
        cache["wfn_B"].set_jk(native)

    # Make a preconditioner function
    P_A = core.Matrix(cache["eps_occ_A"].shape[0], cache["eps_vir_A"].shape[0])
    P_A.np[:] = (cache["eps_occ_A"].np.reshape(-1, 1) - cache["eps_vir_A"].np)

    P_B = core.Matrix(cache["eps_occ_B"].shape[0], cache["eps_vir_B"].shape[0])
    P_B.np[:] = (cache["eps_occ_B"].np.reshape(-1, 1) - cache["eps_vir_B"].np)

    def _as_matrix(value, shape):
        if isinstance(value, core.Matrix):
            mat = value.clone()
        else:
            mat = core.Matrix.from_array(np.ascontiguousarray(value, dtype=float))
        if mat.shape != shape:
            raise ValueError(f"trial response shape must be {shape}; got {mat.shape}.")
        return mat

    def _dse_action(label, trial):
        dse_cphf = cache.get(f"dse_cphf_{label}", None)
        if dse_cphf is not None and dse_cphf.is_active():
            return dse_cphf.hx_matrix(trial)
        return core.Matrix(trial.shape[0], trial.shape[1])

    def _psi4_standard_action(label, trial):
        if label == "A":
            shape = (cache["eps_occ_A"].shape[0], cache["eps_vir_A"].shape[0])
            trial = _as_matrix(trial, shape)
            return cache["wfn_A"].cphf_Hx([trial])[0]
        elif label == "B":
            shape = (cache["eps_occ_B"].shape[0], cache["eps_vir_B"].shape[0])
            trial = _as_matrix(trial, shape)
            return cache["wfn_B"].cphf_Hx([trial])[0]
        else:
            raise ValueError(f"response label must be 'A' or 'B'; got {label!r}.")

    def _matrix_free_standard_action(label, trial):
        if label == "A":
            Cocc = cache["Cocc_A"]
            Cvir = cache["Cvir_A"]
            eps_occ = cache["eps_occ_A"].np.reshape(-1, 1)
            eps_vir = cache["eps_vir_A"].np.reshape(1, -1)
            response_jk = native_A
        elif label == "B":
            Cocc = cache["Cocc_B"]
            Cvir = cache["Cvir_B"]
            eps_occ = cache["eps_occ_B"].np.reshape(-1, 1)
            eps_vir = cache["eps_vir_B"].np.reshape(1, -1)
            response_jk = native_B
        else:
            raise ValueError(f"response label must be 'A' or 'B'; got {label!r}.")

        trial = _as_matrix(trial, (Cocc.shape[1], Cvir.shape[1]))

        # Apply the ordinary ERI response block in the same occupied-virtual
        # convention as DSECPHF, then convert to Psi4 cg_solver's H x = rhs
        # convention: H_std X = (eps_occ - eps_vir) X - G[X].
        left = core.doublet(Cocc, trial, False, False)
        response_jk.C_clear()
        response_jk.C_left_add(left)
        response_jk.C_right_add(Cvir)
        response_jk.compute()
        J = response_jk.J()[0]
        K = response_jk.K()[0]
        K_T = K.clone()
        K_T.transpose_this()

        G = J.clone()
        G.scale(4.0)
        G.axpy(-1.0, K)
        G.axpy(-1.0, K_T)

        response = core.triplet(Cocc, G, Cvir, True, False, False)
        action = trial.clone()
        action.np[:] *= (eps_occ - eps_vir)
        action.axpy(-1.0, response)
        return action

    def _standard_action(label, trial):
        if standard_action == "matrix_free":
            return _matrix_free_standard_action(label, trial)
        if standard_action == "psi4":
            return _psi4_standard_action(label, trial)
        raise ValueError(
            "standard_action must be 'matrix_free' or 'psi4'; "
            f"got {standard_action!r}."
        )

    def _response_actions(label, trial):
        standard = _standard_action(label, trial)
        trial = _as_matrix(trial, standard.shape)

        dse = _dse_action(label, trial)
        total = standard.clone()
        total.axpy(-1.0, dse)
        return standard, dse, total

    def _standard_response_action(label, trial):
        return _response_actions(label, trial)[0].np.copy()

    def _psi4_standard_response_action(label, trial):
        return _psi4_standard_action(label, trial).np.copy()

    def _dse_response_action(label, trial):
        if label == "A":
            shape = (cache["eps_occ_A"].shape[0], cache["eps_vir_A"].shape[0])
        elif label == "B":
            shape = (cache["eps_occ_B"].shape[0], cache["eps_vir_B"].shape[0])
        else:
            raise ValueError(f"response label must be 'A' or 'B'; got {label!r}.")
        trial = _as_matrix(trial, shape)
        return _dse_action(label, trial).np.copy()

    def _total_response_action(label, trial):
        return _response_actions(label, trial)[2].np.copy()

    if diagnostics is not None:
        diagnostics["hessian_action_convention"] = {
            "standard": standard_action,
            "combined": f"{standard_action} standard - dse",
            "solver": "Psi4 cg_solver solves H x = rhs",
            "shape": "(nocc, nvir)",
        }
        diagnostics["hessian_actions"] = {
            "A<-B": {
                "psi4_standard": lambda X: _psi4_standard_response_action("A", X),
                "standard": lambda X: _standard_response_action("A", X),
                "dse": lambda X: _dse_response_action("A", X),
                "total": lambda X: _total_response_action("A", X),
            },
            "A->B": {
                "psi4_standard": lambda X: _psi4_standard_response_action("B", X),
                "standard": lambda X: _standard_response_action("B", X),
                "dse": lambda X: _dse_response_action("B", X),
                "total": lambda X: _total_response_action("B", X),
            },
        }

    # Preconditioner function
    def apply_precon(x_vec, act_mask):
        if act_mask[0]:
            pA = x_vec[0].clone()
            pA.apply_denominator(P_A)
        else:
            pA = False

        if act_mask[1]:
            pB = x_vec[1].clone()
            pB.apply_denominator(P_B)
        else:
            pB = False

        return [pA, pB]

    # Hx function
    def hessian_vec(x_vec, act_mask):
        if act_mask[0]:
            xA = _response_actions("A", x_vec[0])[2]
        else:
            xA = False

        if act_mask[1]:
            xB = _response_actions("B", x_vec[1])[2]
        else:
            xB = False

        return [xA, xB]

    # Manipulate the printing
    active_print = not output.is_quiet()
    sep_size = 51
    if active_print:
        core.print_out("   " + ("-" * sep_size) + "\n")
        core.print_out("   " + "SAPT Coupled Induction Solver".center(sep_size) + "\n")
        core.print_out("   " + ("-" * sep_size) + "\n")
        core.print_out("    Maxiter             = %11d\n" % maxiter)
        core.print_out("    Convergence         = %11.3E\n" % conv)
        core.print_out("   " + ("-" * sep_size) + "\n")

    tstart = time.time()
    if active_print:
        core.print_out("     %4s %12s     %12s     %9s\n" % ("Iter", "(A<-B)", "(B->A)", "Time [s]"))
        core.print_out("   " + ("-" * sep_size) + "\n")

    start_resid = [rhsA.sum_of_squares(), rhsB.sum_of_squares()]

    # print function
    def pfunc(niter, x_vec, r_vec):
        if niter == 0:
            niter = "Guess"
        else:
            niter = ("%5d" % niter)

        # Compute IndAB
        valA = (r_vec[0].sum_of_squares() / start_resid[0])**0.5
        if valA < conv:
            cA = "*"
        else:
            cA = " "

        # Compute IndBA
        valB = (r_vec[1].sum_of_squares() / start_resid[1])**0.5
        if valB < conv:
            cB = "*"
        else:
            cB = " "

        if active_print:
            core.print_out("    %5s %15.6e%1s %15.6e%1s %9d\n" % (niter, valA, cA, valB, cB, time.time() - tstart))
        return [valA, valB]

    # Compute the solver
    vecs, resid = solvers.cg_solver([rhsA, rhsB],
                                    hessian_vec,
                                    apply_precon,
                                    maxiter=maxiter,
                                    rcond=conv,
                                    printlvl=0,
                                    printer=pfunc)
    if active_print:
        core.print_out("   " + ("-" * sep_size) + "\n")

    if diagnostics is not None:
        diagnostics["cphf_residual_norms"] = {
            "A<-B": float(resid[0].sum_of_squares() ** 0.5),
            "A->B": float(resid[1].sum_of_squares() ** 0.5),
        }
        diagnostics["cphf_relative_residual_norms"] = {
            "A<-B": float((resid[0].sum_of_squares() / start_resid[0]) ** 0.5),
            "A->B": float((resid[1].sum_of_squares() / start_resid[1]) ** 0.5),
        }

    return vecs
