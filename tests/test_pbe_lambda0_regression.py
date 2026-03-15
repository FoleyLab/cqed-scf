import numpy as np
import psi4

from cqed_rhf.scf import CQEDSCF
from cqed_rhf.gradients import CQEDRHFGradient


WATER = """
0 1
O  0.000000000000   0.000000000000  -0.124721924329
H  0.000000000000  -1.429937284075   0.989898061465
H  0.000000000000   1.429937284075   0.989898061465
units bohr
no_reorient
no_com
symmetry c1
"""


def test_pbe_lambda0_energy_and_gradient_regression():
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_memory("2 GB")

    psi4_options = {
        "basis": "6-311G",
        "scf_type": "df",
        "e_convergence": 1.0e-10,
        "d_convergence": 1.0e-10,
    }

    lambda_vector = np.zeros(3)
    omega = 0.1

    # CQED-DFT at lambda = 0
    calc = CQEDSCF(
        geometry=WATER,
        lambda_vector=lambda_vector,
        psi4_options=psi4_options,
        omega=omega,
        density_fitting=True,
        functional="PBE",
        debug=False,
    )

    E_qed, scf_data = calc.run()

    grad_driver = CQEDRHFGradient(
        lambda_vector=lambda_vector,
        canonical="psi4",
        debug=False,
    )
    grad_qed = np.array(grad_driver.compute(scf_data))

    # Reference Psi4 PBE
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(psi4_options)

    mol = psi4.geometry(WATER)
    E_ref, wfn = psi4.energy("PBE", molecule=mol, return_wfn=True)
    grad_ref = np.array(psi4.gradient("PBE", molecule=mol, ref_wfn=wfn))

    # Tolerances
    energy_tol = 1.0e-8
    grad_rms_tol = 1.0e-6
    grad_max_tol = 1.0e-5

    dE = abs(E_qed - E_ref)
    grad_diff = grad_qed - grad_ref
    grad_rms = np.sqrt(np.mean(grad_diff**2))
    grad_max = np.max(np.abs(grad_diff))

    assert dE < energy_tol, f"Energy mismatch too large: {dE:.3e}"
    assert grad_rms < grad_rms_tol, f"Gradient RMS mismatch too large: {grad_rms:.3e}"
    assert grad_max < grad_max_tol, f"Gradient max mismatch too large: {grad_max:.3e}"
