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


def test_pbe_lambda0_energy_and_gradient():

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_memory("2 GB")
    psi4.core.set_output_file("pytest.out", False)

    options = {
        "basis": "6-311G",
        "scf_type": "df",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
    }

    lambda_vec = np.zeros(3)

    calc = CQEDSCF(
        geometry=WATER,
        lambda_vector=lambda_vec,
        psi4_options=options,
        omega=0.1,
        density_fitting=True,
        functional="PBE",
    )

    E_qed, data = calc.run()

    grad_engine = CQEDRHFGradient(lambda_vec)
    grad_qed = np.array(grad_engine.compute(data))

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(options)

    mol = psi4.geometry(WATER)

    E_ref, wfn = psi4.energy("PBE", molecule=mol, return_wfn=True)
    grad_ref = np.array(psi4.gradient("PBE", molecule=mol, ref_wfn=wfn))

    dE = abs(E_qed - E_ref)
    grad_diff = grad_qed - grad_ref

    grad_rms = np.sqrt(np.mean(grad_diff**2))
    grad_max = np.max(np.abs(grad_diff))

    assert dE < 1e-8
    assert grad_rms < 1e-6
    assert grad_max < 1e-5
