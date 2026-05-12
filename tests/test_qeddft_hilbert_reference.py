import numpy as np
import psi4

from cqed_scf.scf import CQEDSCF


GEOM = """
0 1
         O            0.000000000000     0.000000000000    -0.068516219320    
         H            0.000000000000    -0.790689573744     0.543701060715
         H            0.000000000000     0.790689573744     0.543701060715
no_reorient
no_com
symmetry c1
"""


def test_qed_dft_reference_energy():

    psi4.core.clean()
    psi4.core.clean_options()

    options = {
        "basis": "sto-3g",
        "scf_type": "df",
        "points" : 3,
        "dft_spherical_points" : 590
    }

    omega = 0.07349864501573

    # field free
    calc0 = CQEDSCF(
        geometry=GEOM,
        lambda_vector=np.zeros(3),
        psi4_options=options,
        omega=omega,
        density_fitting=True,
        functional="PBE",
    )

    E0, _ = calc0.run()

    # cavity field
    calc = CQEDSCF(
        geometry=GEOM,
        lambda_vector=np.array([0.0, 0.0, 0.05]),
        psi4_options=options,
        omega=omega,
        density_fitting=True,
        functional="PBE",
    )

    E, _ = calc.run()
    assert abs(E0 - (-75.23445987752154)) < 1e-8
    assert abs(E - (-75.229826432848)) < 1e-8



