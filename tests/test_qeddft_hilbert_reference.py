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


# original water coordinates to run with lambda with equal
# polarization along x, y, and z
GEOM2_ORIGINAL = """
0 1
    O       0.000000000000    0.000000000000   -0.068516219320
    H       0.000000000000   -0.790689573744    0.543701060715
    H       0.000000000000    0.790689573744    0.543701060715
no_reorient
no_com
symmetry c1
"""

# rotated so that relative orientation to [0, 0, 0.05]
# is equivalent to orientation of GEOM2_ORIGINAL to [0.05/sqrt(3), 0.05/sqrt(3), 0.05/sqrt(3)]
GEOM2_REORIENT = """
0 1
    O       0.039557857668    0.039557857668   -0.039557857668
    H      -0.146813586014   -0.937503159758   -0.142598884484
    H      -0.480998321511    0.309691252233    0.770410792009
no_reorient
no_com
symmetry c1
"""



def test_qed_wb97x_water_geom_reoriented():
    """
    Test two different geometries with the same relative lambda and
    against the reference energy from hilbert using GEOM2_REORIENT 
    and lambda = [0, 0, 0.05]
    """

    psi4.core.clean()
    psi4.core.clean_options()

    options = {
        "basis": "def2-tzvppd",
        "scf_type": "df",
        "e_convergence" : 1e-10,
        "d_convergence" : 1e-10
    }

    omega = 0.07349864501573
    # hilbert energy from GEOM2_REORIENT and l = [0, 0, 0.05] using wb97x, def2-tzvppd
    expected_energy = -76.4365131429285

    # original geometry
    calc_original= CQEDSCF(
        geometry=GEOM2_ORIGINAL,
        lambda_vector=np.array([0.05/np.sqrt(3), 0.05/np.sqrt(3), 0.05/np.sqrt(3)]),
        psi4_options=options,
        omega=omega,
        density_fitting=True,
        functional="wb97x",
    )

    E_original, _ = calc_original.run()

    # cavity field
    calc_reorient = CQEDSCF(
        geometry=GEOM2_REORIENT,
        lambda_vector=np.array([0, 0, 0.05]),
        psi4_options=options,
        omega=omega,
        density_fitting=True,
        functional="wb97x",
    )

    E_reorient, _ = calc_reorient.run()

    assert np.isclose(E_reorient, E_original, atol=1e-8, rtol=1e-8)
    assert np.isclose(E_reorient, expected_energy, atol=1e-8, rtol=1e-8)



def test_qed_pbe_water_geom_reoriented():
    """
    Test two different geometries with the same relative lambda and
    against the reference energy from hilbert using GEOM2_REORIENT 
    and lambda = [0, 0, 0.05]
    """

    psi4.core.clean()
    psi4.core.clean_options()

    options = {
        "basis": "def2-tzvppd",
        "scf_type": "df",
        "e_convergence" : 1e-10,
        "d_convergence" : 1e-10
    }

    omega = 0.07349864501573
    # hilbert energy from GEOM2_REORIENT and l = [0, 0, 0.05] using pbe0, def2-tzvppd
    expected_energy = -76.37640896373374

    # original geometry
    calc_original= CQEDSCF(
        geometry=GEOM2_ORIGINAL,
        lambda_vector=np.array([0.05/np.sqrt(3), 0.05/np.sqrt(3), 0.05/np.sqrt(3)]),
        psi4_options=options,
        omega=omega,
        density_fitting=True,
        functional="pbe",
    )

    E_original, _ = calc_original.run()

    # cavity field
    calc_reorient = CQEDSCF(
        geometry=GEOM2_REORIENT,
        lambda_vector=np.array([0, 0, 0.05]),
        psi4_options=options,
        omega=omega,
        density_fitting=True,
        functional="pbe",
    )

    E_reorient, _ = calc_reorient.run()

    assert np.isclose(E_reorient, E_original, atol=1e-8, rtol=1e-8)
    assert np.isclose(E_reorient, expected_energy, atol=1e-8, rtol=1e-8)
