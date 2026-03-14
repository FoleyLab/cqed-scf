import psi4
from .scf import CQEDSCF 
from .gradients import CQEDRHFGradient
import time


class CQEDRHFCalculator:
    def __init__(self, lambda_vector, psi4_options, omega=0.1, charge=0, multiplicity=1, density_fitting=False, functional=None, debug=False):
        self.lambda_vector = lambda_vector
        self.psi4_options = psi4_options
        self.omega = omega
        self.density_fitting = density_fitting
        self.functional = functional
        self.debug = debug
        self.charge = charge
        self.multiplicity = multiplicity

        if multiplicity !=1:
            raise NotImplementedError("Only multiplicity=1 is currently supported.")


    def energy(self, geometry):
        self.geometry = geometry
        scf = CQEDSCF(
            geometry=geometry,
            lambda_vector=self.lambda_vector,
            psi4_options=self.psi4_options,
            omega=self.omega,
            density_fitting=self.density_fitting,
            functional=self.functional,
            debug=self.debug,

        )
        
        E, _ = scf.run()
        psi4.core.clean()
        return E

    def energy_and_gradient(self, geometry, canonical="psi4"):

        self.geometry = geometry

        scf = CQEDSCF(
            geometry,
            self.lambda_vector,
            self.psi4_options,
            self.omega,
            self.density_fitting,
            functional=self.functional,
            debug=self.debug,
        )

        E, data = scf.run()

        grad_engine = CQEDRHFGradient(
            self.lambda_vector,
            canonical=canonical,
            debug=self.debug
        )

        grad = grad_engine.compute(data)

        g = (self.omega / 2) ** 0.5 * data["d_exp"]

        psi4.core.clean()
        psi4.core.clean_options()

        return E, grad, g


