import psi4
import numpy as np
from .scf import CQEDSCF
from .gradients import CQEDRHFGradient


class CQEDRHFCalculator:
    def __init__(
        self,
        lambda_vector,
        psi4_options,
        omega=0.1,
        charge=0,
        multiplicity=1,
        density_fitting=False,
        functional=None,
        debug=False,
    ):
        self.lambda_vector = lambda_vector
        self.psi4_options = psi4_options
        self.omega = omega
        self.density_fitting = density_fitting
        self.functional = functional
        self.debug = debug
        self.charge = charge
        self.multiplicity = multiplicity

        if multiplicity != 1:
            raise NotImplementedError("Only multiplicity=1 is currently supported.")

        # --- Dispersion handling ---
        self._dispersion_map = {
            "wb97x-d": "wb97x",
        }

        self._has_dispersion = functional in self._dispersion_map
        # define the base functional (without dispersion) for the SCF calculation
        self._base_functional = (
            self._dispersion_map[functional] if self._has_dispersion else functional
        )

    # -------------------------
    # internal helpers
    # -------------------------
    def _compute_dispersion_energy(self, geometry):
        if not self._has_dispersion:
            return 0.0

        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_options(self.psi4_options)

        mol = psi4.geometry(geometry)

        E_w_disp = psi4.energy(self.functional, molecule=mol)
        E_no_disp = psi4.energy(self._base_functional, molecule=mol)

        return E_w_disp - E_no_disp

    def _compute_dispersion_gradient(self, geometry):
        if not self._has_dispersion:
            mol = psi4.geometry(geometry)
            return np.zeros((mol.natom(), 3))

        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_options(self.psi4_options)

        mol = psi4.geometry(geometry)

        grad_w = psi4.gradient(self.functional, molecule=mol)
        energy_w = psi4.core.variable('CURRENT ENERGY')
        grad_0 = psi4.gradient(self._base_functional, molecule=mol)
        energy_0 = psi4.core.variable('CURRENT ENERGY')

        disp_gradient = grad_w.np - grad_0.np
        dispersion_energy = energy_w - energy_0

        return dispersion_energy, disp_gradient



    # -------------------------
    # public API
    # -------------------------

    def energy(self, geometry):
        self.geometry = geometry

        # --- CQED SCF (base functional only) ---
        scf = CQEDSCF(
            geometry=geometry,
            lambda_vector=self.lambda_vector,
            psi4_options=self.psi4_options,
            omega=self.omega,
            density_fitting=self.density_fitting,
            functional=self._base_functional,
            debug=self.debug,
        )
        print("\nRunning CQED-SCF energy calculation...\n")
        print(f"Functional: {self._base_functional}")
        E_qed, _ = scf.run()

        # --- dispersion correction ---
        if self._has_dispersion:
            E_disp = self._compute_dispersion_energy(geometry)
            print(f"Dispersion correction energy: {E_disp:.12f} Eh")
            print(f"Total energy (CQED + dispersion): {E_qed + E_disp:.12f} Eh")
        else:
            E_disp = 0.0

        E_total = E_qed + E_disp

        if self.debug:
            print(f"E_QED  = {E_qed: .12f}")
            print(f"E_disp = {E_disp: .12f}")
            print(f"E_tot  = {E_total: .12f}")

        #psi4.core.clean()
        #psi4.core.clean_options()

        return E_total

    def energy_and_gradient(self, geometry, canonical="psi4"):
        self.geometry = geometry

        # --- CQED SCF ---
        scf = CQEDSCF(
            geometry,
            self.lambda_vector,
            self.psi4_options,
            self.omega,
            self.density_fitting,
            functional=self._base_functional,
            debug=self.debug,
        )
        print("\nRunning CQED-SCF energy calculation...\n")
        print(f"Functional: {self._base_functional}")
        E_qed, data = scf.run()

        # --- CQED gradient ---
        grad_engine = CQEDRHFGradient(
            self.lambda_vector,
            canonical=canonical,
            debug=self.debug,
        )

        grad_qed = grad_engine.compute(data)

        # --- dispersion correction ---
        if self._has_dispersion:
            #E_disp = self._compute_dispersion_energy(geometry)
            E_disp, grad_disp = self._compute_dispersion_gradient(geometry)
            print(f"Dispersion correction energy: {E_disp:.12f} Eh")
            print(f"Total energy (CQED + dispersion): {E_qed + E_disp:.12f} Eh")
            print("Dispersion correction gradient norm: {:.6e} Eh/Bohr".format(np.linalg.norm(grad_disp)))
        else:
            E_disp = 0.0
            grad_disp = np.zeros_like(grad_qed)

        # --- total ---
        E_total = E_qed + E_disp
        grad_total = grad_qed + grad_disp

        # photon observable (unchanged)
        g = (self.omega / 2) ** 0.5 * data["d_exp"]

        if self.debug:
            print(f"E_QED      = {E_qed: .12f}")
            print(f"E_disp     = {E_disp: .12f}")
            print(f"E_total    = {E_total: .12f}")
            print(f"|grad_disp|= {np.linalg.norm(grad_disp): .6e}")

        #psi4.core.clean()
        psi4.core.clean_options()

        return E_total, grad_total, g