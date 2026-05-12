import numpy as np


class RotationalProjectionObserver:
    """
    Projects the full nuclear gradient onto rigid-body rotational
    modes corresponding to phi and theta orientation angles.
    """

    def __init__(self, orientation_tracker, masses):
        """
        Parameters
        ----------
        orientation_tracker : NitrobenzeneOrientation
            Provides x_hat, z_hat, normalized_field_hat.
        masses : ndarray, shape (N,)
            Atomic masses in atomic units.
        """
        self.orientation_tracker = orientation_tracker
        self.masses = masses

    def _center_of_mass(self, coords):
        total_mass = np.sum(self.masses)
        return np.sum(coords * self.masses[:, None], axis=0) / total_mass

    def _rotation_axis(self, v_hat, field_hat):
        axis = np.cross(v_hat, field_hat)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return None  # aligned, no defined rotation axis
        return axis / norm
    
    def _moment_of_inertia(self, axis, R):
        """
        Compute moment of inertia about given axis.
        """
        perp = np.cross(axis, R)
        return np.sum(self.masses[:, None] * perp**2)


    def observe(self, coords_bohr, grad):

        x_hat, z_hat = self.orientation_tracker._compute_vectors(coords_bohr)
        # get normalized field hat vector
        normalized_field_hat = self.orientation_tracker.normalized_field_hat

        com = self._center_of_mass(coords_bohr)
        R = coords_bohr - com

        forces = -grad
        torque_vec = np.sum(np.cross(R, forces), axis=0)

        results = {}
        results["torque_vector"] = torque_vec

        # ----- PHI -----
        axis_phi = self._rotation_axis(x_hat, normalized_field_hat)
        if axis_phi is not None:

            tau_phi = np.dot(torque_vec, axis_phi)

            dR_dphi = np.cross(axis_phi, R)
            dE_dphi = np.sum(grad * dR_dphi)

            I_phi = self._moment_of_inertia(axis_phi, R)

            alpha_phi = tau_phi / I_phi

            results["dE_dphi"] = dE_dphi
            results["axis_phi"] = axis_phi
            results["torque_phi"] = tau_phi
            results["I_phi"] = I_phi
            results["alpha_phi"] = alpha_phi

        else:
            results["dE_dphi"] = None
            results["axis_phi"] = None
            results["torque_phi"] = 0.0
            results["I_phi"] = 0.0
            results["alpha_phi"] = 0.0

        # ----- THETA -----
        axis_theta = self._rotation_axis(z_hat, normalized_field_hat)
        if axis_theta is not None:

            tau_theta = np.dot(torque_vec, axis_theta)

            I_theta = self._moment_of_inertia(axis_theta, R)

            dR_dtheta = np.cross(axis_theta, R)
            dE_dtheta = np.sum(grad * dR_dtheta)

            alpha_theta = tau_theta / I_theta

            results["dE_dtheta"] = dE_dtheta
            results["axis_theta"] = axis_theta
            results["torque_theta"] = tau_theta
            results["I_theta"] = I_theta
            results["alpha_theta"] = alpha_theta

        else:
            results["dE_dtheta"] = None
            results["axis_theta"] = None
            results["torque_theta"] = 0.0
            results["I_theta"] = 0.0
            results["alpha_theta"] = 0.0

        return results


