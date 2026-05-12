import numpy as np

class NitrobenzeneOrientation:
    """
    Track orientation of a nitrobenzene molecule relative to an external field.
    """

    def __init__(self, symbols, coords_bohr, field_vector):
        """
        Parameters
        ----------
        symbols : list[str]
            Atomic symbols.
        coords_bohr : ndarray, shape (N,3)
            Initial coordinates (bohr), used to identify reference atoms.
        field_vector : ndarray, shape (3,)
            Electric field direction (need not be normalized).
        """
        self.symbols = symbols
        self.field_hat = np.asarray(field_vector, dtype=float)
        self.normalized_field_hat = self.field_hat / np.linalg.norm(self.field_hat)
        

        # Atom indices
        self.C_indices = [i for i, s in enumerate(symbols) if s == "C"]
        self.N_index = [i for i, s in enumerate(symbols) if s == "N"][0]

        # Identify bonded carbon ONCE
        self.CN_index = self._find_bonded_carbon(coords_bohr)

        # Pick 3 carbons to define ring plane
        self.ring_ref = self._choose_ring_triplet()

    def _find_bonded_carbon(self, coords_bohr):
        N = coords_bohr[self.N_index]
        C_coords = coords_bohr[self.C_indices]
        dists = np.linalg.norm(C_coords - N, axis=1)
        return self.C_indices[np.argmin(dists)]

    def _choose_ring_triplet(self):
        """
        Choose three non-collinear carbons in the benzene ring.
        """
        return self.C_indices[:3]

    def _compute_vectors(self, coords_bohr):
        """
        Compute molecular orientation vectors.
        """
        # ---- x_hat: C-N bond direction
        x_vec = coords_bohr[self.N_index] - coords_bohr[self.CN_index]
        x_hat = x_vec / np.linalg.norm(x_vec)

        # ---- z_hat: ring normal
        i, j, k = self.ring_ref
        v1 = coords_bohr[j] - coords_bohr[i]
        v2 = coords_bohr[k] - coords_bohr[i]
        z_vec = np.cross(v1, v2)
        z_hat = z_vec / np.linalg.norm(z_vec)

        # Enforce consistent orientation (optional but recommended)
        if z_hat[2] < 0:
            z_hat = -z_hat

        return x_hat, z_hat

    def observe(self, coords_bohr):
        """
        Compute orientation observables.

        Returns
        -------
        dict
            Dictionary of orientation observables.
        """
        x_hat, z_hat = self._compute_vectors(coords_bohr)

        cos_phi = np.dot(x_hat, self.normalized_field_hat)
        cos_theta = np.dot(z_hat, self.normalized_field_hat)

        # Numerical safety
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        phi_deg = np.degrees(np.arccos(cos_phi))
        theta_deg = np.degrees(np.arccos(cos_theta))

        return {
            "cos_phi": cos_phi,
            "cos_theta": cos_theta,
            "phi_deg": phi_deg,
            "theta_deg": theta_deg,
            "x_hat": x_hat.copy(),
            "z_hat": z_hat.copy(),
        }

