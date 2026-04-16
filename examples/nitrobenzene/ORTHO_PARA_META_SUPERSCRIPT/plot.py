import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --- Configuration ---
FILE_NAME = "nitrobromo_wb97x_d_field_scan_combined.txt"
DPI = 300  # High resolution for publication
AU_TO_KCAL = 627.509

def plot_energy_diffs():
    # 1. Load the data
    # Col 0: Theta, Col 1: Phi, Col 5: Para, Col 6: Ortho, Col 7: Meta
    data = np.genfromtxt(FILE_NAME, skip_header=2)
    
    theta = data[:, 0]
    phi = data[:, 1]
    e_para = data[:, 5]
    e_ortho = data[:, 6]
    e_meta = data[:, 7]

    # 2. Reshape into 2D grids
    # We know num_theta_vals = 21, num_phi_vals = 21
    num_t = len(np.unique(theta))
    num_p = len(np.unique(phi))
    
    T = theta.reshape(num_t, num_p)
    P = phi.reshape(num_t, num_p)
    
    # Calculate Differences (Convert Hartree to kcal/mol for better readability if desired)
    # 1 Hartree = 627.509 kcal/mol. Let's stay in Hartree but use scientific notation.
    diff_ortho_meta = (e_ortho - e_meta).reshape(num_t, num_p) * AU_TO_KCAL
    diff_para_meta = (e_para - e_meta).reshape(num_t, num_p) * AU_TO_KCAL

    # 3. Setup Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    # Set a divergent colormap (RdBu) centered at 0 or a sequential one
    # Use 'RdBu_r' to show relative stabilization/destabilization
    cmap = 'RdBu_r' 

    # Plot 1: Ortho - Meta
    im1 = ax1.pcolormesh(P, T, diff_ortho_meta, shading='gouraud', cmap=cmap)
    ax1.set_title(r'$\Delta E$ (Ortho $-$ Meta)', fontsize=14)
    ax1.set_xlabel(r'Azimuthal Angle $\phi$ (deg)', fontsize=12)
    ax1.set_ylabel(r'Polar Angle $\theta$ (deg)', fontsize=12)
    fig.colorbar(im1, ax=ax1, label='Energy Diff (Hartree)')

    # Plot 2: Para - Meta
    im2 = ax2.pcolormesh(P, T, diff_para_meta, shading='gouraud', cmap=cmap)
    ax2.set_title(r'$\Delta E$ (Para $-$ Meta)', fontsize=14)
    ax2.set_xlabel(r'Azimuthal Angle $\phi$ (deg)', fontsize=12)
    ax2.set_ylabel(r'Polar Angle $\theta$ (deg)', fontsize=12)
    fig.colorbar(im2, ax=ax2, label='Energy Diff (Hartree)')

    # Formatting tweaks
    for ax in [ax1, ax2]:
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([0, 45, 90, 135, 180])
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('Comparison of Orientational Energy Landscapes', fontsize=16, fontweight='bold')
    
    # 4. Save and Show
    plt.savefig("energy_difference_maps.png", dpi=DPI)
    print("Plot saved as energy_difference_maps.png")
    plt.show()

if __name__ == "__main__":
    plot_energy_diffs()
