import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("nitrobenzene_brplus_3d_scan_postprocessed.csv")

for z in sorted(df["z_scan_A"].unique()):
    sl = df[df["z_scan_A"] == z]

    pivot = sl.pivot(
        index="y_scan_A",
        columns="x_scan_A",
        values="relative_to_global_min_kcal_mol",
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(
        pivot.values,
        origin="lower",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max(),
        ],
        aspect="equal",
    )
    plt.colorbar(label="Relative energy / kcal mol$^{-1}$")
    plt.xlabel("x in ring plane / Angstrom")
    plt.ylabel("y in ring plane / Angstrom")
    plt.title(f"Br+ scan above nitrobenzene, z = {z:.2f} Angstrom")
    plt.tight_layout()
    plt.savefig(f"heatmap_z_{z:.2f}.png", dpi=300)
    plt.close()
