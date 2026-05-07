import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_x = pd.read_csv("benzene_water_qed_wb97xd_cc_pVTZ_x_pol.csv")
df_y = pd.read_csv("benzene_water_qed_wb97xd_aug_cc_pVTZ_x_pol.csv")
# field free
df = pd.read_csv("benzene_water_qed_wb97xd_cc_pVTZ_x_pol.csv")

# Keep only successful calculations
df_x = df_x[df_x["E_int_raw_kcal_mol"].notna()]
df_y = df_y[df_y["E_int_raw_kcal_mol"].notna()]
df = df[df["E_int_raw_kcal_mol"].notna()]

# Sort by distance
df_x = df_x.sort_values("distance_A")
df_y = df_y.sort_values("distance_A")
df = df.sort_values("distance_A")

# Optional: shift minimum to zero (nice for comparison)
shift = False 
if shift:
    df_x["E_int_raw_kcal_mol"] -= df_x["E_int_raw_kcal_mol"].min()
    df_y["E_int_raw_kcal_mol"] -= df_y["E_int_raw_kcal_mol"].min()
    if "E_int_CP_kcal_mol" in df_x.columns:
        df_x["E_int_CP_kcal_mol"] -= df_x["E_int_CP_kcal_mol"].min()
    if "E_int_CP_kcal_mol" in df_y.columns:
        df_y["E_int_CP_kcal_mol"] -= df_y["E_int_CP_kcal_mol"].min()


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))

# plot field-free reference CP if available, else raw
plt.plot(
    df["distance_A"],
    df["E_int_CP_kcal_mol"] if "E_int_CP_kcal_mol" in df.columns else df["E_int_raw_kcal_mol"],
    marker="x",
    label="field-free",
)
# plot df_x (raw and CP if available)
plt.plot(
    df_x["distance_A"],
    df_x["E_int_CP_kcal_mol"] if "E_int_CP_kcal_mol" in df_x.columns else df_x["E_int_raw_kcal_mol"],
    marker="o", linestyle="--",
    label="x-field",
)



# plot df_y (raw and CP if available)
plt.plot(
    df_y["distance_A"],
    df_y["E_int_CP_kcal_mol"] if "E_int_CP_kcal_mol" in df_y.columns else df_y["E_int_raw_kcal_mol"],
    marker="^", linestyle="--",
    label="y-field",
)


# Zero reference line
plt.axhline(0.0, linestyle="--", linewidth=1)

# Highlight minimum (CP if available, else raw) for x-field
if "E_int_CP_kcal_mol" in df_x.columns:
    col = "E_int_CP_kcal_mol"
else:
    col = "E_int_raw_kcal_mol"

min_idx = df_x[col].idxmin()
d_eq = df_x.loc[min_idx, "distance_A"]
E_eq = df_x.loc[min_idx, col]

#plt.scatter(d_eq, E_eq, color="red", zorder=5)
#plt.text(d_eq, E_eq, f"  x-field min @ {d_eq:.2f} Å")

# Highlight minimum (CP if available, else raw) for y-field
if "E_int_CP_kcal_mol" in df_y.columns:
    col = "E_int_CP_kcal_mol"
else:
    col = "E_int_raw_kcal_mol"

min_idx = df_y[col].idxmin()
d_eq = df_y.loc[min_idx, "distance_A"]
E_eq = df_y.loc[min_idx, col]

#plt.scatter(d_eq, E_eq, color="green", zorder=5)
#plt.text(d_eq, E_eq, f"  y-field min @ {d_eq:.2f} Å")


# field free minimum
if "E_int_CP_kcal_mol" in df.columns:
    col = "E_int_CP_kcal_mol"
else:
    col = "E_int_raw_kcal_mol"  

min_idx = df[col].idxmin()
d_eq = df.loc[min_idx, "distance_A"]
E_eq = df.loc[min_idx, col] 
#plt.scatter(d_eq, E_eq, color="blue", zorder=5)
#plt.text(d_eq, E_eq, f"  field-free min @ {d_eq:.2f} Å")
# Labels
plt.xlabel("Separation distance (Å)")
plt.ylabel("Interaction energy (kcal/mol)")
plt.title("Benzene–Water Interaction Curve (QED-wB97X-D)")

plt.legend()
plt.tight_layout()

plt.savefig("benzene_water_interaction_curve_x_pol.png", dpi=300)
plt.show()