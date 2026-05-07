import pandas as pd

hartree_to_kcal_mol = 627.509474

df = pd.read_csv("nitrobenzene_brplus_3d_scan_parallel.csv") #nitrobenzene_brplus_3d_scan.csv")
ok = df[df["status"] == "ok"].copy()

emin = ok["energy_Ha"].min()
ok["relative_to_global_min_kcal_mol"] = (ok["energy_Ha"] - emin) * hartree_to_kcal_mol

ok_sorted = ok.sort_values("energy_Ha")

print("Lowest-energy scan points:")
print(
    ok_sorted[
        [
            "x_scan_A",
            "y_scan_A",
            "z_scan_A",
            "Br_x_A",
            "Br_y_A",
            "Br_z_A",
            "min_Br_heavy_atom_distance_A",
            "energy_Ha",
            "relative_to_global_min_kcal_mol",
        ]
    ].head(20)
)

ok.to_csv("nitrobenzene_brplus_3d_scan_postprocessed.csv", index=False)
