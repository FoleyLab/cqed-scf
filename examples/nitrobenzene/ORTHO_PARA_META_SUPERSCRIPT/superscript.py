import os
import shutil
import numpy as np
import psi4
from concurrent.futures import ProcessPoolExecutor
# Import your specific calculator
from cqed_rhf.calculator import CQEDRHFCalculator

# --- Configuration ---
NUM_CORES = 24
NUM_THETA = 42
NUM_PHI = 42
MEM_PER_CORE = "4 GB"

OUTPUT_FILE = "nitrobromo_wb97x_d_dense_field_scan_combined.txt"


# ----------------------------
# Molecular Geometries
# ----------------------------
GEOMS = {
    "para": """
1 1
         C           -0.511618296797     1.244386024531     0.732140048697
         C            0.856500593203     1.251903714531     0.717948218697
         C            1.524118723203     0.024661924531     0.713927788697
         H           -1.071804396797     2.172682314531     0.745925708697
         H            1.436128963203     2.163921874531     0.712099008697
         N            3.008539583203     0.046097104531     0.698823798697
         O            3.575097303203    -1.082768165469     0.699174708697
         O            3.542114363203     1.190870854531     0.689202018697
         C           -0.475464946797    -1.253402765469     0.742118638697
         H           -1.008574426797    -2.197377955469     0.762945788697
         C            0.892227703203    -1.221407805469     0.728065818697
         H            1.498048423203    -2.116244695469     0.729653208697
         C           -1.267906576797    -0.015841805469     0.712127418697
         H           -2.116520796797    -0.025293215469     1.403161498697
         BR          -2.114966986797    -0.034666925469    -1.121874081303
units angstrom
no_reorient
no_com
symmetry c1
""",
    "ortho": """
1 1
C           -1.804928163307     1.957993763262     0.703312273806
C           -0.379708783307     1.994122833262     0.698532703806
C            0.296125016693     0.817793533262     0.710271493806
C           -2.520286433307     0.755089873262     0.736288843806
H           -2.344947113307     2.899196893262     0.691895063806
H            0.158564066693     2.933869823262     0.699142733806
H           -3.601954283307     0.764862203262     0.746931053806
N            1.767881836693     0.820900013262     0.771891313806
O            2.315054046693    -0.296733496738     0.879853723806
O            2.340645916693     1.923356243262     0.711986073806
C           -1.829967733307    -0.442167236738     0.756258983806
H           -2.356763623307    -1.389967436738     0.789740873806
C           -0.361341153307    -0.491572936738     0.714148383806
H            0.119338216693    -1.238105076738     1.350400383806
BR          -0.151212663307    -1.224162306738    -1.170925976194
units angstrom
no_reorient
no_com
symmetry c1
""",
    "meta": """
1 1
         C           -0.929257263947     2.021527608578     0.744707683350
         C            0.476075706053     1.968481358578     0.682883583350
         C            1.153033166053     0.732862858578     0.67108907335
         C            0.486309286053    -0.455398891422     0.707696283350
         C           -1.646688783947     0.850023888578     0.786483593350
         H           -1.430027043947     2.980198348578     0.754644003350
         H            1.068570756053     2.878318968578     0.644324213350
         H            1.030908186053    -1.394630481422     0.699715393350
         H           -2.730391873947     0.862207158578     0.834726773350
         N            2.627601876053     0.732774608578     0.609077593350
         O            3.188360516053    -0.377859281422     0.588451963350
         O            3.186221516053     1.845711198578     0.586422223350
         C           -0.982368843947    -0.464026221422     0.760065283350
         H           -1.395507033947    -1.190671951422     1.465426213350
         BR          -1.494673453947    -1.187920261422    -1.064256256650
units angstrom
no_reorient
no_com
symmetry c1
"""
}

psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
}

# ----------------------------
# Worker Function
# ----------------------------
def run_point(task):
    theta, phi, worker_id = task
    
    # 1. Setup Isolated Scratch Directory
    scratch_path = f"/tmp/psi_work_{worker_id}"
    os.makedirs(scratch_path, exist_ok=True)
    
    # 2. Configure Psi4 - FIXED SYNTAX
    psi4.core.IOManager.shared_object().set_default_path(scratch_path)
    psi4.set_memory(MEM_PER_CORE)
    psi4.core.set_num_threads(1)
    # Redirect output to the scratch dir to avoid mixing up logs
    psi4.core.set_output_file(os.path.join(scratch_path, "output.dat"), False)

    # 3. Field Vector Generation
    theta_rad, phi_rad = np.radians(theta), np.radians(phi)
    field_vec = np.array([
        np.sin(theta_rad) * np.cos(phi_rad),
        np.sin(theta_rad) * np.sin(phi_rad),
        np.cos(theta_rad)
    ]) * 0.1

    results = {"theta": theta, "phi": phi, "vec": field_vec, "energies": {}}

    # 4. Calculate for each geometry
    for name, geom_string in GEOMS.items():
        try:
            # We re-initialize the calculator to ensure it picks up the local psi4 settings
            calc = CQEDRHFCalculator(
                lambda_vector=field_vec,
                psi4_options=psi4_options,
                omega=0.06615,
                charge=1,
                multiplicity=1,
                functional="wb97x-d",
            )
            results["energies"][name] = calc.energy(geom_string)
            psi4.core.clean() 
        except Exception as e:
            results["energies"][name] = np.nan
            print(f"Error at {theta}, {phi} for {name}: {e}")

    # OPTIONAL: Cleanup scratch files immediately to save space
    shutil.rmtree(scratch_path, ignore_errors=True)
    
    return results

# ----------------------------
# Main Execution Block
# ----------------------------
if __name__ == "__main__":
    theta_list = np.linspace(0, 180, NUM_THETA)
    phi_list = np.linspace(0, 360, NUM_PHI)
    
    tasks = []
    counter = 0
    for t in theta_list:
        for p in phi_list:
            tasks.append((t, p, counter))
            counter += 1

    print(f"Submitting {len(tasks)} points to {NUM_CORES} cores...")

    all_data = []
    # We use ProcessPoolExecutor to truly utilize the 24 cores of the M2 Ultra
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        all_data = list(executor.map(run_point, tasks))

    # Sort results
    all_data.sort(key=lambda x: (x["theta"], x["phi"]))

    # 5. Write Combined Output
    with open(OUTPUT_FILE, "w") as f:
        header = f"{'Theta':>10} {'Phi':>10} {'Ex':>12} {'Ey':>12} {'Ez':>12} {'Para_E':>20} {'Ortho_E':>20} {'Meta_E':>20}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")

        for d in all_data:
            t, p = d["theta"], d["phi"]
            vx, vy, vz = d["vec"]
            e_p = d["energies"].get("para", np.nan)
            e_o = d["energies"].get("ortho", np.nan)
            e_m = d["energies"].get("meta", np.nan)
            
            line = f"{t:10.3f} {p:10.3f} {vx:12.6f} {vy:12.6f} {vz:12.6f} {e_p:20.12f} {e_o:20.12f} {e_m:20.12f}\n"
            f.write(line)

    print(f"Scan complete. Data written to {OUTPUT_FILE}")
