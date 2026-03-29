import psi4
import numpy as np
from scipy.optimize import minimize
import time
psi4.core.set_output_file("MD_test.out", append=False)

from .utils import (
    build_psi4_geometry,
    parse_psi4_geometry,
    write_xyz,
    ANGSTROM_TO_BOHR,
    AMU_TO_AU,
)


def initialize_md_from_geometry(geometry_string):
    """
    Initialize MD coordinates, symbols, and masses from a Psi4 geometry string.

    Returns
    -------
    coords_bohr : ndarray, shape (N,3)
    symbols : list[str]
    masses : ndarray, shape (N,)
        Atomic masses in electron mass units.
    mol : psi4.core.Molecule
    """
    mol = psi4.geometry(geometry_string)
    

    natom = mol.natom()
    symbols = [mol.symbol(i) for i in range(natom)]
    masses = np.array([mol.mass(i) for i in range(natom)]) * AMU_TO_AU
    coords_bohr = mol.geometry().to_array()


    return coords_bohr, symbols, masses, mol


def velocity_verlet_md(
    calculator,
    geometry=None,
    coords=None,
    symbols=None,
    velocities=None,
    dt=10.0,
    nsteps=10,
    canonical="psi4",
    observers=None,
    debug=False,
    freeze_atoms=None,   # NEW
):
    """
    Velocity-Verlet molecular dynamics.

    Parameters
    ----------
    calculator : CQEDRHFCalculator
    geometry : str, optional
        Psi4 geometry string (angstroms).
    coords : ndarray, shape (N,3), optional
        Cartesian coordinates in angstroms.
    symbols : list[str]
        Atomic symbols.
    velocities : ndarray, shape (N,3), optional
        Initial velocities (bohr / a.u. time).
    dt : float
        Time step in atomic units.
    nsteps : int
        Number of MD steps.
    canonical : {'psi4', 'exact'}
        Gradient backend.
    observers : list, optional
        Objects with an observe(coords_bohr) method.
    debug : bool
        Print energies each step.

    Returns
    -------
    traj : list of dict
        Trajectory data.
    observer_data : dict
        Mapping observer -> list of observations.
    """
    t0 = time.time()
    print(f"Starting MD simulation for {nsteps} steps with dt={dt:.2f} a.u. (started at {t0:.2f} s)"    )
    if observers is None:
        observers = []

    observer_data = {obs: [] for obs in observers}
    t1 = time.time()
    print(f"Observer Initialization took {t1 - t0:.2f} seconds.")

    # -------------------------
    # Initialize system
    # -------------------------
    if geometry is not None:
        print("Initializing from geometry string...")
        print(geometry)
        coords_bohr, symbols, masses, mol = initialize_md_from_geometry(geometry)
        calculator.charge = mol.molecular_charge()
        calculator.multiplicity = mol.multiplicity()

    elif coords is not None and symbols is not None:
        print("Initializing from coords and symbols...")
        print("charge =", calculator.charge)
        print("multiplicity =", calculator.multiplicity)
        
        coords = np.asarray(coords)
        coords_bohr = coords * ANGSTROM_TO_BOHR

        # Build temporary molecule to get masses
        geom = build_psi4_geometry(coords, symbols, units="angstrom", charge=calculator.charge, multiplicity=calculator.multiplicity)
        mol = psi4.geometry(geom)

        masses = np.array([mol.mass(i) for i in range(mol.natom())]) * AMU_TO_AU

    else:
        raise ValueError("Provide either geometry or coords+symbols.")

    natom = len(symbols)
    t2 = time.time()
    print(f"Geometry Initialization took {t2 - t1:.2f} seconds.")

    # -------------------------
    # Velocities (bohr / a.u.)
    # -------------------------
    if velocities is None:
        velocities = np.zeros((natom, 3))
    else:
        velocities = np.asarray(velocities)

    # -------------------------
    # Freeze constraints    
    # -------------------------

    if freeze_atoms is None:
        freeze_atoms = []

    freeze_atoms = np.array(freeze_atoms, dtype=int)

    # -------------------------
    # Initial forces
    # -------------------------
    coords_angstrom = coords_bohr / ANGSTROM_TO_BOHR
    geom = build_psi4_geometry(coords_angstrom, symbols, units="angstrom", charge=calculator.charge, multiplicity=calculator.multiplicity)
    t3 = time.time()
    print(f"Initial Geometry Build took {t3 - t2:.2f} seconds")   
    E, grad, g = calculator.energy_and_gradient(
        geom, canonical=canonical
    )
    t4 = time.time()
    print(f"Initial Energy and Gradient Calculation took {t4 - t3:.2f} seconds")

    forces = -grad  # Hartree / bohr

    # Apply freeze constraints
    coords_bohr, velocities, forces = apply_freeze_constraints(
        coords_bohr, velocities, forces, freeze_atoms
        )

    traj = []

    # -------------------------
    # MD loop
    # -------------------------
    for step in range(nsteps):
        t5 = time.time()
        print(f"Starting step {step} at {t5:.2f} seconds"   )

        # Half-step velocity update
        velocities += 0.5 * dt * forces / masses[:, None]

        # enforce frozen atoms
        velocities[freeze_atoms, :] = 0.0

        # Position update (bohr)
        coords_bohr += dt * velocities

        # Rebuild geometry
        coords_angstrom = coords_bohr / ANGSTROM_TO_BOHR
        geom = build_psi4_geometry(coords_angstrom, symbols, units="angstrom", charge=calculator.charge, multiplicity=calculator.multiplicity)
        t6 = time.time()
        print(f"Geometry Build for step {step} took {t6 - t5:.2f} seconds")
        # New forces
        E, grad, g = calculator.energy_and_gradient(
            geom, canonical=canonical
        )
        t7 = time.time()
        print(f"Energy and Gradient Calculation for step {step} took {t7 - t6:.2f} seconds")
        forces = -grad

        coords_bohr, velocities, forces = apply_freeze_constraints(
            coords_bohr, velocities, forces, freeze_atoms
            )

        # Final half-step velocity update
        velocities += 0.5 * dt * forces / masses[:, None]

        # enforce frozen atoms again
        velocities[freeze_atoms, :] = 0.0

        # ---- Observers ----
        for obs in observers:
            observer_data[obs].append(
                obs.observe(coords_bohr)
            )

        t8 = time.time()
        print(f"Step {step} completed in {t8 - t5:.2f} seconds")

        # Store step
        traj.append(
            dict(
                step=step,
                energy=E,
                coords=coords_angstrom.copy(),
                coords_bohr=coords_bohr.copy(),
                velocities=velocities.copy(),
                forces=forces.copy(),
                coupling=g,
            )
        )
        t9 = time.time()
        print(f"Data storage for step {step} took {t9 - t8:.2f} seconds")
        if debug:
            print(
                f"Step {step:4d} | "
                f"E = {E: .8f} Ha | "
                f"|F| = {np.linalg.norm(forces):.4e}"
            )

    return traj, observer_data

def apply_freeze_constraints(coords, velocities, gradients, freeze_atoms):
    """ Apply freeze constraints by zeroing out forces and velocities on specified atoms."""
    coords = coords.copy()
    velocities = velocities.copy()
    gradients = gradients.copy()

    # zero force/gradient on frozen atoms
    gradients[freeze_atoms, :] = 0.0

    # zero velocity on frozen atoms
    velocities[freeze_atoms, :] = 0.0

    return coords, velocities, gradients


def bfgs_optimize(
    calculator,
    geometry,
    canonical="psi4",
    gtol=1e-5,
    maxiter=5,
    observers=None,
    debug=False,
):
    """
    Geometry optimization using BFGS.

    Parameters
    ----------
    calculator : CQEDRHFCalculator
    geometry : str
        Psi4 geometry string (angstrom).
    canonical : {'psi4', 'exact'}
        Gradient backend.
    gtol : float
        Gradient norm tolerance (Ha/bohr).
    maxiter : int
        Maximum iterations.
    observers : list, optional
        Objects with an observe(coords_bohr) method.
    debug : bool
        Print progress.

    Returns
    -------
    result : OptimizeResult
        SciPy optimization result.
    observer_data : dict
        Mapping observer -> list of observations.
    """

    if observers is None:
        observers = []

    observer_data = {obs: [] for obs in observers}

    # Parse initial geometry
    mol = psi4.geometry(geometry)
    symbols = [mol.symbol(i) for i in range(mol.natom())]
    x0_bohr = mol.geometry().to_array()

    if mol.molecular_charge() != calculator.charge:
        raise ValueError("Charge mismatch between geometry and calculator")

    if mol.multiplicity() != calculator.multiplicity:
        raise ValueError("Multiplicity mismatch between geometry and calculator")


    def objective(x_flat):
        coords_bohr = x_flat.reshape(-1, 3)
        coords_angstrom = coords_bohr / ANGSTROM_TO_BOHR

        geom = build_psi4_geometry(
            coords_angstrom, symbols, units="angstrom", charge=calculator.charge, multiplicity=calculator.multiplicity
        )

        E, grad, g = calculator.energy_and_gradient(
            geom, canonical=canonical
        )

        if debug:
            print("Current Grad is")
            print(grad)
            print("Norm of grad is")
            print(np.linalg.norm(grad))
            write_xyz(
                "opt_traj.xyz",
                symbols,
                coords_angstrom,
                comment=f"E={E:.10f} |grad|={np.linalg.norm(grad):.3e}",
                mode="a",
            )

        return E, grad.reshape(-1)

    def callback(x_flat):
        coords_bohr = x_flat.reshape(-1, 3)
        for obs in observers:
            observer_data[obs].append(
                obs.observe(coords_bohr)
            )

    print("Starting BFGS optimization...")
    print(f"Going to update for {maxiter} iterations or until gradient norm < {gtol:.2e} Ha/bohr")
    result = minimize(
        fun=lambda x: objective(x)[0],
        x0=x0_bohr.reshape(-1),
        jac=lambda x: objective(x)[1],
        method="BFGS",
        callback=callback,
        options=dict(
            gtol=gtol,
            maxiter=maxiter,
            disp=debug,
        ),
    )

    return result, observer_data

