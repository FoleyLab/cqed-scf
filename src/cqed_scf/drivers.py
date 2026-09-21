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
    calculator : CQEDCalculator
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

        # Check rather than overwrite, matching bfgs_optimize below.  Silently
        # adopting the geometry's values would contradict that check, and would
        # also push an open-shell multiplicity into a restricted config, which
        # CQEDConfig now rejects -- surfacing as a ValueError from a property
        # setter in the middle of a driver.
        if mol.molecular_charge() != calculator.charge:
            raise ValueError(
                f"Charge mismatch between geometry ({mol.molecular_charge()}) "
                f"and calculator ({calculator.charge})"
            )

        if mol.multiplicity() != calculator.multiplicity:
            raise ValueError(
                f"Multiplicity mismatch between geometry ({mol.multiplicity()}) "
                f"and calculator ({calculator.multiplicity})"
            )

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
        print(f"Updated Psi4 geometry for step {step}:\n{geom}")
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


def project_cartesian_gradient_remove_translation_rotation(
    coords_bohr,
    grad,
    masses,
    return_diagnostics=False,
):
    """
    Project a Cartesian nuclear gradient away from rigid-body modes.

    This is a Cartesian projected-gradient operation for geometry
    optimization: it removes components along mass-weighted translation and
    rigid-body rotation modes before the gradient is passed to the optimizer.
    It is not an internal-coordinate optimization; it suppresses net translational
    and rotational components of the Cartesian gradient.

    Parameters
    ----------
    coords_bohr : ndarray, shape (N, 3)
        Cartesian coordinates in bohr.
    grad : ndarray, shape (N, 3)
        Cartesian gradient in Hartree/bohr.
    masses : ndarray, shape (N,)
        Atomic masses in atomic units/electron masses.
    return_diagnostics : bool
        If True, return a diagnostics dictionary with force/torque residuals.

    Returns
    -------
    grad_proj : ndarray, shape (N, 3)
        Projected Cartesian gradient in Hartree/bohr.
    diagnostics : dict, optional
        Projection diagnostics, returned only when requested.
    """
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    grad = np.asarray(grad, dtype=float)
    masses = np.asarray(masses, dtype=float)

    if coords_bohr.ndim != 2 or coords_bohr.shape[1] != 3:
        raise ValueError("coords_bohr must have shape (N, 3)")

    if grad.shape != coords_bohr.shape:
        raise ValueError("grad must have the same shape as coords_bohr")

    if masses.ndim != 1 or masses.shape[0] != coords_bohr.shape[0]:
        raise ValueError("masses must have shape (N,), matching coords_bohr")

    if not np.all(np.isfinite(coords_bohr)):
        raise ValueError("coords_bohr must contain only finite values")

    if not np.all(np.isfinite(grad)):
        raise ValueError("grad must contain only finite values")

    if not np.all(np.isfinite(masses)):
        raise ValueError("masses must contain only finite values")

    if np.any(masses <= 0.0):
        raise ValueError("masses must be strictly positive")

    projectors = build_reference_translation_rotation_projectors(
        coords_bohr,
        masses,
    )
    x = projectors["centered_coords_bohr"]
    sqrtm = np.sqrt(masses)

    # A gradient is a covector, so its Cartesian projector is
    # M^(1/2) P_mw M^(-1/2).  This is the transpose of the projector that
    # must be applied to a Cartesian displacement; see the builder below.
    grad_flat = grad.reshape(-1)
    grad_proj = (projectors["gradient_projector"] @ grad_flat).reshape(grad.shape)

    if not return_diagnostics:
        return grad_proj

    forces_raw = -grad
    forces_proj = -grad_proj
    diagnostics = {
        "forces_raw": forces_raw,
        "forces_proj": forces_proj,
        "net_force_raw": np.sum(forces_raw, axis=0),
        "net_force_proj": np.sum(forces_proj, axis=0),
        "torque_raw": np.sum(np.cross(x, forces_raw), axis=0),
        "torque_proj": np.sum(np.cross(x, forces_proj), axis=0),
        "raw_grad_norm": np.linalg.norm(grad),
        "proj_grad_norm": np.linalg.norm(grad_proj),
        "raw_force_norm": np.linalg.norm(forces_raw),
        "proj_force_norm": np.linalg.norm(forces_proj),
        "rank": projectors["rank"],
        "singular_values": projectors["singular_values"],
    }

    return grad_proj, diagnostics


def build_reference_translation_rotation_projectors(
    coords_bohr,
    masses,
    svd_tol=1e-10,
):
    """Build fixed-reference projectors for gradients and BFGS steps.

    The orthogonal projector is first constructed in mass-weighted Cartesian
    coordinates.  Transforming it back to ordinary Cartesian coordinates
    produces two different matrices:

    ``gradient_projector = M^(1/2) P_mw M^(-1/2)``
        Projects a Cartesian energy gradient (a covector).

    ``step_projector = M^(-1/2) P_mw M^(1/2)``
        Projects a Cartesian coordinate displacement (a vector).

    These matrices are transposes of one another.  Using the gradient
    projector on a BFGS displacement is only correct when all masses are the
    same.  Keeping both explicitly is what lets :func:`bfgs_optimize` apply
    the constrained inverse-Hessian action

        step = -step_projector @ B_inv @ gradient_projector @ gradient.

    The modes are built from one reference geometry and remain fixed during
    the optimization.  Consequently accepted coordinates obey the affine
    Eckart-style condition

        Q.T @ M^(1/2) @ (x - x_reference) = 0,

    rather than merely having the instantaneous torque removed from each
    gradient.
    """
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    masses = np.asarray(masses, dtype=float)

    if coords_bohr.ndim != 2 or coords_bohr.shape[1] != 3:
        raise ValueError("coords_bohr must have shape (N, 3)")
    if masses.ndim != 1 or masses.shape[0] != coords_bohr.shape[0]:
        raise ValueError("masses must have shape (N,), matching coords_bohr")
    if not np.all(np.isfinite(coords_bohr)):
        raise ValueError("coords_bohr must contain only finite values")
    if not np.all(np.isfinite(masses)):
        raise ValueError("masses must contain only finite values")
    if np.any(masses <= 0.0):
        raise ValueError("masses must be strictly positive")
    if not np.isfinite(svd_tol) or svd_tol <= 0.0:
        raise ValueError("svd_tol must be a positive finite number")

    total_mass = np.sum(masses)
    natom = coords_bohr.shape[0]
    com = np.sum(coords_bohr * masses[:, None], axis=0) / total_mass
    centered = coords_bohr - com
    sqrtm = np.sqrt(masses)

    modes = []
    for axis in range(3):
        mode = np.zeros((natom, 3))
        mode[:, axis] = sqrtm
        modes.append(mode.reshape(-1))

    for axis in range(3):
        axis_vec = np.eye(3)[axis]
        mode = np.cross(axis_vec[None, :], centered)
        mode *= sqrtm[:, None]
        modes.append(mode.reshape(-1))

    rigid_modes = np.column_stack(modes)
    U, singular_values, _ = np.linalg.svd(rigid_modes, full_matrices=False)
    keep = singular_values > svd_tol
    Q = U[:, keep]

    # P_mw is symmetric and orthogonal in mass-weighted coordinates.
    dimension = 3 * natom
    P_mw = np.eye(dimension) - Q @ Q.T
    sqrtm_flat = np.repeat(sqrtm, 3)

    # Avoid materializing diagonal mass matrices: row multiplication applies
    # the left diagonal matrix and column division/multiplication applies the
    # right one.
    gradient_projector = (
        sqrtm_flat[:, None] * P_mw / sqrtm_flat[None, :]
    )
    step_projector = (
        P_mw * sqrtm_flat[None, :] / sqrtm_flat[:, None]
    )

    return {
        "mass_weighted_projector": P_mw,
        "gradient_projector": gradient_projector,
        "step_projector": step_projector,
        "rigid_basis_mass_weighted": Q,
        "rank": int(Q.shape[1]),
        "singular_values": singular_values,
        "center_of_mass_bohr": com,
        "centered_coords_bohr": centered,
    }


def bfgs_optimize(
    calculator,
    geometry,
    canonical="psi4",
    gtol=1e-5,
    maxiter=5,
    observers=None,
    debug=False,
    project_tr_rot=False,
    projection_debug=False,
):
    """
    Geometry optimization using BFGS with cached energy/gradient evaluations.

    Parameters
    ----------
    calculator : CQEDCalculator
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
    project_tr_rot : bool
        If True, constrain both gradients and BFGS coordinate steps to the
        non-rigid subspace defined by the initial geometry.  The optimizer sees
        the projected gradient, while every trial displacement is passed
        through the transpose (vector) projector before a geometry is built.
        Thus the effective inverse-Hessian action is
        ``P_step @ B_inv @ P_grad`` and cannot reintroduce a rigid rotation.
    projection_debug : bool
        If True and project_tr_rot is enabled, print compact projection
        diagnostics during debug output.

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

    # -------------------------
    # Parse initial geometry
    # -------------------------
    mol = psi4.geometry(geometry)
    symbols = [mol.symbol(i) for i in range(mol.natom())]
    x0_bohr = mol.geometry().to_array()
    masses = np.array([mol.mass(i) for i in range(mol.natom())]) * AMU_TO_AU

    if mol.molecular_charge() != calculator.charge:
        raise ValueError("Charge mismatch between geometry and calculator")

    if mol.multiplicity() != calculator.multiplicity:
        raise ValueError("Multiplicity mismatch between geometry and calculator")

    # ------------------------------------------------------------------
    # Fixed-reference constrained coordinates
    # ------------------------------------------------------------------
    # SciPy's ordinary BFGS is unconstrained.  Supplying only a projected
    # gradient is insufficient because its approximate inverse Hessian can mix
    # the retained internal components back into translations and rotations.
    #
    # For project_tr_rot=True, SciPy therefore optimizes a Cartesian-sized
    # displacement variable ``u`` and the physical coordinates are
    #
    #     x(u) = x_reference + P_step @ u.
    #
    # The chain-rule gradient with respect to u is
    #
    #     dE/du = P_step.T @ dE/dx = P_grad @ dE/dx.
    #
    # Consequently a BFGS trial step is mapped to physical coordinates as
    #
    #     dx = -P_step @ B_inv @ P_grad @ g,
    #
    # projecting both sides of B_inv.  The redundant rigid components of u do
    # not affect x or E.  This retains the historical Cartesian gradient units
    # and gtol interpretation while enforcing a fixed affine Eckart-style
    # constraint relative to the initial geometry.
    x_reference_flat = x0_bohr.reshape(-1).copy()
    reference_projectors = None
    if project_tr_rot:
        reference_projectors = build_reference_translation_rotation_projectors(
            x0_bohr,
            masses,
        )
        gradient_projector = reference_projectors["gradient_projector"]
        step_projector = reference_projectors["step_projector"]
        optimizer_x0 = np.zeros_like(x_reference_flat)

        def optimizer_to_cartesian(optimizer_x):
            return x_reference_flat + step_projector @ optimizer_x
    else:
        optimizer_x0 = x_reference_flat.copy()

        def optimizer_to_cartesian(optimizer_x):
            return np.asarray(optimizer_x, dtype=float)

    # -------------------------
    # Cache for last evaluation
    # -------------------------
    last_x = None
    last_E = None
    last_grad = None

    # -------------------------
    # Core evaluator (single SCF per geometry)
    # -------------------------
    def evaluate(optimizer_x):
        coords_flat = optimizer_to_cartesian(optimizer_x)
        coords_bohr = coords_flat.reshape(-1, 3)
        coords_angstrom = coords_bohr / ANGSTROM_TO_BOHR

        geom = build_psi4_geometry(
            coords_angstrom,
            symbols,
            units="angstrom",
            charge=calculator.charge,
            multiplicity=calculator.multiplicity,
        )
        if project_tr_rot:
            # build_psi4_geometry already disables automatic reorientation.
            # The fixed-reference constraint also fixes the center of mass, so
            # prevent Psi4 from applying a separate coordinate translation
            # before the electronic-structure calculation.  Leave the legacy
            # unconstrained path unchanged.
            geom += "\nno_com"

        E, grad, g = calculator.energy_and_gradient(
            geom, canonical=canonical
        )

        grad_raw = grad.copy()
        proj_info = None

        if project_tr_rot:
            grad_raw_flat = grad_raw.reshape(-1)
            grad = (gradient_projector @ grad_raw_flat).reshape(grad_raw.shape)

            # The fixed projector is defined at x_reference, so its rotational
            # residual is measured against the reference-centered coordinates.
            # We also report the ordinary torque about the current COM: unlike
            # the reference residual it need not vanish after a large internal
            # distortion, because the two quantities represent different frame
            # conventions.
            reference_centered = reference_projectors["centered_coords_bohr"]
            current_com = np.sum(coords_bohr * masses[:, None], axis=0) / np.sum(masses)
            current_centered = coords_bohr - current_com
            forces_raw = -grad_raw
            forces_proj = -grad
            sqrtm_flat = np.repeat(np.sqrt(masses), 3)
            constraint_residual = (
                reference_projectors["rigid_basis_mass_weighted"].T
                @ (sqrtm_flat * (coords_flat - x_reference_flat))
            )
            proj_info = {
                "forces_raw": forces_raw,
                "forces_proj": forces_proj,
                "net_force_raw": np.sum(forces_raw, axis=0),
                "net_force_proj": np.sum(forces_proj, axis=0),
                "torque_raw": np.sum(np.cross(reference_centered, forces_raw), axis=0),
                "torque_proj": np.sum(np.cross(reference_centered, forces_proj), axis=0),
                "current_frame_torque_raw": np.sum(
                    np.cross(current_centered, forces_raw), axis=0
                ),
                "current_frame_torque_proj": np.sum(
                    np.cross(current_centered, forces_proj), axis=0
                ),
                "raw_grad_norm": np.linalg.norm(grad_raw),
                "proj_grad_norm": np.linalg.norm(grad),
                "raw_force_norm": np.linalg.norm(forces_raw),
                "proj_force_norm": np.linalg.norm(forces_proj),
                "rank": reference_projectors["rank"],
                "singular_values": reference_projectors["singular_values"],
                "constraint_residual": constraint_residual,
                "constraint_residual_norm": np.linalg.norm(constraint_residual),
            }

        if debug:
            print("Current Grad is")
            print(grad)

            if project_tr_rot:
                print("Norm of projected grad is")
                print(np.linalg.norm(grad))
                print("Norm of raw grad is")
                print(np.linalg.norm(grad_raw))
            else:
                print("Norm of grad is")
                print(np.linalg.norm(grad))

            if projection_debug and project_tr_rot:
                print("Projection diagnostics:")
                print(f"  raw |grad|       = {proj_info['raw_grad_norm']:.6e}")
                print(f"  projected |grad| = {proj_info['proj_grad_norm']:.6e}")
                print(f"  raw |net force|  = {np.linalg.norm(proj_info['net_force_raw']):.6e}")
                print(f"  proj |net force| = {np.linalg.norm(proj_info['net_force_proj']):.6e}")
                print(f"  raw |torque|     = {np.linalg.norm(proj_info['torque_raw']):.6e}")
                print(f"  proj |torque|    = {np.linalg.norm(proj_info['torque_proj']):.6e}")
                print(f"  rigid mode rank  = {proj_info['rank']}")
                print(f"  constraint resid = {proj_info['constraint_residual_norm']:.6e}")

            if project_tr_rot:
                comment = (
                    f"E={E:.10f} "
                    f"|grad_proj|={np.linalg.norm(grad):.3e} "
                    f"|grad_raw|={np.linalg.norm(grad_raw):.3e}"
                )
            else:
                comment = f"E={E:.10f} |grad|={np.linalg.norm(grad):.3e}"

            write_xyz(
                "opt_traj.xyz",
                symbols,
                coords_angstrom,
                comment=comment,
                mode="a",
            )

        return E, grad.reshape(-1)

    # -------------------------
    # Energy function (cached)
    # -------------------------
    def fun(x):
        nonlocal last_x, last_E, last_grad

        if last_x is None or not np.allclose(x, last_x):
            last_E, last_grad = evaluate(x)
            last_x = x.copy()

        return last_E

    # -------------------------
    # Gradient function (cached)
    # -------------------------
    def jac(x):
        nonlocal last_x, last_E, last_grad

        if last_x is None or not np.allclose(x, last_x):
            last_E, last_grad = evaluate(x)
            last_x = x.copy()

        return last_grad

    # -------------------------
    # Observer callback
    # -------------------------
    def callback(optimizer_x):
        coords_bohr = optimizer_to_cartesian(optimizer_x).reshape(-1, 3)
        for obs in observers:
            observer_data[obs].append(
                obs.observe(coords_bohr)
            )

    # -------------------------
    # Run optimization
    # -------------------------
    print("Starting BFGS optimization...")
    print(f"Going to update for {maxiter} iterations or until gradient norm < {gtol:.2e} Ha/bohr")

    result = minimize(
        fun=fun,
        x0=optimizer_x0,
        jac=jac,
        method="BFGS",
        callback=callback,
        options=dict(
            gtol=gtol,
            maxiter=maxiter,
            disp=debug,
        ),
    )

    if project_tr_rot:
        # Preserve the public result.x contract: callers and examples expect
        # optimized Cartesian coordinates in bohr, not the internal redundant
        # optimizer displacement u.
        optimizer_displacement = np.asarray(result.x, dtype=float).copy()
        result.optimizer_displacement = optimizer_displacement
        result.x = optimizer_to_cartesian(optimizer_displacement)

        # SciPy's hess_inv acts in u-space.  Map it to the physical Cartesian
        # constrained inverse Hessian so downstream inspection sees the same
        # P_step @ B_inv @ P_grad sandwich used for coordinate updates.
        if hasattr(result, "hess_inv"):
            result.hess_inv_optimizer = np.asarray(result.hess_inv).copy()
            result.hess_inv = (
                step_projector
                @ result.hess_inv_optimizer
                @ gradient_projector
            )

        # result.jac is already P_grad @ g and therefore has the shape and units
        # of a Cartesian projected gradient.  Keep a named copy to make that
        # convention explicit in saved OptimizeResult objects.
        if hasattr(result, "jac"):
            result.jac_projected_cartesian = np.asarray(result.jac).copy()

        result.constraint_type = "fixed_reference_mass_weighted_translation_rotation"
        result.constraint_rank = reference_projectors["rank"]
        result.constraint_residual = (
            reference_projectors["rigid_basis_mass_weighted"].T
            @ (
                np.repeat(np.sqrt(masses), 3)
                * (result.x - x_reference_flat)
            )
        )

    return result, observer_data
