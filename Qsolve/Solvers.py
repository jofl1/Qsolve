"""
quantum/solvers.py - Quantum eigenvalue solvers
"""

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
import time
from typing import Optional, Dict, Any, Literal
import logging

from .Core import System, Result, normalize_wavefunction, Grid, Grid2D, Grid3D

logger = logging.getLogger(__name__)


def build_hamiltonian(system: System, sparse: bool = True):
    """Build the Hamiltonian matrix for the system.

    Args:
        system: Quantum system to build Hamiltonian for
        sparse: If True, return sparse matrix (recommended for large systems)

    Returns:
        Hamiltonian matrix (sparse or dense)
    """
    grid = system.grid
    kinetic_factor = -system.hbar**2 / (2 * system.mass)

    if isinstance(grid, Grid):
        n = grid.points
        dx2 = grid.dx ** 2
        if sparse:
            T = kinetic_factor / dx2 * diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format="csr")
            V = diags(system.potential(grid.x), 0, format="csr")
            return T + V
        else:
            T = kinetic_factor / dx2 * (
                np.diag(np.ones(n - 1), 1)
                + np.diag(-2 * np.ones(n), 0)
                + np.diag(np.ones(n - 1), -1)
            )
            V = np.diag(system.potential(grid.x))
            return T + V

    from scipy.sparse import kron, identity

    if isinstance(grid, Grid2D):
        nx, ny = grid.shape
        dx, dy = grid.dx, grid.dy
        Lx = diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx), format="csr") / dx**2
        Ly = diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny), format="csr") / dy**2
        I_x = identity(nx, format="csr")
        I_y = identity(ny, format="csr")
        if sparse:
            Lap = kron(Lx, I_y) + kron(I_x, Ly)
            V = diags(system.potential(*grid.coordinates()).reshape(-1), 0, format="csr")
            return kinetic_factor * Lap + V
        else:
            Lap = np.kron(Lx.toarray(), np.eye(ny)) + np.kron(np.eye(nx), Ly.toarray())
            V = np.diag(system.potential(*grid.coordinates()).reshape(-1))
            return kinetic_factor * Lap + V

    if isinstance(grid, Grid3D):
        nx, ny, nz = grid.shape
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        Lx = diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx), format="csr") / dx**2
        Ly = diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny), format="csr") / dy**2
        Lz = diags([1, -2, 1], [-1, 0, 1], shape=(nz, nz), format="csr") / dz**2
        I_x = identity(nx, format="csr")
        I_y = identity(ny, format="csr")
        I_z = identity(nz, format="csr")
        if sparse:
            Lap = kron(kron(Lx, I_y), I_z) + kron(kron(I_x, Ly), I_z) + kron(kron(I_x, I_y), Lz)
            V = diags(system.potential(*grid.coordinates()).reshape(-1), 0, format="csr")
            return kinetic_factor * Lap + V
        else:
            Lap = (
                np.kron(np.kron(Lx.toarray(), np.eye(ny)), np.eye(nz))
                + np.kron(np.kron(np.eye(nx), Ly.toarray()), np.eye(nz))
                + np.kron(np.kron(np.eye(nx), np.eye(ny)), Lz.toarray())
            )
            V = np.diag(system.potential(*grid.coordinates()).reshape(-1))
            return kinetic_factor * Lap + V

    raise TypeError("Unsupported grid type")


def solve_ground_state(
    system: System,
    method: Literal["auto", "sparse", "dense"] = "auto",
    initial_guess: Optional[np.ndarray] = None,
    tolerance: float = 1e-9,
    max_iterations: int = 500,
    verbose: bool = False,
) -> Result:
    """Solve for the ground state of a quantum system.

    Args:
        system: Quantum system to solve
        method: Solution method ('auto', 'sparse', or 'dense')
        initial_guess: Initial wavefunction guess
        tolerance: Convergence tolerance for iterative methods
        max_iterations: Maximum iterations for iterative methods
        verbose: Print progress information

    Returns:
        Result object containing energies, wavefunctions, and metadata
    """
    return solve_eigenstates(
        system,
        n_states=1,
        method=method,
        initial_guess=initial_guess,
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=verbose,
    )


def solve_eigenstates(
    system: System,
    n_states: int = 1,
    method: Literal["auto", "sparse", "dense", "davidson"] = "auto",
    initial_guess: Optional[np.ndarray] = None,
    tolerance: float = 1e-9,
    max_iterations: int = 500,
    verbose: bool = False,
) -> Result:
    """Solve for multiple eigenstates of a quantum system.

    Args:
        system: Quantum system to solve
        n_states: Number of eigenstates to compute
        method: Solution method
        initial_guess: Initial wavefunction guess for iterative methods
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        verbose: Print progress

    Returns:
        Result object with all computed eigenstates
    """
    # Input validation
    if not isinstance(n_states, int) or n_states < 1:
        raise ValueError(f"n_states must be a positive integer, got {n_states}")

    if n_states > system.grid.points // 2:
        raise ValueError(
            f"Cannot compute {n_states} states with only {system.grid.points} grid points. "
            f"Maximum reasonable number of states is {system.grid.points // 2}"
        )

    if method not in ["auto", "sparse", "dense", "davidson"]:
        raise ValueError(
            f"Unknown method '{method}'. Must be 'auto', 'sparse', 'dense', or 'davidson'"
        )

    if tolerance <= 0:
        raise ValueError(f"Tolerance must be positive, got {tolerance}")

    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")

    if verbose:
        logger.info(
            "Solving %s system (%s points)...", system.name, system.grid.points
        )

    # Choose method automatically based on system size
    if method == "auto":
        if system.grid.points < 500:
            method = "davidson" if n_states <= 3 else "dense"
        elif n_states <= 10:  # Davidson is best for few states
            method = "davidson"
        else:
            method = "sparse"

    start_time = time.time()

    # Build Hamiltonian
    H = build_hamiltonian(system, sparse=(method == "sparse"))

    # Solve eigenvalue problem
    if method == "dense":
        # Full diagonalization for small systems
        if verbose:
            logger.info("Using dense solver (full diagonalization)...")

        H_dense = H.toarray() if hasattr(H, "toarray") else H
        eigenvalues, eigenvectors = np.linalg.eigh(H_dense)

        # Take the lowest n_states
        energies = eigenvalues[:n_states]
        wavefunctions = eigenvectors[:, :n_states]

        info = {
            "method": "dense",
            "converged": True,
            "iterations": 1,
        }

    elif method == "sparse":
        # Iterative solver for large systems
        if verbose:
            logger.info("Using sparse iterative solver...")

        # Create initial guess if not provided
        if initial_guess is None:
            if isinstance(system.grid, Grid):
                x = system.grid.x
                V = system.potential(x)
                min_idx = np.argmin(V)
                initial_guess = np.exp(
                    -((np.arange(len(x)) - min_idx) ** 2) / (len(x) * 0.01)
                )
                initial_guess = normalize_wavefunction(initial_guess, x)
            else:
                initial_guess = np.random.randn(system.grid.points)
                initial_guess = normalize_wavefunction(initial_guess, system.grid)

        # Solve using ARPACK
        try:
            energies, wavefunctions = eigsh(
                H,
                k=n_states,
                which="SA",  # Smallest algebraic eigenvalues
                v0=initial_guess,
                tol=tolerance,
                maxiter=max_iterations,
                return_eigenvectors=True,
            )
            converged = True
        except Exception as e:
            if verbose:
                logger.warning("Warning: Solver did not fully converge: %s", e)
            converged = False
            # Return best estimate
            energies, wavefunctions = eigsh(
                H,
                k=n_states,
                which="SA",
                tol=tolerance * 10,  # Relaxed tolerance
                maxiter=max_iterations * 2,
                return_eigenvectors=True,
            )

        info = {
            "method": "sparse",
            "converged": converged,
            "tolerance": tolerance,
            "max_iterations": max_iterations,
        }

    elif method == "davidson":
        # Use our new Davidson solver!
        from .solvers.davidson import solve_davidson

        davidson_result = solve_davidson(
            system,
            n_states=n_states,
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        energies = davidson_result.energies
        wavefunctions = davidson_result.wavefunctions
        info = davidson_result.info

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize wavefunctions
    for i in range(n_states):
        wavefunctions[:, i] = normalize_wavefunction(wavefunctions[:, i], system.grid)
        # Ensure real for visualization (ground state should be real)
        if i == 0 and np.max(np.abs(wavefunctions[:, i].imag)) < 1e-10:
            wavefunctions[:, i] = wavefunctions[:, i].real

    solve_time = time.time() - start_time

    # Add timing and performance info
    info.update(
        {
            "solve_time": solve_time,
            "matrix_size": system.grid.points,
            "n_states": n_states,
        }
    )

    if verbose:
        logger.info("Solved in %.3f seconds", solve_time)
        logger.info("Ground state energy: E₀ = %.8f", energies[0])

    return Result(
        energies=energies, wavefunctions=wavefunctions, system=system, info=info
    )


# Advanced solvers (stubs for future implementation)


def solve_davidson(system: System, n_states: int = 1, **kwargs) -> Result:
    """Davidson algorithm for large sparse eigenvalue problems.

    More efficient than Lanczos for computing a few lowest eigenvalues.
    """
    from .solvers.davidson import solve_davidson as davidson_solve

    return davidson_solve(system, n_states=n_states, **kwargs)


def solve_lobpcg(system: System, n_states: int = 1, **kwargs) -> Result:
    """Locally Optimal Block Preconditioned Conjugate Gradient solver.

    This provides an alternative to ARPACK when a good preconditioner is
    available.  The implementation here is a thin wrapper around
    :func:`scipy.sparse.linalg.lobpcg` so that basic functionality is
    available instead of raising ``NotImplementedError``.
    """

    from scipy.sparse.linalg import lobpcg, LinearOperator

    # Build sparse Hamiltonian
    H = build_hamiltonian(system, sparse=True)

    n = system.grid.points

    # Initial guess vectors: use random normalized vectors
    X = np.random.randn(n, n_states)
    for i in range(n_states):
        X[:, i] = normalize_wavefunction(X[:, i], system.grid)

    # Diagonal preconditioner improves convergence
    diag = H.diagonal()

    def precond(x):
        return x / (diag + 1e-8)

    M = LinearOperator(matvec=precond, dtype=float, shape=H.shape)

    eigenvalues, eigenvectors = lobpcg(
        H,
        X,
        M=M,
        tol=kwargs.get("tolerance", 1e-9),
        maxiter=kwargs.get("max_iterations", 500),
    )

    # ``lobpcg`` returns unordered eigenpairs
    idx = np.argsort(eigenvalues)[:n_states]
    energies = eigenvalues[idx]
    wavefunctions = eigenvectors[:, idx]

    info = {
        "method": "lobpcg",
        "converged": True,
    }

    return Result(
        energies=energies,
        wavefunctions=wavefunctions,
        system=system,
        info=info,
    )


def solve_imaginary_time(system: System, beta: float = 10.0, **kwargs) -> Result:
    """Imaginary time propagation to approximate the ground state.

    Parameters
    ----------
    system : System
        Quantum system to solve.
    beta : float
        Total imaginary time.  Larger values yield better convergence.
    """

    dt = kwargs.get("dt", 0.1)
    n_steps = int(beta / dt)

    H = build_hamiltonian(system, sparse=True).tocsc()
    psi = np.random.randn(system.grid.points)
    psi = normalize_wavefunction(psi, system.grid)

    for _ in range(n_steps):
        psi = psi - dt * (H @ psi)
        psi = normalize_wavefunction(psi, system.grid)

    if isinstance(system.grid, Grid):
        energy = np.real(np.trapz(np.conj(psi) * (H @ psi), system.grid.x))
    else:
        energy = np.real(np.sum(np.conj(psi) * (H @ psi)) * np.prod(system.grid.dxs))

    info = {
        "method": "imaginary_time",
        "steps": n_steps,
        "beta": beta,
    }

    return Result(
        energies=np.array([energy]),
        wavefunctions=psi.reshape(-1, 1),
        system=system,
        info=info,
    )


# Convenience function
def solve(system: System, **kwargs) -> Result:
    """Convenience function that defaults to ground state."""
    return solve_ground_state(system, **kwargs)
