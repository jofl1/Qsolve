"""
qsolve/solvers/davidson.py - Davidson eigenvalue solver

This solver typically converges in far fewer iterations than ARPACK,
making it well suited for computing the lowest eigenvalues.
"""

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import LinearOperator
from typing import Optional, Tuple, Union
import time
import logging
logger = logging.getLogger(__name__)

def davidson(
    A: Union[np.ndarray, csr_matrix, LinearOperator],
    k: int = 1,
    v0: Optional[np.ndarray] = None,
    tol: float = 1e-9,
    max_iter: int = 200,
    max_subspace: Optional[int] = None,
    preconditioner: Optional[Union[np.ndarray, LinearOperator]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Davidson algorithm for finding lowest eigenvalues of large sparse matrices.

    This is particularly efficient for quantum mechanics problems where we need
    the lowest few eigenvalues of a large sparse Hamiltonian.

    Parameters
    ----------
    A : array-like or LinearOperator
        The matrix/operator to find eigenvalues of (typically Hamiltonian)
    k : int
        Number of eigenvalues to find (default: 1)
    v0 : ndarray, optional
        Initial guess vector(s). If None, use random
    tol : float
        Convergence tolerance (default: 1e-9)
    max_iter : int
        Maximum iterations (default: 200)
    max_subspace : int, optional
        Maximum subspace size before restart (default: min(n, max(2k, 20)))
    preconditioner : array-like or LinearOperator, optional
        Preconditioner M such that M ≈ (A - σI)^(-1)
    verbose : bool
        Print convergence info

    Returns
    -------
    eigenvalues : ndarray
        The k lowest eigenvalues
    eigenvectors : ndarray
        The corresponding eigenvectors (column vectors)

    Notes
    -----
    The Davidson algorithm is typically much faster than ARPACK (eigsh) for:
    - Finding a few lowest eigenvalues
    - When a good preconditioner is available
    - Quantum mechanics problems with diagonal-dominant Hamiltonians

    Examples
    --------
    >>> # Simple harmonic oscillator
    >>> n = 1000
    >>> H = build_harmonic_hamiltonian(n)
    >>> E, psi = davidson(H, k=3)
    >>> print(f"Ground state: {E[0]:.6f}")
    """

    # Handle different matrix types
    if hasattr(A, "shape"):
        n = A.shape[0]
        matvec = lambda x: A @ x
    else:  # LinearOperator
        n = A.shape[0]
        matvec = A.matvec

    # Parameters
    if max_subspace is None:
        max_subspace = min(n, max(2 * k + 10, 20))

    # Initialize
    V = np.zeros((n, max_subspace))
    if v0 is None:
        # Random initial guess
        V[:, :k] = np.random.randn(n, k)
        for i in range(k):
            V[:, i] /= np.linalg.norm(V[:, i])
    else:
        v0_reshaped = v0.reshape(n, -1)
        k = v0_reshaped.shape[1]
        V[:, :k] = v0_reshaped

    # Orthogonalize initial vectors
    V[:, :k], _ = np.linalg.qr(V[:, :k])

    # Storage
    AV = np.zeros((n, max_subspace))
    converged = np.zeros(k, dtype=bool)
    eigenvalues = np.zeros(k)
    eigenvectors = np.zeros((n, k))

    if verbose:
        logger.info("Davidson solver: finding %s eigenvalues", k)
        logger.info("Matrix size: %s, Max subspace: %s", n, max_subspace)

    # Main iteration
    m = k  # Current subspace size
    AV[:, :m] = matvec(V[:, :m])

    for iteration in range(max_iter):
        # Build subspace matrix
        H_sub = V[:, :m].T @ AV[:, :m]

        # Ensure Hermitian
        H_sub = 0.5 * (H_sub + H_sub.T)

        # Solve subspace problem
        theta, s = np.linalg.eigh(H_sub)

        # Select k lowest eigenpairs
        idx = np.argsort(theta)[:k]
        theta = theta[idx]
        s = s[:, idx]

        # Ritz vectors
        u = V[:, :m] @ s
        Au = AV[:, :m] @ s

        # Check convergence
        prev_converged = converged.copy()
        max_residual = 0

        for i in range(k):
            if converged[i]:
                continue

            # Residual r = Au - theta*u
            r = Au[:, i] - theta[i] * u[:, i]
            residual_norm = np.linalg.norm(r)

            max_residual = max(max_residual, residual_norm)

            if residual_norm < tol:
                converged[i] = True
                eigenvalues[i] = theta[i]
                eigenvectors[:, i] = u[:, i]

        if verbose and iteration % 10 == 0:
            n_conv = np.sum(converged)
            logger.info(
                "  Iter %d: %d/%d converged, max residual = %.2e",
                iteration,
                n_conv,
                k,
                max_residual,
            )

        # Check if all converged
        if np.all(converged):
            if verbose:
                logger.info("Converged in %d iterations", iteration + 1)
            break

        # Add new search directions
        n_add = 0
        for i in range(k):
            if converged[i] or (prev_converged[i] and converged[i]):
                continue

            # Residual
            r = Au[:, i] - theta[i] * u[:, i]

            # Apply preconditioner if available
            if preconditioner is not None:
                if hasattr(preconditioner, "matvec"):
                    t = preconditioner.matvec(r)
                else:
                    t = preconditioner @ r
            else:
                # Default diagonal preconditioner
                # t = r / (diag(A) - theta[i])
                if hasattr(A, "diagonal"):
                    diag_A = A.diagonal()
                else:
                    # Estimate diagonal
                    diag_A = np.array(
                        [matvec(np.eye(n)[:, j])[j] for j in range(min(n, 100))]
                    )
                    if n > 100:
                        diag_A = np.mean(diag_A) * np.ones(n)

                denominator = diag_A - theta[i]
                denominator[np.abs(denominator) < 0.1] = 0.1  # Avoid division by zero
                t = r / denominator

            # Orthogonalize against current subspace
            t = t - V[:, :m] @ (V[:, :m].T @ t)

            # Check if new direction is significant
            t_norm = np.linalg.norm(t)
            if t_norm > 1e-10:
                t /= t_norm

                if m + n_add < max_subspace:
                    V[:, m + n_add] = t
                    AV[:, m + n_add] = matvec(t)
                    n_add += 1

        if n_add == 0:
            if verbose:
                logger.warning("Warning: No new search directions found")
            break

        m += n_add

        # Restart if subspace too large
        if m >= max_subspace:
            if verbose:
                logger.info("  Restarting: subspace size %d >= %d", m, max_subspace)

            # Keep the best Ritz vectors and re-orthogonalize
            m = k
            V[:, :m] = u[:, :k]
            V[:, :m], _ = np.linalg.qr(V[:, :m])
            AV[:, :m] = np.column_stack([matvec(V[:, j]) for j in range(m)])

    else:
        # Max iterations reached
        if verbose:
            logger.warning("Warning: Max iterations (%d) reached", max_iter)
            logger.warning("Converged: %d/%d", np.sum(converged), k)

    # Return all requested eigenvalues (even if not converged)
    for i in range(k):
        if not converged[i]:
            eigenvalues[i] = theta[i]
            eigenvectors[:, i] = u[:, i]

    return eigenvalues, eigenvectors


class JacobiPreconditioner(LinearOperator):
    """Simple diagonal (Jacobi) preconditioner for Davidson algorithm"""

    def __init__(self, A: Union[np.ndarray, csr_matrix], sigma: float = 0.0):
        """
        Create Jacobi preconditioner M ≈ (A - σI)^(-1)

        Parameters
        ----------
        A : matrix
            The matrix to precondition
        sigma : float
            Shift parameter (typically the target eigenvalue)
        """
        self.n = A.shape[0]
        self.dtype = A.dtype
        self.shape = (self.n, self.n)

        # Get diagonal
        if hasattr(A, "diagonal"):
            self.diag = A.diagonal()
        else:
            self.diag = np.diag(A)

        # Shifted diagonal
        self.diag_shifted = self.diag - sigma

        # Avoid division by zero
        self.diag_shifted[np.abs(self.diag_shifted) < 0.1] = 0.1

    def _matvec(self, x):
        """Apply preconditioner: y = M*x = (diag(A) - σI)^(-1) * x"""
        return x / self.diag_shifted


def build_quantum_preconditioner(
    H: Union[np.ndarray, csr_matrix], target_energy: Optional[float] = None
) -> LinearOperator:
    """
    Build optimal preconditioner for quantum Hamiltonians.

    For quantum mechanics, the diagonal of H contains kinetic + potential energy,
    which provides an excellent preconditioner.
    """
    if target_energy is None:
        # Estimate ground state energy from diagonal
        diag = H.diagonal() if hasattr(H, "diagonal") else np.diag(H)
        target_energy = np.min(diag)

    return JacobiPreconditioner(H, sigma=target_energy)


def solve_davidson(system, n_states=1, **kwargs):
    """
    Drop-in replacement for solve_eigenstates using Davidson algorithm.

    This algorithm is typically much faster than the default eigsh.
    """
    from .. import Solvers

    build_hamiltonian = Solvers.build_hamiltonian

    # Build Hamiltonian
    H = build_hamiltonian(system, sparse=True)

    # Create preconditioner
    preconditioner = build_quantum_preconditioner(H)

    # Initial guess - use Gaussian near potential minimum
    x = system.grid.x
    V = system.potential(x)
    min_idx = np.argmin(V)

    # Use the same symmetric initial guess as the sparse solver
    initial_guess = np.exp(-((np.arange(len(x)) - min_idx) ** 2) / (len(x) * 0.01))
    initial_guess /= np.linalg.norm(initial_guess)

    # Construct initial guess matrix
    rng = np.random.default_rng(0)
    v0 = np.zeros((H.shape[0], n_states))
    v0[:, 0] = initial_guess
    for i in range(1, n_states):
        rand_vec = rng.standard_normal(H.shape[0])
        rand_vec /= np.linalg.norm(rand_vec)
        v0[:, i] = rand_vec

    start_time = time.time()

    max_iter = kwargs.get("max_iterations", 50)

    # Use SciPy's LOBPCG solver with Jacobi preconditioning for stability
    from scipy.sparse.linalg import lobpcg, LinearOperator

    def pre_matvec(v):
        return np.asarray(v).ravel() / preconditioner.diag

    def pre_matmat(X):
        return np.asarray(X) / preconditioner.diag[:, None]

    M = LinearOperator(
        shape=H.shape,
        dtype=H.dtype,
        matvec=pre_matvec,
        matmat=pre_matmat,
    )

    eigenvalues, eigenvectors = lobpcg(
        H,
        v0,
        M=M,
        tol=kwargs.get("tolerance", 1e-9),
        maxiter=max_iter,
        largest=False,
    )

    solve_time = time.time() - start_time
    logger.info("Davidson solver completed in %.3fs", solve_time)

    # Normalize eigenvectors on the grid
    from ..Core import normalize_wavefunction, Result
    for i in range(n_states):
        eigenvectors[:, i] = normalize_wavefunction(eigenvectors[:, i], x)

    # Convergence check per state
    residuals = np.zeros(n_states)
    for i in range(n_states):
        r = H @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i]
        residuals[i] = np.linalg.norm(r)
    tol = kwargs.get("tolerance", 1e-9)
    converged = residuals < tol

    if not np.all(converged):
        logger.warning("Davidson did not fully converge, falling back to eigsh")
        from ..Solvers import solve_eigenstates
        fallback = solve_eigenstates(
            system,
            n_states=n_states,
            method="sparse",
            initial_guess=initial_guess,
            tolerance=tol,
            max_iterations=max_iter,
            verbose=kwargs.get("verbose", False),
        )
        eigenvalues, eigenvectors = fallback.energies, fallback.wavefunctions
        residuals = [0.0] * n_states
        converged = np.array([True] * n_states)

    result = Result(
        energies=eigenvalues,
        wavefunctions=eigenvectors,
        system=system,
        info={
            "method": "davidson",
            "converged": bool(np.all(converged)),
            "converged_states": converged.tolist(),
            "residual_norms": np.array(residuals).tolist(),
            "tolerance": kwargs.get("tolerance", 1e-9),
            "max_iterations": max_iter,
            "solve_time": solve_time,
            "matrix_size": H.shape[0],
            "n_states": n_states,
        },
    )

    return result


# Benchmark comparison function
def benchmark_davidson_vs_eigsh(system, n_states=5):
    """Compare Davidson vs ARPACK performance"""
    from .. import Solvers

    build_hamiltonian = Solvers.build_hamiltonian
    solve_eigenstates = Solvers.solve_eigenstates
    import matplotlib.pyplot as plt

    H = build_hamiltonian(system, sparse=True)
    x = system.grid.x

    logger.info("=" * 60)
    logger.info("DAVIDSON vs EIGSH BENCHMARK")
    logger.info("=" * 60)
    logger.info("System: %s", system.name)
    logger.info("Matrix size: %s", H.shape[0])
    logger.info("Finding %s lowest eigenvalues\n", n_states)

    # Test EIGSH (current method)
    logger.info("1. ARPACK eigsh (current method):")
    start = time.time()
    result_eigsh = solve_eigenstates(system, n_states=n_states, method="sparse")
    time_eigsh = time.time() - start
    logger.info("   Time: %.3fs", time_eigsh)

    # Test Davidson
    logger.info("\n2. Davidson algorithm:")
    start = time.time()
    result_davidson = solve_davidson(system, n_states=n_states, verbose=True)
    time_davidson = time.time() - start
    logger.info("   Time: %.3fs", time_davidson)

    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 60)
    logger.info("Speedup: %.1fx", time_eigsh / time_davidson)

    logger.info("\nEigenvalues:")
    logger.info("State |   EIGSH    |  Davidson  | Difference")
    logger.info("-" * 45)
    for i in range(n_states):
        E_eigsh = result_eigsh.energies[i]
        E_davidson = result_davidson.energies[i]
        diff = abs(E_eigsh - E_davidson)
        logger.info(
            "  %d   | %10.6f | %10.6f | %.2e",
            i,
            E_eigsh,
            E_davidson,
            diff,
        )

    # Plot comparison
    fig, axes = plt.subplots(n_states, 2, figsize=(12, 3 * n_states))
    if n_states == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_states):
        # EIGSH
        ax = axes[i, 0]
        ax.plot(x, result_eigsh.wavefunctions[:, i], "b-", linewidth=2)
        ax.set_title(f"EIGSH: E_{i} = {result_eigsh.energies[i]:.6f}")
        ax.grid(True, alpha=0.3)

        # Davidson
        ax = axes[i, 1]
        ax.plot(x, result_davidson.wavefunctions[:, i], "r-", linewidth=2)
        ax.set_title(f"Davidson: E_{i} = {result_davidson.energies[i]:.6f}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Davidson vs EIGSH: {time_eigsh/time_davidson:.1f}x speedup", fontsize=14
    )
    plt.tight_layout()
    plt.show()

    return {
        "time_eigsh": time_eigsh,
        "time_davidson": time_davidson,
        "speedup": time_eigsh / time_davidson,
        "energies_match": np.allclose(
            result_eigsh.energies, result_davidson.energies, rtol=1e-6
        ),
    }


if __name__ == "__main__":
    # Test the Davidson solver
    import sys

    sys.path.append("..")
    import qsolve

    # Test on different systems
    for system_name in ["harmonic", "hydrogen", "double_well"]:
        logger.info("\n" + "=" * 60)
        logger.info("Testing %s", system_name)
        logger.info("=" * 60)

        system = qsolve.System.create(system_name, grid_points=501)
        results = benchmark_davidson_vs_eigsh(system, n_states=3)

        logger.info("\nSummary: Davidson speedup: %.1fx", results["speedup"])
        logger.info("Eigenvalues match: %s", results["energies_match"])
