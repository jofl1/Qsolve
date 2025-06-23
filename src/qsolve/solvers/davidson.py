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
        if hasattr(A, "dot"):
            matvec = lambda x: A.dot(x)
        else:
            matvec = lambda x: A @ x
    else:  # LinearOperator
        n = A.shape[0]
        matvec = A.matvec

    # Parameters
    if max_subspace is None:
        max_subspace = min(n, max(2 * k + 10, 20))
    
    # Ensure max_subspace is reasonable
    max_subspace = min(max_subspace, n)

    # Initialize
    V = np.zeros((n, max_subspace), dtype=np.float64)
    W = np.zeros((n, max_subspace), dtype=np.float64)  # W = A*V
    
    # Initial guess
    if v0 is None:
        # Random initial guess
        rng = np.random.default_rng(42)
        for i in range(k):
            V[:, i] = rng.standard_normal(n)
            V[:, i] /= np.linalg.norm(V[:, i])
    else:
        if v0.ndim == 1:
            v0 = v0.reshape(-1, 1)
        nv0 = v0.shape[1]
        for i in range(min(nv0, k)):
            V[:, i] = v0[:, i] / np.linalg.norm(v0[:, i])

    # Orthogonalize initial vectors using QR
    V_init, _ = np.linalg.qr(V[:, :k])
    V[:, :k] = V_init
    
    # Compute A*V for initial vectors
    for i in range(k):
        W[:, i] = matvec(V[:, i])

    # Storage
    converged = np.zeros(k, dtype=bool)
    eigenvalues = np.zeros(k)
    eigenvectors = np.zeros((n, k))
    
    # Get diagonal for default preconditioner
    if preconditioner is None:
        if hasattr(A, "diagonal"):
            diag_A = A.diagonal()
        elif hasattr(A, "toarray"):
            diag_A = A.diagonal()
        else:
            # Estimate diagonal by computing A*e_i for a few basis vectors
            diag_A = np.zeros(n)
            for i in range(min(n, 100)):
                ei = np.zeros(n)
                ei[i] = 1.0
                diag_A[i] = matvec(ei)[i]
            if n > 100:
                # Use average for remaining elements
                diag_A[100:] = np.mean(diag_A[:100])

    if verbose:
        logger.info("Davidson solver: finding %d eigenvalues", k)
        logger.info("Matrix size: %d, Max subspace: %d", n, max_subspace)

    # Main iteration
    m = k  # Current subspace size
    for iteration in range(max_iter):
        # Build subspace matrix H = V^T * A * V
        H_sub = V[:, :m].T @ W[:, :m]
        
        # Ensure symmetric
        H_sub = 0.5 * (H_sub + H_sub.T)
        
        # Solve subspace problem
        theta, s = np.linalg.eigh(H_sub)
        
        # Sort by eigenvalue
        idx = np.argsort(theta)
        theta = theta[idx[:k]]
        s = s[:, idx[:k]]
        
        # Ritz vectors
        u = V[:, :m] @ s
        Au = W[:, :m] @ s
        
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

        if verbose and (iteration % 10 == 0 or np.any(converged != prev_converged)):
            n_conv = np.sum(converged)
            logger.info(
                "  Iter %d: %d/%d converged, max residual = %.2e",
                iteration, n_conv, k, max_residual
            )

        # Check if all converged
        if np.all(converged):
            if verbose:
                logger.info("Converged in %d iterations", iteration + 1)
            break

        # Add new search directions
        n_add = 0
        for i in range(k):
            if converged[i] or m + n_add >= max_subspace:
                continue
                
            # Residual
            r = Au[:, i] - theta[i] * u[:, i]
            
            # Skip if residual is too small
            if np.linalg.norm(r) < 1e-14:
                continue
            
            # Apply preconditioner
            if preconditioner is not None:
                if hasattr(preconditioner, "matvec"):
                    t = preconditioner.matvec(r)
                else:
                    t = preconditioner @ r
            else:
                # Default diagonal (Jacobi) preconditioner
                # t = r / (diag(A) - theta[i])
                denominator = diag_A - theta[i]
                
                # Safe division: avoid very small denominators
                # Use a threshold based on matrix norm estimate
                threshold = max(0.1, 1e-3 * np.max(np.abs(diag_A)))
                safe_denom = np.where(
                    np.abs(denominator) < threshold,
                    np.sign(denominator + 1e-16) * threshold,  # Add tiny value to handle exact zeros
                    denominator
                )
                t = r / safe_denom

            # Orthogonalize against current subspace
            for j in range(m + n_add):
                t -= np.dot(V[:, j], t) * V[:, j]
            
            # Normalize
            t_norm = np.linalg.norm(t)
            if t_norm > 1e-10:
                t /= t_norm
                V[:, m + n_add] = t
                W[:, m + n_add] = matvec(t)
                n_add += 1

        m += n_add

        # Restart if necessary
        if m >= max_subspace or n_add == 0:
            if verbose and m >= max_subspace:
                logger.info("  Restarting: subspace size %d >= %d", m, max_subspace)
            
            # Keep best k vectors
            m = k
            V[:, :k] = u[:, :k]
            W[:, :k] = Au[:, :k]
            
            # Re-orthonormalize for numerical stability
            V[:, :k], _ = np.linalg.qr(V[:, :k])
            for i in range(k):
                W[:, i] = matvec(V[:, i])
                
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


class JacobiDavidsonPreconditioner(LinearOperator):
    """Jacobi-Davidson preconditioner for improved convergence"""
    
    def __init__(self, A: Union[np.ndarray, csr_matrix], sigma: float = 0.0):
        """
        Create Jacobi-Davidson preconditioner M ≈ (A - σI)^(-1)
        
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
        
        # Safe division threshold based on matrix scale
        self.threshold = max(0.1, 1e-3 * np.max(np.abs(self.diag)))
        
        # Ensure no zero denominators
        self.safe_diag = np.where(
            np.abs(self.diag_shifted) < self.threshold,
            np.sign(self.diag_shifted + 1e-16) * self.threshold,
            self.diag_shifted
        )
        
    def _matvec(self, x):
        """Apply preconditioner: y = M*x = (diag(A) - σI)^(-1) * x"""
        return x / self.safe_diag


def build_quantum_preconditioner(
    H: Union[np.ndarray, csr_matrix], 
    target_energy: Optional[float] = None
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
        
    return JacobiDavidsonPreconditioner(H, sigma=target_energy)


def solve_davidson(system, n_states=1, **kwargs):
    """
    Solve for eigenstates using the Davidson algorithm.
    
    This is a drop-in replacement for solve_eigenstates that uses
    the Davidson algorithm instead of ARPACK/LOBPCG.
    """
    from ..solvers_main import build_hamiltonian
    from ..core import normalize_wavefunction, Result
    
    # Build Hamiltonian
    H = build_hamiltonian(system, sparse=True)
    
    # Ensure CSR format for optimal performance
    if hasattr(H, 'tocsr'):
        H = H.tocsr()
    
    # Create preconditioner
    preconditioner = build_quantum_preconditioner(H)
    
    # Initial guess - use a better strategy
    x = system.grid.x
    V = system.potential(x)
    
    # Find the minimum of the potential
    min_idx = np.argmin(V)
    x_min = x[min_idx]
    
    # Estimate characteristic length scale
    dx = x[1] - x[0]
    L = x[-1] - x[0]
    
    # Create initial guesses based on harmonic oscillator eigenstates
    # centered at the potential minimum
    v0 = np.zeros((len(x), n_states))
    
    # Width parameter for Gaussian
    sigma = L / 10  # Adjust based on system size
    
    for i in range(n_states):
        if i == 0:
            # Ground state: Gaussian centered at potential minimum
            v0[:, 0] = np.exp(-((x - x_min)**2) / (2 * sigma**2))
        else:
            # Excited states: Hermite polynomials * Gaussian
            # Simple approximation using oscillating functions
            v0[:, i] = np.exp(-((x - x_min)**2) / (4 * sigma**2)) * np.cos(i * np.pi * (x - x_min) / L)
        
        # Normalize
        v0[:, i] = normalize_wavefunction(v0[:, i], x)
    
    # Extract solver parameters
    max_iter = kwargs.get("max_iterations", 200)
    tol = kwargs.get("tolerance", 1e-9)
    verbose = kwargs.get("verbose", False)
    
    # Set optimal max_subspace for quantum problems
    max_subspace = kwargs.get("max_subspace", min(H.shape[0], max(20, 10 * n_states)))
    
    start_time = time.time()
    
    # Call Davidson solver
    eigenvalues, eigenvectors = davidson(
        H,
        k=n_states,
        v0=v0,
        tol=tol,
        max_iter=max_iter,
        max_subspace=max_subspace,
        preconditioner=preconditioner,
        verbose=verbose
    )
    
    solve_time = time.time() - start_time
    
    # Normalize eigenvectors on the grid
    for i in range(n_states):
        eigenvectors[:, i] = normalize_wavefunction(eigenvectors[:, i], x)
    
    # Check convergence by computing residuals
    residuals = np.zeros(n_states)
    converged = np.zeros(n_states, dtype=bool)
    
    for i in range(n_states):
        if hasattr(H, 'dot'):
            Hv = H.dot(eigenvectors[:, i])
        else:
            Hv = H @ eigenvectors[:, i]
        r = Hv - eigenvalues[i] * eigenvectors[:, i]
        residuals[i] = np.linalg.norm(r)
        converged[i] = residuals[i] < tol
    
    if verbose:
        logger.info("Davidson solver completed in %.3fs", solve_time)
        for i in range(n_states):
            logger.info("State %d: E = %.6f, residual = %.2e, converged = %s", 
                       i, eigenvalues[i], residuals[i], converged[i])
    
    # If Davidson didn't converge well, fall back to eigsh
    if not np.all(converged) and kwargs.get("fallback", True):
        logger.warning("Davidson did not fully converge, falling back to eigsh")
        from ..solvers_main import solve_eigenstates
        
        # Use Davidson's best guess as initial vector for eigsh
        fallback_result = solve_eigenstates(
            system,
            n_states=n_states,
            method="sparse",
            initial_guess=eigenvectors[:, 0] if n_states == 1 else None,
            tolerance=tol,
            max_iterations=max_iter,
            verbose=verbose
        )
        
        return fallback_result
    
    # Create result object compatible with qsolve
    result = Result(
        energies=eigenvalues,
        wavefunctions=eigenvectors,
        system=system,
        info={
            "method": "davidson",
            "converged": bool(np.all(converged)),
            "converged_states": converged.tolist(),
            "residual_norms": residuals.tolist(),
            "tolerance": tol,
            "max_iterations": max_iter,
            "solve_time": solve_time,
            "matrix_size": H.shape[0],
            "n_states": n_states,
            "preconditioner": "jacobi-davidson",
        },
    )
    
    return result


# Benchmark comparison function
def benchmark_davidson_vs_eigsh(system, n_states=5):
    """Compare Davidson vs ARPACK performance"""
    from ..solvers_main import build_hamiltonian, solve_eigenstates
    import matplotlib.pyplot as plt
    
    H = build_hamiltonian(system, sparse=True)
    x = system.grid.x
    
    logger.info("=" * 60)
    logger.info("DAVIDSON vs EIGSH BENCHMARK")
    logger.info("=" * 60)
    logger.info("System: %s", system.name)
    logger.info("Matrix size: %d", H.shape[0])
    logger.info("Finding %d lowest eigenvalues\n", n_states)
    
    # Test EIGSH (current method)
    logger.info("1. ARPACK eigsh (current method):")
    start = time.time()
    result_eigsh = solve_eigenstates(system, n_states=n_states, method="sparse")
    time_eigsh = time.time() - start
    logger.info("   Time: %.3fs", time_eigsh)
    
    # Test Davidson
    logger.info("\n2. Davidson algorithm:")
    start = time.time()
    result_davidson = solve_davidson(system, n_states=n_states, verbose=True, fallback=False)
    time_davidson = time.time() - start
    logger.info("   Time: %.3fs", time_davidson)
    
    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 60)
    if time_davidson > 0:
        logger.info("Speedup: %.1fx", time_eigsh / time_davidson)
    
    logger.info("\nEigenvalues:")
    logger.info("State |   EIGSH    |  Davidson  | Difference")
    logger.info("-" * 45)
    for i in range(n_states):
        E_eigsh = result_eigsh.energies[i]
        E_davidson = result_davidson.energies[i]
        diff = abs(E_eigsh - E_davidson)
        logger.info("  %d   | %10.6f | %10.6f | %.2e", i, E_eigsh, E_davidson, diff)
    
    # Plot comparison
    fig, axes = plt.subplots(n_states, 2, figsize=(12, 3 * n_states))
    if n_states == 1:
        axes = axes.reshape(1, -1)
        
    for i in range(n_states):
        # EIGSH
        ax = axes[i, 0]
        ax.plot(x, result_eigsh.wavefunctions[:, i], 'b-', linewidth=2)
        ax.set_title(f"EIGSH: E_{i} = {result_eigsh.energies[i]:.6f}")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("ψ(x)")
        
        # Davidson
        ax = axes[i, 1]
        ax.plot(x, result_davidson.wavefunctions[:, i], 'r-', linewidth=2)
        ax.set_title(f"Davidson: E_{i} = {result_davidson.energies[i]:.6f}")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("ψ(x)")
        
    if time_davidson > 0:
        fig.suptitle(f"Davidson vs EIGSH: {time_eigsh/time_davidson:.1f}x speedup", fontsize=14)
    else:
        fig.suptitle("Davidson vs EIGSH Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return {
        "time_eigsh": time_eigsh,
        "time_davidson": time_davidson,
        "speedup": time_eigsh / time_davidson if time_davidson > 0 else 0,
        "energies_match": np.allclose(result_eigsh.energies, result_davidson.energies, rtol=1e-6),
    }


if __name__ == "__main__":
    # Test the Davidson solver
    import sys
    sys.path.append("..")
    import qsolve
    
    # Enable detailed logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Test on different systems
    for system_name in ["harmonic", "hydrogen", "double_well"]:
        logger.info("\n" + "=" * 60)
        logger.info("Testing %s", system_name)
        logger.info("=" * 60)
        
        system = qsolve.System.create(system_name, grid_points=501)
        results = benchmark_davidson_vs_eigsh(system, n_states=3)
        
        logger.info("\nSummary: Davidson speedup: %.1fx", results["speedup"])
        logger.info("Eigenvalues match: %s", results["energies_match"])