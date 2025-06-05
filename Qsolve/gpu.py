"""
qsolve/gpu.py - GPU acceleration utilities for quantum mechanics.

Provides significant speedup when CUDA or Metal GPUs are available.
Supports both NVIDIA GPUs (via CuPy) and Apple Silicon (via MLX).
"""

import numpy as np
from typing import Optional, Union, Callable, Literal

# Import unified GPU backend
try:
    from .gpu_unified import (
        UnifiedGPUSystem, unified_gpu_evolve, detect_gpu_capabilities,
        print_gpu_info, HAS_CUPY, HAS_MLX, HAS_JAX,
        UnifiedSplitOperator
    )
    HAS_UNIFIED = True
except ImportError:
    HAS_UNIFIED = False

# Sparse matrix detection helpers
try:
    from scipy.sparse import issparse as scipy_issparse
except Exception:  # pragma: no cover - scipy may not be installed
    scipy_issparse = lambda x: False  # type: ignore

try:
    from cupyx.scipy.sparse import issparse as cupy_issparse  # type: ignore
except Exception:  # pragma: no cover - CuPy may not be installed
    cupy_issparse = lambda x: False  # type: ignore
import time
import logging

# Try importing GPU libraries
logger = logging.getLogger(__name__)

try:
    import cupy as cp

    HAS_CUPY = True
    logger.info("CuPy available - GPU acceleration enabled.")
except ImportError:
    cp = np  # Fallback to NumPy
    HAS_CUPY = False
    logger.info("CuPy not found. Install with 'pip install cupy-cuda12x'.")

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap

    HAS_JAX = True
    # Set default device
    jax.config.update("jax_platform_name", "gpu" if HAS_CUPY else "cpu")
except ImportError:
    HAS_JAX = False
    jnp = np


def get_array_module(x):
    """Get the appropriate array module (NumPy or CuPy) for an array"""
    if HAS_CUPY and hasattr(x, "__cuda_array_interface__"):
        return cp
    return np


class GPUSystem:
    """GPU-accelerated quantum system with unified backend support"""

    def __init__(self, system, use_gpu=True, backend='auto'):
        self.cpu_system = system
        
        # Use unified backend if available
        if HAS_UNIFIED and use_gpu:
            self._unified_system = UnifiedGPUSystem(system, backend=backend)
            self.use_gpu = True
            self.backend_type = self._unified_system.backend.name
            
            # Copy attributes for compatibility
            self.grid_x = self._unified_system.grid_x
            self.grid_dx = self._unified_system.grid_dx
            self.n_points = self._unified_system.n_points
            self.V = self._unified_system.V
        else:
            # Fallback to original CUDA-only implementation
            self.use_gpu = use_gpu and HAS_CUPY
            self.backend_type = "CUDA (CuPy)" if self.use_gpu else "CPU (NumPy)"
            self._unified_system = None

            if self.use_gpu:
                # Transfer to GPU
                self.grid_x = cp.asarray(system.grid.x)
                self.grid_dx = system.grid.dx
                self.n_points = system.grid.points

                # Precompute potential on GPU
                self.V = cp.asarray(system.potential(system.grid.x))
            else:
                self.grid_x = system.grid.x
                self.grid_dx = system.grid.dx
                self.n_points = system.grid.points
                self.V = system.potential(system.grid.x)

    @property
    def xp(self):
        """Array module (cupy, mlx, or numpy)"""
        if self._unified_system:
            return self._unified_system.xp
        return cp if self.use_gpu else np


def gpu_hamiltonian(system: GPUSystem, sparse=True):
    """Build Hamiltonian matrix on GPU.

    When ``sparse`` is ``True`` and a GPU is available this function will
    construct the matrix using ``cupyx.scipy.sparse`` which keeps the matrix in
    GPU memory.  On CPU it falls back to ``scipy.sparse``.  If ``sparse`` is
    ``False`` a dense matrix is returned using either NumPy or CuPy arrays.
    """

    xp = system.xp
    n = system.n_points
    dx2 = system.grid_dx**2

    if sparse:
        if system.use_gpu:
            # GPU sparse using CuPy
            from cupyx.scipy.sparse import diags as gpu_diags

            kinetic = -0.5 / dx2 * gpu_diags(
                [1, -2, 1], [-1, 0, 1], shape=(n, n), format="csr"
            )
            potential = gpu_diags(system.V, 0, format="csr")
            return kinetic + potential
        else:
            # CPU sparse (same as before)
            from scipy.sparse import diags

            kinetic = -0.5 / dx2 * diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
            potential = diags(system.V, 0)
            return kinetic + potential

    # Dense construction (CPU or GPU)
    # Kinetic energy operator
    T = xp.zeros((n, n))
    xp.fill_diagonal(T, -2.0)
    if n > 1:
        xp.fill_diagonal(T[1:], 1.0)
        xp.fill_diagonal(T[:, 1:], 1.0)
    T *= -0.5 / dx2

    # Potential energy operator
    V = xp.diag(system.V)

    return T + V


def gpu_matvec_jax(H, psi):
    """GPU matrix-vector multiplication using JAX"""
    return H @ psi


if HAS_JAX:
    gpu_matvec_jax = jit(gpu_matvec_jax)


def gpu_eigensolver(H_gpu, k=1, method="power", max_iter=1000, tol=1e-9):
    """
    GPU-accelerated eigenvalue solver.

    Methods:
    - 'power': Power iteration (good for ground state)
    - 'lanczos': Lanczos algorithm
    - 'full': Full diagonalization (small systems only)
    """
    xp = get_array_module(H_gpu)
    n = H_gpu.shape[0]
    is_sparse = scipy_issparse(H_gpu) or cupy_issparse(H_gpu)

    if method == "full" and n < 2000 and not is_sparse:
        # Full diagonalization on GPU
        if xp is cp and HAS_CUPY:
            E, V = cp.linalg.eigh(H_gpu)
        else:
            E, V = np.linalg.eigh(H_gpu)
        return E[:k], V[:, :k]

    elif method == "power":
        # Power iteration for ground state
        v = xp.random.randn(n)
        v /= xp.linalg.norm(v)

        for i in range(max_iter):
            # v_new = H @ v
            v_new = H_gpu @ v

            # Rayleigh quotient
            lambda_new = xp.real(xp.vdot(v, v_new))

            # Normalize
            v_new /= xp.linalg.norm(v_new)

            # Check convergence
            if xp.linalg.norm(v_new - v) < tol:
                break

            v = v_new

        return xp.array([lambda_new]), v.reshape(n, 1)

    elif method == "lanczos" or (is_sparse and method == "full"):
        # Lanczos algorithm works for both dense and sparse matrices
        return gpu_lanczos(H_gpu, k, max_iter, tol)
    elif method == "eigsh" and is_sparse:
        if xp is cp and HAS_CUPY:
            from cupyx.scipy.sparse.linalg import eigsh  # type: ignore
        else:
            from scipy.sparse.linalg import eigsh
        E, V = eigsh(H_gpu, k=k, which="SA", tol=tol, maxiter=max_iter)
        return E, V
    else:
        raise ValueError(f"Unknown method: {method}")


def gpu_lanczos(H, k=1, max_iter=100, tol=1e-9):
    """Lanczos algorithm on GPU"""
    xp = get_array_module(H)
    n = H.shape[0]

    # Initialize
    v = xp.random.randn(n)
    v /= xp.linalg.norm(v)

    V = xp.zeros((n, max_iter))
    T = xp.zeros((max_iter, max_iter))

    V[:, 0] = v
    beta = 0
    v_prev = xp.zeros(n)

    for j in range(min(k + 20, max_iter)):
        # Lanczos iteration
        w = H @ v - beta * v_prev
        alpha = xp.real(xp.vdot(v, w))
        w = w - alpha * v

        # Reorthogonalization (for numerical stability)
        for i in range(j):
            w = w - xp.vdot(V[:, i], w) * V[:, i]

        beta = xp.linalg.norm(w)

        if beta < tol:
            break

        v_prev = v
        v = w / beta
        V[:, j + 1] = v

        # Build tridiagonal matrix
        T[j, j] = alpha
        if j > 0:
            T[j - 1, j] = beta
            T[j, j - 1] = beta

    # Solve tridiagonal eigenvalue problem
    j = min(j + 1, max_iter)
    T_sub = T[:j, :j]

    if xp is cp and HAS_CUPY:
        theta, s = cp.linalg.eigh(T_sub)
    else:
        theta, s = np.linalg.eigh(T_sub)

    # Get k lowest eigenvalues
    idx = xp.argsort(theta)[:k]
    eigenvalues = theta[idx]

    # Compute Ritz vectors
    eigenvectors = V[:, :j] @ s[:, idx]

    return eigenvalues, eigenvectors


class GPUSplitOperator:
    """GPU-accelerated split-operator propagator"""

    def __init__(self, system: GPUSystem, dt: float):
        self.system = system
        self.dt = dt
        self.xp = system.xp

        # Precompute on GPU
        n = system.n_points
        dx = system.grid_dx

        # Momentum space (using GPU FFT frequencies)
        if self.xp is cp and HAS_CUPY:
            self.k = 2 * np.pi * cp.fft.fftfreq(n, dx)
        else:
            self.k = 2 * np.pi * np.fft.fftfreq(n, dx)

        # Evolution operators
        T_k = 0.5 * self.k**2
        self.U_T = self.xp.exp(-1j * T_k * dt)
        self.U_V_half = self.xp.exp(-0.5j * system.V * dt)

    def step(self, psi_gpu):
        """Single time step on GPU"""
        xp = self.xp

        # Split-operator steps
        psi_gpu = self.U_V_half * psi_gpu

        if xp is cp and HAS_CUPY:
            psi_gpu = cp.fft.ifft(self.U_T * cp.fft.fft(psi_gpu))
        else:
            psi_gpu = np.fft.ifft(self.U_T * np.fft.fft(psi_gpu))

        psi_gpu = self.U_V_half * psi_gpu

        return psi_gpu


def gpu_evolve(system, psi0, time_span, dt=0.01, use_gpu=True, backend='auto'):
    """GPU-accelerated time evolution with unified backend support"""
    
    # Use unified backend if available
    if HAS_UNIFIED and use_gpu:
        return unified_gpu_evolve(system, psi0, time_span, dt=dt, backend=backend)
    
    # Fallback to original CUDA-only implementation
    if not HAS_CUPY and use_gpu:
        logger.warning("CuPy not available, falling back to CPU")
        use_gpu = False

    # Create GPU system
    gpu_system = GPUSystem(system, use_gpu=use_gpu)
    xp = gpu_system.xp

    # Transfer initial state to GPU
    psi_gpu = xp.asarray(psi0)

    # Setup evolution
    times = np.arange(time_span[0], time_span[1], dt)
    n_steps = len(times)

    # Create propagator
    propagator = GPUSplitOperator(gpu_system, dt)

    # Storage (keep on CPU to save GPU memory)
    wavefunctions = np.zeros((n_steps, len(psi0)), dtype=complex)
    wavefunctions[0] = psi0

    logger.info("GPU Evolution: %s steps on %s...", n_steps, "GPU" if use_gpu else "CPU")

    start_time = time.time()

    # Time evolution
    for i in range(1, n_steps):
        psi_gpu = propagator.step(psi_gpu)

        # Transfer back to CPU for storage (optional)
        if i % 10 == 0 or i == n_steps - 1:
            if use_gpu:
                wavefunctions[i] = cp.asnumpy(psi_gpu)
            else:
                wavefunctions[i] = psi_gpu

    elapsed = time.time() - start_time
    logger.info("Evolution completed in %.3fs", elapsed)
    logger.info("  Performance: %d steps/second", int(n_steps / elapsed))

    # Create result
    from .evolution import EvolutionResult

    return EvolutionResult(times=times, wavefunctions=wavefunctions, system=system)


def benchmark_gpu(system_name="hydrogen", grid_points=2000):
    """Benchmark GPU vs CPU performance.

    The benchmark focuses on large grid sizes where sparse matrices and GPU
    acceleration provide a noticeable advantage.  Results are printed using the
    logging system.
    """
    import qsolve

    logger.info("=" * 60)
    logger.info("GPU ACCELERATION BENCHMARK")
    logger.info("=" * 60)

    if not HAS_CUPY:
        logger.info("CuPy not installed. Install with:")
        logger.info("  pip install cupy-cuda12x  # For CUDA 12.x")
        logger.info("  pip install cupy-cuda11x  # For CUDA 11.x")
        return

    # Create system
    system = qsolve.System.create(system_name, grid_points=grid_points)

    # Test 1: Eigenvalue solving
    logger.info("\n1. EIGENVALUE SOLVING (%s, %s points)", system_name, grid_points)
    logger.info("-" * 40)

    # CPU solve
    start = time.time()
    result_cpu = qsolve.solve_ground_state(system, method="sparse")
    time_cpu = time.time() - start
    logger.info("CPU (sparse): %.3fs, E₀ = %.6f", time_cpu, result_cpu.energy)

    # GPU solve
    gpu_system = GPUSystem(system)
    H_gpu = gpu_hamiltonian(gpu_system, sparse=True)

    start = time.time()
    E_gpu, psi_gpu = gpu_eigensolver(H_gpu, k=1, method="lanczos")
    time_gpu = time.time() - start

    if HAS_CUPY:
        E0 = float(cp.asnumpy(E_gpu[0]))
    else:
        E0 = float(E_gpu[0])

    logger.info("GPU (sparse): %.3fs, E₀ = %.6f", time_gpu, E0)
    logger.info("Speedup: %.1fx", time_cpu / time_gpu)

    # Test 2: Time evolution
    logger.info("\n2. TIME EVOLUTION (100 time steps)")
    logger.info("-" * 40)

    # Initial Gaussian wave packet
    x = system.grid.x
    psi0 = np.exp(-((x - 2) ** 2)) * np.exp(2j * x)
    psi0 /= np.sqrt(np.trapz(np.abs(psi0) ** 2, x))

    # CPU evolution
    start = time.time()
    from .evolution import evolve_wavefunction

    result_cpu = evolve_wavefunction(system, psi0, (0, 1), dt=0.01)
    time_cpu = time.time() - start
    logger.info("CPU: %.3fs", time_cpu)

    # GPU evolution
    start = time.time()
    result_gpu = gpu_evolve(system, psi0, (0, 1), dt=0.01)
    time_gpu = time.time() - start
    logger.info("GPU: %.3fs", time_gpu)
    logger.info("Speedup: %.1fx", time_cpu / time_gpu)

    logger.info("\n" + "=" * 60)
    logger.info("Total GPU speedup: %.1fx", time_cpu / time_gpu)
    logger.info("=" * 60)

    # Memory usage
    if HAS_CUPY:
        mempool = cp.get_default_memory_pool()
        logger.info("\nGPU Memory used: %.1f MB", mempool.used_bytes() / 1e6)
        logger.info("GPU Memory total: %.1f MB", mempool.total_bytes() / 1e6)


# JAX-specific optimizations
if HAS_JAX:

    @jit
    def jax_hamiltonian(x, dx, potential_params):
        """JAX-compatible Hamiltonian construction"""
        n = len(x)

        # Kinetic energy (tridiagonal)
        T = -0.5 / dx**2 * (-2 * jnp.eye(n) + jnp.eye(n, k=1) + jnp.eye(n, k=-1))

        # Potential (example: parameterized harmonic)
        V = jnp.diag(0.5 * potential_params[0] * x**2)

        return T + V

    @jit
    def jax_split_operator_step(psi, U_V_half, U_T):
        """JIT-compiled split-operator step"""
        psi = U_V_half * psi
        psi = jnp.fft.ifft(U_T * jnp.fft.fft(psi))
        psi = U_V_half * psi
        return psi

    def optimize_potential_jax(
        target_density, initial_params, system, learning_rate=0.1, n_steps=100
    ):
        """
        Use JAX to optimize potential parameters to match target density.

        This demonstrates automatic differentiation for quantum mechanics!
        """
        x = jnp.array(system.grid.x)
        dx = system.grid.dx

        @jit
        def loss_fn(params):
            """Loss function: difference from target density"""
            # Build Hamiltonian with current parameters
            H = jax_hamiltonian(x, dx, params)

            # Solve for ground state (simplified - use power iteration)
            psi = jnp.ones(len(x)) / jnp.sqrt(len(x))
            for _ in range(50):
                psi = H @ psi
                psi = psi / jnp.linalg.norm(psi)

            # Compute density
            density = jnp.abs(psi) ** 2

            # Loss: mean squared error
            return jnp.mean((density - target_density) ** 2)

        # Optimize using gradient descent
        params = initial_params
        losses = []

        for i in range(n_steps):
            loss = loss_fn(params)
            losses.append(float(loss))

            # Compute gradient
            grads = grad(loss_fn)(params)

            # Update parameters
            params = params - learning_rate * grads

            if i % 20 == 0:
                logger.info("Step %d: Loss = %.6f", i, loss)

        return params, losses


# Convenience function for users
def enable_gpu():
    """Check GPU availability and provide setup instructions"""
    
    # Use unified backend if available
    if HAS_UNIFIED:
        print_gpu_info()
        info = detect_gpu_capabilities()
        return info['cuda']['available'] or info['metal']['available']
    
    # Fallback to original implementation
    logger.info("GPU Acceleration Status")
    logger.info("=" * 40)

    # Check CUDA
    try:
        import subprocess

        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected")

            # Parse GPU info
            lines = result.stdout.split("\n")
            for line in lines:
                if "NVIDIA" in line and "MiB" in line:
                    logger.info("  %s", line.strip())
        else:
            logger.info("No NVIDIA GPU found")
    except Exception:
        logger.info("nvidia-smi not found")

    # Check CuPy
    if HAS_CUPY:
        logger.info("\nCuPy is installed")
        logger.info("  CuPy version: %s", cp.__version__)

        # Test GPU
        try:
            a = cp.array([1, 2, 3])
            b = cp.array([4, 5, 6])
            c = a + b
            logger.info("  GPU computation test: PASSED")
        except Exception as e:
            logger.info("  GPU computation test: FAILED - %s", e)
    else:
        logger.info("\nCuPy not installed")
        logger.info("  Install with: pip install cupy-cuda12x")

    # Check JAX
    if HAS_JAX:
        logger.info("\nJAX is installed")
        logger.info("  JAX version: %s", jax.__version__)
        logger.info("  Default device: %s", jax.default_backend())
    else:
        logger.info("\nJAX not installed")
        logger.info("  Install with: pip install jax[cuda12_pip]")

    logger.info("\n" + "=" * 40)

    return HAS_CUPY or HAS_JAX


if __name__ == "__main__":
    # Run GPU benchmark
    enable_gpu()

    if HAS_CUPY:
        logger.info("\nRunning GPU benchmark...")
        benchmark_gpu()
