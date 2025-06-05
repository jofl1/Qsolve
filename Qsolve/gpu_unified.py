"""
qsolve/gpu_unified.py - Unified GPU acceleration for NVIDIA CUDA and Apple Metal.

Provides transparent GPU acceleration across different hardware platforms.
"""

import numpy as np
from typing import Optional, Union, Callable, Literal, Any
import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Platform detection
import platform
PLATFORM = platform.system().lower()
IS_MACOS = PLATFORM == "darwin"
IS_LINUX = PLATFORM == "linux"
IS_WINDOWS = PLATFORM == "windows"

# Try importing GPU libraries
HAS_CUPY = False
HAS_MLX = False
HAS_JAX = False
HAS_MPS = False  # PyTorch MPS backend

# NVIDIA CUDA support via CuPy
try:
    import cupy as cp
    HAS_CUPY = True
    logger.info("CuPy available - NVIDIA GPU acceleration enabled.")
except ImportError:
    cp = None
    logger.debug("CuPy not found. Install with 'pip install cupy-cuda12x'.")

# Apple Metal support via MLX
if IS_MACOS:
    try:
        import mlx
        import mlx.core as mx
        import mlx.nn as nn
        HAS_MLX = True
        logger.info("MLX available - Apple Metal GPU acceleration enabled.")
    except ImportError:
        mx = None
        logger.debug("MLX not found. Install with 'pip install mlx'.")
    
    # Also check PyTorch MPS backend
    try:
        import torch
        if torch.backends.mps.is_available():
            HAS_MPS = True
            logger.info("PyTorch MPS backend available for Metal GPU.")
    except ImportError:
        pass

# JAX support (works on both CUDA and Metal via Metal plugin)
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    HAS_JAX = True
    
    # Configure JAX backend
    if IS_MACOS and HAS_MLX:
        # Try to use Metal backend if available
        try:
            jax.config.update("jax_platform_name", "metal")
            logger.info("JAX configured for Metal backend.")
        except:
            jax.config.update("jax_platform_name", "cpu")
            logger.debug("JAX Metal backend not available, using CPU.")
    elif HAS_CUPY:
        jax.config.update("jax_platform_name", "gpu")
        logger.info("JAX configured for CUDA backend.")
except ImportError:
    HAS_JAX = False
    jnp = np


class GPUBackend(ABC):
    """Abstract base class for GPU backends"""
    
    @abstractmethod
    def array(self, data, dtype=None):
        """Create GPU array from data"""
        pass
    
    @abstractmethod
    def to_cpu(self, gpu_array):
        """Transfer GPU array to CPU"""
        pass
    
    @abstractmethod
    def zeros(self, shape, dtype=np.float64):
        """Create zero array on GPU"""
        pass
    
    @abstractmethod
    def ones(self, shape, dtype=np.float64):
        """Create ones array on GPU"""
        pass
    
    @abstractmethod
    def matmul(self, a, b):
        """Matrix multiplication"""
        pass
    
    @abstractmethod
    def eigh(self, matrix):
        """Eigenvalue decomposition for hermitian matrices"""
        pass
    
    @abstractmethod
    def fft(self, array):
        """Fast Fourier Transform"""
        pass
    
    @abstractmethod
    def ifft(self, array):
        """Inverse Fast Fourier Transform"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name"""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass


class CuPyBackend(GPUBackend):
    """NVIDIA CUDA backend using CuPy"""
    
    def __init__(self):
        if not HAS_CUPY:
            raise RuntimeError("CuPy not installed")
        self.cp = cp
    
    def array(self, data, dtype=None):
        return cp.asarray(data, dtype=dtype)
    
    def to_cpu(self, gpu_array):
        return cp.asnumpy(gpu_array)
    
    def zeros(self, shape, dtype=np.float64):
        return cp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=np.float64):
        return cp.ones(shape, dtype=dtype)
    
    def matmul(self, a, b):
        return cp.matmul(a, b)
    
    def eigh(self, matrix):
        return cp.linalg.eigh(matrix)
    
    def fft(self, array):
        return cp.fft.fft(array)
    
    def ifft(self, array):
        return cp.fft.ifft(array)
    
    @property
    def name(self):
        return "CUDA (CuPy)"
    
    @property
    def is_available(self):
        return HAS_CUPY


class MLXBackend(GPUBackend):
    """Apple Metal backend using MLX"""
    
    def __init__(self):
        if not HAS_MLX:
            raise RuntimeError("MLX not installed")
        self.mx = mx
    
    def array(self, data, dtype=None):
        # MLX uses different dtype notation
        if dtype is not None:
            if dtype == np.complex128 or dtype == np.complex64:
                # MLX doesn't have complex types, we'll handle this specially
                real = mx.array(np.real(data))
                imag = mx.array(np.imag(data))
                return (real, imag)  # Return tuple for complex
            dtype_map = {
                np.float64: mx.float32,  # MLX primarily uses float32
                np.float32: mx.float32,
                np.int64: mx.int32,
                np.int32: mx.int32,
            }
            mlx_dtype = dtype_map.get(dtype, mx.float32)
            return mx.array(data, dtype=mlx_dtype)
        return mx.array(data)
    
    def to_cpu(self, gpu_array):
        if isinstance(gpu_array, tuple):  # Complex number handling
            real, imag = gpu_array
            return np.array(real) + 1j * np.array(imag)
        return np.array(gpu_array)
    
    def zeros(self, shape, dtype=np.float64):
        if dtype in [np.complex64, np.complex128]:
            return (mx.zeros(shape), mx.zeros(shape))
        return mx.zeros(shape)
    
    def ones(self, shape, dtype=np.float64):
        if dtype in [np.complex64, np.complex128]:
            return (mx.ones(shape), mx.zeros(shape))
        return mx.ones(shape)
    
    def matmul(self, a, b):
        if isinstance(a, tuple) and isinstance(b, tuple):
            # Complex matrix multiplication
            a_real, a_imag = a
            b_real, b_imag = b
            real = mx.matmul(a_real, b_real) - mx.matmul(a_imag, b_imag)
            imag = mx.matmul(a_real, b_imag) + mx.matmul(a_imag, b_real)
            return (real, imag)
        elif isinstance(a, tuple):
            # Complex @ real
            a_real, a_imag = a
            return (mx.matmul(a_real, b), mx.matmul(a_imag, b))
        elif isinstance(b, tuple):
            # Real @ complex
            b_real, b_imag = b
            return (mx.matmul(a, b_real), mx.matmul(a, b_imag))
        return mx.matmul(a, b)
    
    def eigh(self, matrix):
        # MLX doesn't have built-in eigh, fall back to NumPy
        if isinstance(matrix, tuple):
            # Handle complex matrices
            real, imag = matrix
            cpu_matrix = np.array(real) + 1j * np.array(imag)
        else:
            cpu_matrix = np.array(matrix)
        
        E, V = np.linalg.eigh(cpu_matrix)
        
        # Convert back to MLX arrays
        E_mx = mx.array(E)
        if np.iscomplexobj(V):
            V_mx = (mx.array(V.real), mx.array(V.imag))
        else:
            V_mx = mx.array(V)
        
        return E_mx, V_mx
    
    def fft(self, array):
        # MLX has FFT support
        if isinstance(array, tuple):
            real, imag = array
            # Convert to numpy for FFT
            cpu_array = np.array(real) + 1j * np.array(imag)
            result = np.fft.fft(cpu_array)
            return (mx.array(result.real), mx.array(result.imag))
        else:
            # Real FFT
            return mx.fft.fft(array)
    
    def ifft(self, array):
        if isinstance(array, tuple):
            real, imag = array
            # Convert to numpy for IFFT
            cpu_array = np.array(real) + 1j * np.array(imag)
            result = np.fft.ifft(cpu_array)
            return (mx.array(result.real), mx.array(result.imag))
        else:
            return mx.fft.ifft(array)
    
    @property
    def name(self):
        return "Metal (MLX)"
    
    @property
    def is_available(self):
        return HAS_MLX


class NumpyBackend(GPUBackend):
    """CPU fallback using NumPy"""
    
    def array(self, data, dtype=None):
        return np.asarray(data, dtype=dtype)
    
    def to_cpu(self, array):
        return array  # Already on CPU
    
    def zeros(self, shape, dtype=np.float64):
        return np.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=np.float64):
        return np.ones(shape, dtype=dtype)
    
    def matmul(self, a, b):
        return np.matmul(a, b)
    
    def eigh(self, matrix):
        return np.linalg.eigh(matrix)
    
    def fft(self, array):
        return np.fft.fft(array)
    
    def ifft(self, array):
        return np.fft.ifft(array)
    
    @property
    def name(self):
        return "CPU (NumPy)"
    
    @property
    def is_available(self):
        return True


class UnifiedGPUSystem:
    """Unified GPU-accelerated quantum system supporting multiple backends"""
    
    def __init__(self, system, backend: Optional[Literal['auto', 'cuda', 'metal', 'cpu']] = 'auto'):
        self.cpu_system = system
        
        # Select backend
        if backend == 'auto':
            if HAS_MLX and IS_MACOS:
                self.backend = MLXBackend()
            elif HAS_CUPY:
                self.backend = CuPyBackend()
            else:
                self.backend = NumpyBackend()
        elif backend == 'cuda':
            if not HAS_CUPY:
                logger.warning("CUDA backend requested but CuPy not available, falling back to CPU")
                self.backend = NumpyBackend()
            else:
                self.backend = CuPyBackend()
        elif backend == 'metal':
            if not HAS_MLX:
                logger.warning("Metal backend requested but MLX not available, falling back to CPU")
                self.backend = NumpyBackend()
            else:
                self.backend = MLXBackend()
        else:  # cpu
            self.backend = NumpyBackend()
        
        logger.info(f"Using {self.backend.name} backend")
        
        # Transfer data to GPU
        self.grid_x = self.backend.array(system.grid.x)
        self.grid_dx = system.grid.dx
        self.n_points = system.grid.points
        
        # Precompute potential on GPU
        V_cpu = system.potential(system.grid.x)
        self.V = self.backend.array(V_cpu)
    
    @property
    def xp(self):
        """Array module for compatibility"""
        if isinstance(self.backend, CuPyBackend):
            return cp
        elif isinstance(self.backend, MLXBackend):
            return mx
        else:
            return np
    
    def build_hamiltonian(self, sparse=False):
        """Build Hamiltonian matrix on GPU"""
        n = self.n_points
        dx2 = self.grid_dx**2
        
        if sparse and isinstance(self.backend, CuPyBackend):
            # Use CuPy sparse matrices
            from cupyx.scipy.sparse import diags as gpu_diags
            
            kinetic = -0.5 / dx2 * gpu_diags(
                [1, -2, 1], [-1, 0, 1], shape=(n, n), format="csr"
            )
            potential = gpu_diags(self.V, 0, format="csr")
            return kinetic + potential
        
        elif sparse and isinstance(self.backend, NumpyBackend):
            # Use SciPy sparse matrices
            from scipy.sparse import diags
            
            kinetic = -0.5 / dx2 * diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
            potential = diags(self.backend.to_cpu(self.V), 0)
            return kinetic + potential
        
        else:
            # Dense matrix construction
            T = self.backend.zeros((n, n))
            
            # Kinetic energy - tridiagonal
            if isinstance(self.backend, MLXBackend):
                # MLX doesn't have fill_diagonal, so we use indexing
                if isinstance(T, tuple):  # Complex
                    T_real, T_imag = T
                    T_real = mx.scatter(T_real, mx.arange(n), mx.arange(n), -2.0)
                    if n > 1:
                        T_real = mx.scatter(T_real, mx.arange(n-1), mx.arange(1, n), 1.0)
                        T_real = mx.scatter(T_real, mx.arange(1, n), mx.arange(n-1), 1.0)
                    T = (T_real * (-0.5 / dx2), T_imag * 0)
                else:
                    # Create indices for diagonal
                    diag_indices = mx.arange(n)
                    T = mx.scatter(T, diag_indices, diag_indices, -2.0)
                    if n > 1:
                        # Off-diagonals
                        T = mx.scatter(T, mx.arange(n-1), mx.arange(1, n), 1.0)
                        T = mx.scatter(T, mx.arange(1, n), mx.arange(n-1), 1.0)
                    T = T * (-0.5 / dx2)
            else:
                # NumPy/CuPy style
                xp = self.xp
                xp.fill_diagonal(T, -2.0)
                if n > 1:
                    xp.fill_diagonal(T[1:], 1.0)
                    xp.fill_diagonal(T[:, 1:], 1.0)
                T *= -0.5 / dx2
            
            # Potential energy - diagonal
            if isinstance(self.backend, MLXBackend):
                V_diag = self.backend.zeros((n, n))
                if isinstance(self.V, tuple):
                    # Complex potential
                    V_real, V_imag = self.V
                    V_diag_real = mx.scatter(mx.zeros((n, n)), mx.arange(n), mx.arange(n), V_real)
                    V_diag_imag = mx.scatter(mx.zeros((n, n)), mx.arange(n), mx.arange(n), V_imag)
                    V_diag = (V_diag_real, V_diag_imag)
                else:
                    diag_indices = mx.arange(n)
                    V_diag = mx.scatter(V_diag, diag_indices, diag_indices, self.V)
            else:
                V_diag = self.xp.diag(self.V)
            
            # Add kinetic and potential
            if isinstance(T, tuple) and isinstance(V_diag, tuple):
                T_real, T_imag = T
                V_real, V_imag = V_diag
                return (T_real + V_real, T_imag + V_imag)
            elif isinstance(T, tuple):
                T_real, T_imag = T
                return (T_real + V_diag, T_imag)
            elif isinstance(V_diag, tuple):
                V_real, V_imag = V_diag
                return (T + V_real, V_imag)
            else:
                return T + V_diag


class UnifiedSplitOperator:
    """Split-operator propagator supporting multiple GPU backends"""
    
    def __init__(self, gpu_system: UnifiedGPUSystem, dt: float):
        self.system = gpu_system
        self.backend = gpu_system.backend
        self.dt = dt
        
        # Precompute evolution operators
        n = gpu_system.n_points
        dx = gpu_system.grid_dx
        
        # Momentum space grid
        if isinstance(self.backend, CuPyBackend):
            k = 2 * np.pi * cp.fft.fftfreq(n, dx)
        elif isinstance(self.backend, MLXBackend):
            k_cpu = 2 * np.pi * np.fft.fftfreq(n, dx)
            k = mx.array(k_cpu)
        else:
            k = 2 * np.pi * np.fft.fftfreq(n, dx)
        
        # Kinetic evolution operator
        T_k = 0.5 * k**2
        if isinstance(self.backend, MLXBackend):
            # MLX complex exponential
            angle = -T_k * dt
            self.U_T = (mx.cos(angle), mx.sin(angle))  # exp(i*angle) = cos + i*sin
        else:
            self.U_T = np.exp(-1j * T_k * dt)
            if isinstance(self.backend, CuPyBackend):
                self.U_T = cp.asarray(self.U_T)
        
        # Potential evolution operator
        if isinstance(self.backend, MLXBackend):
            if isinstance(gpu_system.V, tuple):
                V_real, V_imag = gpu_system.V
                # exp(-0.5j * V * dt) for complex V
                angle = -0.5 * dt
                self.U_V_half = (
                    mx.cos(angle * V_real) * mx.exp(0.5 * angle * V_imag),
                    -mx.sin(angle * V_real) * mx.exp(0.5 * angle * V_imag)
                )
            else:
                angle = -0.5 * gpu_system.V * dt
                self.U_V_half = (mx.cos(angle), mx.sin(angle))
        else:
            self.U_V_half = np.exp(-0.5j * gpu_system.V * dt)
            if isinstance(self.backend, CuPyBackend):
                self.U_V_half = cp.asarray(self.U_V_half)
    
    def step(self, psi):
        """Single time evolution step"""
        # Apply half potential
        if isinstance(self.backend, MLXBackend):
            psi = self._complex_multiply(self.U_V_half, psi)
        else:
            psi = self.U_V_half * psi
        
        # FFT to momentum space
        psi_k = self.backend.fft(psi)
        
        # Apply kinetic evolution
        if isinstance(self.backend, MLXBackend):
            psi_k = self._complex_multiply(self.U_T, psi_k)
        else:
            psi_k = self.U_T * psi_k
        
        # IFFT back to position space
        psi = self.backend.ifft(psi_k)
        
        # Apply half potential again
        if isinstance(self.backend, MLXBackend):
            psi = self._complex_multiply(self.U_V_half, psi)
        else:
            psi = self.U_V_half * psi
        
        return psi
    
    def _complex_multiply(self, a, b):
        """Complex multiplication for MLX backend"""
        if isinstance(a, tuple) and isinstance(b, tuple):
            a_real, a_imag = a
            b_real, b_imag = b
            real = a_real * b_real - a_imag * b_imag
            imag = a_real * b_imag + a_imag * b_real
            return (real, imag)
        elif isinstance(a, tuple):
            a_real, a_imag = a
            return (a_real * b, a_imag * b)
        elif isinstance(b, tuple):
            b_real, b_imag = b
            return (a * b_real, a * b_imag)
        else:
            return a * b


def unified_gpu_evolve(system, psi0, time_span, dt=0.01, backend='auto'):
    """Time evolution using unified GPU backend"""
    
    # Create GPU system
    gpu_system = UnifiedGPUSystem(system, backend=backend)
    
    # Transfer initial state to GPU
    psi_gpu = gpu_system.backend.array(psi0, dtype=np.complex128)
    
    # Setup time array
    times = np.arange(time_span[0], time_span[1], dt)
    n_steps = len(times)
    
    # Create propagator
    propagator = UnifiedSplitOperator(gpu_system, dt)
    
    # Storage
    wavefunctions = np.zeros((n_steps, len(psi0)), dtype=complex)
    wavefunctions[0] = psi0
    
    logger.info(f"Evolution: {n_steps} steps on {gpu_system.backend.name}")
    start_time = time.time()
    
    # Time evolution
    for i in range(1, n_steps):
        psi_gpu = propagator.step(psi_gpu)
        
        # Store results periodically
        if i % 10 == 0 or i == n_steps - 1:
            wavefunctions[i] = gpu_system.backend.to_cpu(psi_gpu)
    
    elapsed = time.time() - start_time
    logger.info(f"Evolution completed in {elapsed:.3f}s ({int(n_steps/elapsed)} steps/s)")
    
    # Create result
    from .evolution import EvolutionResult
    return EvolutionResult(times=times, wavefunctions=wavefunctions, system=system)


def detect_gpu_capabilities():
    """Detect available GPU acceleration options"""
    
    info = {
        'cuda': {
            'available': False,
            'device_count': 0,
            'devices': [],
            'library': None
        },
        'metal': {
            'available': False,
            'devices': [],
            'library': None
        },
        'cpu': {
            'available': True,
            'cores': None
        }
    }
    
    # Check CUDA
    if HAS_CUPY:
        info['cuda']['available'] = True
        info['cuda']['library'] = f"CuPy {cp.__version__}"
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            info['cuda']['device_count'] = device_count
            
            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                info['cuda']['devices'].append({
                    'name': props['name'].decode(),
                    'compute_capability': f"{props['major']}.{props['minor']}",
                    'memory': props['totalGlobalMem'] / 1e9  # GB
                })
        except:
            pass
    
    # Check Metal
    if IS_MACOS:
        if HAS_MLX:
            info['metal']['available'] = True
            info['metal']['library'] = f"MLX"
            # MLX doesn't provide detailed device info yet
            info['metal']['devices'].append({
                'name': 'Apple Silicon GPU',
                'unified_memory': True
            })
        elif HAS_MPS:
            info['metal']['available'] = True
            info['metal']['library'] = "PyTorch MPS"
            info['metal']['devices'].append({
                'name': 'Metal Performance Shaders',
                'unified_memory': True
            })
    
    # CPU info
    try:
        import os
        info['cpu']['cores'] = os.cpu_count()
    except:
        pass
    
    return info


def print_gpu_info():
    """Print detailed GPU information"""
    info = detect_gpu_capabilities()
    
    print("=" * 60)
    print("GPU ACCELERATION CAPABILITIES")
    print("=" * 60)
    
    # CUDA/NVIDIA
    print("\nNVIDIA CUDA:")
    if info['cuda']['available']:
        print(f"  Status: Available ({info['cuda']['library']})")
        print(f"  Devices: {info['cuda']['device_count']}")
        for i, dev in enumerate(info['cuda']['devices']):
            print(f"    [{i}] {dev['name']}")
            print(f"        Compute: {dev['compute_capability']}")
            print(f"        Memory: {dev['memory']:.1f} GB")
    else:
        print("  Status: Not available")
        print("  Install: pip install cupy-cuda12x")
    
    # Metal/Apple
    print("\nApple Metal:")
    if info['metal']['available']:
        print(f"  Status: Available ({info['metal']['library']})")
        for dev in info['metal']['devices']:
            print(f"    {dev['name']}")
            if 'unified_memory' in dev:
                print("        Unified Memory Architecture")
    else:
        if IS_MACOS:
            print("  Status: Not available")
            print("  Install: pip install mlx")
        else:
            print("  Status: Not applicable (requires macOS)")
    
    # CPU fallback
    print("\nCPU Fallback:")
    print("  Status: Always available")
    if info['cpu']['cores']:
        print(f"  Cores: {info['cpu']['cores']}")
    
    print("\n" + "=" * 60)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if info['cuda']['available'] and info['metal']['available']:
        print("  Multiple GPU options available!")
        print("  Use backend='auto' for automatic selection")
    elif info['cuda']['available']:
        print("  NVIDIA GPU detected - will use CUDA acceleration")
    elif info['metal']['available']:
        print("  Apple GPU detected - will use Metal acceleration")
    else:
        print("  No GPU detected - using optimized CPU backend")
        if IS_MACOS:
            print("  Consider: pip install mlx")
        else:
            print("  Consider: pip install cupy-cuda12x")
    
    print("=" * 60)


# Update the original enable_gpu function
def enable_gpu():
    """Check GPU availability and provide setup instructions"""
    print_gpu_info()
    info = detect_gpu_capabilities()
    return info['cuda']['available'] or info['metal']['available']


# Backwards compatibility exports
GPUSystem = UnifiedGPUSystem
gpu_evolve = unified_gpu_evolve


if __name__ == "__main__":
    # Test GPU detection
    enable_gpu()