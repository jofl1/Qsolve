"""
Qsolve - A modern quantum mechanics library

Quick start:
    >>> import Qsolve
    >>> result = Qsolve.solve("hydrogen")
    >>> print(f"Ground state energy: {result.energy:.6f}")
    >>> result.plot()
"""

__version__ = "0.1.0"

# Core imports
from .Core import System, Grid, Grid2D, Grid3D, Result
from .Solvers import (
    solve_ground_state,
    solve_eigenstates,
    solve,
    build_hamiltonian,
)
from .potentials import (
    harmonic_oscillator,
    particle_box,
    hydrogen_atom,
    double_well,
    morse_potential,
    list_potentials,
)

# New features imports
from .evolution import (
    evolve_wavefunction, 
    gaussian_wave_packet,
    coherent_state,
    simulate_tunneling,
    EvolutionResult
)

from .solvers.davidson import (
    davidson,
    solve_davidson,
    benchmark_davidson_vs_eigsh
)

from .gpu import (
    enable_gpu,
    gpu_evolve,
    benchmark_gpu,
    GPUSystem,
    HAS_CUPY
)

# Import unified GPU capabilities if available
try:
    from .gpu_unified import (
        UnifiedGPUSystem,
        unified_gpu_evolve,
        detect_gpu_capabilities,
        print_gpu_info,
        HAS_MLX,
    )
    HAS_UNIFIED_GPU = True
except ImportError:
    HAS_MLX = False
    HAS_UNIFIED_GPU = False

# Two-electron systems
from .two_electron import TwoElectronSystem
from .visualisation import plot_two_electron_result

# Convenience imports for common use cases
def solve_quick(system_name: str = "harmonic", **kwargs) -> Result:
    """Quick solve for common systems.
    
    Examples:
        >>> result = Qsolve.solve_quick("hydrogen")
        >>> result = Qsolve.solve_quick("double_well", n_states=3)
    """
    # Extract solver-specific kwargs
    n_states = kwargs.pop('n_states', 1)
    method = kwargs.pop('method', 'auto')
    tolerance = kwargs.pop('tolerance', 1e-10)
    max_iterations = kwargs.pop('max_iterations', 1000)
    verbose = kwargs.pop('verbose', False)
    
    # Create system with remaining kwargs (grid_points, bounds, etc.)
    system = System.create(system_name, **kwargs)
    
    # Create solver kwargs
    solver_kwargs = {
        'method': method,
        'tolerance': tolerance,
        'max_iterations': max_iterations,
        'verbose': verbose
    }
    
    if n_states == 1:
        return solve_ground_state(system, **solver_kwargs)
    else:
        return solve_eigenstates(system, n_states=n_states, **solver_kwargs)


# Aliases for convenience
solve_system = solve_quick


def load_result(filename: str) -> Result:
    """Load a saved quantum result from file.
    
    Args:
        filename: Path to saved .npz file
        
    Returns:
        Result object with loaded data
        
    Examples:
        >>> result = Qsolve.load_result("hydrogen_states.npz")
        >>> print(result.energy)
    """
    return Result.load(filename)


# Package metadata
__all__ = [
    # Core classes
    "System",
    "Grid",
    "Grid2D",
    "Grid3D",
    "Result",
    "TwoElectronSystem",
    # Solver functions
    "solve",
    "solve_ground_state",
    "solve_eigenstates",
    "build_hamiltonian",
    # Potentials
    "harmonic_oscillator",
    "particle_box",
    "hydrogen_atom",
    "double_well",
    "morse_potential",
    "list_potentials",
    # Convenience
    "solve_quick",
    "solve_system",
    "load_result",
    # New features
    "evolve_wavefunction", 
    "gaussian_wave_packet", 
    "simulate_tunneling",
    "davidson", 
    "solve_davidson", 
    "enable_gpu", 
    "gpu_evolve",
    "EvolutionResult",
    "benchmark_davidson_vs_eigsh",
    "benchmark_gpu",
    "GPUSystem",
    "coherent_state",
    "HAS_CUPY",
    "plot_two_electron_result",
    # Unified GPU support
    "HAS_UNIFIED_GPU",
    "HAS_MLX",
]

# Add unified GPU exports if available
if HAS_UNIFIED_GPU:
    __all__.extend([
        "UnifiedGPUSystem",
        "unified_gpu_evolve",
        "detect_gpu_capabilities",
        "print_gpu_info",
    ])


def info():
    """Print package information."""
    print(f"Qsolve v{__version__}")
    print("A modern quantum mechanics library")
    print(f"\nAvailable systems: {', '.join(list_potentials())}")
    print("\nExample usage:")
    print("  import Qsolve")
    print("  result = Qsolve.solve_quick('hydrogen')")
    print("  result.plot()")
