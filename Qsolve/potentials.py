"""
quantum/potentials.py - Common quantum mechanical potentials
"""

import numpy as np
from typing import Callable
from numpy.typing import NDArray


def harmonic_oscillator(x: NDArray, omega: float = 1.0, center: float = 0.0) -> NDArray:
    """Harmonic oscillator potential.
    
    V(x) = 0.5 * m * ω² * (x - x₀)²
    
    Args:
        x: Position array
        omega: Angular frequency (default: 1.0)
        center: Center position (default: 0.0)
        
    Returns:
        Potential energy array
    """
    return 0.5 * omega**2 * (x - center)**2


def particle_box(x: NDArray) -> NDArray:
    """Particle in a box (infinite square well).
    
    V(x) = 0 (free particle)
    
    Note: Infinite walls are handled by the finite grid bounds.
    """
    return np.zeros_like(x)


def hydrogen_atom(x: NDArray, a0: float = 1.0) -> NDArray:
    """Hydrogen atom potential (1D approximation).
    
    V(x) = -e²/(4πε₀|x|) ≈ -1/|x|
    
    Args:
        x: Position array
        a0: Bohr radius for regularization
        
    Returns:
        Coulomb potential (regularized to avoid singularity)
    """
    # Regularize to avoid division by zero
    return -1.0 / (np.abs(x) + 0.1 * a0)


def double_well(
    x: NDArray, 
    barrier_height: float = 2.0, 
    well_separation: float = 2.0
) -> NDArray:
    """Double well potential.
    
    V(x) = V₀ * [(x/a)² - 1]²
    
    Args:
        x: Position array
        barrier_height: Height of central barrier
        well_separation: Distance between well minima
        
    Returns:
        Double well potential array
    """
    scaled_x = x / well_separation
    return barrier_height * (scaled_x**2 - 1)**2


def morse_potential(
    x: NDArray,
    D: float = 1.0,
    a: float = 1.0,
    x0: float = 0.0
) -> NDArray:
    """Morse potential (model for molecular vibrations).
    
    V(x) = D * [1 - exp(-a(x-x₀))]²
    
    Args:
        x: Position array
        D: Dissociation energy
        a: Width parameter
        x0: Equilibrium position
        
    Returns:
        Morse potential array
    """
    return D * (1 - np.exp(-a * (x - x0)))**2


def finite_well(
    x: NDArray,
    depth: float = 1.0,
    width: float = 2.0
) -> NDArray:
    """Finite square well potential.
    
    V(x) = -V₀ for |x| < a, 0 otherwise
    
    Args:
        x: Position array
        depth: Well depth (V₀)
        width: Half-width of well (a)
        
    Returns:
        Finite well potential array
    """
    V = np.zeros_like(x)
    V[np.abs(x) < width] = -depth
    return V


def linear_potential(x: NDArray, field_strength: float = 0.1) -> NDArray:
    """Linear potential (constant force/electric field).
    
    V(x) = F * x
    
    Args:
        x: Position array
        field_strength: Electric field strength
        
    Returns:
        Linear potential array
    """
    return field_strength * x


def gaussian_well(
    x: NDArray,
    depth: float = 1.0,
    width: float = 1.0,
    center: float = 0.0
) -> NDArray:
    """Gaussian well potential.
    
    V(x) = -V₀ * exp(-(x-x₀)²/2σ²)
    
    Args:
        x: Position array
        depth: Well depth
        width: Standard deviation
        center: Center position
        
    Returns:
        Gaussian well array
    """
    return -depth * np.exp(-0.5 * ((x - center) / width)**2)


def woods_saxon(
    x: NDArray,
    V0: float = 50.0,
    R: float = 1.2,
    a: float = 0.5
) -> NDArray:
    """Woods-Saxon potential (nuclear physics).
    
    V(x) = -V₀ / (1 + exp((|x|-R)/a))
    
    Args:
        x: Position array
        V0: Potential depth
        R: Nuclear radius
        a: Surface thickness
        
    Returns:
        Woods-Saxon potential array
    """
    return -V0 / (1 + np.exp((np.abs(x) - R) / a))


def poschl_teller(
    x: NDArray,
    lambda_param: float = 2.0,
    alpha: float = 1.0
) -> NDArray:
    """Pöschl-Teller potential (exactly solvable).
    
    V(x) = -V₀ / cosh²(αx)
    
    Args:
        x: Position array
        lambda_param: Depth parameter
        alpha: Width parameter
        
    Returns:
        Pöschl-Teller potential array
    """
    V0 = 0.5 * alpha**2 * lambda_param * (lambda_param + 1)
    return -V0 / np.cosh(alpha * x)**2


def periodic_cosine(
    x: NDArray,
    amplitude: float = 0.5,
    period: float = 2.0
) -> NDArray:
    """Periodic cosine potential (Mathieu equation).
    
    V(x) = V₀ * cos(2πx/L)
    
    Args:
        x: Position array
        amplitude: Potential amplitude
        period: Spatial period
        
    Returns:
        Periodic potential array
    """
    return amplitude * np.cos(2 * np.pi * x / period)


# Composite potentials

def harmonic_plus_quartic(
    x: NDArray,
    omega: float = 1.0,
    lambda_4: float = 0.1
) -> NDArray:
    """Anharmonic oscillator: harmonic + quartic term.
    
    V(x) = 0.5*ω²x² + λx⁴
    """
    return 0.5 * omega**2 * x**2 + lambda_4 * x**4


def double_well_asymmetric(
    x: NDArray,
    asymmetry: float = 0.1
) -> NDArray:
    """Asymmetric double well.
    
    Adds linear term to break symmetry.
    """
    return double_well(x) + asymmetry * x


# Registry of all available potentials
POTENTIALS = {
    'harmonic': harmonic_oscillator,
    'box': particle_box,
    'hydrogen': hydrogen_atom,
    'double_well': double_well,
    'morse': morse_potential,
    'finite_well': finite_well,
    'linear': linear_potential,
    'gaussian_well': gaussian_well,
    'woods_saxon': woods_saxon,
    'poschl_teller': poschl_teller,
    'periodic': periodic_cosine,
    'anharmonic': harmonic_plus_quartic,
    'asymmetric_double_well': double_well_asymmetric,
}


def list_potentials() -> list[str]:
    """List all available potential names."""
    return list(POTENTIALS.keys())


def get_potential(name: str) -> Callable:
    """Get potential function by name."""
    if name not in POTENTIALS:
        available = ", ".join(POTENTIALS.keys())
        raise ValueError(f"Unknown potential '{name}'. Available: {available}")
    return POTENTIALS[name]
