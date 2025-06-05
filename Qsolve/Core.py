"""
quantum/core.py - Core data structures and system definitions
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class Grid:
    """Spatial grid for quantum calculations.
    
    Examples:
        >>> grid = Grid(points=101, bounds=(-10, 10))
        >>> grid.dx  # Grid spacing
        0.2
    """
    points: int = 101
    bounds: tuple[float, float] = (-10.0, 10.0)
    
    def __post_init__(self):
        # Validate number of points
        if not isinstance(self.points, int):
            raise TypeError(f"Grid points must be an integer, got {type(self.points).__name__}")
        if self.points < 10:
            raise ValueError(f"Grid must have at least 10 points, got {self.points}")
        if self.points > 100000:
            raise ValueError(f"Grid size {self.points} is too large (max 100,000). "
                           "Consider using sparse methods or reducing grid size.")
        
        # Validate bounds
        if len(self.bounds) != 2:
            raise ValueError(f"Bounds must be a tuple of (xmin, xmax), got {self.bounds}")
        if self.bounds[1] <= self.bounds[0]:
            raise ValueError(f"Invalid bounds: xmin={self.bounds[0]} must be less than xmax={self.bounds[1]}")
        
        # Check for reasonable grid span
        span = self.bounds[1] - self.bounds[0]
        if span < 0.01:
            raise ValueError(f"Grid span too small ({span:.3e}). Minimum recommended span is 0.01")
        
        self.x = np.linspace(self.bounds[0], self.bounds[1], self.points)
        self.dx = self.x[1] - self.x[0]

    def coordinates(self) -> Tuple[NDArray]:
        """Return coordinate arrays for the grid"""
        return (self.x,)

    @property
    def dxs(self) -> Tuple[float]:
        """Spacing along each dimension"""
        return (self.dx,)

    @property
    def shape(self) -> Tuple[int]:
        return (self.points,)
    
    @property
    def size(self) -> float:
        """Total size of the grid"""
        return self.bounds[1] - self.bounds[0]
    
    def __repr__(self) -> str:
        return f"Grid(points={self.points}, bounds={self.bounds}, dx={self.dx:.3f})"


@dataclass
class Grid2D:
    """Two-dimensional uniform grid"""

    shape: Tuple[int, int] = (64, 64)
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-10.0, 10.0), (-10.0, 10.0))

    def __post_init__(self):
        for n in self.shape:
            if n < 4:
                raise ValueError("Grid dimensions must be at least 4")
        (x0, x1), (y0, y1) = self.bounds
        self.x = np.linspace(x0, x1, self.shape[0])
        self.y = np.linspace(y0, y1, self.shape[1])
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.points = self.shape[0] * self.shape[1]

    def coordinates(self) -> Tuple[NDArray, NDArray]:
        return (self.X, self.Y)

    @property
    def dxs(self) -> Tuple[float, float]:
        return (self.dx, self.dy)

    @property
    def size(self) -> Tuple[float, float]:
        return (self.bounds[0][1] - self.bounds[0][0], self.bounds[1][1] - self.bounds[1][0])

    def __repr__(self) -> str:
        return (
            f"Grid2D(shape={self.shape}, bounds={self.bounds}, dx={self.dx:.3f}, dy={self.dy:.3f})"
        )


@dataclass
class Grid3D:
    """Three-dimensional uniform grid"""

    shape: Tuple[int, int, int] = (32, 32, 32)
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
    )

    def __post_init__(self):
        for n in self.shape:
            if n < 4:
                raise ValueError("Grid dimensions must be at least 4")
        (x0, x1), (y0, y1), (z0, z1) = self.bounds
        self.x = np.linspace(x0, x1, self.shape[0])
        self.y = np.linspace(y0, y1, self.shape[1])
        self.z = np.linspace(z0, z1, self.shape[2])
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.points = self.shape[0] * self.shape[1] * self.shape[2]

    def coordinates(self) -> Tuple[NDArray, NDArray, NDArray]:
        return (self.X, self.Y, self.Z)

    @property
    def dxs(self) -> Tuple[float, float, float]:
        return (self.dx, self.dy, self.dz)

    @property
    def size(self) -> Tuple[float, float, float]:
        return (
            self.bounds[0][1] - self.bounds[0][0],
            self.bounds[1][1] - self.bounds[1][0],
            self.bounds[2][1] - self.bounds[2][0],
        )

    def __repr__(self) -> str:
        return (
            f"Grid3D(shape={self.shape}, bounds={self.bounds}, "
            f"dx={self.dx:.3f}, dy={self.dy:.3f}, dz={self.dz:.3f})"
        )


@dataclass
class System:
    """Quantum system definition.
    
    Args:
        grid: Spatial grid for calculations
        potential: Function V(x) defining the potential energy
        mass: Particle mass (default: 1.0 in atomic units)
        hbar: Reduced Planck constant (default: 1.0 in atomic units)
        
    Examples:
        >>> # Harmonic oscillator
        >>> system = System(
        ...     grid=Grid(points=201),
        ...     potential=lambda x: 0.5 * x**2
        ... )
    """
    grid: object
    potential: Callable
    mass: float = 1.0
    hbar: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate potential function for the provided grid
        coords = self.grid.coordinates()
        try:
            V = self.potential(*coords)
            if not isinstance(V, np.ndarray) or V.shape != coords[0].shape:
                raise ValueError("Potential function must return array matching grid shape")
        except Exception as e:
            raise ValueError(f"Invalid potential function: {e}")
    
    @classmethod
    def create(cls, name: str, **kwargs) -> "System":
        """Factory method to create common quantum systems.
        
        Args:
            name: System name ('harmonic', 'box', 'hydrogen', etc.)
            **kwargs: Additional parameters
            
        Returns:
            Configured System instance
            
        Examples:
            >>> system = System.create('harmonic', grid_points=201)
            >>> system = System.create('hydrogen', box_size=40.0)
        """
        # Import here to avoid circular dependency
        from .potentials import POTENTIALS
        
        if name not in POTENTIALS:
            available = sorted(POTENTIALS.keys())
            suggestions = []
            # Simple fuzzy matching for common typos
            for pot_name in available:
                if name.lower() in pot_name.lower() or pot_name.lower() in name.lower():
                    suggestions.append(pot_name)
            
            error_msg = f"Unknown system '{name}'.\n"
            if suggestions:
                error_msg += f"Did you mean: {', '.join(suggestions)}?\n"
            error_msg += f"Available systems: {', '.join(available)}"
            raise ValueError(error_msg)
        
        # Extract grid parameters
        grid_params = {}
        if 'grid_points' in kwargs:
            grid_params['points'] = kwargs.pop('grid_points')
        if 'box_size' in kwargs:
            size = kwargs.pop('box_size')
            grid_params['bounds'] = (-size/2, size/2)
        if 'bounds' in kwargs:
            grid_params['bounds'] = kwargs.pop('bounds')
        
        # Create grid
        grid = Grid(**grid_params)
        
        # Get potential function
        potential_func = POTENTIALS[name]

        # Extract potential-specific parameters
        import inspect

        sig = inspect.signature(potential_func)
        pot_params = {}
        for p in sig.parameters.values():
            if p.name == "x":
                continue
            if p.name in kwargs:
                pot_params[p.name] = kwargs.pop(p.name)

        if pot_params:
            def bound_potential(x, func=potential_func, params=pot_params):
                return func(x, **params)
            potential_func = bound_potential
        
        # Create system
        return cls(
            grid=grid,
            potential=potential_func,
            metadata={'name': name},
            **kwargs
        )
    
    @property
    def name(self) -> str:
        """System name if created with factory method"""
        return self.metadata.get('name', 'custom')
    
    def info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'name': self.name,
            'grid_points': self.grid.points,
            'grid_bounds': self.grid.bounds,
            'grid_spacing': self.grid.dxs,
            'mass': self.mass,
            'hbar': self.hbar,
        }
    
    def __repr__(self) -> str:
        return f"System(name='{self.name}', grid={self.grid})"


@dataclass
class Result:
    """Container for solver results.
    
    Attributes:
        energies: Array of energy eigenvalues
        wavefunctions: Array of wavefunctions (columns are states)
        system: The quantum system that was solved
        info: Additional solver information
    """
    energies: NDArray
    wavefunctions: NDArray
    system: System
    info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_states(self) -> int:
        """Number of computed states"""
        return len(self.energies)
    
    @property
    def energy(self) -> float:
        """Ground state energy (convenience property)"""
        return self.energies[0]
    
    @property
    def wavefunction(self) -> NDArray:
        """Ground state wavefunction (convenience property)"""
        return self.wavefunctions[:, 0]
    
    @property
    def density(self) -> NDArray:
        """Ground state probability density"""
        return np.abs(self.wavefunction)**2
    
    def get_state(self, n: int) -> tuple[float, NDArray]:
        """Get nth eigenstate (energy, wavefunction)"""
        if n >= self.n_states:
            raise IndexError(f"Only {self.n_states} states computed")
        return self.energies[n], self.wavefunctions[:, n]
    
    def overlap(self, other_wavefunction: NDArray) -> float:
        """Compute overlap <ψ|other> with ground state"""
        return np.abs(np.trapz(
            np.conj(self.wavefunction) * other_wavefunction, 
            self.system.grid.x
        ))
    
    def expectation_value(self, operator_func: callable, state: int = 0) -> float:
        """Calculate expectation value <ψ|Ô|ψ> for any operator.
        
        Args:
            operator_func: Function that takes (x, psi) and returns Ô|ψ>
            state: Which eigenstate to use (default: ground state)
            
        Examples:
            >>> # Position expectation value
            >>> result.expectation_value(lambda x, psi: x * psi)
            >>> # Kinetic energy
            >>> result.expectation_value(lambda x, psi: -0.5 * np.gradient(np.gradient(psi)))
        """
        psi = self.wavefunctions[:, state]
        x = self.system.grid.x
        operator_psi = operator_func(x, psi)
        return np.real(np.trapz(np.conj(psi) * operator_psi, x))
    
    def position_expectation(self, state: int = 0) -> float:
        """Calculate <x> for given state"""
        psi = self.wavefunctions[:, state]
        x = self.system.grid.x
        return np.trapz(x * np.abs(psi)**2, x)
    
    def momentum_expectation(self, state: int = 0) -> float:
        """Calculate <p> for given state"""
        psi = self.wavefunctions[:, state]
        x = self.system.grid.x
        # <p> = -i * integral(ψ* dψ/dx dx)
        dpsi_dx = np.gradient(psi, x)
        return np.real(-1j * np.trapz(np.conj(psi) * dpsi_dx, x))
    
    def position_uncertainty(self, state: int = 0) -> float:
        """Calculate position uncertainty Δx for given state"""
        x = self.system.grid.x
        psi = self.wavefunctions[:, state]
        prob = np.abs(psi)**2
        
        x_mean = np.trapz(x * prob, x)
        x2_mean = np.trapz(x**2 * prob, x)
        
        return np.sqrt(x2_mean - x_mean**2)
    
    def momentum_uncertainty(self, state: int = 0) -> float:
        """Calculate momentum uncertainty Δp for given state"""
        x = self.system.grid.x
        psi = self.wavefunctions[:, state]
        
        # Calculate <p^2> using -d²/dx²
        d2psi_dx2 = np.gradient(np.gradient(psi, x), x)
        p2_mean = np.real(np.trapz(np.conj(psi) * (-d2psi_dx2), x))
        
        # Calculate <p>²
        p_mean = self.momentum_expectation(state)
        
        return np.sqrt(p2_mean - p_mean**2)
    
    def uncertainty_product(self, state: int = 0) -> float:
        """Calculate uncertainty product ΔxΔp (should be ≥ 0.5 in atomic units)"""
        return self.position_uncertainty(state) * self.momentum_uncertainty(state)
    
    def probability_current(self, state: int = 0) -> NDArray:
        """Calculate quantum probability current j(x) = (ℏ/2mi)[ψ*∇ψ - ψ∇ψ*]"""
        psi = self.wavefunctions[:, state]
        x = self.system.grid.x
        dpsi_dx = np.gradient(psi, x)
        
        # In atomic units, ℏ=1, m=1
        return np.real(np.conj(psi) * dpsi_dx - psi * np.conj(dpsi_dx)) / 2
    
    def transition_dipole(self, initial: int, final: int) -> float:
        """Calculate transition dipole moment <ψf|x|ψi>"""
        x = self.system.grid.x
        psi_i = self.wavefunctions[:, initial]
        psi_f = self.wavefunctions[:, final]
        
        return np.trapz(np.conj(psi_f) * x * psi_i, x)
    
    def plot(self, **kwargs):
        """Plot the results (convenience method).

        This automatically selects an appropriate plotting routine based on
        the system type.
        """
        from .visualisation import plot_result, plot_two_electron_result
        from .two_electron import TwoElectronSystem

        if isinstance(self.system, TwoElectronSystem):
            return plot_two_electron_result(self, **kwargs)
        return plot_result(self, **kwargs)
    
    def save(self, filename: str):
        """Save results to file (.npz format).
        
        Args:
            filename: Output filename (will add .npz extension if not present)
            
        Examples:
            >>> result.save("ground_state")  # saves as ground_state.npz
            >>> result.save("results/hydrogen.npz")
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        # Save all relevant data
        save_dict = {
            'energies': self.energies,
            'wavefunctions': self.wavefunctions,
            'grid_x': getattr(self.system.grid, "x", None),
            'grid_points': self.system.grid.points,
            'grid_bounds': self.system.grid.bounds,
            'system_name': self.system.name,
            'n_states': self.n_states,
            'solver_time': self.info.get('solver_time', 0.0),
        }
        
        # Add info dict items with string keys
        for key, value in self.info.items():
            if isinstance(value, (int, float, str, bool, np.ndarray)):
                save_dict[f'info_{key}'] = value
        
        np.savez_compressed(filename, **save_dict)
        
    @classmethod
    def load(cls, filename: str) -> "Result":
        """Load results from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Result object with loaded data
            
        Examples:
            >>> result = Result.load("ground_state.npz")
            >>> result.plot()
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        with np.load(filename) as data:
            # Reconstruct grid
            grid = Grid(
                points=int(data['grid_points']),
                bounds=tuple(data['grid_bounds'])
            )
            
            # Reconstruct system (with dummy potential)
            system = System(
                grid=grid,
                potential=lambda x: np.zeros_like(x),  # Dummy potential
                metadata={'name': str(data.get('system_name', 'loaded'))}
            )
            
            # Reconstruct info dict
            info = {}
            for key in data.files:
                if key.startswith('info_'):
                    info[key[5:]] = data[key]
            
            # Add solver_time to info if it exists
            if 'solver_time' in data:
                info['solver_time'] = float(data['solver_time'])
            
            # Create Result object
            return cls(
                system=system,
                energies=data['energies'],
                wavefunctions=data['wavefunctions'],
                info=info
            )
    
    def __repr__(self) -> str:
        return (
            f"Result(E0={self.energy:.6f}, n_states={self.n_states}, "
            f"time={self.info.get('solve_time', 0):.3f}s)"
        )


# Utility functions
def normalize_wavefunction(psi: NDArray, grid_or_x) -> NDArray:
    """Normalize a wavefunction for 1D or higher dimensional grids."""
    if isinstance(grid_or_x, np.ndarray):
        norm = np.sqrt(np.trapz(np.abs(psi) ** 2, grid_or_x))
    else:
        grid = grid_or_x
        norm = np.sqrt(np.sum(np.abs(psi) ** 2) * np.prod(grid.dxs))
    return psi / norm


def expectation_value(operator: NDArray, psi: NDArray, grid_or_x) -> complex:
    """Compute expectation value <ψ|O|ψ> for 1D or ND grids."""
    psi_star = np.conj(psi)
    if operator.ndim == 2:
        O_psi = operator @ psi
    else:
        O_psi = operator * psi

    if isinstance(grid_or_x, np.ndarray):
        return np.trapz(psi_star * O_psi, grid_or_x)
    else:
        grid = grid_or_x
        return np.sum(psi_star * O_psi) * np.prod(grid.dxs)
