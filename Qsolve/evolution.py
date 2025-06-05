"""
qsolve/evolution.py - Time-dependent quantum mechanics utilities
"""

import numpy as np
from numpy.fft import fftfreq
from dataclasses import dataclass
from typing import Optional, Callable, Union, List
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

from .Core import System, Result, Grid

logger = logging.getLogger(__name__)


@dataclass
class EvolutionResult:
    """Container for time evolution results"""

    times: np.ndarray
    wavefunctions: np.ndarray  # Shape: (n_times, n_points)
    observables: dict = None
    system: System = None

    @property
    def n_times(self) -> int:
        return len(self.times)

    def get_density(self, time_idx: int) -> np.ndarray:
        """Get probability density at specific time"""
        return np.abs(self.wavefunctions[time_idx]) ** 2

    def get_observable(self, name: str) -> np.ndarray:
        """Get time series of an observable"""
        if self.observables and name in self.observables:
            return self.observables[name]
        raise KeyError(f"Observable '{name}' not computed")

    def animate(self, interval: int = 50, save_path: Optional[str] = None):
        """Create animation of time evolution"""
        return animate_evolution(self, interval, save_path)


def evolve_wavefunction(
    system: System,
    psi0: np.ndarray,
    time_span: tuple[float, float],
    dt: float = 0.01,
    method: str = "split-operator",
    observables: Optional[List[str]] = None,
    store_every: int = 1,
) -> EvolutionResult:
    """
    Solve time-dependent Schrödinger equation: iℏ ∂ψ/∂t = Ĥψ

    Parameters
    ----------
    system : System
        Quantum system with potential V(x)
    psi0 : np.ndarray
        Initial wavefunction
    time_span : tuple[float, float]
        (t_initial, t_final) for evolution
    dt : float
        Time step (default: 0.01)
    method : str
        Evolution method: 'split-operator', 'crank-nicolson', 'rk4'
    observables : List[str], optional
        Observables to track: ['energy', 'position', 'momentum', 'norm']
    store_every : int
        Store wavefunction every N steps (default: 1)

    Returns
    -------
    EvolutionResult
        Object containing times, wavefunctions, and observables

    Examples
    --------
    >>> # Gaussian wave packet in harmonic oscillator
    >>> system = qsolve.System.create("harmonic")
    >>> x = system.grid.x
    >>> psi0 = np.exp(-(x-2)**2) * np.exp(2j*x)  # Moving Gaussian
    >>> psi0 /= np.sqrt(np.trapz(np.abs(psi0)**2, x))
    >>>
    >>> result = qsolve.evolve_wavefunction(system, psi0, (0, 10))
    >>> result.animate()
    """

    # Time points
    t0, tf = time_span
    n_steps = int((tf - t0) / dt)
    times = np.linspace(t0, tf, n_steps + 1)

    # Select evolution method
    if method == "split-operator":
        evolver = SplitOperatorEvolver(system, dt)
    elif method == "crank-nicolson":
        evolver = CrankNicolsonEvolver(system, dt)
    elif method == "rk4":
        evolver = RungeKutta4Evolver(system, dt)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Initialize storage
    store_times = times[::store_every]
    wf_size = system.grid.points
    wavefunctions = np.zeros((len(store_times), wf_size), dtype=complex)
    wavefunctions[0] = psi0.reshape(-1)

    # Observable tracking
    obs_tracker = ObservableTracker(system, observables) if observables else None
    if obs_tracker:
        obs_tracker.measure(psi0, t0)

    # Time evolution loop
    psi = psi0.copy()
    store_idx = 1

    logger.info(
        "Evolving from t=%s to t=%s with dt=%s (%s steps)...",
        t0,
        tf,
        dt,
        n_steps,
    )

    for i in range(1, n_steps + 1):
        # Evolve one time step
        psi = evolver.step(psi)

        # Store wavefunction
        if i % store_every == 0 and store_idx < len(store_times):
            wavefunctions[store_idx] = psi.reshape(-1)
            store_idx += 1

        # Measure observables
        if obs_tracker and i % store_every == 0:
            obs_tracker.measure(psi, times[i])

        # Progress update
        if n_steps >= 10 and i % (n_steps // 10) == 0:
            logger.info("  Progress: %.0f%%", 100 * i / n_steps)

    logger.info("Evolution completed.")

    return EvolutionResult(
        times=store_times,
        wavefunctions=wavefunctions,
        observables=obs_tracker.results if obs_tracker else None,
        system=system,
    )


class SplitOperatorEvolver:
    """Split-operator method for time evolution (most accurate and GPU-friendly)"""

    def __init__(self, system: System, dt: float):
        self.system = system
        self.dt = dt

        # Precompute operators for arbitrary dimensions
        coords = system.grid.coordinates()
        shape = coords[0].shape

        k_axes = [2 * np.pi * fftfreq(n, d) for n, d in zip(system.grid.shape, system.grid.dxs)]
        k_grids = np.meshgrid(*k_axes, indexing="ij")
        k2 = np.zeros(shape)
        for K in k_grids:
            k2 += K**2

        T_k = 0.5 * k2 / system.mass
        V = system.potential(*coords)
        self.U_V_half = np.exp(-0.5j * V * dt / system.hbar)
        self.U_T = np.exp(-1j * T_k * dt / system.hbar)
        self.shape = shape

    def step(self, psi: np.ndarray) -> np.ndarray:
        """Evolve one time step using Trotter decomposition"""
        # exp(-iHt) ≈ exp(-iVt/2) exp(-iTt) exp(-iVt/2)
        psi = self.U_V_half * psi
        psi = np.fft.ifftn(self.U_T * np.fft.fftn(psi))
        psi = self.U_V_half * psi
        return psi


class CrankNicolsonEvolver:
    """Crank-Nicolson method (implicit, unconditionally stable)"""

    def __init__(self, system: System, dt: float):
        self.system = system
        self.dt = dt

        # Build Hamiltonian
        from . import Solvers

        build_hamiltonian = Solvers.build_hamiltonian
        H = build_hamiltonian(system, sparse=True)

        # Crank-Nicolson matrices
        from scipy.sparse import identity
        from scipy.sparse.linalg import factorized

        I = identity(H.shape[0])
        self.A = I + 0.5j * dt * H / system.hbar  # (1 + iHdt/2)
        self.B = I - 0.5j * dt * H / system.hbar  # (1 - iHdt/2)

        # Precompute LU factorization for efficiency
        self.solve = factorized(self.A.tocsc())

    def step(self, psi: np.ndarray) -> np.ndarray:
        """Implicit step: (1 + iHdt/2)ψ(t+dt) = (1 - iHdt/2)ψ(t)"""
        return self.solve(self.B @ psi)


class RungeKutta4Evolver:
    """4th-order Runge-Kutta (general purpose, explicit)"""

    def __init__(self, system: System, dt: float):
        self.system = system
        self.dt = dt

        # Build Hamiltonian
        from . import Solvers

        build_hamiltonian = Solvers.build_hamiltonian
        self.H = build_hamiltonian(system, sparse=True)

    def step(self, psi: np.ndarray) -> np.ndarray:
        """RK4 step for iℏ∂ψ/∂t = Hψ"""
        dt = self.dt
        H = self.H
        hbar = self.system.hbar

        # RK4 stages
        k1 = -1j * H @ psi / hbar
        k2 = -1j * H @ (psi + 0.5 * dt * k1) / hbar
        k3 = -1j * H @ (psi + 0.5 * dt * k2) / hbar
        k4 = -1j * H @ (psi + dt * k3) / hbar

        return psi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class ObservableTracker:
    """Track expectation values during time evolution"""

    def __init__(self, system: System, observables: List[str]):
        self.system = system
        self.observables = observables
        self.results = {obs: [] for obs in observables}

        # Precompute operators
        self.operators = {}
        if hasattr(system.grid, "x"):
            x = system.grid.x
            if "position" in observables:
                self.operators["position"] = x

            if "momentum" in observables:
                dx = system.grid.dx
                n = len(x)
                D = np.zeros((n, n))
                for i in range(1, n - 1):
                    D[i, i + 1] = 1 / (2 * dx)
                    D[i, i - 1] = -1 / (2 * dx)
                self.operators["momentum"] = -1j * system.hbar * D

        if "energy" in observables:
            from . import Solvers

            build_hamiltonian = Solvers.build_hamiltonian
            self.operators["energy"] = build_hamiltonian(system, sparse=False)

    def measure(self, psi: np.ndarray, t: float):
        """Measure all observables at current time"""
        grid = self.system.grid

        # Norm (should always be 1)
        if "norm" in self.observables:
            if hasattr(grid, "x"):
                norm = np.trapz(np.abs(psi) ** 2, grid.x)
            else:
                norm = np.sum(np.abs(psi) ** 2) * np.prod(grid.dxs)
            self.results["norm"].append(norm)

        if hasattr(grid, "x"):
            if "position" in self.observables:
                pos = np.real(np.trapz(np.conj(psi) * grid.x * psi, grid.x))
                self.results["position"].append(pos)

            if "momentum" in self.observables:
                p_psi = self.operators["momentum"] @ psi
                mom = np.real(np.trapz(np.conj(psi) * p_psi, grid.x))
                self.results["momentum"].append(mom)

            if "position_uncertainty" in self.observables:
                x_mean = np.real(np.trapz(np.conj(psi) * grid.x * psi, grid.x))
                x2_mean = np.real(np.trapz(np.conj(psi) * grid.x ** 2 * psi, grid.x))
                sigma_x = np.sqrt(x2_mean - x_mean**2)
                self.results["position_uncertainty"].append(sigma_x)

        if "energy" in self.observables:
            H_psi = self.operators["energy"] @ psi
            if hasattr(grid, "x"):
                energy = np.real(np.trapz(np.conj(psi) * H_psi, grid.x))
            else:
                energy = np.real(np.sum(np.conj(psi) * H_psi) * np.prod(grid.dxs))
            self.results["energy"].append(energy)


def animate_evolution(
    result: EvolutionResult,
    interval: int = 50,
    save_path: Optional[str] = None,
    show_potential: bool = True,
    show_phase: bool = True,
) -> FuncAnimation:
    """Create beautiful animation of quantum time evolution"""

    x = result.system.grid.x
    V = result.system.potential(x)

    # Setup figure
    if show_phase:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # Wavefunction plot
    (line_real,) = ax1.plot([], [], "b-", linewidth=2, label="Re(ψ)")
    (line_imag,) = ax1.plot([], [], "r--", linewidth=2, label="Im(ψ)")

    if show_potential:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, V, "k-", alpha=0.3, linewidth=1)
        ax1_twin.set_ylabel("V(x)", alpha=0.5)

    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(-2, 2)  # Adjust based on your system
    ax1.set_ylabel("Wavefunction")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Probability density plot
    (line_prob,) = ax2.plot([], [], "g-", linewidth=2)
    fill_prob = ax2.fill_between(x, 0, 0, alpha=0.3, color="green")

    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(0, 1)  # Adjust based on normalization
    ax2.set_xlabel("Position")
    ax2.set_ylabel("|ψ|²")
    ax2.grid(True, alpha=0.3)

    # Phase plot (optional)
    if show_phase:
        (line_phase,) = ax3.plot([], [], "purple", linewidth=2)
        ax3.set_xlim(x[0], x[-1])
        ax3.set_ylim(-np.pi, np.pi)
        ax3.set_xlabel("Position")
        ax3.set_ylabel("Phase arg(ψ)")
        ax3.grid(True, alpha=0.3)

    time_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes, fontsize=12)

    def init():
        line_real.set_data([], [])
        line_imag.set_data([], [])
        line_prob.set_data([], [])
        if show_phase:
            line_phase.set_data([], [])
        return [line_real, line_imag, line_prob]

    def animate(frame):
        # Get wavefunction at this time
        psi = result.wavefunctions[frame]
        prob = np.abs(psi) ** 2

        # Update plots
        line_real.set_data(x, psi.real)
        line_imag.set_data(x, psi.imag)
        line_prob.set_data(x, prob)

        # Update fill
        ax2.collections.clear()
        ax2.fill_between(x, 0, prob, alpha=0.3, color="green")

        if show_phase:
            phase = np.angle(psi)
            line_phase.set_data(x, phase)

        time_text.set_text(f"t = {result.times[frame]:.2f}")

        return [line_real, line_imag, line_prob]

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=result.n_times,
        interval=interval,
        blit=False,
        repeat=True,
    )

    if save_path:
        logger.info("Saving animation to %s...", save_path)
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        else:
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval)

    plt.show()
    return anim


# Example applications


def gaussian_wave_packet(
    grid: Grid, x0: float = 0.0, p0: float = 0.0, sigma: float = 1.0
) -> np.ndarray:
    """Create a Gaussian wave packet"""
    x = grid.x
    psi = np.exp(-((x - x0) ** 2) / (4 * sigma**2), dtype=complex)
    psi *= np.exp(1j * p0 * x)
    # Normalize
    psi /= np.sqrt(np.trapz(np.abs(psi) ** 2, x))
    return psi


def coherent_state(system: System, alpha: complex, n_max: int = 50) -> np.ndarray:
    """Create coherent state for harmonic oscillator"""
    # This is a superposition of energy eigenstates
    # |α⟩ = exp(-|α|²/2) Σ(α^n/√n!) |n⟩
    from scipy.special import factorial

    # Get energy eigenstates
    from . import Solvers

    solve_eigenstates = Solvers.solve_eigenstates
    result = solve_eigenstates(system, n_states=n_max)

    # Build coherent state
    psi = np.zeros_like(result.wavefunctions[:, 0], dtype=complex)
    prefactor = np.exp(-0.5 * np.abs(alpha) ** 2)

    for n in range(n_max):
        coeff = alpha**n / np.sqrt(factorial(n))
        psi += coeff * result.wavefunctions[:, n]

    psi *= prefactor

    # Normalize
    x = system.grid.x
    psi /= np.sqrt(np.trapz(np.abs(psi) ** 2, x))

    return psi


def simulate_tunneling(
    barrier_height: float = 5.0,
    barrier_width: float = 2.0,
    particle_energy: float = 3.0,
    time_span: tuple = (0, 20),
    save_animation: bool = True,
) -> EvolutionResult:
    """Simulate quantum tunneling through a barrier"""

    # Create barrier potential
    def barrier_potential(x):
        V = np.zeros_like(x)
        V[np.abs(x) < barrier_width / 2] = barrier_height
        return V

    # Setup system
    system = System(
        grid=Grid(points=512, bounds=(-20, 20)), potential=barrier_potential
    )

    # Initial wave packet approaching barrier
    k0 = np.sqrt(2 * particle_energy)  # Initial momentum
    psi0 = gaussian_wave_packet(system.grid, x0=-8, p0=k0, sigma=1.5)

    # Evolve
    result = evolve_wavefunction(
        system, psi0, time_span, dt=0.01, observables=["position", "energy", "norm"]
    )

    if save_animation:
        result.animate(save_path="tunneling.gif")

    # Calculate transmission probability
    final_psi = result.wavefunctions[-1]
    x = system.grid.x
    trans_prob = np.trapz(
        np.abs(final_psi[x > barrier_width / 2]) ** 2, x[x > barrier_width / 2]
    )
    logger.info("Transmission probability: %.2f%%", trans_prob * 100)

    return result
