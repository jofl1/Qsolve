"""
quantum/visualization.py - Visualisation utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Union, List
import matplotlib.cm as cm

from .core import Result, System
from .two_electron import TwoElectronSystem


def plot_result(
    result: Result,
    states: Optional[Union[int, List[int]]] = None,
    show_potential: bool = True,
    show_probability: bool = True,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot quantum states and probability densities.
    
    Args:
        result: Result object from solver
        states: Which states to plot (default: ground state only)
        show_potential: Show potential energy curve
        show_probability: Show probability density
        figsize: Figure size
        save_path: Path to save figure
        show: Display figure
        
    Returns:
        Matplotlib figure object
    """
    # Determine which states to plot
    if states is None:
        states = [0]  # Ground state only
    elif isinstance(states, int):
        states = list(range(states))
    
    n_states = len(states)
    n_cols = 2 if show_probability else 1
    
    fig, axes = plt.subplots(n_states, n_cols, figsize=figsize)
    if n_states == 1:
        axes = np.array([axes]).reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    x = result.system.grid.x
    V = result.system.potential(x)
    
    # Color map for different states
    colors = cm.tab10(np.linspace(0, 1, n_states))
    
    for idx, state_num in enumerate(states):
        if state_num >= result.n_states:
            continue
            
        E, psi = result.get_state(state_num)
        color = colors[idx]
        
        # Plot wavefunction
        ax = axes[idx, 0]
        ax.plot(x, psi, color=color, linewidth=2, label=f'ψ_{state_num}')
        ax.fill_between(x, 0, psi, alpha=0.3, color=color)
        
        # Add potential if requested
        if show_potential:
            ax2 = ax.twinx()
            ax2.plot(x, V, 'k--', alpha=0.5, linewidth=1)
            ax2.set_ylabel('V(x)', color='k', alpha=0.7)
            # Older Matplotlib versions do not support the ``alpha`` argument in
            # ``tick_params``. Simply omit it to maintain compatibility across
            # versions.
            ax2.tick_params(axis='y', labelcolor='k')
            
            # Add energy level
            ax2.axhline(y=E, color=color, linestyle=':', alpha=0.7)
        
        ax.set_ylabel(f'ψ_{state_num}(x)')
        ax.set_title(f'State {state_num}: E = {E:.6f}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x[0], x[-1])
        
        # Plot probability density
        if show_probability:
            ax = axes[idx, 1]
            prob = np.abs(psi)**2
            ax.plot(x, prob, color=color, linewidth=2)
            ax.fill_between(x, 0, prob, alpha=0.3, color=color)
            ax.set_ylabel(f'|ψ_{state_num}|²')
            ax.set_title(f'Probability Density')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x[0], x[-1])
    
    # Set x-labels only on bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel('Position x')
    
    # Overall title
    fig.suptitle(f'{result.system.name.title()} System - Quantum States', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_two_electron_result(
    result: Result,
    state: int = 0,
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot densities for a :class:`TwoElectronSystem` result."""

    if not isinstance(result.system, TwoElectronSystem):
        raise TypeError("Result is not for a TwoElectronSystem")

    system = result.system
    x = system.grid.x
    psi = result.wavefunctions[:, state]
    density = system.compute_density(psi)
    pair_density = system.compute_pair_density(psi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(x, density, 'b-', linewidth=2)
    ax1.fill_between(x, 0, density, alpha=0.3, color='b')
    ax1.set_xlabel('x')
    ax1.set_ylabel('n(x)')
    ax1.set_title('One-body Density')
    ax1.grid(True, alpha=0.3)

    im = ax2.pcolormesh(x, x, pair_density, shading='auto', cmap='viridis')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('Pair Density |ψ(x₁,x₂)|²')
    fig.colorbar(im, ax=ax2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_energy_levels(
    result: Result,
    n_levels: Optional[int] = None,
    show_degeneracy: bool = True,
    figsize: tuple = (8, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot energy level diagram.
    
    Args:
        result: Result object with computed energies
        n_levels: Number of levels to show (default: all)
        show_degeneracy: Mark degenerate levels
        figsize: Figure size
        save_path: Path to save figure
        show: Display figure
        
    Returns:
        Matplotlib figure object
    """
    if n_levels is None:
        n_levels = result.n_states
    else:
        n_levels = min(n_levels, result.n_states)
    
    energies = result.energies[:n_levels]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot energy levels
    for i, E in enumerate(energies):
        ax.hlines(E, 0, 1, colors='b', linewidth=2)
        ax.text(1.1, E, f'n={i}, E={E:.4f}', 
                verticalalignment='center', fontsize=10)
    
    # Mark degeneracies if requested
    if show_degeneracy:
        tol = 1e-6
        for i in range(len(energies)-1):
            if abs(energies[i+1] - energies[i]) < tol:
                ax.text(-0.1, energies[i], 'deg.', 
                       horizontalalignment='right',
                       verticalalignment='center',
                       color='red', fontsize=10)
    
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(min(energies) - 0.5, max(energies) + 0.5)
    ax.set_ylabel('Energy')
    ax.set_title(f'Energy Levels - {result.system.name.title()} System')
    ax.set_xticks([])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def animate_superposition(
    result: Result,
    coefficients: np.ndarray,
    time_points: np.ndarray,
    repeat: bool = True,
    interval: int = 50,
    save_path: Optional[str] = None,
    show: bool = True
) -> FuncAnimation:
    """Animate time evolution of a superposition state.
    
    Args:
        result: Result with eigenstates
        coefficients: Superposition coefficients
        time_points: Time values for animation
        repeat: Loop animation
        interval: Delay between frames (ms)
        save_path: Path to save animation (.gif or .mp4)
        show: Display animation
        
    Returns:
        Animation object
    """
    x = result.system.grid.x
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Initialize plots
    line_real, = ax1.plot([], [], 'b-', linewidth=2, label='Re(ψ)')
    line_imag, = ax1.plot([], [], 'r--', linewidth=2, label='Im(ψ)')
    line_prob, = ax2.plot([], [], 'g-', linewidth=2)
    
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(-2, 2)  # Adjust based on your system
    ax1.set_ylabel('Wavefunction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(0, 1)  # Adjust based on normalization
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('|ψ|²')
    ax2.grid(True, alpha=0.3)
    
    title = ax1.set_title('')
    
    def init():
        line_real.set_data([], [])
        line_imag.set_data([], [])
        line_prob.set_data([], [])
        return line_real, line_imag, line_prob, title
    
    def animate(frame):
        t = time_points[frame]
        
        # Compute time-dependent wavefunction
        psi_t = np.zeros_like(x, dtype=complex)
        for i, c in enumerate(coefficients):
            if i < result.n_states:
                E = result.energies[i]
                psi = result.wavefunctions[:, i]
                psi_t += c * psi * np.exp(-1j * E * t)
        
        # Update plots
        line_real.set_data(x, psi_t.real)
        line_imag.set_data(x, psi_t.imag)
        line_prob.set_data(x, np.abs(psi_t)**2)
        
        title.set_text(f'Time Evolution: t = {t:.2f}')
        
        return line_real, line_imag, line_prob, title
    
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(time_points), interval=interval,
        blit=True, repeat=repeat
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval)
        else:
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
    
    if show:
        plt.show()
    
    return anim


def compare_results(
    results: List[Result],
    labels: Optional[List[str]] = None,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Compare multiple results side by side.
    
    Args:
        results: List of Result objects to compare
        labels: Labels for each result
        figsize: Figure size
        save_path: Path to save figure
        show: Display figure
        
    Returns:
        Matplotlib figure object
    """
    n_results = len(results)
    if labels is None:
        labels = [f'Result {i+1}' for i in range(n_results)]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = cm.tab10(np.linspace(0, 1, n_results))
    
    # Compare ground state wavefunctions
    ax = axes[0]
    for i, (result, label) in enumerate(zip(results, labels)):
        x = result.system.grid.x
        psi = result.wavefunction
        ax.plot(x, psi, color=colors[i], linewidth=2, label=label)
    
    ax.set_xlabel('Position x')
    ax.set_ylabel('Ground State ψ₀(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Wavefunction Comparison')
    
    # Compare energy levels
    ax = axes[1]
    x_positions = np.arange(n_results)
    width = 0.8
    
    for i, (result, label) in enumerate(zip(results, labels)):
        n_show = min(5, result.n_states)
        energies = result.energies[:n_show]
        
        for j, E in enumerate(energies):
            ax.barh(j, width, left=i-width/2, height=0.1,
                   color=colors[i], alpha=0.7)
            ax.text(i, j, f'{E:.3f}', ha='center', va='center',
                   fontsize=9)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Energy Level')
    ax.set_title('Energy Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig
