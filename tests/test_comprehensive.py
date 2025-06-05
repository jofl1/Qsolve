#!/usr/bin/env python3
"""
Comprehensive test suite for Qsolve library

This tests all major functionality to ensure everything works correctly
after the integration of new features.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import Qsolve
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def test_basic_solving():
    """Test basic eigenvalue solving functionality"""
    print("Testing basic eigenvalue solving...")
    
    # Test harmonic oscillator accuracy
    result = Qsolve.solve_quick('harmonic', grid_points=201)
    expected = 0.5
    error = abs(result.energy - expected)
    assert error < 0.01, f"Harmonic oscillator error too large: {error}"
    
    # Test multiple states
    result = Qsolve.solve_quick('harmonic', n_states=3, grid_points=201)
    for i in range(3):
        expected = i + 0.5
        error = abs(result.energies[i] - expected)
        assert error < 0.01, f"Energy level {i} error too large: {error}"
    
    print("  ✓ Basic solving tests passed")


def test_all_potentials():
    """Test all built-in potential systems"""
    print("Testing all built-in potentials...")
    
    potentials = Qsolve.list_potentials()

    for pot in potentials:
        result = Qsolve.solve_quick(pot, grid_points=51)
        # Basic sanity checks
        assert np.isfinite(result.energy), f"{pot}: Energy not finite"
        assert len(result.wavefunction) > 0, f"{pot}: Empty wavefunction"
        norm = np.trapezoid(np.abs(result.wavefunction)**2, result.system.grid.x)
        assert abs(norm - 1.0) < 0.01, f"{pot}: Poor normalization: {norm}"

    print(f"  ✓ All {len(potentials)} potentials working")


def test_time_evolution():
    """Test time evolution functionality"""
    print("Testing time evolution...")
    
    # Basic evolution test
    system = Qsolve.System.create('harmonic', grid_points=101)
    psi0 = Qsolve.gaussian_wave_packet(system.grid, x0=1.0, sigma=1.0)
    
    result = Qsolve.evolve_wavefunction(
        system, psi0, (0, 1), dt=0.1,
        observables=['norm', 'position']
    )
    
    # Check norm conservation
    final_norm = result.observables['norm'][-1]
    assert abs(final_norm - 1.0) < 0.01, f"Poor norm conservation: {final_norm}"
    
    # Check we have the right number of time steps
    expected_steps = int(1.0 / 0.1) + 1
    assert len(result.times) == expected_steps, f"Wrong number of time steps"
    
    # Test different evolution methods
    methods = ['split-operator', 'crank-nicolson', 'rk4']
    for method in methods:
        result = Qsolve.evolve_wavefunction(
            system, psi0, (0, 0.2), dt=0.1, method=method
        )
        assert len(result.times) > 1, f"Method {method} failed"

    print("  ✓ Time evolution tests passed")


def test_davidson_solver():
    """Test Davidson eigenvalue solver"""
    print("Testing Davidson solver...")

    system = Qsolve.System.create('harmonic', grid_points=201)

    # Solve using Davidson
    start = time.time()
    result_dav = Qsolve.solve_davidson(system, n_states=3, max_iterations=50)
    time_dav = time.time() - start

    # Reference using sparse ARPACK
    start = time.time()
    result_ref = Qsolve.solve_eigenstates(system, n_states=3, method='sparse')
    time_ref = time.time() - start

    # Energies should closely match reference solver
    assert np.allclose(result_dav.energies, result_ref.energies, atol=1e-6)

    ratio = time_dav / time_ref

    # Auto selection should still pick Davidson
    result_auto = Qsolve.solve_eigenstates(system, n_states=2, method='auto')
    assert 'davidson' in result_auto.info['method'], "Auto method should use Davidson"

    print(f"  ✓ Davidson solver tests passed ({ratio:.2f}x vs sparse)")


def test_gpu_framework():
    """Test GPU framework (without actual GPU)"""
    print("Testing GPU framework...")
    
    # Test GPU availability check
    gpu_available = Qsolve.enable_gpu()
    print(f"    GPU available: {gpu_available}")
    
    # Test GPU system creation (should work even without GPU)
    system = Qsolve.System.create('harmonic', grid_points=51)
    gpu_system = Qsolve.GPUSystem(system, use_gpu=False)  # Force CPU
    assert gpu_system is not None, "GPU system creation failed"
    print("  ✓ GPU framework tests passed")


def test_advanced_features():
    """Test advanced quantum mechanics features"""
    print("Testing advanced features...")
    
    # Test Gaussian wave packet creation
    grid = Qsolve.Grid(points=101, bounds=(-5, 5))
    psi = Qsolve.gaussian_wave_packet(grid, x0=1.0, p0=2.0, sigma=0.8)
    assert psi.dtype == complex, "Wave packet should be complex"
    
    norm = np.trapezoid(np.abs(psi)**2, grid.x)
    assert abs(norm - 1.0) < 0.01, f"Wave packet not normalized: {norm}"
    
    # Test coherent states
    system = Qsolve.System.create('harmonic', grid_points=101)
    psi_coherent = Qsolve.coherent_state(system, alpha=1.5, n_max=10)
    norm = np.trapezoid(np.abs(psi_coherent)**2, system.grid.x)
    assert abs(norm - 1.0) < 0.01, f"Coherent state not normalized: {norm}"
    
    # Test quantum tunneling simulation
    result = Qsolve.simulate_tunneling(
        barrier_height=3.0, barrier_width=1.0, particle_energy=2.0,
        time_span=(0, 1), save_animation=False
    )
    assert len(result.times) > 1, "Tunneling simulation failed"

    print("  ✓ Advanced features tests passed")


def test_cli_functionality():
    """Test command-line interface"""
    print("Testing CLI functionality...")
    
    # Test list command
    import subprocess
    result = subprocess.run([
        'python', '-m', 'Qsolve.cli', 'list'
    ], capture_output=True, text=True)

    assert result.returncode == 0, "CLI list command failed"
    assert 'harmonic' in result.stdout, "CLI list output missing harmonic"

    # Test batch solving with grid range and auto-save
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..")
        result = subprocess.run([
            'python', '-m', 'Qsolve.cli', 'harmonic',
            '--grid-range', '30', '40', '10',
            '--batch-save', '--batch-plot'
        ], cwd=tmp, capture_output=True, text=True, env=env)

        assert result.returncode == 0, "CLI batch command failed"
        assert os.path.exists(os.path.join(tmp, 'harmonic_g30.npz'))
        assert os.path.exists(os.path.join(tmp, 'harmonic_g40.npz'))
        assert os.path.exists(os.path.join(tmp, 'comparison.png'))

    print("  ✓ CLI tests passed")


def test_performance():
    """Test performance characteristics"""
    print("Testing performance...")
    
    # Test scaling with system size
    sizes = [101, 201, 401]
    times = []
    
    for n in sizes:
        system = Qsolve.System.create('harmonic', grid_points=n)
        
        start = time.time()
        result = Qsolve.solve_ground_state(system)
        elapsed = time.time() - start
        
        times.append(elapsed)
        assert elapsed < 2.0, f"Solving took too long: {elapsed:.3f}s for {n} points"
    
    # Test that Davidson is at least as fast as sparse for small systems
    system = Qsolve.System.create('hydrogen', grid_points=201)
    
    start = time.time()
    result1 = Qsolve.solve_eigenstates(system, n_states=3, method='sparse')
    time_sparse = time.time() - start
    
    start = time.time()
    result2 = Qsolve.solve_davidson(system, n_states=3)
    time_davidson = time.time() - start
    
    # Davidson should be competitive (within 3x)
    ratio = time_davidson / time_sparse
    assert ratio < 3.0, f"Davidson too slow: {ratio:.1f}x slower than sparse"

    print(f"  ✓ Performance tests passed (Davidson: {ratio:.1f}x vs sparse)")


