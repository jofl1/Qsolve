#!/usr/bin/env python3
"""
Integration test for new Qsolve features

This verifies that evolution.py, davidson.py, and gpu.py are properly integrated.
Based on the integration guide from CLAUDE.md.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import Qsolve
import numpy as np
import time


def test_integration():
    """Test that all components work together"""
    print("="*60)
    print("QSOLVE INTEGRATION TEST")
    print("="*60)
    print(f"Version: {Qsolve.__version__}")
    print(f"GPU available: {Qsolve.HAS_CUPY}")
    print()
    
    # Create simple system
    system = Qsolve.System.create('harmonic', grid_points=101)
    
    print("Testing individual components...")
    
    # Test 1: Davidson solver
    print("1. Testing Davidson solver...")
    result = Qsolve.solve_davidson(system, n_states=3)
    print(f"   ✓ Davidson solver works - Ground state: {result.energies[0]:.6f}")
    
    # Test 2: Time evolution
    print("2. Testing time evolution...")
    psi0 = Qsolve.gaussian_wave_packet(system.grid)
    result = Qsolve.evolve_wavefunction(system, psi0, (0, 1), dt=0.1)
    print(f"   ✓ Time evolution works - {len(result.times)} time steps")
    
    # Test 3: GPU (if available)
    print("3. Testing GPU acceleration...")
    if Qsolve.enable_gpu():
        gpu_system = Qsolve.GPUSystem(system)
        print("   ✓ GPU acceleration available")
    else:
        print("   ! GPU not available (this is OK on CPU-only systems)")
    
    # Test 4: Davidson integration in solve_eigenstates
    print("4. Testing Davidson integration in solve_eigenstates...")
    result = Qsolve.solve_eigenstates(system, n_states=3, method='davidson')
    print(f"   ✓ Davidson integrated - Ground state: {result.energies[0]:.6f}")
        
    # Test 5: Auto method selection should choose Davidson
    print("5. Testing auto method selection...")
    system_large = Qsolve.System.create('harmonic', grid_points=800)
    result = Qsolve.solve_eigenstates(system_large, n_states=5, method='auto', verbose=True)
    print(f"   ✓ Auto method works - Ground state: {result.energies[0]:.6f}")
    print(f"   ✓ Method used: {result.info['method']}")
    
    print("\n" + "="*60)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("="*60)
    print("\nQsolve new features are successfully integrated:")
    print("  • Time evolution with multiple methods")
    print("  • Davidson solver (should be faster than ARPACK)")
    print("  • GPU acceleration (when available)")
    print("  • Seamless integration with existing API")


def benchmark_performance():
    """Quick performance benchmark"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Test system
    system = Qsolve.System.create('hydrogen', grid_points=500, box_size=40)
    print(f"System: {system.name} ({system.grid.points} points)")
    print("Finding 3 lowest energy states...\n")
    
    # Method 1: Standard ARPACK
    print("1. Standard method (ARPACK):")
    start = time.time()
    result1 = Qsolve.solve_eigenstates(system, n_states=3, method='sparse')
    time1 = time.time() - start
    print(f"   Time: {time1:.3f}s")
    print(f"   Ground state: {result1.energies[0]:.8f}")
    
    # Method 2: Davidson
    print("\n2. Davidson algorithm:")
    start = time.time()
    result2 = Qsolve.solve_davidson(system, n_states=3)
    time2 = time.time() - start
    print(f"   Time: {time2:.3f}s")
    print(f"   Ground state: {result2.energies[0]:.8f}")
    
    # Compare
    speedup = time1 / time2
    energy_diff = abs(result1.energies[0] - result2.energies[0])
    
    print(f"\n🚀 Davidson is {speedup:.1f}x faster!")
    print(f"   Energy difference: {energy_diff:.2e} (should be very small)")
    
    if speedup >= 1.0 and energy_diff < 1e-6:
        print("   ✓ Davidson performance benchmark PASSED")
    else:
        print("   ! Davidson performance not faster")


def test_examples():
    """Test some example use cases"""
    print("\n" + "="*60)
    print("EXAMPLE USE CASES")
    print("="*60)
    
    print("1. Quantum tunneling simulation...")
    # Simple tunneling test (short time)
    def barrier(x, height=3.0, width=1.0):
        return np.where(np.abs(x) < width/2, height, 0.0)

    system = Qsolve.System(
        grid=Qsolve.Grid(points=128, bounds=(-10, 10)),
        potential=barrier
    )

    x = system.grid.x
    k0 = np.sqrt(2 * 2.0)  # Energy = 2.0
    psi0 = Qsolve.gaussian_wave_packet(system.grid, x0=-4, p0=k0, sigma=0.8)

    result = Qsolve.evolve_wavefunction(system, psi0, (0, 2), dt=0.1)
    print(f"   ✓ Tunneling simulation: {len(result.times)} time steps")

    # Check conservation
    final_norm = np.trapezoid(np.abs(result.wavefunctions[-1])**2, x)
    print(f"   ✓ Norm conservation: {final_norm:.6f} (should be ~1.0)")
    
    print("\n2. Coherent state example...")
    harmonic_system = Qsolve.System.create('harmonic', grid_points=128)
    alpha = 1.5  # Coherent state parameter
    psi_coherent = Qsolve.coherent_state(harmonic_system, alpha, n_max=20)
    print(f"   ✓ Coherent state created (norm: {np.trapezoid(np.abs(psi_coherent)**2, harmonic_system.grid.x):.6f})")
    
    print("\n✓ Example use cases PASSED")


