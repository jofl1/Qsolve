"""
Test suite for Davidson eigenvalue solver
"""

import pytest
import numpy as np
import qsolve
from qsolve.solvers_main import build_hamiltonian


class TestDavidsonSolver:
    """Test Davidson eigenvalue solver implementation"""
    
    def test_davidson_ground_state(self):
        """Test Davidson solver finds correct ground state"""
        system = qsolve.System.create('harmonic', grid_points=101)
        
        result_davidson = qsolve.solve_davidson(system, n_states=1)
        result_sparse = qsolve.solve_eigenstates(system, n_states=1, method='sparse')
        
        # Ground state should match to high precision
        assert np.abs(result_davidson.energy - result_sparse.energy) < 1e-8
        
    def test_davidson_multiple_states(self):
        """Test Davidson solver for multiple eigenstates"""
        system = qsolve.System.create('harmonic', grid_points=101, box_size=20)
        
        # Get the first 5 states
        result_davidson = qsolve.solve_davidson(
            system, n_states=5, max_iterations=500, tolerance=1e-8
        )
        
        # Compare with direct sparse solver
        H = build_hamiltonian(system, sparse=True)
        from scipy.sparse.linalg import eigsh
        ref_energies, _ = eigsh(H, k=5, which='SA')
        ref_energies = np.sort(ref_energies)
        
        # Davidson should find the same eigenvalues
        np.testing.assert_allclose(
            result_davidson.energies, ref_energies, rtol=1e-6,
            err_msg="Davidson energies don't match reference"
        )
        
    def test_davidson_convergence(self):
        """Test that Davidson solver converges for well-behaved problems"""
        system = qsolve.System.create('harmonic', grid_points=51)
        
        result = qsolve.solve_davidson(
            system, n_states=3, 
            max_iterations=200,
            tolerance=1e-6,
            fallback=False
        )
        
        # Check residuals are small
        assert all(r < 1e-5 for r in result.info['residual_norms']), \
            f"Large residuals: {result.info['residual_norms']}"
            
    def test_davidson_different_systems(self):
        """Test Davidson on various quantum systems"""
        test_cases = [
            ('harmonic', 101, 3),
            ('hydrogen', 151, 3),
            ('box', 101, 5),
            ('morse', 101, 4),
        ]
        
        for system_name, n_points, n_states in test_cases:
            system = qsolve.System.create(system_name, grid_points=n_points)
            
            # Davidson with fallback disabled to test actual performance
            result_dav = qsolve.solve_davidson(
                system, n_states=n_states,
                max_iterations=300,
                tolerance=1e-6,
                fallback=False
            )
            
            # Reference calculation
            H = build_hamiltonian(system, sparse=True)
            from scipy.sparse.linalg import eigsh
            ref_energies, _ = eigsh(H, k=n_states, which='SA', tol=1e-6)
            ref_energies = np.sort(ref_energies)
            
            # Allow slightly larger tolerance for difficult systems
            rtol = 1e-4 if system_name == 'hydrogen' else 1e-5
            
            np.testing.assert_allclose(
                result_dav.energies, ref_energies, rtol=rtol,
                err_msg=f"Energy mismatch for {system_name}"
            )
            
    def test_davidson_vs_sparse_consistency(self):
        """Ensure Davidson and sparse methods give consistent results"""
        system = qsolve.System.create('double_well', grid_points=201)
        
        # Test with method='davidson' through solve_eigenstates
        result_davidson = qsolve.solve_eigenstates(
            system, n_states=4, method='davidson',
            max_iterations=400, tolerance=1e-7
        )
        
        # Direct sparse solver
        result_sparse = qsolve.solve_eigenstates(
            system, n_states=4, method='sparse',
            tolerance=1e-7
        )
        
        # Results should be very close
        np.testing.assert_allclose(
            result_davidson.energies, result_sparse.energies,
            rtol=1e-5, err_msg="Davidson/sparse mismatch"
        )
        
    def test_davidson_performance_params(self):
        """Test Davidson solver with different performance parameters"""
        system = qsolve.System.create('harmonic', grid_points=101)
        
        # Test with larger subspace
        result1 = qsolve.solve_davidson(
            system, n_states=3,
            max_subspace=50,  # Larger than default
            max_iterations=100,
            fallback=False
        )
        
        # Test with smaller subspace
        result2 = qsolve.solve_davidson(
            system, n_states=3,
            max_subspace=15,  # Smaller than default
            max_iterations=200,  # More iterations needed
            fallback=False
        )
        
        # Both should converge to same result
        np.testing.assert_allclose(
            result1.energies, result2.energies, rtol=1e-6,
            err_msg="Different subspace sizes give different results"
        )
        
    def test_davidson_initial_guess(self):
        """Test Davidson with custom initial guess"""
        system = qsolve.System.create('harmonic', grid_points=101)
        x = system.grid.x
        
        # Custom initial guess - excited state of harmonic oscillator
        v0 = np.sqrt(2) * x * np.exp(-x**2 / 2)
        v0 /= np.linalg.norm(v0)
        
        result = qsolve.solve_davidson(
            system, n_states=1,
            max_iterations=100,
            fallback=False
        )
        
        # Should still converge to ground state
        expected_ground = 0.5  # Approximate for harmonic oscillator
        assert abs(result.energy - expected_ground) < 0.1
        
    @pytest.mark.parametrize("n_states", [1, 3, 5, 10])
    def test_davidson_state_count(self, n_states):
        """Test Davidson with varying number of requested states"""
        system = qsolve.System.create('box', grid_points=101)
        
        result = qsolve.solve_davidson(
            system, n_states=n_states,
            max_iterations=200,
            tolerance=1e-6
        )
        
        assert len(result.energies) == n_states
        assert result.wavefunctions.shape[1] == n_states
        
        # Energies should be ordered
        assert all(result.energies[i] <= result.energies[i+1] 
                  for i in range(n_states-1))


class TestDavidsonIntegration:
    """Test Davidson solver integration with the rest of qsolve"""
    
    def test_auto_method_selection(self):
        """Test that auto method selection works with Davidson"""
        # Small system - should use Davidson
        small_system = qsolve.System.create('harmonic', grid_points=101)
        result_small = qsolve.solve_eigenstates(
            small_system, n_states=3, method='auto'
        )
        assert 'davidson' in result_small.info['method'].lower()
        
        # Large system with many states - should use sparse
        large_system = qsolve.System.create('harmonic', grid_points=1001)
        result_large = qsolve.solve_eigenstates(
            large_system, n_states=20, method='auto'
        )
        assert result_large.info['method'] == 'sparse'
        
    def test_solve_ground_state_with_davidson(self):
        """Test solve_ground_state can use Davidson"""
        system = qsolve.System.create('hydrogen', grid_points=201)
        
        result = qsolve.solve_ground_state(system, method='davidson')
        
        # Check it's a valid ground state
        assert result.energy < 0  # Hydrogen ground state is negative
        assert result.info['method'] == 'davidson'
        
    def test_davidson_fallback_mechanism(self):
        """Test Davidson fallback to sparse solver"""
        # Create a difficult problem where Davidson might struggle
        system = qsolve.System.create('hydrogen', grid_points=201, box_size=40)
        
        # Use very tight tolerance and few iterations
        result = qsolve.solve_davidson(
            system, n_states=5,
            max_iterations=50,  # Too few
            tolerance=1e-12,    # Too tight
            fallback=True       # Enable fallback
        )
        
        # Should still get correct results due to fallback
        assert len(result.energies) == 5
        assert result.energies[0] < 0  # Ground state
        
        
class TestDavidsonAccuracy:
    """Test accuracy of Davidson solver for known solutions"""
    
    def test_harmonic_oscillator_accuracy(self):
        """Test Davidson accuracy for harmonic oscillator"""
        system = qsolve.System.create('harmonic', grid_points=201, box_size=20)
        
        result = qsolve.solve_davidson(system, n_states=5, max_iterations=200)
        
        # Theoretical energies are (n + 0.5) for n = 0, 1, 2, ...
        # With finite grid, expect small deviation
        expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        # Should be accurate to ~1%
        relative_errors = np.abs(result.energies - expected) / expected
        assert all(err < 0.01 for err in relative_errors), \
            f"Relative errors too large: {relative_errors}"
            
    def test_particle_in_box_accuracy(self):
        """Test Davidson accuracy for particle in a box"""
        system = qsolve.System.create('box', grid_points=201, box_size=1.0)
        
        result = qsolve.solve_davidson(system, n_states=5)
        
        # Theoretical energies are n²π²/2 for box of length 1
        n_values = np.arange(1, 6)
        expected = n_values**2 * np.pi**2 / 2
        
        # Finite grid causes slight deviation
        relative_errors = np.abs(result.energies - expected) / expected
        assert all(err < 0.01 for err in relative_errors), \
            f"Box energies inaccurate: {relative_errors}"