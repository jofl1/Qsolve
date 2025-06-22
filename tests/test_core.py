import pytest
import numpy as np
import qsolve

class TestBasicFunctionality:
    def test_import(self):
        """Test that package imports correctly."""
        assert hasattr(qsolve, 'System')
        assert hasattr(qsolve, 'solve_quick')
        assert hasattr(qsolve, 'Grid')
    
    def test_version(self):
        """Test version is accessible."""
        assert hasattr(qsolve, '__version__')
        assert isinstance(qsolve.__version__, str)

class TestBuiltinSystems:
    @pytest.mark.parametrize("system_name", [
        "harmonic", "hydrogen", "box", "double_well", "morse"
    ])
    def test_solve_quick(self, system_name):
        """Test quick solve for all built-in systems."""
        result = qsolve.solve_quick(system_name)
        assert hasattr(result, 'energy')
        assert hasattr(result, 'wavefunction')
        assert result.energy is not None
        
    def test_hydrogen_ground_state(self):
        """Test hydrogen atom ground state energy."""
        result = qsolve.solve_quick("hydrogen")
        # 1D hydrogen model ground state
        assert result.energy < 0
        assert abs(result.energy + 0.5) < 0.1  # Roughly -0.5 a.u.

class TestDavidsonSolver:
    def test_davidson_vs_standard(self):
        """Test Davidson solver gives similar results to standard method."""
        system = qsolve.System.create('hydrogen', grid_points=200, box_size=20)
        
        result_davidson = qsolve.solve_davidson(system, n_states=3)
        result_standard = qsolve.solve_eigenstates(system, n_states=3, method='sparse')
        
        # Check energies are close
        np.testing.assert_allclose(
            result_davidson.energies[:3], 
            result_standard.energies[:3], 
            rtol=1e-3
        )

class TestTimeEvolution:
    def test_norm_conservation(self):
        """Test that wavefunction norm is conserved during evolution."""
        system = qsolve.System.create('harmonic')
        grid = system.grid
        psi0 = qsolve.gaussian_wave_packet(grid, x0=0, p0=1.0, sigma=1.0)
        
        result = qsolve.evolve_wavefunction(
            system, psi0, (0, 1), dt=0.01,
            observables=['norm']
        )
        
        norms = result.observables['norm']
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    @pytest.mark.parametrize("method", ["split-operator", "crank-nicolson"])
    def test_evolution_methods(self, method):
        """Test different evolution methods work."""
        system = qsolve.System.create('harmonic')
        psi0 = qsolve.gaussian_wave_packet(system.grid, x0=1.0)
        
        result = qsolve.evolve_wavefunction(
            system, psi0, (0, 0.1), dt=0.01, method=method
        )
        assert result is not None
        assert len(result.times) > 0

class TestGPU:
    @pytest.mark.skipif(not qsolve.HAS_CUPY, reason="GPU not available")
    def test_gpu_evolution(self):
        """Test GPU evolution if available."""
        system = qsolve.System.create('harmonic')
        psi0 = qsolve.gaussian_wave_packet(system.grid)
        
        result = qsolve.gpu_evolve(system, psi0, (0, 0.1), dt=0.01)
        assert result is not None