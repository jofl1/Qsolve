import pytest
import numpy as np
import qsolve
import tempfile
import os

class TestFullWorkflows:
    def test_tunneling_simulation(self):
        """Test complete tunneling simulation workflow."""
        result = qsolve.simulate_tunneling(
            barrier_height=5.0,
            barrier_width=2.0,
            particle_energy=3.0,
            save_animation=False
        )
        
        assert hasattr(result, 'transmission')
        assert 0 <= result.transmission <= 1
        
    def test_save_load_results(self):
        """Test saving and loading results."""
        result = qsolve.solve_quick("hydrogen")
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            result.save(f.name)
            loaded = qsolve.load_result(f.name)
            
        assert np.allclose(loaded.energy, result.energy)
        os.unlink(f.name)

    def test_two_electron_system(self):
        """Test two-electron system functionality."""
        grid = qsolve.Grid(points=31, bounds=(-3, 3))
        system = qsolve.TwoElectronSystem(
            grid,
            external_potential=qsolve.harmonic_oscillator,
            interaction_type='coulomb',
            interaction_strength=0.5,
        )
        result = system.solve_ground_state()
        assert result.energy > 0  # Should be higher than non-interacting