import pytest
import numpy as np
import qsolve

class TestPotentials:
    """Test all built-in potential functions."""
    
    @pytest.mark.parametrize("potential_name", [
        "harmonic_oscillator", "hydrogen_atom", "particle_box",
        "double_well", "morse", "finite_well", "woods_saxon",
        "poschl_teller", "gaussian_well", "anharmonic",
        "asymmetric_double_well", "periodic", "linear"
    ])
    def test_potential_callable(self, potential_name):
        """Test that all potentials are callable."""
        potential = getattr(qsolve.potentials, potential_name)
        x = np.linspace(-5, 5, 100)
        V = potential(x)
        assert V.shape == x.shape
        assert np.all(np.isfinite(V))

    def test_custom_potential(self):
        """Test using custom potential function."""
        def custom_V(x):
            return x**4 - 2*x**2
        
        grid = qsolve.Grid(points=100, bounds=(-3, 3))
        system = qsolve.System(grid, custom_V)
        result = qsolve.solve_eigenstates(system, n_states=2)
        assert result.energies[0] < result.energies[1]