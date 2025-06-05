import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import Qsolve
import numpy as np


def test_two_electron_basic():
    grid = Qsolve.Grid(points=21, bounds=(-5, 5))
    system = Qsolve.TwoElectronSystem(
        grid,
        external_potential=lambda x: np.zeros_like(x),
        interaction_type="contact",
        interaction_strength=1.0,
    )
    result = system.solve_ground_state()
    assert np.isfinite(result.energy), "Energy should be finite"
    density = system.compute_density(result.wavefunctions[:, 0])
    assert density.shape[0] == grid.points, "Density shape mismatch"


def test_two_electron_plot():
    import matplotlib
    matplotlib.use('Agg')

    grid = Qsolve.Grid(points=11, bounds=(-2, 2))
    system = Qsolve.TwoElectronSystem(
        grid,
        external_potential=Qsolve.harmonic_oscillator,
        interaction_type="contact",
    )
    result = system.solve_ground_state()
    fig = result.plot(show=False)
    assert fig is not None

