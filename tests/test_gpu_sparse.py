import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import Qsolve
import numpy as np
from scipy.sparse import isspmatrix
from Qsolve import gpu


def test_gpu_sparse_cpu_fallback():
    system = Qsolve.System.create('harmonic', grid_points=50)
    gpu_sys = Qsolve.GPUSystem(system, use_gpu=False)
    H = gpu.gpu_hamiltonian(gpu_sys, sparse=True)
    assert isspmatrix(H), "Hamiltonian should be scipy sparse on CPU"

    E, psi = gpu.gpu_eigensolver(H, k=1, method='eigsh', max_iter=300)
    assert np.isfinite(E[0])
    assert abs(float(E[0]) - 0.5) < 0.05

