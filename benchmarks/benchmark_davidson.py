import time
import numpy as np
import matplotlib.pyplot as plt
import qsolve

def benchmark_davidson():
    """Benchmark Davidson vs standard eigensolvers."""
    grid_sizes = [100, 200, 400, 800, 1600]
    davidson_times = []
    arpack_times = []
    
    for n_points in grid_sizes:
        system = qsolve.System.create('hydrogen', grid_points=n_points, box_size=40)
        
        # Time ARPACK
        start = time.time()
        qsolve.solve_eigenstates(system, n_states=5, method='sparse')
        arpack_times.append(time.time() - start)
        
        # Time Davidson
        start = time.time()
        qsolve.solve_davidson(system, n_states=5)
        davidson_times.append(time.time() - start)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.loglog(grid_sizes, arpack_times, 'o-', label='ARPACK', linewidth=2)
    plt.loglog(grid_sizes, davidson_times, 's-', label='Davidson', linewidth=2)
    plt.xlabel('Grid Size')
    plt.ylabel('Time (seconds)')
    plt.title('Davidson vs ARPACK Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('benchmarks/davidson_performance.png', dpi=150)
    
    return grid_sizes, davidson_times, arpack_times

if __name__ == "__main__":
    benchmark_davidson()