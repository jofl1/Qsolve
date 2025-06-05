# Qsolve - Modern Quantum Mechanics Library

Qsolve is a modern quantum mechanics library designed for high performance compared to existing solutions while maintaining an intuitive API.

## Key Features

### **Fast Performance**
- **Davidson eigenvalue solver**: often requires significantly fewer iterations than ARPACK
- **GPU acceleration**: can provide substantial speedups with CUDA
- **Smart method selection**: Automatically chooses the best algorithm

### **Time Evolution & Dynamics**
- **Multiple evolution methods**: Split-operator, Crank-Nicolson, Runge-Kutta
- **Real-time dynamics**: Quantum tunneling, wave packet propagation
- **Observable tracking**: Monitor position, momentum, energy during evolution
- **Beautiful animations**: Visualize quantum dynamics with built-in tools

### **Comprehensive Physics**
- **13+ built-in potentials**: More than any other library
- **Custom potentials**: Easy to define and solve
- **Quantum analysis**: Expectation values, uncertainties, transition dipoles
- **Coherent states**: Advanced quantum state preparation
- **Two-electron Hamiltonians**: Model interacting electrons with various
  interaction types
- **2D and 3D grids**: Solve higher-dimensional problems

### **Modern Development**
- **GPU-first design**: CUDA/CuPy and JAX support
- **Clean API**: Intuitive and consistent interface
- **Extensive testing**: Comprehensive test suite
- **Professional packaging**: Easy installation and distribution

## Installation

### Basic Installation
```bash
pip install -e .
```

### With GPU Support (NVIDIA)
```bash
pip install -e .
pip install cupy-cuda12x  # For CUDA 12.x
# or
pip install jax[cuda12_pip]
```

### Development Installation
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Quantum Solving
```python
import Qsolve

# Solve hydrogen atom ground state quickly
result = Qsolve.solve_quick("hydrogen")
print(f"Ground state energy: {result.energy:.6f}")
result.plot()
```

### Time Evolution & Quantum Dynamics
```python
# Create a quantum tunneling simulation
def barrier(x, height=5.0, width=2.0):
    return np.where(np.abs(x) < width/2, height, 0.0)

system = Qsolve.System(
    grid=Qsolve.Grid(points=512, bounds=(-15, 15)),
    potential=barrier
)

# Initial wave packet with momentum
psi0 = Qsolve.gaussian_wave_packet(system.grid, x0=-8, p0=2.0, sigma=1.0)

# Evolve and watch the magic!
result = Qsolve.evolve_wavefunction(
    system, psi0, (0, 15), dt=0.01,
    observables=['position', 'norm']
)

# Create beautiful animation
result.animate(save_path='tunneling.gif')
```

### High-Performance Solving with Davidson
```python
# For large systems, Davidson is much faster
system = Qsolve.System.create('hydrogen', grid_points=2001, box_size=80)

# This automatically uses Davidson for best performance
result = Qsolve.solve_eigenstates(system, n_states=5)

# Or explicitly request Davidson
result = Qsolve.solve_davidson(system, n_states=5)
```

### GPU Acceleration
```python
# Check GPU availability
if Qsolve.enable_gpu():
    print("GPU acceleration available!")
    
    # GPU-accelerated time evolution
    result = Qsolve.gpu_evolve(system, psi0, (0, 10), dt=0.01)
    
    # Benchmark GPU vs CPU
    Qsolve.benchmark_gpu()
```

### Advanced Features
```python
# Coherent states for harmonic oscillator
system = Qsolve.System.create('harmonic')
psi_coherent = Qsolve.coherent_state(system, alpha=2.0)

# Multiple evolution methods
result1 = Qsolve.evolve_wavefunction(system, psi0, (0, 1), method="split-operator")
result2 = Qsolve.evolve_wavefunction(system, psi0, (0, 1), method="crank-nicolson")

# Performance benchmarking
Qsolve.benchmark_davidson_vs_eigsh(system, n_states=5)
```

## Performance Showcase

```python
# See the potential speedup in action
import time

# Large hydrogen system
system = Qsolve.System.create('hydrogen', grid_points=1000, box_size=40)

# Standard ARPACK method
start = time.time()
result1 = Qsolve.solve_eigenstates(system, n_states=5, method='sparse')
time_arpack = time.time() - start

# Our Davidson algorithm
start = time.time()
result2 = Qsolve.solve_davidson(system, n_states=5)
time_davidson = time.time() - start

print(f"Davidson is {time_arpack/time_davidson:.1f}x faster!")
```

## Time Evolution Examples

### Quantum Tunneling
```python
# Barrier tunneling simulation
result = Qsolve.simulate_tunneling(
    barrier_height=5.0,
    barrier_width=2.0,
    particle_energy=3.0,
    save_animation=True
)
print(f"Transmission probability: {result.transmission:.1%}")
```

### Wave Packet Dynamics
```python
# Harmonic oscillator with initial displacement
system = Qsolve.System.create('harmonic')
psi0 = Qsolve.gaussian_wave_packet(system.grid, x0=2.0, p0=0.0)

result = Qsolve.evolve_wavefunction(
    system, psi0, (0, 20), dt=0.1,
    observables=['position', 'energy']
)

# Oscillatory motion in <x>
import matplotlib.pyplot as plt
plt.plot(result.times, result.observables['position'])
plt.xlabel('Time')
plt.ylabel('⟨x⟩')
plt.title('Classical-like Oscillation')
plt.show()
```

### Two-Electron Example
```python
grid = Qsolve.Grid(points=51, bounds=(-5, 5))
system = Qsolve.TwoElectronSystem(
    grid,
    external_potential=Qsolve.harmonic_oscillator,
    interaction_type='contact',
    interaction_strength=1.0,
)
result = system.solve_ground_state()
result.plot()
```

### 2D Harmonic Oscillator
```python
grid = Qsolve.Grid2D(shape=(32, 32), bounds=((-4, 4), (-4, 4)))
system = Qsolve.System(grid=grid, potential=lambda x, y: 0.5 * (x**2 + y**2))
result = Qsolve.solve_ground_state(system, method="dense")
print("Ground state energy:", result.energy)
```


## Available Quantum Systems

**Built-in potentials** (more than any other library):
- `harmonic` - Harmonic oscillator  
- `hydrogen` - Hydrogen atom (1D model)
- `box` - Particle in a box
- `double_well` - Symmetric double-well potential
- `morse` - Morse potential
- `finite_well` - Finite square well
- `woods_saxon` - Woods-Saxon potential
- `poschl_teller` - Pöschl-Teller potential
- `gaussian_well` - Gaussian well
- `anharmonic` - Anharmonic oscillator
- `asymmetric_double_well` - Asymmetric double well
- `periodic` - Periodic potential
- `linear` - Linear potential

## Testing

Run the full test suite using `pytest`:

```bash
pytest
```

You can also run the example script to see the library in action:

```bash
python examples.py
```

## Command-Line Interface

```bash
# List available systems
qsolve-cli list

# Solve with Davidson for best performance
qsolve-cli hydrogen --method davidson --states 5

# Plot wavefunctions for three states
qsolve-cli harmonic --states 3 --plot

# Save results to a file
qsolve-cli hydrogen --save hydrogen_states.npz

# Batch solve across grid sizes
qsolve-cli harmonic --grid-range 200 400 100 --batch-save --batch-plot

# Vary potential parameters
qsolve-cli morse --param-range D 1 3 1 --batch-save
```

## API Reference

### Core Classes
- `System` - Quantum system with grid and potential
- `Grid` - Spatial discretization
- `Result` - Eigenvalue results with analysis methods
- `EvolutionResult` - Time evolution results with animation

### Solver Functions
- `solve_quick()` - Fast solver for built-in systems
- `solve_eigenstates()` - Multi-state solver with method selection
- `solve_davidson()` - High-performance Davidson algorithm
- `evolve_wavefunction()` - Time evolution with multiple methods

### Advanced Features
- `gaussian_wave_packet()` - Create Gaussian wave packets
- `coherent_state()` - Create coherent states
- `gpu_evolve()` - GPU-accelerated evolution
- `enable_gpu()` - Check and enable GPU support

## Requirements

**Minimum:**
- Python 3.9+
- NumPy
- SciPy  
- Matplotlib

**For GPU acceleration:**
- CuPy (NVIDIA CUDA)
- JAX (NVIDIA/Google TPU)

**For development:**
- pytest
- black
- isort

## Known Issues

### Davidson Solver Numerical Precision
The Davidson eigenvalue solver occasionally shows small numerical differences compared to ARPACK (typically <0.1 in energy units) for certain systems, particularly:
- Large grid systems (>800 points) with many states (>5)
- Systems with very flat or irregular potentials
- High-precision requirements (<1e-8 tolerance)

**Workarounds:**
- Use `method='sparse'` for highest precision critical calculations
- The auto method selection avoids problematic cases
- For most physics applications, the differences are negligible
- Performance benefits often outweigh small precision trade-offs

## Performance Notes

Actual speedups depend on the problem size, algorithm choice, and available hardware. Large grids and GPU acceleration tend to provide the biggest gains, while smaller problems may show only modest improvements.

Because Qsolve relies on numerical methods, results can differ slightly from analytic solutions. For calculations requiring very high precision, validate results against reference data when possible.

## Contributing

I welcome contributions! Areas where help is especially appreciated:

- **New potentials**: Add more quantum systems
- **GPU kernels**: Optimize CUDA/JAX implementations  
- **Documentation**: Examples and tutorials
- **Testing**: Edge cases and validation
- **Features**: 2D/3D systems, many-body physics


