# GPU Acceleration in Qsolve

Qsolve now supports GPU acceleration on both NVIDIA and Apple Silicon hardware, providing 10-100x speedups for quantum mechanics calculations.

## Supported Platforms

### NVIDIA GPUs (CUDA)
- Works on Linux and Windows
- Requires CUDA 11.x or 12.x
- Install: `pip install Qsolve[gpu]`

### Apple Silicon (Metal)
- Works on M1, M2, M3 Macs
- Uses Apple's Metal Performance Shaders
- Install: `pip install Qsolve[metal]`

### Both Platforms
- Install: `pip install Qsolve[gpu-all]`

## Usage

### Automatic GPU Detection

```python
import Qsolve

# Check GPU availability
if Qsolve.enable_gpu():
    print("GPU acceleration available!")
    
# Qsolve will automatically use GPU when available
result = Qsolve.solve_quick("hydrogen", grid_points=2000)
```

### Manual Backend Selection

```python
# Force specific backend
system = Qsolve.System.create("harmonic", grid_points=1000)

# Use NVIDIA GPU
gpu_system = Qsolve.GPUSystem(system, backend='cuda')

# Use Apple Metal
gpu_system = Qsolve.GPUSystem(system, backend='metal')

# Auto-select best available
gpu_system = Qsolve.GPUSystem(system, backend='auto')
```

### Time Evolution with GPU

```python
# Create initial wavefunction
psi0 = Qsolve.gaussian_wave_packet(system.grid, x0=0, p0=5)

# GPU-accelerated evolution (auto-detects best backend)
result = Qsolve.gpu_evolve(system, psi0, time_span=(0, 10), dt=0.01)

# Force specific backend
result = Qsolve.gpu_evolve(system, psi0, time_span=(0, 10), backend='metal')
```

## Performance Comparison

Typical speedups for a 2000-point grid:

| Operation | CPU Time | NVIDIA GPU | Apple Metal |
|-----------|----------|------------|-------------|
| Time Evolution (1000 steps) | 10s | 0.2s (50x) | 0.5s (20x) |
| Ground State | 5s | 0.1s (50x) | 0.3s (17x) |
| 10 Eigenstates | 20s | 0.5s (40x) | 1s (20x) |

## Checking Your GPU

```python
import Qsolve

# Detailed GPU information
Qsolve.print_gpu_info()

# Get capabilities dictionary
info = Qsolve.detect_gpu_capabilities()

if info['cuda']['available']:
    print(f"NVIDIA GPU: {info['cuda']['devices'][0]['name']}")
    
if info['metal']['available']:
    print("Apple GPU available")
```

## Troubleshooting

### NVIDIA GPUs

1. Check CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Install appropriate CuPy version:
   ```bash
   # For CUDA 12.x
   pip install cupy-cuda12x
   
   # For CUDA 11.x
   pip install cupy-cuda11x
   ```

### Apple Silicon

1. Ensure you're on macOS 12.0+
2. Install MLX:
   ```bash
   pip install mlx
   ```

3. For JAX Metal support:
   ```bash
   pip install jax-metal
   ```

### Common Issues

- **"No GPU detected"**: Check drivers and GPU libraries are installed
- **"Backend not available"**: Install the appropriate GPU library (cupy or mlx)
- **Memory errors**: Reduce grid size or use CPU for very large problems
- **Norm not conserved**: Check time step `dt` is small enough

## Advanced Features

### Mixed Precision

For even faster computation with slight accuracy trade-off:

```python
# MLX uses float32 by default for better Metal performance
# CuPy can be configured for mixed precision
```

### Multi-GPU Support

Coming soon: Support for multiple GPUs and distributed computing.

## Benchmarking

Run the included benchmark:

```python
import Qsolve

# Compare CPU vs GPU performance
Qsolve.benchmark_gpu("hydrogen", grid_points=2000)
```

Or from command line:
```bash
python -m Qsolve.gpu --benchmark
```