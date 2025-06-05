#!/usr/bin/env python3
"""
Qsolve CLI - Command line interface for quick quantum calculations

Examples:
    qsolve-cli hydrogen
    qsolve-cli harmonic --states 5
    qsolve-cli double_well --grid 501 --plot
    qsolve-cli list
"""

import argparse
import sys
import numpy as np
import Qsolve


def list_systems():
    """List all available quantum systems"""
    systems = Qsolve.list_potentials()
    print("\nAvailable quantum systems:")
    print("-" * 30)
    for name in sorted(systems):
        print(f"  {name}")
    print(f"\nTotal: {len(systems)} systems")
    print("\nExample: qsolve-cli hydrogen --states 3")


def format_energy(energy, precision=6):
    """Format energy value with appropriate precision"""
    return f"{energy:.{precision}f}"


def main():
    parser = argparse.ArgumentParser(
        description="Qsolve CLI - Quick quantum calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qsolve-cli hydrogen                    # Solve hydrogen ground state
  qsolve-cli harmonic --states 5         # Get 5 harmonic oscillator states
  qsolve-cli double_well --grid 501      # Use 501 grid points
  qsolve-cli morse --plot                # Show wavefunction plot
  qsolve-cli list                        # List all available systems
        """
    )
    
    # Positional argument
    parser.add_argument(
        'system',
        nargs='?',
        help='Quantum system to solve (e.g., hydrogen, harmonic, double_well)'
    )
    
    # Optional arguments
    parser.add_argument(
        '-n', '--states',
        type=int,
        default=1,
        help='Number of eigenstates to compute (default: 1)'
    )
    
    parser.add_argument(
        '-g', '--grid',
        type=int,
        default=301,
        help='Number of grid points (default: 301)'
    )
    
    parser.add_argument(
        '-b', '--bounds',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        help='Grid boundaries (default: system-specific)'
    )
    
    parser.add_argument(
        '-p', '--plot',
        action='store_true',
        help='Display wavefunction plot'
    )
    
    parser.add_argument(
        '-s', '--save',
        type=str,
        metavar='FILE',
        help='Save results to file'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show detailed information about the result'
    )
    
    parser.add_argument(
        '--uncertainty',
        action='store_true',
        help='Calculate position/momentum uncertainties'
    )
    
    parser.add_argument(
        '--expectation',
        action='store_true',
        help='Calculate expectation values'
    )

    parser.add_argument(
        '--grid-range',
        type=int,
        nargs=3,
        metavar=('START', 'STOP', 'STEP'),
        help='Range of grid points for batch runs'
    )

    parser.add_argument(
        '--param-range',
        action='append',
        nargs=4,
        metavar=('NAME', 'START', 'STOP', 'STEP'),
        help='Potential parameter range (can be repeated)'
    )

    parser.add_argument(
        '--batch-save',
        action='store_true',
        help='Automatically save batch results'
    )

    parser.add_argument(
        '--batch-plot',
        action='store_true',
        help='Generate comparison plot for batch run'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle special cases
    if not args.system or args.system == 'list':
        list_systems()
        return
    
    # Check if system exists
    available = Qsolve.list_potentials()
    if args.system not in available:
        print(f"Error: Unknown system '{args.system}'")
        print(f"Use 'qsolve-cli list' to see available systems")
        
        # Suggest similar names
        suggestions = [s for s in available if args.system.lower() in s.lower()]
        if suggestions:
            print(f"\nDid you mean: {', '.join(suggestions)}?")
        return 1
    
    # Solve the system
    try:
        # Build base kwargs
        base_kwargs = {
            'n_states': args.states,
        }
        if args.bounds:
            base_kwargs['bounds'] = tuple(args.bounds)

        # Determine grid values
        if args.grid_range:
            start, stop, step = args.grid_range
            grid_values = list(range(start, stop + step, step))
        else:
            grid_values = [args.grid]

        # Parameter ranges
        param_ranges = {}
        if args.param_range:
            for name, pstart, pstop, pstep in args.param_range:
                pstart = float(pstart)
                pstop = float(pstop)
                pstep = float(pstep)
                param_ranges[name] = np.arange(pstart, pstop + pstep, pstep)

        from itertools import product
        varying = bool(args.grid_range or param_ranges)
        combos = product(grid_values, *param_ranges.values())
        results = []

        for combo in combos:
            grid = combo[0]
            local_kwargs = base_kwargs | {'grid_points': grid}
            param_values = combo[1:]
            for name, val in zip(param_ranges.keys(), param_values):
                local_kwargs[name] = val

            desc = f"grid={grid}"
            for n, v in zip(param_ranges.keys(), param_values):
                desc += f", {n}={v}"
            print(f"\nSolving {args.system} system ({desc})...")
            result = Qsolve.solve_quick(args.system, **local_kwargs)
            results.append((combo, result))
            print(f"  E_0 = {format_energy(result.energy)}")

            if args.batch_save:
                fname = f"{args.system}_g{grid}"
                for n, v in zip(param_ranges.keys(), param_values):
                    fname += f"_{n}{v}"
                fname += ".npz"
                result.save(fname)
                print(f"  Saved {fname}")

        # If not varying parameters, handle analysis/plot/save normally
        if not varying:
            result = results[0][1]

            if args.info:
                print(f"\nSystem information:")
                print(f"  Grid points: {result.system.grid.points}")
                print(f"  Grid bounds: {result.system.grid.bounds}")
                print(f"  Grid spacing: {result.system.grid.dx:.6f}")
                if 'solver_time' in result.info:
                    print(f"  Solver time: {result.info['solver_time']:.3f}s")
                if 'method' in result.info:
                    print(f"  Method: {result.info['method']}")

            if args.uncertainty:
                print(f"\nUncertainty analysis (ground state):")
                print(f"  Δx = {result.position_uncertainty():.6f}")
                print(f"  Δp = {result.momentum_uncertainty():.6f}")
                print(f"  ΔxΔp = {result.uncertainty_product():.6f}")
                print(f"  (Heisenberg limit: 0.5)")

            if args.expectation:
                print(f"\nExpectation values (ground state):")
                print(f"  <x> = {result.position_expectation():.6f}")
                print(f"  <p> = {result.momentum_expectation():.6f}")
                if args.system == 'harmonic':
                    x2 = result.expectation_value(lambda x, psi: x**2 * psi)
                    print(f"  <x²> = {x2:.6f}")

            if args.save:
                result.save(args.save)
                print(f"\nResults saved to: {args.save}")
                if not args.save.endswith('.npz'):
                    print(f"  (saved as {args.save}.npz)")

            if args.plot:
                try:
                    result.plot()
                    print("\nPlot displayed. Close window to exit.")
                except Exception as e:
                    print(f"\nWarning: Could not display plot ({e})")
                    print("Make sure you have matplotlib installed and a display available.")

        if varying and args.batch_plot:
            if len(param_ranges) + (1 if args.grid_range else 0) != 1:
                print("\nComparison plot requires a single varying parameter")
            else:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                if args.grid_range and not param_ranges:
                    xs = [c[0] for c, _ in results]
                    label = 'grid points'
                else:
                    name = next(iter(param_ranges.keys()))
                    xs = [c[1] for c, _ in results]
                    label = name

                ys = [r.energy for _, r in results]
                plt.figure()
                plt.plot(xs, ys, marker='o')
                plt.xlabel(label)
                plt.ylabel('ground state energy')
                plt.tight_layout()
                plt.savefig('comparison.png')
                print('\nComparison plot saved to comparison.png')
        
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
