#!/usr/bin/env python3
"""
Generate RNG Stream for LUMINA-SN Validation
=============================================

This script generates random numbers using NumPy's Mersenne Twister
(the same RNG used by TARDIS) and saves them to a file that the C
code can consume.

Usage:
    python generate_rng_stream.py [--seed N] [--count N] [--output FILE]
"""

import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Generate TARDIS-compatible RNG stream for LUMINA validation')
    parser.add_argument('--seed', type=int, default=23111963,
                       help='Random seed (default: 23111963)')
    parser.add_argument('--count', type=int, default=10000,
                       help='Number of random numbers (default: 10000)')
    parser.add_argument('--output', type=str, default='tardis_rng_stream.txt',
                       help='Output file (default: tardis_rng_stream.txt)')

    args = parser.parse_args()

    print(f"Generating {args.count} random numbers with seed {args.seed}...")

    # Use NumPy's Mersenne Twister (same as TARDIS)
    np.random.seed(args.seed)

    # Generate random numbers
    rng_values = np.random.random(args.count)

    # Write to file with high precision
    with open(args.output, 'w') as f:
        # Header with metadata
        f.write(f"# TARDIS RNG Stream\n")
        f.write(f"# Seed: {args.seed}\n")
        f.write(f"# Count: {args.count}\n")
        f.write(f"# Algorithm: NumPy Mersenne Twister (MT19937)\n")
        f.write(f"# Format: One double per line, %.18e precision\n")

        for val in rng_values:
            f.write(f"{val:.18e}\n")

    print(f"Wrote {args.count} values to: {args.output}")

    # Print first few values for verification
    print("\nFirst 10 values:")
    for i, val in enumerate(rng_values[:10]):
        print(f"  [{i}] {val:.18e}")

    # Show what mu and nu would be for packet 0
    print("\nExpected Packet 0 initialization:")
    xi_mu = rng_values[0]
    mu = np.sqrt(xi_mu)
    print(f"  xi[0] = {xi_mu:.18e}")
    print(f"  mu = sqrt(xi) = {mu:.18e}")

    # For blackbody sampling, TARDIS uses rejection sampling
    # so we can't easily predict which RNG values will be used
    print(f"  (nu sampling uses rejection method - varies)")

if __name__ == '__main__':
    main()
