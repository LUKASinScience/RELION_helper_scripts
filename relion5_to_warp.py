"""Convert RELION5 subtomogram STAR to WarpTools-compatible format for ts_export_particles.

WarpTools ts_export_particles --input_star crashes with RELION5 run_data.star because
the worker internally writes rlnTomoParticleId, rlnTomoParticleName, rlnImageName, and
rlnTomoVisibleFrames into its temp table, then tries to copy those same columns from the
input STAR — resulting in "An item with the same key has already been added".

This script makes the minimal changes required:
  1. rlnTomoName        -> rlnMicrographName  (WarpTools matches particles by this)
  2. Drop the 4 colliding columns:
       rlnTomoParticleId, rlnTomoParticleName, rlnImageName, rlnTomoVisibleFrames

Everything else — coordinates, Euler angles, shifts, class numbers, optics — is left
completely unchanged.

Reference: https://warpem.github.io/warp/reference/warptools/tomogram_particle_files/
Bug report: https://groups.google.com/g/warp-em/c/AcXXqe_I8uc

Usage:
    python3 relion5_to_warp.py particles.star particles_warp.star

Script by Lukas W. Bauer and Claude (Anthropic), Weiss Group UZH, 2026.
No success guaranteed. Use at your own risk.
"""

import sys
import starfile
import pandas as pd


# Columns that WarpTools writes internally — cause AddColumn duplicate-key crash
COLLIDING_COLS = {
    'rlnTomoParticleId',
    'rlnTomoParticleName',
    'rlnImageName',
    'rlnTomoVisibleFrames',
}


def convert(input_star: str, output_star: str) -> None:
    data = starfile.read(input_star)

    if not isinstance(data, dict) or 'particles' not in data:
        print("ERROR: No 'particles' block found.")
        sys.exit(1)

    particles = data['particles'].copy()
    print(f"Input : {len(particles)} particles, {len(particles.columns)} columns")

    # 1. Rename rlnTomoName -> rlnMicrographName
    if 'rlnTomoName' not in particles.columns:
        print("ERROR: rlnTomoName not found.")
        sys.exit(1)
    particles = particles.rename(columns={'rlnTomoName': 'rlnMicrographName'})
    print("Renamed : rlnTomoName -> rlnMicrographName")

    # 2. Drop colliding columns
    to_drop = [c for c in COLLIDING_COLS if c in particles.columns]
    particles = particles.drop(columns=to_drop)
    print(f"Dropped : {to_drop}")

    # Write — preserve all other blocks (optics, general) unchanged
    out_data = {k: v for k, v in data.items() if k != 'particles'}
    out_data['particles'] = particles
    starfile.write(out_data, output_star)

    print(f"Output: {len(particles)} particles, {len(particles.columns)} columns")
    print(f"Written to: {output_star}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 relion5_to_warp.py <input.star> <output.star>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
