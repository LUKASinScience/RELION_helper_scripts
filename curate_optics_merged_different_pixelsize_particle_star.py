#!/usr/bin/env python3
"""
Remap particle optics group assignments to match tomograms.star,
and validate pixel size consistency in optics table.

What it does:
  - Ensures each rlnOpticsGroupName from tomograms.star exists in the particles.star optics table
  - If tilt pixel size mismatches occur, generates a new opticsGroupN (number-only)
  - Updates tomograms.star to use new optics group names if renamed
  - Remaps rlnOpticsGroup values in particles to match tomogram optics group names
  - Appends missing optics rows using a template (first row or matched tilt px)
  - Validates that: rlnImagePixelSize ≈ rlnTomoTiltSeriesPixelSize / rlnTomoSubtomogramBinning
  - Prompts user before writing any changes
  - Does not modify any pixel size values

Script by Lukas W. Bauer, 2025, with ChatGPT-4 as copilot.
No guaranteed success. Use at your own risk.
"""

# =========================
# User input (edit below)
# =========================
PARTICLES_IN   = "merged_joinstar_particles.star"
TOMOS_IN       = "merged_tomogramset_tomograms.star"
PARTICLES_OUT  = "curated_merged_joinstar_particles.star"
TOMOS_OUT      = "curated_merged_tomogramset_tomograms.star"

TEMPLATE_STRATEGY = "match_tilt"  # "match_tilt" or "first"
TILT_MATCH_TOL    = 1e-6
PIXEL_MATCH_TOL   = 1e-5
STRICT_MODE       = False
# =========================

import sys, math
from collections import OrderedDict
import pandas as pd

try:
    import starfile
except ImportError:
    print("ERROR: 'starfile' is required (pip install starfile)")
    sys.exit(1)

REQ_TOMO = {"rlnTomoName", "rlnOpticsGroupName", "rlnTomoTiltSeriesPixelSize"}
REQ_PART = {"rlnTomoName", "rlnOpticsGroup"}
REQ_OPT  = {
    "rlnOpticsGroup", "rlnOpticsGroupName", "rlnImagePixelSize",
    "rlnTomoTiltSeriesPixelSize", "rlnTomoSubtomogramBinning"
}

COL_TILT = "rlnTomoTiltSeriesPixelSize"
COL_BIN  = "rlnTomoSubtomogramBinning"
COL_IMG  = "rlnImagePixelSize"

def _load_star_any(path):
    data = starfile.read(path)
    return OrderedDict(data) if isinstance(data, dict) else OrderedDict([("data", data)])

def _find_block_with(blocks, required_cols):
    for name, df in sorted(blocks.items(), key=lambda x: -len(x[1]) if isinstance(x[1], pd.DataFrame) else 0):
        if isinstance(df, pd.DataFrame) and required_cols.issubset(df.columns):
            return name, df
    return None, None

def _to_float(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None

def _pick_template_row(optics_df, target_tilt):
    if len(optics_df) == 0:
        return None
    if TEMPLATE_STRATEGY == "match_tilt" and COL_TILT in optics_df.columns and target_tilt is not None:
        for i, v in optics_df[COL_TILT].items():
            vv = _to_float(v)
            if vv is not None and abs(vv - target_tilt) <= TILT_MATCH_TOL:
                return optics_df.loc[i]
    return optics_df.iloc[0]

def main():
    print("RELION optics group remapper + pixel size validator (numbered, dual-output version)")
    print(f"  Tomograms In : {TOMOS_IN}")
    print(f"  Particles In : {PARTICLES_IN}")
    print(f"  Particles Out: {PARTICLES_OUT}")
    print(f"  Tomograms Out: {TOMOS_OUT}")
    print("---------------------------------------------------")

    # Load tomograms
    tomo_blocks = _load_star_any(TOMOS_IN)
    tomo_block, tomos = _find_block_with(tomo_blocks, REQ_TOMO)
    if tomos is None:
        sys.exit("ERROR: tomograms.star is missing required columns.")

    tomos["rlnTomoName"] = tomos["rlnTomoName"].astype(str)
    tomos["rlnOpticsGroupName"] = tomos["rlnOpticsGroupName"].astype(str)

    # Load particles and optics
    part_blocks = _load_star_any(PARTICLES_IN)
    par_block, particles = _find_block_with(part_blocks, REQ_PART)
    opt_block, optics = _find_block_with(part_blocks, REQ_OPT)
    if particles is None or optics is None:
        sys.exit("ERROR: particles.star is missing required tables or columns.")

    particles["rlnTomoName"] = particles["rlnTomoName"].astype(str)
    optics["rlnOpticsGroupName"] = optics["rlnOpticsGroupName"].astype(str)

    existing_names = set(optics["rlnOpticsGroupName"].unique())
    name_to_index = dict(zip(optics["rlnOpticsGroupName"], optics["rlnOpticsGroup"]))
    current_max = int(pd.to_numeric(optics["rlnOpticsGroup"], errors="coerce").max() or 0)

    # Update tomograms with unique and pixel-consistent optics groups
    rename_map = {}
    new_rows = []

    for i, row in tomos.iterrows():
        orig_name = row["rlnOpticsGroupName"]
        tilt_px = _to_float(row.get(COL_TILT))
        if orig_name in optics["rlnOpticsGroupName"].values:
            existing_row = optics[optics["rlnOpticsGroupName"] == orig_name].iloc[0]
            existing_tilt = _to_float(existing_row.get(COL_TILT))
            if existing_tilt is not None and abs(existing_tilt - tilt_px) > TILT_MATCH_TOL:
                # Conflict: must create a new optics group with a new number
                current_max += 1
                new_name = f"opticsGroup{current_max}"
                template = _pick_template_row(optics, tilt_px)
                new_row = {c: pd.NA for c in optics.columns} if template is None else template.to_dict()
                new_row["rlnOpticsGroupName"] = new_name
                new_row["rlnOpticsGroup"] = current_max
                new_row[COL_TILT] = tilt_px
                new_rows.append(new_row)
                name_to_index[new_name] = current_max
                rename_map[i] = new_name
        else:
            # Missing group: assign new one
            current_max += 1
            new_name = f"opticsGroup{current_max}"
            template = _pick_template_row(optics, tilt_px)
            new_row = {c: pd.NA for c in optics.columns} if template is None else template.to_dict()
            new_row["rlnOpticsGroupName"] = new_name
            new_row["rlnOpticsGroup"] = current_max
            new_row[COL_TILT] = tilt_px
            new_rows.append(new_row)
            name_to_index[new_name] = current_max
            rename_map[i] = new_name

    for idx, new_name in rename_map.items():
        tomos.at[idx, "rlnOpticsGroupName"] = new_name

    # Apply mapping from tomogram to opticsGroup index
    tomo_to_groupname = dict(zip(tomos["rlnTomoName"], tomos["rlnOpticsGroupName"]))

    def _remap(row):
        tname = row["rlnTomoName"]
        gname = tomo_to_groupname.get(tname)
        return name_to_index.get(gname, row["rlnOpticsGroup"])

    particles["rlnOpticsGroup"] = particles.apply(_remap, axis=1)

    # Append new optics rows if needed
    if new_rows:
        optics = pd.concat([optics, pd.DataFrame(new_rows)], ignore_index=True)

    # Validate optics pixel sizes
    print("\nOptics group pixel size table:")
    print(f"{'GroupName':<20} {'TiltPx':>8} {'Binning':>8} {'ImgPx':>8} {'Expected':>10}  Status")
    for _, row in optics.iterrows():
        name = row["rlnOpticsGroupName"]
        tilt = _to_float(row.get(COL_TILT))
        binning = _to_float(row.get(COL_BIN))
        img = _to_float(row.get(COL_IMG))
        expected = tilt * binning if tilt and binning else None
        status = "OK"
        if img is None:
            status = "MISSING"
        elif expected and abs(expected - img) > PIXEL_MATCH_TOL:
            status = "MISMATCH"
        print(f"{name:<20} {tilt:8.3f} {binning:8.3f} {img:8.3f} {expected if expected else '':>10.3f}  {status}")

    try:
        confirm = input("\nApply and write both particle + tomogram outputs? (yes/no): ").strip().lower()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)

    if confirm != "yes":
        print("No changes written.")
        return

    # Save both updated files
    part_blocks[par_block] = particles
    part_blocks[opt_block] = optics
    starfile.write(part_blocks, PARTICLES_OUT, overwrite=True)

    tomo_blocks[tomo_block] = tomos
    starfile.write(tomo_blocks, TOMOS_OUT, overwrite=True)

    print(f"\n Written: {PARTICLES_OUT}")
    print(f" Written: {TOMOS_OUT}")
    print("Done. Your optics are now morally and numerically consistent.")

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 300)
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)


