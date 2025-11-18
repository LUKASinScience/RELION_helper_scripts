#!/usr/bin/env python3
"""
Merge identical optics groups across particles.star and tomograms.star
while preserving original STAR file structure and formatting.

- Particles.star is left unchanged except for updated optics group names/IDs
- Tomograms.star is updated with merged optics group names/IDs
- All original data blocks are preserved in both outputs

Script by Lukas W. Bauer & ChatGPT, 2025. Use at your own risk.
"""

# ============ USER INPUT ============
PARTICLES_IN  = "merged_joinstar_particles.star"
TOMOS_IN      = "merged_tomogramset_tomograms.star"
PARTICLES_OUT = "merged_same_optics_particles.star"
TOMOS_OUT     = "merged_same_optics_tomograms.star"
DRY_RUN       = False
# ====================================

import sys
import pandas as pd
from collections import OrderedDict
import starfile

REQ_OPTICS = {"rlnOpticsGroup", "rlnOpticsGroupName"}
REQ_PARTICLES = {"rlnOpticsGroup", "rlnTomoName"}
REQ_TOMOS = {"rlnOpticsGroupName", "rlnTomoName"}

def _load_star(path):
    data = starfile.read(path)
    return OrderedDict(data) if isinstance(data, dict) else OrderedDict([("data", data)])

def _find_block_with(blocks, required_cols):
    for name, df in blocks.items():
        if isinstance(df, pd.DataFrame) and required_cols.issubset(df.columns):
            return name, df
    return None, None

def main():
    print("=== Merging identical RELION optics groups ===")
    print(f" Input particles : {PARTICLES_IN}")
    print(f" Input tomograms : {TOMOS_IN}")
    print(f" Output particles: {PARTICLES_OUT}")
    print(f" Output tomograms: {TOMOS_OUT}")
    print(f" Mode: {'DRY-RUN (no files written)' if DRY_RUN else 'LIVE MODE'}")
    print("-------------------------------------------------------------")

    # Load particle STAR blocks
    part_blocks = _load_star(PARTICLES_IN)
    par_block, particles = _find_block_with(part_blocks, REQ_PARTICLES)
    opt_block, optics = _find_block_with(part_blocks, REQ_OPTICS)

    if any(df is None for df in [particles, optics]):
        sys.exit("ERROR: particles.star is missing required tables.")

    # Load tomo STAR blocks
    tomo_blocks = _load_star(TOMOS_IN)
    tomo_block, tomos = _find_block_with(tomo_blocks, REQ_TOMOS)
    if tomos is None:
        sys.exit("ERROR: tomograms.star is missing required tables.")

    # Hash optics by all metadata except group name/id
    meta_cols = [c for c in optics.columns if c not in {"rlnOpticsGroup", "rlnOpticsGroupName"}]
    optics["_hash"] = optics[meta_cols].astype(str).agg("|".join, axis=1)

    # Unique optics
    unique_optics = optics.drop_duplicates("_hash").reset_index(drop=True).copy()
    unique_optics["rlnOpticsGroup"] = range(1, len(unique_optics)+1)
    unique_optics["rlnOpticsGroupName"] = [f"opticsGroup{i}" for i in unique_optics["rlnOpticsGroup"]]

    # Mapping from old -> new
    mapping = optics[["rlnOpticsGroup", "rlnOpticsGroupName", "_hash"]].merge(
        unique_optics[["_hash", "rlnOpticsGroup", "rlnOpticsGroupName"]], on="_hash",
        suffixes=("_old", "_new")
    )

    print("\nMerged Optics Group Mapping:")
    for _, row in mapping.drop_duplicates("rlnOpticsGroup_old").iterrows():
        if row["rlnOpticsGroup_old"] != row["rlnOpticsGroup_new"]:
            print(f"  - {row['rlnOpticsGroupName_old']} → {row['rlnOpticsGroupName_new']}")
    if len(mapping.query("rlnOpticsGroup_old != rlnOpticsGroup_new")) == 0:
        print("  No merges were necessary.")

    # Update particles group ID only (not names)
    remap_ids = dict(zip(mapping["rlnOpticsGroup_old"], mapping["rlnOpticsGroup_new"]))
    particles["rlnOpticsGroup"] = particles["rlnOpticsGroup"].map(remap_ids).astype(int)

    # Update tomograms group name
    name_map = dict(zip(mapping["rlnOpticsGroupName_old"], mapping["rlnOpticsGroupName_new"]))
    tomos["rlnOpticsGroupName"] = tomos["rlnOpticsGroupName"].map(name_map).fillna(tomos["rlnOpticsGroupName"])

    if DRY_RUN:
        print("\nDRY-RUN: No changes written.")
        return

    # Save particle.star with original blocks but new optics + particles
    part_blocks[opt_block] = unique_optics.drop(columns="_hash")
    part_blocks[par_block] = particles
    starfile.write(part_blocks, PARTICLES_OUT, overwrite=True)

    # Save tomograms.star with updated tomos table only, preserving all blocks
    for name, df in tomo_blocks.items():
        if isinstance(df, pd.DataFrame) and set(df.columns).issuperset(REQ_TOMOS):
            tomo_blocks[name] = tomos
    starfile.write(tomo_blocks, TOMOS_OUT, overwrite=True)

    print(f"\n✔ Written: {PARTICLES_OUT}")
    print(f"✔ Written: {TOMOS_OUT}")

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 200)
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
