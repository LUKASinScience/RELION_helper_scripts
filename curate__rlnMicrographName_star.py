#!/usr/bin/env python3
"""
Curate a .star file for Warp particle extraction: rename data_ to data_particles
and replace micrograph paths with their corresponding .tomostar filenames
based on the position prefix (e.g., pos38_ts_003_unsorted).

Script by Lukas W. Bauer, 2025, with ChatGPT-4 as copilot.
No guaranteed success. Use at your own risk.
"""
import os
import argparse
import re
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Curate .star for Warp particle extraction.")
    parser.add_argument("input_star", help="Input .star file")
    parser.add_argument("output_star", help="Output .star file to write")
    parser.add_argument("tomostar_dir", help="Directory containing .tomostar files (left untouched)")
    args = parser.parse_args()

    input_star = Path(args.input_star)
    output_star = Path(args.output_star)
    tomostar_dir = Path(args.tomostar_dir)

    if not input_star.is_file():
        print(f"ERROR: Input star file '{input_star}' does not exist or is not a file.", file=sys.stderr)
        sys.exit(1)
    if not tomostar_dir.is_dir():
        print(f"ERROR: Tomostar directory '{tomostar_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Gather available tomostar stems (without extension)
    tomostar_files = [p for p in tomostar_dir.iterdir() if p.is_file() and p.suffix == ".tomostar"]
    tomostar_stems = {p.stem for p in tomostar_files}
    if not tomostar_stems:
        print(f"WARNING: No .tomostar files found in '{tomostar_dir}'. Micrograph names will not be updated.", file=sys.stderr)

    # Read input star file
    with open(input_star, newline='') as f:
        lines = f.read().splitlines()

    # Replace first line starting with data_ to data_particles
    data_line_replaced = False
    for idx, line in enumerate(lines):
        if re.match(r"^data_.*", line):
            if line.strip() != "data_particles":
                print(f"Replacing line {idx+1} '{line}' -> 'data_particles'")
                lines[idx] = "data_particles"
            else:
                print(f"Line {idx+1} already 'data_particles'; no change.")
            data_line_replaced = True
            break
    if not data_line_replaced:
        print("WARNING: No line starting with 'data_' found; did not replace data section name.", file=sys.stderr)

    # Locate loop_ and header labels
    loop_idx = None
    label_lines = []
    for i, line in enumerate(lines):
        if line.strip() == "loop_":
            loop_idx = i
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("_"):
                    label_lines.append(lines[j])
                else:
                    break
            break
    if loop_idx is None:
        print("ERROR: Could not find 'loop_' line in star file header.", file=sys.stderr)
        sys.exit(1)

    # Parse label ordering via # numbering
    numbered = []
    for lbl in label_lines:
        m = re.match(r"^(_\S+)\s*#\s*(\d+)", lbl)
        if not m:
            m = re.match(r"^(_\S+)\s*#(\d+)", lbl)
        if not m:
            print(f"WARNING: Could not parse label line: '{lbl}'. Skipping.", file=sys.stderr)
            continue
        name, num = m.group(1), int(m.group(2))
        numbered.append((num, name))
    if not numbered:
        print("ERROR: No valid label lines parsed.", file=sys.stderr)
        sys.exit(1)
    numbered.sort(key=lambda x: x[0])
    ordered_labels = [name for (_, name) in numbered]
    try:
        micro_idx = ordered_labels.index("_rlnMicrographName")
    except ValueError:
        print("ERROR: '_rlnMicrographName' label not found among header labels.", file=sys.stderr)
        sys.exit(1)

    # Data rows start after loop_ + label lines
    data_start = loop_idx + 1 + len(label_lines)
    replaced_count = 0
    missing_match = 0
    malformed = 0

    pos_regex = re.compile(r"(pos\d+_ts_\d+_unsorted)")

    for i in range(data_start, len(lines)):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) <= micro_idx:
            malformed += 1
            print(f"WARNING: Line {i+1} is too short to contain micrograph name: '{line}'", file=sys.stderr)
            continue
        original_micro = parts[micro_idx]
        m = pos_regex.search(original_micro)
        if not m:
            missing_match += 1
            print(f"WARNING: Could not extract position prefix from micrograph '{original_micro}' on line {i+1}. Leaving unchanged.", file=sys.stderr)
            continue
        prefix = m.group(1)
        candidate = f"{prefix}.tomostar"
        if prefix in tomostar_stems:
            if parts[micro_idx] != candidate:
                parts[micro_idx] = candidate
                replaced_count += 1
                lines[i] = "\t".join(parts)
        else:
            missing_match += 1
            print(f"WARNING: No matching .tomostar file for prefix '{prefix}' (expected '{candidate}') on line {i+1}. Leaving unchanged.", file=sys.stderr)

    # Write output star
    with open(output_star, "w", newline="\n") as out:
        for l in lines:
            out.write(l + "\n")

    print(f"Done. Replaced {replaced_count} micrograph name(s). {missing_match} line(s) had no valid tomostar match. {malformed} malformed line(s) skipped.")
    if replaced_count == 0:
        print("NOTE: No micrograph names were updated; check that your tomostar directory and naming pattern match the input star.", file=sys.stderr)

if __name__ == "__main__":
    main()
