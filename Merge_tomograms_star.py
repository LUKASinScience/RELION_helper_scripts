#!/usr/bin/env python3
# use at your own risk
# based on an internal script from the Structural Biochemistry group at Utrecht University


import sys
import re
import pandas as pd
import starfile  # Make sure you have the 'starfile' module installed

def extract_optics_group_number(name):
    """Extract the trailing number from optics group name like 'opticsGroup23' -> 23"""
    match = re.search(r'(\d+)$', name)
    return int(match.group(1)) if match else None

def increment_opticsgroup_name(name, offset):
    """Increment optics group name number by offset, e.g. opticsGroup23 + 10 -> opticsGroup33"""
    base = re.sub(r'(\d+)$', '', name)
    num = extract_optics_group_number(name)
    if num is None:
        raise ValueError(f"Cannot parse optics group number from '{name}'")
    return f"{base}{num + offset}"

def merge_star_files(input_files, output_file):
    merged_global_rows = []
    merged_data_blocks = {}
    opticsgroup_offset = 0

    global_columns = None
    data_columns_per_block = {}

    for filename in input_files:
        print(f"Reading {filename}")
        star = starfile.read(filename)

        # The global block is named 'global' in your files
        if 'global' not in star:
            raise KeyError(f"'global' block not found in {filename}")

        global_block = star['global']

        if global_columns is None:
            global_columns = list(global_block.columns)

        # Adjust opticsGroup names and collect global rows
        for _, row in global_block.iterrows():
            row = row.copy()
            old_name = row['rlnOpticsGroupName']
            new_name = increment_opticsgroup_name(old_name, opticsgroup_offset)
            row['rlnOpticsGroupName'] = new_name
            merged_global_rows.append(row)

        # For all other blocks except 'global', merge data blocks
        for block_name in star.keys():
            if block_name == 'global':
                continue

            block = star[block_name]

            if block_name not in merged_data_blocks:
                merged_data_blocks[block_name] = []
                data_columns_per_block[block_name] = list(block.columns)

            for _, row in block.iterrows():
                row = row.copy()
                if 'rlnOpticsGroupName' in block.columns:
                    old_name = row['rlnOpticsGroupName']
                    new_name = increment_opticsgroup_name(old_name, opticsgroup_offset)
                    row['rlnOpticsGroupName'] = new_name
                merged_data_blocks[block_name].append(row)

        # Update offset for next file based on max optics group number in current file
        max_group_num = 0
        for _, row in global_block.iterrows():
            num = extract_optics_group_number(row['rlnOpticsGroupName'])
            if num and num > max_group_num:
                max_group_num = num
        opticsgroup_offset += max_group_num

    # Create merged star dictionary with ordered columns preserved
    merged_star = {}
    merged_star['global'] = pd.DataFrame(merged_global_rows)[global_columns]

    for block_name, rows in merged_data_blocks.items():
        cols = data_columns_per_block[block_name]
        df = pd.DataFrame(rows)[cols]
        merged_star[block_name] = df

    starfile.write(merged_star, output_file)
    print(f"Merged STAR written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python Merge_tomograms_star.py input1.star input2.star [...] output.star")
        sys.exit(1)

    input_files = sys.argv[1:-1]
    output_file = sys.argv[-1]
    merge_star_files(input_files, output_file)
