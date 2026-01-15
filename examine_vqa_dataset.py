#!/usr/bin/env python3
"""Examine VQA v2 Parquet dataset structure."""

import pandas as pd
import pyarrow.parquet as pq
import os
import sys

def examine_single_parquet(file_path):
    """Examine a single Parquet file."""
    print(f"\n=== Examining {os.path.basename(file_path)} ===")
    
    # Read the Parquet file
    table = pq.read_table(file_path)
    df = table.to_pandas()
    
    # Print basic information
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"\nColumns: {', '.join(df.columns)}")
    
    # Print data types
    print("\nData types:")
    print(df.dtypes)
    
    # Print first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    return df.columns.tolist()

def examine_all_parquet(directory):
    """Examine all Parquet files in a directory."""
    parquet_files = sorted([f for f in os.listdir(directory) if f.endswith('.parquet')])
    print(f"Found {len(parquet_files)} Parquet files")
    
    if not parquet_files:
        print("No Parquet files found")
        return
    
    # Examine the first file
    first_file = os.path.join(directory, parquet_files[0])
    columns = examine_single_parquet(first_file)
    
    # Check if all files have the same structure
    print("\n=== Checking consistency across files ===")
    for i, file_name in enumerate(parquet_files[1:2]):  # Check next one file
        file_path = os.path.join(directory, file_name)
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        if list(df.columns) == columns:
            print(f"File {i+2} ({file_name}) has the same columns")
        else:
            print(f"File {i+2} ({file_name}) has different columns!")
            print(f"  Columns: {', '.join(df.columns)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python examine_vqa_dataset.py <parquet_directory>")
        return
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        return
    
    examine_all_parquet(directory)

if __name__ == "__main__":
    main()
