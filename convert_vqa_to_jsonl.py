#!/usr/bin/env python3
"""Convert VQA v2 Parquet with binary images to JSONL format."""

import pandas as pd
import pyarrow.parquet as pq
import json
import os
import glob
from PIL import Image
from io import BytesIO


def save_image_from_bytes(image_bytes, output_dir, image_id):
    """Save image from bytes to disk."""
    try:
        image = Image.open(BytesIO(image_bytes))
        image_path = os.path.join(output_dir, f"{image_id}.jpg")
        image.save(image_path, "JPEG")
        return image_path
    except Exception as e:
        print(f"Error saving image {image_id}: {e}")
        return None


def convert_parquet_to_jsonl(parquet_dir, output_file, image_output_dir, max_samples=None):
    """Convert VQA v2 Parquet files with binary images to JSONL format."""
    # Create image output directory if it doesn't exist
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Get all Parquet files
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} Parquet files")
    
    if not parquet_files:
        print("No Parquet files found")
        return
    
    with open(output_file, "w") as fout:
        sample_count = 0
        
        for file_idx, file_path in enumerate(parquet_files):
            print(f"Processing file {file_idx + 1}/{len(parquet_files)}: {os.path.basename(file_path)}")
            
            # Read Parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            for _, row in df.iterrows():
                # Save image from bytes
                if isinstance(row["image"], dict) and "bytes" in row["image"]:
                    image_bytes = row["image"]["bytes"]
                else:
                    image_bytes = row["image"]
                
                image_path = save_image_from_bytes(image_bytes, image_output_dir, row["image_id"])
                
                if not image_path:
                    continue  # Skip if image saving fails
                
                # Create JSONL entry
                entry = {
                    "question_id": row["question_id"],
                    "category": "vqa",
                    "image": image_path,
                    "turns": [row["question"]],
                    "reference": []
                }
                
                # Add reference answers if available
                if row["answers"] is not None:
                    entry["reference"] = [ans["answer"] for ans in row["answers"]]  # Adjust based on actual structure
                elif row["multiple_choice_answer"] is not None:
                    entry["reference"] = [row["multiple_choice_answer"]]
                
                # Write to JSONL file
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
                sample_count += 1
                if max_samples and sample_count >= max_samples:
                    print(f"\nReached maximum sample count: {max_samples}")
                    return
    
    print(f"\nConverted {sample_count} samples to {output_file}")
    print(f"Saved images to {image_output_dir}")


def main():
    # Configuration - adjust these paths
    parquet_dir = "/workspace/datasets/vqav2/data"
    output_file = "/workspace/datasets/vqav2/vqa_questions.jsonl"
    image_output_dir = "/workspace/datasets/vqav2/images"
    max_samples = None  # Set to a number for testing, None for all samples
    
    convert_parquet_to_jsonl(parquet_dir, output_file, image_output_dir, max_samples)


if __name__ == "__main__":
    main()
