"""
Data preprocessing script to filter samples by labels.

This script filters gene expression data and label files to keep only samples
with specified labels. For example, if you have "normal", "stage I", "stage II", 
and "stage III", you can filter to keep only "normal" and "stage I" samples.

Usage:
    python scripts/preprocess_filter_labels.py \\
        --config config.yaml 

    --data_file: Path to the gene expression data file (tab-separated, sample IDs as index)
    --label_file: Path to the label file (tab-separated, sample IDs in first column, labels in second)
    --output_data: Path to save filtered gene expression data
    --output_labels: Path to save filtered labels
    --keep_labels: One or more label names to keep (e.g., normal "stage I")
"""

import argparse
import pandas as pd
import os
from pathlib import Path
from omegaconf import OmegaConf



def filter_data_by_labels(data_file, label_file, output_data, output_labels, keep_labels):
    """
    Filter gene expression data and labels to keep only specified labels.
    
    :param data_file: Path to gene expression data file (tab-separated, sample IDs as index)
    :param label_file: Path to label file (tab-separated, sample IDs in first column, labels in second)
    :param output_data: Path to save filtered gene expression data
    :param output_labels: Path to save filtered labels
    :param keep_labels: List of label names to keep
    """
    # Read label file
    print(f"Reading label file: {label_file}")
    df_labels = pd.read_csv(label_file, sep='\t', header=None)
    
    # Check format: should have at least 2 columns (sample ID and label)
    if df_labels.shape[1] < 2:
        raise ValueError(f"Label file must have at least 2 columns. Found {df_labels.shape[1]} columns.")
    
    # Get unique labels in the dataset
    unique_labels = sorted(set(df_labels[1]))
    print(f"Found {len(unique_labels)} unique labels in dataset: {unique_labels}")
    
    # Validate that all requested labels exist
    missing_labels = [label for label in keep_labels if label not in unique_labels]
    if missing_labels:
        raise ValueError(f"The following labels were not found in the dataset: {missing_labels}")
    
    # Filter labels to keep only specified ones
    print(f"Filtering to keep labels: {keep_labels}")
    mask = df_labels[1].isin(keep_labels)
    filtered_labels = df_labels[mask].copy()
    
    print(f"Original samples: {len(df_labels)}, Filtered samples: {len(filtered_labels)}")
    
    # Get sample IDs to keep
    sample_ids_to_keep = set(filtered_labels[0].values)
    
    # Read gene expression data
    print(f"Reading gene expression data file: {data_file}")
    df_data = pd.read_csv(data_file, sep='\t', index_col=0)
    
    print(f"Original data shape: {df_data.shape[0]} samples x {df_data.shape[1]} genes")
    
    # Filter data to keep only samples with specified labels
    # Check if all sample IDs from labels exist in data
    missing_samples = sample_ids_to_keep - set(df_data.index)
    if missing_samples:
        print(f"Warning: {len(missing_samples)} sample IDs in label file not found in data file.")
        print(f"First few missing samples: {list(missing_samples)[:5]}")
        # Remove missing samples from filtered_labels
        filtered_labels = filtered_labels[filtered_labels[0].isin(df_data.index)]
        sample_ids_to_keep = set(filtered_labels[0].values)
    
    # Filter data
    filtered_data = df_data.loc[df_data.index.isin(sample_ids_to_keep)].copy()
    
    print(f"Filtered data shape: {filtered_data.shape[0]} samples x {filtered_data.shape[1]} genes")
    
    # Ensure filtered_labels matches filtered_data order (if needed)
    # Sort both by sample ID to maintain consistency
    filtered_data = filtered_data.sort_index()
    filtered_labels = filtered_labels.sort_values(by=0)
    
    # Verify consistency
    data_sample_ids = set(filtered_data.index)
    label_sample_ids = set(filtered_labels[0].values)
    
    if data_sample_ids != label_sample_ids:
        print("Warning: Some samples in data don't have labels or vice versa.")
        # Keep only samples that exist in both
        common_samples = data_sample_ids & label_sample_ids
        filtered_data = filtered_data.loc[filtered_data.index.isin(common_samples)]
        filtered_labels = filtered_labels[filtered_labels[0].isin(common_samples)]
        print(f"After alignment: {len(common_samples)} samples")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_data) if os.path.dirname(output_data) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_labels) if os.path.dirname(output_labels) else '.', exist_ok=True)
    
    # Save filtered data
    print(f"Saving filtered gene expression data to: {output_data}")
    filtered_data.to_csv(output_data, sep='\t')
    
    # Save filtered labels (without header, maintaining original format)
    print(f"Saving filtered labels to: {output_labels}")
    filtered_labels.to_csv(output_labels, sep='\t', header=False, index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Filtering Summary:")
    print("="*60)
    print(f"Original samples: {len(df_labels)}")
    print(f"Filtered samples: {len(filtered_labels)}")
    print(f"\nLabel distribution in filtered data:")
    label_counts = filtered_labels[1].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  {label}: {count} samples")
    print("="*60)
    
    return filtered_data, filtered_labels

def main(args):
    basic_conf = create_config()
    input_conf = OmegaConf.load(args.config)
    config = OmegaConf.merge(basic_conf, input_conf)
    filter_data_by_labels(
        config.data_file,
        config.label_file,
        config.output_data,
        config.output_labels,
        config.keep_labels
    )
    
    print("\nFiltering completed successfully!")
    

def create_config():
    
    defaults = {
        "data_file": "data/example_train.txt",
        "label_file": "data/example_train_labels.txt",
        "output_data": "data/example_train_filtered.txt",
        "output_labels": "data/example_train_labels_filtered.txt",
        "keep_labels": ["normal", "tumor"]
    }
    return defaults 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args)