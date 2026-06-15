#!/usr/bin/env python

import argparse
import pandas as pd
from project.data import load_fasta_to_df
import os

def get_sequences(positive_fasta_file, low_quality_positives_fasta_file, output_csv_file, min_length, max_length):
    # Load data from positive and negative CSV files
    positive_df = load_fasta_to_df(positive_fasta_file)

    # Add a column 'IsAMP' to label the data
    positive_df['IsAMP'] = 1
    
    # Filter sequences based on their length
    filtered_positive_df = positive_df[positive_df['Sequence'].apply(lambda x: min_length <= len(x) <= max_length)].drop(columns=["Id"])

    if low_quality_positives_fasta_file and os.path.exists(low_quality_positives_fasta_file):
        print("Low quality positives dataset provided. Saving both high and low quality datasets.")
        
        low_quality_df = load_fasta_to_df(low_quality_positives_fasta_file)

        print(f"Loaded {len(low_quality_df)} low quality sequences")
    
        low_quality_df['IsAMP'] = 0
    
        filtered_low_quality_df = low_quality_df[low_quality_df['Sequence'].apply(lambda x: min_length <= len(x) <= max_length)].drop(columns=["Id"])

        # Concatenate both datasets
        filtered_df = pd.concat([filtered_positive_df, filtered_low_quality_df], ignore_index=True)
    else:
        print("No low quality dataset provided. Saving only the high quality dataset.")

        filtered_df = filtered_positive_df
    
    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_csv_file, index=False)
    print(f"Filtered data saved to {output_csv_file}")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process FASTA files and filter data by sequence length.')
    parser.add_argument('--positive_fasta_file', type=str, default='data/generative-model-data/AMPs.fasta',
                        help='Path to the positive FASTA file')
    parser.add_argument('--low_quality_positives_fasta_file', type=str, default='data/generative-model-data/LQ-AMPs.fasta',
                        help='Path to the low quality positives FASTA file')
    parser.add_argument('--output_csv_file', type=str, default='data/generative-model-data/generative-model-dataset.csv',
                        help='Output CSV file path')
    parser.add_argument('--min_length', type=int, default=0,
                        help='Minimum sequence length (default: 0)')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum sequence length (default: 100)')
    
    # Parse arguments
    args = parser.parse_args()

    # Execute filtering
    get_sequences(args.positive_fasta_file, args.low_quality_positives_fasta_file, args.output_csv_file, args.min_length, args.max_length)

if __name__ == "__main__":
    main()
