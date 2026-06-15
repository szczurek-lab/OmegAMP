#!/usr/bin/env python

import argparse
import os
import pandas as pd
from project.data import load_fasta_to_df
from project.classifiers import AMPClassifier

def main(curated_amp_file_path, curated_non_amp_file_path, non_amp_file_path, random_sequences_file_path, shuffled_amp_sequences_file_path, mutated_amp_sequences_file_path, output_csv):
    # Load required positive examples
    if not os.path.exists(curated_amp_file_path):
        raise FileNotFoundError(f"Required curated AMP file not found: {curated_amp_file_path}")
    curated_amp_df = load_fasta_to_df(curated_amp_file_path)
    
    # Initialize list for negative datasets
    negative_dfs = []
    
    # Load and process each negative dataset if it exists
    if os.path.exists(curated_non_amp_file_path):
        print(f"Loading curated non-AMP file: {curated_non_amp_file_path}")
        curated_non_amp_df = load_fasta_to_df(curated_non_amp_file_path)
        curated_non_amp_df['label'] = 0
        curated_non_amp_df['high_quality'] = 1
        negative_dfs.append(curated_non_amp_df)
    
    if os.path.exists(non_amp_file_path):
        print(f"Loading non-AMP file: {non_amp_file_path}")
        non_amp_df = load_fasta_to_df(non_amp_file_path)
        non_amp_df['label'] = 0
        non_amp_df['high_quality'] = 0
        negative_dfs.append(non_amp_df)
    
    if os.path.exists(random_sequences_file_path):
        print(f"Loading random sequences file: {random_sequences_file_path}")
        random_sequences_df = load_fasta_to_df(random_sequences_file_path)
        random_sequences_df['label'] = 0
        random_sequences_df['high_quality'] = 0
        negative_dfs.append(random_sequences_df)
    
    if os.path.exists(shuffled_amp_sequences_file_path):
        print(f"Loading shuffled AMP sequences file: {shuffled_amp_sequences_file_path}")
        shuffled_amp_sequences_df = load_fasta_to_df(shuffled_amp_sequences_file_path)
        shuffled_amp_sequences_df['label'] = 0
        shuffled_amp_sequences_df['high_quality'] = 0
        negative_dfs.append(shuffled_amp_sequences_df)
    
    if os.path.exists(mutated_amp_sequences_file_path):
        print(f"Loading mutated AMP sequences file: {mutated_amp_sequences_file_path}")
        mutated_amp_sequences_df = load_fasta_to_df(mutated_amp_sequences_file_path)
        mutated_amp_sequences_df['label'] = 0
        mutated_amp_sequences_df['high_quality'] = 0
        negative_dfs.append(mutated_amp_sequences_df)

    if not negative_dfs:
        raise ValueError("No negative example files were found. At least one negative dataset is required.")

    # Process positive examples
    curated_amp_df['label'] = 1
    curated_amp_df['high_quality'] = 1
    
    # Combine negative datasets
    negative_df = pd.concat(negative_dfs, ignore_index=True)
    positive_df = curated_amp_df

    overlap = positive_df[positive_df['Sequence'].isin(negative_df['Sequence'])]

    print(f"Overlapping sequences found: {len(overlap)}")
        
    positive_df = positive_df[~positive_df['Sequence'].isin(overlap['Sequence'])]
    negative_df = negative_df[~negative_df['Sequence'].isin(overlap['Sequence'])]

    sequence_label_df = pd.concat([positive_df, negative_df], ignore_index=True)

    print(f"Duplicate sequences found: {len(sequence_label_df) - len(sequence_label_df.drop_duplicates(subset='Sequence'))}")

    sequence_label_df = sequence_label_df.drop_duplicates(subset='Sequence')
    sequence_label_df.reset_index(drop=True, inplace=True)

    classifier = AMPClassifier(model_path=None)

    sequences = sequence_label_df['Sequence'].tolist()

    print(f"Extracting features for {len(sequences)} sequences")

    features_df = classifier.get_input_features(sequences)

    sequence_label_df = pd.concat([sequence_label_df, features_df], axis=1)

    sequence_label_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and validate an XGBoost classifier for AMPs using provided datasets.')
    parser.add_argument('--curated_amp_file_path', type=str, default='data/activity-data/curated-AMPs.fasta', help='Path to the curated AMP FASTA file (default: data/curated-AMPs.fasta)')
    parser.add_argument('--curated_non_amp_file_path', type=str, default='data/activity-data/curated-Non-AMPs.fasta', help='Path to the curated non-AMP FASTA file (default: data/curated-Non-AMPs.fasta)')
    parser.add_argument('--non_amp_file_path', type=str, default='data/activity-data/Non-AMPs.fasta', help='Path to the non-AMP FASTA file (default: data/Non-AMPs.fasta)')
    parser.add_argument('--random_sequences_file_path', type=str, default='data/activity-data/synthetic-data/random-sequences.fasta', help='Path to the random sequences file (default: data/random-sequences.fasta)')
    parser.add_argument('--shuffled_amp_sequences_file_path', type=str, default='data/activity-data/synthetic-data/shuffled-AMP-sequences.fasta', help='Path to the shuffled AMP sequences file (default: data/shuffled-AMP-sequences.fasta)')
    parser.add_argument('--mutated_amp_sequences_file_path', type=str, default='data/activity-data/synthetic-data/mutated-AMP-sequences.fasta', help='Path to the mutated AMP sequences file (default: data/mutated-AMP-sequences.fasta)')
    parser.add_argument('--output_csv', type=str, default='data/activity-data/classifier-dataset.csv', help='Path where the classifier dataset will be saved (default: data/classifier-dataset.csv)')
    args = parser.parse_args()

    main(args.curated_amp_file_path, args.curated_non_amp_file_path, args.non_amp_file_path, args.random_sequences_file_path, args.shuffled_amp_sequences_file_path, args.mutated_amp_sequences_file_path, args.output_csv)
