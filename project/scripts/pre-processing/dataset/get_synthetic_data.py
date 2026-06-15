#!/usr/bin/env python

import argparse
from project.synthetic_data import (
    generate_random_sequences_length_adjusted,
    generate_shuffled_sequences,
    generate_mutated_sequences,
    generate_sequences_with_addition_deletion
)
from project.data import load_fasta_to_df

def main():
    parser = argparse.ArgumentParser(description='Generate non-AMP sequences emulating the length distribution of AMP sequences and filter by pseudo perplexity.')
    parser.add_argument('--path_to_amps_fasta', type=str, default='data/activity-data/curated-AMPs.fasta', help='Path to the AMP FASTA file')
    parser.add_argument('--path_to_fasta', type=str, default='data/activity-data/synthetic-data/random-sequences.fasta', help='Path to the output FASTA file for synthetic sequences')
    parser.add_argument('--number_of_non_amps', type=int, default=100000, help='Number of non-AMP sequences to generate')
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'shuffled', 'mutated', 'added-deleted'], help='Mode for generating non-AMP sequences')
    parser.add_argument('--length_filter', type=int, default=5, help='Length for filtering sequences')
    parser.add_argument('--mutations', type=int, default=5, help='Number of mutations per sequence (only for mutated mode)')
    parser.add_argument('--additions', type=int, default=5, help='Number of additions per sequence (only for added-deleted mode)')

    args = parser.parse_args()

    amp_df = load_fasta_to_df(args.path_to_amps_fasta)

    if args.length_filter > 0:
        amp_df = amp_df[amp_df['Sequence'].apply(len) >= args.length_filter]

    amp_sequences = amp_df['Sequence'].values.tolist()
    ids = amp_df['Id'].values.tolist() 

    non_amp_sequences = []

    if args.mode == 'random':
        non_amp_sequences = generate_random_sequences_length_adjusted(
            args.number_of_non_amps, amp_sequences
        )
        original_ids = [i for i in range(len(non_amp_sequences))] 
    elif args.mode == 'shuffled':
        original_ids, non_amp_sequences = generate_shuffled_sequences(
            args.number_of_non_amps, ids, amp_sequences
        )
    elif args.mode == 'mutated':
        original_ids, non_amp_sequences = generate_mutated_sequences(
            args.number_of_non_amps, ids, amp_sequences, args.mutations
        )
    elif args.mode == 'added-deleted':
        original_ids, non_amp_sequences = generate_sequences_with_addition_deletion(
            args.number_of_non_amps, ids, amp_sequences, length_adjustment=args.additions
        )

    with open(args.path_to_fasta, 'w') as output_file:
        for i, seq in enumerate(non_amp_sequences):
            output_file.write(f'>{args.mode}-{original_ids[i]}\n{seq}\n')

    print(f"Generated {len(non_amp_sequences)}/{args.number_of_non_amps} non-AMP sequences saved to {args.path_to_fasta}")

if __name__ == "__main__":
    main()
