#!/usr/bin/env python

import argparse
import pandas as pd
import h5py
from project.wrappers import HydrophobicScaleWrapper, combine_scales
from project.constants import wimley_white_scale, wimley_white_scale_with_min_spacing, PADDING_VALUE, pI_scale, levitt_scale, aasi_scale, transmembrane_propensity_scale

def main(csv_file, scale, output_h5_file, max_length, batch_size=100):

    if scale == 'Wimley-White':
        hydrophobic_scale_model = HydrophobicScaleWrapper(wimley_white_scale, max_length, PADDING_VALUE)
    elif scale == 'Wimley-White-with-min-spacing':
        hydrophobic_scale_model = HydrophobicScaleWrapper(wimley_white_scale_with_min_spacing, max_length, PADDING_VALUE)
    elif scale == 'WWMS_and_pI':
        hydrophobic_scale_model = HydrophobicScaleWrapper(combine_scales([wimley_white_scale_with_min_spacing, pI_scale]), max_length, PADDING_VALUE)
    elif scale == 'WWMS_and_levitt':
        hydrophobic_scale_model = HydrophobicScaleWrapper(combine_scales([wimley_white_scale_with_min_spacing, levitt_scale]), max_length, PADDING_VALUE)
    elif scale == 'WWMS_and_aasi':
        hydrophobic_scale_model = HydrophobicScaleWrapper(combine_scales([wimley_white_scale_with_min_spacing, aasi_scale]), max_length, PADDING_VALUE)
    elif scale == 'WWMS_and_transmembrane_propensity':
        hydrophobic_scale_model = HydrophobicScaleWrapper(combine_scales([wimley_white_scale_with_min_spacing, transmembrane_propensity_scale]), max_length, PADDING_VALUE)
    elif scale == 'WWMS_pI_levitt_transmembrane_propensity_aasi':
        hydrophobic_scale_model = HydrophobicScaleWrapper(combine_scales([wimley_white_scale_with_min_spacing, pI_scale, levitt_scale, transmembrane_propensity_scale, aasi_scale]), 
                                                            max_length, PADDING_VALUE)
    else:
        raise ValueError("Scale not found")

    # Initialize HDF5 file for writing
    with h5py.File(output_h5_file, 'w') as h5f:
        # Process the CSV in chunks
        with pd.read_csv(csv_file, chunksize=batch_size) as reader:
            for i, chunk in enumerate(reader):
                # Get sequences
                sequences = chunk["Sequence"].tolist()
                
                # Generate embeddings
                embeddings = hydrophobic_scale_model.encode(sequences)
                
                # Save embeddings to HDF5
                if 'embeddings' not in h5f:
                    # Create dataset if it does not exist
                    max_shape = (None,) + embeddings.shape[1:]  # None implies extendable dimension
                    h5f.create_dataset('embeddings', data=embeddings.cpu().numpy(), maxshape=max_shape, chunks=True)
                else:
                    # Append to dataset
                    old_shape = h5f['embeddings'].shape[0]
                    new_shape = old_shape + embeddings.shape[0]
                    h5f['embeddings'].resize((new_shape,) + embeddings.shape[1:])
                    h5f['embeddings'][old_shape:new_shape] = embeddings.cpu().numpy()
                
                print("{} embeddings stored".format((i+1) * batch_size))

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Generate embeddings for sequences from a CSV file and store in HDF5 format.')
    parser.add_argument('--csv_file', type=str, default='data/generative-model-data/generative-model-dataset.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--scale', type=str, default='Wimley-White-with-min-spacing',
                        help='Model to get the embeddings from')
    parser.add_argument('--output_h5_file', type=str, default='data/generative-model-data/generative-model-embeddings.h5',
                        help='Output HDF5 file path')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length of sequences')

    # Parse arguments
    args = parser.parse_args()

    # Execute main function
    main(args.csv_file, args.scale, args.output_h5_file, args.max_length)
