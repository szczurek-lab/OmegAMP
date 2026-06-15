#!/usr/bin/env python

import argparse
import os
import pandas as pd
from project.classifiers import AMPClassifier, HemolyticClassifier
from project.data import load_fasta_to_df
from project.constants import CLASSIFIER_MODELS

def main(fasta_file=None, classifier_choice=None, output_csv=None, predict_proba=False, batch_size=100000, sequences=None):
    if sequences is not None:
        sequences = pd.DataFrame({"Id": range(len(sequences)), "Sequence": sequences})
    elif fasta_file is not None:
        sequences = load_fasta_to_df(fasta_file)
    else:
        raise ValueError("Either fasta_file or sequences must be provided.")

    # Check if the user wants to run a single classifier or all classifiers
    if classifier_choice in CLASSIFIER_MODELS:
        model_path = CLASSIFIER_MODELS[classifier_choice]

        # Load the specified classifier model
        model = AMPClassifier(model_path=model_path) if classifier_choice != 'hemolytic-classifier' else HemolyticClassifier(model_path=model_path)
        model.eval()

        # Process sequences in batches
        all_predictions = []
        for i in range(0, len(sequences), batch_size):
            batch_df = sequences.iloc[i:i+batch_size]
            batch_sequences = batch_df["Sequence"].to_list()

            # Run inference for the batch
            if predict_proba:
                predictions = model.predict_proba(batch_sequences)
            else:
                predictions = model(batch_sequences)
            all_predictions.extend(predictions)

        sequences["Prediction"] = all_predictions

        if output_csv is not None:
            sequences.to_csv(output_csv, index=False)
            print(f"Predictions for {len(sequences)} sequences using classifier {classifier_choice} saved to {output_csv}")

        return sequences

    elif classifier_choice == "all":
        # Process sequences in batches
        all_results = []
        for i in range(0, len(sequences), batch_size):
            print(f"Processing batch {i//batch_size + 1}/{(len(sequences) + batch_size - 1)//batch_size}")
            batch_df = sequences.iloc[i:i+batch_size].copy()
            batch_sequences = batch_df["Sequence"].to_list()

            # Pre-compute AMP features for the batch
            amp_features = AMPClassifier(model_path=None).get_input_features(batch_sequences)

            # Iterate over all classifiers
            for classifier, model_path in CLASSIFIER_MODELS.items():
                model = AMPClassifier(model_path=model_path) if classifier != 'hemolytic-classifier' else HemolyticClassifier(model_path=model_path)
                model.eval()

                if classifier == 'hemolytic-classifier':
                    predictions = model.predict_proba(batch_sequences) if predict_proba else model(batch_sequences)
                else:
                    predictions = model.predict_from_features(amp_features, proba=predict_proba)
                
                batch_df[classifier] = predictions
            
            all_results.append(batch_df)

        final_results = pd.concat(all_results, ignore_index=True)
        if output_csv is not None:
            final_results.to_csv(output_csv, index=False)
            print(f"Predictions for {len(sequences)} sequences using all classifiers saved to {output_csv}")

        return final_results
    else:
        raise ValueError(f"Classifier {classifier_choice} not found. Available classifiers: {', '.join(CLASSIFIER_MODELS.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a classifier model or ensemble and run inference on protein sequences.')
    parser.add_argument('fasta_file', type=str, help='Path to the fasta file with sequences to predict')
    parser.add_argument('--classifier', type=str, default='broad-classifier', help='Specify a classifier or "all" to run all classifiers')
    parser.add_argument('--output_csv', type=str, default='results/classifier-results/script-classifier-results.csv', help='Path to the output CSV file for predictions')
    parser.add_argument('--predict_proba', action='store_true', help='Output probabilities instead of predictions')
    parser.add_argument('--batch_size', type=int, default=100000, help='Batch size for processing sequences')
    
    args = parser.parse_args()

    main(args.fasta_file, args.classifier, args.output_csv, args.predict_proba, args.batch_size)
