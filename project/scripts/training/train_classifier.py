#!/usr/bin/env python

import argparse
import os
import pandas as pd
from project.data import load_fasta_to_df
from project.classifiers import AMPClassifier, HemolyticClassifier
from project.data import get_dataset_for_activity_classifier, get_input_features_labels_mask_high_quality_idxs
from project.constants import CLASSIFIER_MODELS, CLASSIFIER_DATASETS, MODEL_DIR

def main(classifier_name, with_secret_data, objective):
    # Load appropriate dataset based on classifier type
    if classifier_name == 'hemolytic-classifier':
        sequence_label_df = pd.read_csv(CLASSIFIER_DATASETS["toxicity"])
    else:
        sequence_label_df = pd.read_csv(CLASSIFIER_DATASETS["activity"])
    
    if with_secret_data:
        secret_data_df = pd.read_csv(CLASSIFIER_DATASETS["activity-secret"])
    else:
        secret_data_df = None

    if classifier_name == 'all':
        for classifier_name in CLASSIFIER_MODELS.keys():
            print(f"Training classifier: {classifier_name}")
            if classifier_name == 'hemolytic-classifier':
                dataset = pd.read_csv(CLASSIFIER_DATASETS["toxicity"])
                classifier = HemolyticClassifier(model_path=None)
            else:
                dataset = get_dataset_for_activity_classifier(classifier_name, sequence_label_df, secret_data_df=secret_data_df)
                classifier = AMPClassifier(model_path=None)

            input_features, labels, mask_high_quality_idxs = get_input_features_labels_mask_high_quality_idxs(dataset)
            
            feature_importances = classifier.train_classifier(input_features, labels, mask_high_quality_idxs=mask_high_quality_idxs, return_feature_importances=True, objective=objective)
            classifier.save(f"{MODEL_DIR}/{classifier_name}.json")
            
            columns = dataset.drop(columns=['Id', 'Sequence', 'label', 'high_quality']).columns
            print_feature_importances(columns, feature_importances)
            print(f"Finished training classifier: {classifier_name}\n")
    else:
        dataset = sequence_label_df if classifier_name == 'hemolytic-classifier' else get_dataset_for_activity_classifier(classifier_name, sequence_label_df, secret_data_df=secret_data_df)
        
        input_features, labels, mask_high_quality_idxs = get_input_features_labels_mask_high_quality_idxs(dataset)
        classifier = AMPClassifier(model_path=None) if classifier_name != 'hemolytic-classifier' else HemolyticClassifier(model_path=None)
        
        feature_importances = classifier.train_classifier(input_features, labels, mask_high_quality_idxs=mask_high_quality_idxs, return_feature_importances=True, objective=objective)
        classifier.save(f"{MODEL_DIR}/{classifier_name}.json")
        
        columns = dataset.drop(columns=['Id', 'Sequence', 'label', 'high_quality']).columns
        print_feature_importances(columns, feature_importances)

def print_feature_importances(feature_names, feature_importances):
    named_importances = list(zip(feature_names, feature_importances))
    named_importances_sorted = sorted(named_importances, key=lambda x: x[1], reverse=True)

    print("Feature Importances:")
    for name, importance in named_importances_sorted:
        print(f"{name}: {importance:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and validate an XGBoost classifier for AMPs using provided datasets.')
    parser.add_argument('--classifier', type=str, default='broad-classifier', help='Specify a classifier or "all" to run all classifiers')
    parser.add_argument('--with-secret-data', action='store_true', help='Use secret data for training')
    parser.add_argument('--objective', type=str, default='binary:logistic', choices=['focal_loss', 'binary:logistic'], help='The objective function to use for training.')
    args = parser.parse_args()

    main(args.classifier, args.with_secret_data, args.objective)
