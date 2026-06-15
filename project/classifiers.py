import numpy as np
from sklearn.utils import compute_class_weight
import torch.nn as nn
import xgboost as xgb
from xgboost import DMatrix
import torch
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from project.constants import HQ_AMPs_FILE
from project.data import get_signal_and_metabolic_sequences
from project.synthetic_data import generate_synthetic_sequences
from .sequence_properties import calculate_average_esm2_embeddings, calculate_physchem_prop, calculate_aa_frequency, calculate_positional_encodings
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef, precision_score, average_precision_score, roc_curve
    

def focal_loss(predt: np.ndarray, dtrain: xgb.DMatrix, gamma=2.0):
    """
    Computes the gradient and hessian for Focal Loss in XGBoost.

    The loss function is:
    -alpha * [(1-p)^gamma * y * log(p) + p^gamma * (1-y) * log(1-p)]

    This function computes the derivatives of the loss with respect to the
    raw margin score 'predt', not the probability 'p'.
    """

    # 1. Get true labels and alpha weights
    y = dtrain.get_label()
    alpha = dtrain.get_weight()

    # 2. Transform raw margin score 'predt' to probabilities 'p'
    p = 1.0 / (1.0 + np.exp(-predt))
    
    # Clip probabilities to avoid log(0) and division by zero
    p = np.clip(p, 1e-15, 1 - 1e-15)

    # 3. Compute the first derivative of the loss with respect to p (your original grad)
    grad_wrt_p = alpha * (
        -(p**gamma * (y - 1))/(1 - p) 
        + gamma * p**(gamma - 1) * (y - 1) * np.log(1 - p) 
        - ((1 - p)**gamma * y)/p 
        + gamma * (1 - p)**(gamma - 1) * y * np.log(p)
    )

    # 4. Compute the second derivative of the loss with respect to p (your original hess)
    hess_wrt_p = - alpha * (
        (p**gamma * (y - 1))/(1 - p)**2 
        + (2 * gamma * p**(gamma - 1) * (y - 1))/(1 - p) 
        - (gamma - 1) * gamma * p**(gamma - 2) * (y - 1) * np.log(1 - p) 
        - ((1 - p)**gamma * y)/p**2 
        - (2 * gamma * (1 - p)**(gamma - 1) * y)/p 
        + (gamma - 1) * gamma * (1 - p)**(gamma - 2) * y * np.log(p)
    )
    
    # 5. Apply the chain rule to get derivatives with respect to 'predt'
    # Derivative of sigmoid: p * (1 - p)
    sigmoid_deriv = p * (1 - p)

    # Gradient with respect to predt
    grad = grad_wrt_p * sigmoid_deriv

    # Hessian with respect to predt (using the approximation for stability)
    # hess ≈ (d²L/dp²) * (dp/dpredt)²
    hess = hess_wrt_p * (sigmoid_deriv**2) # + grad_wrt_p * (p * (1 - p) * (1 - 2*p))

    return grad, hess


def calculate_tpr_at_fpr(y_true, y_scores, target_fpr=0.01):
    """
    Calculate TPR at a specific FPR threshold.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        target_fpr: Target false positive rate (default 0.01 for 1%)
    
    Returns:
        TPR at the threshold that gives the target FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find the index where FPR is closest to target_fpr
    idx = np.argmin(np.abs(fpr - target_fpr))
    
    return tpr[idx]


class PeptideClassifier(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        if model_path is not None:
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
        else:
            self.model = xgb.XGBClassifier(eval_metric='logloss', 
                                         early_stopping_rounds=50, n_estimators=5000)
        self.model.n_classes_ = 2
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.decision_threshold = 0.5

    def get_input_features(self, sequences):
        """To be implemented by child classes"""
        raise NotImplementedError

    def train_classifier(self, input_features, labels, weight_balancing="balanced_with_adjustment_for_high_quality", mask_high_quality_idxs=[], return_feature_importances=False, verbose=True, objective='focal_loss'):
        """To be implemented by child classes"""
        raise NotImplementedError

    def eval_with_k_fold_cross_validation(self, input_features, labels, seed=42, weight_balancing="balanced_with_adjustment_for_high_quality", k=5, mask_high_quality_idxs=[], reference_file=HQ_AMPs_FILE, objective='focal_loss'):
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

        fold_rows = []
        
        all_accuracies = []
        all_f1_scores = []
        all_mcc_scores = []
        all_auprc_scores = []
        all_tpr_at_001_fpr = []
        all_confusion_matrices = []
        
        biological_accuracies = []
        biological_f1_scores = []
        biological_mcc_scores = []
        biological_auprc_scores = []
        biological_tpr_at_001_fpr = []
        biological_confusion_matrices = []
        
        random_hit_rate = []
        shuffled_hit_rate = []
        mutated_hit_rate = []
        added_deleted_hit_rate = []
        signal_hit_rate = []
        metabolic_hit_rate = []
        precision_at_100, biological_precision_at_100 = [], []

        mutations=5 # FIXME hardcoded values
        additions=5 # FIXME hardcoded values
        signal_sequences, metabolic_sequences = get_signal_and_metabolic_sequences()
        random_sequences, shuffled_sequences, mutated_sequences, added_deleted_sequences = generate_synthetic_sequences(reference_file, 10000, mutations, additions) # FIXME hardcoded values

        signal_input_features = self.get_input_features(signal_sequences)
        metabolic_input_features = self.get_input_features(metabolic_sequences)
        random_input_features = self.get_input_features(random_sequences)
        shuffled_input_features = self.get_input_features(shuffled_sequences)
        mutated_input_features = self.get_input_features(mutated_sequences)
        added_deleted_input_features = self.get_input_features(added_deleted_sequences)

        for fold_idx, (train_index, test_index) in enumerate(kf.split(input_features), start=1):
            train_features = [input_features[i] for i in train_index]
            test_features = [input_features[i] for i in test_index]
            train_labels = [labels[i] for i in train_index]
            test_labels = [labels[i] for i in test_index]
            train_mask_high_quality_idxs = [mask_high_quality_idxs[i] for i in train_index]
            test_mask_high_quality_idxs = [mask_high_quality_idxs[i] for i in test_index]

            self.train_classifier(train_features, train_labels, weight_balancing=weight_balancing, mask_high_quality_idxs=train_mask_high_quality_idxs, verbose=False, objective=objective)

            predictions = self.predict_from_features(test_features)
            scores = self.predict_from_features(test_features, proba=True)
            
            # Biological Data: HQ test subset + signal + metabolic
            high_quality_test_labels = np.array(test_labels)[test_mask_high_quality_idxs]
            high_quality_predictions = predictions[test_mask_high_quality_idxs]
            high_quality_scores = scores[test_mask_high_quality_idxs]

            random_predictions = self.predict_from_features(random_input_features)
            random_scores = self.predict_from_features(random_input_features, proba=True)
            shuffled_predictions = self.predict_from_features(shuffled_input_features)
            shuffled_scores = self.predict_from_features(shuffled_input_features, proba=True)
            mutated_predictions = self.predict_from_features(mutated_input_features)
            mutated_scores = self.predict_from_features(mutated_input_features, proba=True)
            added_deleted_predictions = self.predict_from_features(added_deleted_input_features)
            added_deleted_scores = self.predict_from_features(added_deleted_input_features, proba=True)
            signal_predictions = self.predict_from_features(signal_input_features)
            signal_scores = self.predict_from_features(signal_input_features, proba=True)
            metabolic_predictions = self.predict_from_features(metabolic_input_features)
            metabolic_scores = self.predict_from_features(metabolic_input_features, proba=True)

            random_hit_rate.append(random_predictions.mean())
            shuffled_hit_rate.append(shuffled_predictions.mean())
            mutated_hit_rate.append(mutated_predictions.mean())
            added_deleted_hit_rate.append(added_deleted_predictions.mean())
            signal_hit_rate.append(signal_predictions.mean())
            metabolic_hit_rate.append(metabolic_predictions.mean())

            # Build Biological Data aggregates (HQ + signal + metabolic)
            biological_predictions_fold = np.concatenate([
                high_quality_predictions,
                signal_predictions,
                metabolic_predictions,
            ])
            biological_scores_fold = np.concatenate([
                high_quality_scores,
                signal_scores,
                metabolic_scores,
            ])
            biological_labels_fold = np.concatenate([
                high_quality_test_labels,
                np.zeros(len(signal_scores), dtype=int),
                np.zeros(len(metabolic_scores), dtype=int),
            ])

            biological_accuracies.append(accuracy_score(biological_labels_fold, biological_predictions_fold))
            biological_f1_scores.append(f1_score(biological_labels_fold, biological_predictions_fold))
            biological_mcc_scores.append(matthews_corrcoef(biological_labels_fold, biological_predictions_fold))
            biological_auprc_scores.append(average_precision_score(biological_labels_fold, biological_scores_fold))
            biological_tpr_at_001_fpr.append(calculate_tpr_at_fpr(biological_labels_fold, biological_scores_fold))
            biological_cm_fold = confusion_matrix(
                biological_labels_fold,
                biological_predictions_fold,
                labels=[0, 1],
                normalize='true',
            )
            biological_confusion_matrices.append(biological_cm_fold)

            # Build All Data aggregates (Biological + added_deleted sequences only)
            all_predictions_fold = np.concatenate([
                biological_predictions_fold,
                added_deleted_predictions,
            ])
            all_scores_fold = np.concatenate([
                biological_scores_fold,
                added_deleted_scores,
            ])
            all_labels_fold = np.concatenate([
                biological_labels_fold,
                np.zeros(len(added_deleted_scores), dtype=int),
            ])

            all_accuracies.append(accuracy_score(all_labels_fold, all_predictions_fold))
            all_f1_scores.append(f1_score(all_labels_fold, all_predictions_fold))
            all_mcc_scores.append(matthews_corrcoef(all_labels_fold, all_predictions_fold))
            all_auprc_scores.append(average_precision_score(all_labels_fold, all_scores_fold))
            all_tpr_at_001_fpr.append(calculate_tpr_at_fpr(all_labels_fold, all_scores_fold))
            all_cm_fold = confusion_matrix(
                all_labels_fold,
                all_predictions_fold,
                labels=[0, 1],
                normalize='true',
            )
            all_confusion_matrices.append(all_cm_fold)

            top_100_idxs = np.argsort(all_scores_fold)[-100:]
            precision_at_100.append(precision_score(np.array(all_labels_fold)[top_100_idxs], all_predictions_fold[top_100_idxs]))
            
            # Precision@100 for Biological Data only
            top_100_biological_idxs = np.argsort(biological_scores_fold)[-100:]
            biological_precision_at_100.append(
                precision_score(biological_labels_fold[top_100_biological_idxs], biological_predictions_fold[top_100_biological_idxs])
            )

            fold_rows.append({
                "fold": fold_idx,

                "all_accuracy": all_accuracies[-1],
                "all_f1": all_f1_scores[-1],
                "all_mcc": all_mcc_scores[-1],
                "all_auprc": all_auprc_scores[-1],
                "all_tpr_at_0p01_fpr": all_tpr_at_001_fpr[-1],
                "all_confusion_matrix_norm_true_0_0": float(all_cm_fold[0, 0]),
                "all_confusion_matrix_norm_true_0_1": float(all_cm_fold[0, 1]),
                "all_confusion_matrix_norm_true_1_0": float(all_cm_fold[1, 0]),
                "all_confusion_matrix_norm_true_1_1": float(all_cm_fold[1, 1]),
                "all_positive_likelihood_ratio": float(all_cm_fold[1, 1] / (all_cm_fold[0, 1] + 1e-10)),
                "all_precision_at_100": precision_at_100[-1],

                "biological_accuracy": biological_accuracies[-1],
                "biological_f1": biological_f1_scores[-1],
                "biological_mcc": biological_mcc_scores[-1],
                "biological_auprc": biological_auprc_scores[-1],
                "biological_tpr_at_0p01_fpr": biological_tpr_at_001_fpr[-1],
                "biological_confusion_matrix_norm_true_0_0": float(biological_cm_fold[0, 0]),
                "biological_confusion_matrix_norm_true_0_1": float(biological_cm_fold[0, 1]),
                "biological_confusion_matrix_norm_true_1_0": float(biological_cm_fold[1, 0]),
                "biological_confusion_matrix_norm_true_1_1": float(biological_cm_fold[1, 1]),
                "biological_positive_likelihood_ratio": float(biological_cm_fold[1, 1] / (biological_cm_fold[0, 1] + 1e-10)),
                "biological_precision_at_100": biological_precision_at_100[-1],

                "p_amp_random": random_hit_rate[-1],
                "p_amp_shuffled": shuffled_hit_rate[-1],
                "p_amp_mutated": mutated_hit_rate[-1],
                "p_amp_added_deleted": added_deleted_hit_rate[-1],
                "p_amp_signal": signal_hit_rate[-1],
                "p_amp_metabolic": metabolic_hit_rate[-1],
            })

        print(f"Average All Data Accuracy: {np.mean(all_accuracies):.4f} (+/- {np.std(all_accuracies):.4f})")
        print(f"Average All Data F1 Score: {np.mean(all_f1_scores):.4f} (+/- {np.std(all_f1_scores):.4f})")
        print(f"Average All Data MCC Score: {np.mean(all_mcc_scores):.4f} (+/- {np.std(all_mcc_scores):.4f})")
        print(f"Average All Data AUPRC Score: {np.mean(all_auprc_scores):.4f} (+/- {np.std(all_auprc_scores):.4f})")
        print(f"Average All Data TPR @ 0.01 FPR: {np.mean(all_tpr_at_001_fpr):.4f} (+/- {np.std(all_tpr_at_001_fpr):.4f})")
        print("Average All Data Confusion Matrix:")
        average_all_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
        print(average_all_confusion_matrix)
        print("All Data Positive Likelihood Ratio:")
        print(average_all_confusion_matrix[1, 1] / (average_all_confusion_matrix[0, 1] + 1e-10))

        if biological_accuracies:
            print(f"Average Biological Data Accuracy: {np.mean(biological_accuracies):.4f} (+/- {np.std(biological_accuracies):.4f})")
        if biological_f1_scores:
            print(f"Average Biological Data F1 Score: {np.mean(biological_f1_scores):.4f} (+/- {np.std(biological_f1_scores):.4f})")
        if biological_mcc_scores:
            print(f"Average Biological Data MCC Score: {np.mean(biological_mcc_scores):.4f} (+/- {np.std(biological_mcc_scores):.4f})")
        if biological_auprc_scores:
            print(f"Average Biological Data AUPRC Score: {np.mean(biological_auprc_scores):.4f} (+/- {np.std(biological_auprc_scores):.4f})")
        if biological_tpr_at_001_fpr:
            print(f"Average Biological Data TPR @ 0.01 FPR: {np.mean(biological_tpr_at_001_fpr):.4f} (+/- {np.std(biological_tpr_at_001_fpr):.4f})")

        print("Average Biological Data Confusion Matrix:")
        average_biological_confusion_matrix = np.mean(biological_confusion_matrices, axis=0)
        print(average_biological_confusion_matrix)
        print("Biological Data Positive Likelihood Ratio:")
        print(average_biological_confusion_matrix[1, 1] / (average_biological_confusion_matrix[0, 1] + 1e-10))
        
        print(f"Probability of random sequences being AMPs: {np.mean(random_hit_rate):.4f}")
        print(f"Probability of shuffled sequences being AMPs: {np.mean(shuffled_hit_rate):.4f}")
        print(f"Probability of mutated sequences (mutations={mutations}) being AMPs: {np.mean(mutated_hit_rate):.4f}")
        print(f"Probability of added-deleted sequences (added-deleted={additions}) being AMPs: {np.mean(added_deleted_hit_rate):.4f}")
        print(f"Probability of signal sequences being AMPs: {np.mean(signal_hit_rate):.4f}")
        print(f"Probability of metabolic sequences being AMPs: {np.mean(metabolic_hit_rate):.4f}")
        print(f"Precision at Top 100 (All Data): {np.mean(precision_at_100):.4f} (+/- {np.std(precision_at_100):.4f})")
        print(f"Biological Data Precision at Top 100: {np.mean(biological_precision_at_100):.4f} (+/- {np.std(biological_precision_at_100):.4f})")

        return pd.DataFrame(fold_rows)

    def forward(self, sequences):
        input = self.get_input_features(sequences)
        probas = self.model.predict_proba(input)[:, 1]
        return (probas >= self.decision_threshold).astype(int)
    
    def predict_from_features(self, input_features, proba=False):
        probas = self.model.predict_proba(input_features)[:, 1]
        if proba:
            return probas
        return (probas >= self.decision_threshold).astype(int)
    
    def predict_proba(self, sequences):
        input = self.get_input_features(sequences)
        return self.model.predict_proba(input)[:, 1]

    def save(self, path):
        self.model.save_model(path)

class AMPClassifier(PeptideClassifier):
    def get_input_features(self, sequences, esm2_embeddings=False):
        positional_encodings = pd.DataFrame(calculate_positional_encodings(sequences))
        properties = pd.DataFrame(calculate_physchem_prop(sequences, all_scales=True))
        frequencies = pd.DataFrame(calculate_aa_frequency(sequences))
        if esm2_embeddings:
            esm2_embeddings = pd.DataFrame(calculate_average_esm2_embeddings(sequences))
            return pd.concat([properties, frequencies, positional_encodings, esm2_embeddings], axis=1)
        else:
            return pd.concat([properties, frequencies, positional_encodings], axis=1)

    def train_classifier(self, input_features, labels, weight_balancing="balanced_with_adjustment_for_high_quality", mask_high_quality_idxs=[], return_feature_importances=False, verbose=True, objective='focal_loss'):
        train_input, val_input, train_labels, val_labels, train_mask_high_quality_idxs, _ = train_test_split(
            input_features, labels, mask_high_quality_idxs, test_size=0.03, random_state=42, stratify=labels
        )

        if weight_balancing.startswith("balanced"):
            class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
            weights = np.array([class_weights[i] for i in train_labels])
        elif weight_balancing == "balanced_with_adjustment_for_high_quality":
            weights[train_mask_high_quality_idxs] = max(class_weights)
        else:
            weights = np.ones(len(train_labels))

        dtrain = DMatrix(train_input, label=train_labels, weight=weights)
        dval = DMatrix(val_input, label=val_labels)
        
        params = self.model.get_xgb_params()

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.model.n_estimators,
            evals=[(dval, 'eval')],
            early_stopping_rounds=self.model.early_stopping_rounds,
            obj=focal_loss if objective == 'focal_loss' else None,
            verbose_eval=verbose,
        )
        self.model._Booster = booster

        if return_feature_importances:
            return self.model.feature_importances_

class HemolyticClassifier(PeptideClassifier):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.decision_threshold = 0.5

    def get_input_features(self, sequences):
        positional_encodings = pd.DataFrame(calculate_positional_encodings(sequences))
        properties = pd.DataFrame(calculate_physchem_prop(sequences, all_scales=True))
        frequencies = pd.DataFrame(calculate_aa_frequency(sequences))
        return pd.concat([properties, frequencies, positional_encodings], axis=1)

    def train_classifier(self, input_features, labels, weight_balancing="balanced_with_adjustment_for_high_quality", mask_high_quality_idxs=[], return_feature_importances=False, verbose=True, objective='focal_loss'):
        train_input, val_input, train_labels, val_labels, train_mask_high_quality_idxs, _ = train_test_split(
            input_features, labels, mask_high_quality_idxs, test_size=0.03, random_state=42, stratify=labels
        )

        high_quality_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_mask_high_quality_idxs), y=train_mask_high_quality_idxs)
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
        
        weights = np.array([class_weights[c] + high_quality_weights[int(hq)] for (c,hq) in zip(train_labels, train_mask_high_quality_idxs)])

        dtrain = DMatrix(train_input, label=train_labels, weight=weights)
        dval = DMatrix(val_input, label=val_labels)

        params = self.model.get_xgb_params()

        obj_param = focal_loss if objective == 'focal_loss' else 'binary:logistic'

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.model.n_estimators,
            evals=[(dval, 'eval')],
            early_stopping_rounds=self.model.early_stopping_rounds,
            obj=obj_param,
            verbose_eval=verbose,
        )
        self.model._Booster = booster

        if return_feature_importances:
            return self.model.feature_importances_

