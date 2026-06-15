from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kstest
import math

from project.classifiers import AMPClassifier
from sklearn.metrics import mean_absolute_error

from project.constants import AMINO_ACIDS
from project.helpers import calculate_precision_recall, filter_valid_sequences, calculate_normalized_alignment_similarity, calculate_normalized_levenshtein_distance
from project.sequence_properties import *
from Levenshtein import distance as levenshtein

from .utils import get_logger
import enum

log = get_logger()

class MetricInputType(enum.Enum):
    Sequence = 1
    Embedding = 2
    Conditioning = 3


class SampleMetricsCollection(nn.Module):
    def __init__(self, prefix: str, metrics: list[nn.Module]):
        super().__init__()

        self.prefix = prefix
        self.metrics = nn.ModuleList(metrics)

    def forward(self, embeddings, sequences, conditioning_used):
        values = {}

        valid_sequences, idxs = filter_valid_sequences(sequences)

        if len(valid_sequences) == 0:
            log.warning("No valid sequences to evaluate metrics on.")
            return values
        elif len(valid_sequences) != len(sequences):
            if embeddings != None:
                embeddings = embeddings[idxs]
            if conditioning_used != None:
                conditioning_used = (conditioning_used[0][idxs], conditioning_used[1][idxs])

        # Evaluate the metrics
        for metric in self.metrics:
            if metric.input_type == MetricInputType.Sequence:
                metrics = metric(valid_sequences)
                if type(metrics) == dict:
                    for value in metrics:
                        values[self.log_name(f"{metric.name}/{value}")] = metrics[value]
                else:
                    values[self.log_name(metric.name)] = metrics
            elif metric.input_type == MetricInputType.Embedding:
                values[self.log_name(metric.name)] = metric(embeddings)
            else:
                conditioning_metrics = metric(valid_sequences, conditioning_used)
                if type(conditioning_metrics) == dict:
                    for value in conditioning_metrics:
                        values[self.log_name(f"{metric.name}/{value}")] = conditioning_metrics[value]
                else:
                    values[self.log_name(metric.name)] = metric(valid_sequences, conditioning_used)
        
        return values

    def log_name(self, metric: str):
        return f"{self.prefix}/{metric}"

class ConditioningPropertiesMetricsMAE(nn.Module):
    def __init__(self, conditioning, conditioning_masking):
        super().__init__()
        self.loss = mean_absolute_error
        self.name = "mae"
        self.input_type = MetricInputType.Conditioning
        self.conditioning = conditioning
        self.conditioning_masking = conditioning_masking

    def forward(self, sequences, conditioning_used):
        conditioning, conditioning_mask = conditioning_used
        conditioning_mask = conditioning_mask.to(conditioning.device)
        computable_conditioning_generated = self.conditioning.get_computable_conditioning_vectors(sequences).to(conditioning.device)
        cond_metrics = {}

        # Single feature unmasked metrics
        for idx, name in enumerate(self.conditioning.computable_names):
            single_feature_mask = self.conditioning_masking.conditioning_mask.default_mask.clone()
            single_feature_mask[idx] = self.conditioning_masking.conditioning_mask.positive_val

            valid_idxs = torch.all(conditioning_mask == single_feature_mask, dim=1).to(conditioning.device)
            if torch.any(valid_idxs):
                cond_metrics[f"{name}_single"] = self.loss(
                    computable_conditioning_generated[valid_idxs, idx].tolist(),
                    conditioning[valid_idxs, idx].tolist()
                )
            else:
                cond_metrics[f"{name}_single"] = float('nan')

        # All features unmasked metrics
        all_unmasked = torch.all(conditioning_mask == self.conditioning_masking.conditioning_mask.positive_val, dim=1).to(conditioning.device)
        if torch.any(all_unmasked):
            for idx, name in enumerate(self.conditioning.computable_names):
                cond_metrics[f"{name}_all"] = self.loss(
                    computable_conditioning_generated[all_unmasked, idx].tolist(),
                    conditioning[all_unmasked, idx].tolist()
                )
        else:
            for name in self.conditioning.computable_names:
                cond_metrics[f"{name}_all"] = float('nan')

        return cond_metrics

class ConditioningPropertiesMetricsNormalizedMAE(ConditioningPropertiesMetricsMAE):
    def __init__(self, conditioning, conditioning_masking):
        super().__init__(conditioning, conditioning_masking)
        self.name = "mae-by-std-unit"

    def forward(self, sequences, conditioning_used):
        cond_metrics = super().forward(sequences, conditioning_used)
        # Normalize single
        for name in self.conditioning.computable_names:
            idx = self.conditioning.computable_names.index(name)
            if not math.isnan(cond_metrics[f"{name}_single"]):
                cond_metrics[f"{name}_single"] = cond_metrics[f"{name}_single"] / self.conditioning.std[idx]
            if not math.isnan(cond_metrics[f"{name}_all"]):
                cond_metrics[f"{name}_all"] = cond_metrics[f"{name}_all"] / self.conditioning.std[idx]

        # Normalize all
        for name in self.conditioning.computable_names:
            idx = self.conditioning.computable_names.index(name)
            if not math.isnan(cond_metrics[f"{name}_all"]):
                cond_metrics[f"{name}_all"] = cond_metrics[f"{name}_all"] / self.conditioning.std[idx]
        
        return cond_metrics

class KolmogorovSmirnovTest(nn.Module):
    """
    https://docs.scipy.org/doc/scipy-1.14.0/reference/generated/scipy.stats.kstest.html
    """
    def forward(self, real_datapoints, sampled_datapoints):
        return kstest(np.array(real_datapoints), np.array(sampled_datapoints))


class PropertyKSDistance(KolmogorovSmirnovTest):
    def __init__(self, properties, real_sequences, p_value=False):
        super().__init__()
        self.properties = properties
        self.name = "kolmogorov-smirnov-distance"
        self.real_datapoints = [self.get_function_for_property(property)(real_sequences) for property in self.properties]
        self.p_value = p_value
        self.input_type = MetricInputType.Sequence

    def get_function_for_property(self, property):
        if property == "length":
            return calculate_length
        elif property == "charge":
            return calculate_charge
        elif property == "hydrophobicity_eisenberg":
            return calculate_hydrophobicity
        elif property == "instability_index":
            return calculate_instability_index
    
    def forward(self, generated_sequences):
        sampled_datapoints = [self.get_function_for_property(property)(generated_sequences) for property in self.properties]
        results = {}
        for idx, property in enumerate(self.properties):
            if self.p_value:
                results[property] = super().forward(self.real_datapoints[idx], sampled_datapoints[idx]).pvalue
            else:
                results[property] = super().forward(self.real_datapoints[idx], sampled_datapoints[idx]).statistic
        return results
        
        
    
class FrechetDistance(nn.Module):
    """
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Code adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    """
    def __init__(self, real_embeddings, eps=1e-6):
        super().__init__()

        self.mu_real = torch.mean(real_embeddings, dim=0)
        self.cov_real = torch.cov(real_embeddings.t(), correction=1)
        self.name = "frechet-distance"
        self.input_type = MetricInputType.Embedding
        self.eps=eps

    def forward(self, generated_embeddings):
        mu_generated = torch.mean(generated_embeddings, dim=0)
        cov_generated = torch.cov(generated_embeddings.t(), correction=1)

        self.mu_real, self.cov_real = self.mu_real.to(mu_generated.device), self.cov_real.to(cov_generated.device)
        
        mu1, mu2 = self.mu_real, mu_generated
        sigma1, sigma2 = self.cov_real, cov_generated

        a = (mu1 - mu2).square().sum(dim=-1)
        b = sigma1.trace() + sigma2.trace()
        c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

        return a + b - 2 * c

class FrechetAminoacidEmbeddingDistance(FrechetDistance):
    def __init__(self, encoder_model, real_sequences):
        real_embeddings = encoder_model.encode(real_sequences)
        real_embeddings = real_embeddings.view(real_embeddings.shape[0], -1)

        super().__init__(real_embeddings)

        self.input_type = MetricInputType.Sequence
        self.encoder_model = encoder_model
        self.name = "frechet-aa-embedding-distance"
    
    def forward(self, generated_sequences):
        generated_embeddings = self.encoder_model.encode(generated_sequences)

        generated_embeddings = generated_embeddings.view(generated_embeddings.shape[0], -1)
        
        return super().forward(generated_embeddings)


class FrechetPCADistance(FrechetDistance):
    def __init__(self, real_embeddings, components=320):
        real_embeddings = torch.tensor(real_embeddings)
        real_embeddings = real_embeddings.reshape(real_embeddings.shape[0], -1)

        if real_embeddings.shape[1] < components:
            self.projector_matrix = torch.eye(real_embeddings.shape[1])
        else:
            _, _ , self.projector_matrix = torch.pca_lowrank(real_embeddings, q=components)

        projected_real_embeddings = real_embeddings @ self.projector_matrix

        super().__init__(projected_real_embeddings)

        self.name = "frechet-pca-distance"
    
    def forward(self, generated_embeddings):
        self.projector_matrix = self.projector_matrix.to(generated_embeddings.device)

        generated_embeddings = generated_embeddings.reshape(generated_embeddings.shape[0], -1)
        projected_generated_embeddings = generated_embeddings @ self.projector_matrix
        return super().forward(projected_generated_embeddings)
    

class FrechetMeanDistance(FrechetDistance):
    def __init__(self, real_embeddings):
        real_embeddings = torch.tensor(real_embeddings)
        
        averaged_real_embeddings = torch.mean(real_embeddings, dim=2)

        super().__init__(averaged_real_embeddings)

        self.name = "frechet-length-average-distance"
    
    def forward(self, generated_embeddings):
        averaged_generated_embeddings = torch.mean(generated_embeddings, dim=2)
        return super().forward(averaged_generated_embeddings)
    

class PseudoPerplexity(nn.Module):
    """
    Computes the pseudo-perplexity of generated sequences as described in 
    https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf
    """

    def __init__(self, esm_model):
        super().__init__()
        self.esm_model = esm_model
        self.name = "pseudo-perplexity"
        self.input_type = MetricInputType.Sequence

    def forward(self, generated_sequences, all_results=False):
        results = []
        for sequence in generated_sequences:
            results.append(self.esm_model.compute_pseudo_perplexity(sequence))
        if all_results:
            return np.array(results).mean(), np.array(results)
        return np.array(results).mean()
    
class BatchEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "batch-aminoacid-entropy"
        self.input_type = MetricInputType.Sequence
        self.amino_acids = AMINO_ACIDS

    def forward(self, generated_sequences):
        total_freqs = Counter()
        total_length = 0

        for sequence in generated_sequences:
            total_freqs.update(Counter(sequence))
            total_length += len(sequence)

        for amino_acid in self.amino_acids:
            if amino_acid not in total_freqs:
                total_freqs[amino_acid] = 0
        
        probabilities = np.array([total_freqs[aa] / total_length for aa in self.amino_acids if total_length > 0])
        
        entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))  # np.finfo(float).eps prevents log(0)

        return entropy
    

class SequenceEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "sequence-aminoacid-entropy"
        self.input_type = MetricInputType.Sequence
        self.amino_acids = AMINO_ACIDS

    def forward(self, generated_sequences):
        entropies = []

        for sequence in generated_sequences:
            freqs = Counter(sequence)
            length = len(sequence)

            for amino_acid in self.amino_acids:
                if amino_acid not in freqs:
                    freqs[amino_acid] = 0

            probabilities = np.array([freqs[aa] / length for aa in self.amino_acids])
            entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
            entropies.append(entropy)

        mean_entropy = np.mean(entropies)

        return mean_entropy


class Precision(nn.Module):
    def __init__(self, encoder_model, real_sequences):
        super().__init__()
        self.name = "precision"
        self.input_type = MetricInputType.Sequence
        self.encoder_model = encoder_model
        real_embeddings = self.encoder_model.encode(real_sequences)
        self.real_embeddings = real_embeddings.view(real_embeddings.shape[0], -1)
    
    def forward(self, generated_sequences):
        valid_sequences = [seq for seq in generated_sequences if 'X' not in seq and len(seq) > 0]
        
        generated_embeddings = self.encoder_model.encode(valid_sequences)

        generated_embeddings = generated_embeddings.view(generated_embeddings.shape[0], -1)

        precision, _ = calculate_precision_recall(self.real_embeddings, generated_embeddings)

        return precision

class Recall(nn.Module):
    def __init__(self, encoder_model, real_sequences):
        super().__init__()
        self.name = "recall"
        self.input_type = MetricInputType.Sequence
        self.encoder_model = encoder_model
        real_embeddings = self.encoder_model.encode(real_sequences)
        self.real_embeddings = real_embeddings.view(real_embeddings.shape[0], -1)
    
    def forward(self, generated_sequences):
        valid_sequences = [seq for seq in generated_sequences if 'X' not in seq and len(seq) > 0]
        
        generated_embeddings = self.encoder_model.encode(valid_sequences)

        generated_embeddings = generated_embeddings.view(generated_embeddings.shape[0], -1)

        _ , recall = calculate_precision_recall(self.real_embeddings, generated_embeddings)

        return recall


class AMPProbability(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.name = "amp-probability"
        self.input_type = MetricInputType.Sequence
        self.model_wrapper = AMPClassifier(model_path)
    
    def forward(self, generated_sequences, n=5000):
        return np.mean(self.model_wrapper(generated_sequences[:n]))

class NovelAMP(nn.Module):
    def __init__(self, real_sequences):
        super().__init__()
        self.name = "novel-amp"
        self.input_type = MetricInputType.Sequence
        self.real_sequences = set(real_sequences)
    
    def forward(self, generated_sequences):
        novel_count = 0
        for seq in generated_sequences:
            if seq not in self.real_sequences:
                novel_count +=1
        return novel_count / len(generated_sequences)
    
class Diversity(nn.Module):
    def __init__(self, distance="alignment"):
        super().__init__()
        self.name = "diversity"
        self.input_type = MetricInputType.Sequence
        if distance == "levenshtein":
            self.distance_fn = lambda seq1, seq2: calculate_normalized_levenshtein_distance(seq1, seq2)
        elif distance == "alignment":
            self.distance_fn = lambda seq1, seq2: (1 - calculate_normalized_alignment_similarity(seq1, seq2))

    def forward(self, generated_sequences, n=2000):
        total_distance = 0
        total_pairs = 0
        if len(generated_sequences) == 0:
            return 0
        elif len(generated_sequences) == 1:
            return 1
        sequences = np.random.choice(generated_sequences, n)
        for idx, seq1 in enumerate(sequences):
            for seq2 in sequences[idx + 1:]:
                total_distance += self.distance_fn(seq1, seq2)
                total_pairs += 1
        return total_distance / total_pairs
    
class DiversityPredictedPositives(nn.Module):
    def __init__(self, path_to_classifier, distance="alignment"):
        super().__init__()
        self.name = "diversity-predicted-positives"
        self.input_type = MetricInputType.Sequence
        self.classifier = AMPClassifier(path_to_classifier)
        self.diversity_wrapper = Diversity(distance=distance)
    
    def forward(self, generated_sequences, n=2000):
        predicted_positives = self.classifier(generated_sequences)
        valid_sequences = [seq for idx, seq in enumerate(generated_sequences) if predicted_positives[idx] > 0.5]
        return self.diversity_wrapper(valid_sequences, n)

class Uniqueness(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "uniqueness"
        self.input_type = MetricInputType.Sequence
    
    def forward(self, generated_sequences):
        unique_sequences = set(generated_sequences)
        return len(unique_sequences) / len(generated_sequences)

class FitnessScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "fitness-score"
        self.input_type = MetricInputType.Sequence
        
    def forward(self, generated_sequences, all_results=False):
        fitness_scores = compute_fitness_scores(generated_sequences)
        if all_results:
            return np.mean(fitness_scores), fitness_scores
        return np.mean(fitness_scores)


class EncoderAverageDistance(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.name = "encoder-average-distance"
        self.input_type = MetricInputType.Embedding
        self.wrapper = wrapper

    def forward(self, generated_embeddings):
        predicted_sequences = self.wrapper.decode(generated_embeddings)
        corresponding_embeddings = self.wrapper.encode(predicted_sequences)
        corresponding_embeddings = corresponding_embeddings.to(generated_embeddings.device)
        return torch.sqrt(torch.mean(torch.pow(generated_embeddings - corresponding_embeddings, 2)))
