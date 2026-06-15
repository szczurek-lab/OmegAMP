import torch
from project.constants import AMINO_ACIDS
from Bio import pairwise2
from Levenshtein import distance as levenshtein



def update_scale(scale, min_clamp, max_clamp):
    items = list(scale.items())

    # Sort items by value
    items.sort(key=lambda x: x[1])

    # Clamp values
    for i in range(len(items) - 1):
        diff = items[i+1][1] - items[i][1]
        if diff < min_clamp:
            for j in range(i+1, len(items)):
                items[j] = [items[j][0], items[j][1] + min_clamp - diff]
        if diff > max_clamp:
            for j in range(i+1, len(items)):
                items[j] = [items[j][0], items[j][1] + max_clamp - diff]

    # Update scale
    scale = {k: v for k, v in items}
    return scale

def filter_valid_sequences(sequences):
    valid_sequences = []
    valid_indices = []
    for idx, seq in enumerate(sequences):
        if len(seq) > 0 and all(char in AMINO_ACIDS for char in seq):
            valid_sequences.append(seq)
            valid_indices.append(idx)
    return valid_sequences, valid_indices

def compute_mean_and_std(embeddings, chunk_size=1024):
    num_embeddings = embeddings.shape[0]

    x_shape = embeddings.shape[1:]
    
    sum = torch.zeros(x_shape)

    for i in range(0, num_embeddings, chunk_size):
        end = min(i + chunk_size, num_embeddings)
        chunk = torch.tensor(embeddings[i:end]) if not torch.is_tensor(embeddings) else embeddings[i:end]
        sum += torch.sum(chunk, dim=0)

    mean = sum / num_embeddings

    variance_sum = torch.zeros(x_shape)

    for i in range(0, num_embeddings, chunk_size):
        end = min(i + chunk_size, num_embeddings)
        chunk = torch.tensor(embeddings[i:end]) if not torch.is_tensor(embeddings) else embeddings[i:end]
        variance_sum += torch.sum((chunk - mean) ** 2, dim=0)

    variance = variance_sum / num_embeddings
    std = torch.sqrt(variance)

    return mean, std

"""Code from https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_prc.py"""

def calc_cdist_part(features_1, features_2, batch_size=10000):
    dists = []
    for feat2_batch in features_2.split(batch_size):
        dists.append(torch.cdist(features_1, feat2_batch).cpu())
    return torch.cat(dists, dim=1)


def calculate_precision_recall(features_1, features_2, neighborhood=3, batch_size=10000):
    # Precision
    dist_nn_1 = []
    for feat_1_batch in features_1.split(batch_size):
        dist_nn_1.append(calc_cdist_part(feat_1_batch, features_1, batch_size).kthvalue(neighborhood + 1).values)
    dist_nn_1 = torch.cat(dist_nn_1)
    precision = []
    for feat_2_batch in features_2.split(batch_size):
        dist_2_1_batch = calc_cdist_part(feat_2_batch, features_1, batch_size)
        precision.append((dist_2_1_batch <= dist_nn_1).any(dim=1).float())
    precision = torch.cat(precision).mean().item()
    # Recall
    dist_nn_2 = []
    for feat_2_batch in features_2.split(batch_size):
        dist_nn_2.append(calc_cdist_part(feat_2_batch, features_2, batch_size).kthvalue(neighborhood + 1).values)
    dist_nn_2 = torch.cat(dist_nn_2)
    recall = []
    for feat_1_batch in features_1.split(batch_size):
        dist_1_2_batch = calc_cdist_part(feat_1_batch, features_2, batch_size)
        recall.append((dist_1_2_batch <= dist_nn_2).any(dim=1).float())
    recall = torch.cat(recall).mean().item()
    return precision, recall

def calculate_normalized_alignment_similarity(ref_seq, target_seq):
    ref_seq = ref_seq.strip().upper()
    target_seq = target_seq.strip().upper()
    
    alignments = pairwise2.align.globalxx(ref_seq, target_seq, one_alignment_only=True)
    best_alignment = alignments[0]
    _, _, similarity, _, _ = best_alignment

    return similarity / min(len(ref_seq), len(target_seq))

def calculate_normalized_levenshtein_distance(ref_seq, target_seq):
    return levenshtein(ref_seq, target_seq) / max(len(ref_seq), len(target_seq))
