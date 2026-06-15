from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import pandas as pd
from project.conditioning import Conditioning
from project.constants import SIGNAL_SEQUENCES_FILE, METABOLIC_SEQUENCES_FILE
from project.helpers import compute_mean_and_std
import pandas as pd
from Bio import SeqIO

def save_sequences_to_fasta(sequences, output_path, inspired_sequences=None):
    """Save sequences to a fasta file."""
    with open(output_path, 'w') as output_file:
        for i, seq in enumerate(sequences):
            if inspired_sequences is not None:
                output_file.write(f'>sequence_{i + 1}_inspired_{inspired_sequences[i]}\n{seq}\n')
            else:
                output_file.write(f'>sequence_{i + 1}\n{seq}\n')

def load_fasta_to_df(fasta_path):
    ids = []
    sequences = []
    with open(fasta_path, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            ids.append(record.id)
            sequences.append(str(record.seq))
    df = pd.DataFrame({
        'Id': ids,
        'Sequence': sequences
    })
    return df

def get_dataset_for_activity_classifier(classifier, sequence_label_df, secret_data_df=None):
    if secret_data_df is not None:
        sequence_label_df = pd.concat([sequence_label_df, secret_data_df], ignore_index=True)
    
    if classifier == 'broad-classifier':
        return sequence_label_df
    elif "species" in classifier or "strains" in classifier:
        if "species" in classifier:
            taxonomy, name = "species", classifier.split("-")[1]
        else:
            taxonomy, name = "strains", classifier.split("-")[1]

        positive_ids = load_fasta_to_df(f"data/activity-data/strain-species-data/{taxonomy}/{name}_positive.fasta")['Id'].to_list()
        negative_ids = load_fasta_to_df(f"data/activity-data/strain-species-data/{taxonomy}/{name}_negative.fasta")['Id'].to_list()

        if secret_data_df is not None and "species" in classifier:
            positive_ids += load_fasta_to_df(f"data/activity-data/secret-data/{name}_positive.fasta")['Id'].to_list()
            negative_ids += load_fasta_to_df(f"data/activity-data/secret-data/{name}_negative.fasta")['Id'].to_list()

        high_quality_sequences_features_dataset = sequence_label_df[sequence_label_df["high_quality"] == 1]
        low_quality_sequences_features_dataset = sequence_label_df[sequence_label_df["high_quality"] == 0]

        # Filter out high quality sequences not in positive ids or negative ids and fix the label of the ones that are
        taxonomy_high_quality_feature_dataset = high_quality_sequences_features_dataset[high_quality_sequences_features_dataset["Id"].isin(positive_ids + negative_ids)].copy()
        taxonomy_high_quality_feature_dataset.loc[taxonomy_high_quality_feature_dataset["Id"].isin(positive_ids), "label"] = 1
        taxonomy_high_quality_feature_dataset.loc[taxonomy_high_quality_feature_dataset["Id"].isin(negative_ids), "label"] = 0

        taxonomy_specific_feature_dataset = pd.concat([taxonomy_high_quality_feature_dataset, low_quality_sequences_features_dataset], ignore_index=True)
        
        # Run sanity check
        assert taxonomy_specific_feature_dataset[taxonomy_specific_feature_dataset["Id"].isin(positive_ids)]["label"].all() == 1
        assert taxonomy_specific_feature_dataset[taxonomy_specific_feature_dataset["Id"].isin(negative_ids)]["label"].all() == 0

        return taxonomy_specific_feature_dataset
    else:
        raise ValueError(f"Invalid classifier: {classifier}")


def get_input_features_labels_mask_high_quality_idxs(sequence_label_df):
    input_features_df = sequence_label_df.drop(columns=['Id', 'Sequence', 'label', 'high_quality']) 
    input_features = input_features_df.to_numpy()
    labels = sequence_label_df['label'].to_numpy()
    mask_high_quality_idxs = (sequence_label_df['high_quality'] == 1).to_numpy()
    return input_features, labels, mask_high_quality_idxs


def get_signal_and_metabolic_sequences():
    signal_sequences = load_fasta_to_df(SIGNAL_SEQUENCES_FILE)
    metabolic_sequences = load_fasta_to_df(METABOLIC_SEQUENCES_FILE)
    return signal_sequences["Sequence"].to_list(), metabolic_sequences["Sequence"].to_list()

class MinMaxNormalization:
    def __init__(self, embeddings):
        self.min = torch.min(torch.tensor(embeddings), dim=0)[0]
        max = torch.max(torch.tensor(embeddings), dim=0)[0]
        self.range = max - self.min
        self.range = torch.where(self.range == 0, torch.tensor(1, device=self.range.device), self.range)
    
    def normalize(self, x: torch.Tensor):
        self.min = self.min.to(x.device)
        self.range = self.range.to(x.device)
        return ((x - self.min) / self.range) * 2 - 1
    
    def denormalize(self, x: torch.Tensor):
        self.min = self.min.to(x.device)
        self.range = self.range.to(x.device)
        return ((x + 1) / 2) * self.range + self.min


class Standardization:
    def __init__(self, embeddings):
        self.mean , self.std = compute_mean_and_std(embeddings)
        self.std = torch.where(self.std == 0, torch.tensor(1, device=self.std.device), self.std)

    def normalize(self, x: torch.Tensor):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        return torch.addcmul(-self.mean / self.std, torch.reciprocal(self.std), x)

    def denormalize(self, x: torch.Tensor):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        return torch.addcmul(self.mean, self.std, x)


class AMPDataset(Dataset):
    def __init__(self, original_amp_file, embeddings_file, computable_conditioning_names, uncomputable_conditioning_names, only_amps):
        self.sequences = pd.read_csv(original_amp_file)
        self.embeddings = h5py.File(embeddings_file,'r')["embeddings"]
        
        if only_amps:
            self.embeddings = self.embeddings[self.sequences["IsAMP"] == 1]
            self.sequences = self.sequences[self.sequences["IsAMP"] == 1]

        uncomputable_cond = torch.tensor(self.sequences["IsAMP"].tolist(), dtype=torch.float32)
        self.computable_conditioning_names = computable_conditioning_names
        self.uncomputable_conditioning_names = uncomputable_conditioning_names
        self.conditioning_names = computable_conditioning_names + uncomputable_conditioning_names

        self.conditioning = Conditioning(self.sequences["Sequence"].tolist(), self.computable_conditioning_names, 
                                         self.uncomputable_conditioning_names, uncomputable_cond.reshape(uncomputable_cond.shape[0], 1))

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        sequence = self.sequences["Sequence"][idx]
        embedding = self.embeddings[idx]
        conditioning = self.conditioning.conditioning_vectors[idx]

        return (sequence, embedding, conditioning)

    def get_all_embeddings(self):
        return self.embeddings[:]

    def get_all_sequences(self):
        return self.sequences["Sequence"].tolist()
    
    def get_all_amp_sequences(self):
        amp_mask = self.conditioning.conditioning_vectors[:, -1] == 1
        return self.sequences["Sequence"][amp_mask.tolist()].tolist()

    def get_all_amp_embeddings(self):
        amp_mask = self.conditioning.conditioning_vectors[:, -1] == 1
        return self.embeddings[amp_mask]


class AMPDataModule(pl.LightningDataModule):
    def __init__(self, 
                 original_amp_file: Path, 
                 embeddings_file: Path, 
                 split_seed: int = 80672983, 
                 batch_size: int = 32,
                 num_workers: int = 0,
                 computable_conditioning_names: list = [],
                 uncomputable_conditioning_names: list = [],
                 only_amps: bool = False):
        super().__init__()

        self.original_amp_file = original_amp_file
        self.embeddings_file = embeddings_file
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.computable_conditioning_names = computable_conditioning_names
        self.uncomputable_conditioning_names = uncomputable_conditioning_names
        self.conditioning_names = computable_conditioning_names + uncomputable_conditioning_names
        self.only_amps = only_amps

    def prepare_data(self):
        self.dataset = AMPDataset(self.original_amp_file, self.embeddings_file, self.computable_conditioning_names, self.uncomputable_conditioning_names, self.only_amps)

    def setup(self, stage: str):
        self.train_data, self.val_data, self.test_data = random_split(
            self.dataset,
            (self.dataset.__len__() - 2 * self.batch_size, self.batch_size, self.batch_size),
            generator=torch.Generator().manual_seed(self.split_seed),
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)
