import torch

from project.constants import PADDING_VALUE
from project.helpers import compute_mean_and_std
from project.sequence_properties import calculate_physchem_prop, calculate_length, calculate_charge, calculate_hydrophobicity, calculate_instability_index
import pandas as pd
from enum import Enum

class PartialConditioningTypes(Enum):
    UNDEFINED = 0
    INTERVAL = 1
    LIST_POSSIBLE_VALUES = 2
    DEFINED = 3

class Conditioning:
    def __init__(self, sequences, computable_names, uncomputable_names, uncomputable_cond):
        self.computable_names = computable_names
        self.uncomputable_names = uncomputable_names
        self.conditioning_names = computable_names + uncomputable_names
        self.uncomputable_cond = uncomputable_cond

        computable_cond = self.get_computable_conditioning_vectors(sequences)        

        self.conditioning_vectors = torch.cat((computable_cond, uncomputable_cond), dim=1)
        self.mean = torch.mean(self.conditioning_vectors, dim=0)
        self.std = torch.std(self.conditioning_vectors, dim=0)

        self.number_of_amps = torch.sum(self.conditioning_vectors[:, -1] == 1)
        self.number_of_non_amps = torch.sum(self.conditioning_vectors[:, -1] == 0)
        
    def get_computable_conditioning_vectors(self, sequences):
        properties = {}
        
        if "length" in self.computable_names:
            properties["length"] = calculate_length(sequences).tolist()
        
        if "charge" in self.computable_names:
            properties["charge"] = calculate_charge(sequences).tolist()
        
        if "hydrophobicity_eisenberg" in self.computable_names:
            properties["hydrophobicity_eisenberg"] = calculate_hydrophobicity(sequences, scale="eisenberg").tolist()

        if "instability_index" in self.computable_names:
            properties["instability_index"] = calculate_instability_index(sequences).tolist()
        # properties = calculate_physchem_prop(sequences, all_scales=True) for complete set of properties
        
        df = pd.DataFrame(properties)
        return torch.tensor(df[self.computable_names].values, dtype=torch.float32)
        

class ConditioningSampler:
    def __init__(self, conditioning_vectors, conditioning_names):
        mask = conditioning_vectors[:, -1] == 1
        self.amp_conditioning_vectors = conditioning_vectors[mask]
        self.conditioning_names = conditioning_names

    def sample(self, batch_size, return_idxs=None):
        idxs = torch.randint(len(self.amp_conditioning_vectors), (batch_size,))
        if return_idxs is not None:
            return_idxs["idxs"] = idxs
        return self.amp_conditioning_vectors[idxs]

class ConditioningMasking:
    def __init__(self, computable_names, uncomputable_names):
        self.conditioning_mask = ConditioningMask(computable_names, uncomputable_names)
    
    def get_conditioning_mask(self, masked_conditioning):
        return masked_conditioning[:, :, 1]
    
    def mask_idxs(self, conditioning, conditioning_mask):
        conditioning_mask = conditioning_mask.to(conditioning.device)
        masked_conditioning = conditioning * (conditioning_mask + 0.5) + PADDING_VALUE * (0.5 - conditioning_mask)
        return torch.stack((masked_conditioning, conditioning_mask), dim=-1)
    

class ConditioningMask:
    def __init__(self, computable_names, uncomputable_names):
        self.computable_names = computable_names
        self.uncomputable_names = uncomputable_names
        self.conditioning_names = computable_names + uncomputable_names
        self.positive_val = 0.5
        self.negative_val = -0.5
        self.default_mask = torch.cat((torch.ones(len(self.computable_names)) * self.negative_val, torch.ones(len(self.uncomputable_names)) * self.positive_val))

    def get_full_mask(self, batch_size):
        return torch.broadcast_to(self.default_mask, (batch_size, len(self.conditioning_names)))
    
    def get_no_mask(self, batch_size):
        return torch.ones(batch_size, len(self.conditioning_names)) * self.positive_val
    
    def get_partial_mask(self, partial_conditioning_info, batch_size):
        conditioning_mask = self.default_mask.clone()
        for property in partial_conditioning_info:
            partial_conditioning_type = partial_conditioning_info[property][0]
            idx = self.conditioning_names.index(property)
            if partial_conditioning_type == PartialConditioningTypes.DEFINED:
                conditioning_mask[idx] = self.positive_val
            elif partial_conditioning_type == PartialConditioningTypes.INTERVAL:
                conditioning_mask[idx] = self.positive_val
        return torch.broadcast_to(conditioning_mask, (batch_size, len(self.conditioning_names)))
    
    def get_random_mask(self, batch_size):
        number_unmasked_values = torch.randint(0, len(self.computable_names) + 1, (batch_size,))
        unmasked_idxs = [torch.randperm(len(self.computable_names))[:cond_number] for cond_number in number_unmasked_values]
        
        conditioning_mask = self.default_mask.expand(batch_size, -1).clone()
        for i, idxs in enumerate(unmasked_idxs):
            conditioning_mask[i, idxs] = self.positive_val
        return conditioning_mask
    
    def get_evaluation_mask(self, batch_size):
        partitions = (len(self.computable_names) + 1)
        partition_size = batch_size // partitions
        evaluation_mask = self.default_mask.expand(batch_size, -1).clone()
        for i in range(partitions - 1): # Only one unmasked value
            evaluation_mask[i * partition_size:(i + 1) * partition_size, i] = self.positive_val
        
        # Last partition is all unmasked
        evaluation_mask[(partitions - 1) * partition_size:, :len(self.computable_names)] = self.positive_val
        return evaluation_mask