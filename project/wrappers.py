import torch
import torch.nn as nn
import esm
from project.constants import AMINO_ACIDS

def combine_scales(scales):
    combined_scale = {}
    for aa in scales[0].keys():
        combined_scale[aa] = sum([scales[i][aa] for i in range(len(scales))], [])
    return combined_scale
    
class HydrophobicScaleWrapper(nn.Module):
    def __init__(self, scale, max_length, padding_value):
        super().__init__()
        self.scale = scale
        self.scale_size = len(list(scale.values())[0])

        self.max_length = max_length
        self.padding_value = padding_value

        self.idx_to_aa = {i: aa for i, aa in enumerate(self.scale.keys())}
        self.idx_to_aa[len(self.scale)] = 'PAD'
        aa_embeddings = [[*self.scale[aa], 1] for aa in self.scale.keys()]
        aa_embeddings.append([*self.scale_size * [padding_value], 0])
        self.aa_embeddings = torch.tensor(aa_embeddings, dtype=torch.float32) 
        
    def encode_sequence(self, sequence):
        encoded_sequence = torch.zeros(self.scale_size + 1, self.max_length)
        for i, aa in enumerate(sequence):
            encoded_sequence[:self.scale_size, i] = torch.tensor(self.scale[aa], dtype=torch.float32)
            encoded_sequence[self.scale_size, i] = 1
        encoded_sequence[:self.scale_size, len(sequence):] = self.padding_value
        encoded_sequence[self.scale_size, len(sequence):] = 0 # unnecessary but included for clarity
        return encoded_sequence
        
    def encode(self, sequences):
        encoded_sequences = []
        for sequence in sequences:
            encoded_sequence = self.encode_sequence(sequence)
            encoded_sequences.append(encoded_sequence)
        return torch.stack(encoded_sequences, dim=0)
    
    def decode(self, embeddings):
        self.aa_embeddings = self.aa_embeddings.to(embeddings.device)
        decoded_sequences = []
        for idx_embedding in range(embeddings.shape[0]):
            decoded_sequence = []
            for idx_aa in range(embeddings.shape[2]):
                aa_idx = torch.argmin(torch.norm(self.aa_embeddings - embeddings[idx_embedding, : , idx_aa], dim=1))
                decoded_sequence.append(self.idx_to_aa[aa_idx.item()])
            decoded_sequence_without_pad = decoded_sequence[:decoded_sequence.index('PAD') if 'PAD' in decoded_sequence else len(decoded_sequence)]
            decoded_sequences.append("".join(decoded_sequence_without_pad))
        return decoded_sequences

class NumericWrapper(nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length
        self.ordinal_map = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        self.ordinal_map["PAD"] = len(self.ordinal_map)
    
    def encode(self, sequences):
        encoded_sequences = torch.zeros(len(sequences), 1, self.max_length)
        for i, sequence in enumerate(sequences):
            for j, aa in enumerate(sequence):
                encoded_sequences[i, 0, j] = self.ordinal_map[aa]
            encoded_sequences[i, 0, len(sequence):] = self.ordinal_map["PAD"]
        return encoded_sequences
    
    def decode(self, encoded_sequences):
        decoded_sequences = []
        for i in range(encoded_sequences.shape[0]):
            decoded_sequence = []
            for j in range(encoded_sequences.shape[2]):
                aa_idx = torch.round(encoded_sequences[i, 0, j]).to(torch.int64)
                aa = AMINO_ACIDS[aa_idx.item()] if aa_idx.item() < len(AMINO_ACIDS) else "PAD"
                if aa == "PAD":
                    break
                decoded_sequence.append(aa)
            decoded_sequences.append("".join(decoded_sequence))
        return decoded_sequences
    
class OneHotWrapper(nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length
        self.ordinal_map = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        self.ordinal_map["PAD"] = len(self.ordinal_map)
    
    def encode(self, sequences):
        encoded_sequences = torch.zeros(len(sequences), len(self.ordinal_map), self.max_length)
        for i, sequence in enumerate(sequences):
            for j, aa in enumerate(sequence):
                encoded_sequences[i, self.ordinal_map[aa], j] = 1
            encoded_sequences[i, self.ordinal_map["PAD"], len(sequence):] = 1
        return encoded_sequences
    
    def decode(self, encoded_sequences):
        decoded_sequences = []
        for i in range(encoded_sequences.shape[0]):
            decoded_sequence = []
            for j in range(encoded_sequences.shape[2]):
                aa_idx = torch.argmax(encoded_sequences[i, :, j])
                aa = AMINO_ACIDS[aa_idx.item()] if aa_idx.item() < len(AMINO_ACIDS) else "PAD"
                if aa == "PAD":
                    break
                decoded_sequence.append(aa)
            decoded_sequences.append("".join(decoded_sequence))
        return decoded_sequences

class EsmWrapper(nn.Module):
    def __init__(self, embedding_dim, max_length, device, quantization=False, path_to_model=None):
        super().__init__()

        if embedding_dim == 320:
            self.model_name = "esm2_t6_8M_UR50D"
            self.last_layer = 6
        elif embedding_dim == 480:
            self.model_name = "esm2_t12_35M_UR50D"
            self.last_layer = 12
        elif embedding_dim == 640:
            self.model_name = "esm2_t30_150M_UR50D"
            self.last_layer = 30
        elif embedding_dim == 1280:
            self.model_name = "esm2_t33_650M_UR50D"
            self.last_layer = 33
        elif embedding_dim == 2560:
            self.model_name = "esm2_t36_3B_UR50D"
            self.last_layer = 36
        elif embedding_dim == 5120:
            self.model_name = "esm2_t48_15B_UR50D"
            self.last_layer = 48

        if quantization == True and path_to_model:
            self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
            self.model = torch.load(path_to_model, map_location=device)
        else:
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_hub(self.model_name)
                
        self.model.eval()

        self.max_length = max_length

        self.dummy_param = nn.Parameter(torch.empty(0))
    
    def get_indices(self, alphabet, sequences, max_length):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(sequences)
        seq_str_list = sequences
        seq_encoded_list = [alphabet.encode(seq_str) for seq_str in seq_str_list]
        
        seq_encoded_list = [seq_str[:max_length] for seq_str in seq_encoded_list]
        
        max_len = max_length
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(alphabet.prepend_bos) + int(alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(alphabet.padding_idx)
        
        for i, (seq_encoded) in enumerate(seq_encoded_list):
            if alphabet.prepend_bos:
                tokens[i, 0] = alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(alphabet.prepend_bos) : len(seq_encoded)
                + int(alphabet.prepend_bos),
            ] = seq
            if alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(alphabet.prepend_bos)] = alphabet.eos_idx

        return tokens

    def get_sequences(self, alphabet, batch_token_ids):
        all_sequences = []

        no_sequences = batch_token_ids.shape[0]

        for i in range(no_sequences):
            raw_sequence = "".join(list(map(lambda x: alphabet.get_tok(x), batch_token_ids[i,:])))
            sequence = raw_sequence.replace(" ", "").replace("<cls>", "").replace("<eos>", "")

            all_sequences.append(sequence)
        
        return all_sequences
    
    def encode(self, sequences):

        batch_tokens = self.get_indices(self.alphabet, sequences, self.max_length)

        with torch.no_grad():
            model_output = self.model(batch_tokens.to(self.dummy_param.device), repr_layers=[self.last_layer])
        
        embeddings = model_output["representations"][self.last_layer]

        return torch.transpose(embeddings, 1, 2)
    
    def decode(self, embeddings):
        """ Based on model architecture in https://github.com/facebookresearch/esm/blob/main/esm/model/esm2.py"""

        embeddings = torch.transpose(embeddings, 1, 2)

        with torch.no_grad():
            model_output = self.model.lm_head(embeddings)

            model_output = torch.softmax(model_output, dim=2)

        batch_token_ids = torch.argmax(model_output, dim=2)

        predictions = self.get_sequences(self.alphabet, batch_token_ids)

        return predictions
    
    def compute_pseudo_perplexity(self, sequence):
        batch_tokens = self.get_indices(self.alphabet, [sequence], len(sequence))

        all_batch_tokens = []

        for i in range(1, len(sequence)+1):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = self.alphabet.mask_idx
            all_batch_tokens.append(batch_tokens_masked)

        all_batch_tokens_mask = torch.concat(all_batch_tokens, dim=0).to(self.dummy_param.device)

        with torch.no_grad():
            log_probs = torch.log_softmax(self.model(all_batch_tokens_mask)["logits"], dim=-1)

        all_log_probs = [log_probs[i-1, i, self.alphabet.get_idx(sequence[i-1])].item() for i in range(1, len(sequence)+1)]

        return torch.exp(-torch.tensor(all_log_probs).mean())