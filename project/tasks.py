import pandas as pd
import pytorch_lightning as pl
from project.blocks import Unet1D
from project.conditioning import Conditioning, ConditioningMasking, ConditioningSampler, PartialConditioningTypes
from project.data import MinMaxNormalization, Standardization, save_sequences_to_fasta
from project.diffusion import GaussianDiffusion1D
import torch
import torch.nn as nn
import math
from .wrappers import EsmWrapper, HydrophobicScaleWrapper, NumericWrapper, combine_scales, OneHotWrapper
from .metrics import FitnessScore, EncoderAverageDistance, AMPProbability, Diversity, FrechetAminoacidEmbeddingDistance, NovelAMP, ConditioningPropertiesMetricsMAE, ConditioningPropertiesMetricsNormalizedMAE, PropertyKSDistance, SequenceEntropy, BatchEntropy, PseudoPerplexity, SampleMetricsCollection, Precision, Recall, Uniqueness
from .constants import AA_SCALES, PADDING_VALUE, kyle_doolittle_scale, eisenberg_scale

class DiffusionTraining(pl.LightningModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule | None = None,
        beta_schedule: str = "sigmoid",
        timesteps: int = 100,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-5,
        lr_decay: float | None = None,
        dim: int = 320,
        optimizer: str = "adam",
        max_epochs: int = 10,
        no_test_samples: int = 50000,
        dim_mults: list = [1,2],
        evaluation_model_embedding_dim: int = 320,
        self_condition: bool = False,
        objective: str = "pred_noise",
        classifier_model_path = None,
        rescaled_phi: float = 0.0,
        encoder_decoder_model: str = "ESM2",
        aa_scales: list = ["wimley_white_with_min_spacing"],
        sample_batch_size: int = 32,
        noise_strength: float = 0.05,
        tau1: float = 0.5,
        tau2: float = 1.0,
        variable_conditioning: bool = True,
        loss_weighting: bool = False,
        no_saved_samples: int = 0,
        sample_save_path: str = "data/generated-samples.fasta"   
        ):
        super().__init__()

        embedding_dim = datamodule.dataset.embeddings.shape[1]
        max_length = datamodule.dataset.embeddings.shape[2]

        embedding_shape = (embedding_dim, max_length)
        
        self.variable_conditioning = variable_conditioning

        model = Unet1D(max_length,
                       dim,
                       channels = embedding_dim,
                       dim_mults=tuple(dim_mults),
                       self_condition=self_condition,
                       num_conditional_features=datamodule.dataset.conditioning.conditioning_vectors.shape[1],
                       variable_conditioning=variable_conditioning)

        self.latent_model = GaussianDiffusion1D(model,
                                                seq_length=max_length,
                                                timesteps=timesteps,
                                                beta_schedule=beta_schedule,
                                                objective=objective,
                                                rescaled_phi=rescaled_phi,
                                                noise_strength = noise_strength,
                                                tau1 = tau1,
                                                tau2 = tau2)
        
        self.conditioning = datamodule.dataset.conditioning

        self.conditioning_sampler = ConditioningSampler(self.conditioning.conditioning_vectors, self.conditioning.conditioning_names)

        self.normalization = MinMaxNormalization(datamodule.dataset.get_all_embeddings())

        self.conditioning_normalization = Standardization(self.conditioning.conditioning_vectors)

        self.conditioning_masking = ConditioningMasking(self.conditioning.computable_names, self.conditioning.uncomputable_names)

        self.no_test_samples = no_test_samples
        self.datamodule = datamodule
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.optimizer = optimizer
        self.max_train_steps = max_epochs * len(self.datamodule.train_dataloader())
        self.classifier_model_path = classifier_model_path
        self.encoder_decoder_model = encoder_decoder_model

        evaluation_scale = combine_scales([eisenberg_scale, kyle_doolittle_scale])
        self.hydrophobic_scale_wrapper = HydrophobicScaleWrapper(evaluation_scale, embedding_shape[1], PADDING_VALUE)

        if self.encoder_decoder_model == "ESM2":
            self.encoder_decoder = EsmWrapper(embedding_shape[0], embedding_shape[1] - 2, self.device)
            self.evaluation_model = EsmWrapper(evaluation_model_embedding_dim, embedding_shape[1] - 2, self.device)
        elif self.encoder_decoder_model == "HydrophobicScaleWrapper":
            scale = combine_scales([AA_SCALES[scale] for scale in aa_scales])
            self.encoder_decoder = HydrophobicScaleWrapper(scale, embedding_shape[1], PADDING_VALUE)
            self.evaluation_model = EsmWrapper(evaluation_model_embedding_dim, embedding_shape[1], self.device)
        elif self.encoder_decoder_model == "OneHotWrapper":
            self.encoder_decoder = OneHotWrapper(embedding_shape[1])
            self.evaluation_model = EsmWrapper(evaluation_model_embedding_dim, embedding_shape[1], self.device)
        elif self.encoder_decoder_model == "NumericWrapper":
            self.encoder_decoder = NumericWrapper(embedding_shape[1])
            self.evaluation_model = EsmWrapper(evaluation_model_embedding_dim, embedding_shape[1], self.device)

        self.no_saved_samples = no_saved_samples
        self.sample_save_path = sample_save_path
        self.sample_batch_size = sample_batch_size
        self.val_sample_metrics = self._sample_metrics("val")
        self.val_conditioning_metrics = self._conditioning_metrics("val")
        self.test_sample_metrics = self._sample_metrics("test")
        self.test_conditioning_metrics = self._conditioning_metrics("test")

        self.loss_weighting = loss_weighting

        number_of_positives = self.conditioning.number_of_amps
        number_of_negatives = self.conditioning.number_of_non_amps

        if self.loss_weighting and number_of_positives > 0 and number_of_negatives > 0:
            total_number_of_samples = number_of_positives + number_of_negatives
            self.positive_weight = torch.sqrt(total_number_of_samples / (2 * number_of_positives))
            self.negative_weight = torch.sqrt(total_number_of_samples / (2 * number_of_negatives))
        else:
            self.positive_weight = 1.0
            self.negative_weight = 1.0
        
        
    def _conditioning_metrics(self, phase: str):
        return SampleMetricsCollection(phase, [ConditioningPropertiesMetricsMAE(self.conditioning, self.conditioning_masking), ConditioningPropertiesMetricsNormalizedMAE(self.conditioning, self.conditioning_masking)])
    
    def _sample_metrics(self, phase: str):
        original_embeddings = self.datamodule.dataset.get_all_amp_embeddings()
        original_sequences = self.datamodule.dataset.get_all_amp_sequences()
        if phase == "val":
            metrics = [SequenceEntropy(), BatchEntropy(), AMPProbability(self.classifier_model_path), NovelAMP(original_sequences), EncoderAverageDistance(self.encoder_decoder),
                        Uniqueness(), FitnessScore()]
        elif phase == "test":
            metrics = [PseudoPerplexity(self.evaluation_model), FrechetAminoacidEmbeddingDistance(self.hydrophobic_scale_wrapper, original_sequences), 
                    SequenceEntropy(), BatchEntropy(), AMPProbability(self.classifier_model_path), NovelAMP(original_sequences), 
                    Precision(self.hydrophobic_scale_wrapper, original_sequences), Recall(self.hydrophobic_scale_wrapper, original_sequences), 
                    PropertyKSDistance(self.conditioning.computable_names, original_sequences), Diversity(), Uniqueness(), FitnessScore()]
        return SampleMetricsCollection(phase, metrics)

    def conditioning_to_model_conditioning(self, conditioning, conditioning_mask, no_actual_samples):
        if self.variable_conditioning:
            if conditioning is None:
                conditioning = self.conditioning_sampler.sample(no_actual_samples)
                conditioning_mask = self.conditioning_masking.conditioning_mask.get_full_mask(no_actual_samples)
            if conditioning is not None and conditioning_mask is None:
                conditioning = conditioning[:no_actual_samples]
                conditioning_mask = self.conditioning_masking.conditioning_mask.get_no_mask(no_actual_samples)
            norm_conditioning = self.conditioning_normalization.normalize(conditioning)
            masked_conditioning = self.conditioning_masking.mask_idxs(norm_conditioning, conditioning_mask)
            model_conditioning = masked_conditioning
        else:
            if conditioning is None:
                conditioning = self.conditioning_sampler.sample(no_actual_samples)
            norm_conditioning = self.conditioning_normalization.normalize(conditioning)
            model_conditioning = norm_conditioning
        
        return conditioning, model_conditioning

    def model_conditioning_to_conditioning_mask(self, model_conditioning):
        if self.variable_conditioning:
            return self.conditioning_masking.get_conditioning_mask(model_conditioning)
        else:
            masked_conditioning = self.conditioning_masking.no_mask(model_conditioning)
            return self.conditioning_masking.get_conditioning_mask(masked_conditioning)

    def sample(self, no_samples, output_embedding=False, batch_size=100, conditioning=None, conditioning_mask=None, 
               reference_embed=None, reference_mask=None, guidance_strength=None, 
               initial_embed=None, initial_timestep=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        no_batches = no_samples // batch_size
        if no_batches * batch_size < no_samples:
            no_batches += 1

        conditioning, model_conditioning = self.conditioning_to_model_conditioning(conditioning, conditioning_mask, no_samples)
        model_conditioning = model_conditioning.to(self.device)
        if reference_embed is not None:
            reference_embed = reference_embed.to(self.device)
        if reference_mask is not None:
            reference_mask = reference_mask.to(self.device)
        if initial_embed is not None:
            initial_embed = initial_embed.to(self.device)
            
        embed_list = []
        sequences = []
        for i in range(no_batches):
            batch_initial_embed = None
            batch_reference_embed = None
            batch_reference_mask = None
            if initial_embed is not None:
                batch_initial_embed = initial_embed[i * batch_size:(i + 1) * batch_size]
            if reference_embed is not None:
                batch_reference_embed = reference_embed[i * batch_size:(i + 1) * batch_size]
            if reference_mask is not None:
                batch_reference_mask = reference_mask[i * batch_size:(i + 1) * batch_size]
            
            batch_conditioning = model_conditioning[i * batch_size:(i + 1) * batch_size]

            batch_embeddings = self.latent_model.sample(
                batch_conditioning,
                batch_size=len(batch_conditioning),
                reference_img=batch_reference_embed,
                reference_mask=batch_reference_mask,
                guidance_strength=guidance_strength,
                initial_img=batch_initial_embed,
                initial_timestep=initial_timestep
            )
            batch_embeddings = self.normalization.denormalize(batch_embeddings)
            
            if output_embedding:
                embed_list.append(batch_embeddings)
            
            batch_sequences = self.encoder_decoder.decode(batch_embeddings)
            sequences += batch_sequences
        
        conditioning_mask = self.model_conditioning_to_conditioning_mask(model_conditioning)

        conditioned_features = (conditioning, conditioning_mask)

        if output_embedding:
            return sequences, torch.cat(embed_list, dim=0), conditioned_features

        return sequences, conditioned_features

    def update_partial_conditioning_params(self, kwargs, list_of_partial_conditioning_info_to_samples):
        conditioning_list = []
        conditioning_mask_list = []
        for partial_conditioning_info, no_samples in list_of_partial_conditioning_info_to_samples:
            conditioning = torch.zeros((no_samples, len(self.conditioning.conditioning_names)))
            for property in partial_conditioning_info:
                partial_conditioning_type = partial_conditioning_info[property][0]
                partial_conditioning_value = partial_conditioning_info[property][1]
                idx = self.conditioning.conditioning_names.index(property)
                if partial_conditioning_type == PartialConditioningTypes.DEFINED:
                    conditioning[:, idx] = partial_conditioning_value
                elif partial_conditioning_type == PartialConditioningTypes.UNDEFINED:
                    pass
                elif partial_conditioning_type == PartialConditioningTypes.INTERVAL:
                    uniform_samples = torch.distributions.Uniform(partial_conditioning_value[0], partial_conditioning_value[1]).sample((no_samples,))
                    if property == "length":
                        conditioning[:, idx] = uniform_samples.round().long()
                    else:
                        conditioning[:, idx] = uniform_samples
            partial_mask = self.conditioning_masking.conditioning_mask.get_partial_mask(partial_conditioning_info, no_samples)

            conditioning_list.append(conditioning)
            conditioning_mask_list.append(partial_mask)

        kwargs["conditioning"] = torch.cat(conditioning_list, dim=0)
        kwargs["conditioning_mask"] = torch.cat(conditioning_mask_list, dim=0)

        return kwargs

    def update_prototype_derived_conditioning_params(self, kwargs, prototype_sequences, conditioning_closeness, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        prototype_derived_conditioning = Conditioning(prototype_sequences, self.conditioning.computable_names, self.conditioning.uncomputable_names, torch.ones((len(prototype_sequences), 1)))
        prototype_derived_conditioning_samples = prototype_derived_conditioning.conditioning_vectors
        
        property_stds = self.conditioning.std.clone()
        property_stds[-1] = 0 # Do not change isAMP
        unif_deviations = torch.distributions.Uniform(0, 1).sample((len(prototype_sequences), len(property_stds))) * 2 - 1
        deviations = unif_deviations * property_stds * conditioning_closeness

        conditioning_samples = prototype_derived_conditioning_samples + deviations
        
        kwargs["conditioning"] = conditioning_samples
        return kwargs
    
    def update_motif_conditioning_params(self, kwargs, raw_motifs, guidance_strength):
        motifs = []
        sequence_masks = []
        for motif in raw_motifs:
            motifs.append(motif.replace("-", "A"))
            sequence_masks.append(torch.zeros(len(motif)))
            for idx, aa in enumerate(motif):
                if aa != "-":
                    sequence_masks[-1][idx] = 1

        motif_embeddings = self.encoder_decoder.encode(motifs)
        motif_embeddings = self.normalization.normalize(motif_embeddings)
        no_motifs, channels, max_length = motif_embeddings.shape[0], motif_embeddings.shape[1], motif_embeddings.shape[2] 

        full_sequence_masks = torch.zeros((no_motifs, channels, max_length), device=self.device)

        for idx, sequence_mask in enumerate(sequence_masks):
            full_sequence_masks[idx, :, :len(sequence_mask)] = sequence_mask

        kwargs["reference_embed"] = motif_embeddings
        kwargs["reference_mask"] = full_sequence_masks
        kwargs["guidance_strength"] = guidance_strength

        return kwargs
    
    def update_analog_conditioning_params(self, kwargs, analogs, initial_timestep):
        analog_embedding = self.encoder_decoder.encode(analogs)
        analog_embedding = self.normalization.normalize(analog_embedding).to(self.device)

        kwargs["initial_embed"] = analog_embedding
        kwargs["initial_timestep"] = initial_timestep

        return kwargs

    def compute_loss(self, batch):
        embeddings = self.normalization.normalize(batch[1])
        conditioning = batch[2]
        random_mask = self.conditioning_masking.conditioning_mask.get_random_mask(conditioning.shape[0])
        _ , model_conditioning = self.conditioning_to_model_conditioning(conditioning, random_mask if self.variable_conditioning else None, conditioning.shape[0])

        if self.loss_weighting:
            class_loss_weights = torch.ones(conditioning.shape[0]) * self.positive_weight
            class_loss_weights[conditioning[:, -1] == 0] = self.negative_weight
            class_loss_weights = class_loss_weights.to(self.device)
            loss = self.latent_model(embeddings, model_conditioning, class_loss_weights=class_loss_weights)
        else:
            loss = self.latent_model(embeddings, model_conditioning)
        return loss

    def get_batch_size(self, batch):
        return batch[1].shape[0]

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        self.log("train/loss", loss, batch_size=self.get_batch_size(batch), prog_bar=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        batch_size = self.get_batch_size(batch)
        
        # Loss metrics
        
        loss = self.compute_loss(batch)
        
        self.log("val/loss", loss, batch_size=batch_size)

        # Unconditional Sampling Metrics

        sequences, embeddings, conditioning_used = self.sample(batch_size, output_embedding=True, batch_size=batch_size)

        metrics = self.val_sample_metrics(embeddings, sequences, conditioning_used)

        self.log_dict(metrics)

        # Conditional Sampling Metrics

        conditioning = self.conditioning_sampler.sample(batch_size)
        evaluation_mask = self.conditioning_masking.conditioning_mask.get_evaluation_mask(batch_size)
        sequences, conditioning_used = self.sample(batch_size, output_embedding=False, batch_size=batch_size, 
                                                    conditioning=conditioning, conditioning_mask=evaluation_mask)
        
        metrics = self.val_conditioning_metrics(embeddings, sequences, conditioning_used)

        self.log_dict(metrics)
        
        return {"loss": loss}
    
    def test_step(self, batch, batch_idx):
        batch_size = self.get_batch_size(batch)
        
        # Loss metrics

        loss = self.compute_loss(batch)
        
        self.log("test/loss", loss, batch_size=batch_size)
        
        # Sampling Metrics

        sequences, conditioning_used = self.sample(self.no_test_samples, output_embedding=False, batch_size=self.sample_batch_size)
        
        embeddings = None # Currently not used in the metrics

        metrics = self.test_sample_metrics(embeddings, sequences, conditioning_used)

        self.log_dict(metrics)

        # Conditional Sampling Metrics

        conditioning = self.conditioning_sampler.sample(batch_size)
        evaluation_mask = self.conditioning_masking.conditioning_mask.get_evaluation_mask(batch_size)
        sequences, conditioning_used = self.sample(batch_size, output_embedding=False, batch_size=batch_size, 
                                                    conditioning=conditioning, conditioning_mask=evaluation_mask)
        
        metrics = self.test_conditioning_metrics(embeddings, sequences, conditioning_used)

        self.log_dict(metrics)
        
        return {"loss": loss}
    
    def on_test_end(self):
        if self.no_saved_samples > 0:
            sequences, _ = self.sample(self.no_saved_samples, output_embedding=False, batch_size=self.sample_batch_size)

            save_sequences_to_fasta(sequences, self.sample_save_path)
            

    def configure_optimizers(self):
        if self.optimizer == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "radam":
            opt = torch.optim.RAdam(self.parameters(), lr=self.learning_rate)
        else:
            raise RuntimeError(f"Unknown optimizer {self.optimizer}")

        # Decay exponentially to the min learning rate over the course of the timesteps
        def decay(step):
            decay_step = (
                math.log(self.min_learning_rate / self.learning_rate)
                / self.max_train_steps
            )
            return math.exp(decay_step * min(step, self.max_train_steps))

        config = {"optimizer": opt}
        if self.lr_decay == "exp":
            config["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, decay),
                "interval": "step",
            }

        return config