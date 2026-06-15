#!/usr/bin/env python

import argparse
import warnings
import torch
from project.config import load_model_for_inference
from project.data import load_fasta_to_df, save_sequences_to_fasta
from hydra import compose, initialize
from project.conditioning import PartialConditioningTypes
import pandas as pd

def load_targeted_conditioning_csv(targeted_conditioning_csv):
    df = pd.read_csv(targeted_conditioning_csv)
    list_of_partial_conditioning_info_to_samples = []
    for index, row in df.iterrows():
        partial_conditioning_info = {}
        partial_conditioning_info['length'] = parse_specification(row['length'])
        partial_conditioning_info['charge'] = parse_specification(row['charge'])
        partial_conditioning_info['hydrophobicity_eisenberg'] = parse_specification(row['hydrophobicity_eisenberg'])
        partial_conditioning_info['isAMP'] = (PartialConditioningTypes.DEFINED, 1)
        list_of_partial_conditioning_info_to_samples.append((partial_conditioning_info, row['num_samples']))
    return list_of_partial_conditioning_info_to_samples

def validate_kwargs(kwargs, num_samples):
    if ('conditioning' in kwargs) and ('reference_embed' in kwargs):
        if kwargs['conditioning'].shape[0] != kwargs['reference_embed'].shape[0]:
            raise ValueError("The number of conditioning samples and reference embeddings must be the same")
    if ('conditioning' not in kwargs) and ('reference_embed' in kwargs):
        if kwargs['reference_embed'].shape[0] != num_samples:
            raise ValueError("The number of reference embeddings must be the same as the number of samples")
    if ('conditioning' in kwargs) and ('initial_embed' in kwargs):
        if kwargs['conditioning'].shape[0] != kwargs['initial_embed'].shape[0]:
            raise ValueError("The number of conditioning samples and initial embeddings must be the same")
    if ('conditioning' not in kwargs) and ('initial_embed' in kwargs):
        if kwargs['initial_embed'].shape[0] != num_samples:
            raise ValueError("The number of initial embeddings must be the same as the number of samples")
    if ('reference_embed' in kwargs) and ('initial_embed' in kwargs):
        if kwargs['reference_embed'].shape[0] != kwargs['initial_embed'].shape[0]:
            raise ValueError("The number of reference embeddings and initial embeddings must be the same")

def obtain_kwargs(generation_mode, 
                  conditioning_strategy, 
                  list_of_partial_conditioning_info_to_samples,
                  analog_sequences, 
                  motif_sequences, 
                  prototype_sequences, 
                  tau,
                  sigma,
                  guidance_strength,
                  seed,
                  timesteps,
                  batch_size,
                  model,
                  num_samples):
    kwargs = {}
    kwargs['output_embedding'] = False
    kwargs['batch_size'] = batch_size
    if seed is not None:
        kwargs['seed'] = seed

    initial_timestep = round(tau * timesteps)
    conditioning_closeness = sigma

    if generation_mode == 'de-novo':
        pass
    elif generation_mode == 'analog':
        analog_sequences = load_fasta_to_df(analog_sequences)['Sequence'].tolist()
        kwargs = model.update_analog_conditioning_params(kwargs, analog_sequences, initial_timestep)
    elif generation_mode == 'motif':
        motif_sequences = load_fasta_to_df(motif_sequences)['Sequence'].tolist()
        kwargs = model.update_motif_conditioning_params(kwargs, motif_sequences, guidance_strength)
    elif generation_mode == 'analog-motif':
        analog_sequences = load_fasta_to_df(analog_sequences)['Sequence'].tolist()
        motif_sequences = load_fasta_to_df(motif_sequences)['Sequence'].tolist()
        kwargs = model.update_analog_conditioning_params(kwargs, analog_sequences, initial_timestep)
        kwargs = model.update_motif_conditioning_params(kwargs, motif_sequences, guidance_strength)

    if conditioning_strategy == 'unconditional':
        pass
    elif conditioning_strategy == 'targeted':
        kwargs = model.update_partial_conditioning_params(kwargs, list_of_partial_conditioning_info_to_samples)
    elif conditioning_strategy == 'prototype-derived':
        prototype_sequences = load_fasta_to_df(prototype_sequences)['Sequence'].tolist()
        kwargs = model.update_prototype_derived_conditioning_params(kwargs, prototype_sequences, conditioning_closeness, seed=seed)

    validate_kwargs(kwargs, num_samples)
    
    return kwargs


def main(generation_mode='de-novo',
         conditioning_strategy='unconditional',
         length='-',
         charge='-',
         hydrophobicity='-',
         targeted_conditioning_csv=None,
         analog_sequences='inference-examples/analog-sequences.fasta',
         motif_sequences='inference-examples/motif-sequences.fasta',
         prototype_sequences='inference-examples/prototype-sequences.fasta',
         tau=0.25,
         sigma=0.0,
         guidance_strength=1.0,
         checkpoint_path='models/generative_model.ckpt',
         output_fasta='results/generative-model-results/script-omegamp-generated-samples.fasta',
         conditioning_output_path='results/generative-model-results/script-omegamp-generated-conditioning.pt',
         num_samples=32,
         batch_size=32,
         seed=None,
         model=None):
    if model is None:
        with initialize(version_base=None, config_path="../../../config"):
            config = compose(config_name="train")
        model = load_model_for_inference(config, checkpoint_path)
    
    timesteps = 1000 # FIXME Hardcoded for now

    if (targeted_conditioning_csv is not None) and (length != '-' or charge != '-' or hydrophobicity != '-'):
        raise ValueError("Targeted conditioning CSV cannot be used with length, charge, or hydrophobicity specifications")

    if targeted_conditioning_csv is not None:
        list_of_partial_conditioning_info_to_samples = load_targeted_conditioning_csv(targeted_conditioning_csv)
    else:
        partial_conditioning_info = {}
        partial_conditioning_info['length'] = parse_specification(length)
        partial_conditioning_info['charge'] = parse_specification(charge)
        partial_conditioning_info['hydrophobicity_eisenberg'] = parse_specification(hydrophobicity)
        partial_conditioning_info['isAMP'] = (PartialConditioningTypes.DEFINED, 1)
        list_of_partial_conditioning_info_to_samples = [(partial_conditioning_info, num_samples)]

    kwargs = obtain_kwargs(generation_mode=generation_mode, 
                           conditioning_strategy=conditioning_strategy, 
                           list_of_partial_conditioning_info_to_samples=list_of_partial_conditioning_info_to_samples,
                           analog_sequences=analog_sequences, 
                           motif_sequences=motif_sequences, 
                           prototype_sequences=prototype_sequences, 
                           tau=tau, 
                           sigma=sigma, 
                           guidance_strength=guidance_strength, 
                           seed=seed,
                           timesteps=timesteps, 
                           batch_size=batch_size, 
                           model=model,
                           num_samples=num_samples)
    
    if 'conditioning' in kwargs:
        n = kwargs['conditioning'].shape[0]
    elif 'initial_embed' in kwargs:
        n = kwargs['initial_embed'].shape[0]
    elif 'reference_embed' in kwargs:
        n = kwargs['reference_embed'].shape[0]
    else:
        n = num_samples
    if n != num_samples:
        warnings.warn(
            f"Generating {n} sequences; --num_samples={num_samples} was ignored because "
            "batch size is determined by FASTA inputs (one output per entry) or by "
            "--targeted-conditioning-csv row counts.",
            UserWarning,
            stacklevel=2,
        )
    sequences, conditioning = model.sample(n, **kwargs)

    # Save conditioning information to a file
    if conditioning_output_path is not None:
        torch.save(conditioning, conditioning_output_path)
        print(f"Conditioning information saved to {conditioning_output_path}")

    # Save sequences to a fasta file
    if output_fasta is not None:
        save_sequences_to_fasta(sequences, output_fasta)
        print(f"Generated {len(sequences)} sequences saved to {output_fasta}")

    return sequences, conditioning

def parse_specification(spec):
    if spec == '-':
        return (PartialConditioningTypes.UNDEFINED, None)
    if ":" in spec:
        return (PartialConditioningTypes.INTERVAL, list(map(float, spec.split(":"))))
    return (PartialConditioningTypes.DEFINED, float(spec))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a model checkpoint and generate sequences.')
    parser.add_argument('generation_mode', type=str, choices=['de-novo', 'analog', 'motif', 'analog-motif'], help='Generation mode: de-novo, analog, motif, analog-motif')
    parser.add_argument('conditioning_strategy', type=str, choices=['unconditional', 'targeted', 'prototype-derived'], help='Conditioning strategy: unconditional, targeted, prototype-derived')
    parser.add_argument('--length', type=str, default='-', help='Length specification for conditional sampling')
    parser.add_argument('--charge', type=str, default='-', help='Charge specification for conditional sampling')
    parser.add_argument('--hydrophobicity', type=str, default='-', help='Hydrophobicity specification for conditional sampling')
    parser.add_argument('--targeted-conditioning-csv', type=str, default=None, help='Path to CSV file with property conditioning information')
    parser.add_argument('--analog_sequences', type=str, default='data/activity-data/curated-AMPs.fasta', help='Path to FASTA file with analog sequences for conditional sampling')
    parser.add_argument('--motif_sequences', type=str, default='data/activity-data/curated-AMPs.fasta', help='Path to FASTA file with motif sequences for conditional sampling')
    parser.add_argument('--prototype_sequences', type=str, default='data/activity-data/curated-AMPs.fasta', help='Path to FASTA file with prototype sequences for prototype-derived conditioning')
    parser.add_argument('--checkpoint_path', type=str, default='models/generative_model.ckpt', help='Path to the model checkpoint')
    parser.add_argument('--tau', type=float, default=0.25, help='Exploration strength for analog sampling')
    parser.add_argument('--sigma', type=float, default=0.0, help='Relaxation strength for analog sampling')
    parser.add_argument('--guidance_strength', type=float, default=1.0, help='Guidance strength for sampling')
    parser.add_argument('--output_fasta', type=str, default='results/generative-model-results/script-omegamp-generated-samples.fasta', help='Path to the output FASTA file for generated sequences')
    parser.add_argument('--conditioning_output_path', type=str, default='results/generative-model-results/script-omegamp-generated-conditioning.pt', help='Path to save the conditioning information')
    parser.add_argument('--num_samples', type=int, default=32, help='Number of sequences to sample')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for sampling')
    parser.add_argument('--seed', type=int, default=None, help='Seed for sampling')

    args = parser.parse_args()


    main(
        generation_mode=args.generation_mode,
        conditioning_strategy=args.conditioning_strategy,
        length=args.length,
        charge=args.charge,
        hydrophobicity=args.hydrophobicity,
        targeted_conditioning_csv=args.targeted_conditioning_csv,
        analog_sequences=args.analog_sequences,
        motif_sequences=args.motif_sequences,
        prototype_sequences=args.prototype_sequences,
        checkpoint_path=args.checkpoint_path,
        tau=args.tau,
        sigma=args.sigma,
        guidance_strength=args.guidance_strength,
        output_fasta=args.output_fasta,
        conditioning_output_path=args.conditioning_output_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )