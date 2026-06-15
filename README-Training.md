# OmegAMP — Training Guide

This guide covers data pre-processing and model training. For installation and the web interface, see the [main README](README.md).

## Table of Contents

- [Generative model](#generative-model)
  - [Pre-processing](#generative-model-pre-processing)
  - [Training](#generative-model-training)
- [Classifier](#classifier)
  - [Additional installation](#classifier-additional-installation)
  - [Pre-processing](#classifier-pre-processing)
  - [Training](#classifier-training)

---

## Generative model

### Pre-processing

#### Generative model dataset

Create a dataset for training the generative model by combining AMP and non-AMP sequences:

```sh
python project/scripts/pre-processing/dataset/get_generative_dataset.py [--positive_fasta_file PATH] [--output_csv_file PATH] [--min_length N] [--max_length N]
```

#### Embeddings generation

Generate embeddings using amino-acid scale based encoders:

```sh
python project/scripts/pre-processing/embeddings/get_scale_embeddings.py [--csv_file PATH] [--scale SCALE] [--output_h5_file PATH] [--max_length N]
```

### Training

To train the generative model, call `train.py` (all the necessary hyperparameter configs are in **config/**). Note: you need to have a [W&B](https://wandb.ai) account.

```sh
python train.py
```

The model will be saved in the **wandb/latest-run/** folder. Afterwards, replace **models/generative_model.ckpt** with your newly trained checkpoint to use it for sampling.

---

## Classifier

### Classifier — Additional installation

Download the activity dataset:

```sh
gdown "https://drive.google.com/uc?id=1t-lCR-SLy1Vhh8UpFb4SvR9GZfcwznFo" -O data/activity-data/activity-dataset.csv
```

### Pre-processing

#### Synthetic negative samples

Generate synthetic negative samples using different strategies:

```sh
# Random sequences
python project/scripts/pre-processing/dataset/get_synthetic_data.py --mode random [--number_of_non_amps N] [--length_filter N]

# Shuffled AMP sequences
python project/scripts/pre-processing/dataset/get_synthetic_data.py --mode shuffled [--number_of_non_amps N] [--length_filter N]

# Mutated AMP sequences
python project/scripts/pre-processing/dataset/get_synthetic_data.py --mode mutated [--mutations N] [--number_of_non_amps N] [--length_filter N]
```

#### Activity dataset

Create the classifier dataset by combining positive and negative samples and extracting input features:

```sh
python project/scripts/pre-processing/dataset/get_activity_dataset.py [--output_csv PATH] [--curated_amp_file_path PATH] [--curated_non_amp_file_path PATH]
```

### Training

Train individual classifiers or all classifiers at once:

```sh
# Train a single classifier
python project/scripts/training/train_classifier.py --classifier broad-classifier

# Train all classifiers
python project/scripts/training/train_classifier.py --classifier all
```

Trained models are saved to **models/** (e.g. `models/broad-classifier.json`).

