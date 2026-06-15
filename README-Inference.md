# OmegAMP — Inference Guide

Generate AMP sequences with `project/scripts/inference/generate_samples.py`. Inference combines two choices:

- **Generation mode** — how the diffusion trajectory starts and which positions are constrained
- **Conditioning strategy** — how the property profile `[length, charge, hydrophobicity_eisenberg, isAMP]` is set

Any valid pair can be combined (e.g. analog + targeted → **OmegAMP-AT**).

---

## Programmatic usage

Load the model once, then call `generate_samples.main()`:

```python
from hydra import compose, initialize
from project.config import load_model_for_inference
from project.scripts.inference import generate_samples

with initialize(version_base=None, config_path="config/"):
    config = compose(config_name="train")

model = load_model_for_inference(config, "models/generative_model.ckpt")

sequences, _ = generate_samples.main(
    generation_mode="de-novo",
    conditioning_strategy="targeted",
    length="20",
    charge="6",
    hydrophobicity="-",
    num_samples=1,
    batch_size=1,
    model=model,
    output_fasta=None,              # skip writing to disk
    conditioning_output_path=None,
)

print(f"OmegAMP generated {sequences}")
```

---

## CLI usage

```sh
python project/scripts/inference/generate_samples.py <generation_mode> <conditioning_strategy> [options]
```

**Examples:**

```sh
# De novo, unconditional
python project/scripts/inference/generate_samples.py de-novo unconditional --num_samples 32

# Analog + prototype-derived properties
python project/scripts/inference/generate_samples.py analog prototype-derived --analog_sequences inference-examples/analog-sequences.fasta --tau 0.15 --prototype_sequences inference-examples/prototype-sequences.fasta --sigma 0.75

# Multiple targeted profiles from CSV
python project/scripts/inference/generate_samples.py de-novo targeted --targeted-conditioning-csv inference-examples/targeted-properties.csv
```

---

## Generation modes


| Mode           | CLI value      | Description                                                   | Required inputs                           |
| -------------- | -------------- | ------------------------------------------------------------- | ----------------------------------------- |
| De novo        | `de-novo`      | Trajectory starts from Gaussian noise                         | —                                         |
| Analog         | `analog`       | Trajectory starts from a partially noised prototype embedding | `--analog_sequences` (FASTA)              |
| Motif-guided   | `motif`        | Fixed positions are held during denoising                     | `--motif_sequences` (FASTA)               |
| Analog + Motif | `analog-motif` | Combines analog initialization and fixed motif positions      | `--analog_sequences`, `--motif_sequences` |


**Motif syntax:** use amino acids for fixed positions and `-` for positions to design (e.g. `KR--L---WK`). One FASTA entry per sample.

**FASTA inputs:** for `--analog_sequences`, `--motif_sequences`, and `--prototype_sequences`, the model generates one sequence per FASTA entry, in the same order as in the input file. The total number of generated sequences equals the number of entries in the relevant FASTA file(s); when both analog and motif inputs are used, each file must contain the same number of entries and outputs are paired by position. In these modes, `--num_samples` is ignored—output count always follows the FASTA entry count (or per-row `num_samples` in `--targeted-conditioning-csv`). To generate fewer sequences, use a smaller FASTA file; to pair specific analogs, motifs, and prototypes, keep files aligned by line order.

---

## Conditioning strategies


| Strategy          | CLI value           | Description                                                              | Required inputs                                                             |
| ----------------- | ------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Unconditional     | `unconditional`     | Only `isAMP=1` is fixed                                                  | —                                                                           |
| Targeted          | `targeted`          | Explicit values or ranges for length, charge, and/or hydrophobicity      | `--length`, `--charge`, `--hydrophobicity` or `--targeted-conditioning-csv` |
| Prototype-derived | `prototype-derived` | Properties computed from prototype sequence(s), with optional relaxation | `--prototype_sequences` (FASTA)                                             |


**Property specification** (for `--length`, `--charge`, `--hydrophobicity`):


| Syntax      | Example | Meaning                                           |
| ----------- | ------- | ------------------------------------------------- |
| Unspecified | `-`     | Property is sampled from the learned distribution |
| Exact value | `20`    | Fixed target                                      |
| Interval    | `15:25` | Uniform sample in `[lo, hi]` per sequence         |


**Targeted CSV columns:** `length`, `charge`, `hydrophobicity_eisenberg`, `num_samples`. Cannot be combined with `--length` / `--charge` / `--hydrophobicity` flags.

---

## Hyperparameters


| Parameter                   | CLI flag                     | Default                                                                     | Applies to             | Description                                                                                                 |
| --------------------------- | ---------------------------- | --------------------------------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| Number of samples           | `--num_samples`              | `32`                                                                        | De novo, targeted (CLI flags) | Sequences to generate; ignored when FASTA inputs or `--targeted-conditioning-csv` set the batch size (see above) |
| Batch size                  | `--batch_size`               | `32`                                                                        | All                    | Inference batch size                                                                                        |
| Random seed                 | `--seed`                     | `None`                                                                      | All                    | Reproducible sampling                                                                                       |
| Tau (exploration strength)  | `--tau`                      | `0.25`                                                                      | Analog, Analog + Motif | Fraction of forward noise applied before reverse diffusion; `initial_timestep = round(tau × 1000)`          |
| Sigma (property relaxation) | `--sigma`                    | `0.0`                                                                       | Prototype-derived      | Allowed deviation from prototype properties; `0` = exact match, `1` = up to ±1 training-set SD per property |
| Motif guidance strength     | `--guidance_strength`        | `1.0`                                                                       | Motif, Analog + Motif  | Multiplier on the reconstruction gradient at fixed motif positions; values above 1 can lead to unstable denoising |
| Checkpoint                  | `--checkpoint_path`          | `models/generative_model.ckpt`                                              | All                    | Model weights                                                                                               |
| Output FASTA                | `--output_fasta`             | `results/generative-model-results/script-omegamp-generated-samples.fasta`   | All                    | Generated sequences                                                                                         |
| Conditioning output         | `--conditioning_output_path` | `results/generative-model-results/script-omegamp-generated-conditioning.pt` | All                    | Saved conditioning tensors                                                                                  |


---

## Mode combinations (paper labels)


| Label       | Generation mode | Conditioning strategy |
| ----------- | --------------- | --------------------- |
| OmegAMP-DU  | `de-novo`       | `unconditional`       |
| OmegAMP-DT  | `de-novo`       | `targeted`            |
| OmegAMP-DP  | `de-novo`       | `prototype-derived`   |
| OmegAMP-AU  | `analog`        | `unconditional`       |
| OmegAMP-AT  | `analog`        | `targeted`            |
| OmegAMP-AP  | `analog`        | `prototype-derived`   |
| OmegAMP-MU  | `motif`         | `unconditional`       |
| OmegAMP-MT  | `motif`         | `targeted`            |
| OmegAMP-MP  | `motif`         | `prototype-derived`   |
| OmegAMP-AMU | `analog-motif`  | `unconditional`       |
| OmegAMP-AMT | `analog-motif`  | `targeted`            |
| OmegAMP-AMP | `analog-motif`  | `prototype-derived`   |


---

## Predict sequences

To classify generated sequences, use `project/scripts/inference/predict_sequences.py`:

```sh
python project/scripts/inference/predict_sequences.py <path/to/sequences.fasta> --classifier broad-classifier
```

For a browser-based workflow, see the [Web Interface](README.md#web-interface) (`streamlit run app.py`).