# OmegAMP

This project presents a generative model designed for the conditional generation of novel Antimicrobial Peptides (AMPs), a classifier to distinguish between AMPs and non-AMPs and a series of fine-grained classifiers that distinguish between active and inactive peptides against specific species, strains. Together, these components form a framework that helps you discover new AMPs.

## Table of Contents

- [Installation](#installation)
- [Documentation](#documentation)
- [Web Interface](#web-interface)

## Installation

Suggested setup (paste into your terminal):

```sh
conda create -n public-omegamp python=3.11 -y && \
conda activate public-omegamp && \
pip install -e . && \
gdown "https://drive.google.com/uc?id=1KO-6Aa7K5_G03DTiwfCa-gPAIOsuxigd" -O models/generative_model.ckpt && \
gdown "https://drive.google.com/uc?id=1divlvNxsmjYacqb7wb6nK06b8XF8JlOw" -O data/generative-model-data/generative-model-embeddings.h5
```

## Documentation

For detailed usage instructions, see the dedicated guides:

- **[Training Guide](README-Training.md)** — data pre-processing, classifier and generative model training
- **[Inference Guide](README-Inference.md)** — sequence generation and prediction with trained models for advanced programmatic usage

## Web Interface

To run the Streamlit app locally:

```sh
streamlit run app.py
```

## Citation

If you use this project in your research, please cite:

```bibtex
@article{
soares2026omegamp,
title={Omeg{AMP}: Targeted {AMP} Discovery via Biologically Informed Generation},
author={Diogo Soares and Leon Hetzel and Paulina Szymczak and Marcelo Der Torossian Torres and Johanna Sommer and Cesar de la Fuente-Nunez and Fabian J Theis and Stephan G{\"u}nnemann and Ewa Szczurek},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=hAq3XLZ9ex},
note={}
}
```

## Acknowledgements

The code used in this project leverages previous implementations:

- [Denoising Diffusion Probabilistic Models](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main)

