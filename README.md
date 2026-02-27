# Verified-Synthetic-Data

This repository hosts the implementation for our research on iterative retraining with verified synthetic data in generative models ([arXiv: 2510.16657](https://arxiv.org/abs/2510.16657)).
The code is maintained for research purposes and academic reproducibility.

---

## Overview

Synthetic data can be used to augment or replace real data in training generative models. However, repeatedly retraining models on their own generated data may lead to degradation or collapse. This repository explores **verification-based synthetic retraining**, where an external verifier (e.g., a discriminator) filters synthetic samples before they are reused for training.

The experiments focus on understanding:

- Short-term improvements from verified synthetic data
- Long-term behavior under iterative retraining
- The impact of verifier quality and filtering strength
- Trade-offs between sample quality and diversity

---

## MNIST CVAE Experiments

The MNIST experiments investigate iterative retraining of a **Conditional Variational Autoencoder (CVAE)** under low-resource settings.

### Setting

- An initial model is trained on a small subset of real MNIST samples
- Synthetic samples are generated from the current model at each round
- A verifier (implemented as a discriminator) scores synthetic samples
- Only top-ranked synthetic samples are retained for retraining

### Experimental Variations

- **Synthetic sample schedule**
  - Fixed number of retained samples per round
  - Linearly increasing number of retained samples
- **Verifier quality**
  - Verifiers trained with different amounts of real data
- **Filtering strength**
  - Different retention thresholds for synthetic samples

### Metrics

- Fréchet Inception Distance (FID)
- Reconstruction and likelihood-based metrics 

---

## Research Design

This codebase reflects an **exploratory research workflow**:

- Prioritizes explicit experiment logic over complex APIs.
- Emphasis on interpretability and analysis of training dynamics.
- Scripts are designed to be read and modified by researchers.
- Interfaces may change to accommodate research needs.

---


## Replication Instructions

### Quick Start (defaults)

Run from the **repo root**:

```bash
python MNIST/scripts/run_full_pipeline.py
```

**What this does by default:**
- Downloads MNIST to `./data`
- Uses `base_seed = 0` for `torch / numpy / random`
- Trains an initial Conv-CVAE on a balanced subset of size `init_size = 500`
- Runs `k = 20` synthetic retraining rounds with:
  - `threshold = 0.1` (discriminator selection threshold)
  - a linear synthetic size schedule from `30,000` to `1,000,000`
  - CVAE training: `epochs=200`, `patience=5`, `lr=1e-3`
  - discriminator training: `epochs=80`, `patience=5`, `lr=1e-3`
- Saves results (CSV), model checkpoints, and sample grids

---

### Basic run with explicit parameters

The script currently exposes **one** CLI argument:

- `--fixed-size` (default: `FIXED_SIZE` env var if set, otherwise `5000`)

> Note: In the current version of the script, `--fixed-size` is parsed and stored
> as `FIXED`, but it is **not used downstream** (i.e., changing it will not change
> training behavior unless you wire it into the pipeline).

Example:

```bash
FIXED_SIZE=5000 python MNIST/scripts/run_full_pipeline.py --fixed-size 5000
```

---

### Key settings (from code)

| Setting | Default | Description |
|---|---:|---|
| Device | `cuda` if available else `cpu` | Automatic device selection |
| Seed | `0` | Set for `torch`, `torch.cuda`, `numpy`, `random` |
| `init_size` | `500` | Balanced real subset for initial model |
| `k` | `20` | Number of retraining rounds |
| `threshold` | `0.1` | Discriminator filtering threshold |
| Synthetic size schedule | `30k → 1,000k` over 20 rounds | Rounded to multiples of 10 |
| CVAE epochs | `200` | With early stopping (`patience=5`) |
| Discriminator epochs | `80` | With early stopping (`patience=5`) |
| Latent dim | `20` | CVAE latent dimension |
| FID generation size | `6000` | Images generated per FID evaluation |

---

### Checkpoint dependency (required)

This pipeline **loads** a pretrained Conv-CVAE checkpoint trained on the full dataset
to build the discriminator training set:

```
MNIST/conv_cvae/model_saved_full_dataset/full_dataset_model.pth
```

If this file is missing, the script will fail at `load_state_dict(...)`.

---

### Outputs

The pipeline writes outputs under:

```
MNIST/conv_cvae/one_strong_vae_appendix/
```

Expected structure:

```
MNIST/conv_cvae/one_strong_vae_appendix/
├── model_saved_more/                    # CVAE checkpoints for each round
├── data_saved_more/                     # round temp synthetic dirs (removed after each round)
├── results_saved_more/
│   └── results_table_{init_size}_{k}rounds.csv
└── picture_saved_more/
    ├── initial_model_samples_{init_size}.png
    └── round{round_id}_model_samples.png
```

---


### MNIST ELBO experiments

```bash
cd MNIST/scripts
python3 ELBO_experiments.py
```


