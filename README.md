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

### MNIST ELBO experiments

```bash
cd MNIST/scripts
python3 ELBO_experiments.py
```


