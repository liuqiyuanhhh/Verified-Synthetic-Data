# XSUM Iterative Retraining Pipeline

Iterative retraining pipeline for news summarization on the XSUM dataset. The pipeline fine-tunes a language model on article-summary pairs, generates synthetic summaries, selects a subset (filtered by ROUGE-1 or randomly), and retrains on combined data.

## Overview

1. **Initial training**: Train on 12.5% of the dataset (configurable via `--base_data_ratio`)
2. **Synthetic generation**: Generate summaries for training articles using the current model
3. **Selection**:
   - **Filtered** (default): Select top N% synthetic samples by ROUGE-1 vs ground truth
   - **Unfiltered** (`--use_unfiltered_synthetic_data`): Randomly select N% synthetic samples (N = 12.5%)
4. **Retrain** on selected synthetic data and repeat

## Installation

```bash
# From project root
pip install -r requirements.txt
```

Dependencies: `torch`, `transformers`, `datasets`, `numpy`, `tqdm`, `loguru`, `rouge-score`

## Replication Instructions

### Quick start (defaults)

```bash
cd XSUM
python -m trainer
```

### Basic run with explicit parameters

```bash
python -m XSUM.trainer \
    --model_name "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --max_length 1024 \
    --dataset_name "xsum" \
    --dataset_split "train" \
    --base_data_ratio 0.125 \
    --num_iterations 5 \
    --synthetic_samples_per_iteration 1000 \
    --batch_size 32 \
    --num_epochs 1 \
    --learning_rate 5e-5 \
    --max_new_tokens 512 \
    --output_dir "./output"
```

### Filtered vs unfiltered synthetic data

- **Filtered** (default): Top N% by ROUGE-1
```bash
python -m XSUM.trainer --output_dir ./output_filtered
```

- **Unfiltered**: Randomly select N% samples
```bash
python -m XSUM.trainer --use_unfiltered_synthetic_data --output_dir ./output_unfiltered
```

### Toy run (small dataset, few iterations)

```bash
python -m XSUM.trainer \
    --max_samples 5000 \
    --num_iterations 2 \
    --synthetic_samples_per_iteration 100 \
    --batch_size 8
```

### Full replication with checkpoints

```bash
python -m XSUM.trainer \
    --model_name "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --max_length 1024 \
    --max_samples 100000 \
    --num_iterations 20 \
    --synthetic_samples_per_iteration 5000 \
    --batch_size 32 \
    --num_epochs 1 \
    --learning_rate 5e-5 \
    --output_dir "./output" \
    --save_checkpoints \
    --checkpoint_frequency 1 \
    --save_optimizer
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | HuggingFaceTB/SmolLM2-135M-Instruct | Model for training and generation |
| `--max_length` | 256 | Max sequence length for tokenization |
| `--max_samples` | None | Max samples to load (None = full dataset) |
| `--base_data_ratio` | 0.125 | Initial data fraction and synthetic selection ratio |
| `--num_iterations` | 20 | Retraining rounds |
| `--synthetic_samples_per_iteration` | 10 | Synthetic samples per round |
| `--use_unfiltered_synthetic_data` | False | Use random selection instead of ROUGE-1 filtering |
| `--use_multi_gpu` | True | Multi-GPU for generation |
| `--output_dir` | /net/scratch/yuweicheng/output_linear | Output directory |

Run `python -m XSUM.trainer --help` for all options.

## Output Structure

```
output_dir/
├── training_YYYYMMDD_HHMMSS.log   # Log file
└── round_{i}/
    └── final/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer files
        ├── training_metadata.json
        └── optimizer.pt (if --save_optimizer)
```

## Logging

- Logs are saved to `output_dir/training_YYYYMMDD_HHMMSS.log`
- Use `--log_level DEBUG` for detailed logs
- Default: `INFO`
