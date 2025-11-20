import argparse
from loguru import logger
import os
from pathlib import Path
from datetime import datetime
import sys
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

class NewsSummarizationDataset(Dataset):
    """Dataset for news summarization"""
    def __init__(self, articles: List[str], summaries: List[str], tokenizer, max_length: int):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = self.articles[idx]
        summary = self.summaries[idx]
        
        # Format: "Article: {article}\n\nSummary: {summary}"
        text = f"Article: {article}\n\nSummary: {summary}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),  # Tokenied input sequence
            "attention_mask": encoding["attention_mask"].squeeze(),  # Attention mask for padding tokens
            "labels": encoding["input_ids"].squeeze()  # Labels for the input sequence
        }

def _multiprocessing_generate_worker(
    rank: int,
    gpu_id: int,
    prompts: List[str],
    model_dir: str,
    tokenizer_dir: str,
    max_new_tokens: int,
    prompt_max_length: int,
    temperature: float,
    top_p: float,
    batch_size_per_gpu: int,
    num_generations: int,
    return_dict
):
    """
    Worker process for multi-GPU generation. Each worker loads a model replica on
    a dedicated GPU, processes its chunk of prompts, and returns the decoded
    outputs (list of list[str], preserving all generation candidates).
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        model.eval()
        
        generated_per_prompt: List[List[str]] = []
        
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size_per_gpu):
                batch_prompts = prompts[start:start + batch_size_per_gpu]
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=prompt_max_length,
                    padding=True
                ).to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=num_generations
                )
                
                outputs = outputs.view(len(batch_prompts), num_generations, -1).cpu()
                
                for prompt_idx, prompt in enumerate(batch_prompts):
                    prompt_generations: List[str] = []
                    for seq in outputs[prompt_idx]:
                        text = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                        prompt_generations.append(text)
                    generated_per_prompt.append(prompt_generations)
        
        model.cpu()
        torch.cuda.empty_cache()
        return_dict[rank] = generated_per_prompt
    except Exception as exc:
        return_dict[rank] = exc

def setup_logger(log_dir: str, log_level: str = "INFO"):
    """
    Set up loguru logger that writes to both console and file
    
    Args:
        log_dir: Directory to save log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Remove default handler
    logger.remove()
    
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Add file handler (detailed, all levels)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",  # Rotate when file reaches 10MB
        retention="7 days",  # Keep logs for 7 days
        compression="zip",  # Compress old logs
        enqueue=True,  # Thread-safe logging
        backtrace=True,  # Show full stack trace
        diagnose=True  # Show variable values in traceback
    )
    
    # Add console handler (configurable level)
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level=log_level.upper(),
        colorize=True
    )
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    logger.info(f"Log level: {log_level}")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Iterative Retraining Pipeline for News Summarization"
    )
    
    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Path to the model (e.g., smolm2-135MB-Instruct or HuggingFace model name)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length (block size) for tokenization (default: 1024, matching Hugging Face Transformers v2 training recipe)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If None, auto-detect"
    )
    parser.add_argument(
        "--use_multi_gpu",
        default=True,
        action="store_true",
        help="If set, use all available GPUs with DataParallel for generation (training uses single GPU)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of CPU workers for data loading (default: 4, increase to use more CPUs)"
    )
    
    # Data parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="xsum",
        help="Dataset name (e.g., 'xsum')"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use (default: 'train')"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500000,
        help="Maximum number of samples to load from dataset (default: 500000 for full XSum run)"
    )
    parser.add_argument(
        "--train_test_split_ratio",
        type=float,
        default=0.8,
        help="Ratio for train/test split (default: 0.8 for 80%% train, 20%% test)"
    )
    parser.add_argument(
        "--base_data_ratio",
        type=float,
        default=0.125,
        help="Ratio of data to use for initial training (e.g., 0.125 for 12.5%%)"
    )
    parser.add_argument(
        "--initial_selection_method",
        type=str,
        default="random",
        choices=["random", "first", "diverse"],
        help="Method to select initial samples (default: 'random')"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_generations",
        type=int,
        default=2,
        help="Number of candidate generations to create before filtering (e.g., 10)"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="Generate a news article and its summary.\n\nArticle:",
        help="Prompt template for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=256,
        help="Maximum length for prompt tokenization (default: 256)"
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=256,
        help="Batch size per generation step (default: 256; effective batch grows with multi-GPU)"
    )
    parser.add_argument(
        "--use_ground_truth_articles",
        default=True,
        action="store_true",
        help="If set, use ground truth articles and only generate summaries"
    )
    
    # Training parameters
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="Number of iterative retraining rounds (default: 1 for toy example)"
    )
    parser.add_argument(
        "--synthetic_samples_per_iteration",
        type=int,
        default=10,
        help="Number of synthetic samples to generate per iteration (default: 10 for toy example)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32 to match HF Transformers v2 recipe; increase if memory allows)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs per round (default: 1 for toy example)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for training (default: 5e-5, matching HF Transformers v2 recipe)"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup","none"],
        help="Learning rate scheduler (default: cosine, matching HF Transformers v2 recipe)"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup ratio for LR scheduler (fraction of total steps, default: 0.0)"
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizer type (default: 'adamw')"
    )
    
    # Output and misc parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/net/scratch/yuweicheng/output",
        help="Output directory for saving models (default: '/net/scratch/yuweicheng/output')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Checkpoint parameters
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="If set, save checkpoints during training"
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1, only at end of training)"
    )
    parser.add_argument(
        "--save_optimizer",
        action="store_true",
        help="If set, save optimizer state in checkpoints"
    )
    
    # Logging parameters
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    
    return parser.parse_args()