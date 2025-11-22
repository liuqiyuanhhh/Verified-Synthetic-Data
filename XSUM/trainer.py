"""
Iterative Retraining Pipeline for News Summarization
Uses smolm2 135MB-Instruct model for synthetic data generation
"""

import os
import json
import math
import shutil
import tempfile
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_scheduler
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import random
from pathlib import Path
from utils import setup_logger, parse_args, _multiprocessing_generate_worker, NewsSummarizationDataset
from loguru import logger
from rouge_score import rouge_scorer


class IterativeSummarizationTrainer:
    """Main class for iterative retraining pipeline"""
    
    def __init__(
        self,
        model_name: str,
        base_data_ratio: float,
        num_generations: int,
        max_length: int,
        device: Optional[str],
        output_dir: str,
        seed: int,
        use_multi_gpu: bool = False,
        num_workers: int = 4
    ):
        self.model_name = model_name
        self.base_data_ratio = base_data_ratio
        self.num_generations = num_generations
        self.max_length = max_length
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.seed = seed
        self.use_multi_gpu = use_multi_gpu
        self.num_workers = num_workers
        
        # Use global loguru logger
        self.logger = logger
        
        # Log initialization
        self.logger.info("=" * 60)
        self.logger.info("Initializing IterativeSummarizationTrainer")
        self.logger.info("=" * 60)
        self.logger.info(f"Model name: {model_name}")
        self.logger.info(f"Base data ratio: {base_data_ratio}")
        self.logger.info(f"Number of generations: {num_generations}")
        self.logger.info(f"Max length: {max_length}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Seed: {seed}")
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.logger.debug(f"Random seeds set to {seed}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer and model
        self.logger.info(f"Loading model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.debug("Set pad_token to eos_token")
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            
            # Training uses single GPU only (not wrapped with DataParallel)
            # Multi-GPU is only used for generation (handled in generation methods)
            self.model.train()
            
            self.logger.info(f"Model loaded successfully on device: {self.device} (training uses single GPU)")
            if self.use_multi_gpu and torch.cuda.device_count() > 1:
                self.logger.info(f"Multi-GPU ({torch.cuda.device_count()} GPUs) will be used for generation only")
            self.logger.debug(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            self.logger.debug(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        except Exception as e:
            self.logger.exception(f"Failed to load model: {e}")
            raise
    
    def _create_temp_inference_assets(self) -> Tuple[str, str, str]:
        """Save current model/tokenizer to a temporary directory for worker processes."""
        temp_root = tempfile.mkdtemp(prefix="inference_", dir=self.output_dir)
        model_dir = os.path.join(temp_root, "model")
        tokenizer_dir = os.path.join(temp_root, "tokenizer")
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(tokenizer_dir)
        return temp_root, model_dir, tokenizer_dir
    
    def _single_gpu_generate_prompts(
        self,
        prompts: List[str],
        max_new_tokens: int,
        prompt_max_length: int,
        temperature: float,
        top_p: float,
        generation_batch_size: int,
        num_generations: int,
        description: str
    ) -> List[List[str]]:
        """Generate texts sequentially on the current device."""
        self.logger.info(f"{description} on single GPU (batch size: {generation_batch_size})")
        self.model.eval()
        results: List[List[str]] = []
        
        total_batches = (len(prompts) + generation_batch_size - 1) // generation_batch_size
        batch_indices = range(0, len(prompts), generation_batch_size)
        
        with torch.no_grad():
            for start in tqdm(
                batch_indices,
                total=total_batches,
                desc=description,
                unit="batch"
            ):
                batch_prompts = prompts[start:start + generation_batch_size]
                original_padding_side = self.tokenizer.padding_side
                self.tokenizer.padding_side = 'left'
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=prompt_max_length,
                    padding=True
                ).to(self.device)
                self.tokenizer.padding_side = original_padding_side
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=num_generations
                )
                
                outputs = outputs.view(len(batch_prompts), num_generations, -1).cpu()
                
                for prompt_idx, _ in enumerate(batch_prompts):
                    prompt_generations: List[str] = []
                    for seq in outputs[prompt_idx]:
                        text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                        prompt_generations.append(text)
                    results.append(prompt_generations)
        
        self.model.train()
        return results
    
    def _multi_gpu_generate_prompts(
        self,
        prompts: List[str],
        max_new_tokens: int,
        prompt_max_length: int,
        temperature: float,
        top_p: float,
        generation_batch_size: int,
        num_generations: int,
        description: str
    ) -> List[List[str]]:
        """Generate texts in parallel using multiple GPUs via multiprocessing."""
        num_gpus = torch.cuda.device_count()
        if num_gpus <= 1:
            return self._single_gpu_generate_prompts(
                prompts,
                max_new_tokens,
                prompt_max_length,
                temperature,
                top_p,
                generation_batch_size,
                num_generations,
                description
            )
        
        self.logger.info(f"{description} using {num_gpus} GPUs in parallel...")
        temp_root, model_dir, tokenizer_dir = self._create_temp_inference_assets()
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()
        
        processes = []
        chunk_size = math.ceil(len(prompts) / num_gpus)
        
        try:
            # Start all processes
            for rank, gpu_id in enumerate(range(num_gpus)):
                start = rank * chunk_size
                chunk_prompts = prompts[start:start + chunk_size]
                if not chunk_prompts:
                    continue
                
                process = ctx.Process(
                    target=_multiprocessing_generate_worker,
                    args=(
                        rank,
                        gpu_id,
                        chunk_prompts,
                        model_dir,
                        tokenizer_dir,
                        max_new_tokens,
                        prompt_max_length,
                        temperature,
                        top_p,
                        generation_batch_size,
                        num_generations,
                        return_dict
                    )
                )
                process.start()
                processes.append((rank, process))
            
            # Wait for processes with progress bar
            with tqdm(total=len(processes), desc=f"{description} (multi-GPU)", unit="GPU") as pbar:
                for rank, process in processes:
                    process.join()
                    if process.exitcode != 0:
                        raise RuntimeError(f"Generation worker on GPU {rank} exited with code {process.exitcode}")
                    pbar.update(1)
                    pbar.set_postfix({"completed": f"{pbar.n}/{pbar.total} GPUs"})
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
        
        results: List[List[str]] = []
        for rank in sorted(return_dict.keys()):
            value = return_dict[rank]
            if isinstance(value, Exception):
                raise value
            results.extend(value)
        
        return results
    
    def _generate_prompts(
        self,
        prompts: List[str],
        max_new_tokens: int,
        prompt_max_length: int,
        temperature: float,
        top_p: float,
        generation_batch_size: int,
        num_generations: int,
        description: str
    ) -> List[List[str]]:
        """Dispatch generation to single or multi-GPU path."""
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            return self._multi_gpu_generate_prompts(
                prompts,
                max_new_tokens,
                prompt_max_length,
                temperature,
                top_p,
                generation_batch_size,
                num_generations,
                description
            )
        return self._single_gpu_generate_prompts(
            prompts,
            max_new_tokens,
            prompt_max_length,
            temperature,
            top_p,
            generation_batch_size,
            num_generations,
            description
        )
    
    def load_checkpoint(
        self,
        checkpoint_dir: str,
        load_optimizer: bool = False,
        optimizer_type: str = "adamw",
        learning_rate: float = 5e-5
    ) -> Optional[torch.optim.Optimizer]:
        """
        Load a checkpoint (model, tokenizer, and optionally optimizer)
        
        Args:
            checkpoint_dir: Directory containing the checkpoint
            load_optimizer: Whether to load optimizer state
            optimizer_type: Type of optimizer (if creating new one)
            learning_rate: Learning rate (if creating new optimizer)
        
        Returns:
            Optimizer if load_optimizer is True, None otherwise
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load model and tokenizer
        try:
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            self.model.to(self.device)
            # Training uses single GPU only
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            self.logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            self.logger.exception(f"Failed to load checkpoint: {e}")
            raise
        
        optimizer = None
        if load_optimizer:
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(optimizer_path):
                # Create optimizer first
                if optimizer_type.lower() == "adamw":
                    optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
                elif optimizer_type.lower() == "adam":
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
                elif optimizer_type.lower() == "sgd":
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
                else:
                    raise ValueError(f"Unknown optimizer type: {optimizer_type}")
                
                # Load optimizer state
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
                self.logger.info(f"Optimizer state loaded from {optimizer_path}")
            else:
                self.logger.warning(f"Optimizer checkpoint not found at {optimizer_path}")
        
        # Load metadata if available
        metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.logger.info(f"Checkpoint metadata: {json.dumps(metadata, indent=2)}")
        else:
            self.logger.debug(f"No metadata file found at {metadata_path}")
        
        self.logger.info(f"Checkpoint loaded successfully")
        return optimizer
    
    def load_dataset(self, dataset_name: str, split: str = None, max_samples: int = None, test_max_samples: int = None, train_test_split_ratio: float = None):
        """
        Load news summarization dataset from train and validation splits
        
        Args:
            dataset_name: Name of the dataset (e.g., 'xsum')
            split: Deprecated - kept for backward compatibility but ignored. 
                   Data is loaded from 'train' and 'validation' splits.
            max_samples: Deprecated - kept for backward compatibility but will be ignored.
                        All samples from 'train' split will be loaded.
            test_max_samples: Maximum number of samples to load from 'validation' split (None for all)
            train_test_split_ratio: Deprecated - kept for backward compatibility but ignored.
                                    Test data comes from 'validation' split.
        
        Returns:
            (train_articles, train_summaries, test_articles, test_summaries)
        """
        self.logger.info(f"Loading dataset: {dataset_name} from 'train' and 'validation' splits")
        
        # Warn about deprecated max_samples parameter
        if max_samples is not None:
            self.logger.warning(
                "The 'max_samples' parameter is deprecated and will be ignored. "
                "All samples from the 'train' split will be loaded. "
                "Use dataset filtering after loading if you need to limit samples."
            )
        
        # Warn about deprecated train_test_split_ratio parameter
        if train_test_split_ratio is not None:
            self.logger.warning(
                "The 'train_test_split_ratio' parameter is deprecated and will be ignored. "
                "Test data comes from the 'validation' split of the dataset. "
                "Use 'test_max_samples' to limit validation/test samples if needed."
            )
        
        try:
            # Load full dataset (similar to toy_example.py)
            raw_datasets = load_dataset(dataset_name)
            
            # Load train split - deprecated max_samples is ignored, load all samples
            train_dataset = raw_datasets["train"]
            
            train_articles = [item["document"] for item in train_dataset]
            train_summaries = [item["summary"] for item in train_dataset]
            
            # Load validation split for test data
            test_dataset = raw_datasets["validation"]
            if test_max_samples is not None and test_max_samples > 0:
                self.logger.info(f"Selecting {test_max_samples} samples from validation split")
                test_dataset = test_dataset.select(range(test_max_samples))
            
            test_articles = [item["document"] for item in test_dataset]
            test_summaries = [item["summary"] for item in test_dataset]
            
            self.logger.info(f"Loaded {len(train_articles)} train examples and {len(test_articles)} test examples")
            self.logger.debug(f"Sample train article length: {len(train_articles[0]) if train_articles else 0} chars")
            self.logger.debug(f"Sample train summary length: {len(train_summaries[0]) if train_summaries else 0} chars")
            self.logger.debug(f"Sample test article length: {len(test_articles[0]) if test_articles else 0} chars")
            self.logger.debug(f"Sample test summary length: {len(test_summaries[0]) if test_summaries else 0} chars")
            
            return train_articles, train_summaries, test_articles, test_summaries
        except Exception as e:
            self.logger.exception(f"Failed to load dataset: {e}")
            raise
    
    def select_initial_samples(
        self, 
        articles: List[str], 
        summaries: List[str],
        selection_method: str = "random"
    ) -> Tuple[List[str], List[str]]:
        """
        Step 1: Select 12.5% of data as initial training set
        selection_method: 'random', 'first', or 'diverse' (can be extended)
        """
        total_size = len(articles)
        sample_size = int(total_size * self.base_data_ratio)
        
        if selection_method == "random":
            indices = random.sample(range(total_size), sample_size)
        elif selection_method == "first":
            indices = list(range(sample_size))
        elif selection_method == "diverse":
            # Simple diversity: take evenly spaced samples
            indices = [int(i * total_size / sample_size) for i in range(sample_size)]
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        selected_articles = [articles[i] for i in indices]
        selected_summaries = [summaries[i] for i in indices]
        
        self.logger.info(f"Selected {len(selected_articles)} samples ({self.base_data_ratio*100}%) using method: {selection_method}")
        self.logger.debug(f"Selection indices range: {min(indices)} to {max(indices)}")
        return selected_articles, selected_summaries
    
    def generate_synthetic_data(
        self,
        num_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        prompt_max_length: int,
        ground_truth_articles: List[str],
        generation_batch_size: int = 4
    ) -> List[Dict[str, str]]:
        """
        Step 2: Generate synthetic summaries for all articles in the full training dataset
        
        Args:
            num_samples: Deprecated - now generates summaries for all articles in ground_truth_articles
            ground_truth_articles: List of all ground truth articles from training dataset to generate summaries for
            generation_batch_size: Batch size for parallel generation (speeds up processing)
        
        Returns list of {article, summary} pairs for all articles in the training dataset
        """
        if not ground_truth_articles:
            self.logger.warning("No ground truth articles provided for synthetic data generation")
            return []
        
        # Generate summaries for ALL articles in the full training dataset
        articles_to_process = ground_truth_articles
        total_articles = len(articles_to_process)
        self.logger.info(f"Generating summaries for all {total_articles} articles in the training dataset")
        
        prompts = [f"Article: {article}\n\nSummary:" for article in articles_to_process]
        
        self.logger.info("Starting generation process...")
        generated_texts = self._generate_prompts(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            prompt_max_length=prompt_max_length,
            temperature=temperature,
            top_p=top_p,
            generation_batch_size=generation_batch_size,
            num_generations=self.num_generations,
            description="Generating synthetic summaries for all training articles"
        )
        
        self.logger.info("Processing generated summaries...")
        synthetic_data: List[Dict[str, str]] = []
        for article, prompt, generations in tqdm(
            zip(articles_to_process, prompts, generated_texts),
            total=total_articles,
            desc="Processing generated summaries",
            unit="article"
        ):
            if not generations:
                continue
            
            if self.num_generations == 1:
                full_text = generations[0]
            else:
                full_text = self.filter_best_generation(generations)
            
            summary = full_text.replace(prompt, "").strip()
            
            if summary:
                synthetic_data.append({
                    "article": article,
                    "summary": summary
                })
        
        self.logger.info(f"Generated {len(synthetic_data)} valid synthetic samples out of {total_articles} articles")
        if len(synthetic_data) < total_articles:
            self.logger.warning(f"Only {len(synthetic_data)}/{total_articles} samples were valid. Some generations may have failed parsing.")
        
        self.model.train()
        return synthetic_data
    
    def calculate_rouge1(self, reference: str, candidate: str) -> float:
        """
        Calculate ROUGE-1 F1 score between reference and candidate summaries
        """
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return scores['rouge1'].fmeasure
    
    def evaluate_on_test_set(
        self,
        test_articles: List[str],
        test_summaries: List[str],
        max_new_tokens: int,
        prompt_max_length: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        generation_batch_size: int = 8
    ) -> float:
        """
        Evaluate the model on test set by generating summaries and calculating average ROUGE-1 score
        
        Args:
            test_articles: List of test article texts
            test_summaries: List of ground truth summaries
            max_new_tokens: Maximum tokens to generate
            prompt_max_length: Maximum prompt length
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Average ROUGE-1 F1 score
        """
        if not test_articles or not test_summaries:
            self.logger.warning("Test set is empty, cannot evaluate")
            return 0.0
        
        if len(test_articles) != len(test_summaries):
            self.logger.warning(f"Mismatch: {len(test_articles)} articles but {len(test_summaries)} summaries")
            min_len = min(len(test_articles), len(test_summaries))
            test_articles = test_articles[:min_len]
            test_summaries = test_summaries[:min_len]
        
        prompts = [f"Article: {article}\n\nSummary:" for article in test_articles]
        # Use greedy decoding (temperature=0) for evaluation
        generated_texts = self._generate_prompts(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            prompt_max_length=prompt_max_length,
            temperature=0.0,  # Greedy decoding
            top_p=1.0,  # Not used with greedy decoding but set for consistency
            generation_batch_size=generation_batch_size,
            num_generations=1,
            description="Evaluating summaries (greedy decoding)"
        )
        
        generated_summaries: List[str] = []
        rouge_scores: List[float] = []
        
        for prompt, reference, generations in zip(prompts, test_summaries, generated_texts):
            summary_full = generations[0] if generations else ""
            summary = summary_full.replace(prompt, "").strip()
            generated_summaries.append(summary)
            rouge_scores.append(self.calculate_rouge1(reference, summary))
        
        # Calculate average ROUGE-1 score
        avg_rouge1 = np.mean(rouge_scores) if rouge_scores else 0.0
        
        self.logger.info(f"Average ROUGE-1 score: {avg_rouge1:.4f}")
        self.logger.debug(f"ROUGE-1 score range: {min(rouge_scores):.4f} - {max(rouge_scores):.4f}")
        
        self.model.train()
        return avg_rouge1
    
    def filter_best_generation(self, generated_texts: List[str]) -> str:
        """
        Step 3: Filter by selecting the best from multiple generations
        Uses simple heuristics: length, structure, coherence
        Can be extended with more sophisticated metrics
        """
        if len(generated_texts) == 1:
            return generated_texts[0]
        
        # Score each generation
        scores = []
        for text in generated_texts:
            score = 0
            
            # Prefer reasonable length (not too short, not too long)
            length = len(text.split())
            if 50 <= length <= 500:
                score += 10
            elif 20 <= length < 50 or 500 < length <= 1000:
                score += 5
            
            # Prefer text that contains both "Article:" and "Summary:" markers
            if "Article:" in text and "Summary:" in text:
                score += 20
            
            # Prefer text with proper sentence structure (has periods)
            if text.count('.') >= 2:
                score += 10
            
            # Prefer text without excessive repetition
            words = text.split()
            unique_ratio = len(set(words)) / len(words) if words else 0
            if unique_ratio > 0.5:
                score += 10
            
            scores.append(score)
        
        # Return the highest scoring generation
        best_idx = np.argmax(scores)
        return generated_texts[best_idx]
    
    def parse_generated_text(self, text: str) -> Optional[Dict[str, str]]:
        """Parse generated text to extract article and summary"""
        # Try to extract article and summary from generated text
        # Format: "Article: ... Summary: ..."
        
        if "Article:" in text and "Summary:" in text:
            parts = text.split("Summary:")
            if len(parts) == 2:
                article_part = parts[0].replace("Article:", "").strip()
                summary_part = parts[1].strip()
                
                # Clean up article (remove prompt if present)
                if "Generate a news article" in article_part:
                    article_part = article_part.split("Article:")[-1].strip()
                
                if article_part and summary_part:
                    return {
                        "article": article_part,
                        "summary": summary_part
                    }
        
        # Fallback: try to split by newlines or other patterns
        lines = text.split("\n")
        if len(lines) >= 2:
            # Assume first part is article, second part is summary
            article = lines[0].replace("Article:", "").strip()
            summary = "\n".join(lines[1:]).replace("Summary:", "").strip()
            
            if article and summary:
                return {"article": article, "summary": summary}
        
        return None
    
    def train_model(
        self,
        articles: List[str],
        summaries: List[str],
        epoch: int,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        optimizer_type: str,
        lr_scheduler: str,
        warmup_ratio: float,
        save_checkpoints: bool,
        checkpoint_frequency: int,
        save_optimizer: bool
    ):
        """Train the model on given articles and summaries"""
        self.logger.info("=" * 60)
        self.logger.info(f"Training model (Round {epoch}) on {len(articles)} samples...")
        self.logger.info(
            f"Training parameters: batch_size={batch_size}, num_epochs={num_epochs}, "
            f"lr={learning_rate}, optimizer={optimizer_type}, scheduler={lr_scheduler}, warmup_ratio={warmup_ratio}"
        )
        
        # Create dataset
        self.logger.debug(f"Creating dataset with {len(articles)} samples")
        dataset = NewsSummarizationDataset(articles, summaries, self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device.startswith('cuda') else False,
            persistent_workers=True if self.num_workers > 0 else False
        )
        self.logger.debug(f"DataLoader created with {len(dataloader)} batches (num_workers={self.num_workers}, pin_memory={self.device.startswith('cuda')})")
        
        # Setup optimizer
        self.logger.debug(f"Setting up {optimizer_type} optimizer with lr={learning_rate}")
        parameters = self.model.parameters()
        if optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        elif optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=learning_rate)
        else:
            self.logger.error(f"Unknown optimizer type: {optimizer_type}")
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        scheduler = None
        total_training_steps = num_epochs * len(dataloader)
        scheduler_name = lr_scheduler.lower()
        if scheduler_name != "none" and total_training_steps > 0:
            warmup_steps = int(total_training_steps * warmup_ratio)
            scheduler = get_scheduler(
                scheduler_name,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps
            )
            self.logger.info(
                f"Initialized {scheduler_name} scheduler "
                f"(warmup_steps={warmup_steps}, total_steps={total_training_steps})"
            )
        elif scheduler_name != "none":
            self.logger.warning("Scheduler requested but no training steps available; skipping scheduler setup.")
        
        # Training loop
        self.model.train()
        total_loss = 0
        
        # Base checkpoint directory for this round
        base_checkpoint_dir = os.path.join(self.output_dir, f"round_{epoch}")
        Path(base_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        for epoch_idx in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{num_epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            self.logger.info(f"Epoch {epoch_idx+1}/{num_epochs} - Average loss: {avg_loss:.4f}")
            self.logger.debug(f"Epoch {epoch_idx+1} - Total batches: {num_batches}, Total loss: {epoch_loss:.4f}")
            
            # Save checkpoint during training if enabled
            if save_checkpoints and (epoch_idx + 1) % checkpoint_frequency == 0:
                self.logger.info(f"Saving checkpoint at epoch {epoch_idx+1}")
                checkpoint_dir = os.path.join(base_checkpoint_dir, f"epoch_{epoch_idx+1}")
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                
                # Save model and tokenizer (training uses single GPU, no unwrapping needed)
                self.model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)
                
                # Save optimizer state if requested
                if save_optimizer:
                    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
                    torch.save(optimizer.state_dict(), optimizer_path)
                
                # Save training metadata
                metadata = {
                    "round": epoch,
                    "epoch": epoch_idx + 1,
                    "total_epochs": num_epochs,
                    "avg_loss": avg_loss,
                    "learning_rate": learning_rate,
                    "optimizer_type": optimizer_type,
                    "lr_scheduler": lr_scheduler,
                    "warmup_ratio": warmup_ratio,
                    "batch_size": batch_size,
                    "num_samples": len(articles)
                }
                metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
                if save_optimizer:
                    self.logger.debug(f"Optimizer state saved to {optimizer_path}")
        
        # Save final model checkpoint
        self.logger.info("Saving final checkpoint...")
        final_checkpoint_dir = os.path.join(base_checkpoint_dir, "final")
        Path(final_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model and tokenizer (training uses single GPU, no unwrapping needed)
            self.model.save_pretrained(final_checkpoint_dir)
            self.tokenizer.save_pretrained(final_checkpoint_dir)
            self.logger.debug("Model and tokenizer saved")
        except Exception as e:
            self.logger.exception(f"Failed to save model: {e}")
            raise
        
        # Save optimizer state if requested
        if save_optimizer:
            optimizer_path = os.path.join(final_checkpoint_dir, "optimizer.pt")
            try:
                torch.save(optimizer.state_dict(), optimizer_path)
                self.logger.debug(f"Optimizer state saved to {optimizer_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save optimizer state: {e}")
                self.logger.exception("Optimizer save error details:")
        
        # Save final training metadata
        final_avg_loss = total_loss / (num_epochs * num_batches) if num_batches > 0 else 0
        final_metadata = {
            "round": epoch,
            "total_epochs": num_epochs,
            "final_avg_loss": final_avg_loss,
            "learning_rate": learning_rate,
            "optimizer_type": optimizer_type,
            "lr_scheduler": lr_scheduler,
            "warmup_ratio": warmup_ratio,
            "batch_size": batch_size,
            "num_samples": len(articles)
        }
        metadata_path = os.path.join(final_checkpoint_dir, "training_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(final_metadata, f, indent=2)
        
        self.logger.info(f"Final model saved to {final_checkpoint_dir}")
        self.logger.info(f"Final average loss: {final_avg_loss:.4f}")
        return final_avg_loss
    
    def run_iterative_training(
        self,
        dataset_name: str,
        dataset_split: str,
        num_iterations: int,
        synthetic_samples_per_iteration: int,
        initial_selection_method: str,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        optimizer_type: str,
        lr_scheduler: str,
        warmup_ratio: float,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        prompt_max_length: int,
        save_checkpoints: bool,
        checkpoint_frequency: int,
        save_optimizer: bool,
        max_samples: Optional[int] = None,
        train_test_split_ratio: Optional[float] = None,
        generation_batch_size: int = 4
    ):
        """
        Main pipeline: Run iterative retraining
        Step 1: Initial training on 12.5% of data
        Step 2-4: Iteratively generate synthetic data, filter, and retrain
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Iterative Retraining Pipeline")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - Dataset: {dataset_name} ({dataset_split})")
        self.logger.info(f"  - Max samples: {max_samples}")
        self.logger.info(f"  - Train/test split ratio: {train_test_split_ratio}")
        self.logger.info(f"  - Iterations: {num_iterations}")
        self.logger.info(f"  - Synthetic samples per iteration: {synthetic_samples_per_iteration}")
        self.logger.info(f"  - Initial selection method: {initial_selection_method}")
        self.logger.info(
            "Training recipe: Hugging Face Transformers v2 (Apache 2.0) defaults "
            "(pretrained Llama-2 init, lr=5e-5, scheduler=cosine, epochs=1, total batch size=32, block size=1024)."
        )
        
        # Load dataset from train and validation splits
        train_articles, train_summaries, test_articles, test_summaries = self.load_dataset(
            dataset_name, 
            dataset_split, 
            max_samples=max_samples,
            train_test_split_ratio=train_test_split_ratio
        )
        
        # Use train split for training
        articles = train_articles
        summaries = train_summaries
        # Store test set (from validation split) for evaluation
        self.test_articles = test_articles
        self.test_summaries = test_summaries
        self.logger.info(f"Using train set with {len(articles)} samples for training")
        self.logger.info(f"Test set (from validation split) with {len(test_articles)} samples stored for evaluation")
        
        # Step 1: Select initial 12.5% of data
        initial_articles, initial_summaries = self.select_initial_samples(
            articles, summaries, selection_method=initial_selection_method
        )
        
        # Store all articles and summaries for potential use in synthetic generation
        all_available_articles = articles.copy()
        all_available_summaries = summaries.copy()
        
        # Step 1: Train on initial data
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ROUND 0: Initial Training")
        self.logger.info("=" * 60)
        self.train_model(
            initial_articles,
            initial_summaries,
            epoch=0,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            lr_scheduler=lr_scheduler,
            warmup_ratio=warmup_ratio,
            save_checkpoints=save_checkpoints,
            checkpoint_frequency=checkpoint_frequency,
            save_optimizer=save_optimizer
        )
        
        # Evaluate on test set after initial training
        if hasattr(self, 'test_articles') and hasattr(self, 'test_summaries') and self.test_articles:
            self.logger.info("\n" + "-" * 60)
            self.logger.info("Evaluating model on test set after ROUND 0")
            self.logger.info("-" * 60)
            avg_rouge1 = self.evaluate_on_test_set(
                test_articles=self.test_articles,
                test_summaries=self.test_summaries,
                max_new_tokens=max_new_tokens,
                prompt_max_length=prompt_max_length,
                temperature=0.7,
                top_p=0.9,
                generation_batch_size=generation_batch_size
            )
            self.logger.info(f"ROUND 0 - Average ROUGE-1: {avg_rouge1:.4f}")
        
        # Keep track of all training data
        all_articles = initial_articles.copy()
        all_summaries = initial_summaries.copy()
        
        # Steps 2-4: Iterative retraining
        for iteration in range(1, num_iterations + 1):
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f"ROUND {iteration}: Iterative Retraining")
            self.logger.info("=" * 60)
            
            # Step 2: Generate synthetic data using ground truth articles
            synthetic_data = self.generate_synthetic_data(
                num_samples=synthetic_samples_per_iteration,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                prompt_max_length=prompt_max_length,
                ground_truth_articles=all_available_articles,
                generation_batch_size=generation_batch_size
            )
            
            if not synthetic_data:
                self.logger.warning("No valid synthetic data generated. Skipping iteration.")
                continue
            
            # Calculate ROUGE-1 scores for synthetic summaries against ground truth summaries
            self.logger.info("Calculating ROUGE-1 scores for synthetic summaries...")
            
            # Create a mapping from article to ground truth summary for quick lookup
            article_to_ground_truth_summary = dict(zip(all_available_articles, all_available_summaries))
            
            # Score each synthetic summary against its ground truth summary
            synthetic_data_with_scores = []
            for item in tqdm(synthetic_data, desc="Calculating ROUGE-1 scores", unit="sample"):
                article = item["article"]
                synthetic_summary = item["summary"]
                
                # Find corresponding ground truth summary
                ground_truth_summary = article_to_ground_truth_summary.get(article)
                if ground_truth_summary is None:
                    self.logger.warning(f"Ground truth summary not found for article, skipping...")
                    continue
                
                # Calculate ROUGE-1 score
                rouge1_score = self.calculate_rouge1(ground_truth_summary, synthetic_summary)
                synthetic_data_with_scores.append({
                    "article": article,
                    "summary": synthetic_summary,
                    "rouge1_score": rouge1_score
                })
            
            if not synthetic_data_with_scores:
                self.logger.warning("No synthetic data with valid ground truth matches. Skipping iteration.")
                continue
            
            # Sort by ROUGE-1 score (descending) and select top base_data_ratio percentage
            synthetic_data_with_scores.sort(key=lambda x: x["rouge1_score"], reverse=True)
            num_to_select = int(len(synthetic_data_with_scores) * self.base_data_ratio)
            selected_synthetic_data = synthetic_data_with_scores[:num_to_select]
            
            self.logger.info(
                f"Selected top {len(selected_synthetic_data)} synthetic samples "
                f"({self.base_data_ratio*100}%) with highest ROUGE-1 scores "
                f"(range: {selected_synthetic_data[-1]['rouge1_score']:.4f} - {selected_synthetic_data[0]['rouge1_score']:.4f})"
            )
            
            # Extract articles and summaries from selected synthetic data
            selected_synthetic_articles = [item["article"] for item in selected_synthetic_data]
            selected_synthetic_summaries = [item["summary"] for item in selected_synthetic_data]
            
            # Add only selected synthetic data to training set
            all_articles = selected_synthetic_articles.copy()
            all_summaries = selected_synthetic_summaries.copy()
            
            self.logger.info(f"Total training samples: {len(all_articles)}")
            self.logger.info(f"  - Initial: {len(initial_articles)}")
            self.logger.info(f"  - Synthetic (this round, selected): {len(selected_synthetic_articles)}/{len(synthetic_data)}")
            self.logger.info(f"  - Total synthetic: {len(all_articles) - len(initial_articles)}")
            self.logger.debug(f"Training data growth: {len(all_articles) - len(initial_articles)} new samples")
            
            # Step 4: Retrain on combined data
            self.train_model(
                all_articles,
                all_summaries,
                epoch=iteration,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                optimizer_type=optimizer_type,
                lr_scheduler=lr_scheduler,
                warmup_ratio=warmup_ratio,
                save_checkpoints=save_checkpoints,
                checkpoint_frequency=checkpoint_frequency,
                save_optimizer=save_optimizer
            )
            
            # Evaluate on test set after this round
            if hasattr(self, 'test_articles') and hasattr(self, 'test_summaries') and self.test_articles:
                self.logger.info("\n" + "-" * 60)
                self.logger.info(f"Evaluating model on test set after ROUND {iteration}")
                self.logger.info("-" * 60)
                avg_rouge1 = self.evaluate_on_test_set(
                    test_articles=self.test_articles,
                    test_summaries=self.test_summaries,
                    max_new_tokens=max_new_tokens,
                    prompt_max_length=prompt_max_length,
                    temperature=0.7,
                    top_p=0.9,
                    generation_batch_size=generation_batch_size
                )
                self.logger.info(f"ROUND {iteration} - Average ROUGE-1: {avg_rouge1:.4f}")
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Iterative Training Complete!")
        self.logger.info("=" * 60)
        final_model_path = os.path.join(self.output_dir, f'round_{num_iterations}', 'final')
        self.logger.info(f"Final model saved to: {final_model_path}")
        self.logger.info(f"Total rounds completed: {num_iterations}")
        self.logger.info(f"Total training samples used: {len(all_articles)}")



def main():
    """Main function"""
    args = parse_args()
    
    # Setup logger first
    setup_logger(args.output_dir, args.log_level)
    logger.info("=" * 60)
    logger.info("Iterative Summarization Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Command-line arguments: {vars(args)}")
    
    # Initialize trainer
    try:
        trainer = IterativeSummarizationTrainer(
            model_name=args.model_name,
            base_data_ratio=args.base_data_ratio,
            num_generations=args.num_generations,
            max_length=args.max_length,
            device=args.device,
            output_dir=args.output_dir,
            seed=args.seed,
            use_multi_gpu=args.use_multi_gpu,
            num_workers=args.num_workers
        )
    except Exception as e:
        logger.exception(f"Failed to initialize trainer: {e}")
        raise
    
    # Run iterative training
    try:
        trainer.run_iterative_training(
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            num_iterations=args.num_iterations,
            synthetic_samples_per_iteration=args.synthetic_samples_per_iteration,
            initial_selection_method=args.initial_selection_method,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer_type,
            lr_scheduler=args.lr_scheduler,
            warmup_ratio=args.warmup_ratio,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            prompt_max_length=args.prompt_max_length,
            save_checkpoints=args.save_checkpoints,
            checkpoint_frequency=args.checkpoint_frequency,
            save_optimizer=args.save_optimizer,
            max_samples=args.max_samples,
            train_test_split_ratio=args.train_test_split_ratio,
            generation_batch_size=args.generation_batch_size
        )
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    finally:
        logger.info("Training session ended")


if __name__ == "__main__":
    main()

