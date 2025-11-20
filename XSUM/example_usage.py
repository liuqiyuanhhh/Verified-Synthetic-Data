"""
Example usage script for iterative news summarization training

This file shows example command-line invocations.
Run the trainer.py script with command-line arguments instead.
"""

# Example 1: Basic usage with all required parameters
"""
python trainer.py \
    --model_name "path/to/smolm2-135MB-Instruct" \
    --max_length 512 \
    --dataset_name "xsum" \
    --dataset_split "train" \
    --base_data_ratio 0.125 \
    --num_generations 10 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 512 \
    --num_iterations 5 \
    --synthetic_samples_per_iteration 1000 \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --output_dir "./iterative_models"
"""

# Example 2: Using ground truth articles (only generate summaries)
"""
python trainer.py \
    --model_name "path/to/smolm2-135MB-Instruct" \
    --max_length 512 \
    --dataset_name "xsum" \
    --dataset_split "train" \
    --base_data_ratio 0.125 \
    --num_generations 10 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 512 \
    --num_iterations 5 \
    --synthetic_samples_per_iteration 1000 \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --output_dir "./iterative_models" \
    --use_ground_truth_articles
"""

# Example 3: Custom prompt template and optimizer
"""
python trainer.py \
    --model_name "path/to/smolm2-135MB-Instruct" \
    --max_length 512 \
    --dataset_name "xsum" \
    --dataset_split "train" \
    --base_data_ratio 0.125 \
    --initial_selection_method "diverse" \
    --num_generations 10 \
    --prompt_template "Write a news article with a summary:\n\nArticle:" \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_new_tokens 512 \
    --num_iterations 5 \
    --synthetic_samples_per_iteration 1000 \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --optimizer_type "adam" \
    --output_dir "./iterative_models" \
    --seed 123
"""

# Example 4: With checkpoint saving (save every epoch and optimizer state)
"""
python trainer.py \
    --model_name "path/to/smolm2-135MB-Instruct" \
    --max_length 512 \
    --dataset_name "xsum" \
    --base_data_ratio 0.125 \
    --num_generations 10 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 512 \
    --num_iterations 5 \
    --synthetic_samples_per_iteration 1000 \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --output_dir "./iterative_models" \
    --save_checkpoints \
    --checkpoint_frequency 1 \
    --save_optimizer
"""

# Example 5: Save checkpoints every 2 epochs
"""
python trainer.py \
    --model_name "path/to/smolm2-135MB-Instruct" \
    --max_length 512 \
    --dataset_name "xsum" \
    --base_data_ratio 0.125 \
    --num_generations 10 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 512 \
    --num_iterations 5 \
    --synthetic_samples_per_iteration 1000 \
    --batch_size 4 \
    --num_epochs 6 \
    --learning_rate 5e-5 \
    --output_dir "./iterative_models" \
    --save_checkpoints \
    --checkpoint_frequency 2 \
    --save_optimizer
"""

# Example 6: With detailed logging (DEBUG level)
"""
python trainer.py \
    --model_name "path/to/smolm2-135MB-Instruct" \
    --max_length 512 \
    --dataset_name "xsum" \
    --base_data_ratio 0.125 \
    --num_generations 10 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 512 \
    --num_iterations 5 \
    --synthetic_samples_per_iteration 1000 \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --output_dir "./iterative_models" \
    --save_checkpoints \
    --save_optimizer \
    --log_level DEBUG
"""

print("See the comments above for example command-line invocations.")
print("Run: python trainer.py --help to see all available options.")
print("\nLogging (using loguru):")
print("  - Install loguru: pip install loguru")
print("  - Logs are automatically saved to: output_dir/training_YYYYMMDD_HHMMSS.log")
print("  - Log files rotate at 10MB and are compressed")
print("  - Logs are retained for 7 days")
print("  - Use --log_level DEBUG for detailed debug information")
print("  - Use --log_level INFO for standard information (default)")
print("  - Use --log_level WARNING for only warnings and errors")

