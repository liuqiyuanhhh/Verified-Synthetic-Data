# smol2_135M_xsum_fullft.py

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import torch
import math

# <<< CHANGE THIS TO YOUR REAL CHECKPOINT >>>
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"  # or SmolLM2-135M
BLOCK_SIZE = 1024

# 1) Load XSum
raw_datasets = load_dataset("xsum")  # you already fixed datasets version

train_ds = raw_datasets["train"]
valid_ds = raw_datasets["validation"]

# 2) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3) Build instruction-style text
PROMPT_SUFFIX = "Write a one-sentence summary of the article:\n"

def build_text(example):
    article = example["document"]
    summary = example["summary"]

    prompt = (
        "You are an assistant that writes concise English news summaries.\n\n"
        "Article:\n"
        f"{article}\n\n"
        f"{PROMPT_SUFFIX}"
    )
    full_text = prompt + summary  # teacher forcing on gold summary
    return {"text": full_text}

train_ds = train_ds.map(build_text, remove_columns=train_ds.column_names)
valid_ds = valid_ds.map(build_text, remove_columns=valid_ds.column_names)

# 4A) Simple LM-style tokenization: loss on prompt + summary (easiest)
def tokenize_fn(batch):
    enc = tokenizer(
        batch["text"],
        max_length=BLOCK_SIZE,
        truncation=True,
        padding="max_length",
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
valid_tok = valid_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# If you prefer loss only on summary tokens, see 4B below and swap.

train_tok.set_format(type="torch")
valid_tok.set_format(type="torch")

# 5) Load full model (no LoRA, no quantization)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.config.use_cache = False  # safer for training

# 6) Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Effective batch size ≈ 32 = per_device * grad_accum * n_gpus
per_device_batch = 4
grad_accum = 8  # if 1 GPU → 4 * 8 = 32

training_args = TrainingArguments(
    output_dir="./smol2_135M_xsum_fullft",
    num_train_epochs=1,                         # as in the paper
    per_device_train_batch_size=per_device_batch,
    per_device_eval_batch_size=per_device_batch,
    gradient_accumulation_steps=grad_accum,
    learning_rate=5e-5,                         # as in the paper for generator
    logging_steps=50,
    save_steps=1000,
    fp16=torch.cuda.is_available(),
    # If your transformers is new enough, you can ADD:
    # lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=valid_tok,    # optional, can remove if you don’t need eval
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./smol2_135M_xsum_fullft")
tokenizer.save_pretrained("./smol2_135M_xsum_fullft")
