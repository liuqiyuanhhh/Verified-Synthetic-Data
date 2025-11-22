from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import torch

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


# 1. Load XSum (toy subset)
raw_datasets = load_dataset("xsum")

# Take a tiny subset to test pipeline
train_dataset = raw_datasets["train"]
print(f"Train dataset size: {len(train_dataset)}")
val_dataset   = raw_datasets["validation"]
print(f"Validation dataset size: {len(val_dataset)}")

# # 2. Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # 3. Build prompt templates
# def build_prompt(example):
#     article = example["document"]
#     summary = example["summary"]

#     # Simple instruction-style format
#     prompt = (
#         "You are an assistant that writes concise English news summaries.\n\n"
#         "Article:\n"
#         f"{article}\n\n"
#         "Write a one-sentence summary of the article:\n"
#     )

#     # For causal LM fine-tuning, labels = shifted version of full input
#     # Easiest: input_ids = prompt + summary
#     full_text = prompt + summary
#     return {"text": full_text}

# train_dataset = train_dataset.map(build_prompt, remove_columns=train_dataset.column_names)
# val_dataset   = val_dataset.map(build_prompt,   remove_columns=val_dataset.column_names)

# # 4. Tokenize
# max_length = 512  # for toy example; you might use 1024–2048 in practice

# def tokenize_fn(examples):
#     return tokenizer(
#         examples["text"],
#         max_length=max_length,
#         truncation=True,
#         padding="max_length",
#     )

# tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
# tokenized_val   = val_dataset.map(tokenize_fn,   batched=True, remove_columns=["text"])

# # Transformers expects "labels" for LM; copy input_ids
# def add_labels(examples):
#     examples["labels"] = examples["input_ids"].copy()
#     return examples

# tokenized_train = tokenized_train.map(add_labels, batched=True)
# tokenized_val   = tokenized_val.map(add_labels,   batched=True)

# # 5. Load base model in 4-bit for QLoRA (still very heavy in practice!)
# dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     dtype=dtype,
#     device_map="auto",
# )

# # 6. Set up LoRA
# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()  # sanity check

# # 7. Trainer setup
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,
# )

# training_args = TrainingArguments(
#     output_dir="./smol2-xsum-toy",
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=8,
#     learning_rate=2e-4,
#     num_train_epochs=1.0,
#     # evaluation_strategy="steps",
#     # eval_steps=100,
#     save_steps=200,
#     logging_steps=50,
#     warmup_steps=50,
#     fp16=torch.cuda.is_available(),
#     # gradient_checkpointing=True,
#     gradient_checkpointing=False,
#     report_to="none",
# )

# trainer = Trainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_val,
#     data_collator=data_collator,
# )

# trainer.train()

# # 8. Save LoRA adapter
# trainer.save_model("./smol2-xsum-toy-lora")
# tokenizer.save_pretrained("./smol2-xsum-toy-lora")
