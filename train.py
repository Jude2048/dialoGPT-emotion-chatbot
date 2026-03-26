import sys
import os

sys.path.append(os.path.abspath("."))

print("STARTING TRAINING SCRIPT...\n")

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

import torch


print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
# -------------------------
# STEP 1: LOAD DATA
# -------------------------
print("[1] Loading dataset...")

dataset = load_dataset("text", data_files={"train": "train_final.txt"})

print("Dataset loaded")

# -------------------------
# STEP 2: TOKENIZER
# -------------------------
print("[2] Loading tokenizer...")

model_name = "microsoft/DialoGPT-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer ready")

# -------------------------
# STEP 3: TOKENIZATION
# -------------------------
print("[3] Tokenizing...")

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("Tokenization done")

# -------------------------
# STEP 4: GROUP TEXT
# -------------------------
print("[4] Grouping text...")

block_size = 128

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])

    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

print("Grouping done")

# -------------------------
# STEP 5: LOAD MODEL
# -------------------------
print("[5] Loading model...")

model = AutoModelForCausalLM.from_pretrained(model_name)

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Using device:", device)

# -------------------------
# STEP 6: TRAINING SETUP
# -------------------------
print("[6] Setting up training...")

training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=3, 
    per_device_train_batch_size=2,   # SAFE for 4GB GPU
    save_steps=10000, # Save every 10k steps
    logging_steps=200, # Log every 200 steps
    save_total_limit=2,
    gradient_accumulation_steps = 8, # Effective batch size = 2 * 8 = 16
    prediction_loss_only=True, # Only compute loss for efficiency
    fp16=torch.cuda.is_available(),  # auto use fp16 if GPU
    gradient_checkpointing=False,  # Save memory with gradient checkpointing
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("Trainer ready")

# -------------------------
# STEP 7: TRAIN
# -------------------------
print("[7] TRAINING STARTED...\n")

trainer.train()

print("\n TRAINING COMPLETE")

print("\n[8] Saving model...")

trainer.save_model("./model")
tokenizer.save_pretrained("./model")

print("Model saved to ./model")