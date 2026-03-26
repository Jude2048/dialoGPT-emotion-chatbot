import sys
import os

sys.path.append(os.path.abspath("."))

from src.data_loader import load_split_from_local_files
from src.preprocessing import preprocess_dataset, create_training_texts
from src.training import oversample_data

from transformers import AutoTokenizer

# -------------------------
# STEP 1: LOAD DATA
# -------------------------
print("\n[1] Loading data...")

train_data = load_split_from_local_files(
    "sample_data/Train",
    "train"
)

print("Loaded:", len(train_data))


# -------------------------
# STEP 2: PREPROCESS
# -------------------------
print("\n[2] Preprocessing...")

cleaned = preprocess_dataset(train_data)

print("After cleaning:", len(cleaned))


# -------------------------
# STEP 3: TOKENIZER
# -------------------------
print("\n[3] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
tokenizer.pad_token = tokenizer.eos_token


# -------------------------
# STEP 4: FORMAT DATA
# -------------------------
print("\n[4] Formatting data...")

training_texts = create_training_texts(cleaned, tokenizer.eos_token)

print("Sample:\n", training_texts[0][:200])


# -------------------------
# STEP 5: OVERSAMPLING
# -------------------------
print("\n[5] Oversampling...")

oversampled = oversample_data(training_texts)

print("Final dataset size:", len(oversampled))


# -------------------------
# STEP 6: SAVE FILE
# -------------------------
print("\n[6] Saving dataset...")

with open("train_final.txt", "w", encoding="utf-8") as f:
    for text in oversampled:
        f.write(text + "\n")

print("Saved: train_final.txt")


print("\n✅ PIPELINE COMPLETE")