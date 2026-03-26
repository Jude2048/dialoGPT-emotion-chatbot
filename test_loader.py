import sys
import os
import torch

print(torch.cuda.is_available())
sys.path.append(os.path.abspath("."))

from src.data_loader import load_split_from_local_files

train_data = load_split_from_local_files(
    "sample_data/train",
    "train"
)

print(len(train_data))
print(train_data[0])