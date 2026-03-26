# Emotion-Aware Dialogue Generation using DialoGPT

## Overview
This project builds a conversational AI system that generates emotionally controlled responses using fine-tuned DialoGPT on the DailyDialog dataset.

## Features
- Emotion-controlled text generation
- Multi-turn dialogue processing
- NLP pipeline with preprocessing, training, and evaluation
- Evaluation using:
  - ROUGE
  - BERTScore
  - Emotion classification accuracy

## Project Structure

src/            # Core modules
main.py         # Data preprocessing pipeline
train.py        # Model training
evaluate.py     # Evaluation metrics

## Dataset

DailyDialog dataset with emotion labels.

## Dataset Setup

This project uses the DailyDialog dataset.

Due to size constraints, the dataset is not included in this repository.

### Steps to use:

1. Download DailyDialog dataset from:
   https://aclanthology.org/I17-1099/

2. Place files in the following structure:

sample_data/

    Train/
─ dialogues_train.txt
─ dialogues_emotion_train.txt
─ dialogues_act_train.txt

    Test/
─ dialogues_test.txt
─ dialogues_emotion_test.txt
─ dialogues_act_test.txt

    Validation/
─ dialogues_validation.txt
─ dialogues_emotion_validation.txt
─ dialogues_act_validation.txt

## Technologies
- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn

## How to Run

1. Prepare Data
    python main.py

2. Train Model
    python train.py

3. Evaluate
    python evaluate.py

## Author
Jude Silveira
