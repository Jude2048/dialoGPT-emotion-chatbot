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