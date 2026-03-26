import sys
import os

sys.path.append(os.path.abspath("."))

from src.data_loader import load_split_from_local_files

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pandas as pd
from tqdm import tqdm

import evaluate
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# STEP 1: LOAD MODEL
# -------------------------
print("[1] Loading trained model...")

model_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Using device:", device)


# -------------------------
# STEP 2: LOAD TEST DATA
# -------------------------
print("[2] Loading test data...")

test_data = load_split_from_local_files(
    "sample_data/Test",
    "test"
)


# -------------------------
# STEP 3: EMOTION MAP
# -------------------------
emotion_map = {
    0: 'no emotion',
    1: 'anger',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',
    5: 'sadness',
    6: 'surprise'
}


# -------------------------
# STEP 4: GENERATE RESPONSES
# -------------------------
print("[3] Generating responses...")

predictions = []
references = []
actual_emotions = []

for dialogue in tqdm(test_data[:500]):   # limit for speed

    if len(dialogue["dialog"]) < 2:
        continue

    prompt = dialogue["dialog"][0]
    true_response = dialogue["dialog"][1]

    target_emotion = dialogue["emotion"][1]
    emotion_token = f"<|{emotion_map[target_emotion]}|>"

    input_str = f"{emotion_token} {prompt} {tokenizer.eos_token}"

    inputs = tokenizer(input_str, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.replace(prompt, "").strip()

    predictions.append(response)
    references.append(true_response)
    actual_emotions.append(target_emotion)

print("Generated:", len(predictions))


# -------------------------
# STEP 5: ROUGE
# -------------------------
print("\n[4] ROUGE...")

rouge = evaluate.load("rouge")
rouge_results = rouge.compute(predictions=predictions, references=references)

print(rouge_results)


# -------------------------
# STEP 6: BERTScore
# -------------------------
print("\n[5] BERTScore...")

bertscore = evaluate.load("bertscore")

bert_results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en"
)

avg_f1 = sum(bert_results["f1"]) / len(bert_results["f1"])

print("BERTScore F1:", round(avg_f1, 4))


# -------------------------
# STEP 7: EMOTION CLASSIFIER
# -------------------------
print("\n[6] Emotion classification...")

emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=1,
    device=0 if torch.cuda.is_available() else -1
)

classifier_to_our_map = {
    'sadness': 5,
    'anger': 1,
    'surprise': 6,
    'fear': 3,
    'joy': 4,
    'neutral': 0
}


predicted_emotions = []

for text in tqdm(predictions):

    if not text:
        predicted_emotions.append(0)
        continue

    pred_label = emotion_classifier(text)[0][0]["label"]
    pred_idx = classifier_to_our_map.get(pred_label, 0)

    predicted_emotions.append(pred_idx)


# -------------------------
# STEP 8: CLASSIFICATION REPORT
# -------------------------
print("\n[7] Emotion Metrics...")

report = classification_report(
    actual_emotions,
    predicted_emotions,
    target_names=list(emotion_map.values())
)

print(report)


# -------------------------
# STEP 9: VISUALIZATION
# -------------------------
print("\n[8] Visualization...")

report_dict = classification_report(
    actual_emotions,
    predicted_emotions,
    target_names=list(emotion_map.values()),
    output_dict=True
)

df = pd.DataFrame(report_dict).T.iloc[:-3]

df[['precision','recall','f1-score']].plot(kind='bar', figsize=(10,6))
plt.title("Precision / Recall / F1 per Emotion")
plt.ylim(0,1)
plt.tight_layout()
plt.show()


# -------------------------
# STEP 10: CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(actual_emotions, predicted_emotions)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("\n EVALUATION COMPLETE")