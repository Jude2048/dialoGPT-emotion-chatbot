from src.data_loader import load_split_from_local_files

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

model_path = "./model"  # or your checkpoint folder

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded")

emotion_map = {
    0: 'no emotion',
    1: 'anger',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',
    5: 'sadness',
    6: 'surprise'
}

path_to_test = "data/Test"  # change if needed

test_data = load_split_from_local_files(path_to_test, "test")

print("Test samples:", len(test_data))

def generate_response(prompt_text, emotion_token):

    input_str = f"{emotion_token} {prompt_text} {tokenizer.eos_token}"

    inputs = tokenizer(input_str, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,

        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    response = decoded.replace(prompt_text, "").strip()

    return response

results = []

num_samples = 20  # increase later

for dialogue in test_data[:num_samples]:

    if len(dialogue["dialog"]) < 2:
        continue

    prompt = dialogue["dialog"][0]
    true_response = dialogue["dialog"][1]

    target_emotion = dialogue["emotion"][1]
    emotion_token = f"<|{emotion_map[target_emotion]}|>"

    generated = generate_response(prompt, emotion_token)

    results.append({
        "Prompt": prompt,
        "Target Emotion": emotion_map[target_emotion],
        "Generated Response": generated,
        "Actual Response": true_response
    })

df = pd.DataFrame(results)

df.head()

df.to_csv("test_results.csv", index=False)

print("Saved results")