import re
import copy

# Emotion map (keep consistent everywhere)
emotion_map = {
    0: 'no emotion',
    1: 'anger',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',
    5: 'sadness',
    6: 'surprise'
}

def clean_text(text):
    """
    Clean a single utterance
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.?!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_dataset(data):
    """
    Full preprocessing pipeline:
    - clean text
    - remove emotionless dialogues
    - remove empty utterances
    """

    cleaned_data = copy.deepcopy(data)

    # Clean text
    for dialogue in cleaned_data:
        dialogue['dialog'] = [clean_text(turn) for turn in dialogue['dialog']]

    # Remove emotionless dialogues
    cleaned_data = [
        d for d in cleaned_data
        if not all(e == 0 for e in d['emotion'])
    ]

    # Remove empty utterances
    final_data = []

    for dialogue in cleaned_data:

        new_dialog = []
        new_emotion = []
        new_act = []

        for i in range(len(dialogue['dialog'])):
            if dialogue['dialog'][i]:
                new_dialog.append(dialogue['dialog'][i])
                new_emotion.append(dialogue['emotion'][i])
                new_act.append(dialogue['act'][i])

        if new_dialog:
            final_data.append({
                'dialog': new_dialog,
                'emotion': new_emotion,
                'act': new_act
            })

    return final_data


def format_dialogue_for_model(dialogue, eos_token):
    """
    Convert dialogue to DialoGPT format
    """
    formatted_turns = []

    for i, turn_text in enumerate(dialogue['dialog']):
        emotion_idx = dialogue['emotion'][i]
        emotion_name = emotion_map[emotion_idx]

        formatted_turn = f"<|{emotion_name}|> {turn_text} {eos_token}"
        formatted_turns.append(formatted_turn)

    return "".join(formatted_turns)


def create_training_texts(cleaned_data, eos_token):
    """
    Create list of training strings
    """
    return [format_dialogue_for_model(d, eos_token) for d in cleaned_data]