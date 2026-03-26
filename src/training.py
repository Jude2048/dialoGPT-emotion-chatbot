import re

# Emotion map
emotion_map = {
    0: 'no emotion',
    1: 'anger',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',
    5: 'sadness',
    6: 'surprise'
}

# Rare emotions to oversample
rare_emotion_indices = {1, 2, 3, 5, 6}

# Mapping
name_to_idx = {v: k for k, v in emotion_map.items()}


def extract_emotion_indices(dialogue_string):
    """
    Extract emotion indices from formatted dialogue
    """
    names = re.findall(r"<\|(.*?)\|>", dialogue_string)
    return [name_to_idx[n] for n in names if n in name_to_idx]


def oversample_data(formatted_data, duplication_factor=4):
    """
    Oversample dialogues containing rare emotions
    """
    oversampled_data = []

    for dialogue_string in formatted_data:

        oversampled_data.append(dialogue_string)

        emo_idxs = set(extract_emotion_indices(dialogue_string))

        if emo_idxs & rare_emotion_indices:
            oversampled_data.extend([dialogue_string] * duplication_factor)

    print(f"Original: {len(formatted_data)}")
    print(f"Oversampled: {len(oversampled_data)}")

    return oversampled_data