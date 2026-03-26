import os   

def load_split_from_local_files(directory_path, split_name):
    """
    Load DailyDialog dataset split from local files.

    Args:
        directory_path (str): Path to folder containing dataset files
        split_name (str): 'train', 'validate', or 'test'

    Returns:
        list of dicts:
        [
            {
                'dialog': [...],
                'emotion': [...],
                'act': [...]
            }
        ]
    """

    # -------------------------------
    # File selection based on split
    # -------------------------------
    if split_name == 'train':
        dialogues_file = os.path.join(directory_path, 'dialogues_train.txt')
        emotions_file = os.path.join(directory_path, 'dialogues_emotion_train.txt')
        acts_file = os.path.join(directory_path, 'dialogues_act_train.txt')

    elif split_name == 'test':
        dialogues_file = os.path.join(directory_path, 'dialogues_test.txt')
        emotions_file = os.path.join(directory_path, 'dialogues_emotion_test.txt')
        acts_file = os.path.join(directory_path, 'dialogues_act_test.txt')

    elif split_name == 'validate':
        dialogues_file = os.path.join(directory_path, 'dialogues_validation.txt')
        emotions_file = os.path.join(directory_path, 'dialogues_emotion_validation.txt')
        acts_file = os.path.join(directory_path, 'dialogues_act_validation.txt')

    else:
        raise ValueError("split_name must be: 'train', 'validate', or 'test'")

    # -------------------------------
    # Check files exist
    # -------------------------------
    for file_path in [dialogues_file, emotions_file, acts_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")

    dataset = []
    skipped = 0

    # -------------------------------
    # Read files together
    # -------------------------------
    with open(dialogues_file, 'r', encoding='utf-8') as f_dialog, \
         open(emotions_file, 'r', encoding='utf-8') as f_emotion, \
         open(acts_file, 'r', encoding='utf-8') as f_act:

        for dialog_line, emotion_line, act_line in zip(f_dialog, f_emotion, f_act):

            # Split dialogue into turns
            turns = [
                t.strip()
                for t in dialog_line.strip().split('__eou__')
                if t.strip()
            ]

            # Convert labels
            emotions = [int(e) for e in emotion_line.strip().split()]
            acts = [int(a) for a in act_line.strip().split()]

            # Validate alignment
            if len(turns) != len(emotions) or len(turns) != len(acts):
                skipped += 1
                continue

            dataset.append({
                'dialog': turns,
                'emotion': emotions,
                'act': acts
            })

    print(f"Loaded {len(dataset)} dialogues | Skipped {skipped} mismatched")

    return dataset