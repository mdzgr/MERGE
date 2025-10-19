import json
from collections import defaultdict, Counter

def extract_base_id_nv(full_id: str):
    # Trim suffixes like ':h:bert' etc.; adjust to your real ID format.
    return full_id.split(':')[0]

def collect_labels_per_base_id(*json_files):
    """
    Collect all labels per base ID across multiple JSON files.
    Returns:
      - dict: base_id -> list of all labels
      - set:  all unique base_ids
    """
    base_id_to_labels = defaultdict(list)

    for file_path in json_files:
        if not file_path:  
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)

        for item in data:
            full_id = item.get('id')
            label = item.get('label')
            if not full_id or not label:
                continue
            base_id = extract_base_id_nv(full_id)
            base_id_to_labels[base_id].append(label)

    all_unique_base_ids = set(base_id_to_labels.keys())
    return base_id_to_labels, all_unique_base_ids

def label_distribution_across_base_ids(base_id_to_labels):
    """
    Count how many unique base IDs have each label at least once.
    """
    label_presence_counter = Counter()
    for _, labels in base_id_to_labels.items():
        for lbl in set(labels):  # deduplicate per base ID
            label_presence_counter[lbl] += 1
    return label_presence_counter

