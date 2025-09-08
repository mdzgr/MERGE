
def file_comparison(input_file1, input_file2):
    '''compares 2 files and outputs how many entries are the same, and how many are diff'''
    with open(input_file1, 'r') as f:
        entries = json.load(f)

    with open(input_file2, 'r') as f:
        reference_entries = json.load(f)

    entry_ids = {entry['id'] for entry in entries}
    reference_ids = {entry['id'] for entry in reference_entries}

    filtered_entries = [entry for entry in entries if entry['id'] in reference_ids]
    missing_entries = [entry for entry in reference_entries if entry['id'] not in entry_ids]

    print(f"Found {len(filtered_entries)} matching entries from {input_file1}.")
    print(f"Found {len(missing_entries)} missing entries from {input_file2}.")

    return filtered_entries, missing_entries



def filter_entries_by_id(input_file1, input_file2, output_file="filtered_entries.json", missing_output_file="missing_entries.json"):
    '''compares 2 files and splits them into common and missing entries, creating 2 files'''
    with open(input_file1, 'r') as f:
        entries = json.load(f)

    with open(input_file2, 'r') as f:
        reference_entries = json.load(f)

    entry_ids = {entry['id'] for entry in entries}
    reference_ids = {entry['id'] for entry in reference_entries}

    filtered_entries = [entry for entry in entries if entry['id'] in reference_ids]
    missing_entries = [entry for entry in reference_entries if entry['id'] not in entry_ids]

    with open(output_file, 'w') as f:
        json.dump(filtered_entries, f, indent=2)

    with open(missing_output_file, 'w') as f:
        json.dump(missing_entries, f, indent=2)

    print(f"Found {len(filtered_entries)} matching entries from {input_file1}. Saved to {output_file}")
    print(f"Found {len(missing_entries)} missing entries from {input_file2}. Saved to {missing_output_file}")

    return filtered_entries, missing_entries



def save_dataset_to_json(dataset, directory, filename):
    """
    Saves a dataset (list of dictionaries) to a JSON file within a specified directory.
    """
    os.makedirs(directory, exist_ok=True) #dictionary to save file, if not existent create

    filepath = os.path.join(directory, filename) #dictionary+filepath

    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset successfully saved to {filepath}")

def load_data(json_input):
  '''load function'''
  if isinstance(json_input, str):
      with open(json_input, "r", encoding="utf-8") as f:
          data = json.load(f)
  elif isinstance(json_input, list):
      data = json_input
  else:
      raise ValueError("Input must be a filepath or a list of dictionaries.")
  return data


def merge_helper(a: dict, b: dict):
    """Return a shallow merge of two dicts (b overwrites a)."""
    out = a.copy()
    out.update(b)
    return out
