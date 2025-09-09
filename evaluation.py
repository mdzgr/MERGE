import json, evaluate
from collections import Counter, defaultdict
def transform_results(flat_list):
  '''formats results into a dictionary with different entries for standard and pattern accuracy'''
  structured_results = defaultdict(list)

  for entry in flat_list: # for each entry
      model = entry.get("model") # get the model
      input_file = entry.get("input_file") #get the input file
      metrics = entry.get("metrics", {}) #get the metrics, otherwise empty dictionary

      pattern_accuracy = {
          k.replace("pattern_accuracy_", ""): v #replace it with nothing to avoid repetition
          for k, v in metrics.items() # for keys in metric
          if k.startswith("pattern_accuracy_") #if the key starts with pattern accuracy
      }
      normal_accuracy = metrics.get("normal_accuracy")
      new_entry = { #store them in a dictionary
          "input_file": input_file,
          "normal_accuracy": normal_accuracy,
          "pattern_accuracy": pattern_accuracy
      }

      structured_results[model].append(new_entry)

  return dict(structured_results)


def map_labels_to_numbers(dataset, model_name):
    """
    for item labels converts them to numbers corresponding to each model.
    """
    #if cretain token is in model name
    #instantiate label_mapping with certain values

    if "_bert" in model_name.lower():
        label_mapping = {'entailment': 1, 'neutral':2 , 'contradiction': 0}
    if "_roberta" in model_name.lower():
        label_mapping = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    if "_deberta" in model_name.lower():
        label_mapping = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    if "albert-xxlarge-v2" in model_name.lower():
          label_mapping = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    if "_bart" in model_name.lower():
        label_mapping = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    new_dataset = []
    for entry in dataset: #  for each entry

        new_entry = entry.copy() #copy it
        original_label = new_entry.get('label') #get label mapping correspondance
        new_entry['label'] = label_mapping.get(original_label, None) #return -1 if value is not found, return None
        new_dataset.append(new_entry) # append new entry with converted label
    return new_dataset # return dataset with converted labels

def predictions_nli(model_name, model_name_dif_token, data_json_file, batch_size_number, device_g_c, batch_function, tok_model_function):
    #not double-checked
    """
    calculates predictions for a dataset

    args:
    model_name: model_name
    data_json_file: json file with stimuli for predictions
    batch_size_number: batch number for predictions
    device_g_c:  cuda or cpu
    batch_function: takes function from assign.tools to eval in batches
    tok_model_function: tokenizer function from assign tools

    outputs: a json file with the input file name, model name and its predictions
    ***Note json file has to contain premise and hypothesis
    """
    with open(data_json_file, "r") as f:
        data = json.load(f)
    data = map_labels_to_numbers(data, model_name_dif_token)

    tokenizer, model_cpu = tok_model_function(model_name)
    model_cpu.to(device_g_c)
    prem_hypo_list = [(item['premise'], item['hypothesis']) for item in data]
    preds2 = batch_function(tokenizer, model_cpu, prem_hypo_list, batch_size=batch_size_number, device=device_g_c)
    output = {
        "input_file": data_json_file,
        "model": model_name,
        "predictions": preds2
    }
    output_filename = f"{data_json_file.rsplit('.', 1)[0]}_{model_name_dif_token}_predictions.json"
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=4, default=lambda o: o.item() if hasattr(o, "item") else o)
    return output, output_filename
def merge_data_and_predictions(data_json_file, predictions_file, model_name, model_name_dif_tok):
    #not double-checked
    '''merges documents of the inflated dataset and the predictions from the model by zipping the data from json'''
    with open(data_json_file, "r") as f:
        data = json.load(f)
    with open(predictions_file, "r") as f:
        predictions_dict = json.load(f)
    merged = []
    data = map_labels_to_numbers(data, model_name_dif_tok)
    file_name=predictions_dict.get("input_file")
    model=predictions_dict.get("model")
    predictions = predictions_dict.get("predictions", [])
    if len(data) != len(predictions):
        print(f"Warning: Number of data items ({len(data)}) and predictions ({len(predictions)}) do not match.")
    for original, pred in zip(data, predictions):
        merged_entry = {
            'input_file': file_name,
            'model': model,
            'id': original.get('id'),
            'premise': original.get('premise'),
            'hypothesis': original.get('hypothesis'),
            'gold_label': original.get('label'),
            'label_index': pred.get('label_index'),
            'label': pred.get('label'),
            'prob': pred.get('prob'),
            'probs': pred.get('probs')
        }
        merged.append(merged_entry)
    output_filename = data_json_file.rsplit('.', 1)[0] + '_merged.json'
    with open(output_filename, 'w') as f:
        json.dump(merged, f, indent=4)


    return merged, output_filename

def compute_all_metrics(json_filepath, dictionary_result, type_evaluation, thresholds_list:list, id_type: int, calculate_per_label=False):
    #not-double-checked
  '''calculates various metrics: SA, PA Acc, Consistency (first, average-based) separately (or not) per label
  args:
      json_filepath: file with stimuli and predictions
      dictionary_result: dict where to store results
      type_evaluation: the reference taken in eval has 3 poss values:
              0 = gold label comparison (for SA and PA)
              1 = first-prediction (consistency first-choice based)
              2 = avr prediction label (consistency average-base)
      thresholds_list: list of threshold levels for the metrics
      id_type: what type of id the dataset has (only original id from SNLI, or one that encodes info about the pos tag replaced, position etc.)
              values are used for:
              0 = seed datasets  e.g. 6160193920.jpg#4r1e
              1 = inflated datasets, e.g. 6160193920.jpg#4r1e:very:RB:49:53:13:17:really
      calculate_per_label: if specified it will calculate PA for every label separately
  '''
  if '.json' in json_filepath:
    with open(json_filepath, "r") as f:
        data = json.load(f)
  else:
    data=json_filepath
  model=data[0]['model']
  input_file=data[0]['input_file']

  predictions = [entry["label_index"] for entry in data]
  references = [entry["gold_label"] for entry in data]
  metric = evaluate.load("accuracy")

  normal_result = metric.compute(predictions=predictions, references=references)
  normal_accuracy = normal_result["accuracy"]
  groups = defaultdict(lambda: {"predictions": [], "label": None})
  for entry in data:
      if id_type==0:
        id_prefix = entry["id"].split(":")[0]
      else:
        id_prefix=entry["id"]
      groups[id_prefix]["predictions"].append(entry["label_index"])
      if groups[id_prefix]["label"] is None:
          groups[id_prefix]["label"] = entry["gold_label"]

  thresholds = thresholds_list
  nested_accuracies = {}

  if calculate_per_label:
        unique_labels = set(references)
        per_label_accuracies = {label: {} for label in unique_labels}

  for threshold in thresholds:
      nested_final_predictions = []
      nested_labels = []
      if calculate_per_label:
            label_predictions = {label: [] for label in unique_labels}
            label_references = {label: [] for label in unique_labels}
      for group in groups.values():
          if type_evaluation==0:
            true_label = group["label"]
          if type_evaluation==1:
            true_label=group["predictions"][0]
          if type_evaluation == 2:
            true_label = Counter(group["predictions"]).most_common(1)[0][0]

          preds = group["predictions"]
          correct_ratio = sum(1 for pred in preds if pred == true_label) / len(preds)

          if correct_ratio >= threshold:
              nested_final_predictions.append(true_label)
          else:
              nested_final_predictions.append(-1)
          nested_labels.append(true_label)
          if calculate_per_label:
                label_predictions[true_label].append(nested_final_predictions[-1])
                label_references[true_label].append(true_label)
      nested_result = metric.compute(predictions=nested_final_predictions, references=nested_labels)
      nested_accuracies[threshold] = nested_result["accuracy"]
      if calculate_per_label:
            for label in unique_labels:
                if len(label_references[label]) > 0:
                    label_result = metric.compute(
                        predictions=label_predictions[label],
                        references=label_references[label]
                    )
                    per_label_accuracies[label][threshold] = label_result["accuracy"]
                else:
                    per_label_accuracies[label][threshold] = None

  if model not in dictionary_result:
      dictionary_result[model] = []
  key_name = (
            "pattern_accuracy" if type_evaluation == 0 else
            "consistency_accuracy_first" if type_evaluation == 1 else
            "majority_accuracy"
        )

  result_dict = {
    "input_file": json_filepath,
    "normal_accuracy": normal_accuracy,
    key_name: nested_accuracies
  }

  if calculate_per_label:
      result_dict["per_label_accuracies"] = per_label_accuracies

  dictionary_result[model].append(result_dict)

  print("SAMPLE Accuracy:", normal_accuracy)
  for threshold, acc in nested_accuracies.items():
      print(f"PATTERN Accuracy at threshold {threshold}: {acc}")

  if calculate_per_label:
        print("\nPer-label accuracies:")
        for label in unique_labels:
            print(f"Label {label}:")
            for threshold, acc in per_label_accuracies[label].items():
                print(f"  Threshold {threshold}: {acc}")

  return_dict = {
        "normal_accuracy": normal_accuracy,
        "nested_accuracies": nested_accuracies,
        "dictionary_results": dictionary_result
    }

  return return_dict




def get_base_ids_pos(filepath):
    """Reads a JSON file and extracts the first part of the 'id' for each entry."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file {filepath}: {e}")
        return set()

    base_ids = set()
    for entry in data:
        if 'id' in entry:
            parts = entry['id'].split(':')
            if parts:
                base_ids.add(parts[0])
    return base_ids
def get_base_ids_2pos(filepath):
    """Reads a JSON file and extracts the first part of the 'id' for each entry."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file {filepath}: {e}")
        return set()

    base_ids = set()
    for entry in data:
        if 'id' in entry:
            parts = entry['id'].split(':')
            if parts:
                base_ids.add(parts[0])
    return base_ids

def filter_dataset_by_ids_pos(dataset_path, id_set):
    """
    Filters a dataset (JSON file) to include only entries whose base ID is in the provided set.

    Args:
        dataset_path (str): Path to the input JSON dataset file.
        id_set (set): A set of base IDs to keep.

    Returns:
        list: A list containing the filtered dataset entries.
              Returns an empty list if the file cannot be read or is empty.
    """
    try:
        with open(dataset_path, 'r') as f:
            dataset_entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        return []

    filtered_dataset = []
    for entry in dataset_entries:
        if 'id' in entry:
            parts = entry['id'].split(':')
            if parts and parts[0] in id_set:
                filtered_dataset.append(entry)
    return filtered_dataset
