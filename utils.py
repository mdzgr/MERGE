# Standard library
import gc
import glob
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import login
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
def load_data(json_input):
  '''loads data from a json file with nothing special'''
  if isinstance(json_input, str):
      with open(json_input, "r", encoding="utf-8") as f:
          data = json.load(f)
  return data

def return_id_list_items(items_list):
  '''returns a list of item ids'''
  return [item['id'] for item in items_list]

def make_directory(directory):
  '''makes directory if not existent'''
  os.makedirs(directory, exist_ok=True)

def join_directory_and_filename(file_path, directory):
  '''returns path of file +path'''
  return os.path.join(directory, file_path)

def create_json_from_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Dataset successfully saved to {path}")

def save_data_to_json_to_directory(data, json_file_name, directory):
  '''saves json file to directory'''
  make_directory(directory)
  file_path = join_directory_and_filename(json_file_name, directory)
  create_json_from_data(data, file_path)

def split_to_get_base_id(id_item):
  return id_item.split(":")[0]

def combine_json_files(file_paths, output_file):
    """
    Combines list of multiple JSON files into a single json file.
    """
    combined_data = []

    combined_data.extend(load_data(file_path) for file_path in file_paths)

    return create_json_from_data(combined_data, output_file)

def extract_base_id_with_hashtag(full_id):
    return full_id.split("#")[0]

def get_base_ids_from_data(data, type_data):
    """returns a set of base ids for a nested file or a list for a flat file."""
    if type_data =='nested':
      return set([split_to_get_base_id(values['id']) for values in data.values()])
    if type_data =='flat':
      return [split_to_get_base_id(item['id']) for item in data]
def return_basename_file_path(file_path):
  '''rteurns file base name'''
  return os.path.basename(file_path)

def get_destination_path(local_file_path, output_directory):
    '''gets base name and creates file in a directory'''
    return join_directory_and_filename(
        return_basename_file_path(local_file_path),
        output_directory
    )

def save_files_folder(pattern_glob, output_drive_directory):
  paths=glob.glob(pattern_glob)
  make_directory(output_drive_directory)
  for local_file_path in paths:
      if os.path.exists(local_file_path):
          drive_file_path = get_destination_path(local_file_path, output_drive_directory)
          shutil.copy(local_file_path, drive_file_path)


def group_by_base_var(data_var):
    '''Group variant items by their base id (prefix before the first ":").
    -> {base_id: [list of variant items]}'''
    return build_id_lookup(data_var, 'base')


def group_by_base_id_nested(validation):
  '''takes nested dataset and groups by base ids
  output structure #grouped_base_id defaultdict(<function <lambda> at 0x7ce7f9c011c0>, {'final_all_1': defaultdict(<class 'list'>, {'101001n': [False, False, False,'''
  grouped = defaultdict(lambda: defaultdict(list))

  for dataset, dataset_dict in validation.items():
    for key, value in dataset_dict.items():
        grouped[dataset][key.split(':', 1)[0]].append(value)

  return grouped


def build_id_lookup(dataset, form_id):
  '''for each item, make that item s id its key in a dict'''
  if form_id=='full':
    return {item['id']: item for item in dataset}
  if form_id == 'base':
    grouped = defaultdict(list)
    for item in dataset:
        grouped[split_to_get_base_id(item['id'])].append(item)
    return grouped

def data_subsample(name_file, number_subsample, name_output_file):
  '''function opening files, saving an x random subsample into new file, and returning ids'''
  data=load_data(name_file)
  subsample = random.sample(data, number_subsample)
  ids_subsample=return_id_list_items(subsample)
  create_json_from_data(subsample, name_output_file)
  return ids_subsample


def return_common_ids(dataset, ids_list, type_id):
  '''returns the dataset items that have ids in a id list'''
  if type_id=='full':
    return [item for item in dataset if item['id'] in ids_list]
  if type_id=='base':
    return [item for item in dataset if split_to_get_base_id(item['id']) in ids_list]



def subsample_nested_dataset_from_id_list(nested_data, id_list, type_id):
    """subsamples nested file"""
    id_set = set(id_list)
    return {
        dataset_name: return_common_ids(records, id_set, type_id)
        for dataset_name, records in nested_data.items()
    }

def data_subsample_nested(name_file, ids_already_subsampled, name_sampled_nested, type_id):
  '''function creating new subsampled file from nested file
  this is split id'''
  data=load_data(name_file)
  new_dict=defaultdict(list)
  new_dict=subsample_nested_dataset_from_id_list(data, ids_already_subsampled, type_id)
  create_json_from_data(new_dict, name_sampled_nested)


def check_nested(subsampled_seed_problems_file, nested_file_variant):
  '''test to check if subsampling of nested was succesful based on a flat json file of subsamled items'''

  data_seed=load_data(subsampled_seed_problems_file)
  data_nested=load_data(nested_file_variant)
  ids_seed_problem= return_id_list_items(data_seed)
  nested_groupbed_by_original_id = {split_to_get_base_id(item['id']) for values in data_nested.values() for item in values}
  assert set(ids_seed_problem) == set(nested_groupbed_by_original_id)


def print_number_of_entries_found(entries, name_file):
  '''prints number of matching entries in a file'''
  print(f"Found {len(entries)} entries in {name_file}.")

def file_comparison(input_file1, input_file2):
    '''compares 2 files and outputs how many entries are the same, and how many are diff'''

    entries=load_data(input_file1)
    reference_entries=load_data(input_file2)

    entry_ids = return_id_list_items(entries)
    reference_ids = return_id_list_items(reference_entries)

    filtered_entries= return_common_ids(entries, reference_ids, 'full')
    missing_entries= return_common_ids(reference_entries, entry_ids, 'full')
    print_number_of_entries_found(filtered_entries, input_file1)
    print_number_of_entries_found(missing_entries, input_file2)

    return filtered_entries, missing_entries



def filter_entries_by_id(input_file1, input_file2, output_file="filtered_entries.json", missing_output_file="missing_entries.json"):
    '''compares 2 files and splits them into common and missing entries, creating 2 files'''

    filtered_entries, missing_entries=file_comparison(input_file1, input_file2)
    create_json_from_data(filtered_entries, output_file)
    create_json_from_data(missing_entries, missing_output_file)
    return filtered_entries, missing_entries


def delete_files_containing_word(folder, word):
    deleted = []
    for filename in os.listdir(folder):
        if word in filename:
            path = os.path.join(folder, filename)
            os.remove(path)
            deleted.append(filename)
            print(f"Deleted: {filename}")
    print(f"\n{len(deleted)} files deleted.")
    return deleted

def load_predictions_LLMs_with_key(path, key):
    """Load flat prediction list from a JSON file under a given key."""
    if key:
      with open(path, 'r') as f:
          return json.load(f)[key]
    else:
      return load_data(path)


def trim_prediction_ids(prediction_map):
    """Strip the last segment of each ID (everything before the last ':')."""
    return {':'.join(k.split(':')[:-1]): v for k, v in prediction_map.items()}


def filter_nested_by_predictions(nested_data, prediction_map):
    """Keep only records whose ID appears in the prediction map."""
    result = {}
    for dataset_name, records in nested_data.items():
        result[dataset_name] = {
            item['id']: prediction_map[item['id']]
            for item in records
            if item['id'] in prediction_map
        }
    return result



def transpose_predictions(scr_, predictions):
  '''checks if scrambled so that the ids of predictions are cut'''
  if scr_ == True:
    return trim_prediction_ids(predictions)
  else:
    return predictions

def recast_predictions_to_nested(pred_path, key_predictions, scr_, nested_path, output_path):
    """Full pipeline: load predictions, match to nested data, save result."""
    predictions     = load_predictions_LLMs_with_key(pred_path, key_predictions)
    prediction_map  = transpose_predictions(scr_, predictions)
    nested_data     = load_data(nested_path)
    result          = filter_nested_by_predictions(nested_data, prediction_map)
    create_json_from_data(result, output_path)
    return result


def subsample_nested_file(nested_path, id_list, output_path, type_id):
    """Full pipeline: load predictions, match to nested data, save result. this is full id"""
    nested_data = load_data(nested_path)
    result      = subsample_nested_dataset_from_id_list(nested_data, id_list, type_id)
    create_json_from_data(result, output_path)
    return result

def sample_random_examples_from_variants_minimum(variants, number_variants_to_sample):
  '''dataset out of which an x number of variants are sampled with mminimum'''
  return random.sample(variants, min(number_variants_to_sample, len(variants)))

def book_of_base_ids_per_label(dataset):
  '''resturctures items by label, & base_ids {: {: []}}
  Organize items by label -> base_id -> items'''
  grouped = defaultdict(lambda: defaultdict(list))
  for item in dataset:
      grouped[item['label']][split_to_get_base_id(item['id'])].append(item)
  return grouped


def find_few_shot_examples(data, n_variants=2):
    """
    return a tuple of id and label chosen for few shot
    """

    grouped = book_of_base_ids_per_label(data)
    selected = []
    for label, base_groups in grouped.items():

        representative_base_id = random.choice(list(base_groups.keys()))
        sampled = sample_random_examples_from_variants_minimum(base_groups[representative_base_id], n_variants)
        selected.extend((item['id'], label) for item in sampled)
    return selected

def load_two_files(file_1, file_2):
    '''load two datasets at the same time maybe could add ** to make more datasets'''
    return load_data(file_1), load_data(file_2)

def find_few_shot_for_two_datasets(data_1, data_2):
    '''find few shots for two datasets'''
    return find_few_shot_examples(data_1), find_few_shot_examples(data_2)

def print_few_shot_examples(item):
    print(f"Premise: {item['premise']}")
    print(f"Hypothesis: {item['hypothesis']}")
    print("""A. Entailment\nB. Neutral\nC. Contradiction""")

def print_few_shot_examples_for_dataset(dataset_name, examples, lookup, mapping_labels):
    '''mapping_labels_printin={'contradiction':'C', 'entailment':'A', 'neutral':'B'}'''
    print(f"\n=== {dataset_name} Few-Shot Examples ===")
    for vid, label in examples:
        item = lookup[vid]
        print_few_shot_examples(item)
        print(f"Answer: {mapping_labels.get(label)}")


def select_few_shot_and_print(file_snli, file_mnli, mapping_labels):

    snli_data, mnli_data = load_two_files(file_snli, file_mnli)
    snli_examples, mnli_examples = find_few_shot_for_two_datasets(snli_data, mnli_data)
    snli_lookup, mnli_lookup = build_id_lookup(snli_data, 'full'), build_id_lookup(mnli_data, 'full')

    print_few_shot_examples_for_dataset("SNLI", snli_examples, snli_lookup, mapping_labels)
    print_few_shot_examples_for_dataset("MNLI", mnli_examples, mnli_lookup, mapping_labels)

def reverse_dictionary_values_to_keys(dictionary):
    '''reverses dictionary values to keys
    if 2 values they become two different entires
    value_1: same key
    value_2: same key'''
    return {v: k for k, values in dictionary.items() for v in values}

def make_filename_split_by_open_class(category, output_suffix):
    return f"{category}_dataset_{output_suffix}.json"

def write_json_files_where_variants_split_by_open_class(splits, output_suffix):
  '''creates from a dictionary that contains {cat: [items],} different files'''
  for cat, content in splits.items():
        out_file = make_filename_split_by_open_class(cat, output_suffix)
        create_json_from_data(content, out_file)
        print(f"({sum(len(v) for v in content.values())} items)")

def get_tag_of_id(id_item):
  '''returns the tag of an id'''
  return id_item.split(":")[2]

def reorganize_nested_data_by_value_in_id(nested_data, dictionary_map_id_part_to_category, function_to_split_id):
  '''re-organizez nested data by a specific value in the id of items'''
  splits={cat: {} for cat in dictionary_map_id_part_to_category}
  for dataset_key, items in nested_data.items():
    for item in items:
      cat=dictionary_map_id_part_to_category.get(function_to_split_id(item['id']))
      if cat:
        splits[cat].setdefault(dataset_key, []).append(item)
  return splits


def pos_tag_split_dataset(nested_file, pos_tag_dictionary, function_id_split,output_suffix="bothcpAh_ran_20"):
    '''creates nested files split by open class categories from nested files
    function for id split is get_tag_of_id here
    reference:
    POS_TAG_DICTIONARY = {
    'noun': {'NN', 'NNS', 'NNP', 'NNPS'},
    'verb': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
    'adjective': {'JJ'},
    'adverb': {'RB'}
    }'''
    data=load_data(nested_file)
    tag_to_cat = reverse_dictionary_values_to_keys(pos_tag_dictionary)
    splits = reorganize_nested_data_by_value_in_id(data, tag_to_cat, function_id_split)
    write_json_files_where_variants_split_by_open_class(splits, output_suffix)
def entires_in_re_written_file(data):
  '''returns dictionary with entires from data, data can either be data[1] for re-structur or just data if no multiple items'''
  return {'model': data['model'], 'input_file': data['input_file'], 'items': []}

def merge_items(original_item, updated_item):
    return {
        "id": updated_item["id"],
        "gold_label": original_item["gold_label"],
        "label_index": original_item["label_index"],
        "probs": original_item["probs"]
    }

def label_index_normalization(item, label_mapping=None):
  '''returns label index considering if label mapping is present'''
  if label_mapping is None:
    return item["label_index"]
  else:
    return label_mapping[str(item.get("label_index", item.get("label")))]

def normalize_item(item, label_mapping=None):
    '''resturctures item, depning on label mapping or not'''

    output = {
        "id": item['id'],
        "gold_label": item["gold_label"],
        "label_index": label_index_normalization(item, label_mapping)
    }
    if "probs" in item:
      output["probs"] = item["probs"]
    return output


def structure_for_re_written_file(data, dictionary):
  '''gives back dictionary with added items but re written to have
  'id': id,
  'gold_label': gold_label,
  'label_index': label_index,
  'probs': probs'''
  dictionary['items'] = [normalize_item(it) for it in data]

def convert_file_predictions(inp, out):
    data = load_data(inp)

    dictionary_new= entires_in_re_written_file(data[1])
    structure_for_re_written_file(data, dictionary_new)

    create_json_from_data(dictionary_new, out)

def merge_files(orig_path, updated_path, out_path):
    '''merges 2 files, but with the ids of the second as main ids'''

    orig, upd=load_two_files(orig_path, updated_path)
    comapre_values_of_two_entries_dict(orig, upd, "items", "items")

    merged_items = []

    for o, u in zip(orig["items"], upd["items"]):
        merged_items.append(merge_items(o, u))
    merged=entires_in_re_written_file(orig)
    merged['items']=merged_items
    create_json_from_data(merged, out_path)
def comapre_values_of_two_entries_dict(data_1, data_2, key_name_1, key_name_2):
  '''returns if the count of twy dictinnaries keys is the same'''
  assert len(data_1[key_name_1]) == len(data_2[key_name_2]), "Item counts differ!"

def model_family(model, dictionary_for_model_family):
  '''standard dict
  MODEL_FAMILIES = {
    "roberta-large-mnli": "uppercase",
    "textattack/roberta-base-mnli": "0_2_reversed",
    "deberta-v3": "non-standard_hf",
    "bart-base-snli": "non-standard_hf",
    "/bert": "non-standard_hf",
    "roberta-base": "non-standard_hf",
    "gpt2-large": "non-standard_hf"
}
  '''
  for pattern, family in dictionary_for_model_family.items():
        if pattern in model.lower():
            return family

  return "standard"

def model_prediction_mapping(model_family):
  '''returns mappinf for models' prediction'''
  label_mapping= {'standard': {'entailment': 0, 'neutral': 1, 'contradiction': 2},
                'uppercase': {'ENTAILMENT': 0, 'NEUTRAL': 1, 'CONTRADICTION': 2},
                '0_2_reversed': {'LABEL_0': 2, 'LABEL_2': 0, 'LABEL_1': 1},
                'non-standard_hf': {'LABEL_0': 0, 'LABEL_2': 2, 'LABEL_1': 1}}
  return label_mapping[model_family]


def from_model_family_to_model_label(model, model_family_dictionary):
  '''returns mappinf for models' prediction'''
  return model_prediction_mapping(model_family(model, model_family_dictionary))



def rewrite_labels(inp, out, model_families):
    '''takes prediction file of a model and re maps its labels for evaluation
    -> load data
    -> gets faimily model name bc there are several mappins
    -> normalizes the prediction file strutcure to only keep essential data: id, gold_label, label_index & probs'''
    data=load_data(inp)
    label_mapping = from_model_family_to_model_label(data['model'], model_families)
    print(data['model'], 'label_mapping', label_mapping)
    data["items"] = [normalize_item(it, label_mapping) for it in data["items"]]

    create_json_from_data(data, out)


def x_number_of_random_subsamples(dataset, number_of_random_samples, number_items_per_sample):
  '''takes dataset [items]-> outputs{sample_x_number}: [items.number_of_items]'''
  samples=defaultdict(dict)
  for i in range(number_of_random_samples):
    sample = random.sample(dataset, number_items_per_sample)
    samples[f'sample{i}']=sample
  return samples

def choose_sample(subsamples, type_choice):
  '''from a dictionary of random samples returns one randomly'''
  if type_choice=='random':
    return subsamples[random.choice(list(subsamples.keys()))]

def create_seed_problems(folder, name_file, number_samples, number_seed_problems):
    '''creates an x number of random problems'''
    seed_data=load_data(f'{folder}{name_file}')
    seed_problems=x_number_of_random_subsamples(seed_data, number_samples, number_seed_problems)

    chosen=choose_sample(seed_problems, 'random')
    ID_subsmapled_SNLI=return_id_list_items(chosen)
    return seed_problems, chosen, ID_subsmapled_SNLI

def is_prediction_correct(item):
    return item["gold_label"] == item["label_index"]

def correctness_scores(data):
    return [int(is_prediction_correct(item)) for item in data]

def accuracy_calculation(model_scores, if_values=None):
  '''outputs accuracy sum/len'''
  if if_values:
    return sum(model_scores.values())/len(model_scores)
  else:
    return sum(model_scores)/len(model_scores)



def rename_files_in_folder(folder, old_pattern, new_pattern):
    for filename in os.listdir(folder):
        if old_pattern in filename:
            new_name = filename.replace(old_pattern, new_pattern)
            os.rename(
                os.path.join(folder, filename),
                os.path.join(folder, new_name)
            )
            print(f"Renamed {filename} → {new_name}")

def copy_file_with_new_name(path, new_name):
    folder = os.path.dirname(path)
    new_path = os.path.join(folder, new_name)
    shutil.copy2(path, new_path)
    print(f"Copied to: {new_path}")
    return new_path
itself_map = {
    'bert-base-S':          'bert',
    'bert-base-M':          'bert',
    'bert-large-S':         'bertl',
    'roberta-base-S':       'roberta',
    'roberta-base-M':       'roberta',
    'roberta-large-SMFA':   'robertal',
    'roberta-large-M':      'robertal',
    'electra-large-SMFA':   'electral',
    'albert-xxlarge-SMFA':  'albertxxl',
}

size_map = {
    'bert-base-S':          'bertl',
    'bert-base-M':          'bertl',
    'bert-large-S':         'bert',
    'roberta-base-S':       'robertal',
    'roberta-base-M':       'robertal',
    'roberta-large-SMFA':   'roberta',
    'roberta-large-M':      'roberta',
    'electra-large-SMFA':   'electrab',
    'albert-xxlarge-SMFA':  'albert',
}

def scores_threshold_comparison(scores, threshold):
  '''returns 1 if accuracy is bigger than a threshold'''
  return 1 if accuracy_calculation(scores, None)>= threshold else 0

def PA_score(predictions_by_id, subsampled_variants_initial, thresholds):
    thresholds_scores = {}
    #for each threshold in a list of threshold have an operation
    for threshold in thresholds:
        scores_per_minidataset = {}
        for k, v in subsampled_variants_initial.items():
            base_ids_with_90_threshold = []
            for each_base_id_list in v:
                scores=[int(is_prediction_correct(predictions_by_id[item])) for item in each_base_id_list]
                base_ids_with_90_threshold.append(scores_threshold_comparison(scores, threshold))
            scores_per_minidataset[k] = accuracy_calculation(base_ids_with_90_threshold, None)

        thresholds_scores[threshold] = accuracy_calculation(scores_per_minidataset, True)
    return thresholds_scores

#structure_file_store_ids #varNLI #mutate_items_to_ids
def mutate_items_to_ids(type_eval, data, key_name_special):
    '''keeps structure of file dict: [] but saves ids of items
    key_name_special is key for file that has multiple nested dictionaries'''
    return (
        {k: return_id_list_items(v) for k, v in data.items()}
        if type_eval == key_name_special
        else {type_eval: return_id_list_items(data)}
    )
def group_items_by_base_id(data_nested_by_id):
    '''returns dict of which elements are id.split: item'''
    subsampled_variants_initial = defaultdict(list)
    for k, v in data_nested_by_id.items():
        subsampled_variants_initial[k] = build_id_lookup(v, 'base')
    return subsampled_variants_initial

def get_last_element_of_id(item_id):
    '''for id returns last element after split(:)'''
    return item_id.split(':')[-1]

def get_origin_category(item_id, eval_model, rename_map=None):
    o = get_last_element_of_id(item_id)
    if rename_map and o in rename_map:
        o = rename_map[o]
    if o == 'both':
        return 'both'
    elif o == itself_map.get(eval_model):
        return 'itself'
    elif o == size_map.get(eval_model):
        return 'itself_size'
    else:
        return 'other'

def filter_by_origin_model(data_nested_by_id, target_origin, eval_model, rename_map=None):
    return {
        k: [item_id for item_id in v if get_origin_category(item_id, eval_model, rename_map) == target_origin]
        for k, v in data_nested_by_id.items()
    }

def get_model_names(folder_pattern, avoid_):
    return {
        Path(fp).stem.split("__n__")[-1]
        for fp in glob.glob(folder_pattern)
        if "__n__" in Path(fp).name and avoid_ not in Path(fp).name
    }


def return_names_files(name_eval, file_j_dict, small_file_dict):
    file_j = file_j_dict[name_eval]
    small_files = small_file_dict[name_eval]
    return file_j, small_files

def return_look_up_id_gold_label_label_index(preds):
  '''build look up id : {gold_label, label_index}'''
  return {r['id']: {'gold_label': r['gold_label'], 'label_index': r['label_index']} for r in preds['items']}



def load_and_process_model_preditcions(directory, model_names, processor):
    '''
    a list of models
    -> get their predictions {directory/model_name}
    -> do something with those predictions / motidify them
    -> return a dictionary with model_name: {predictions}
    '''
    results = {}

    for model in model_names:
        data = load_data(f"{directory}{model}.json")
        results[model] = processor(data)

    return results

def model_scores_function(small_file_name, all_model_predictions, names_models, data_nested_by_id, subsampled_variants_initial, thresholds):
    model_scores, SA_scores, PA_scores = {}, {}, {}
    for model in tqdm(names_models):
        predictions_by_id = all_model_predictions[model]
        SA_all_small_datasets = {}
        for small_dataset, list_ids in data_nested_by_id.items():
            scores=[int(is_prediction_correct(predictions_by_id[item])) for item in list_ids]
            SA_all_small_datasets[small_dataset] = accuracy_calculation(scores, None)

        SA_scores[model] = accuracy_calculation(SA_all_small_datasets, True)
        PA_scores[model] = PA_score(predictions_by_id, subsampled_variants_initial, thresholds)
    return {small_file_name: {"normal_accuracy": SA_scores, "PA_scores": PA_scores}}

def return_name_files_predictions(seed_var_test, split, type_var, folder_predictions, folder_nested, type_pos=None):
    if seed_var_test in ['test', 'seedNLI']:
        nested = f'{folder_nested}{seed_var_test}__{split}__{type_var}.json'
    if seed_var_test in ['varNLI']:
        if type_var=='scr':
            nested = f'{folder_nested}{seed_var_test}__{split}__pah__nest.json'
        else:
          nested = f'{folder_nested}{seed_var_test}__{split}__{type_var}__nest.json'
    if type_pos is not None and type_pos in ['nouns', 'verbs', 'adjectives']:
        nested = f'{folder_nested}{type_pos}_dataset_bothcpAh_ran_20.json'
    if type_pos is not None:
        nested = f'{folder_nested}all.1.{type_pos}.te.inf.20.json'
    return f'{folder_predictions}{seed_var_test}__{split}__{type_var}__n__', [nested]

def return_accuracy_scores(scores):
    '''returns normal and PA scores from dictionary'''
    return scores['normal_accuracy'], scores['PA_scores']

def refromat_PA_scores(PA, model):
  '''takes a dictionary with PA scores
  finds the model scores
  then reformats into thr: score and gives back dict'''
  return {f"pattern_accuracy_{thr}": pa for thr, pa in PA[model].items()}

def reformat_accuracy_metrics(SA, PA, model):
  '''reformat to nomral acc, pat acc'''
  metrics = {"normal_accuracy": SA} | refromat_PA_scores(PA, model)
  return metrics

def format_suffix(value):
  '''adds _ + value'''
  return f"_{value}" if value else ""

def return_structure_metric_file(model, small_file, metrics):
    return {"model": model, "input_file": small_file, "metrics": metrics}

def save_results_models_scores(model_scores, seed_var_test, split, type_var, type_pos):
    json_friendly = []
    for small_file, d in model_scores.items():
        SA, PA = return_accuracy_scores(d)
        for model, sa in SA.items():
            metrics = reformat_accuracy_metrics(sa, PA, model)
            json_friendly.append(return_structure_metric_file(model, small_file, metrics))
    type_pos_name = format_suffix(type_pos)
    create_json_from_data(json_friendly, f'/content/r_{seed_var_test}_{split}_{type_var}{type_pos_name}.json')

def eval_NLI(seed_var_test, sample_file, split, type_var, type_pos, folder_predictions, folder_nested, names_models, thresholds, origins=None, rename_map=None):
    rename_map = {'None': 'bert'}
    for i in seed_var_test:
        file_j, small_files = return_name_files_predictions(i, split, type_var, folder_predictions, folder_nested, type_pos)
        if sample_file:
            small_files = [sample_file]
        print(f"\n=== Processing FILE: {file_j} ===")
        all_model_predictions = load_and_process_model_preditcions(file_j, names_models, return_look_up_id_gold_label_label_index)
        print(small_files)
        for small_file in tqdm(small_files):
            print(f"\n=== Processing SMALL FILE: {small_file} ===")
            small_data = load_data(small_file)
            data_nested_by_id = mutate_items_to_ids(i, small_data, 'varNLI')

            subsampled_variants_initial = group_items_by_base_id(data_nested_by_id)
            model_scores = model_scores_function(small_file, all_model_predictions, names_models, data_nested_by_id, subsampled_variants_initial, thresholds)
            save_results_models_scores(model_scores, i, split, type_var, type_pos)

    return model_scores


def filter_validation_by_origin(validation, origin, rename_map=None):
    def get_origin(item_id):
        o = item_id.split(':')[-1]
        if rename_map and o in rename_map:
            return rename_map[o]
        return o

    return {
        dataset: {item_id: val for item_id, val in dataset_dict.items()
                  if get_origin(item_id) == origin}
        for dataset, dataset_dict in validation.items()
    }

def return_datawkey_if_value(type_test, list_type, dataset):
  '''appends a key to a data structure if type_test is in a certain type of data'''
  if type_test in list_type:
      return {type_test: dataset}
  else:
      return dataset

def replace_name_keys(data, pattern, pattern_to_replace):
  '''give back dictionary with some keys replaces'''
  return {k.replace(pattern, pattern_to_replace): v for k, v in data.items()}


def return_prefix_file(kind, firs_value, second_value):
  '''returns the prefix of file as kind if == first value or kind+combined if else'''
  return kind if kind ==  firs_value else f"{kind}{second_value}"




def load_evaluation_data_in_context(model, split, kind, type_data, pos_tag, prompt_type,
                                    folder_preds, folder_truth):

    '''load data for evaluation, new file structure'''
    model_name = name_models_dictionary.get(model)
    prefix=return_prefix_file(kind, 'test', 'LLM')
    pred_path = (
        f"{folder_preds}{prefix}__{split}__{type_data}__"
        f"{prompt_type}__{model_name}.json"
    )

    suffix =  "__nest.json" if kind == "var" else '.json'
    if pos_tag in ['nouns', 'verbs', 'adjectives']:
      truth_path = f"{folder_truth}{pos_tag}_dataset_bothcpAh_ran_20.json"
    if pos_tag in ['nvv', 'nvn', 'adjvadj', 'adjvv', 'adjnadj', 'adjnn']:
      truth_path=f'{folder_truth}all.1.{pos_tag}.te.inf.20.json'
    else:
      truth_path = f"{folder_truth}{prefix}__{split}__{type_data}{suffix}"

    results_2, data= load_two_files(pred_path, truth_path)

    mapping_datasedt={'nvv': 'verbs',
                      'nvn': 'nouns',
                      'adjvv': 'verbs',
                      'adjvadj': 'adjectives',
                      'adjnn': 'nouns',
                      'adjnadj': 'adjectives'}

    if pos_tag in ['nvv', 'nvn', 'adjvadj', 'adjvv', 'adjnadj', 'adjnn']:
      results_2=replace_name_keys(results_2, "final_all_", f"final_{mapping_datasedt.get(pos_tag)}_dataset_bothcpAh__random_")

    variants= return_datawkey_if_value(kind, ['test', 'seed', 'dev'], data)

    return pred_path, truth_path, results_2, variants

def map_nested_dataset_truth_values(_variants_omie, results_2, mapping_labels):
  '''gets a final all 1 nested dataset> checks for predictions> maps labels to truth values
  and returns dataset with true/false values
  output structure {'name_dataset_1': {'102777c:tracked:VBN:53:60:8:15:strapped:ph:both': False}, 'name_dataset_2': '''

  validation=defaultdict(dict)
  map_lbl = lambda x: mapping_labels.get(x, x)

  for dataset, elements in _variants_omie.items():
      res = results_2[dataset]

      validation[dataset].update({
          e['id']: map_lbl(e['label']) == map_lbl(res[e['id']])
          for e in elements if e['id'] in res
      })
  return validation

def keep_items_if_variants_bigger_than_n(dictionary, n):
  return {k: v for k, v in dictionary.items() if len(v) >= n}

def sanity_check_min_examples(grouped, n):
    out = {}
    for d, base in grouped.items():
        kept = keep_items_if_variants_bigger_than_n(base, n)
        if not kept:
          print(f"⚠️ {d}: none ≥ {n}")
        if kept:
            out[d] = kept
    return out

def PA_scores_CAL(thresholds, filtered_only_20):
  '''calculate PA scores'''
  PA_scores={}
  #threshold function
  for threshold in thresholds:
      pa_per_dataset = []
      for dataset, base_dict in filtered_only_20.items():
          base_level_scores = []

          for base_id, matches in base_dict.items():
              score=scores_threshold_comparison(matches, threshold)
              base_level_scores.append(score)
          pa_per_dataset.append(accuracy_calculation(base_level_scores, None))
      PA_scores[f"pattern_accuracy_{threshold}"] = (
          accuracy_calculation(pa_per_dataset, None)
          if pa_per_dataset else 0
      )
  return PA_scores

def score_choices_nll_stacked_one_tokenization(prompts, model, tokenizer, labels_spec):
    choices = labels_spec
    B = len(prompts)
    C = len(choices)

    choice_enc = tokenizer(
        choices,
        add_special_tokens=False
    )
    choice_lens = [len(ids) for ids in choice_enc["input_ids"]]
    all_texts = [p + c for p in prompts for c in choices]

    enc = tokenizer(
        all_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True
    ).to(model.device)

    with torch.inference_mode():
        logits = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask
        ).logits[:, :-1]


    labels = enc.input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)


    seq_len = labels.size(1)
    pos = torch.arange(seq_len, device=model.device).unsqueeze(0)
    full_lens = enc.attention_mask[:, 1:].sum(dim=1)
    choice_lens_expanded = torch.tensor(
        [l for _ in prompts for l in choice_lens],
        device=model.device
    )
    prompt_lens = full_lens - choice_lens_expanded

    mask = pos >= prompt_lens.unsqueeze(1)
    mask &= (labels != tokenizer.pad_token_id)

    nll = -(token_log_probs * mask).sum(dim=1)

    nll = nll.view(B, C)

    return {
        labels_spec[0]: nll[:, 0].tolist(),
        labels_spec[1]: nll[:, 1].tolist(),
        labels_spec[2]: nll[:, 2].tolist()
    }


def select_predictions(nlls, labels_spec):
    LABEL_MAP = {
        labels_spec[0]: "entailment",
        labels_spec[1]: "neutral",
        labels_spec[2]: "contradiction"
    }

    predictions = []
    batch_size = len(next(iter(nlls.values())))
    for i in range(batch_size):
        best_choice = min(
            labels_spec,
            key=lambda c: nlls[c][i]
        )
        predictions.append(LABEL_MAP[best_choice])

    return predictions

def print_length(list_l, name_of_length):
  '''prints length of list'''
  print(f"length of {name_of_length}: {len(list_l)}")

def run_nli_inference(
    input_path,
    output_path,
    type_test,
    model,
    token,
    prompts,
    variant_path,
    type_few_shot,
    number_type_few_shot,
    batch_size=8,
    labels_spec=[' Entailment', ' Neutral', ' Contradiction']
):
    import json
    from tqdm import tqdm

    datasets=load_data(input_path)
    print_length(datasets, 'dataset')
    if type_few_shot!=None:
      print('the variant path', variant_path)
      data_var=load_data(variant_path)
      grouped_seed_data=group_by_base_var(data_var)
    else:
      grouped_seed_data=None
    datasets=return_datawkey_if_value(type_test, ['seed', 'test', 'var'], datasets)

    results = {}

    for dataset, items in datasets.items():
        results[dataset] = {}

        print_length(items, 'items')

        for i in tqdm(range(0, len(items), batch_size), position=0, leave=True):
            batch = items[i:i + batch_size]
            preds = classify_nli_batch(
                batch,
                model,
                token,
                prompts,
                labels_spec,
                grouped_seed_data,
                type_few_shot,
                number_type_few_shot
            )

            for e, p in zip(batch, preds):
              results[dataset][e["id"]] = p
            torch.cuda.empty_cache()
    create_json_from_data(results, output_path)
    return results
from collections import Counter
def classify_nli_batch(batch, model, tokenizer, prompts_mix, labels_spec, grouped_seed_data, type_few_shot, number):

    prompts = build_prompts(batch, prompts_mix, type_few_shot, grouped_seed_data, number)

    nlls = score_choices_nll_stacked_one_tokenization(prompts, model, tokenizer, labels_spec)
    num_samples=1
    preds=select_predictions(nlls, labels_spec)
    if num_samples==1:
      return preds
    else:
        grouped_preds = [
        preds[i:i+num_samples]
        for i in range(0, len(preds), num_samples)
        ]
        majority_preds = [Counter(group).most_common(1)[0][0]
                          for group in grouped_preds
                          ]
        return majority_preds

def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="sdpa"

    )
    model.eval()
    model.config.use_cache = True
    torch.set_grad_enabled(False)

    return model, tokenizer

def input_filename(dataset, split, variant):
    if split=='seed' or split=='var':
      split=split+'LLM'
    return f"{split}__{variant}__pah.json"

def output_filename(dataset, split, variant, prompt_name, model_name):
    if split=='seed' or split=='var':
      split=split+'LLM'
    return f"{split}__{variant}__pah__{prompt_name}__{model_name}.json"

def print_statement_for_in_file_LLMs(in_file):
  print(f"=== Processing FILE: {in_file} ===")

def print_details_llms(model_name, dataset, split, variant, prompt_name, batch_size):
  print(f"{model_name} | {dataset} | {split} | {variant} | {prompt_name}")
  print('the bacth size in context', batch_size)

def run_in_context_predictions(dataset, model, tok, labels_spec, in_folder_path, out_folder_path, batch_size_1, split, variant, prompt_name, prompts, hf_model, model_name, variant_path, type_few_shot, number_type_few_shot):
    in_file = input_filename(dataset, split, variant)
    print_statement_for_in_file_LLMs(in_file)
    out_file = output_filename(dataset, split, variant, prompt_name, model_name)

    input_path, output_path = f"{in_folder_path}/{in_file}", f"{out_folder_path}/{out_file}"
    print_details_llms(model_name, dataset, split, variant, prompt_name, batch_size_1)

    run_nli_inference(
        input_path=input_path,
        output_path=output_path,
        type_test=split,
        model=model,
        token=tok,
        prompts=prompts,
        variant_path=variant_path,
        type_few_shot=type_few_shot,
        number_type_few_shot=number_type_few_shot,
        batch_size=batch_size_1,
        labels_spec=labels_spec
    )
import random
from collections import Counter

LABEL_TO_LETTER = {'contradiction': 'C', 'entailment': 'A', 'neutral': 'B'}


def format_few_shot_examples(examples: list[dict]) -> str:
    """Format a list of examples into a prompt string."""
    formatted = []
    for ex in examples:
        formatted.append(
            f"Premise: {ex['premise']}\n"
            f"Hypothesis: {ex['hypothesis']}\n"
            f"A. Entailment\n"
            f"B. Neutral\n"
            f"C. Contradiction\n"
            f"Answer: {ex['label']}\n"
        )
    return "\n".join(formatted)


def add_test_example(prompts, few_shot, example) -> list:
    """Append a test example (without answer) to each few-shot prompt."""

    test_block = (
        f"Premise: {example['premise']}\n"
        f"Hypothesis: {example['hypothesis']}\n"
        f"A. Entailment\n"
        f"B. Neutral\n"
        f"C. Contradiction\n"
        f"Answer:"
    )
    if isinstance(few_shot, list):
        prompts.extend([fs + test_block for fs in few_shot])
    else:
        prompts.append(few_shot + test_block)
    return prompts


def ensure_double_newline(text: str) -> str:
    """Ensure a prompt string ends with a double newline."""
    return text if text.endswith("\n\n") else text + "\n\n"


def parse_prompt_string(prompts_str: str) -> list[dict]:
    """Parse a raw prompt string back into a list of example dicts."""
    lines = prompts_str.split('\n')
    premises = [line.split('Premise: ')[1] for line in lines if line.startswith('Premise:')]
    hypotheses = [line.split('Hypothesis: ')[1] for line in lines if line.startswith('Hypothesis:')]
    answers = [line.split('Answer: ')[1] for line in lines if line.startswith('Answer:')]
    return [
        {'premise': p, 'hypothesis': h, 'label': a}
        for p, h, a in zip(premises, hypotheses, answers)
    ]


def scramble_text(text: str) -> str:
    """Randomly shuffle the words in a string."""
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)


def scramble_example(example: dict) -> dict:
    """Return a copy of an example with premise and hypothesis word-scrambled."""
    return {
        'id': example.get('id'),
        'premise': scramble_text(example['premise']),
        'hypothesis': scramble_text(example['hypothesis']),
        'label': example.get('label'),
    }


def scramble_examples(examples: list[dict]) -> list[dict]:
    """Scramble words in premise and hypothesis for a list of examples."""
    return [scramble_example(ex) for ex in examples]


def sample_few_shot_examples(seed_id: str, grouped_data: dict, n: int) -> list[dict]:
    """
    Sample n examples for a given seed id and map labels to letter format.
    """
    examples = random.sample(grouped_data[seed_id], n)
    mapped = []
    for ex in examples:
        ex = ex.copy()
        ex['label'] = LABEL_TO_LETTER[ex['label'].lower()]
        mapped.append(ex)
    return mapped


def build_replace_prompt(example: dict, prompts_mix: str, grouped_data: dict, n: int) -> str:
    """
    Build a prompt by replacing one few-shot example with a variant
    that shares the same label, drawn from the seed's variants.
    """
    few_shot_list = parse_prompt_string(prompts_mix)
    new_examples = sample_few_shot_examples(example['id'], grouped_data, n)

    replacement_idx = None
    replacement_example = None

    for new_ex in new_examples:
        for idx, old_ex in enumerate(few_shot_list):
            if old_ex['label'] == new_ex['label']:
                replacement_idx = idx
                replacement_example = new_ex
                break
        if replacement_idx is not None:
            break

    few_shot_list[replacement_idx] = replacement_example

    label_counts = Counter(ex['label'] for ex in few_shot_list)
    if any(count > 2 for count in label_counts.values()):
        print('WARNING: imbalanced labels in few-shot prompt')

    return ensure_double_newline(format_few_shot_examples(few_shot_list))


def build_inference_prompt(example: dict, grouped_data: dict, n: int) -> str:
    """Build a prompt by sampling n examples fresh from the seed's variants."""
    few_shot = sample_few_shot_examples(example['id'], grouped_data, n)
    if len(few_shot) != n:
        print('WARNING: fewer few-shot examples than requested')
    return ensure_double_newline(format_few_shot_examples(few_shot))


def build_self_consistency_prompts(example: dict, grouped_data: dict, n: int, k: int = 2) -> list[str]:
    """
    Build k independently sampled few-shot prompts for self-consistency voting.
    """
    return [
        ensure_double_newline(format_few_shot_examples(
            sample_few_shot_examples(example['id'], grouped_data, n)
        ))
        for _ in range(k)
    ]


def build_scrambled_prompt(prompts_mix: str) -> str:
    """Build a prompt by scrambling the words in all few-shot examples."""
    few_shot_list = parse_prompt_string(prompts_mix)
    return ensure_double_newline(format_few_shot_examples(scramble_examples(few_shot_list)))


def build_example(example, type_prompting_build):
  if type_prompting_build=='thinking_token':
    return

def build_prompts(
    batch: list[dict],
    prompts_mix: str,
    type_prompting_build: str | None,
    grouped_seed_data: dict,
    number_few_shot: int,
) -> list[str]:
    """
    Build a list of prompts for a batch of examples.

    Args:
        batch: List of test examples.
        prompts_mix: Raw string of fixed few-shot examples.
        type_prompting_build: One of 'replace', 'inference', 'self-consistency',
                              'scrambled', or None (use prompts_mix as-is).
        grouped_seed_data: Dict mapping seed id to list of variant examples.
        number_few_shot: Number of few-shot examples to include.

    Returns:
        List of prompt strings ready for inference.
    """
    prompts = []

    for example in batch:
        if type_prompting_build == 'replace':
            few_shot = build_replace_prompt(example, prompts_mix, grouped_seed_data, number_few_shot)

        elif type_prompting_build == 'inference':
            few_shot = build_inference_prompt(example, grouped_seed_data, number_few_shot)

        elif type_prompting_build == 'self-consistency':
            few_shot = build_self_consistency_prompts(example, grouped_seed_data, number_few_shot)

        elif type_prompting_build == 'scrambled':
            few_shot = build_scrambled_prompt(prompts_mix)

        elif type_prompting_build is None:
            few_shot = ensure_double_newline(prompts_mix)

        else:
            raise ValueError(f"Unknown prompting type: '{type_prompting_build}'")
        add_test_example(prompts, few_shot, example)
    return prompts
def sample_k_per_label(snli_split, k=3, seed=42):
    list_smaples={}
    random.seed(seed)

    by_label = defaultdict(list)
    for ex_id, ex in snli_split.items():
        by_label[ex['g']].append(ex_id)

    sampled_ids = {
        label: random.sample(ids, k)
        for label, ids in by_label.items()
    }
    for ex_id, ex in snli_split.items():
        if ex_id in sampled_ids[ex['g']]:
            list_smaples[ex_id]={'Premise': ex["p"], 'Hypothesis': ex["h"], 'Answer': ex["g"]}
    return list_smaples, sampled_ids
