from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from typing import Dict, Callable, List, Tuple, List, Any
import nltk, re, os, json, string, torch, glob, sys, torch, random, numpy as np, pandas as pd
from nltk import pos_tag, word_tokenize
from tqdm import tqdm
from collections import defaultdict, Counter
from datasets import load_dataset
from typing import Dict,Tuple
import datasets, evaluate
from evaluate import evaluator
from google.colab import files
import torch.nn.functional as F
from contextlib import redirect_stdout
from wordfreq import zipf_frequency
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statistics import mean

def extract__pos_position(pos_tags, tokens, source, pos_type, sentence):
  #not modified for optimality
    pos_tag_map = {
        'noun': {'NN', 'NNS'},
        'verb': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
        'adjective': {'JJ'},
        'adverb': {'RB'},
        'merged_n_a': {'NN', 'NNS', 'JJ'},
        'merged_v_n': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS'},
        'merged_v_a': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB'},
        'merged_v_a_n': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'NN', 'NNS'}
    }

    ignore, dictionary_positions, token_counts = [], {}, defaultdict(int)
    valid_tags = pos_tag_map.get(pos_type, set())
    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
        if pos not in valid_tags or token in ignore:
          continue

        pattern = r'\b' + re.escape(token) + r'\b'
        matches = list(re.finditer(pattern, sentence))
        if not matches:
            continue
        token_counts[token] += 1
        occurrence_index = token_counts[token]
        matchy=matches[occurrence_index - 1]
        preceding_text = sentence[:matchy.start()]
        preceding_length = len(preceding_text)
        start = preceding_length
        end = preceding_length + len(token)
        offset = (start, end)
        if token not in dictionary_positions:
            dictionary_positions[token] = {'positions': [offset], 'pos': pos, 'source': source, 'preceding_text': preceding_text}
        else:
            dictionary_positions[token]['positions'].append(offset)
    return dictionary_positions



def generate_mask_predictions(model, tokenizer, context, mask_token, target_word=None, top_k=50):
    #not modified
    """
    Generate predictions with detailed token debugging.

    Args:
        model: The masked language model
        tokenizer: The corresponding tokenizer
        context: The input string containing the mask token
        mask_token: The mask token (e.g., <mask> or [MASK])
        target_word: The word whose probability we want to retrieve (optional)
        top_k: The number of predictions to return
        debug: Whether to print debugging information
    """
    if mask_token not in context:
        raise ValueError(f"Context must contain the mask token: {mask_token}")
    inputs = tokenizer(context, return_tensors="pt")
    mask_positions = torch.where(inputs.input_ids[0] == tokenizer.mask_token_id)[0]
    if len(mask_positions) == 0:
        raise ValueError("Mask token not found in the input context.")
    mask_position = mask_positions.item()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits[0, mask_position]
        probabilities = F.softmax(predictions, dim=-1)
    target_probability = None
    target_tokens = tokenizer(target_word, add_special_tokens=False)['input_ids']
    if len(target_tokens) > 1:
        target_probability='None'
    else:
      target_token_id = target_tokens[0]
      target_probability = probabilities[target_token_id].item()
    top_probs, top_indices = torch.topk(probabilities, top_k)
    prediction_list = []
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.convert_ids_to_tokens(int(idx))
        prediction_list.append({
            "score": prob.item(),
            "token_str": token
        })
    return prediction_list, target_probability



def suggest_mask_fillers(input_str:str, mask_offsets: List[Tuple[int,int]],
                         model, tokenizer, all_single_words, common_tokens, suggestion_n=50):
                             #not double-checked
    """ mask_offsets is a list of integer pairs that mark the part of teh string input taht needs to be masked.
        It is a list because in general it might be needed to mask several parts of the input string.
        Returns a dictionary with character offsets as keys and a list of ranked suggestions as values.
    """
    #not modified for optimality
    model_architecture = model.config.architectures[0].lower()
    mask_token = "<mask>" if model_architecture == "roberta" else "[MASK]"
    suggestions, all_tuples = {},[]
    for w in mask_offsets:
      if len(w)>1:
        for i in w:
          all_tuples.append(i)
      else:
        all_tuples.append(w[0])
    mask_offsets=all_tuples
    for i, j in mask_offsets:
      masked_token_orig = input_str[i:j]
      offset_key = str(i)+':'+str(j)
      if masked_token_orig in common_tokens:
          pos_tag = common_tokens[masked_token_orig].get('pos', 'UNK')
      else:
          pos_tag = "UNK"
      candidate_list = []
      masked_input = input_str[:i] + f'{mask_token}' + input_str[j:]
      if masked_input.endswith(mask_token):
          masked_input += '.'
      if mask_token == '<mask>' and not masked_input.startswith('<mask>'):
        masked_token_orig=' '+masked_token_orig
        if masked_input.startswith('<mask>'):
          print('the mask that is fed to the model for probability when it is the first tokem', masked_token_orig)
      generated, probability_masked_word = generate_mask_predictions(model, tokenizer, masked_input, mask_token, masked_token_orig, suggestion_n)
      if mask_token == '<mask>' and not masked_input.startswith('<mask>'):
        masked_token_orig=masked_token_orig.strip()
      token_key=f"{masked_token_orig}:{pos_tag}"
      if probability_masked_word=='None':
        offset_key = f"{offset_key}:{probability_masked_word}"
      else:
        offset_key = f"{offset_key}:{probability_masked_word:.2e}"
      for k in generated:
          token = k['token_str'].lstrip()
          token_1=token.strip('Ġ')
          candidate_list.append(f"{token_1}:{k['score']:.2e}")
      if len(candidate_list) != suggestion_n:
          print(f"\nWarning: Expected {suggestion_n} suggestions but got {len(candidate_list)}")
          print(f"Input string: {input_str}")
      if token_key not in suggestions:
          suggestions[token_key] = {}
      if offset_key not in suggestions[token_key]:
        suggestions[token_key][offset_key] = {}
        suggestions[token_key][offset_key] = candidate_list
      else:
        suggestions[token_key][offset_key].extend(candidate_list)
    return suggestions

def extract_nouns_and_verbs(pos_tags, tokens, pos_type):
        noun_tags = {'NN', 'NNS'}
        verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        adjective_tags = {'JJ'}
        adverb_tags = {'RB'}
        if pos_type == 'noun':
            return {tokens[i] for i, pos in enumerate(pos_tags) if pos in noun_tags}
        elif pos_type == 'verb':
            return {
                "verbs": {tokens[i] for i, pos in enumerate(pos_tags) if pos in verb_tags}
            }
        elif pos_type == 'adjective':
            return {tokens[i] for i, pos in enumerate(pos_tags) if pos in adjective_tags}
        elif pos_type == 'adverb':
            return {tokens[i] for i, pos in enumerate(pos_tags) if pos in adverb_tags}
        else:
            raise ValueError("Invalid pos_type. Choose 'noun', 'verb', 'adjective', 'adverb', 'merged_n_a', 'merged_v_n', 'merged_v_a', or ' 'merged_v_a_n''.")



def flatten_extracted_words(extracted):
  if isinstance(extracted, set):
      return extracted
  elif isinstance(extracted, dict):
    if all(extracted.values()):
      return set().union(*extracted.values())  #
    else:
        return set()
  return set()

def filter_snli(dataset, mapping, pos_to_mask, min_common_words, num_sentences_to_process, max_filtered_count=None, annotators_agreement_number=int, length_premise=int, length_hypothesis=int):
    '''function to extract sentences that adhere to certain criteria, can't be used for combined pos tags'''
    filtered, count = {}, 0
    dataset_items = list(dataset.items())[:num_sentences_to_process] if num_sentences_to_process else dataset.items()
    for k, p in dataset_items:
        # print(p)
        if p['g'] != '-' and len(p['lcnt']) < annotators_agreement_number and len(p['p'].split()) >= length_premise and len(p['h'].split()) >= length_hypothesis: #
            common_words = (
                flatten_extracted_words(extract_nouns_and_verbs(mapping[p['p']]['pos'], mapping[p['p']]['tok'], pos_to_mask))
                &
                flatten_extracted_words(extract_nouns_and_verbs(mapping[p['h']]['pos'], mapping[p['h']]['tok'], pos_to_mask))
            )
            if len(common_words) >= min_common_words:
                filtered[k] = p
                count += 1
                if max_filtered_count and count >= max_filtered_count:
                    break
    return filtered


def process_unmasked_dataset(filtered_list_1, neutral_number, entailment_number, contradiction_number, id):
  #filtered_list_1 structure  [{'id': f"{base_id}{version}",
            # 'label': p['g'],
            # 'premise': p['p'],
            # 'hypothesis': p['h'],
            # 'p_p': mapping[p['p']]['pos'],
            # 'p_t': mapping[p['p']]['tok'],
            # 'h_p': mapping[p['h']]['pos'],
            # 'h_t': mapping[p['h']]['tok']},]
  new_list4 = []
  label_counts = {'contradiction': 0, 'entailment': 0, 'neutral': 0}
  for i in tqdm(filtered_list_1):
      label = i['label']
      if id=='yes':
        new_list4.append({
            'id':i['id'],
            'premise': i['premise'],
            'hypothesis': i['hypothesis'],
            'label': {'neutral': neutral_number, 'entailment': entailment_number, 'contradiction':contradiction_number}[label]
        })
        label_counts[label] += 1
      if id =='no':
        new_list4.append({
            'premise': i['premise'],
            'hypothesis': i['hypothesis'],
            'label': {'neutral': neutral_number, 'entailment': entailment_number, 'contradiction':contradiction_number}[label]
        })
        label_counts[label] += 1
  print("Label counts:", label_counts)
  return new_list4, label_counts

def pos_toks_extract_from_dataset(list_filtered, mapping):


  ### list_filtered structure {'3827316480.jpg#0r1e': {'g': 'entailment', 'pid': '3827316480.jpg#0r1e', 'cid': '3827316480.jpg#0', 'lnum': 5, 'lcnt': Counter({'entailment': 5}), 'ltype': '500', 'p': 'One tan girl with a wool hat is running and leaning over an object, while another person in a wool hat is sitting on the ground.', 'h': 'A tan girl runs leans over an object'},
  ### grouped_problems defaultdict(<class 'dict'>, {'3827316480.jpg#0r1': {'e': {'g': 'entailment'
    #the code groups first by last letter and then againcombines it bc before i was making sure there are all 3 letters there, but now it is harmless and
  filtered_list_1 = []
  grouped_problems = defaultdict(dict)
  for k, p in list_filtered.items(): ########this is not necessary but i will leave it as it is for now
      base_id = k[:-1]
      version = k[-1]
      grouped_problems[base_id][version] = p
  for base_id, versions in grouped_problems.items():
      for version, p in versions.items():
        filtered_list_1.append({
            'id': f"{base_id}{version}",
            'label': p['g'],
            'premise': p['p'],
            'hypothesis': p['h'],
            'p_p': mapping[p['p']]['pos'],
            'p_t': mapping[p['p']]['tok'],
            'h_p': mapping[p['h']]['pos'],
            'h_t': mapping[p['h']]['tok']
        })
  print(f"no. problems filtered after criteria: {len(filtered_list_1)}")
  return filtered_list_1


def is_sentence_fully_processed(sentence, filler_data, common_tokens_dictionary):
  if sentence not in filler_data:
      return False
  existing_keys = set(filler_data[sentence].keys())
  required_keys = {f"{token}:{data['pos']}" for token, data in common_tokens_dictionary.items()}
  return required_keys.issubset(existing_keys)

def common(sentence1, sentence2, pos_sent_1, pos_sent_2, toks_sent_1, toks_sent_2, pos_type, source_1, source_2, singles='yes'):
    ##
            # extracted_1 {'black': {'positions': [(11, 16)], 'pos': 'JJ', 'source': 'premise', 'preceding_text': 'A man in a '}, 'commercial': {'positions': [(29, 39)], 'pos': 'JJ', 'source': 'premise', 'preceding_text': 'A man in a black shirt, in a '}}
            # extracted_2 {'black': {'positions': [(13, 18)], 'pos': 'JJ', 'source': 'hypthesis', 'preceding_text': 'A woman in a '}, 'commercial': {'positions': [(31, 41)], 'pos': 'JJ', 'source': 'hypthesis', 'preceding_text': 'A woman in a black shirt, in a '}}
            # common tokens {'black', 'commercial'}
            # common dict {'black': {'positions': [(11, 16)], 'pos': 'JJ', 'source': 'premise', 'preceding_text': 'A man in a '}, 'commercial': {'positions': [(29, 39)], 'pos': 'JJ', 'source': 'premise', 'preceding_text': 'A man in a black shirt, in a '}}
            # mask positions 1 [[(11, 16)], [(29, 39)]]
            # mask positions 2 [[(13, 18)], [(31, 41)]]

    extracted_1 = extract__pos_position(pos_sent_1, toks_sent_1, source_1, pos_type, sentence1)
    extracted_2 = extract__pos_position(pos_sent_2, toks_sent_2, source_2, pos_type, sentence2)
    common_tokens = set(extracted_1.keys()) & set(extracted_2.keys())
    common_dict = {token: extracted_1[token] for token in common_tokens}
    all_nouns_singles = {' ' + k for d in [extracted_1, extracted_2] for k, v in d.items()} if singles=='yes' else None
    mask_positions_1 = [extracted_1[token]["positions"] for token in common_tokens]
    mask_positions_2 = [extracted_2[token]["positions"] for token in common_tokens]
    return common_dict, mask_positions_1, mask_positions_2, all_nouns_singles


def suggest_mask_fillers(input_str:str, mask_offsets: List[Tuple[int,int]],
                         model, tokenizer, all_single_words, common_tokens, suggestion_n=50):
                             #not double-checked
    """ mask_offsets is a list of integer pairs that mark the part of teh string input taht needs to be masked.
        It is a list because in general it might be needed to mask several parts of the input string.
        Returns a dictionary with character offsets as keys and a list of ranked suggestions as values.
    """
    model_architecture = getattr(model.config, "architectures", None)
    model_architecture = model_architecture[0].lower() if model_architecture else model.config.model_type.lower()
    mask_token = "<mask>" if model_architecture == "roberta" else "[MASK]"
    suggestions, all_tuples=  {}, []
    for w in mask_offsets:
      if len(w)>1:
        for i in w:
          all_tuples.append(i)
      else:
        all_tuples.append(w[0])
    mask_offsets=all_tuples
    for i, j in mask_offsets:
      masked_token_orig = input_str[i:j]
      offset_key = str(i)+':'+str(j)
      if masked_token_orig in common_tokens:
          pos_tag = common_tokens[masked_token_orig].get('pos', 'UNK')
      else:
          pos_tag = "UNK"
      candidate_list = []
      masked_input = input_str[:i] + f'{mask_token}' + input_str[j:]
      if masked_input.endswith(mask_token):
          masked_input += '.'
      if mask_token == '<mask>' and not masked_input.startswith('<mask>'):
        masked_token_orig=' '+masked_token_orig
        if masked_input.startswith('<mask>'):
          print('the mask that is fed to the model for probability when it is the first tokem', masked_token_orig)
      generated, probability_masked_word = generate_mask_predictions(model, tokenizer, masked_input, mask_token, masked_token_orig, suggestion_n)
      if mask_token == '<mask>' and not masked_input.startswith('<mask>'):
        masked_token_orig=masked_token_orig.strip()
      token_key=f"{masked_token_orig}:{pos_tag}"
      if probability_masked_word=='None':
        offset_key = f"{offset_key}:{probability_masked_word}"
      else:
        offset_key = f"{offset_key}:{probability_masked_word:.2e}"
      for k in generated:
          token = k['token_str'].lstrip()
          token_1=token.strip('Ġ')
          candidate_list.append(f"{token_1}:{k['score']:.2e}")
      if len(candidate_list) != suggestion_n:
          print(f"\nWarning: Expected {suggestion_n} suggestions but got {len(candidate_list)}")
          print(f"Input string: {input_str}")
      if token_key not in suggestions:
          suggestions[token_key] = {}
      if offset_key not in suggestions[token_key]:
        suggestions[token_key][offset_key] = {}
        suggestions[token_key][offset_key] = candidate_list
      else:
        suggestions[token_key][offset_key].extend(candidate_list)
    return suggestions


#already_exsiting_words> exclude_words_part_of_p_and_h
def create_filler_file(
    model_name: str,
    dataset: pd.DataFrame,
    split: str,
    pos_to_mask: str,
    min_common_words: int,
    num_filler_suggestions: int,
    source_1: str,
    source_2: str,
    mapping,
    exclude_words_part_of_p_and_h: str,
    no_neutral,
    no_contradiction,
    no_ential,
    number_of_labels,
    number_words_premise,
    number_words_hypothesis,
    num_sentences_to_process_dataset: int = None,
    num_sentences_compliant_criteria: int = None,
    mock_test: bool = False,
    add_id_to_dataset: bool = False,
    output_file: str = None,
):
#not double-checked
    """
        Function generating a new inflated dataset with suggestion from a language model, alongside the initial split of sentences modified

        model_name : str // name of language model used for generating masked token predictions (e.g., "bert-base-uncased").
        dataset : Dataset // The dataset to process
        split : str // The dataset split to use (e.g., "train", "test", "validation").
        pos_to_mask : str // a str indicating what pos to be masked in sentences ('noun' or 'verb'), and what sentences to be picked for masking considering trhe min_common_words (e.g. 3 common nouns)
        min_common_words : int// the minimum number of common words required between premise and hypothesis
        num_filler_suggestions : int// The number of suggested filler words for each masked token by model
        source_1: name of the first sentence in dataset
        source_2: name of the second sentence in dataset
        mapping: mapping function
        exclude_words_part_of_p_and_h: if we want to exclude or not 3xclude alreadye xisting words
        num_sentences_to_process_dataset : int /// The number of sentences to process from the dataset.
        num_sentences_compliant_criteria : int // argument that sopecifies after how many sentences compliant to the crteria to select
        mock_test: if yes it will do the generation for one sentence that has a word with 2 occurances
        add_id_to_dataset: if yes it will add an id to the dataset
        output_file : str /// file name where the masked dataset will be saved
        #returns the list of processed sentences with ids and a sepearte file with the suggestions
    """

    label_counts = {'contradiction': 0, 'entailment': 0, 'neutral': 0}
    new_list4 = []

    dataset = dataset[split]
    SNLI_filtered_2 = filter_snli(dataset, mapping, pos_to_mask, min_common_words,
                                  num_sentences_to_process_dataset, num_sentences_compliant_criteria, number_of_labels, number_words_premise, number_words_hypothesis)

    filtered_list_1 = pos_toks_extract_from_dataset(SNLI_filtered_2, mapping)
    seed_dataset, lab = process_unmasked_dataset(filtered_list_1, no_neutral, no_ential, no_contradiction, id=add_id_to_dataset)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results_dict = {}
    for p in tqdm(filtered_list_1):
        id, premise, hypothesis, tok_p, pos_p, tok_h, pos_h = (p['id'], p['premise'], p['hypothesis'], p['p_t'], p['p_p'], p['h_t'], p['h_p'] )
        if mock_test and id != "3827316480.jpg#0r1e": 
          continue
        common_tokens_dictionary, p_off, h_off, all_nouns_singles = common(
            premise, hypothesis, pos_p, pos_h, tok_p, tok_h, pos_to_mask, source_1, source_2, exclude_words_part_of_p_and_h
        )
        p_off_filler = suggest_mask_fillers(premise, p_off, model, tokenizer, all_nouns_singles, common_tokens_dictionary, num_filler_suggestions)
        if p_off_filler:
          if premise in results_dict:
              results_dict[premise].update(p_off_filler)
          else:
              results_dict[premise] = p_off_filler
        h_off_filler = suggest_mask_fillers(hypothesis, h_off, model, tokenizer, all_nouns_singles, common_tokens_dictionary, num_filler_suggestions)
        if h_off_filler:
          if hypothesis in results_dict:
            results_dict[hypothesis].update(h_off_filler)
          else:
            results_dict[hypothesis] = h_off_filler
    with open(output_file, "w") as f:
        json.dump(results_dict, f)
    return seed_dataset
