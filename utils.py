from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from typing import Dict, Callable, List, Tuple, List, Any
import nltk
import re
import numpy as np
from nltk import pos_tag, word_tokenize
from tqdm import tqdm
import pandas as pd
import json
from collections import defaultdict, Counter
import datasets
from datasets import load_dataset
import json
import string
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from typing import Dict,Tuple
import datasets, evaluate
from evaluate import evaluator
from google.colab import files
import torch
import torch.nn.functional as F



def extract__pos_position(pos_tags, tokens, source, pos_type, sentence):
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

    ignore=[]
    valid_tags = pos_tag_map.get(pos_type, set())
    dictionary_positions = {}
    token_counts = defaultdict(int)
    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
        if pos not in valid_tags:
            continue
        if token in ignore:
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
                         model, tokenizer, all_single_words, common_tokens, suggestion_n=50) -> Dict[Tuple[int,int], List[str]]:
    """ mask_offsets is a list of integer pairs that mark the part of teh string input taht needs to be masked.
        It is a list because in general it might be needed to mask several parts of the input string.
        Returns a dictionary with character offsets as keys and a list of ranked suggestions as values.
    """
    model_architecture = model.config.architectures[0].lower()
    if 'bert' in model_architecture and 'roberta' not in model_architecture:
      mask_token = '[MASK]'
    else:
      mask_token = '<mask>'
    suggestions = {}
    all_tuples=[]
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
        elif pos_type == 'merged_n_a':
            merged_dict = {
            "nouns": {tokens[i] for i, pos in enumerate(pos_tags) if pos in noun_tags},
            "adjectives": {tokens[i] for i, pos in enumerate(pos_tags) if pos in adjective_tags}
          }
            return merged_dict
        elif pos_type == 'merged_v_n':  # Verbs + Nouns
          merged_dict = {
              "verbs": {tokens[i] for i, pos in enumerate(pos_tags) if pos in verb_tags},
              "nouns": {tokens[i] for i, pos in enumerate(pos_tags) if pos in noun_tags}
          }
          return merged_dict
        elif pos_type == 'merged_v_a':  # Verbs + Adverbs
            merged_dict = {
                "verbs": {tokens[i] for i, pos in enumerate(pos_tags) if pos in verb_tags},  # Fixed to include verbs
                "adverbs": {tokens[i] for i, pos in enumerate(pos_tags) if pos in adverb_tags}
            }
            return  merged_dict
        elif pos_type == 'merged_v_a_n':  # Verbs + Adverbs + Nouns
            merged_dict = {
                "verbs": {tokens[i] for i, pos in enumerate(pos_tags) if pos in verb_tags},
                "adverbs": {tokens[i] for i, pos in enumerate(pos_tags) if pos in adverb_tags},
                "nouns": {tokens[i] for i, pos in enumerate(pos_tags) if pos in noun_tags}
            }
            return merged_dict
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
    filtered = {}
    count = 0
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


def process_unmasked_dataset(filtered_list_1, neutral_number, entailment_number, contradiction_number, id) -> List[Dict]:
  new_list4 = []
  label_counts = {'contradiction': 0, 'entailment': 0, 'neutral': 0}
  if isinstance(filtered_list_1, dict):
        filtered_list_1 = [
            {"label": p["g"], "premise": p["p"], "hypothesis": p["h"]}
            for k, p in filtered_list_1.items()
        ]

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
      filtered_list_1 = []
      grouped_problems = defaultdict(dict)
      for k, p in list_filtered.items():
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

    extracted_1 = extract__pos_position(pos_sent_1, toks_sent_1, source_1, pos_type, sentence1)
    extracted_2 = extract__pos_position(pos_sent_2, toks_sent_2, source_2, pos_type, sentence2)
    common_tokens = set(extracted_1.keys()) & set(extracted_2.keys())
    common_dict = {token: extracted_1[token] for token in common_tokens}
    all_nouns_singles = {' ' + k for d in [extracted_1, extracted_2] for k, v in d.items()} if singles=='yes' else None
    mask_positions_1 = [extracted_1[token]["positions"] for token in common_tokens]
    mask_positions_2 = [extracted_2[token]["positions"] for token in common_tokens]
    return common_dict, mask_positions_1, mask_positions_2, all_nouns_singles

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
    already_exsiting_words: str,
    no_neutral,
    no_contradiction,
    no_ential,
    number_of_labels,
    number_words_premise,
    number_words_hypothesis,
    num_sentences_to_process_dataset: int = None,
    num_sentences_compliant_criteria: int = None,
    output_file: str = None,
) -> List[Dict]:

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
        already_exsiting_words: if we want to exclude or not 3xclude alreadye xisting words
        num_sentences_to_process_dataset : int /// The number of sentences to process from the dataset.
        num_sentences_compliant_criteria : int // argument that sopecifies after how many sentences compliant to the crteria to select
        output_format : str // The format of the output file
        output_file : str /// file name where the masked dataset will be saved
        #returns the list of processed sentences with ids and a sepearte file with the suggestions
    """

    label_counts = {'contradiction': 0, 'entailment': 0, 'neutral': 0}
    new_list4 = []

    dataset = dataset[split]
    SNLI_filtered_2 = filter_snli(dataset, mapping, pos_to_mask, min_common_words,
                                  num_sentences_to_process_dataset, num_sentences_compliant_criteria, number_of_labels, number_words_premise, number_words_hypothesis)

    filtered_list_1 = pos_toks_extract_from_dataset(SNLI_filtered_2, mapping)
    new_list3, lab = process_unmasked_dataset(filtered_list_1, no_neutral, no_ential, no_contradiction, id='yes')
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results_dict = {}
    for p in tqdm(filtered_list_1):
        id, premise, hypothesis, tok_p, pos_p, tok_h, pos_h = (p['id'], p['premise'], p['hypothesis'], p['p_t'], p['p_p'], p['h_t'], p['h_p'] )
        common_tokens_dictionary, p_off, h_off, all_nouns_singles = common(
            premise, hypothesis, pos_p, pos_h, tok_p, tok_h, pos_to_mask, source_1, source_2, already_exsiting_words
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
    return new_list3

def process_and_save_dataset(result, output_file, neutral_number, entailment_number, contradiction_number, id='yes'):
    """
   maps labels to numbers
    """
    processed = process_unmasked_dataset(
        result,
        neutral_number=neutral_number,
        entailment_number=entailment_number,
        contradiction_number=contradiction_number,
        id=id
    )
    with open(output_file, "w") as f:
        json.dump(processed, f, indent=4)
    try:
        # files.download(output_file)
        return processed
    except NameError:
        print("Skipping file download (not running in Google Colab)")



def filter_candidates(candidates, all_singles=None, excluded_words=None):
    """
   filter out unwatned words from json file
    """
    if excluded_words is None:
        excluded_words = {
            "n't", "not", "no", "never", "neither", "none", "nowise",
            "nothing", "nobody", "nowhere", "non", "absent", "lacking",
            "minus", "without", "'s", "'n'", "'re", "'m"
        }

    filtered = []

    for candidate in candidates:
        # print('THE CANDIDATE', candidate)
        word = candidate.split(":")[0]
        if word.startswith('##'):
            continue

        if all(char in string.punctuation for char in word):
            continue

        if word in excluded_words:
            continue

        if all_singles is not None:
            word_with_space = ' ' + word

            word_lower = word.lower()
            word_lower_with_space = ' ' + word_lower
            word_stripped = word.strip()
            word_stripped_lower = word_stripped.lower()
            word_stripped_with_space = ' ' + word_stripped
            word_stripped_lower_with_space = ' ' + word_stripped_lower

            if (word in all_singles or
                word_with_space in all_singles or
                word_lower in all_singles or
                word_lower_with_space in all_singles or
                word_stripped in all_singles or
                word_stripped_with_space in all_singles or
                word_stripped_lower in all_singles or
                word_stripped_lower_with_space in all_singles):
                continue
        filtered.append(candidate)

    return filtered


def ranked_overlap(list_of_lists, probs):
    n = len(list_of_lists)
    s = set(list_of_lists[0]).intersection(*map(set, list_of_lists[1:]))
    s_ranks = dict()
    for element in s:
        ranks = [ l.index(element) for l in list_of_lists ]
        probs1=[z[l.index(element)] for l, z in zip (list_of_lists, probs) if element in l]
        avg_prob=sum(probs1)/n
        s_ranks[element] = {
            'average_rank': sum(ranks)/n,
            'ranks' :ranks,
            'average_prob': f"{avg_prob:.2e}",
            "individual_probs": [f"{p:.2e}" for p in probs1]}
    return s_ranks

def process_dataset(first_data, initial_dataset,split, output_file, min_common_words, mapping, ranked_overlap, pos_to_mask, neutral_number, source_1, source_2, entailment_number, contradiction_number, an_no, prem_n, hypo_n, number_of_minimal_suggestions, rank_option='top', sort_by_pos='no', id='no', num_sentences_to_process_dataset: int = None, num_sentences_compliant_criteria: int = None,  debug:str=None):
    """
    Matches premise and hypothesis from second_data with first_data, replaces words, applies ranking,
    transforms the dataset, and optionally groups it by POS tags.

    :first_data: The dataset containing suggestions
    :second_data: The dataset containing 'id', 'premise', 'hypothesis', and 'label'.
    :ranked_overlap: The function that ranks words based on probability.
    :neutral_number: number for neutral label
    :entailment_number: number for entailment label
    :contradiction_number: number for contradiction label
    :rank_option: 'top' for highest-ranked, int for specific rank, slice for multiple replacements.
    :sort_by_pos: 'yes' to group the dataset by POS tags.
    :id: 'yes' to process a first_data file with masked suggestions that were recorded as sentence:id
    :return: Processed dataset with replaced words and transformed labels.
    """

    processed_data = []
    pos_tagged_data = defaultdict(list)
    dataset = initial_dataset[split]
    SNLI_filtered_2 = filter_snli(dataset, mapping, pos_to_mask, min_common_words,
                                  num_sentences_to_process_dataset, num_sentences_compliant_criteria, an_no, prem_n, hypo_n)
    processed_second_data = pos_toks_extract_from_dataset(SNLI_filtered_2, mapping)
    new_list3, labels_sample = process_unmasked_dataset(processed_second_data, neutral_number, entailment_number, contradiction_number, id='yes')
    expected_generation = {'neutral': 0, 'entailment': 0, 'contradiction': 0}


    actual_generation = {'neutral': 0, 'entailment': 0, 'contradiction': 0}
    for entry in tqdm(processed_second_data):
        id, premise, hypothesis, tok_p, pos_p, tok_h, pos_h, label = (entry['id'], entry['premise'], entry['hypothesis'], entry['p_t'],entry['p_p'], entry['h_t'], entry['h_p'], entry['label'] )


        if id =='yes':
          premise_id = f"{premise}:{id}"
          hypothesis_id = f"{hypothesis}:{id}"
        else:
          premise_id = premise
          hypothesis_id = hypothesis


        word2fillers = defaultdict(list)
        word2probabilities = defaultdict(list)
        word2pos = defaultdict(list)
        token_counts = defaultdict(int)
        offsets_for_tokens_premise=defaultdict(list)
        offsets_for_tokens_hypothesis=defaultdict(list)

        if premise_id in first_data.keys() and hypothesis_id in first_data.keys():

          common_dict, p_positions, h_positions, singles = common(premise, hypothesis, pos_p, pos_h, tok_p, tok_h, pos_to_mask, source_1, source_2)
          if debug=='yes':
            print('premise', premise)
            print('hypot', hypothesis)
            print('common dict', common_dict)
            print('p_positions', p_positions)
            print('h_positions', h_positions)

          words = list(common_dict.keys())
          for i, word in enumerate(words):
            pos = common_dict[word]['pos']
            word_with_pos = f"{word}:{pos}"
            if debug=='yes':
              print('word_with_pos', word_with_pos)
            premise_pos_list = p_positions[i]
            hypothesis_pos_list = h_positions[i]

            for j, (p_pos, h_pos) in enumerate(zip(premise_pos_list, hypothesis_pos_list)):
                p_start, p_end = p_pos
                key_prefix_p = f"{p_start}:{p_end}:"
                h_start, h_end = h_pos
                key_prefix_h = f"{h_start}:{h_end}:"
                key = f"{word_with_pos}:{p_start}:{p_end}:{h_start}:{h_end}"
                premise_data=first_data[premise]
                hypothesis_data=first_data[hypothesis]
                if debug=='yes':
                  print('premise_data', premise_data)
                  print('hypothesis_data', hypothesis_data)

                matching_key_p = next(
                    (k for k in premise_data[word_with_pos].keys() if k.startswith(key_prefix_p)),
                    None
                    )

                matching_key_h = next(
                    (k for k in hypothesis_data[word_with_pos].keys() if k.startswith(key_prefix_h)),
                    None
                    )
                premise_suggestions = premise_data[word_with_pos][matching_key_p]
                hypothesis_suggestions = hypothesis_data[word_with_pos][matching_key_h]
                if debug=='yes':
                  print('length of premise suggestions', len(premise_suggestions))
                  print('length of hypothesis suggestions', len(hypothesis_suggestions))
                  print('premise_suggestions', premise_suggestions)
                  print('hypothesis_suggestions', hypothesis_suggestions)

                premise_cleaned=filter_candidates(premise_suggestions, singles)
                hypothesis_cleaned= filter_candidates(hypothesis_suggestions, singles)
                if debug=='yes':
                  print('length of premise cleaned', len(premise_cleaned))
                  print('length of hypothesis cleaned', len(hypothesis_cleaned) )
                  print('premise_cleaned', premise_cleaned)
                  print('hypothesis_cleaned', hypothesis_cleaned)

                premise_fillers= [c.split(":")[0] for c in premise_cleaned]
                hypothesis_fillers= [c.split(":")[0] for c in hypothesis_cleaned]

                premise_keys = set(premise_fillers)
                hypothesis_keys = set(hypothesis_fillers)
                common_words = premise_keys & hypothesis_keys
                if len(common_words) < number_of_minimal_suggestions:
                  if debug=='yes':
                    print(word_with_pos)
                    print('the id', id)
                    print('premise', premise)
                    print('hypo', hypothesis)
                    print('premise_fillers', premise_fillers)
                    print('hypothesis_fillers', hypothesis_fillers)
                  continue

                premise_probabilities = [float(c.split(":")[1]) for c in premise_cleaned]
                hypothesis_probabilities = [float(c.split(":")[1]) for c in hypothesis_cleaned]
                if debug=='yes':
                  print('premise_probabilities', premise_probabilities)
                  print('hypothesis_probabilities', hypothesis_probabilities)
                word2fillers[key] = [premise_fillers, hypothesis_fillers]
                word2probabilities[key] = [premise_probabilities, hypothesis_probabilities]
                word2pos[key] = [pos, pos]

        if word2fillers:
          words = {}
          if debug=='yes':
            print('word2fillers', word2fillers)
          for w in word2fillers:
              words[w] = ranked_overlap(word2fillers[w], word2probabilities[w]).items()
              words[w] = sorted(words[w], key=lambda x: x[1]["average_rank"])

          if debug=='yes':
            print('the words',words)
          assigned_pos_tags = set()

          for w, ranked_fillers in words.items():
              if debug=='yes':
                print('the word entry', w)
                print('the ranked fillers', ranked_fillers)
                print('len of rank', len(ranked_fillers))

              parts = w.split(':')
              if len(parts) != 6:
                  print(f"Unexpected key format: {w}")
                  continue
              word_only = parts[0]
              pos=parts[1]
              premise_start = int(parts[2])
              premise_end = int(parts[3])
              hypothesis_start = int(parts[4])
              hypothesis_end = int(parts[5])
              if debug=='yes':
                print('word_only',word_only)
                print('premise_start',premise_start)
                print('premise_end',premise_end)
                print('hypothesis_start',hypothesis_start)
                print('hypothesis_end',hypothesis_end)
              expected_variants = 0

              if isinstance(rank_option, int):
                  if len(ranked_fillers) >= rank_option:
                      expected_variants = 1
              elif isinstance(rank_option, slice):
                  start, stop, step = rank_option.indices(len(ranked_fillers))
                  expected_variants = len(range(start, stop, step))
              sentence_variants = []
              if label == 'neutral':
                  expected_generation['neutral'] += expected_variants
              elif label == 'entailment':
                  expected_generation['entailment'] += expected_variants
              elif label == 'contradiction':
                  expected_generation['contradiction'] += expected_variants


              if debug=='yes':
                print('the premise', premise_id)
                print('the hypothesis', hypothesis_id)
              try:
                if isinstance(rank_option, int):
                    if len(ranked_fillers) < rank_option - 1:
                        continue
                    best_ = ranked_fillers[rank_option][0].strip()

                    p_variant = premise_id[:premise_start] + best_ + premise_id[premise_end:]
                    if debug=='yes':
                      print('p_variant',p_variant)
                    h_variant = hypothesis_id[:hypothesis_start] + best_ + hypothesis_id[hypothesis_end:]
                    if debug=='yes':
                      print('h_variant', h_variant)
                    sentence_variants.append((p_variant, h_variant, best_))
                    if debug=='yes':
                      print('sentence_var', sentence_variants)

                elif isinstance(rank_option, slice):
                    for i in range(*rank_option.indices(len(ranked_fillers))):
                        best_ = ranked_fillers[i][0].strip()
                        p_variant = premise_id[:premise_start] + best_ + premise_id[premise_end:]
                        if debug=='yes':
                          print('p_variant',p_variant)
                        h_variant = hypothesis_id[:hypothesis_start] + best_ + hypothesis_id[hypothesis_end:]
                        sentence_variants.append((p_variant, h_variant, best_))
                        if debug=='yes':
                          print('h_variant', h_variant)
                          print('sentence_var', sentence_variants)
                if debug=='yes':
                  print('all sentence variants', sentence_variants)
                assigned_pos_tags.update(word2pos[w])
                for idx, (p_variant, h_variant, best_) in enumerate(sentence_variants):
                  numeric_label = None

                  if label == 'neutral':
                      numeric_label = 'neutral'
                      actual_generation['neutral'] += 1
                  elif label == 'entailment':
                      numeric_label = 'entailment'
                      actual_generation['entailment'] += 1
                  elif label == 'contradiction':
                      numeric_label = 'contradiction'
                      actual_generation['contradiction'] += 1

                  processed_entry = {
                      'id': f"{id}:{word_only}:{pos}:{premise_start}:{premise_end}:{hypothesis_start}:{hypothesis_end}:{best_}",
                      'premise': p_variant,
                      'hypothesis': h_variant,
                      'label': numeric_label
                  }

                  if id == 'yes':
                      processed_entry['id'] = f"{id}_{idx}"

                  if sort_by_pos == 'yes':
                          for pos_tag in assigned_pos_tags:
                              pos_tagged_data[pos_tag].append(processed_entry)
                  else:
                      processed_data.append(processed_entry)
              except Exception as e:
                  print("Error processing variants for key", w, ":", e)

    print("\nLabel Counts:")
    print(f"Neutral: {actual_generation['neutral']} (Expected: {expected_generation['neutral']})")
    print(f"Entailment: {actual_generation['entailment']} (Expected: {expected_generation['entailment']})")
    print(f"Contradiction: {actual_generation['contradiction']} (Expected: {expected_generation['contradiction']})\n")

    file_counts={output_file:actual_generation, 'sample': labels_sample}
    if sort_by_pos == 'yes':
        sorted_data = []
        for pos, entries in sorted(pos_tagged_data.items()):
            sorted_data.append({pos: entries})
        return sorted_data, new_list3
    if debug=='yes':
      print('processed_data', processed_data)
    with open(output_file, "w") as f:
        json.dump(processed_data, f)
    return processed_data, new_list3,file_counts

def generate_output_filenames(suggestion_file, number_inflation="10"):
    """
    Given a suggestion file path like:
      /.../robert-base-cased.1.noun.200.test.json
    extract parts of the name and automatically generate the output file names.

    Returns:
        output_processed_dataset, output_initial, output_all_inflated, output_all_sample, pos_to_mask
    """
    basename = os.path.basename(suggestion_file)
    parts = basename.split('.')

    if len(parts) < 6:
        raise ValueError("Filename does not follow expected naming convention.")

    pos_map = {
        "noun": "n",
        "verb": "v",
        "adjective": "adj",
        "adverb": "adv"
    }
    moodels={
        "bert-base-cased": "b",
        "roberta-base": "r"
    }

    model_tested = parts[0]
    model_name=moodels.get(model_tested)
    model_number = parts[1]
    pos_full = parts[2]
    pos_abbrev = pos_map.get(pos_full.lower(), pos_full.lower())
    size = parts[3]
    split_str = parts[4][:2]

    output_processed_dataset = f"{model_name}.{model_number}.{pos_abbrev}.{size}.{split_str}.inf.{number_inflation}.json"
    output_initial = f"{model_name}.{model_number}.{pos_abbrev}.{size}.{split_str}.samp.{number_inflation}.json"
    output_all_inflated = f"{model_name}.{model_number}.all.{size}.{split_str}.inf.{number_inflation}.json"
    output_all_sample = f"{model_name}.{model_number}.all.{size}.{split_str}.samp.{number_inflation}.json"

    # Use the full pos tag as pos_to_mask if that's what you need.
    pos_to_mask = pos_full

    return output_processed_dataset, output_initial, output_all_inflated, output_all_sample, pos_to_mask
