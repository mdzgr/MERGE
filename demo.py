import re
import torch
import torch.nn.functional as F
from typing import List, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from MERGE.utils import *
from assigntools.LoLa.read_nli import (
    snli_jsonl2dict,
    sen2anno_from_nli_problems
)

from assigntools.LoLa.deep_nli import (
    predict_nli,
    batch_predict_nli,
    load_tok_model
)


def return_pos_tag_for_class(class_pos):
  '''returns a set of accepted pos tags for each open class'''
  dictionar_classes_to_pos_tags_={'noun': {'NN', 'NNS'},
                                  'verb': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
                                  'adjective': {'JJ'}
                                  }
  return dictionar_classes_to_pos_tags_[class_pos]

def extract_pos_position_matches(i, pos, valid_tags, pos_type, tokens, sentence):
    '''returns all regex matches of tokens[i] in sentence, or None if this token doesn't qualify'''
    if pos not in valid_tags:
        return None
    if pos_type == "verb" and i + 1 < len(tokens) and tokens[i + 1] == "n't":
        return None
    token = tokens[i]
    matches = list(re.finditer(r'\b' + re.escape(token) + r'\b', sentence))
    if not matches:
        return None
    return matches


def return_offset_preceding_text(token_counts_dictionary, matches, token, sentence):
    token_counts_dictionary[token] += 1
    occurrence_index = token_counts_dictionary[token]
    matchy = matches[occurrence_index - 1]
    offset = (len(sentence[:matchy.start()]), len(sentence[:matchy.start()]) + len(token))
    return offset, sentence[:matchy.start()]


def extract__pos_position(pos_tags, tokens, source, pos_type, sentence):
    dictionary_positions, token_counts = {}, defaultdict(int)
    valid_tags = return_pos_tag_for_class(pos_type)

    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
        matches = extract_pos_position_matches(i, pos, valid_tags, pos_type, tokens, sentence)
        if not matches:
            continue

        offset, preceding_text = return_offset_preceding_text(token_counts, matches, token, sentence)
        if token not in dictionary_positions:
            dictionary_positions[token] = {'positions': [offset], 'pos': pos, 'source': source, 'preceding_text': preceding_text}
        else:
            dictionary_positions[token]['positions'].append(offset)

    return dictionary_positions


def get_mask_position(inputs, tokenizer):
    '''returns the single mask position in the input, raises valueerror if mask not there'''
    mask_positions = torch.where(inputs.input_ids[0] == tokenizer.mask_token_id)[0]
    if len(mask_positions) == 0:
        raise ValueError("Mask token not found in the input context.")
    return mask_positions.item()


def get_target_word_probability(target_word, probabilities, tokenizer):
    '''probability of target_word under the mask distribution, or None if not a single token'''
    if target_word is None:
        return None
    target_tokens = tokenizer(target_word, add_special_tokens=False)['input_ids']
    if len(target_tokens) != 1:
        return None
    return probabilities[target_tokens[0]].item()


def get_top_k_predictions(probabilities, tokenizer, top_k):
    '''top_k (score, token) predictions for the masked position'''
    top_probs, top_indices = torch.topk(probabilities, top_k)
    return [
        {"score": prob.item(), "token_str": tokenizer.convert_ids_to_tokens(int(idx))}
        for prob, idx in zip(top_probs, top_indices)
    ]

def assert_mask_token_exists(mask_token, context):
    if mask_token not in context:
      raise ValueError(f"Context must contain the mask token: {mask_token}")

def generate_mask_predictions(model, tokenizer, context, mask_token, target_word=None, top_k=50):
    """
    Generate replacements for a masked token.

    Args:
        model: The masked language model
        tokenizer: The corresponding tokenizer
        context: The input string containing only one mask token
        mask_token: The mask token (e.g., <mask> or [MASK])
        target_word: The word whose probability we want to retrieve (optional)
        top_k: The number of replacements to return
    """
    assert_mask_token_exists(mask_token, context)

    inputs = tokenizer(context, return_tensors="pt")
    mask_position = get_mask_position(inputs, tokenizer)

    with torch.no_grad():
        predictions = model(**inputs).logits[0, mask_position]
        probabilities = F.softmax(predictions, dim=-1)

    target_probability = get_target_word_probability(target_word, probabilities, tokenizer)
    prediction_list = get_top_k_predictions(probabilities, tokenizer, top_k)

    return prediction_list, target_probability


def extract_pos_words(pos_tags, tokens, pos_type):
  '''from category to pos tags'''
  tags = return_pos_tag_for_class(pos_type)
  return {tokens[i] for i, pos in enumerate(pos_tags) if pos in tags}


def meets_agreement_and_length(problem, annotators_agreement_number, length_premise, length_hypothesis):
    '''gold label exists, annotator disagreement is below threshold, premise/hypothesis meet min length'''
    return (
        problem['g'] != '-'
        and len(problem['lcnt']) < annotators_agreement_number
        and len(problem['p'].split()) >= length_premise
        and len(problem['h'].split()) >= length_hypothesis
    )


def shared_pos_words(problem, mapping, pos_to_mask):
    '''words of pos_to_mask shared between premise and hypothesis'''
    premise, hypothesis = mapping[problem['p']], mapping[problem['h']]
    premise_words = extract_pos_words(premise['pos'], premise['tok'], pos_to_mask)
    hypothesis_words = extract_pos_words(hypothesis['pos'], hypothesis['tok'], pos_to_mask)
    return premise_words & hypothesis_words


def filter_snli(dataset,mapping,pos_to_mask, min_common_words, num_sentences_to_process=None, max_filtered_count=None, annotators_agreement_number: int = 5, length_premise: int = 0,length_hypothesis: int = 0,):
    '''returns problems that have a gold label and:
    - share at least min_common_words words of pos_to_mask between premise and hypothesis
    - meet minimum length for premise and hypothesis
    - meet a minimum annotator agreement threshold
    stops early once max_filtered_count problems have been collected'''
    filtered = {}
    dataset_items = list(dataset.items())[:num_sentences_to_process] if num_sentences_to_process else dataset.items()

    for k, p in dataset_items:
        if not meets_agreement_and_length(p, annotators_agreement_number, length_premise, length_hypothesis):
            continue

        common_words = shared_pos_words(p, mapping, pos_to_mask)
        if len(common_words) < min_common_words:
            continue

        filtered[k] = p
        if max_filtered_count and len(filtered) >= max_filtered_count:
            break

    return filtered

def process_unmasked_dataset(filtered_list_1, neutral_number, entailment_number, contradiction_number, include_id):
    '''converts filtered_list_1 items into model-ready dicts with numeric labels
    ffiltered_list_1 structure  [{'id': f"{base_id}{version}",
            # 'label': p['g'],
            # 'premise': p['p'],
            # 'hypothesis': p['h'],
            # 'p_p': mapping[p['p']]['pos'],
            # 'p_t': mapping[p['p']]['tok'],
            # 'h_p': mapping[p['h']]['pos'],
            # 'h_t': mapping[p['h']]['tok']},]=
    '''
    label_to_number={'neutral': neutral_number, 'entailment': entailment_number, 'contradiction':contradiction_number}
    new_list4 = []
    for item in tqdm(filtered_list_1):
        entry = {
            'premise': item['premise'],
            'hypothesis': item['hypothesis'],
            'label': label_to_number[item['label']],
        }
        if include_id:
            entry['id'] = item['id']
        new_list4.append(entry)

    label_counts = Counter(item['label'] for item in filtered_list_1)
    print("Label counts inside loop:", dict(label_counts))
    return new_list4, dict(label_counts)


def pos_toks_extract_from_dataset(list_filtered, mapping):
  '''  in: ### list_filtered structure {'3827316480.jpg#0r1e': {'g': 'entailment', 'pid': '3827316480.jpg#0r1e', 'cid': '3827316480.jpg#0', 'lnum': 5, 'lcnt': Counter({'entailment': 5}), 'ltype': '500', 'p': '...', 'h': '...'}'''
  filtered_list_1 = [
      {
          'id': k,
          'label': p['g'],
          'premise': p['p'],
          'hypothesis': p['h'],
          'p_p': mapping[p['p']]['pos'],
          'p_t': mapping[p['p']]['tok'],
          'h_p': mapping[p['h']]['pos'],
          'h_t': mapping[p['h']]['tok'],
      }
      for k, p in list_filtered.items()
  ]
  print(f"no. problems filtered after criteria: {len(filtered_list_1)}")
  return filtered_list_1


def is_sentence_fully_processed(sentence, filler_data, common_tokens_dictionary):
  if sentence not in filler_data:
      return False
  existing_keys = set(filler_data[sentence].keys())
  required_keys = {f"{token}:{data['pos']}" for token, data in common_tokens_dictionary.items()}
  return required_keys.issubset(existing_keys)

def common(sentence1, sentence2, pos_sent_1, pos_sent_2, toks_sent_1, toks_sent_2, pos_type, source_1, source_2, singles='yes'):
    ''''extracted_1 {'black': {'positions': [(11, 16)], 'pos': 'JJ', 'source': 'premise', 'preceding_text': 'A man in a '}, 'commercial': {'positions': [(29, 39)], 'pos': 'JJ', 'source': 'premise', 'preceding_text': 'A man in a black shirt, in a '}}
      extracted_2 {'black': {'positions': [(13, 18)], 'pos': 'JJ', 'source': 'hypthesis', 'preceding_text': 'A woman in a '}, 'commercial': {'positions': [(31, 41)], 'pos': 'JJ', 'source': 'hypthesis', 'preceding_text': 'A woman in a black shirt, in a '}}
      common tokens {'black', 'commercial'}
      common dict {'black': {'positions': [(11, 16)], 'pos': 'JJ', 'source': 'premise', 'preceding_text': 'A man in a '}, 'commercial': {'positions': [(29, 39)], 'pos': 'JJ', 'source': 'premise', 'preceding_text': 'A man in a black shirt, in a '}}
      mask positions 1 [[(11, 16)], [(29, 39)]]
      mask positions 2 [[(13, 18)], [(31, 41)]]'''

    extracted_1 = extract__pos_position(pos_sent_1, toks_sent_1, source_1, pos_type, sentence1)
    extracted_2 = extract__pos_position(pos_sent_2, toks_sent_2, source_2, pos_type, sentence2)
    common_tokens = set(extracted_1.keys()) & set(extracted_2.keys())
    common_dict = {token: extracted_1[token] for token in common_tokens}
    all_nouns_singles = {' ' + k for d in [extracted_1, extracted_2] for k, v in d.items()} if singles=='yes' else None
    mask_positions_1 = [extracted_1[token]["positions"] for token in common_tokens]
    mask_positions_2 = [extracted_2[token]["positions"] for token in common_tokens]
    return common_dict, mask_positions_1, mask_positions_2, all_nouns_singles

def return_offset_key(offset, probability_masked_word):
    if probability_masked_word is None:
        return f"{offset}:{probability_masked_word}"
    return f"{offset}:{probability_masked_word:.2e}"

def return_masked_token(input_str, i, j):
        return input_str[i:j]

def return_offse_key_1(i, j):
  return str(i)+':'+str(j)

def return_masked_sentence(mask_tok, string, i, j):
  return string[:i] + f'{mask_tok}' + string[j:]

def return_mask_tok_off_pos_mask_in(input_str, i, j, common_tokens, mask_token):
  masked_token_orig = return_masked_token(input_str, i, j)
  offset_key = return_offse_key_1(i, j)
  pos_tag = common_tokens.get(masked_token_orig, {}).get("pos", "UNK")
  masked_input = return_masked_sentence(mask_token, input_str, i, j)
  return masked_input, masked_token_orig, pos_tag, offset_key

def return_inp_tok_spacing_tokenizer(masked_input, mask_token, mask_token_orig):
  if masked_input.endswith(mask_token):
    masked_input += '.'
  if mask_token == '<mask>' and not masked_input.startswith('<mask>'):
    mask_token_orig=' '+mask_token_orig
  return masked_input, mask_token_orig

def strip_sapce_from_mask_tok(mask_tok, masked_in, masked_token_orig):
  if mask_tok == '<mask>' and not masked_in.startswith('<mask>'):
    masked_token_orig=masked_token_orig.strip()
  return masked_token_orig

def return_processed_suggestions(k):
  return f"{k['token_str'].lstrip().strip('Ġ▁')}:{k['score']:.2e}"

def assert_length_candidates(candidate_list, suggestion_n, input_str):
  if len(candidate_list) != suggestion_n:
    print(f"\nWarning: Expected {suggestion_n} suggestions but got {len(candidate_list)}")
    print(f"Input string: {input_str}")


def add_suggestion(suggestions, masked_token_orig, pos_tag, offset_key, candidate_list):
    '''adds candidate_list under suggestions[token_key][offset_key], creating levels as needed;
    extends the existing list if that offset_key was already seen for this token'''
    token_key = f"{masked_token_orig}:{pos_tag}"
    token_entry = suggestions.setdefault(token_key, {})
    if offset_key not in token_entry:
        token_entry[offset_key] = candidate_list
    else:
        token_entry[offset_key].extend(candidate_list)
    return suggestions


def suggest_mask_fillers(input_str:str, mask_offsets: List[Tuple[int,int]],
                         model, tokenizer, all_single_words, common_tokens, suggestion_n=50):
                             #not double-checked
    """ mask_offsets is a list of integer pairs that mark the part of teh string input taht needs to be masked.
        It is a list because in general it might be needed to mask several parts of the input string.
        Returns a dictionary with character offsets as keys and a list of ranked suggestions as values.
    """
    mask_token = tokenizer.mask_token

    suggestions = {}
    mask_off_flat= [i for w in mask_offsets for i in w]

    for i, j in mask_off_flat:
      candidate_list = []

      masked_input, masked_token_orig, pos_tag, offset_key = return_mask_tok_off_pos_mask_in(input_str, i, j, common_tokens, mask_token)
      masked_input, masked_token_orig = return_inp_tok_spacing_tokenizer(masked_input, mask_token, masked_token_orig)

      generated, probability_masked_word = generate_mask_predictions(model, tokenizer, masked_input, mask_token, masked_token_orig, suggestion_n)

      masked_token_orig = strip_sapce_from_mask_tok(mask_token, masked_input, masked_token_orig)
      offset_key=return_offset_key(offset_key, probability_masked_word)


      for k in generated:
          candidate_list.append(return_processed_suggestions(k))

      assert_length_candidates(candidate_list, suggestion_n, input_str)

      suggestions = add_suggestion(suggestions, masked_token_orig, pos_tag, offset_key, candidate_list)
    return suggestions


def return_model_tok(model_name):
  return AutoModelForMaskedLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)


def merge_filler_results(results_dict, sentence, off_filler):
    '''adds off_filler under results_dict[sentence], creating the entry if needed,
    or merging into the existing dict of fillers for that sentence'''
    if off_filler:
        results_dict.setdefault(sentence, {}).update(off_filler)
    return results_dict

def return_name_models_suggestions():
  return {"google/electra-base-generator": "electrab", #
              "google-bert/bert-base-cased":"bertb",#
              "google-bert/bert-large-cased":"bertl",#
              "FacebookAI/roberta-base":"robertab",#  %
              "FacebookAI/roberta-large":"robertal",#
              "facebook/bart-base": "bartb",
              "facebook/bart-large":"bartl",
              "albert/albert-base-v2": "albertb",
              "albert/albert-xxlarge-v2": "albertxxl",
              "google/electra-large-generator": "electrl"}

def create_filler_file( #this also needs to be split or find an altenrative
    model_name: str,
    dataset: pd.DataFrame,
    split: str,
    pos_to_mask: str,
    min_common_words: int,
    number_replacements: int, #number_replacements
    mapping,
    exclude_words_part_of_p_and_h: str,
    agreement_annotators_no,
    number_words_premise,
    number_words_hypothesis,
    num_sentences_to_process_dataset: int = None,
    num_sentences_compliant_criteria: int = None,
    id_mock_test: str=None,
    add_id_to_dataset: bool = False,
    output_file: str = None,
):
    """
        Function generating a new inflated dataset with suggestion from a language model, alongside the initial split of sentences modified

        model_name : str // name of language model used for generating masked token predictions (e.g., "bert-base-uncased").
        dataset : Dataset // The dataset to process
        split : str // The dataset split to use (e.g., "train", "test", "validation").
        pos_to_mask : str // a str indicating what pos to be masked in sentences ('noun' or 'verb'), and what sentences to be picked for masking considering trhe min_common_words (e.g. 3 common nouns)
        min_common_words : int// the minimum number of common words required between premise and hypothesis
        number_replacements : int// The number of suggested filler words for each masked token by model
        mapping: mapping function
        exclude_words_part_of_p_and_h: if we want to exclude or not 3xclude alreadye xisting words
        num_sentences_to_process_dataset : int /// The number of sentences to process from the dataset.
        add_id_to_dataset: if an id it will add only a specific mock id to the dataset
        output_file : str /// file name where the masked dataset will be saved
        #returns the list of processed sentences with ids and a sepearte file with the suggestions
    """
    results_dict = {}

    dataset = dataset[split]
    SNLI_filtered_2 = filter_snli(dataset, mapping, pos_to_mask, min_common_words, num_sentences_to_process_dataset, num_sentences_compliant_criteria, agreement_annotators_no, number_words_premise, number_words_hypothesis)

    filtered_list_1 = pos_toks_extract_from_dataset(SNLI_filtered_2, mapping)

    seed_dataset, lab = process_unmasked_dataset(filtered_list_1, 'neutral', 'entailment', 'contradiction', include_id=add_id_to_dataset)
    model, tokenizer = return_model_tok(model_name)
    for p in tqdm(filtered_list_1):
        id, premise, hypothesis, tok_p, pos_p, tok_h, pos_h = (p['id'], p['premise'], p['hypothesis'], p['p_t'], p['p_p'], p['h_t'], p['h_p'] )
        if id_mock_test and id != id_mock_test:
          continue
        common_tokens_dictionary, p_off, h_off, all_nouns_singles = common(premise, hypothesis, pos_p, pos_h, tok_p, tok_h, pos_to_mask, 'premise', 'hypothesis', exclude_words_part_of_p_and_h)
        p_off_filler = suggest_mask_fillers(premise, p_off, model, tokenizer, all_nouns_singles, common_tokens_dictionary, number_replacements)
        merge_filler_results(results_dict, premise, p_off_filler)
        h_off_filler = suggest_mask_fillers(hypothesis, h_off, model, tokenizer, all_nouns_singles, common_tokens_dictionary, number_replacements)
        merge_filler_results(results_dict, hypothesis, h_off_filler)
    if output_file:
      create_json_from_data(results_dict, output_file)
    return seed_dataset, results_dict



def return_dictionary_datasets(name_dataset):
  return f'/content/drive/MyDrive/nlp_data/{name_dataset.lower()}_1.0'


def return_dataset_mapping_function(name_dataset): ###this has to be updated in pmi
    '''returns a list of normalized frequancies for words in SNLI train/ MNLI_train
    and a list of seed problems w. unonaimous agreemenet bt annotators'''
    name_codes={'SNLI': 'snli',
                  'SNLI-m': 'mnli_matched',
                  'SNLI-mis': 'mnli_mismatched'}
    dictionary_type_dataset={'snli': '.+_(train|dev|test).jsonl',
                          'mnli_matched': '.+_dev_matched.jsonl$',
                          'mnli_mismatched': '.+_dev_mismatched.jsonl$'}
    names_datasets_files={'snli':'snli_1.0',
                          'mnli_matched': 'multinli',
                          'mnli_mismatched': 'multinli'}
    dataset, S2A = snli_jsonl2dict(return_dictionary_datasets(name_dataset), name_codes.get(name_dataset), dictionary_type_dataset, names_datasets_files)
    return dataset, S2A

from collections import Counter, defaultdict
from wordfreq import zipf_frequency
from contextlib import redirect_stdout
#from MERGE.generate_suggestions import filter_snli, pos_toks_extract_from_dataset, process_unmasked_dataset, common
import string, json, os, re, random
from tqdm import tqdm
import sys, contextlib
from functools import partial
from itertools import chain
import string
import spacy

def excluded_words_default():
  return {"n't", "not", "no", "never", "neither", "none", "nowise", "nothing", "nobody", "nowhere", "non", "absent", "lacking", "minus", "without", "'s", "'n'", "'re", "'m"}

def return_common_affixes():
  return {"anti", "auto", "bi", "co", "contra", "counter", "de", "dis", "en", "em",
      "extra", "hetero", "homo", "hyper", "il", "im", "in", "ir", "inter", "intra",
      "macro", "micro", "mid", "mis", "mono", "multi", "non", "over", "post",
      "pre", "pro", "pseudo", "re", "semi", "sub", "super", "tele", "trans",
      "tri", "ultra", "un", "under",
      "able", "ible", "al", "ally", "ance", "ence", "ant", "ent",
      "ary", "ery", "ory", "ate", "ed", "er", "est", "ful",
      "hood", "ic", "ical", "ify", "ing", "ion", "tion", "sion",
      "ish", "ism", "ist", "ity", "less", "let", "like", "ling",
      "ly", "ment", "ness", "ous", "ship", "y"}


def _word_variants(word):
    """All the case/whitespace variants we check against all_singles."""
    forms = {word, word.lower(), word.strip(), word.strip().lower()}
    forms |= {f" {f}" for f in forms}
    return forms


def _is_excluded(word, excluded_words, all_singles):
    if len(word) == 1 or word in return_common_affixes() or word.startswith("##") or word in excluded_words or set(word) <= set(string.punctuation):
        return True
    if all_singles is not None and _word_variants(word) & set(all_singles): #loved this one
        return True
    return False


def filter_candidates(candidates, all_singles=None, excluded_words=None):
    """Filter out unwanted words from a list of 'word:...' candidate strings."""
    if excluded_words is None:
        excluded_words = excluded_words_default()

    return [candidate for candidate in candidates if not _is_excluded(candidate.split(":", 1)[0], excluded_words, all_singles)]

def flatten_dataset(data):
    '''Creates a list of unique items {p, h, l},
    useful when having a nested, redundant dataset (e.g. same examples across splits).'''
    seen = {}
    for example in chain.from_iterable(data.values()):
        seen.setdefault(example["id"], example)
    return list(seen.values())

def filter_suggestions_by_contextual_pos(suggestions, original_sentence, start_idx, end_idx, allowed_pos_tags, nlp, batch_size_no):
    """
    tags replacements with spacy nlp pipeline: replace original > tag replacement in context
    return list_filter_suggestions_with_pos + count for each word per pos tags
    """
    suggestion_probabilities = {suggestion.split(":")[0]: suggestion.split(":")[1] for suggestion in suggestions}
    docs = [original_sentence[:start_idx] + suggestion.split(":")[0] + original_sentence[end_idx:] for suggestion in suggestions]
    filtered = [f"{token.text}:{suggestion_probabilities.get(token.text)}:{token.tag_}" for doc in nlp.pipe(docs, batch_size=batch_size_no) for token in doc if token.idx == start_idx]
    pos_counts = Counter(entry.split(":")[-1] for entry in filtered)
    return filtered, pos_counts

def build_key_prefix(pos_pair):
    '''joins start:end position into a key prefix, e.g. (11,16) -> "11:16:"'''
    p_start, p_end = pos_pair
    return f"{p_start}:{p_end}:"

def keys_starting_with_prefix(keys, prefix):
    '''returns keys that start with a given prefix'''
    return [k for k in keys if k.startswith(prefix)]

def get_all_matching_keys(data, word_with_pos, pos_list):
    '''match beginning key entries in dictionary that start with positions of words
    # pos_list premise_pos_list [(11, 16)]
    # data=sentence  #{'poses:VBZ': {'26:31:4.23e-04': ['','']}
    # word_with_pos original_word:pos
     ## result get_all_matching_keys the matching keys ['48:51:7.56e-01'] all matching keys ['20:23:9.83e-01', '48:51:7.56e-01'], or single element if single apperance

    '''
    keys = data[word_with_pos].keys()
    return [k for pos_pair in pos_list for k in keys_starting_with_prefix(keys, build_key_prefix(pos_pair))]

def has_pos_tags(suggestions): # outputs True if suggestions have pos tags added besides probability
    '''returns True if all suggestions have suggestion:probability:postag (3 elements)'''
    return all(len(s.split(":")) == 3 for s in suggestions)

def count_pos(suggestions):
  '''return counts per pos tag'''
  return Counter(s.split(":")[2] for s in suggestions)

def ranked_overlap(list_of_lists, probs, type_):
    '''ranks words based on probability or the averaged position of words in two lists
    by taking all elemnents of the list, or only the common ones'''
    n = len(list_of_lists)
    if type_== 'union': # if union
      s = set().union(*list_of_lists) #unite lists of list
    if type_=='intersection':
      s = set(list_of_lists[0]).intersection(*map(set, list_of_lists[1:])) # make set, then intersect
    s_ranks = dict()
    for element in s:
        ranks = [l.index(element) for l in list_of_lists if element in l] #get the rank of the filtered suggestions, for each list
        probs1=[z[l.index(element)] for l, z in zip(list_of_lists, probs) if element in l] #get the probability of the elements that are in s
        avg_prob=sum(probs1)/len(ranks) #sum them and divide them by how many positions does the word have, which is 2
        s_ranks[element] = {
            'average_rank': sum(ranks)/n,
            'ranks' :ranks,
            'average_prob': f"{avg_prob:.2e}",
            "individual_probs": [f"{p:.2e}" for p in probs1]}
    return s_ranks

def without_last_2(full_id):
  '''from 2677109430.jpg#1r1n:church:NN:5:11:4:10:building:h:bert -> to without :h:bert
  '''
  return ':'.join(full_id.split(':')[:-2])


def return_id_list_items(items_list):
  '''returns a list of item ids'''
  return [item['id'] for item in items_list]

def return_word_pos_lookup(ids_across_datasets):
    '''word_pos_lookup structure {'2677109430.jpg#1r1n:church:NN:apartment': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:attic': {'h': {'roberta'}},'''
    word_pos_lookup = {}
    for full_id in ids_across_datasets:
        parts = full_id.split(':')
        word_pos_lookup.setdefault(':'.join(parts[:3] + [parts[-3]]), {}).setdefault(parts[-2], set()).add(parts[-1])
    return word_pos_lookup


def return_ids_across_datasets(pairs_datasets):
  return set([id_ for dataset, _ in pairs_datasets for id_ in return_id_list_items(dataset)])


def return_opposite_origin(origin, word_pos_id, word_pos_lookup):
  opp_origin='p' if origin == 'h' else 'h'
  return opp_origin, word_pos_id in word_pos_lookup and opp_origin in word_pos_lookup[word_pos_id]


def build_new_id(origin, base_id, model, word_pos_id, word_pos_lookup, opposite):
    """
    Build the new id for a suggestion whose origin was 'p' or 'h', if it is also suggeste by other models or not
    """
    opposite_origin, has_opposite = return_opposite_origin(origin, word_pos_id, word_pos_lookup)
    token = "ph" if has_opposite else origin

    if opposite:
        if not has_opposite:
            return None
        end = "both" if any(m != model for m in word_pos_lookup[word_pos_id][opposite_origin]) else model
        new_id = f"{base_id}:ph:{end}"
    else:
        new_id = f"{base_id}:{token}:{model}"

    return new_id


def parse_item_id(item_id):
    """Extract word_pos_id, origin, model, base_id from an item's id."""
    parts = item_id.split(':')
    return ':'.join(parts[:3] + [parts[-3]]), parts[-2], parts[-1], without_last_2(item_id)

def shared_label(label_for_shared_suggestions, num_models: int):
      if label_for_shared_suggestions.lower() == "no_models":
          return f"{num_models}"
      return "both"

def build_ph_id(base_id, model, word_pos_id, word_pos_lookup, label_for_shared_suggestions):
    """Build the new id for a suggestion whose origin was already 'ph' (shared p/h)."""
    origins = word_pos_lookup.get(word_pos_id, {})
    other_origins_exist = any(origins.get(o, set()) - {model} for o in ('p', 'h', 'ph'))

    if not other_origins_exist:
        return f"{base_id}:ph:{model}"

    involved_models = set()
    for o in ('p', 'h', 'ph'):
        involved_models |= origins.get(o, set())
    label = shared_label(label_for_shared_suggestions, len(involved_models))
    return f"{base_id}:ph:{label}"


def rename_items(all_pairs, word_pos_lookup, opposite, label_for_shared_suggestions):
    """Walk all dataset items, rebuild their ids, and dedupe by word_pos_id."""
    seen_ids, renamed = set(), []

    for dataset, _ in all_pairs:
        for item in dataset:
            #print('the item', item)
            word_pos_id, origin, model, base_id = parse_item_id(item['id'])
            if origin not in ('h', 'p', 'ph') or word_pos_id in seen_ids:
                continue

            new_id = (build_ph_id(base_id, model, word_pos_id, word_pos_lookup, label_for_shared_suggestions) if origin == 'ph' else build_new_id(origin, base_id, model, word_pos_id, word_pos_lookup, opposite))
            if new_id is None:
                continue

            seen_ids.add(word_pos_id)
            renamed.append({'id': new_id, 'premise': item['premise'], 'hypothesis': item['hypothesis'],'label': item['label'],})
    return renamed

def filter_qualified_items(renamed, min_count):
    """Keep only items whose base_id appears at least min_count times."""
    final_counts = Counter(get_base_ids_from_data(renamed, 'flat'))
    qualified_base_ids = {bid for bid, count in final_counts.items() if count >= min_count}
    final_dataset = [item for item in sorted(renamed, key=lambda x: x['id']) if final_counts.get(split_to_get_base_id(item['id']), 0) >= min_count]
    return final_dataset, qualified_base_ids


def compute_dataset_stats(final_dataset):
    """Compute source counts, average per source, and label counts for final_dataset."""
    source_counts = Counter(item['id'].split(':')[-1] for item in final_dataset)
    total_instances = sum(source_counts.values())
    average_per_source = ({src: count / total_instances for src, count in source_counts.items()} if total_instances else {})
    label_counts = Counter(item['label'] for item in final_dataset)
    return source_counts, total_instances, average_per_source, label_counts



def print_dataset_report(dual_print, qualified_base_ids, final_dataset, min_count, source_counts, total_instances, average_per_source, label_counts):
    dual_print(f"Qualified seeds (count >= {min_count}): {len(qualified_base_ids)}")
    dual_print(f"Length dataset {len(final_dataset)}",f" Length per origin model: {total_instances}")
    dual_print(f"Source counts: {dict(source_counts)}")

    dual_print("=== Label Counts in Final Dataset ===")
    for label, count in label_counts.items():
        dual_print(f"{label}: {count}")

    dual_print("\n=== Average Instances per Source ===")
    for src, avg in average_per_source.items():
        dual_print(f"{src}: {avg:.2f}")


def compute_and_report_stats(renamed, pos_name_file_name, type, min_count):
    """Filter renamed items by min_count, print/save stats, return final dataset."""
    outfile = f"potential_variants_{pos_name_file_name}_{type}_stats.txt"

    final_dataset, qualified_base_ids = filter_qualified_items(renamed, min_count)
    source_counts, total_instances, average_per_source, label_counts = compute_dataset_stats(final_dataset)

    with open(outfile, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout

        def dual_print(*args, **kwargs):
            print(*args, **kwargs, file=original_stdout)
            print(*args, **kwargs, file=f)

        print_dataset_report(dual_print, qualified_base_ids, final_dataset, min_count, source_counts, total_instances, average_per_source, label_counts)

    return final_dataset, average_per_source


def merge_and_analyze_datasets(dataset1, source1, dataset2, source2, pos_name_file_name, type, min_count, name=None, opposite=True, others=None, label_for_shared_suggestions: str = "general"):
    '''
    Function that merges datasets of several models
    Keeps suggestions for one replacement only if that replacement has sufficient variants
    '''
    print(f"dataset1 size={len(dataset1)}; dataset2 size={len(dataset2)}")

    all_pairs = [(dataset1, source1), (dataset2, source2)]
    if others:
        all_pairs.extend(others)

    ids_across_datasets = return_ids_across_datasets(all_pairs)
    print(f"ids_across_datasets size={len(ids_across_datasets)}")
    word_pos_lookup = return_word_pos_lookup(ids_across_datasets)

    renamed = rename_items(all_pairs, word_pos_lookup, opposite, label_for_shared_suggestions)

    return compute_and_report_stats(renamed, pos_name_file_name, type, min_count)

def return_pos_additional_noun_tags(class_pos_to_mask):
      '''class_pos_to_mask: noun, verb, adj'''
      pos_filter_map = {
      'noun': {'NN', 'NNS', 'NNP', 'NNPS'},
      'verb': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
      'adjective': {'JJ'},
      }
      return pos_filter_map[class_pos_to_mask]

def check_type_return_correct(type_rank):
  '''check if the value of type_rank is one of the allowed ones'''
  allowed_values = {"average_rank", "average_prob", None}
  if type_rank not in allowed_values:
      raise ValueError(f"type_rank must be one of {allowed_values}, got {type_rank}")

def return_key_value_of_dict_in_list(dictionary_f, list_f):
  return {key: value for key, value in dictionary_f.items() if key in list_f}


def return_words_of_2_sentences(sent_1, sent_2):
  '''assumes words are divided by white space
  strips and lowers the tokens'''
  singles_premise, singles_hypothesis = sent_1.split(' '), sent_2.split(' ')
  clean=singles_premise+singles_hypothesis
  return [word.strip(string.punctuation).lower() for word in clean]
def unpack_entry_dataset(entry_dataset):
  return entry_dataset['id'], entry_dataset['premise'], entry_dataset['hypothesis'], entry_dataset['p_t'],entry_dataset['p_p'], entry_dataset['h_t'], entry_dataset['h_p'], entry_dataset['label']

def get_pos_for_word(common_dict, word):
  pos=common_dict[word]['pos']
  return pos, f'{word}:{pos}'


def return_word_start_end_keys(all_matching_keys_p, all_matching_keys_h, pos, word):
  p_start, p_end, _= all_matching_keys_p[0].split(':')
  h_start, h_end, _= all_matching_keys_h[0].split(':')
  return f"{word}:{pos}:{p_start}:{p_end}:{h_start}:{h_end}"

def missing_pos_tagging(premise_suggestions, hypothesis_suggestions):
    '''checks if premise or hypothesis suggestions/filters are missing POS tags'''
    return (not has_pos_tags(premise_suggestions) or not has_pos_tags(hypothesis_suggestions))

def print_not_fully_tagged(pos_tag_filtering, premise_suggestions, hypothesis_suggestions):
    if pos_tag_filtering == 'yes' and missing_pos_tagging(premise_suggestions, hypothesis_suggestions):
      print("Some of the suggestions for premise or hypothesis are not tagged for POS tag")



def return_suggestions(list_suggestions):
  return [c.split(":")[0] for c in list_suggestions]


def return_probabilities(list_suggestions):
  return [float(c.split(":")[1]) for c in list_suggestions if c.split(":")[1] != '']


def get_suggestions_plus_pos_for_key(data, word_with_pos, k, sentence, pos_tag_filtering, allowed_pos_tags, nlp, batch_nlp_classification_no, save_suggestions_in_file, data_with_suggestions=None):
    """
    get suggestions for a key, see if they have pos, if not, tag, if yes just return what was stored

    Returns:
        list: the suggestions for this key.
    """
    data_key = data[word_with_pos][k]
    has_p = has_pos_tags(data_key)

    if not has_p and pos_tag_filtering == 'yes':
        p_start, p_end, _ = k.split(":")
        suggestions, pos_counts = filter_suggestions_by_contextual_pos(data_key, sentence, int(p_start), int(p_end), allowed_pos_tags, nlp, batch_nlp_classification_no)

        if save_suggestions_in_file and data_with_suggestions is not None:
            data_with_suggestions[sentence][word_with_pos][k] = suggestions

        return suggestions

    return data[word_with_pos][k]

def return_tag_of_suggestion(suggestion):
  return suggestion.split(":")[-1]

def return_original_prob(k):
  return float(k.split(":")[2])


def return_probability_suggestion(suggestion):
  return suggestion.split(":")[1]

def return_clean_suggestions_by_pos(list_suggestions, o):
  return [s for s in list_suggestions if return_tag_of_suggestion(s) in o]

def assert_clean_sugg_prob(list_suggestions):
  for c in list_suggestions:
    try:
      float(return_probability_suggestion(c))
    except ValueError:
      print(f"DEBUG: Found problematic entry in: {list_suggestions}")
      print('the problematic entry is', c)
      return False
  return True


def return_clean_suggestions_by_prob(list_suggestions, org_prob):
  return [c for c in list_suggestions if return_probability_suggestion(c) not in ('', 'None') and float(return_probability_suggestion(c)) >= org_prob]

def asseert_condition_met_prob(cleaned_list, orig):
  if any(float(return_probability_suggestion(c)) < orig for c in cleaned_list):
    print(f"DEBUG: Found problematic entry in: {cleaned_list}")
    return True
  return False

def clean_suggestions_prob(suggestions_list, k): #this has a bug
  origin_prob=return_original_prob(k)
  cleaned_list= return_clean_suggestions_by_prob(suggestions_list, origin_prob)
  if asseert_condition_met_prob(cleaned_list, origin_prob):
    return None
  return cleaned_list

def parse_suggestion(s, pos_tag_filtering):
    """
    gives back word, prob, and pos tag if they exist in the dataset
    """
    parts = s.split(":")
    word, val = parts[0], parts[1]
    pos = parts[2] if pos_tag_filtering == 'yes' else None
    return word, float(val), pos


def build_presence_map(per_key_cleaned_lists, pos_tag_filtering):
    """

    """
    presence = defaultdict(list)
    for lst in per_key_cleaned_lists:
        for s in set(lst):
            word, val, pos = parse_suggestion(s, pos_tag_filtering)
            presence[word].append((val, pos))
    return presence

def build_intersected_suggestions(presence, cleaned_suggestions, min_occurrences=2):
    """

    """
    keep = {w: f"{w}:{sum(v for v, _ in vals) / len(vals)}:{vals[0][1]}"  # assuming POS tag is the same for the 2 options
        for w, vals in presence.items()
        if len(vals) >= min_occurrences}

    intersected_suggestions = [keep[s.split(":")[0]] for s in cleaned_suggestions if s.split(":")[0] in keep]
    return intersected_suggestions

def return_allowed_pos_per_class_or_tag(type_pos_filtering, allowed_pos_tags, word_with_pos):
  if type_pos_filtering in ('class', 'pos_tag'):
    return allowed_pos_tags if type_pos_filtering == 'class' else [word_with_pos.split(':')[1]]


def assert_probability_original_exists(k):
  try:
    return_original_prob(k)
    return True
  except (ValueError, IndexError):
    return False


def process_matching_keys(data, sentence, word_with_pos, all_matching_keys,
          allowed_pos_tags,  pos_tag_filtering, prob, nlp, singles, batch_nlp_classification_no,
          save_suggestions_in_file=False, data_with_suggestions=None, avrg=None, type_pos_filtering=None):


  cleaned_suggestions, per_key_cleaned_lists, intersected_suggestions =  [], [], []
  if len(all_matching_keys) == 0:
    print("No matching keys found", sentence)
  for k in all_matching_keys: #


      temporary_suggestions= get_suggestions_plus_pos_for_key(data, word_with_pos, k, sentence, pos_tag_filtering, allowed_pos_tags, nlp, batch_nlp_classification_no, save_suggestions_in_file, data_with_suggestions)
      o=return_allowed_pos_per_class_or_tag(type_pos_filtering, allowed_pos_tags, word_with_pos)

      cleaned_list = filter_candidates(temporary_suggestions, singles)

      if pos_tag_filtering == 'yes':

          cleaned_list = return_clean_suggestions_by_pos(cleaned_list, o)

          if len(cleaned_list) == 0:                                                      #if no suggestions left, continue
              continue

      if prob == "yes":
          if assert_probability_original_exists(k) != True:
            continue
          cleaned_list = clean_suggestions_prob(cleaned_list, k)
          if cleaned_list is None:
            break
      per_key_cleaned_lists.append(cleaned_list); cleaned_suggestions.extend(cleaned_list)

  if len(all_matching_keys) == 1:
    intersected_suggestions = cleaned_suggestions

  elif len(all_matching_keys) > 1:
    presence = build_presence_map(per_key_cleaned_lists, pos_tag_filtering)
    intersected_suggestions = build_intersected_suggestions(presence, cleaned_suggestions)
  return intersected_suggestions

def assert_suggestions_to_skip(premise_fillers, hypothesis_fillers, number_of_minimal_suggestions_common_bt_p_h, words, seed_dataset, item_id, counters, stats):
    """
    Returns True if this item should be skipped. Mutates `seed_dataset`
    and `counters` in place as a side effect.
    counters keys: 'global_words_without_replacements',
                   'words_with_not_enough_replacements_inside_loop_utilitary',
                   'problems_removed_due_to_low_suggestions'
    """
    if len(premise_fillers) == 0 and len(hypothesis_fillers) == 0:
        return True

    common_suggestions = set(premise_fillers) & set(hypothesis_fillers)
    if len(common_suggestions) < number_of_minimal_suggestions_common_bt_p_h:
        counters['global_words_without_replacements'] += 1

        if len(words) == 1 or stats['words_with_not_enough_replacements_inside_loop_utilitary'] == len(words) - 1:
            seed_dataset[:] = [item for item in seed_dataset if item['id'] != item_id]
            counters['problems_removed_due_to_low_suggestions'] += 1
        else:
            stats['words_with_not_enough_replacements_inside_loop_utilitary'] += 1

        return True

    return False


def process_word_for_alignment(word, idx, common_dict, data_with_suggestions, premise, hypothesis, p_positions, h_positions, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, type_pos_filtering, number_of_minimal_suggestions_common_bt_p_h, words, seed_dataset, item_id, counters, stats):
    """
    Process a single word: fetch POS/matching keys, get premise and hypothesis suggestions, and decide whether it should be skipped.

    Returns:
        None if the word should be skipped, otherwise a tuple:
        (key, premise_fillers, hypothesis_fillers, premise_probabilities, hypothesis_probabilities, pos, all_matching_keys_p, all_matching_keys_h)
    """
    pos, word_with_pos = get_pos_for_word(common_dict, word)
    premise_data, hypothesis_data = data_with_suggestions[premise], data_with_suggestions[hypothesis]

    all_matching_keys_p = get_all_matching_keys(premise_data, word_with_pos, p_positions[idx])
    all_matching_keys_h = get_all_matching_keys(hypothesis_data, word_with_pos, h_positions[idx])

    key = return_word_start_end_keys(all_matching_keys_p, all_matching_keys_h, pos, word)

    premise_suggestions = process_matching_keys(premise_data, premise, word_with_pos, all_matching_keys_p, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, data_with_suggestions, avrg='yes', type_pos_filtering=type_pos_filtering)

    hypothesis_suggestions = process_matching_keys(hypothesis_data, hypothesis, word_with_pos, all_matching_keys_h, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, data_with_suggestions, avrg='yes', type_pos_filtering=type_pos_filtering)

    print_not_fully_tagged(pos_tag_filtering, premise_suggestions, hypothesis_suggestions)

    premise_fillers, hypothesis_fillers = return_suggestions(premise_suggestions), return_suggestions(hypothesis_suggestions)

    skip= assert_suggestions_to_skip(premise_fillers, hypothesis_fillers, number_of_minimal_suggestions_common_bt_p_h, words, seed_dataset, item_id, counters, stats)
    if skip:
      return None
    premise_probabilities, hypothesis_probabilities = return_probabilities(premise_suggestions), return_probabilities(hypothesis_suggestions)

    return key, premise_fillers, hypothesis_fillers, premise_probabilities, hypothesis_probabilities, pos, all_matching_keys_p, all_matching_keys_h



def return_fllers_probabilities_pos_positions(data_with_suggestions, premise, hypothesis, pos_p,  pos_h, tok_p, tok_h, pos_to_mask, source_1, source_2, allowed_pos_tags, pos_tag_filtering, prob, nlp, number_batch_for_pos_tagging, save_suggestions_in_file, type_pos_filtering, number_of_minimal_suggestions_common_bt_p_h, seed_dataset, id, counters):
  word2fillers, word2probabilities, positions = [defaultdict(list), defaultdict(list), defaultdict(list)]
  if premise in data_with_suggestions.keys() and hypothesis in data_with_suggestions.keys():
    common_dict, p_positions, h_positions, _ = common(premise, hypothesis, pos_p, pos_h, tok_p, tok_h, pos_to_mask, source_1, source_2)

    singles= return_words_of_2_sentences(premise, hypothesis)
    words = list(common_dict.keys())

    stats={'words_with_not_enough_replacements_inside_loop_utilitary':0}


    for i, word in enumerate(words):
      result = process_word_for_alignment(word, i, common_dict, data_with_suggestions, premise, hypothesis, p_positions, h_positions, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, type_pos_filtering, number_of_minimal_suggestions_common_bt_p_h, words, seed_dataset, id, counters, stats)

      if result is None:
          continue

      key, premise_fillers, hypothesis_fillers, premise_probabilities, hypothesis_probabilities, pos, all_matching_keys_p, all_matching_keys_h = result

      word2fillers[key] = [premise_fillers, hypothesis_fillers]
      word2probabilities[key] = [premise_probabilities, hypothesis_probabilities]
      positions[key] = [all_matching_keys_p, all_matching_keys_h]
  return word2fillers, word2probabilities, positions, words

def return_word_pos(word, pos):
  return f"{word}:{pos}"


def sort_positions(positions):
  return sorted(positions, key=lambda s: int(s.split(':')[0]), reverse=True)

def variant_formation(positions, variant, best_):
  for i in positions:
    start, end= i.split(':')
    variant= variant[:int(start)] + best_ + variant[int(end):]
  return variant


def variants_all_formation(indices, ranked_fillers, premise, hypothesis, positions_to_replace_premise, positions_to_replace_hypothesis, premise_suggestion_for_id_indexing, hypothesis_suggestion_for_id_indexing):
  sentence_variants = []
  for i in indices:
      if i >= len(ranked_fillers):
          continue

      best_ = ranked_fillers[i][0].strip()

      p_variant, h_variant = premise, hypothesis
      p_positions_sorted, h_positions_sorted = sort_positions(positions_to_replace_premise), sort_positions(positions_to_replace_hypothesis)
      p_variant, h_variant=variant_formation(p_positions_sorted, p_variant, best_), variant_formation(h_positions_sorted, h_variant, best_)


      oiringacr = (
        'h' if best_ not in premise_suggestion_for_id_indexing else
        'p' if best_ not in hypothesis_suggestion_for_id_indexing else
        'ph'
        )

      if p_positions_sorted and h_positions_sorted:
          sentence_variants.append((p_variant, h_variant, best_, oiringacr)) #SENTENCE VARIANT STRUCTURE [('This building choir sings to the masses as they sing joyous songs from the book at a building.', 'The building has cracks in the ceiling.', 'building', 'h')]
      else:
          print('❌ Skipped: One of the position lists is empty.')
  return sentence_variants

def calculate_expected_variants(expected_generation, rank_option, label, ranked_fillers):
  """
  Calculate how many variants are expected for `label` given `rank_option`,
  and accumulate that count into `expected_generation[label]`.
  """
  if isinstance(rank_option, int):
      expected_variants = 1 if rank_option < len(ranked_fillers) else 0
  elif isinstance(rank_option, slice):
      expected_variants = len(range(*rank_option.indices(len(ranked_fillers))))
  else:
      expected_variants = 0
  if label in expected_generation:
    expected_generation[label] += expected_variants


def actual_generetion(processed_dataset):
  return Counter(entry['label'] for entry in processed_dataset)

def add_variant_to_dataset(sentence_variants, processed_data, label, id, word_only, pos, premise_start, premise_end, hypothesis_start, hypothesis_end, model_name):
  for idx, (p_variant, h_variant, best_, oiringacr) in enumerate(sentence_variants):
    processed_entry = {
        'id': f"{id}:{word_only}:{pos}:{premise_start}:{premise_end}:{hypothesis_start}:{hypothesis_end}:{best_}:{oiringacr}:{model_id_for_file_name().get(model_name)}",
        'premise': p_variant,
        'hypothesis': h_variant,
        'label': label
        } #processed_entry: {'id': '2677109430.jpg#1r1n:church:NN:5:11:4:10:building:h:bert', 'premise': 'This building choir sings to the masses as they sing joyous songs from the book at a building.', 'hypothesis': 'The building has cracks in the ceiling.', 'label': 'neutral'}

    processed_data.append(processed_entry)


def update_replacement_summary(replacement_summary, id, word_only, pos, positions_to_replace_premise, positions_to_replace_hypothesis):
    """
    Record which positions were replaced for a given word/POS under a given id.
    e.g. {'2677109430.jpg#1r1n': {'church:NN': [['5:11', '83:89'], ['4:10']]}}
    """
    replacement_summary.setdefault(id, {})
    key=return_word_pos(word_only, pos)
    if key in replacement_summary[id]:
        print('RED FLAG — duplicate key, skipping:', return_word_pos(word_only, pos))
    else:
        replacement_summary[id][return_word_pos(word_only, pos)] = [positions_to_replace_premise, positions_to_replace_hypothesis]

    return replacement_summary


def write_general_statistics(output_file, name_file_general, actual_generation, expected_generation, global_words_without_replacements, problems_removed_due_to_low_suggestions):
  """
  prints expacted/actual label counts, no. problems & original words with not enough solution
  """
  with open(name_file_general, "w") as f:
      f.write(f"\nGeneral Statistics for ({output_file}):\n")
      f.write("=" * 50 + "\n")
      f.write("\nLabel Counts:\n")
      for i in ('neutral', 'entailment', 'contradiction'):
          f.write(f"{i}: {actual_generation[i]} (Expected: {expected_generation[i]})\n")

      f.write(f"\nProcessing Issues:\n") #replaced words with not enough solutions, and problems with not en ough solutions
      f.write(f"Words with not enough suggestions: {global_words_without_replacements}\n")
      f.write(f"Problems with not enough suggestion: {problems_removed_due_to_low_suggestions}\n")


def return_sources():
  return 'premise', 'hypothesis'


def return_replacements_ranked(word2fillers, word2probabilities, type_rank_operation, type_rank):
  '''takes a list of replacememts, their prob, type rank operation (union or intersection) and calculate rank
  #Ranked fillers for church:NN:5:11:4:10: [('house', {'average_rank': 0.5, 'ranks': [1], 'average_prob': '1.08e-01', 'individual_probs': ['1.08e-01']}), ('building', {'average_rank': 0.0, 'ranks': [0], 'average_prob': '1.52e-01', 'individual_probs': ['1.52e-01']})]'''
  words = {}
  for w in word2fillers:

      words[w] = ranked_overlap(word2fillers[w], word2probabilities[w], type_rank_operation).items()
      words[w] = sorted(words[w], key=lambda x: x[1][type_rank]) # type of ranking -: average rank or prob, and sort by that
  return words



def print_check_statements(type_eval, type_eval_acro, pos_value):
    print("-" * 40)
    print(f"THE TYPE OF EVAL: {type_eval}")
    print(f"THE TYPE OF EVAL ACRONYM: {type_eval_acro}")
    print(f"THE POS VALUE: {pos_value}")
    print("-" * 40)

def return_base_common_ids(pos, type_of_procesing): #n_nouns_dataset_bothcpAh_ran_var_20.json
  all_base_ids_common_list=set(get_base_ids_from_data(load_data(f"/content/drive/MyDrive/MERGE/variants/samples/snli/final_cpah/{pos}s_dataset_bothcpAh_sel.json"), "flat")) if type_of_procesing!='first' else None
  return all_base_ids_common_list


def assert_minimum_models(pairs_models_datasets, pos_tag):
  """Raise if fewer than two models have suggestions for the given pos_tag."""
  if len(pairs_models_datasets) < 2:
      raise ValueError(f"Need at least two models' suggestions for pos_tag='{pos_tag}'.")


def build_pairs_models_datasets(models, pos_tag, processed_results):
    """ """
    entries = [(model, processed_results.get(f"{model}_{pos_tag}", [])) for model in models]
    pairs_models_datasets = [(data, model) for model, data in entries if data]
    missing = [f"{model}_{pos_tag}" for model, data in entries if not data]
    return pairs_models_datasets, missing

def merge_and_analyze_from_results(
    processed_results: dict,
    *,
    models: list[str],
    pos_tag: str,
    type_evaluation: str,
    min_count: int,
    name: str | None = None,
    opposite: bool = True,
    label_for_shared_suggestions: str = "general",
):

    pairs_models_datasets, missing = build_pairs_models_datasets(models, pos_tag, processed_results)
    assert_minimum_models(pairs_models_datasets, pos_tag)
    (dataset1, source1), (dataset2, source2), *others = pairs_models_datasets
    return merge_and_analyze_datasets(dataset1, source1, dataset2, source2, pos_tag, type_evaluation, min_count, name, opposite,  others, label_for_shared_suggestions)

def build_name(pos_part, suffix, models_dictionary, model_tested, model_number, size, split_str, type_dataset, number_inflation):
  return f"{models_dictionary.get(model_tested)}.{model_number}.{pos_part}.{size}.{split_str}{type_dataset}.{suffix}.{number_inflation}.json"


def generate_output_filenames(suggestion_file, models_dictionary, pos_dicitonary, number_inflation, type_dataset):
    """
    suggestion_file structure  /.../robert-base-cased.1.noun.200.test.json
    returns output_processed_dataset
    """
    parts = os.path.basename(suggestion_file).split('.')

    if len(parts) < 6:
        raise ValueError("Filename does not follow expected naming convention.")
    name = partial(build_name, models_dictionary=models_dictionary, model_tested=parts[0], model_number=parts[1], size=parts[3], split_str=parts[4][:2], type_dataset=type_dataset, number_inflation=number_inflation)

    output_processed_dataset = name(pos_dicitonary.get(parts[2].lower()), "inf")
    return output_processed_dataset

def process_dataset(data_with_suggestions,
                    output_file,
                    pos_to_mask,
                    problems_w_pos_tok,
                    seed_dataset,
                    nlp, # i think again this should be by default
                    number_batch_for_pos_tagging, # this as well
                    mock_test: str=None,
                    pos_tag_filtering:str=None,
                    rank_option='top',
                    save_suggestions_in_file: str = None,
                    prob_filtering:str=None,
                    type_rank:str=None,
                    type_rank_operation:str=None,
                    model_name:str=None,
                    type_pos_filtering:str=None,
                    ):

    """
    Matches premise and hypothesis from second_data with data_with_suggestions, replaces words, applies ranking,
    transforms the dataset, and optionally groups it by POS tags.

    data_with_suggestions: dataset containing the filler/replacement suggestions to merge in
    output_file: name of the output file
    pos_to_mask: pos tag to be looked for to be common between premise and hypothesis
    problems_w_pos_tok: [FILL IN — not documented in the old docstring, describe what this tracks/holds]
    seed_dataset: the initial (unmasked) dataset to match suggestions against
    nlp: loaded NLP pipeline/model used for POS tagging
    number_batch_for_pos_tagging: number of sentences to process at a time for POS tagging
    mock_test: str = None, if specified restricts processing to a single mock test case
    pos_tag_filtering: str = None, if == 'yes' the created dataset will only contain the same pos tags as the initial masked word
    rank_option = 'top': rank function — 'top' for highest-ranked, int for specific rank, slice for multiple replacements
    save_suggestions_in_file: str = None, if specified will replace the entries in data_with_suggestions with filtered ones that match the suggestion pos tags
    prob_filtering: str = None, if 'yes' suggestions with probabilities equal or higher than the original replaced word are kept
    type_rank: str = None, 'average_rank' ranks suggestions based on their average position in the suggested words for premise and hypothesis; 'average_prob' ranks suggestions based on their average probabilities
    type_rank_operation: str = None, when ranking words, the lists of premise and hypothesis can be united or intersected
    model_name: str = None, [FILL IN — not documented in the old docstring, describe what this is used for, e.g. model used for re-tagging suggestions]
    type_pos_filtering: str = None, whether to allow all pos tags of the open class category, or only the original pos tag of the replaced word — values: 'class', 'pos_tag'
    """
    check_type_return_correct(type_rank)
    processed_data, replacement_summary  = [], {}
    counters = {'global_words_without_replacements': 0, 'problems_removed_due_to_low_suggestions': 0}
    allowed_pos_tags = return_pos_additional_noun_tags(pos_to_mask)

    expected_generation = {'neutral': 0, 'entailment': 0, 'contradiction': 0}

    for entry in tqdm(problems_w_pos_tok):
        id, premise, hypothesis, tok_p, pos_p, tok_h, pos_h, label = unpack_entry_dataset(entry)
        if mock_test and id != mock_test:
          continue

        word2fillers, word2probabilities, positions, words = return_fllers_probabilities_pos_positions(data_with_suggestions, premise, hypothesis, pos_p, pos_h, tok_p, tok_h, pos_to_mask,
                                                                                                                 'premise', 'hypothesis', allowed_pos_tags, pos_tag_filtering, prob_filtering, nlp, number_batch_for_pos_tagging, save_suggestions_in_file,
                                                                                                                 type_pos_filtering, 0, seed_dataset, id, counters)
        words = return_replacements_ranked(word2fillers, word2probabilities, type_rank_operation, type_rank)
        for w, ranked_fillers in words.items():

            parts = w.split(':')
            if len(parts) != 6:
                print(f"Unexpected key format: {w}")
                continue
            word_only, pos, premise_start, premise_end, hypothesis_start, hypothesis_end = parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])

            positions_to_replace_premise = [':'.join(i.split(':')[:2]) for i in positions[w][0]] # Positions to replace (premise): ['5:11', '83:89']
            positions_to_replace_hypothesis = [':'.join(i.split(':')[:2]) for i in positions[w][1]] #Positions to replace (hypothesis): ['4:10']

            replacement_summary = update_replacement_summary(replacement_summary, id, word_only, pos, positions_to_replace_premise, positions_to_replace_hypothesis)

            for key, value in word2fillers.items():                                    #Processing word2fillers entry: key church:NN:5:11:4:10 -> value [[], ['building', 'house']] key _> value
              to_look = ':'.join(key.split(':')[:2])
              if to_look == return_word_pos(word_only, pos):
                premise_suggestion_for_id_indexing, hypothesis_suggestion_for_id_indexing = value[0], value[1]

            calculate_expected_variants(expected_generation, rank_option, label, ranked_fillers)
            indices = [rank_option] if isinstance(rank_option, int) else range(*rank_option.indices(len(ranked_fillers))) #modifies rank/_option to be applicable to ranekd fillers
            sentence_variants = variants_all_formation(indices, ranked_fillers, premise, hypothesis, positions_to_replace_premise, positions_to_replace_hypothesis, premise_suggestion_for_id_indexing, hypothesis_suggestion_for_id_indexing)
            add_variant_to_dataset(sentence_variants, processed_data, label, id, word_only, pos, premise_start, premise_end, hypothesis_start, hypothesis_end, model_name)
    if output_file:
      write_general_statistics(output_file,  f"{output_file}_general_stats.txt", actual_generetion(processed_data), expected_generation, counters['global_words_without_replacements'], counters['problems_removed_due_to_low_suggestions'])

    if output_file:
      create_json_from_data(processed_data, output_file)
    if save_suggestions_in_file:
      create_json_from_data(data_with_suggestions, save_suggestions_in_file)

    return processed_data, replacement_summary

def return_models_short_to_acronyms_mapping():
  return {"bertb": "bb", "robertab": "rb", "deberta": "d","albertb": "ab","robertal": "rl","electrl": "el","bertl": "bl","albertxxl": 'axxl',"bartl": 'bal',"bartb": "bab", "electrab": 'elb'} #}


def return_class_to_short_class_mapping():
  return {"noun": "n", "verb": "v", "adjective": "adj", "adverb": "adv"}



def model_id_for_file_name():
  return {"bertv": "bert" , "robertab": "roberta", "deberta": "deberta", "albertb": "albert", "robertal": 'robertal', "electrl": "electral", "bertl": "bertl", "albertxxl": 'albertxxl', "bartl": 'bartl', "bartb": "bartb", "electrab": 'electrab'}



def return_filtered_sent_for_process_function(original_dataset, split, mapping, pos_to_mask,
                                               dataset_filtered_already_for_shared_words, min_common_words,
                                               num_sentences_to_process_dataset, num_sentences_compliant_criteria,
                                               number_of_maximum_annotators, minimum_no_words_premise,
                                               minimum_no_words_hypothesis, if_ids_exist):
    '''
    data_with_suggestions: data with suggestions of models
    if_ids_exist: optional id list to sample from the eligible sentences further
    dataset_filtered_already_for_shared_words: None (defult), or file that has the seed sentences for the inflated dataset we want to obtain, othewrise the SNLI dataset will be filtered
                      for certain criteria to find the seed sentences
    original_dataset: dataset to be filtered, e.g. SNLI
    split: split of dataset to be filtered, e.g. test
    neutral_number: number for neutral label/ string 'neutral' (will appear as ref in the inf dataset)
    entailment_number: number for entailment label/ string 'entailment' (will appear as ref in the inf dataset)
    contradiction_number: number for contradiction label/ string 'contradiction' (will appear as ref in the inf dataset)
    number_of_maximum_annotators: number of maximum annotators (anything lower than)
    minimum_no_words_premise: minimum number of words in premise
    minimum_no_words_hypothesis: minumum numberr of words in hypo'''
    SNLI_filtered_2 = dataset_filtered_already_for_shared_words if dataset_filtered_already_for_shared_words else filter_snli(original_dataset[split], mapping, pos_to_mask, min_common_words, num_sentences_to_process_dataset, num_sentences_compliant_criteria, number_of_maximum_annotators, minimum_no_words_premise, minimum_no_words_hypothesis)

    if if_ids_exist:
        SNLI_filtered_2 = return_key_value_of_dict_in_list(SNLI_filtered_2, if_ids_exist)
        print(f"Filtered length: {len(SNLI_filtered_2)}")

    item_data_plus_tok_pos = pos_toks_extract_from_dataset(SNLI_filtered_2, mapping)
    seed_dataset, labels_sample = process_unmasked_dataset(item_data_plus_tok_pos, 'neutral', 'entailment', 'contradiction', include_id=True)  # <-- fixed name
    return item_data_plus_tok_pos, seed_dataset, labels_sample





def process_model_pos_combination(model, pos, formation_type, dataset, split, mapping,
                                   min_common_words, num_sentences_to_process_dataset,
                                   num_sentences_compliant_criteria, an_no, words_premise_min,
                                   words_hypothesis_min, type_of_procesing, nlp, no_sentences_classified_batch,
                                   mock_test_value, pos_value, number_suggestions_to_be_considered,
                                   prob_value, pos_tag_scope,
                                   number_suggestions_required_per_id,
                                   ):
    """
    Runs the full suggestion pipeline for one (model, pos) pair: filters seed sentences,
    loads that model's suggestions, processes the dataset, writes the per-combo output
    files, and returns the resulting variants.
    """
    all_base_ids_common_list = return_base_common_ids(pos, type_of_procesing)

    problems_w_pos_tok, seed_dataset, labels_sample = return_filtered_sent_for_process_function(
        dataset, split, mapping, pos, None, min_common_words,
        num_sentences_to_process_dataset, num_sentences_compliant_criteria, an_no,
        words_premise_min, words_hypothesis_min, all_base_ids_common_list)

    #FIXME: add function
    suggestion_file = f"/content/drive/MyDrive/MERGE/variants/samples/snli/tagged_sugg/{model}.1.{pos}.200.test.json"
    data_suggestions = load_data(suggestion_file)

    file_name_processed_dataset = generate_output_filenames(
        suggestion_file, models_dictionary=return_models_short_to_acronyms_mapping(),
        pos_dicitonary=return_class_to_short_class_mapping(),
        number_inflation=number_suggestions_required_per_id, type_dataset=formation_type)

    variants, dict_sumarr = process_dataset(
        data_with_suggestions=data_suggestions,
        output_file=file_name_processed_dataset,
        pos_to_mask=pos,
        problems_w_pos_tok=problems_w_pos_tok,
        seed_dataset=seed_dataset,
        nlp=nlp,
        number_batch_for_pos_tagging=no_sentences_classified_batch,
        mock_test=mock_test_value,
        pos_tag_filtering=pos_value,
        rank_option=number_suggestions_to_be_considered,
        save_suggestions_in_file=f'{model}_{pos}_{formation_type}_pos.json',
        prob_filtering=prob_value,
        type_rank='average_prob',
        type_rank_operation='union',
        model_name=model,
        type_pos_filtering=pos_tag_scope
    )

    create_json_from_data(dict_sumarr, f'replacements_positions_{pos}_{formation_type}.json')
    return variants


def run_model_pos_grid(models, pos_to_mask, formation_type, dataset, split, mapping, min_common_words,
                        num_sentences_to_process_dataset, num_sentences_compliant_criteria, an_no,
                        words_premise_min, words_hypothesis_min, type_of_procesing, nlp,
                        no_sentences_classified_batch, mock_test_value, pos_value,
                        number_suggestions_to_be_considered, prob_value, pos_tag_scope,
                        number_suggestions_required_per_id):
    """Runs process_model_pos_combination for every (model, pos) pair, keyed as 'model_pos'."""
    processed_results = {}
    for model in models:
        for pos in pos_to_mask:
            variants = process_model_pos_combination(
                model, pos, formation_type, dataset, split, mapping, min_common_words,
                num_sentences_to_process_dataset, num_sentences_compliant_criteria, an_no,
                words_premise_min, words_hypothesis_min, type_of_procesing, nlp,
                no_sentences_classified_batch, mock_test_value, pos_value,
                number_suggestions_to_be_considered, prob_value, pos_tag_scope,
                number_suggestions_required_per_id)
            processed_results[f"{model}_{pos}"] = variants
    return processed_results

def wrapper_merge_across_models(pos, processed_results, models, type_evaluation, min_count, opposite, label_for_shared_suggestions, name_pos_files, output_directory):
    pos_, average_pos = merge_and_analyze_from_results(processed_results, models=models, pos_tag=pos, type_evaluation=type_evaluation, min_count=min_count, opposite=opposite, label_for_shared_suggestions=label_for_shared_suggestions)
    if output_directory != None:
      create_json_from_data(pos_, output_directory+name_pos_files[pos]) #here the same
    return pos_, average_pos



def maybe_merge_datasets(merge_datasets, pos_to_mask, formation_type, processed_results, models,
                          number_suggestions_required_per_id, replacement_scope,
                          label_type_aggregation_across_models, output_directory):
    """If merge_datasets is truthy, merges suggestions across models for every pos tag."""
    if not merge_datasets:
        return
    name_pos_files = {pos: f"{pos}_dataset_{formation_type}_sel.json" for pos in pos_to_mask}
    merged_by_pos = {}
    for pos in pos_to_mask:
        pos_return, _=wrapper_merge_across_models(pos, processed_results, models, formation_type,
                                     number_suggestions_required_per_id, replacement_scope,
                                     label_type_aggregation_across_models, name_pos_files,
                                     output_directory)
        merged_by_pos[pos] = pos_return
    return pos_return

def assert_problem_in_one_of_the_datasets():
  print('premise and hypothesis not in any of the 3 datasets, check if the p and h are part of the same problem or part of SNLI-test and MNLI-/m and -mm dev')


def re_format_dataset(dataset):
  return {k: [v['p'], v['h']] for k,v in dataset.items()}

def re_format_S2A(S2A, premise, hypothesis):
  re_formatted=defaultdict(dict)
  re_formatted[premise]=S2A[premise]
  re_formatted[hypothesis]=S2A[hypothesis]
  return re_formatted

def identify_dataset(premise, hypothesis):
    for dataset in ['SNLI', 'SNLI-m', 'SNLI-mis']:
        dataset_full, S2A = return_dataset_mapping_function(dataset)
        dataset_split = dataset_full[return_splits_per_dataset(dataset)]
        re_formated_dataset = re_format_dataset(dataset_split)
        for key, v in re_formated_dataset.items():
            if premise in v and hypothesis in v:
                return dataset, dataset_split[key], key, S2A
    assert_problem_in_one_of_the_datasets()
    return None, None, None, None


def return_splits_per_dataset(dataset_name):
  return 'test' if dataset_name == 'SNLI' else 'dev'


def return_config_for_formation():
  '''returns config values of type boolened considering the formation type'''
  return {'prob':  ("yes", None), "pos":   ("no",  "yes"), "none":  ("no",  None), "both":  ("yes", "yes")}

def return_config_replecement_file_generetion_demo():
      return {'min_commom_words_': 1,
              'number_replacements': 200,
              'exclude_words_part_of_p_and_h': 'yes',
              'agreement_annotators_no': 4,
              'number_words_premise': 1,
              'number_words_hypothesis': 1,
              'num_sentences_to_process_dataset': None,
              'num_sentences_compliant_criteria': None,
              'id_mock_test': None,
              'add_id_to_dataset':True,
              'output_file': None}

def standard_settings_demo():
    '''Standard/default args for process_dataset calls inside merge_demo.
    Returns 3 dicts  with **settings for the process_dataset call.'''
    return {
        'rank_option': slice(200),
        'save_suggestions_in_file': None,
        'type_rank': 'average_prob',
        'type_rank_operation': 'union',
        'number_batch_for_pos_tagging': 1024,
        'mock_test': None,
    }, {
        'pos_tag_scope': 'class',
        'replacement_scope': True,
        'folder_ouptut': '/content/',
    },  {
        'merge_datasets': True,
        'label_type_aggregation_across_models': 'general',
        'output_directory': None,
    }, spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "attribute_ruler"]), return_config_replecement_file_generetion_demo()

def merged_print_demo(merged):
    for pos, items in merged.items():
        print(f"=== POS: {pos} ===")
        for i in items:
            print('THE ID', i['id'])
            print('the premise', i['premise'])
            print('the hypothesis', i['hypothesis'])
            print('the label', i['label'])
            print('------------')

def transform_list_model_card(model_cards):
  return [return_name_models_suggestions().get(model_card) for model_card in model_cards]


def build_eval_type(formation_type, pos_tag_scope, replacement_scope, folder_ouptut, num_suggestions):
    '''returns name of file of variants considering the type of fomration of variants'''
    if formation_type not in return_config_for_formation():
        return None, None
    formation_type += "c" if pos_tag_scope == "class" else "pos"
    formation_type += "pAh" if replacement_scope is True else "pph"
    return f"{folder_ouptut}all.1.{formation_type}.te.inf.{num_suggestions}.json", formation_type


def register_eval_type(formation_type_user, pos_tag_scope, replacement_scope, folder_ouptut, number_suggestions_required_per_id):
    """returns filename tyhpe considering arguments set, i.e. type of operation for variants"""
    file_to_predict, formation_type_formatted = build_eval_type(formation_type_user, pos_tag_scope, replacement_scope, folder_ouptut, number_suggestions_required_per_id)
    return formation_type_formatted


def return_data_for_1_problem_demo(premise, hypothesis):
  '''idnetifies from which dataset an item is'''
  dataset_name, dataset_filtered, item_id, S2A=identify_dataset(premise, hypothesis)
  tok_item_S2A=re_format_S2A(S2A, premise, hypothesis)
  return dataset_name, dataset_filtered, item_id, tok_item_S2A

def merge_demo(NLI_problem_p, NLI_problem_H, model_cards, pos_tags, minimum_variants, formation_type_user='both'):
    pd_settings, et_settings, mm_settings, nlp, create_replacements_config = standard_settings_demo()
    prob_value, pos_value = return_config_for_formation()[formation_type_user]
    formation_type_formatted = register_eval_type(formation_type_user, et_settings['pos_tag_scope'], et_settings['replacement_scope'], et_settings['folder_ouptut'], minimum_variants)
    dataset_name, dataset_item, item_id, mapping = return_data_for_1_problem_demo(NLI_problem_p, NLI_problem_H)
    if dataset_name is None:
        return None, None, None
    split = return_splits_per_dataset(dataset_name)
    item_data_plus_tok_pos = pos_toks_extract_from_dataset({item_id: dataset_item}, mapping)
    seed_dataset, _ = process_unmasked_dataset(item_data_plus_tok_pos, 'neutral', 'entailment', 'contradiction', include_id=True)
    processed_results = {}

    for model_card in model_cards:
        model_name_ = return_name_models_suggestions()[model_card]
        print('PROCESSING MODEL', model_name_)
        for pos_to_mask in pos_tags:

            _, dict_res_new = create_filler_file(
                model_card, {split: {item_id: dataset_item}}, split, pos_to_mask, create_replacements_config['min_commom_words_'], create_replacements_config['number_replacements'], mapping, create_replacements_config['exclude_words_part_of_p_and_h'],
                create_replacements_config['agreement_annotators_no'], create_replacements_config['number_words_premise'], create_replacements_config['number_words_hypothesis'],
                create_replacements_config['num_sentences_to_process_dataset'], create_replacements_config['num_sentences_compliant_criteria'],
                create_replacements_config['id_mock_test'], create_replacements_config['add_id_to_dataset'],
                create_replacements_config['output_file']
            )

            data_suggestions = dict_res_new

            variants, _= process_dataset(
                data_with_suggestions=data_suggestions,
                output_file=None,
                pos_to_mask=pos_to_mask,
                problems_w_pos_tok=item_data_plus_tok_pos,
                seed_dataset=seed_dataset,
                nlp=nlp,
                number_batch_for_pos_tagging=pd_settings['number_batch_for_pos_tagging'],
                mock_test=pd_settings['mock_test'],
                pos_tag_filtering=pos_value,
                rank_option=pd_settings['rank_option'],
                save_suggestions_in_file=pd_settings['save_suggestions_in_file'],
                prob_filtering=prob_value,
                type_rank=pd_settings['type_rank'],
                type_rank_operation=pd_settings['type_rank_operation'],
                model_name=model_name_,
                type_pos_filtering=et_settings['pos_tag_scope']
                )
            processed_results[f"{model_name_}_{pos_to_mask}"] = variants

    merged = maybe_merge_datasets(
        mm_settings['merge_datasets'], pos_tags, formation_type_formatted, processed_results, transform_list_model_card(model_cards),
        minimum_variants, et_settings['replacement_scope'],
        mm_settings['label_type_aggregation_across_models'], mm_settings['output_directory'])

    merged_print_demo(merged)

    return processed_results, merged
