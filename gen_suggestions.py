#re-factored code have to still re write the descirption of each function
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
    preceding_text = sentence[:matchy.start()]
    offset = (len(preceding_text), len(preceding_text) + len(token))
    return offset, preceding_text


def extract__pos_position(pos_tags, tokens, source, pos_type, sentence):
    dictionary_positions, token_counts = {}, defaultdict(int)
    valid_tags = return_pos_tag_for_class(pos_type)

    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
        matches = extract_pos_position_matches(i, pos, valid_tags, pos_type, tokens, sentence)
        if not matches:
            continue

        offset, preceding_text = return_offset_preceding_text(token_counts, matches, token, sentence)
        if token not in dictionary_positions:
            dictionary_positions[token] = {
                'positions': [offset], 'pos': pos, 'source': source, 'preceding_text': preceding_text
            }
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
    premise = mapping[problem['p']]
    hypothesis = mapping[problem['h']]
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
  '''  ### list_filtered structure {'3827316480.jpg#0r1e': {'g': 'entailment', 'pid': '3827316480.jpg#0r1e', 'cid': '3827316480.jpg#0', 'lnum': 5, 'lcnt': Counter({'entailment': 5}), 'ltype': '500', 'p': 'One tan girl with a wool hat is running and leaning over an object, while another person in a wool hat is sitting on the ground.', 'h': 'A tan girl runs leans over an object'},'''
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
    id_mock_test: str=None,
    add_id_to_dataset: bool = False,
    modifying_function:bool=None,
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
    results_dict = {}

    dataset = dataset[split]
    SNLI_filtered_2 = filter_snli(dataset, mapping, pos_to_mask, min_common_words, num_sentences_to_process_dataset, num_sentences_compliant_criteria, number_of_labels, number_words_premise, number_words_hypothesis)

    filtered_list_1 = pos_toks_extract_from_dataset(SNLI_filtered_2, mapping)

    seed_dataset, lab = process_unmasked_dataset(filtered_list_1, no_neutral, no_ential, no_contradiction, include_id=add_id_to_dataset)
    model, tokenizer = return_model_tok(model_name)
    for p in tqdm(filtered_list_1):
        id, premise, hypothesis, tok_p, pos_p, tok_h, pos_h = (p['id'], p['premise'], p['hypothesis'], p['p_t'], p['p_p'], p['h_t'], p['h_p'] )
        if mock_test and id != id_mock_test:
          continue
        common_tokens_dictionary, p_off, h_off, all_nouns_singles = common(premise, hypothesis, pos_p, pos_h, tok_p, tok_h, pos_to_mask, source_1, source_2, exclude_words_part_of_p_and_h)
        p_off_filler = suggest_mask_fillers(premise, p_off, model, tokenizer, all_nouns_singles, common_tokens_dictionary, num_filler_suggestions)
        merge_filler_results(results_dict, premise, p_off_filler)
        h_off_filler = suggest_mask_fillers(hypothesis, h_off, model, tokenizer, all_nouns_singles, common_tokens_dictionary, num_filler_suggestions)
        merge_filler_results(results_dict, hypothesis, h_off_filler)

    create_json_from_data(results_dict, output_file)
    return seed_dataset, results_dict

