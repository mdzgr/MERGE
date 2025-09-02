from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from typing import Dict, Callable, List, Tuple, List, Any
import nltk, re, os, json, string, glob, sys, torch, random
import numpy as np
from nltk import pos_tag, word_tokenize
from tqdm import tqdm
import pandas as pd
from collections import defaultdict, Counter
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from typing import Dict,Tuple
import datasets, evaluate
from evaluate import evaluator
from google.colab import files
import torch.nn.functional as F
from contextlib import redirect_stdout
from wordfreq import zipf_frequency
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing_extensions import final
from itertools import zip_longest
from statistics import mean



def extract__pos_position(pos_tags, tokens, source, pos_type, sentence):
    ##not double-checked
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
    #not double_checked
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
                             #not double-checked
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
    ##not double-checked
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
   #not double-checked
    '''function to extract sentences that adhere to certain criteria, can't be used for combined pos tags'''
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
  # print(list_filtered)
  for k, p in list_filtered.items(): ########this is not necessary but i will leave it as it is for now
      base_id = k[:-1]
      version = k[-1]
      grouped_problems[base_id][version] = p
  print('the grouped problems',grouped_problems)
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
    #not double-checked
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
    #not double-checked
    """
   filter out unwanted words from json file
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


def flatten_dataset(data):
    '''creates a list of unique items {p,h,l},
    useful when having a nested random dataset'''
    seen_ids = set()
    flattened = []

    for split_name, examples in data.items():
        for example in examples:
            example_id = example["id"]
            if example_id not in seen_ids:
                seen_ids.add(example_id)
                flattened.append(example)

    return flattened
def filter_suggestions_by_contextual_pos(
    suggestions, # alist of suggestions
    original_sentence, #the original sentence
    start_idx, #where to put the suggestions
    end_idx, #where that ends
    allowed_pos_tags, #what are the allowed pos tags
    nlp, #the nlp pipeline
    batch_size_no # the batch size for how many suggestions will be tagged
    ):
    """
    tags suggestions via spacy nlp pipeline
    replace themn in sentence > tags suggestions in context
    > returns a list with filtered suggestions with their pos tags
    and a count of the pos tags

    """

    docs=[]
    words=[]
    for suggestion in suggestions:
        word = suggestion.split(":")[0] #split to get only the word
        temp_sentence = original_sentence[:start_idx] + word + original_sentence[end_idx:]  #replace it in the sentence !!! why is this based on positions and not regex?
        docs.append(temp_sentence) #append variants ctreated
    for doc in nlp.pipe(docs, batch_size=batch_size_no): # use pipleline to process in batches
      for token in doc:
          if token.idx == start_idx: # if the character onset == the onset of the replaced word
              tokens_with_tags=token.text+':'+token.tag_ #get the token, add the ta
              words.append(tokens_with_tags) #append it to a list

    pos_dict = {w.split(":")[0]: w.split(":")[1] for w in words} # make a dict word: pos tag
    pos_counts = Counter()
    filtered=[]
    for suggestion in suggestions: #for each suggestion
      word = suggestion.split(":")[0] #split get token
      prob=suggestion.split(":")[1] #split get probability
      if word in pos_dict: # if the token is in the list of tagged words
        filtered.append(f"{word}:{prob}:{pos_dict[word]}") #form a new entry for the word with token:prob:pos tag of the word
        pos_counts[pos_dict[word]] += 1 #add the count to pos count
    return filtered, pos_counts




def write_general_statistics(output_file, name_file_general,
                            actual_generation, expected_generation,
                            global_words_without_replacements, problems_removed_due_to_low_suggestions):
    """
    Writes in file:
    label counts, expacted vs. obtained
    no. problems without enough sugestions
    no. replaced words without enough suggestions
    """
    with open(name_file_general, "w") as f:
        f.write(f"\nGeneral Statistics for ({output_file}):\n") #name of file  generated for
        f.write("=" * 50 + "\n")

        f.write("\nLabel Counts:\n") #expacted vs actual generation, should be the same
        f.write(f"Neutral: {actual_generation['neutral']} (Expected: {expected_generation['neutral']})\n")
        f.write(f"Entailment: {actual_generation['entailment']} (Expected: {expected_generation['entailment']})\n")
        f.write(f"Contradiction: {actual_generation['contradiction']} (Expected: {expected_generation['contradiction']})\n")

        f.write(f"\nProcessing Issues:\n") #replaced words with not enough solutions, and problems with not en ough solutions
        f.write(f"Words with not enough solutions: {global_words_without_replacements}\n")
        f.write(f"Problems with not enough solutions: {problems_removed_due_to_low_suggestions}\n")

def write_prob_statistics(output_file, name_file_prob,
                         number_hypothesis_suggestions_remaining_all_filtering, no_hypothesis_words_with_10more_suggestions_higher_probability,
                         total_words_h,  number_premise_suggestions_remaining_all_filtering, no_premise_words_with_10more_suggestions_higher_probability, total_words_p,
                         words_replaced_p, words_replaced_h, premise_replaced_words_had_no_probability, hypothesis_replaced_words_had_no_probability, average_prob_replaced_hypothesis, average_prob_replaced_premise):
    """
    Writes file, when prob=='yes' and average_pos=='no'.

    no. of words processed

    """
    with open(name_file_prob, "w") as f:
        f.write(f"\nProbability Statistics for ({output_file}):\n")
        f.write("=" * 50 + "\n")

        f.write(f"Replaced words with no prob premise: {premise_replaced_words_had_no_probability}\n")
        f.write(f"Replaced words with no prob hypothesis: {hypothesis_replaced_words_had_no_probability}\n")
        f.write(f"Words replaced premise: {words_replaced_p}\n")
        f.write(f"Words replaced hypothesis: {words_replaced_h}\n")
        f.write(f"Total words processed premise: {total_words_p}\n")
        f.write(f"Total words processed hypothesis: {total_words_h}\n")

        if total_words_p > 0 and total_words_h > 0:
            if words_replaced_p > 0:
                average_original_word_premise = average_prob_replaced_premise / words_replaced_p
                f.write(f"\nAverage original probability of word premise: {average_original_word_premise:.4f}\n")
            if words_replaced_h>0:
                average_original_word_hypothesis = average_prob_replaced_hypothesis / words_replaced_h
                f.write(f"Average original probability of word hypothesis: {average_original_word_hypothesis:.4f}\n")

            average_prob_premise = number_premise_suggestions_remaining_all_filtering / total_words_p
            average_prob_hypothesis = number_hypothesis_suggestions_remaining_all_filtering / total_words_h
            f.write(f"\nAverage no of premise suggestions after prob filtering: {average_prob_premise:.2f}\n")
            f.write(f"Average no of hypothesis suggestions after prob filtering: {average_prob_hypothesis:.2f}\n")

        f.write(f"\nPremise - Words with 10+ better suggestions: {no_premise_words_with_10more_suggestions_higher_probability}\n")
        f.write(f"Hypothesis - Words with 10+ better suggestions: {no_hypothesis_words_with_10more_suggestions_higher_probability}\n")


def write_pos_statistics(output_file, name_file_pos, allowed_pos_tags,
                        overall_count_for_pos_p, overall_count_for_pos_h,
                        count_for_most_common_words_their_pos_tag_p, count_for_most_common_words_their_pos_tag_h,
                        num_sentences_both, premise_diff_after_pos_filter, hypothesis_diff_after_pos_filter,
                        pos_to_mask):
    """
    Write POS tag related statistics when average_pos=='yes'.
    Returns the counters for further processing if needed.
    """

    with open(name_file_pos, "w") as f:
        f.write(f"\nPOS Tag Statistics for ({output_file}):\n") # for file x
        f.write(f"POS tags accepted: {allowed_pos_tags}\n") # with the following tags accepted
        f.write("=" * 50 + "\n")

        for label, counts in [("premise", overall_count_for_pos_p), ("hypothesis", overall_count_for_pos_h)]: #################################################
            f.write(f"\nAverage POS tag counts per sentence ({label}):\n")
            for tag, total in counts.items():
                avg = total / num_sentences_both if num_sentences_both else 0
                f.write(f"{tag}: {avg:.2f}\n")

        f.write(f"\n{'='*50}\n")
        f.write("MOST COMMON WORDS BY POS TAG\n")
        f.write(f"{'='*50}\n")

        premise_filtered = {k: v for k, v in count_for_most_common_words_their_pos_tag_p.items() ###########why is this necessary here da faq
                          if k.split(':')[1] in allowed_pos_tags}
        hypothesis_filtered = {k: v for k, v in count_for_most_common_words_their_pos_tag_h.items()
                             if k.split(':')[1] in allowed_pos_tags}

        premise_count = Counter(premise_filtered).most_common()
        hypothesis_count = Counter(hypothesis_filtered).most_common()

        f.write(f"\nTop 20 most common words in PREMISE with allowed POS tags:\n")
        f.write("-" * 60 + "\n")
        for word_pos, count in Counter(premise_filtered).most_common(20):
            word, pos = word_pos.split(':')
            freq = zipf_frequency(word, 'en')
            avg_per_sentence = count / num_sentences_both if num_sentences_both else 0
            f.write(f"{word:15} | {pos:8} | Count: {count:4} | Freq: {freq:4} | Avg: {avg_per_sentence:.2f}\n")
        word_count_freq_premise = []
        word_count_freq_hypothesis = []

        for word_pos, count in premise_count:
            word, pos = word_pos.split(':')
            avg_per_sentence = count / num_sentences_both if num_sentences_both else 0
            freq = zipf_frequency(word, 'en')
            new_entry = f"{word}:{pos}:{count}:{freq}:{avg_per_sentence}"
            word_count_freq_premise.append(new_entry)

        for word_pos, count in hypothesis_count:
            word, pos = word_pos.split(':')
            freq = zipf_frequency(word, 'en')
            avg_per_sentence = count / num_sentences_both if num_sentences_both else 0
            new_entry = f"{word}:{pos}:{count}:{freq}:{avg_per_sentence}"
            word_count_freq_hypothesis.append(new_entry)


        f.write(f"\nTop 20 most common words in HYPOTHESIS with allowed POS tags:\n")
        f.write("-" * 60 + "\n")
        for word_pos, count in Counter(hypothesis_filtered).most_common(20):
            word, pos = word_pos.split(':')
            avg_per_sentence = count / num_sentences_both if num_sentences_both else 0
            f.write(f"{word:15} | {pos:8} | Count: {count:4} | Avg: {avg_per_sentence:.2f}\n")

        f.write(f"\nWORDS in premise top 20 not shared with hypothesis top 20:\n")
        hypothesis_top_20 = [word_pos for word_pos, count in Counter(hypothesis_filtered).most_common(20)]
        for word_pos, count in Counter(premise_filtered).most_common(20):
            if word_pos not in hypothesis_top_20:
                word, pos = word_pos.split(':')
                f.write(f"{word}\n")
        if num_sentences_both:
            f.write(f'\nProcessing count: {num_sentences_both}\n')
            f.write(f"Premise differences: {premise_diff_after_pos_filter}\n")
            f.write(f"Hypothesis differences: {hypothesis_diff_after_pos_filter}\n")
            f.write(f"Average reduction in premise suggestions for {pos_to_mask}: {premise_diff_after_pos_filter / num_sentences_both:.2f}\n")
            f.write(f"Average reduction in hypothesis suggestions for {pos_to_mask}: {hypothesis_diff_after_pos_filter / num_sentences_both:.2f}\n")
        else:
            f.write("No entries processed for reduction after POS tag filtering.\n")


    return premise_count, hypothesis_count, word_count_freq_premise, word_count_freq_hypothesis

def write_statistics_based_on_parameters(output_file, prob, calculate_average_pos, **kwargs):
    """
    Main function to write descirptive files, can add as many arguments as possible with **kwargs
    """
    results = {}


    write_general_statistics(
        output_file,
        kwargs.get('name_file_general', 'general_stats.txt'),
        kwargs['actual_generation'],
        kwargs['expected_generation'],
        kwargs['global_words_without_replacements'],
        kwargs['problems_removed_due_to_low_suggestions']
    )

    if prob == 'yes' and calculate_average_pos == 'no':
        write_prob_statistics(
            output_file,
            kwargs.get('name_file_prob', 'prob_stats.txt'),
            kwargs['number_hypothesis_suggestions_remaining_all_filtering'],
            kwargs['no_hypothesis_words_with_10more_suggestions_higher_probability'],
            kwargs['total_words_h'],
            kwargs['number_premise_suggestions_remaining_all_filtering'],
            kwargs['no_premise_words_with_10more_suggestions_higher_probability'],
            kwargs['total_words_p'],
            kwargs['words_replaced_p'],
            kwargs['words_replaced_h'],
            kwargs['premise_replaced_words_had_no_probability'],
            kwargs['hypothesis_replaced_words_had_no_probability'],
            kwargs['average_prob_replaced_hypothesis'],
            kwargs['average_prob_replaced_premise']
        )
    if calculate_average_pos == 'yes':
        premise_count, hypothesis_count, word_count_freq_premise, word_count_freq_hypothesis = write_pos_statistics(
            output_file,
            kwargs.get('name_file_pos', 'pos_stats.txt'),
            kwargs['allowed_pos_tags'],
            kwargs['overall_count_for_pos_p'],
            kwargs['overall_count_for_pos_h'],
            kwargs['count_for_most_common_words_their_pos_tag_p'],
            kwargs['count_for_most_common_words_their_pos_tag_h'],
            kwargs['num_sentences_both'],
            kwargs['premise_diff_after_pos_filter'],
            kwargs['hypothesis_diff_after_pos_filter'],
            kwargs['pos_to_mask']
        )
        results.update({
            'premise_count': premise_count,
            'hypothesis_count': hypothesis_count,
            'word_count_freq_premise': word_count_freq_premise,
            'word_count_freq_hypothesis': word_count_freq_hypothesis
        })

    return results


#


def save_dataset_to_json(dataset, directory, filename):
    """
    Saves a dataset (list of dictionaries) to a JSON file within a specified directory.
    """
    os.makedirs(directory, exist_ok=True) #dictionary to save file, if not existent create

    filepath = os.path.join(directory, filename) #dictionary+filepath

    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset successfully saved to {filepath}")


def filter_by_id_prefix(input_files, max_per_prefix=5, suffix="_5", grouping_per_problem="no", typegrouping_file=None):
    #i have no idea what this function is supposed to be?
    #on the one hand it filters only 5 elements, or it filters items that are part of another file
    """
    takes a problem
    and selects an x number of inflated variants per problem
    if typegrouping == no > the randomization is done at the level of each replaced word
    if grouping_per_problem == yes > the randomization is done at the level of the problem

    """
    for file_path in input_files: # for each file in input files
        with open(file_path, 'r') as f:
            data = json.load(f) #open file

        filtered_data = []

        if grouping_per_problem == "no": #if groupin per problem == no

            grouped = defaultdict(list)
            for item in data: #for each item
                prefix = ":".join(item.get("id", "").split(":")[:6]) + ":" #split item and get the 6 first elements (id+word_replaced+start_position+end_position)
                grouped[prefix].append(item)  #group variants by replaced word

            for group in grouped.values(): # for each group of items
                filtered_data.extend(group[:max_per_prefix]) #filter until the number of required variants

        elif grouping_per_problem == "yes" and typegrouping_file: # if group_per_problem ==yes
            with open(typegrouping_file, 'r') as f2: #open a file where we will store the variants
                tobe_data = json.load(f2)


            valid_prefixes = set(":".join(item["id"].split(":")[:1]) for item in tobe_data) #create a set of original ids for items
            print(len(valid_prefixes))
            filtered_data = [item for item in data if (":".join(item["id"].split(":")[:1])) in valid_prefixes] # if the original id of item is in the original id of the second file, filter
            print(len(filtered_data))
        else:
            raise ValueError("Invalid 'grouping_per_problem' argument or missing typegrouping_file.")

        base, ext = os.path.splitext(file_path) # split into filename and extenstion
        output_file = f"{base}{suffix}{ext}" # create name for file: original file + number filtered + extension

        with open(output_file, 'w') as f_out:
            json.dump(filtered_data, f_out, indent=2)

        print(f"Filtered file written to: {output_file}")



def filter_by_number_wrongly_classified_variants(nested_dataset, flat_dataset_path, output_path, no_mistamtches, id_split_char=":"):
    """
    gets the variants for which a model gets at least 10 matches incorrectly
    """

    extracted_ids = set()
    for records in nested_dataset.values(): # for each dataset in a nested collection of datasets
        for record in records:
            extracted_ids.add(record.get("id")) ## add the entries of the dataset to a set that keeps traks of the ids

    with open(flat_dataset_path, "r", encoding="utf-8") as f:
        flat_dataset = json.load(f)

    groups = defaultdict(list)
    for record in flat_dataset: #same for the flat
        if record.get("id") in extracted_ids:
            group_key = record.get("id").split(id_split_char)[0] #append by original id, group by original id
            groups[group_key].append(record)


    filtered_dataset = []
    for group_key, records in groups.items(): # for id: entry
        correct_matches = sum(1 for r in records if r.get("gold_label") == r.get("label_index")) # for each original id, sum how many variants are correctly classified
        total_entries = len(records)

        if correct_matches < total_entries: # if the number of correct matches is smaller
          mismatches = total_entries - correct_matches #calculate how many mismatches
          if mismatches < total_entries * no_mistamtches: #if the number of mismtaches is lower than 10\%
            mismatched_records = [r for r in records if r.get("gold_label") != r.get("label_index")] #get the variants which were not classified correclty
            filtered_dataset.extend(mismatched_records) #add to filtered dataset


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_dataset, f, indent=2, ensure_ascii=False) #save in output file

    print(f"Filtered dataset saved as {output_path} with {len(filtered_dataset)} entries.")


def union_unique_items(file1_path, file2_path, output_path=None):
    with open(file1_path, 'r') as f: ## open first file
        data1 = json.load(f)
    with open(file2_path, 'r') as f: #open second file
        data2 = json.load(f)

    all_items = data1 + data2 #combine them
    unique_items = {}
    for item in all_items: # for each entry
        prefix = ':'.join(item['id'].split(':')[:-1]) #split item by : get all items but the last one
        if prefix not in unique_items: #
            unique_items[prefix] = item ## add that id: entry to unique_items

    result = list(unique_items.values())
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Total items from both files: {len(all_items)}")
    print(f"Unique items after deduplication: {len(result)}")
    return result

def generate_mini_datasets(dataset_file_paths, log_for, num_samples=10, sample_size=20):
    """
    Generate mini-datasets from multiple input dataset files and save them directly to disk.
    Args:
        dataset_file_paths (list): list, file paths of files with variants.
        num_samples (int): Number of nested mini-datasets, default to 10.
        sample_size (int): Number of entries per mini-dataset, default to 20.
    """
    log_path = f"log_summary_{log_for}.txt" #log summary text file

    with open(log_path, "w") as log_file: #open log file
        with redirect_stdout(log_file): # all print statements will be directly written in the log file
            for dataset_path in dataset_file_paths: # for each dataset
                try:
                    with open(dataset_path, 'r') as f:
                        dataset_entries = json.load(f) # open dataset
                    dataset_name1 = os.path.splitext(os.path.basename(dataset_path))[0]
                    # Clean version
                    dataset_name = dataset_name1.replace("_sel", "")


                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error loading dataset from {dataset_path}: {e}")
                    continue

                print(f"\nProcessing dataset: {dataset_name}")#proccesing a certain dataset
                base_id_groups = defaultdict(list) #create default liist
                for entry in dataset_entries: # for each entry
                    base_id = ':'.join(entry['id'].split(':')[:1]) #get original id
                    base_id_groups[base_id].append(entry) #group entries by original id
                print(f"Total base IDs: {len(base_id_groups)}") # print the no. of original base ids
                seed_info = {}
                mini_datasets = defaultdict(list)
                all_unique_entries = {}
                for base_id, entries in base_id_groups.items(): #for original id, its entries
                    original_count = len(entries) #count entries
                    for i in range(1, num_samples+1): #for each mini dataset
                        seed_value = hash(f"{base_id}_{i}") % (2**32) #creates a hash for each mini dataset
                        random.seed(seed_value) #set as a seed
                        if original_count <= sample_size: #if the no. of possible variants == the number of rquried variants
                            sampled_entries = entries #choose them
                        else:
                            sampled_entries = random.sample(entries, sample_size) #otherwise randomly sample from them the no. of required variants
                        mini_name = f"final_{dataset_name}__random_{i}" #name for mini dataste
                        mini_datasets[mini_name].extend(sampled_entries) #for that mini dataset add the entries
                        seed_info[mini_name] = seed_value #add for that dataset the seed value to reproduce later
                print(f"Generated {len(mini_datasets)} mini-datasets for {dataset_name}") #print how many mini datasets
                print(f"\nMini-dataset sizes for {dataset_name}:") #for which datas
                for mini_name, entries in mini_datasets.items(): #for each mini dataset
                    print(f"  {mini_name}: {len(entries)} entries") # print how many entries it has
                for entry_list in mini_datasets.values(): #for each minidataset
                    for entry in entry_list: #for each variant
                        full_id = entry['id'] # get the id
                        all_unique_entries[full_id] = entry # store entry as id: entry
                number_s_required=sample_size
                combined_dataset = list(all_unique_entries.values()) #get values to form combined dataset

                combined_file_path = f"{dataset_name}_ran_{number_s_required}.json" #save mini datasets
                with open(combined_file_path, "w") as f:
                    json.dump(mini_datasets, f, indent=2) #

                seed_file_path = f"{dataset_name}_ran_var_{number_s_required}.json" ##all variants from minidatasets but flattened
                with open(seed_file_path, "w") as f:
                    json.dump(combined_dataset, f, indent=2)

                print(f"Generated {len(seed_info)} mini-datasets for {dataset_name}")
                print(f"Final combined dataset has {len(combined_dataset)} unique entries")
                print(f"Seeds used for mini-datasets in {dataset_name}:")
                for mini_name, seed in seed_info.items(): #print the random seed for replication
                    print(f"  {mini_name}: {seed}")
                print(f"Saved files for dataset '{dataset_name}' in current directory")
    print(f"\nAll processing done. Logs (including seed info) saved in {log_path}")
    return

    def get_base_ids_flat(filepath):
    """gets base (original id) for a flat JSON dataset."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f) #open
    except (FileNotFoundError, json.JSONDecodeError) as e: #if file not found or invalid json structure
        print(f"Error reading file {filepath}: {e}")
        return set()

    base_ids = set()
    for entry in data: # for entry in data
        if 'id' in entry: #if it has id
            parts = entry['id'].split(':') #split th id
            if parts:
                base_ids.add(parts[0]) #get the base of the id
    return base_ids

def split_nested_dataset_by_origin(input_files, origins=['bert', 'both', 'roberta'], origin_type='model'):
    """
    Split nested datasets by origin (premise/hypohtesis/ph or model used for generation), keeping nested structure, and saves
    to JSON files.
    Also print the average number of entries per mini dataset for each origin.

    Args:
        input_files (list of str): List of input JSON file paths.
        origins (list of str): List of expected origins to split by.
    """
    for input_file in input_files: # pass a list of input files, for each file of those
        print(f"Processing file: {input_file}") #say which one is being processed

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f) #open it

        if isinstance(data, list): # if the data is in a list
            nested_data = {'default': data} #put it in a nested dictionary ####why do i put this in a nested dictionary?
        elif isinstance(data, dict): # if it s a dictionary already
            nested_data = data #leave it like this
        else:
            print(f"Error: Unrecognized data format in {input_file}. Skipping.")
            continue

        split_datasets = {origin: defaultdict(list) for origin in origins} #

        for dataset_name, entries in nested_data.items(): #for each dataset and its entires
            for entry in entries: #for its entries
                entry_id = entry['id'] #get id
                parts = entry_id.split(':') #split it
                if origin_type == 'ph': #if the origin type is if the suggestion comes form premise/hypothesis
                    origin = parts[-2] if len(parts) > 1 else None #get the second to last
                else:
                    origin = parts[-1] #otherwise get the last
                if origin in split_datasets:
                    split_datasets[origin][dataset_name].append(entry)
                else:
                    print(f"Warning: Unrecognized origin '{origin}' in ID: {entry_id}")

        base_filename = os.path.splitext(os.path.basename(input_file))[0] #get name before .json
        for origin, data in split_datasets.items(): #the save it
            output_filename = f"{base_filename}_{origin}.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            total_entries = sum(len(v) for v in data.values()) #sum all entries across all mini datasets
            num_mini_datasets = len(data) #no of entries
            avg_entries = total_entries / num_mini_datasets if num_mini_datasets > 0 else 0
            print(f"Saved {output_filename} with {total_entries} entries across {num_mini_datasets} mini datasets. Average: {avg_entries:.2f} entries per mini dataset.")

def combine_json_files(file_paths, output_file):
    """
    Combines multiple JSON files into a single json file.
    """
    combined_data = []
    #for each file
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f) #open
                if isinstance(data, list):
                    combined_data.extend(data) #keeps flat structure
                else:
                    combined_data.append(data) #keeps nested structure
            print(f"Successfully loaded and appended data from {file_path}")
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")
    try:
        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        print(f"Combined data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving combined data to {output_file}: {e}")


def scramble_per_item(item):
    '''scrablmes replacement words and gives back scrambled datasets'''
    id_parts = item["id"].split(":") #split the id of the
    word_to_change = id_parts[-3] #get the replacement word
    # to_add=len(word_to_change)
    pattern = re.compile(rf'(?<![A-Za-z]){re.escape(word_to_change)}(?![A-Za-z])') # !!!!! HAVE TO VERIFY IF SCRAMBALING WORKS WITH THIS PATTERN

    old_premise = item["premise"]
    old_hypothesis = item["hypothesis"]

    new_premise = pattern.sub(mixed_word, old_premise)
    new_hypothesis = pattern.sub(mixed_word, old_hypothesis)

    new_id = item["id"] + ":" + mixed_word

    return {
        "id": new_id,
        "premise": new_premise,
        "hypothesis": new_hypothesis,
        "label": item["label"]
    }

def scramble_all_data(data_list):
    """scrambles a list"""
    results = []
    for item in data_list: #for each item
        new_item = scramble_per_item(item) #scramble
        results.append(new_item)
    return results


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



def get_all_matching_keys(data, word_with_pos, pos_list):
  '''match beginning key entries in dicitionary that start with positions of words'''
  # pos_list premise_pos_list [(11, 16)]
  # data=sentence
  #word with_pos orginial_word:itspos
  #{'poses:VBZ': {'26:31:4.23e-04': ['','']}
  all_matching_keys = []
  for i in pos_list: #for each position
    p_start, p_end = i #unpack into start and end, e.g. 11, 16
    key_prefix = f"{p_start}:{p_end}:" #join
    matching_keys = [
        k for k in data[word_with_pos].keys()
        if k.startswith(key_prefix)
    ] #get the keys that start with thes epositions
    # print('the matching keys', matching_keys)
    all_matching_keys.extend(matching_keys)
  # print('all matching keys', all_matching_keys)
  return all_matching_keys

def has_pos_tags(suggestions): # outputs True if suggestions have pos tags added besides probability
    '''returns True if all suggestions have suggestion:probability:postag (3 elements)'''
    return all(len(s.split(":")) == 3 for s in suggestions)

def count_pos(suggestions):
  '''return counts per pos tag'''
  pos_C=Counter()
  for s in suggestions:
    pos=s.split(":")[2]
    pos_C[pos]+=1
  return pos_C

def verify_dataset_replacements(dataset, initial_dict):

    '''replaces the suggestion with the original word and looks for that in the intial dicitonary
    sanity check for replacement'''
    not_found_in_original_dictionary = 0
    for item in dataset:
        full_id = item["id"]
        sentence_id, word, pos, _,_,_,_,replacement_word,_ = full_id.split(":")
        key = f"{word}:{pos}"
        original_word = word
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        pattern = re.compile(rf"\b{re.escape(replacement_word)}(?=\b|-)")
        restored_premise, count_p = pattern.subn(original_word, premise)
        restored_hypothesis, count_h = pattern.subn(original_word, hypothesis)

        if count_p == 0 and count_h == 0:
            raise ValueError(
                f"No replacements made for id={full_id!r} "
                f"(replacement_word={replacement_word!r})"
            ) # if no replacements were done in premise/hyptohesis

        if (restored_premise, restored_hypothesis) not in initial_dict:
            not_found_in_original_dictionary += 1

    print(not_found_in_original_dictionary)
    return not_found_in_original_dictionary

def merge_and_analyze_datasets(dataset1, source1, dataset2, source2, min_count, name=None, opposite=True): #~ modify this such that it can take other models as well
    ######NEEDS TO BE MODIFIED
    '''
    Function that merges 2 datasets
    Keeps suggestions for one replacement only if that replacement has sufficient variants
    '''

    def first_3(full_id):
        return ':'.join(full_id.split(':')[:-3])

    def without_last_2(full_id):
        return ':'.join(full_id.split(':')[:-2])  # everything except origin and model
    def id_0(full_id):
        return full_id.split(':')[0]  # everything except origin and model

    def get_word_pos_id(full_id):
        parts = full_id.split(':')
        return ':'.join(parts[:3])  # image_id, word, POS

    all_ids = set()
    for dataset in [dataset1, dataset2]:
        for item in dataset:
            all_ids.add(item['id'])

    word_pos_lookup = {}
    for full_id in sorted(all_ids):
        parts = full_id.split(':')
        word_pos_id = ':'.join(parts[:3] + [parts[-3]])
        origin = parts[-2]
        model = parts[-1]

        if word_pos_id not in word_pos_lookup:
            word_pos_lookup[word_pos_id] = {}
        if origin not in word_pos_lookup[word_pos_id]:
            word_pos_lookup[word_pos_id][origin] = set()
        word_pos_lookup[word_pos_id][origin].add(model)
    renamed = []
    seen_ids = set()

    for dataset in [dataset1, dataset2]:
        for item in tqdm(dataset):
            # print('hey')
            full_id = item['id']
            parts = full_id.split(':')
            word_pos_id = ':'.join(parts[:3] + [parts[-3]])
            origin = parts[-2]
            model = parts[-1]
            base_id = without_last_2(full_id)

            new_id = None
            if origin in ['h', 'p']: #daca origina e p sau h
              opposite_origin = 'p' if origin == 'h' else 'h' # originea opusa e inver
              has_opposite = (
                  word_pos_id in word_pos_lookup and
                  opposite_origin in word_pos_lookup[word_pos_id] #da daaca are opous
              )
              if has_opposite:
                token='ph'
              else:
                token=origin
              if opposite: #daca trb sa aiba opus
                  if not has_opposite:
                      continue
                  opposite_models = word_pos_lookup[word_pos_id][opposite_origin] #uite te la opus
                  if any(other_model != model for other_model in opposite_models):
                      new_id = f"{base_id}:ph:both"
                  else:
                      new_id = f"{base_id}:ph:{model}"
              else:
                  if opposite==False:
                    name2=f"{token}:{model}"
                    new_id = f"{base_id}:{name2}"



            elif origin == 'ph':
                other_origins_exist = False

                if word_pos_id in word_pos_lookup:
                    for check_origin in ['p', 'h']:
                        if (check_origin in word_pos_lookup[word_pos_id] and
                            any(other_model != model for other_model in sorted(word_pos_lookup[word_pos_id][check_origin]))):
                            other_origins_exist = True
                            break

                    if (not other_origins_exist and
                        'ph' in word_pos_lookup[word_pos_id] and
                        any(other_model != model for other_model in word_pos_lookup[word_pos_id]['ph'])):
                        other_origins_exist = True

                if other_origins_exist and opposite==True:
                    new_id = f"{base_id}:ph:both"
                elif other_origins_exist and opposite==False:
                    new_id = f"{base_id}:{origin}:both"
                else:
                    if opposite==False:
                      name2=f"{origin}:{model}"
                      new_id = f"{base_id}:{name2}"
                    else:
                      new_id = f"{base_id}:ph:{model}"
            else:
                continue

            if word_pos_id not in seen_ids:
                new_item = item.copy()
                new_item['id'] = new_id
                renamed.append(new_item)
                seen_ids.add(word_pos_id)

    final_counts = Counter(id_0(item['id']) for item in renamed)
    qualified_base_ids = {bid for bid, count in final_counts.items() if count >= min_count}
    final_dataset = [item for item in sorted(renamed, key=lambda x: x['id']) if id_0(item['id']) in qualified_base_ids]

    source_counts = Counter()
    for item in final_dataset:
        source = item['id'].split(':')[-1]
        source_counts[source] += 1
    print('length of final dataset', len(final_dataset))
    total_instances = sum(source_counts.values())
    average_per_source = {src: count / total_instances for src, count in source_counts.items()}
    label_counts = Counter(item['label'] for item in final_dataset)

    print("\n=== unique ids in qualied base_ids")
    print(len(qualified_base_ids))
    print("=== Label Counts in Final Dataset ===")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print("\n=== Average Instances per Source ===")
    for src, avg in average_per_source.items():
        print(f"{src}: {avg}")
    return final_dataset, average_per_source

def process_matching_keys(data, sentence, word_with_pos, all_matching_keys,
          allowed_pos_tags,  pos_tag_filtering, prob, nlp, singles, batch_nlp_classification_no,
          save_suggestions_in_file=False, data_with_suggestions=None, outside_function_counter_for_count_for_most_common_words_their_pos_tag=None,
          outside_function_counter_for_count_for_pos=None, avrg=None, type_pos_filtering=None, total_words=None, total_remaining_suggestions=None,
          total_words_with_10_plus=None, no_prob_counter=None, average_prob_suggestions=None, diff_total=None, num_sentences=None, num_sentences_binary=None, words_replaced=None, average_prob_replaced=None):
  # average_prob_suggestions = probability of suggestions in a list for a replacement word
# no_prob_counter = how many occurances marked by positions do not have probabilities bc they were not in the first 200 words
# total_words_with_10_plus = no. replaced words that have more than 10 words after probability filtering
# total_remaining_suggestions = no. of remaning suggestions after all filtering
# count = no that is add if pos tag filtering is yes, for each replaced occurance
# diff total = difference between before and after pos tag filtering, added outside loop
# total_Words = count increasing with each processed occurance
#outputsȘ intersected_suggestions, added_pos_counts

  cleaned_suggestions, added_pos_counts, per_key_cleaned_lists = [], False, []
  #print(all_matching_keys)

  for k in all_matching_keys: #
      count_for_most_common_words_their_pos_tag, count_for_pos=Counter(), Counter()
      data_key = data[word_with_pos][k] #get suggestions for that key

      has_p = has_pos_tags(data_key) #check if everything has POS tag

      temporary_suggestions = []

      if has_p == False and pos_tag_filtering == 'yes': #if not everything has POS tag and the argument for pos_tag_filtering  is yes
          p_start, p_end, _ = k.split(":")
          suggestions, pos_counts = filter_suggestions_by_contextual_pos(
              data_key, sentence, int(p_start), int(p_end), allowed_pos_tags, nlp, batch_nlp_classification_no
          )
          if save_suggestions_in_file and data_with_suggestions is not None: #if save filtered data is yes (which would mean replace suggestions with the new suggestion:pos) and if we have a file with suggestions
              data_with_suggestions[sentence][word_with_pos][k] = suggestions #replace the suggestions we have now in the file with the tagged suggestions
          temporary_suggestions.extend(suggestions) # add suggestion to temporary list
      else:
          temporary_suggestions.extend(data[word_with_pos][k]) #same here

      if pos_tag_filtering == 'yes' and avrg == 'yes':   #per replaced word, bc of k
        for s in temporary_suggestions: #for each suggestion stored
            word_pos = f"{s.split(':')[0]}:{s.split(':')[2]}" # get its pos tag
            count_for_most_common_words_their_pos_tag[word_pos] += 1 # add count to a pos tag
                                                                                                                     
        count_for_pos.update(count_pos(temporary_suggestions))
        added_pos_counts = True
        if outside_function_counter_for_count_for_most_common_words_their_pos_tag is not None:
            outside_function_counter_for_count_for_most_common_words_their_pos_tag.update(count_for_most_common_words_their_pos_tag)
        if outside_function_counter_for_count_for_pos is not None:
            outside_function_counter_for_count_for_pos.update(count_for_pos)
      #~ value of type_pos_filtering > class > all_pos_tags_of_class_of_replaced_word
      #~ value of type_pos_filtering > word > pos_tag_of_replaced_word
      if type_pos_filtering == 'all_pos_tags_of_class_of_replaced_word':
        o=allowed_pos_tags

      if type_pos_filtering == 'pos_tag_of_replaced_word':

        o=[word_with_pos.split(':')[1]]

      cleaned_list = filter_candidates(temporary_suggestions, singles)

      if pos_tag_filtering == 'yes': #THIS IS WHERE POS TAG FILTERING IS DONE #####

          len_before = len(cleaned_list)
          cleaned_list = [s for s in cleaned_list if s.split(":")[-1] in o]

          len_after = len(cleaned_list)
          diff_total += (len_before - len_after)  # for each occurance of a word replaced, calculate how big is the diff
          if len(cleaned_list) == 0: #if no suggestions left, continue
              continue
          average_prob_suggestions += sum(float(c.split(":")[1]) for c in cleaned_list) / len(cleaned_list) #sum the probabilityies and divide them by the length of the list


      if prob == "yes":
          try:
              original_prob = float(k.split(":")[2])
              average_prob_replaced+=original_prob
          except (IndexError, ValueError):
              no_prob_counter += 1 # continuw if the word did not have the prob
              continue

          if original_prob !=None:
            words_replaced+=1
          cleaned_list = [c for c in cleaned_list if float(c.split(":")[1]) >= original_prob]
          if len(cleaned_list) > 10:
              total_words_with_10_plus += 1

          for c in cleaned_list:
              prob1 = float(c.split(":")[1])
              if prob1 < original_prob:
                  print(f"DEBUG: Found problematic entry: {c}, prob={prob1}, threshold={original_prob}")
      total_remaining_suggestions += len(cleaned_list)
      total_words += 1
      per_key_cleaned_lists.append(cleaned_list)
      cleaned_suggestions.extend(cleaned_list)
  if num_sentences_binary =='yes' and pos_tag_filtering == 'yes':
    num_sentences += 1 # add 1 to count
  if len(all_matching_keys) > 1:
    #sanity check: i manually checked the output of each line of code
    presence = defaultdict(list)

    for lst in per_key_cleaned_lists:
      for s in set(lst):  # use set to avoid duplicates in the same list
          word, val, pos = s.split(":")
          presence[word].append((float(val), pos))

      keep = {}
      for word, vals in presence.items():
          if len(vals) >= 2:  # appears in ≥2 keys
              avg_val = sum(v for v, _ in vals) / len(vals)
              pos = vals[0][1]  # assume POS is always same
              keep[word] = f"{word}:{avg_val:.5f}:{pos}"
      intersected_suggestions = [
            keep[s.split(":")[0]] for s in cleaned_suggestions if s.split(":")[0] in keep
        ]
  if len(all_matching_keys) == 1:
    intersected_suggestions=  cleaned_suggestions
  return intersected_suggestions, added_pos_counts


def process_dataset(data_with_suggestions,
                    dataset_filtered_already_for_shared_words,
                    if_ids_exist,
                    original_dataset,
                    split,
                    output_file,
                    min_common_words,
                    mapping,
                    ranked_overlap,
                    pos_to_mask,
                    neutral_number,
                    source_1,
                    source_2,
                    entailment_number,
                    contradiction_number,
                    number_of_maximum_annotators,
                    minimum_no_words_premise,
                    minimum_no_words_hypothesis,
                    number_of_minimal_suggestions_common_bt_p_h,
                    nlp,
                    number_batch_for_pos_tagging,
                    pos_tag_filtering:str=None,
                    rank_option='top',
                    num_sentences_to_process_dataset: int = None,
                    num_sentences_compliant_criteria: int = None,
                    save_suggestions_in_file: str = None,
                    calculate_average_pos: str=None,
                    name_file_pos: str=None,
                    prob:str=None,
                    type_rank:str=None,
                    batch_for_pos_tagging= 200,
                    type_rank_operation:str=None,
                    model_name:str=None,
                    name:str=None,
                    type_pos_filtering:str=None,
                    ):
    """

    Matches premise and hypothesis from second_data with data_with_suggestions, replaces words, applies ranking,
    transforms the dataset, and optionally groups it by POS tags.

    data_with_suggestions: data with suggestions of models
    if_ids_exist: optional id list to sample from the eligible sentences further
    dataset_filtered_already_for_shared_words: None (defult), or file that has the seed sentences for the inflated dataset we want to obtain, othewrise the SNLI dataset will be filtered
                      for certain criteria to find the seed sentences
    original_dataset: dataset to be filtered, e.g. SNLI
    split: split of dataset to be filtered, e.g. test
    output_file: name of the output file
    min_common_words: for filtering, minimum number of common words between premise and hypothesis
    mapping: dict with sent annotations from dataset
    ranked_overlap: function that ranks words based on their position in the lists
    pos_to_mask: pos tag to be looked for to be common bt premise and hypothesis
    neutral_number: number for neutral label/ string 'neutral' (will appear as ref in the inf dataset)
    source_1: the title the first sentence has in the dataset, e.g. premise
    source_2: the title the second sentence has in the dataset, e.g. hypothesis
    entailment_number: number for entailment label/ string 'entailment' (will appear as ref in the inf dataset)
    contradiction_number: number for contradiction label/ string 'contradiction' (will appear as ref in the inf dataset)
    number_of_maximum_annotators: number of maximum annotators (anything lower than)
    minimum_no_words_premise: minimum number of words in premise
    minimum_no_words_hypothesis: minumum numberr of words in hypo
    number_of_minimal_suggestions_common_bt_p_h: the lowest number of common suggestion between premise and hypothesis acceptable
    pos_tag_filtering:str=None, if == 'yes' the created dataset will only contain same pos tags as the initial masked word
    rank_option='top' : rank function || values'top' for highest-ranked, int for specific rank, slice for multiple replacements.
    num_sentences_to_process_dataset: int = None: if specified will stop after this number of sentences are process from the dataset, regardless if they are compliant to filtering criteria or not
    num_sentences_compliant_criteria: int = None: if specified will stop after this number of sentences are process from the dataset that are compliant to filtering criteria
    save_suggestions_in_file: str = None if specified will replace the entries in firtst=data with filtered ones that have for suggestions pos tags
    calculate_average_pos: str=None, if specified calculates average pos per premises and hypothesis
    name_file_pos: str=None, name of the file where to store average pos
    prob: if 'yes' suggestions with probabilities equal or higher than the original replaced word are kept
    type_rank: 'avearge_rank' - ranks suggestions based on their avearge positions in the suggested words for premise and hyptoehsis; average_prob: ranks suggestions based on their average probabilities
    batch_for_pos_tagging: no. of sentences to process at a time for pos tagging
    type_rank_operation: when ranking words, the lists of premise and hypothesis can be united, or intersected
    type_pos_filtering= all pos tags of the open class category, or the original pos tag of the replaced word, values: all_pos_tags_of_class_of_replaced_word', 'pos_tag_of_replaced_word'
    """

    pos_filter_map = {
    'noun': {'NN', 'NNS', 'NNP', 'NNPS'},
    'verb': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
    'adjective': {'JJ'},
    'adverb': {'RB'},
    'merged_n_a': {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'},
    'merged_v_n': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS'},
    'merged_v_a': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB'},
    'merged_v_a_n': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'NN', 'NNS'}
    }

    allowed_values = {"average_rank", "average_prob", None}
    if type_rank not in allowed_values: # check if the value of type_rank is one of the allowed ones
        raise ValueError(f"type_rank must be one of {allowed_values}, got {type_rank}")
    processed_data = [] # processed data will be stored here
    pos_tagged_data = defaultdict(list)
    dataset = original_dataset[split] #get the split of the dataset
    if dataset_filtered_already_for_shared_words != None: # if we have a selection of sentences that adhere to the criteria
      SNLI_filtered_2=dataset_filtered_already_for_shared_words # use them
    else: #otherwise check which sentences adhere
      SNLI_filtered_2 = filter_snli(dataset, mapping, pos_to_mask, min_common_words,
                                    num_sentences_to_process_dataset, num_sentences_compliant_criteria, number_of_maximum_annotators, minimum_no_words_premise, minimum_no_words_hypothesis)
    # print(SNLI_filtered_2)
    print(if_ids_exist)
    if if_ids_exist!=None:
      SNLI_filtered_2={key: value for key, value in SNLI_filtered_2.items() if key in if_ids_exist} # if id is in if_ids_exist
      print(f"Filtered length: {len(SNLI_filtered_2)}") #sub-sample
    processed_second_data = pos_toks_extract_from_dataset(SNLI_filtered_2, mapping)
    print('PROCESSED SECOND DATA', processed_second_data)
    new_list3, labels_sample = process_unmasked_dataset(processed_second_data, neutral_number, entailment_number, contradiction_number, id='yes')
    # new_list3 = seed dataset
    # labels_sample == count of labels of initial sentences seletced
    expected_generation = {'neutral': 0, 'entailment': 0, 'contradiction': 0}
    actual_generation = {'neutral': 0, 'entailment': 0, 'contradiction': 0}

    global_words_without_replacements = problems_removed_due_to_low_suggestions = premise_diff_after_pos_filter= total_words_p = average_prob_replaced_premise= words_replaced_p= number_premise_suggestions_remaining_all_filtering = no_premise_words_with_10more_suggestions_higher_probability =premise_replaced_words_had_no_probability=premise_avg_suggestion_prob=premise_diff_total=num_sentences_both=0
    hypothesis_diff_after_pos_filter= total_words_h= average_prob_replaced_hypothesis=words_replaced_h=number_hypothesis_suggestions_remaining_all_filtering = no_hypothesis_words_with_10more_suggestions_higher_probability =hypothesis_replaced_words_had_no_probability=hypothesis_avg_suggestion_prob=hypothesis_diff_total=0
    overall_count_for_pos_p, overall_count_for_pos_h, count_for_most_common_words_their_pos_tag_p, count_for_most_common_words_their_pos_tag_h = [Counter(), Counter(), Counter(), Counter()]
    total_premise_suggestion_probs = total_hypothesisn_suggestion_probs= 0.0
    skip_entry=False
    if pos_to_mask not in pos_filter_map:
        raise ValueError(f"Invalid pos_to_mask: {pos_to_mask}. Must be one of {list(pos_filter_map.keys())}")
    allowed_pos_tags = pos_filter_map[pos_to_mask]
    ids_multiple_keys=[]
    replacement_summary = {}
    for entry in tqdm(processed_second_data[:100]):

        id, premise, hypothesis, tok_p, pos_p, tok_h, pos_h, label = (entry['id'], entry['premise'], entry['hypothesis'], entry['p_t'],entry['p_p'], entry['h_t'], entry['h_p'], entry['label'] )
        if id != '2876232980.jpg#0r1e':
          continue
        premise_id = premise
        hypothesis_id = hypothesis
        word2fillers, word2probabilities, word2pos, _, _, _, positions = [defaultdict(list), defaultdict(list), defaultdict(int), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
        #


        if premise_id in data_with_suggestions.keys() and hypothesis_id in data_with_suggestions.keys():
          common_dict, p_positions, h_positions, _ = common(premise, hypothesis, pos_p, pos_h, tok_p, tok_h, pos_to_mask, source_1, source_2)
                                                      #common dict {'black': {'positions': [(11, 16)], 'pos': 'JJ', 'source': 'premise', 'preceding_text': 'A man in a '}, 'commercial': {'positions': [(29, 39)], 'pos': 'JJ', 'source': 'premise', 'preceding_text': 'A man in a black shirt, in a '}}
                                                      # p_positions [[(11, 16)], [(29, 39)]]
                                                      # h_positions [[(13, 18)], [(31, 41)]]
          singles_premise=premise.split(' ')
          singles_hypothesis=hypothesis.split(' ')
          clean=singles_premise+singles_hypothesis
          singles = [word.strip(string.punctuation).lower() for word in clean]
          words = list(common_dict.keys()) # common_words actually
          words_with_not_enough_replacements_inside_loop_utilitary=0
          for i, word in enumerate(words):
            pos = common_dict[word]['pos'] #get pos of the original word
            word_with_pos = f"{word}:{pos}" #add it to the tokem
            premise_pos_list = p_positions[i] # get the first list of position which corresponds to the word in premise
            hypothesis_pos_list = h_positions[i] # same here
                                        #premise_pos_list [(11, 16)]
                                        # print('the premise data', premise_data)
                                        #the premise data {'poses:VBZ': {'26:31:4.23e-04': ['','']}
            premise_data=data_with_suggestions[premise] # get the data of the premise from the suggestion file
            hypothesis_data=data_with_suggestions[hypothesis] # same for hypothesis

            for qkey, qvalue in premise_data.items():
              if isinstance(qvalue, dict) and len(qvalue) > 1:
                  print('id', id)
                  print(f"Premise {premise_id} has multiple keys: {list(qvalue.keys())}")
                  ids_multiple_keys.append(id)

            all_matching_keys_p = get_all_matching_keys(premise_data, word_with_pos, premise_pos_list)
                                          ## result get_all_matching_keys the matching keys ['48:51:7.56e-01'] all matching keys ['20:23:9.83e-01', '48:51:7.56e-01'], or single element if single apperance
            all_matching_keys_h = get_all_matching_keys(hypothesis_data, word_with_pos, hypothesis_pos_list)
            p_starrt, p_end, _= all_matching_keys_p[0].split(':')
            h_starrt, h_end, _= all_matching_keys_h[0].split(':')
            key=f"{word}:{pos}:{p_starrt}:{p_end}:{h_starrt}:{h_end}" #this will be indexed only with the first position



            premise_suggestions, premise_pos_filter_applied = process_matching_keys(
                premise_data, premise, word_with_pos, all_matching_keys_p, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, data_with_suggestions, count_for_most_common_words_their_pos_tag_p, overall_count_for_pos_p, avrg='yes', type_pos_filtering=type_pos_filtering, total_words=total_words_p,
                total_remaining_suggestions=number_premise_suggestions_remaining_all_filtering, total_words_with_10_plus=no_premise_words_with_10more_suggestions_higher_probability, no_prob_counter=premise_replaced_words_had_no_probability, average_prob_suggestions=premise_avg_suggestion_prob, diff_total=premise_diff_after_pos_filter, num_sentences=num_sentences_both, num_sentences_binary='yes', words_replaced= words_replaced_p,
                average_prob_replaced=average_prob_replaced_premise
              )

            hypothesis_suggestions, hypothesis_pos_filter_applied = process_matching_keys(
                hypothesis_data, hypothesis, word_with_pos, all_matching_keys_h, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, data_with_suggestions, count_for_most_common_words_their_pos_tag_h, overall_count_for_pos_h, avrg='yes', type_pos_filtering=type_pos_filtering, total_words=total_words_h,
                total_remaining_suggestions=number_hypothesis_suggestions_remaining_all_filtering, total_words_with_10_plus=no_hypothesis_words_with_10more_suggestions_higher_probability, no_prob_counter=hypothesis_replaced_words_had_no_probability,
                average_prob_suggestions=hypothesis_avg_suggestion_prob, diff_total=hypothesis_diff_after_pos_filter, words_replaced= words_replaced_h, average_prob_replaced=average_prob_replaced_hypothesis
              )

            if pos_tag_filtering=='yes' and has_pos_tags(premise_suggestions)==False or has_pos_tags(hypothesis_suggestions)==False or premise_pos_filter_applied != True or hypothesis_pos_filter_applied!= True: #check again if some suggestions do not have pos tags
              print('Some of the suggestions for premise or hypothesis are not tagged for POS tag')

            premise_fillers= [c.split(":")[0] for c in premise_suggestions] #suggestions
            hypothesis_fillers= [c.split(":")[0] for c in hypothesis_suggestions]
            if len(premise_fillers)==0 and len(hypothesis_fillers)==0: #I concluded this does not affect the code
              continue
            common_suggestions = set(premise_fillers) & set(hypothesis_fillers) #~ common_words > common_suggestions
            if len(common_suggestions) < number_of_minimal_suggestions_common_bt_p_h: #this is useful only when setting a number of minnimal suggestions
                global_words_without_replacements += 1




                if len(words) == 1 or (len(words) >= 1 and words_with_not_enough_replacements_inside_loop_utilitary == len(words) - 1):
                    new_list3 = [item for item in new_list3 if item['id'] != id]
                    problems_removed_due_to_low_suggestions += 1
                elif len(words) >= 1:
                    words_with_not_enough_replacements_inside_loop_utilitary += 1
                continue

            premise_probabilities = [float(c.split(":")[1]) for c in premise_suggestions]
            hypothesis_probabilities = [float(c.split(":")[1]) for c in hypothesis_suggestions]

            word2fillers[key] = [premise_fillers, hypothesis_fillers]
            word2probabilities[key] = [premise_probabilities, hypothesis_probabilities]
            word2pos[key] = [pos, pos]
            positions[key]= [all_matching_keys_p, all_matching_keys_h]


        ######wird2filllers {'top:NN:22:25:22:25': [['dress', 'hat', 'shirt', 'cap', 'coat', 'robe', 'sc]]
        if word2fillers and skip_entry==False:
          words = {}
          for w in word2fillers:
              print('w from word2fillers', word2fillers)
              words[w] = ranked_overlap(word2fillers[w], word2probabilities[w], type_rank_operation).items()
              #########[word] = {
            # 'average_rank': sum(ranks)/n,
            # 'ranks' :ranks,
            # 'average_prob': f"{avg_prob:.2e}",
            # "individual_probs": [f"{p:.2e}" for p in probs1]}
              words[w] = sorted(words[w], key=lambda x: x[1][type_rank]) ###################
          assigned_pos_tags = set()
          for w, ranked_fillers in words.items():
              parts = w.split(':')
              if len(parts) != 6:
                  print(f"Unexpected key format: {w}")
                  continue
              word_only, pos, premise_start, premise_end, hypothesis_start, hypothesis_end = parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
              positions_to_replace_premise = [':'.join(i.split(':')[:2]) for i in positions[w][0]]
              positions_to_replace_hypothesis = [':'.join(i.split(':')[:2]) for i in positions[w][1]]
              premise_suggestions_fillers=[]
              if id not in replacement_summary:
                replacement_summary[id] = {}

              key = f"{word_only}:{pos}"
              if key not in replacement_summary[id]:
                replacement_summary[id][key] = [positions_to_replace_premise, positions_to_replace_hypothesis]
              else:
                print('RED FLAG — duplicate key, skipping:', key)

              for key, value in word2fillers.items():
                parts = key.split(':')
                to_look= ':'.join(parts[:2])
                if to_look == word_only+':'+pos:
                  premise_ssssss, hypo_ssss = value[0], value[1]

              expected_variants = 0
              if isinstance(rank_option, int): 
                  expected_variants = 1 if rank_option < len(ranked_fillers) else 0
              elif isinstance(rank_option, slice):
                  expected_variants = len(range(*rank_option.indices(len(ranked_fillers))))
              sentence_variants = []
              if label in expected_generation:
                expected_generation[label] += expected_variants


              indices = [rank_option] if isinstance(rank_option, int) else range(*rank_option.indices(len(ranked_fillers)))

              for i in indices:
                  if i >= len(ranked_fillers):
                      continue

                  best_ = ranked_fillers[i][0].strip()

                  p_variant = premise_id
                  h_variant = hypothesis_id

                  p_positions_sorted = sorted(
                      positions_to_replace_premise,
                      key=lambda s: int(s.split(':')[0]),
                      reverse=True)
                  h_positions_sorted = sorted(
                      positions_to_replace_hypothesis,
                      key=lambda s: int(s.split(':')[0]),
                      reverse=True
                  )

                  for i in p_positions_sorted:
                      start, end = i.split(':')

                      p_variant = p_variant[:int(start)] + best_ + p_variant[int(end):]


                  for i in h_positions_sorted:
                      start, end = i.split(":")

                      h_variant = h_variant[:int(start)] + best_ + h_variant[int(end):]
                  oiringacr = (
                    'h' if best_ not in premise_ssssss else
                    'p' if best_ not in hypo_ssss else
                    'ph'
                    )

                  if p_positions_sorted and h_positions_sorted:
                      sentence_variants.append((p_variant, h_variant, best_, oiringacr))
                  else:
                      print('❌ Skipped: One of the position lists is empty.')


              assigned_pos_tags.update(word2pos[w])
              for idx, (p_variant, h_variant, best_, oiringacr) in enumerate(sentence_variants):
                numeric_label = None

                if label in {'neutral', 'entailment', 'contradiction'}:
                  numeric_label = label
                  actual_generation[label] += 1
                model_name_id=('bert' if model_name=='bert-base-cased' else
                               'roberta') ##########3 here add for the other model
                processed_entry = {
                    'id': f"{id}:{word_only}:{pos}:{premise_start}:{premise_end}:{hypothesis_start}:{hypothesis_end}:{best_}:{oiringacr}:{model_name_id}",
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
                ###### CHECK 4
    statistics_kwargs = {
        'actual_generation': actual_generation,
        'expected_generation': expected_generation,
        'global_words_without_replacements': global_words_without_replacements,
        'problems_removed_due_to_low_suggestions': problems_removed_due_to_low_suggestions,
        'name_file_general': f"{output_file}_general_stats.txt",
        'name_file_prob': f"{output_file}_prob_stats.txt",
        'name_file_pos': name_file_pos if name_file_pos else f"{output_file}_pos_stats.txt",
        'number_hypothesis_suggestions_remaining_all_filtering': number_hypothesis_suggestions_remaining_all_filtering,
        'no_hypothesis_words_with_10more_suggestions_higher_probability': no_hypothesis_words_with_10more_suggestions_higher_probability,
        'total_words_h': total_words_h,
        'number_premise_suggestions_remaining_all_filtering': number_premise_suggestions_remaining_all_filtering,
        'no_premise_words_with_10more_suggestions_higher_probability': no_premise_words_with_10more_suggestions_higher_probability,
        'total_words_p': total_words_p,
        'words_replaced_p': words_replaced_p,
        'words_replaced_h': words_replaced_h,
        'premise_replaced_words_had_no_probability': premise_replaced_words_had_no_probability,
        'hypothesis_replaced_words_had_no_probability': hypothesis_replaced_words_had_no_probability,
        'average_prob_replaced_hypothesis': average_prob_replaced_hypothesis,
        'average_prob_replaced_premise': average_prob_replaced_premise,
        'allowed_pos_tags': allowed_pos_tags,
        'overall_count_for_pos_p': overall_count_for_pos_p,
        'overall_count_for_pos_h': overall_count_for_pos_h,
        'count_for_most_common_words_their_pos_tag_p': count_for_most_common_words_their_pos_tag_p,
        'count_for_most_common_words_their_pos_tag_h': count_for_most_common_words_their_pos_tag_h,                                                        
        'num_sentences_both': num_sentences_both,
        'premise_diff_after_pos_filter': premise_diff_after_pos_filter,
        'hypothesis_diff_after_pos_filter': hypothesis_diff_after_pos_filter,
        'pos_to_mask': pos_to_mask
        }

    results = write_statistics_based_on_parameters(output_file, prob, calculate_average_pos, **statistics_kwargs)


    premise_count = None
    hypothesis_count = None
    word_count_freq_premise = None
    word_count_freq_hypothesis = None

    if results:
        premise_count = results.get('premise_count')
        hypothesis_count = results.get('hypothesis_count')
        word_count_freq_premise = results.get('word_count_freq_premise')
        word_count_freq_hypothesis = results.get('word_count_freq_hypothesis')

    file_counts={output_file:actual_generation, 'sample': labels_sample}
    with open(output_file, "w") as f:
        json.dump(processed_data, f)
    if save_suggestions_in_file:
      with open(save_suggestions_in_file, 'w') as f_out:
          json.dump(data_with_suggestions, f_out)
    return processed_data, new_list3, file_counts, premise_count, hypothesis_count, word_count_freq_premise, word_count_freq_hypothesis, replacement_summary


def get_base_ids(filepath):
    """Reads a JSON file and extracts the first part of the 'id' for each entry."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file {filepath}: {e}")
        return set()

    base_ids = set()

    for key, entry_list in data.items():
        if isinstance(entry_list, list):
            for entry in entry_list:
                if isinstance(entry, dict) and 'id' in entry:
                    parts = entry['id'].split(':')
                    if parts:
                        base_ids.add(parts[0])
    return base_ids

def extract_base_id(full_id):
    return full_id.split("#")[0]



def generate_output_filenames(suggestion_file, number_inflation="10"):
    #not double-checked
    """
    Given a suggestion file path like:
      /.../robert-base-cased.1.noun.200.test.json
    extract parts of the name and automatically generate the output file names required for the processed dataset

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

def predictions_nli(model_name, data_json_file, batch_size_number, device_g_c, batch_function, tok_model_function):
    #not double-checked
    """
    calculates predictions for a dataset

    args:
    model_name: model_name
    data_json_file: json file with stimuli for predictions
    batch_size_number: batch number
    device_g_c:  cuda or cpu
    batch_function: takes function from assign.tools to eval in batches
    tok_model_function: tokenizer function from assign tools
    
    outputs: a json file with the input file name, model name and its predictions
    ***Note json file has to contain premise and hypothesis
    """
    with open(data_json_file, "r") as f:
        data = json.load(f)
    data = map_labels_to_numbers(data, model_name)

    moodels={
        "textattack/bert-base-uncased-snli": "textattack/bert",
        "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli": "ynie/roberta",
        "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli": "ynie/albert",
        "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli": "ynie/bart"
    }
    tokenizer, model_cpu = tok_model_function(model_name)
    model_cpu.to(device_g_c)
    prem_hypo_list = [(item['premise'], item['hypothesis']) for item in data]
    preds2 = batch_function(tokenizer, model_cpu, prem_hypo_list, batch_size=batch_size_number, device=device_g_c)
    output = {
        "input_file": data_json_file,
        "model": model_name,
        "predictions": preds2
    }
    safe_model_name = model_name.replace('/', '_')
    output_filename = f"{data_json_file.rsplit('.', 1)[0]}_{safe_model_name}_predictions.json"
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=4, default=lambda o: o.item() if hasattr(o, "item") else o)
    return output, output_filename

def merge_data_and_predictions(data_json_file, predictions_file, model_name):
    #not double-checked
    '''merges documents of the inflated dataset and the predictions from the model by zipping the data from json'''
    with open(data_json_file, "r") as f:
        data = json.load(f)
    with open(predictions_file, "r") as f:
        predictions_dict = json.load(f)
    merged = []
    data = map_labels_to_numbers(data, model_name)
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
  with open(json_filepath, "r") as f:
      data = json.load(f)
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


def parse_input_file(input_file):
    """
    Given an input file path with a filename like:
    /.../bert-base-cased.1.adjective.200.test.inflated.for.bert.10.json
    returns a tuple: (model_generated, pos_tag, number_inflation)
    """
    base = os.path.basename(input_file)
    parts = base.split('.')

    if len(parts) < 7:
        return ("", "", "")
    model_generated = parts[0]
    pos_tag = parts[2]
    number_inflation = parts[-2]
    inflated=parts[5]
    return model_generated, pos_tag, number_inflation, inflated


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
