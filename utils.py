#FIXME no explicit punctuaion definition but use the sent and tokens to get chr offsets
from transformers import pipeline
from typing import Dict, Callable, List, Tuple
import nltk
import re
from nltk import pos_tag, word_tokenize
from tqdm import tqdm
import pandas as pd
import json
from collections import defaultdict


#do not replace negation in adverb
def extract__pos_position(pos_tags, tokens, source, pos_type):
    noun_tags = {'NN', 'NNS'}
    verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    adjective_tags = {'JJ', 'JJR', 'JJS'}
    adverb_tags = {'RB', 'RBR', 'RBS'}

    tags = [
        ".", ",", "?", "'", ":", ";", "-", "–", "—", "'", "\"", "(", ")", "[", "]", "{", "}",
        "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "_", "~", "`", "<", ">", "="
    ]

    special_tok = ["'s", "'t", "n't"]
    dictionary_positions = {}
    current_position = 0

    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
        start = current_position
        end = current_position + len(token)
        offset = (start, end)


        if pos_type == 'noun' and pos in noun_tags:
            category = "nouns"
        elif pos_type == 'verb' and pos in verb_tags:
            category = "verbs"
        elif pos_type == 'adjective' and pos in adjective_tags:
            category = "adjectives"
        elif pos_type == 'adverb' and pos in adverb_tags:
            category = "adverbs"
        elif pos_type == 'merged_n_a' and (pos in noun_tags or pos in adjective_tags):
            category = "nouns" if pos in noun_tags else "adjectives"
        elif pos_type == 'merged_v_n' and (pos in verb_tags or pos in noun_tags):
            category = "verbs" if pos in verb_tags else "nouns"
        elif pos_type == 'merged_v_a' and (pos in verb_tags or pos in adverb_tags):
            category = "verbs" if pos in verb_tags else "adverbs"
        elif pos_type == 'merged_v_a_n' and (pos in verb_tags or pos in adverb_tags or pos in noun_tags):
            if pos in verb_tags:
                category = "verbs"
            elif pos in adverb_tags:
                category = "adverbs"
            else:
                category = "nouns"
        else:
            category = None

        if category:

            if token not in dictionary_positions:
                dictionary_positions[token] = {'positions': [offset], 'pos': pos, 'source': source}
            else:
                dictionary_positions[token]['positions'].append(offset)

        if pos not in tags:
            current_position += len(token) + 1
        if pos in tags:
            current_position += len(token)
        if token in special_tok:
            current_position += len(token) - 3
    return dictionary_positions


def common(sentence1, sentence2, pos_sent_1, pos_sent_2, toks_sent_1, toks_sent_2, pos_type, source_1, source_2, singles='yes'):

    extracted_1 = extract__pos_position(pos_sent_1, toks_sent_1, source_1, pos_type)

    extracted_2 = extract__pos_position(pos_sent_2, toks_sent_2, source_2, pos_type)
    common_tokens = set(extracted_1.keys()) & set(extracted_2.keys())
    common_dict = {token: extracted_1[token] for token in common_tokens}
    all_nouns_singles = {' ' + k for d in [extracted_1, extracted_2] for k, v in d.items()} if singles=='yes' else None
    mask_positions_1 = [extracted_1[token]["positions"][0] for token in common_tokens]
    mask_positions_2 = [extracted_2[token]["positions"][0] for token in common_tokens]

    return common_dict, mask_positions_1, mask_positions_2, all_nouns_singles


def suggest_mask_fillers(input_str:str, mask_offsets: List[Tuple[int,int]],
                         model_fill_mask, all_single_words, common_tokens, suggestion_n=50) -> Dict[Tuple[int,int], List[str]]:
    """ mask_offsets is a list of integer pairs that mark the part of teh string input taht needs to be masked.
        It is a list because in general it might be needed to mask several parts of the input string.
        Returns a dictionary with character offsets as keys and a list of ranked suggestions as values.
    """
    model_architecture = model_fill_mask.model.config.architectures[0].lower()

    if 'bert' in model_architecture and 'roberta' not in model_architecture:
      mask_token = '[MASK]'
    else:
      mask_token = '<mask>'
    suggestions = {}

    for i, j in mask_offsets:
      masked_token_orig = input_str[i:j]
      offset_key = str(i)+':'+str(j)
      if masked_token_orig in common_tokens:
          pos_tag = common_tokens[masked_token_orig].get('pos', 'UNK')
      else:
          pos_tag = "UNK"
      token_key = f"{masked_token_orig}:{pos_tag}"
      candidate_list = []
      masked_input = input_str[:i] + f'{mask_token}' + input_str[j:]
      # print(masked_input)
      if masked_input.endswith('<mask>'):
          masked_input += '.'
      generated = model_fill_mask(masked_input, top_k=suggestion_n)
      all_singles_stripped = [i.strip(' ') for i in all_single_words]
      all_singles_stripped_lower = [i.strip(' ').lower() for i in all_single_words]
      all_singles_lower = [i.lower() for i in all_single_words]
      for k in generated:
          if k['token_str'] in all_single_words or k['token_str'] in all_singles_stripped or k[
              'token_str'] in all_singles_stripped_lower or k['token_str'] in all_singles_lower:

              continue
          
          if re.match(r' \w+', k['token_str']) or re.match(r'\w+', k['token_str']) and not k[
              'token_str'].startswith('##'):
              candidate_list.append(f"{k['token_str']}:{k['score']:.2e}")

      if input_str not in suggestions:
          suggestions[token_key] = {}
      if offset_key not in suggestions[token_key]:
        suggestions[token_key][offset_key] = candidate_list
      else:
        suggestions[token_key][offset_key].extend(candidate_list)
    return suggestions





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

def extract_nouns_and_verbs(pos_tags, tokens, pos_type):
        noun_tags = {'NN', 'NNS'}
        verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        adjective_tags = {'JJ'}
        adverb_tags = {'RB'} #, 'JJR', 'JJS' 'RBR', 'RBS' without them to not create change of label

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

def filter_snli(dataset, mapping, pos_to_mask, min_common_words, num_sentences_to_process, max_filtered_count=None):

    filtered = {}
    count = 0

    dataset_items = list(dataset.items())[:num_sentences_to_process] if num_sentences_to_process else dataset.items()

    for k, p in dataset_items:
        if len(p['lcnt']) == 1 and len(p['p'].split()) >= 8 and len(p['h'].split()) >= 8:
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


def process_unmasked_dataset(filtered_list_1, neutral_number:int, entailment_number:int, contradiction_number:int, id) -> List[Dict]:
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

  return new_list4

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
    no_neutral:int,
    no_contradiction:int,
    no_ential:int,
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
                                  num_sentences_to_process_dataset, num_sentences_compliant_criteria)

    filtered_list_1 = pos_toks_extract_from_dataset(SNLI_filtered_2, mapping)
    new_list3 = process_unmasked_dataset(filtered_list_1, no_neutral, no_ential, no_contradiction, id='yes')

    filler_pipeline = pipeline("fill-mask", model=model_name)


    results_dict = {}
    
    for p in tqdm(filtered_list_1):
        id, premise, hypothesis, tok_p, pos_p, tok_h, pos_h = (p['id'], p['premise'], p['hypothesis'], p['p_t'], p['p_p'], p['h_t'], p['h_p'] )

        common_tokens_dictionary, p_off, h_off, all_nouns_singles = common(
            premise, hypothesis, pos_p, pos_h, tok_p, tok_h, pos_to_mask, source_1, source_2, already_exsiting_words
        )


      
        p_off_filler = suggest_mask_fillers(premise, p_off, filler_pipeline, all_nouns_singles, common_tokens_dictionary, num_filler_suggestions)
        if p_off_filler:
       
          if premise in results_dict:
              results_dict[premise].update(p_off_filler)
          else:
              results_dict[premise] = p_off_filler

        h_off_filler = suggest_mask_fillers(hypothesis, h_off, filler_pipeline, all_nouns_singles, common_tokens_dictionary, num_filler_suggestions)
        if h_off_filler:
          if hypothesis in results_dict:
              results_dict[hypothesis].update(h_off_filler)
          else:
              results_dict[hypothesis] = h_off_filler      
    with open(output_file, "w") as f:
        json.dump(results_dict, f)

    return new_list3
import re
from collections import defaultdict
from tqdm import tqdm

def process_dataset(first_data, second_data, ranked_overlap, neutral_number, entailment_number, contradiction_number, rank_option='top', sort_by_pos='no', id='no'):
    """
    Matches premise and hypothesis from second_data with first_data, replaces words, applies ranking,
    transforms the dataset, and optionally groups it by POS tags.

    :param first_data: The dataset containing suggestions
    :param second_data: The dataset containing 'id', 'premise', 'hypothesis', and 'label'.
    :param ranked_overlap: The function that ranks words based on probability.
    :param neutral_number: number for neutral label
    :param entailment_number: number for entailment label
    :param contradiction_number: number for contradiction label
    :param rank_option: 'top' for highest-ranked, int for specific rank, slice for multiple replacements.
    :param sort_by_pos: 'yes' to group the dataset by POS tags.
    :param id: 'yes' to process a first_data file with masked suggestions that were recorded as sentence:id 
    :return: Processed dataset with replaced words and transformed labels.
    """

    processed_data = []
    pos_tagged_data = defaultdict(list)  
    label_counts = {'neutral': 0, 'entailment': 0, 'contradiction': 0}
    for entry in tqdm(second_data[:20]):
        id = entry['id']
        premise = entry['premise']
        hypothesis = entry['hypothesis']
        label = entry['label']

        if id =='yes':
          premise_id = f"{premise}:{id}"
          hypothesis_id = f"{hypothesis}:{id}"

        else:
          premise_id = premise
          hypothesis_id = hypothesis

  
        word2fillers = defaultdict(list)
        word2probabilities = defaultdict(list)
        word2pos = defaultdict(list)
     
        
        for sentence_id in [premise_id, hypothesis_id]:
          for i in first_data:
            if sentence_id in i.keys():
                
                token_data = i[sentence_id]
                
                for token_key, offsets in token_data.items():
                    for offset, candidates in offsets.items():
                        word, pos = token_key.split(":")  
                        
                       
                        fillers = [c.split(":")[0] for c in candidates]
                  
                        probabilities = [float(c.split(":")[1]) for c in candidates]

                        word2fillers[word].append(fillers)
                        word2probabilities[word].append(probabilities)
                        word2pos[word].append(pos)  


        words = {}

        for w in word2fillers:
            words[w] = ranked_overlap(word2fillers[w], word2probabilities[w]).items()
            words[w] = sorted(words[w], key=lambda x: x[1]["average_rank"])

   
        assigned_pos_tags = set()  
        sentence_variants = []
           
        for w, ranked_fillers in words.items():
        
            try:
                if isinstance(rank_option, int):
                   
                    if len(ranked_fillers)<rank_option-1:
                      continue
                    else:
                      best_ = ranked_fillers[rank_option][0].strip()
                      
                      p_variant = re.sub(rf'\b{w}\b', best_, premise)
                     
                      h_variant = re.sub(rf'\b{w}\b', best_, hypothesis)
                      sentence_variants.append((p_variant, h_variant))
                    
                elif isinstance(rank_option, slice):
                 
                    for i in range(*rank_option.indices(len(ranked_fillers))):
                        best_ = ranked_fillers[i][0].strip()
                        p_variant = re.sub(rf'\b{w}\b', best_, premise)
                       
                        h_variant = re.sub(rf'\b{w}\b', best_, hypothesis)
                        sentence_variants.append((p_variant, h_variant))

               
                assigned_pos_tags.update(word2pos[w])
                for idx, (p_variant, h_variant) in enumerate(sentence_variants):

                  processed_entry = {
                      'premise': p_variant,
                      'hypothesis': h_variant,
                      'label': label
                  }

                  if id == 'yes':
                      processed_entry['id'] = f"{id}_{idx}"

               
                  if label == neutral_number:
                      label_counts['neutral'] += 1
                  elif label == entailment_number:
                      label_counts['entailment'] += 1
                  elif label == contradiction_number:
                      label_counts['contradiction'] += 1


                  if sort_by_pos == 'yes':
                          for pos_tag in assigned_pos_tags:
                              pos_tagged_data[pos_tag].append(processed_entry)
                  else:
                      processed_data.append(processed_entry)
        
            except (IndexError, ValueError):
                continue  
            
       
            
    print("\nLabel Counts:") 
    print(f"Neutral: {label_counts['neutral']}")
    print(f"Entailment: {label_counts['entailment']}")
    print(f"Contradiction: {label_counts['contradiction']}\n")

    if sort_by_pos == 'yes':
        sorted_data = []
        for pos, entries in sorted(pos_tagged_data.items()):
            sorted_data.append({pos: entries})
        return sorted_data

    return processed_data
