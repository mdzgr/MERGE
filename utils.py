#FIXME no explicit punctuaion definition but use the sent and tokens to get chr offsets
from transformers import pipeline
from typing import Dict, Callable, List, Tuple
import nltk
import re
from nltk import pos_tag, word_tokenize
from tqdm import tqdm
import pandas as pd
from collections import defaultdict


#do not replace negation in adverb
def extract_nouns(pos_tags, tokens, source, pos_type):
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
    # print(dictionary_positions)
    return dictionary_positions

def suggest_mask_fillers(input_str:str, mask_offsets: List[Tuple[int,int]],
                         model_fill_mask, all_single_words, suggestion_n=50, type_mask=None) -> Dict[Tuple[int,int], List[str]]:
    """ mask_offsets is a list of integer pairs that mark the part of teh string input taht needs to be masked.
        It is a list because in general it might be needed to mask several parts of the input string.
        Returns a dictionary with character offsets as keys and a list of ranked suggestions as values.
    """
    suggestions = {}
    suggestions_replaced={}
    if type_mask == 'one':
      mask_token = '[MASK]' if 'bert' in model_fill_mask.model.config.architectures[0].lower() else ' <mask>'
      for i,j in mask_offsets:
        tokens={}
        tokens_repl={}
        masked_input = input_str[:i] +  f' {mask_token}' + input_str[j:] #for roberta is dif token <mask>
        if masked_input.endswith('<mask>'):
          masked_input+='.'

        generated = model_fill_mask(masked_input, top_k=suggestion_n)
        all_singles_stripped=[i.strip(' ') for i in all_single_words]
        all_singles_stripped_lower=[i.strip(' ').lower() for i in all_single_words]
        all_singles_lower=[i.lower() for i in all_single_words]
        for k in generated:

          if k['token_str'] in all_single_words or k['token_str'] in all_singles_stripped or k['token_str'] in all_singles_stripped_lower or k['token_str'] in all_singles_lower:

            continue

          if re.match(r' \w+',  k['token_str']) or re.match(r'\w+',  k['token_str']) and not k['token_str'].startswith('##'):

            tokens[k['token_str']]={'score': k['score'], 'sequence': k['sequence']}

        suggestions[(i, j)] = tokens

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

def filter_snli(dataset, pos_to_mask, min_common_words, num_sentences_to_process):

    filtered = {
        k: p
        for k, p in list(dataset.items())[:num_sentences_to_process]
        if len(p['lcnt']) == 1  # one label
        and len(p['p'].split()) >= 8  # premise len >= 8
        and len(p['h'].split()) >= 8
        and (len

                (flatten_extracted_words(
                    extract_nouns_and_verbs(S2A[p['p']]['pos'], S2A[p['p']]['tok'], pos_to_mask)
                ) &
                flatten_extracted_words(
                    extract_nouns_and_verbs(S2A[p['h']]['pos'], S2A[p['h']]['tok'], pos_to_mask)
                )
            ) >= min_common_words)
    }

    return filtered

def process_unmasked_dataset(filtered_list_1, id) -> List[Dict]:
  new_list4 = []
  
  label_counts = {'contradiction': 0, 'entailment': 0, 'neutral': 0}

  if isinstance(filtered_list_1, dict):
        filtered_list_1 = [
            {"label": p["g"], "premise": p["p"], "hypothesis": p["h"]}
            for k, p in filtered_list_1.items()
        ]

  for i in tqdm(filtered_list_1):
      # print(i)
      label = i['label']
      if id=='yes':
        new_list4.append({
            'id':i['id'],
            'premise': i['premise'],
            'hypothesis': i['hypothesis'],
            'label': {'contradiction': 2, 'entailment': 0, 'neutral': 1}[label]
        })
        label_counts[label] += 1
      if id =='no':
        new_list4.append({
            'premise': i['premise'],
            'hypothesis': i['hypothesis'],
            'label': {'contradiction': 2, 'entailment': 0, 'neutral': 1}[label]
        })
        label_counts[label] += 1

  print("Label counts:", label_counts)

  return new_list4
    
def create_filler_masked_dataset(
    model_name: str,
    dataset: pd.DataFrame,
    split: str,
    pos_to_mask: str,
    min_common_words: int,
    num_filler_suggestions: int,
    rank_w:int,
    to_mask: str,
    num_sentences_to_process: int = None,
    output_format: str = 'list',
    output_file: str = None,
    output_file_2: str= None,
    separate_by_pos: str = 'no'
) -> List[Dict]:

    """
        Function generating a new inflated dataset with suggestion from a language model, alongside the initial split of sentences modified

        model_name : str // name of language model used for generating masked token predictions (e.g., "bert-base-uncased").
        dataset : Dataset // The dataset to process
        split : str // The dataset split to use (e.g., "train", "test", "validation").
        pos_to_mask : str // a str indicating what pos to be masked in sentences ('noun' or 'verb'), and what sentences to be picked for masking considering trhe min_common_words (e.g. 3 common nouns)
        min_common_words : int// the minimum number of common words required between premise and hypothesis
        num_filler_suggestions : int// The number of suggested filler words for each masked token by model
        rank_w : slice or int // the ranking range/ the specific filler suggestions that will be part of the new dataset
        to_mask : str // Whether to return a dataset with inflated options or not, if 'no' > output are 2 datasets, unmasked sentences that have the no. min of common pos tags indicated, with (second returned dictionary) and with no ids
                      // if 'yes' > returns 2 datasets, first dataset containts the new obtained dataset after inlafting with masked suggestions, second dictionaries stores the initial filtered dataset for masking, before masking.
        num_sentences_to_process : int /// The number of sentences to process from the dataset.
        output_format : str // The format of the output file
        output_file : str /// file name where the masked dataset  will be saved
        output_file_2 : str // name of a second output file that will store all the ranked words of first 100 examples of the dataset
        separate_by_pos : str // If 'yes', returnes a masked dictionary and output_file_2 grouped by pos tags

    """

    label_counts = {'contradiction': 0, 'entailment': 0, 'neutral': 0}
    new_list4 = []
    new_list_3=[]
    filtered_list_1 = []


    dataset = dataset[split]
    SNLI_filtered_2 = filter_snli(dataset, pos_to_mask, min_common_words, num_sentences_to_process)

    grouped_problems = defaultdict(dict)
    for k, p in SNLI_filtered_2.items():
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
              'p_p': S2A[p['p']]['pos'],
              'p_t': S2A[p['p']]['tok'],
              'h_p': S2A[p['h']]['pos'],
              'h_t': S2A[p['h']]['tok']
          })
    print(f"no. problems filtered after criteria: {len(filtered_list_1)}")
    if to_mask=='no':
      new_list4 = process_unmasked_dataset(filtered_list_1, id='no')
      new_list_3 = process_unmasked_dataset(filtered_list_1, id='yes')
      if output_format == 'txt' and output_file:
          with open(output_file, 'w') as file:
              model_details = f"\"fill-mask\", model=\"{model_name}\""
              file.write(f"\n{model_details}\n")
              for entry in filtered_list_1:

                  id = entry['id']
                  premise = entry['premise']
                  hypothesis = entry['hypothesis']
                  label = entry['label']

                  file.write(f"\n{id}: {label}\n{premise}\n{hypothesis}\n")

          return new_list4, new_list_3
      return new_list4, new_list_3
    else:
      new_list3 = process_unmasked_dataset(filtered_list_1, id='yes')
      filler_pipeline = pipeline("fill-mask", model=model_name)
      words_overall2 = []
      for p in tqdm(filtered_list_1):
          id, premise, hypothesis, tok_p, pos_p, tok_h, pos_h = (p['id'], p['premise'], p['hypothesis'], p['p_t'], p['p_p'], p['h_t'], p['h_p'])

          p_dictionary = extract_nouns(pos_p, tok_p, 'premise', pos_to_mask)
          h_dictionary = extract_nouns(pos_h, tok_h, 'hypothesis', pos_to_mask)

          all_nouns_singles = {' ' + k for d in [p_dictionary, h_dictionary] for k, v in d.items()}

          common_tokens_dictionary = set(p_dictionary.keys()).intersection(h_dictionary.keys())

          p_dictionary = {k: v for k, v in p_dictionary.items() if k in common_tokens_dictionary}
          h_dictionary = {k: v for k, v in h_dictionary.items() if k in common_tokens_dictionary}

          p_off = [i['positions'][0] for i in p_dictionary.values()]
          h_off = [i['positions'][0] for i in h_dictionary.values()]

          p_off_filler = {offset: suggest_mask_fillers(premise, [offset], filler_pipeline, all_nouns_singles, num_filler_suggestions, 'one')[offset]
                          for offset in p_off}

          h_off_filler = {offset: suggest_mask_fillers(hypothesis, [offset], filler_pipeline, all_nouns_singles, num_filler_suggestions, 'one')[offset]
                          for offset in h_off}

          combined_p_h = defaultdict(list)
          for d in (p_dictionary, h_dictionary):
              for key, value in d.items():
                  combined_p_h[key].append(value)

          word2fillers = defaultdict(list)
          word2probabilities = defaultdict(list)
          word2pos=defaultdict(list)
          for key, value in combined_p_h.items():

              for i in value:
                  positions = i['positions'][0]
                  source = i['source']
                  pos=i['pos']

                  if source == 'premise' and positions in p_off_filler:
                      fillers = p_off_filler[positions]
                      word2fillers[key].append(list(fillers.keys()))
                      word2probabilities[key].append([val['score'] for val in fillers.values() if 'score' in val])
                      word2pos[key].append(pos)

                  if source == 'hypothesis' and positions in h_off_filler:
                      fillers = h_off_filler[positions]
                      word2fillers[key].append(list(fillers.keys()))
                      word2probabilities[key].append([val['score'] for val in fillers.values() if 'score' in val])
                      word2pos[key].append(pos)
          

          words = {}
          for w in word2fillers:

              words[w] = ranked_overlap(word2fillers[w], word2probabilities[w]).items()
              words[w] = sorted(words[w], key=lambda x: x[1]["average_rank"])

          words_overall2.append({
              'id': id,
              'premise': premise,
              'label': p['label'],
              'hypothesis': hypothesis,
              'ranks': words,
              'pos': word2pos

          })


      new_dataset_1 = []
      for entry in words_overall2:
   
          id, label, premise, hypothesis, ranks, pos_en = entry['id'], entry['label'], entry['premise'], entry['hypothesis'], entry['ranks'], entry['pos']

          new_ = {}

          for w, ranked_fillers in ranks.items():
            try:
              if isinstance(rank_w, int):
                  best_ = ranked_fillers[rank_w][0].strip()
                  p_masked = re.sub(rf'\b{w}\b', best_, premise)
                  h_masked = re.sub(rf'\b{w}\b', best_, hypothesis)

                  pos_1= pos_en[w]

                  new_[w] = {'premise': p_masked, 'hypothesis': h_masked, 'pos': pos_1}

              elif isinstance(rank_w, slice):

                  for i in range(*rank_w.indices(len(ranked_fillers))):
                   
                      best_ = ranked_fillers[i][0].strip()
                      p_masked = re.sub(rf'\b{w}\b', best_, premise)
                      h_masked = re.sub(rf'\b{w}\b', best_, hypothesis)
                    
                      pos_1 = pos_en[w]
                    
                      new_[f"{w}_{i}"] = {'premise': p_masked, 'hypothesis': h_masked, 'pos': pos_1}

              else:
                  raise ValueError("rank_w must be an integer or a slice.")
            except IndexError:
              break
          new_dataset_1.append({'id': id, 'premise': premise, 'label': label, 'hypothesis': hypothesis, 'new_h_p': new_})
      new_list4_dict = defaultdict(list) 
      for i in tqdm(new_dataset_1):
          label = i['label']
          pos_tagged_dataset = defaultdict(list)

          for key, value in i['new_h_p'].items():
              if isinstance(value, dict) and 'premise' in value and 'hypothesis' in value:
                  if separate_by_pos == 'yes':
                     
                      pos_tagged_dataset[value['pos'][0]].append({
                          'premise': value['premise'],
                          'hypothesis': value['hypothesis'],
                          'label': {'contradiction': 0, 'entailment': 1, 'neutral': 2}[label],
                      })

                  else:
                      new_list4.append({
                          'premise': value['premise'],
                          'hypothesis': value['hypothesis'],
                          'label': {'contradiction': 0, 'entailment': 1, 'neutral': 2}[label]
                      })
          label_counts[label] += len(i['new_h_p'])
  
          if separate_by_pos == 'yes':
            for pos, entries in pos_tagged_dataset.items():

                new_list4_dict[pos].extend(entries)
            
      if separate_by_pos == 'yes':
        for i, v in new_list4_dict.items():
          new_list4.append({i:v})


      print("Label counts of generated inflated problems:", label_counts)

      if output_format == 'txt' and output_file_2:
          with open(output_file_2, 'w') as file:
              model_details = f"\"fill-mask\", model=\"{model_name}\""
              file.write(f"\n{model_details}\n")
              pos_grouped = defaultdict(list)
              if separate_by_pos == 'yes':
                for entry in words_overall2[:100]:
                  ranks, pos_t=entry['ranks'], entry['pos']

              
                  for w, ranked_fillers in ranks.items():
                      pos_tag = pos_t.get(w, ["UNKNOWN"])[0]
                      pos_grouped[pos_tag].append((entry))

               
                for i, v in pos_grouped.items():
                  file.write(f"\n--- {i.upper()} ---\n")
                 
                  for y in v:
                    id, premise, hypothesis, label, ranks, pos = y['id'], y['premise'], y['hypothesis'], y['label'], y['ranks'], y['pos']
                    file.write(f"\n{id}: {label}\n")
                    for w, ranked_fillers in ranks.items():
                        p_masked = re.sub(rf'\b{w}\b', f'[{w}]', premise)
                        h_masked = re.sub(rf'\b{w}\b', f'[{w}]', hypothesis)
                        file.write(f"\n{p_masked}\n{h_masked}\n")

                        for f, v in ranked_fillers:
                            file.write(f"{round(v['average_rank']):>2} {f:<20} {v['ranks']} {v['average_prob']}   {v['individual_probs']}\n")
              else:
                for entry in words_overall2[:100]:
                  id, premise, hypothesis, label, ranks = entry['id'], entry['premise'], entry['hypothesis'], entry['label'], entry['ranks']
                  for w, ranked_fillers in ranks.items():
                    p_masked = re.sub(rf'\b{w}\b', f'[{w}]', premise)
                    h_masked = re.sub(rf'\b{w}\b', f'[{w}]', hypothesis)

                    file.write(f"\n{id}: {label}\n{p_masked}\n{h_masked}\n")
                    for f, v in ranked_fillers:

                      file.write(f"{round(v['average_rank']):>2} {f:<20} {v['ranks']} {v['average_prob']}   {v['individual_probs']}\n")

      if output_format == 'txt' and output_file:
        with open(output_file, 'w') as file:
            model_details = f"\"fill-mask\", model=\"{model_name}\""
            file.write(f"\n{model_details}\n")
            for entry in new_dataset_1:

                id, premise, hypothesis, label = entry['id'], entry['premise'], entry['hypothesis'], entry['label']

                file.write(f"\n{id}: {label}\n{premise}\n{hypothesis}\n")
                for w, ranked_fillers in entry['new_h_p'].items():
                    file.write(f"{ranked_fillers['premise']}\n{ranked_fillers['hypothesis']}\n")
        return new_list4, new_list3

      return new_list4, new_list3
