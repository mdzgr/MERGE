from collections import Counter, defaultdict
from wordfreq import zipf_frequency
from contextlib import redirect_stdout
from MERGE.generate import *
from MERGE.utils import *
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
    word_pos_lookup = return_word_pos_lookup(ids_across_datasets)

    renamed = rename_items(all_pairs, word_pos_lookup, opposite, label_for_shared_suggestions)

    return compute_and_report_stats(renamed, pos_name_file_name, type, min_count)


def maybe_init_cleaned_entry(save_cleaned_only, cleaned_data_file, sentence, word_with_pos):
    '''initializes nested entry in cleaned_data_file if saving cleaned data only'''
    if save_cleaned_only and cleaned_data_file is not None:
        cleaned_data_file.setdefault(sentence, {}).setdefault(word_with_pos, {})

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

def return_clean_suggestions_by_prob(list_suggestions, org_prob):
  return [c for c in list_suggestions if return_probability_suggestion(c) != '' and float(return_probability_suggestion(c)) >= org_prob]


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
        for s in set(lst):  # use set() to avoid duplicates in the same list
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
          save_suggestions_in_file=False, data_with_suggestions=None, avrg=None, type_pos_filtering=None,
          save_cleaned_only=False, cleaned_data_file=None):


  cleaned_suggestions, per_key_cleaned_lists, intersected_suggestions =  [], [], []
  maybe_init_cleaned_entry(save_cleaned_only, cleaned_data_file, sentence, word_with_pos)

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
      if save_cleaned_only and cleaned_data_file is not None:
        cleaned_data_file[sentence][word_with_pos][k] = cleaned_list
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


def process_word_for_alignment(word, idx, common_dict, data_with_suggestions, premise, hypothesis, p_positions, h_positions, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, type_pos_filtering, save_cleaned_only, cleaned_data_file, number_of_minimal_suggestions_common_bt_p_h, words, seed_dataset, item_id, counters, stats):
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

    premise_suggestions = process_matching_keys(premise_data, premise, word_with_pos, all_matching_keys_p, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, data_with_suggestions, avrg='yes', type_pos_filtering=type_pos_filtering, save_cleaned_only=save_cleaned_only, cleaned_data_file=cleaned_data_file)

    hypothesis_suggestions = process_matching_keys(hypothesis_data, hypothesis, word_with_pos, all_matching_keys_h, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, data_with_suggestions, avrg='yes', type_pos_filtering=type_pos_filtering, save_cleaned_only=save_cleaned_only, cleaned_data_file=cleaned_data_file)

    print_not_fully_tagged(pos_tag_filtering, premise_suggestions, hypothesis_suggestions)

    premise_fillers, hypothesis_fillers = return_suggestions(premise_suggestions), return_suggestions(hypothesis_suggestions)

    skip= assert_suggestions_to_skip(premise_fillers, hypothesis_fillers, number_of_minimal_suggestions_common_bt_p_h, words, seed_dataset, item_id, counters, stats)
    if skip:
      return None
    premise_probabilities, hypothesis_probabilities = return_probabilities(premise_suggestions), return_probabilities(hypothesis_suggestions)

    return key, premise_fillers, hypothesis_fillers, premise_probabilities, hypothesis_probabilities, pos, all_matching_keys_p, all_matching_keys_h



def return_fllers_probabilities_pos_positions(data_with_suggestions, premise, hypothesis, pos_p,  pos_h, tok_p, tok_h, pos_to_mask, source_1, source_2, allowed_pos_tags, pos_tag_filtering, prob, nlp, number_batch_for_pos_tagging, save_suggestions_in_file, type_pos_filtering, svae_cleaned_value, quality_filtered_suggestions, number_of_minimal_suggestions_common_bt_p_h, seed_dataset, id, counters):
  word2fillers, word2probabilities, positions = [defaultdict(list), defaultdict(list), defaultdict(list)]
  if premise in data_with_suggestions.keys() and hypothesis in data_with_suggestions.keys():
    common_dict, p_positions, h_positions, _ = common(premise, hypothesis, pos_p, pos_h, tok_p, tok_h, pos_to_mask, source_1, source_2)

    singles= return_words_of_2_sentences(premise, hypothesis)
    words = list(common_dict.keys())

    stats={'words_with_not_enough_replacements_inside_loop_utilitary':0}


    for i, word in enumerate(words):
      result = process_word_for_alignment(word, i, common_dict, data_with_suggestions, premise, hypothesis, p_positions, h_positions, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, type_pos_filtering, svae_cleaned_value, quality_filtered_suggestions, number_of_minimal_suggestions_common_bt_p_h, words, seed_dataset, id, counters, stats)

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
  '''takes a list of replacememts, their prob, type rank operation (union or intersection) and calculate rank'''
  words = {}
  for w in word2fillers:

      words[w] = ranked_overlap(word2fillers[w], word2probabilities[w], type_rank_operation).items()
      words[w] = sorted(words[w], key=lambda x: x[1][type_rank]) # type of ranking -: average rank or prob, and sort by that
  return words

def build_eval_type(type_of_eval, add_operation_suffix, operation_for_pos_filtering, add_opposite_suffix, opposite_value, folder_flatten_var, num_suggestions):
    if type_of_eval not in return_config():
        return None, None

    if add_operation_suffix:
        type_of_eval += "c" if operation_for_pos_filtering == "all_pos_tags_of_class_of_replaced_word" else "pos"
    if add_opposite_suffix:
        type_of_eval += "pAh" if opposite_value is True else "pph"

    return f"{folder_flatten_var}all.1.{type_of_eval}.te.inf.{num_suggestions}.json", type_of_eval

def print_check_statements(type_eval, type_eval_acro, pos_value, list_files_predict):
    print("-" * 40)
    print(f"THE TYPE OF EVAL: {type_eval}")
    print(f"THE TYPE OF EVAL ACRONYM: {type_eval_acro}")
    print(f"THE POS VALUE: {pos_value}")
    print(f"THE LIST OF FILES TO PREDICT: {list_files_predict}")
    print("-" * 40)

def return_base_common_ids(pos, type_of_procesing): #n_nouns_dataset_bothcpAh_ran_var_20.json
  all_base_ids_common_list=set(get_base_ids_from_data(load_data(f"/content/drive/MyDrive/MERGE/variants/samples/snli/final_cpah/{pos}s_dataset_bothcpAh_sel.json"), "flat")) if type_of_procesing!='first' else None
  return all_base_ids_common_list

def wrapper_merge_across_models(pos, processed_results, models, type_evaluation, min_count, opposite, label_for_shared_suggestions, name_pos_files, output_directory):
    pos_, average_pos = merge_and_analyze_from_results(processed_results, models=models, pos_tag=pos, type_evaluation=type_evaluation, min_count=min_count, opposite=opposite, label_for_shared_suggestions=label_for_shared_suggestions)
    create_json_from_data(pos_, output_directory+name_pos_files[pos])
    return pos_, average_pos


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
                    processed_second_data,
                    seed_dataset,
                    labels_sample,
                    number_of_minimal_suggestions_common_bt_p_h,
                    nlp,
                    svae_cleaned_value, #
                    name_file_cleaned_, #
                    number_batch_for_pos_tagging,
                    mock_test: str=None,
                    pos_tag_filtering:str=None,
                    rank_option='top',
                    save_suggestions_in_file: str = None,
                    prob:str=None,
                    type_rank:str=None,
                    type_rank_operation:str=None,
                    model_name:str=None,
                    type_pos_filtering:str=None,
                    ):
    """

    Matches premise and hypothesis from second_data with data_with_suggestions, replaces words, applies ranking,
    transforms the dataset, and optionally groups it by POS tags.

    output_file: name of the output file
    min_common_words: for filtering, minimum number of common words between premise and hypothesis
    mapping: dict with sent annotations from dataset
    ranked_overlap: function that ranks words based on their position in the lists
    pos_to_mask: pos tag to be looked for to be common bt premise and hypothesis

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
    check_type_return_correct(type_rank)
    processed_data, replacement_summary, quality_filtered_suggestions  = [], {}, {}
    counters = {'global_words_without_replacements': 0, 'problems_removed_due_to_low_suggestions': 0,}
    allowed_pos_tags = return_pos_additional_noun_tags(pos_to_mask)
    source_1, source_2= return_sources()

    expected_generation = {'neutral': 0, 'entailment': 0, 'contradiction': 0}
    for entry in tqdm(processed_second_data):
        id, premise, hypothesis, tok_p, pos_p, tok_h, pos_h, label = unpack_entry_dataset(entry)
        if mock_test and id != mock_test:
          continue

        word2fillers, word2probabilities, positions, words = return_fllers_probabilities_pos_positions(data_with_suggestions, premise, hypothesis, pos_p, pos_h, tok_p, tok_h, pos_to_mask,
                                                                                                                 source_1, source_2, allowed_pos_tags, pos_tag_filtering, prob, nlp, number_batch_for_pos_tagging, save_suggestions_in_file,
                                                                                                                 type_pos_filtering, svae_cleaned_value, quality_filtered_suggestions, number_of_minimal_suggestions_common_bt_p_h, seed_dataset, id, counters)
        words = return_replacements_ranked(word2fillers, word2probabilities, type_rank_operation, type_rank)
        for w, ranked_fillers in words.items():                                                        #Ranked fillers for church:NN:5:11:4:10: [('house', {'average_rank': 0.5, 'ranks': [1], 'average_prob': '1.08e-01', 'individual_probs': ['1.08e-01']}), ('building', {'average_rank': 0.0, 'ranks': [0], 'average_prob': '1.52e-01', 'individual_probs': ['1.52e-01']})]
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

    write_general_statistics(output_file,  f"{output_file}_general_stats.txt", actual_generetion(processed_data), expected_generation, counters['global_words_without_replacements'], counters['problems_removed_due_to_low_suggestions'])

    create_json_from_data(processed_data, output_file)
    if save_suggestions_in_file:
      create_json_from_data(data_with_suggestions, save_suggestions_in_file)
    if svae_cleaned_value:
      create_json_from_data(quality_filtered_suggestions, name_file_cleaned_)

    return processed_data, seed_dataset, replacement_summary

def return_models_short_to_acronyms_mapping():
  return {"bertb": "bb", "robertab": "rb", "deberta": "d","albertb": "ab","robertal": "rl","electrl": "el","bertl": "bl","albertxxl": 'axxl',"bartl": 'bal',"bartb": "bab", "electrab": 'elb'} #}


def return_class_to_short_class_mapping():
  return {"noun": "n", "verb": "v", "adjective": "adj", "adverb": "adv"}


def return_config():
  return {'prob':  ("yes", None), "pos":   ("no",  "yes"), "none":  ("no",  None), "both":  ("yes", "yes")}



def model_id_for_file_name():
  return {"bertv": "bert" , "robertab": "roberta", "deberta": "deberta", "albertb": "albert", "robertal": 'robertal', "electrl": "electral", "bertl": "bertl", "albertxxl": 'albertxxl', "bartl": 'bartl', "bartb": "bartb", "electrab": 'electrab'}



def return_filtered_sent_for_process_function(original_dataset, split, mapping, pos_to_mask,
                                               dataset_filtered_already_for_shared_words, min_common_words,
                                               num_sentences_to_process_dataset, num_sentences_compliant_criteria,
                                               number_of_maximum_annotators, minimum_no_words_premise,
                                               minimum_no_words_hypothesis, neutral_number, entailment_number,
                                               contradiction_number, if_ids_exist):
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

    processed_second_data = pos_toks_extract_from_dataset(SNLI_filtered_2, mapping)
    seed_dataset, labels_sample = process_unmasked_dataset(processed_second_data, neutral_number, entailment_number, contradiction_number, include_id=True)  # <-- fixed name
    return processed_second_data, seed_dataset, labels_sample
def register_eval_type(type_of_eval_acum1, add_operation_suffix, operation_for_pos_filtering,
                        add_opposite_suffix, opposite_value, folder_flatten_var,
                        number_suggestions_required_per_id, list_files_to_predict, type_eval_acronyms):
    """Builds the eval type/filename for one accumulator value and tracks it if valid.
    Mutates list_files_to_predict and type_eval_acronyms in place."""
    file_to_predict, type_of_eval = build_eval_type(
        type_of_eval_acum1, add_operation_suffix, operation_for_pos_filtering,
        add_opposite_suffix, opposite_value, folder_flatten_var, number_suggestions_required_per_id)

    if file_to_predict:
        list_files_to_predict.append(file_to_predict)
        type_eval_acronyms.append(type_of_eval)

    return type_of_eval


def process_model_pos_combination(model, pos, type_of_eval, dataset, split, mapping,
                                   min_common_words, num_sentences_to_process_dataset,
                                   num_sentences_compliant_criteria, an_no, words_premise_min,
                                   words_hypothesis_min, neutral_label_, entailment_label_,
                                   contradiction_label_, type_of_procesing, nlp,
                                   cleaned_file_value_1, no_sentences_classified_batch,
                                   mock_test_value, pos_value, number_suggestions_to_be_considered,
                                   prob_value, operation_for_pos_filtering,
                                   number_suggestions_required_per_id,
                                   number_of_minimal__shared_suggestions_p_h):
    """
    Runs the full suggestion pipeline for one (model, pos) pair: filters seed sentences,
    loads that model's suggestions, processes the dataset, writes the per-combo output
    files, and returns the resulting variants.
    """
    all_base_ids_common_list = return_base_common_ids(pos, type_of_procesing)

    processed_second_data, seed_dataset, labels_sample = return_filtered_sent_for_process_function(
        dataset, split, mapping, pos, None, min_common_words,
        num_sentences_to_process_dataset, num_sentences_compliant_criteria, an_no,
        words_premise_min, words_hypothesis_min, neutral_label_, entailment_label_,
        contradiction_label_, all_base_ids_common_list)

    suggestion_file = f"/content/drive/MyDrive/MERGE/variants/samples/snli/tagged_sugg/{model}.1.{pos}.200.test.json"
    data_suggestions = load_data(suggestion_file)

    output_processed_dataset = generate_output_filenames(
        suggestion_file, models_dictionary=return_models_short_to_acronyms_mapping(),
        pos_dicitonary=return_class_to_short_class_mapping(),
        number_inflation=number_suggestions_required_per_id, type_dataset=type_of_eval)

    variants, seed_dataset, dict_sumarr = process_dataset(
        data_with_suggestions=data_suggestions,
        output_file=output_processed_dataset,
        pos_to_mask=pos,
        processed_second_data=processed_second_data,
        seed_dataset=seed_dataset,
        labels_sample=labels_sample,
        number_of_minimal_suggestions_common_bt_p_h=number_of_minimal__shared_suggestions_p_h,
        nlp=nlp,
        svae_cleaned_value=cleaned_file_value_1,
        name_file_cleaned_=f'conffusion_matrix_cleaned_suggesrions_{pos}_{type_of_eval}_{model}.json',
        number_batch_for_pos_tagging=no_sentences_classified_batch,
        mock_test=mock_test_value,
        pos_tag_filtering=pos_value,
        rank_option=number_suggestions_to_be_considered,
        save_suggestions_in_file=f'{model}_{pos}_{type_of_eval}_pos.json',
        prob=prob_value,
        type_rank='average_prob',
        type_rank_operation='union',
        model_name=model,
        type_pos_filtering=operation_for_pos_filtering
    )

    create_json_from_data(dict_sumarr, f'replacements_positions_{pos}_{type_of_eval}.json')
    return variants


def run_model_pos_grid(models, pos_to_mask, type_of_eval, dataset, split, mapping, min_common_words,
                        num_sentences_to_process_dataset, num_sentences_compliant_criteria, an_no,
                        words_premise_min, words_hypothesis_min, neutral_label_, entailment_label_,
                        contradiction_label_, type_of_procesing, nlp, cleaned_file_value_1,
                        no_sentences_classified_batch, mock_test_value, pos_value,
                        number_suggestions_to_be_considered, prob_value, operation_for_pos_filtering,
                        number_suggestions_required_per_id, number_of_minimal__shared_suggestions_p_h):
    """Runs process_model_pos_combination for every (model, pos) pair, keyed as 'model_pos'."""
    processed_results = {}
    for model in models:
        for pos in pos_to_mask:
            variants = process_model_pos_combination(
                model, pos, type_of_eval, dataset, split, mapping, min_common_words,
                num_sentences_to_process_dataset, num_sentences_compliant_criteria, an_no,
                words_premise_min, words_hypothesis_min, neutral_label_, entailment_label_,
                contradiction_label_, type_of_procesing, nlp, cleaned_file_value_1,
                no_sentences_classified_batch, mock_test_value, pos_value,
                number_suggestions_to_be_considered, prob_value, operation_for_pos_filtering,
                number_suggestions_required_per_id, number_of_minimal__shared_suggestions_p_h)
            processed_results[f"{model}_{pos}"] = variants
    return processed_results


def maybe_merge_datasets(merge_datasets, pos_to_mask, type_of_eval, processed_results, models,
                          number_suggestions_required_per_id, opposite_value,
                          label_type_aggregation_across_models, output_directory):
    """If merge_datasets is truthy, merges suggestions across models for every pos tag."""
    if not merge_datasets:
        return
    name_pos_files = {pos: f"{pos}_dataset_{type_of_eval}_sel.json" for pos in pos_to_mask}
    for pos in pos_to_mask:
        wrapper_merge_across_models(pos, processed_results, models, type_of_eval,
                                     number_suggestions_required_per_id, opposite_value,
                                     label_type_aggregation_across_models, name_pos_files,
                                     output_directory)
