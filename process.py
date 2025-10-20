from collections import Counter, defaultdict
from wordfreq import zipf_frequency
from contextlib import redirect_stdout
from MERGE.generate_suggestions import filter_snli, pos_toks_extract_from_dataset, process_unmasked_dataset, common
import string, json, os, re, random
from tqdm import tqdm
import sys, contextlib
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

    COMMON_PREFIXES = {
    "anti", "auto", "bi", "co", "contra", "counter", "de", "dis", "en", "em",
    "extra", "hetero", "homo", "hyper", "il", "im", "in", "ir", "inter", "intra",
    "macro", "micro", "mid", "mis", "mono", "multi", "non", "over", "post",
    "pre", "pro", "pseudo", "re", "semi", "sub", "super", "tele", "trans",
    "tri", "ultra", "un", "under"
    }

    COMMON_SUFFIXES = {
        "able", "ible", "al", "ally", "ance", "ence", "ant", "ent",
        "ary", "ery", "ory", "ate", "ed", "en", "er", "est", "ful",
        "hood", "ic", "ical", "ify", "ing", "ion", "tion", "sion",
        "ish", "ism", "ist", "ity", "less", "let", "like", "ling",
        "ly", "ment", "ness", "ous", "ship", "y"
    }

    filtered = []

    for candidate in candidates:
        # print('THE CANDIDATE', candidate)
        word = candidate.split(":")[0]
        if len(word)==1:
            continue # word with len(1) usually modify structure
            #'blue shirt' > 'f shirt'
        if word in COMMON_PREFIXES or word in COMMON_SUFFIXES:
            continue
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
    suggestions,                            # alist of suggestions
    original_sentence,                      #the original sentence
    start_idx,                              #where to put the suggestions
    end_idx,                                #where that ends
    allowed_pos_tags,                       #what are the allowed pos tags
    nlp,                                    #the nlp pipeline
    batch_size_no                           # the batch size for how many suggestions will be tagged
    ):
    """
    tags suggestions via spacy nlp pipeline
    replace themn in sentence > tags suggestions in context
    > returns a list with filtered suggestions with their pos tags
    and a count of the pos tags

    """

    docs, words, pos_counts, filtered=[], [], Counter(), []
    for suggestion in suggestions:
        word = suggestion.split(":")[0]           #split to get only the word
        temp_sentence = original_sentence[:start_idx] + word + original_sentence[end_idx:]  #replace it in the sentence !!! why is this based on positions and not regex?
        docs.append(temp_sentence)                #append variants ctreated
    for doc in nlp.pipe(docs, batch_size=batch_size_no): # use pipleline to process in batches
      for token in doc:
          if token.idx == start_idx:                # if the character onset == the onset of the replaced word
              tokens_with_tags=token.text+':'+token.tag_ #get the token, add the ta
              words.append(tokens_with_tags)         #append it to a list

    pos_dict = {w.split(":")[0]: w.split(":")[1] for w in words} # make a dict word: pos tag
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

def get_all_matching_keys(data, word_with_pos, pos_list):
  '''match beginning key entries in dicitionary that start with positions of words'''
                                                    # pos_list premise_pos_list [(11, 16)]
                                                    # data=sentence
                                                    #word with_pos orginial_word:itspos
                                                    #{'poses:VBZ': {'26:31:4.23e-04': ['','']}
  all_matching_keys = []
  for i in pos_list:                                  #for each position
    p_start, p_end = i                                 #unpack into start and end, e.g. 11, 16
    key_prefix = f"{p_start}:{p_end}:"                #join
    matching_keys = [
        k for k in data[word_with_pos].keys()
        if k.startswith(key_prefix)
    ]                                                #get the keys that start with thes epositions
    all_matching_keys.extend(matching_keys)
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

def merge_and_analyze_datasets(dataset1, source1,
    dataset2, source2,
    pos_name_file_name,
    type,
    min_count,
    name=None,
    opposite=True,
    others=None,
    label_for_shared_suggestions: str = "general"): #~ modify this such that it can take other models as well
    '''
    Function that merges datasets of several models
    Keeps suggestions for one replacement only if that replacement has sufficient variants
    '''

    try:
        print(f"dataset1 size={len(dataset1)}; dataset2 size={len(dataset2)}")
    except Exception:
        pass


    def without_last_2(full_id):
      '''without_last_2()  IN : 2677109430.jpg#1r1n:church:NN:5:11:4:10:building:h:bert
      without_last_2()  OUT: 2677109430.jpg#1r1n:church:NN:5:11:4:10:building
      '''
      result = ':'.join(full_id.split(':')[:-2])  # everything except origin and model

      return result

    def id_0(full_id):
      '''id_0()  IN : 2677109430.jpg#1r1n:church:NN:5:11:4:10:structure:h:roberta
        id_0()  OUT: 2677109430.jpg#1r1n'''
      result = full_id.split(':')[0]  # everything except origin and model
      return result

    def shared_label(num_models: int):
        if label_for_shared_suggestions.lower() == "fine-grained":
            return f"shared-{num_models}"
        return "both"

    all_pairs = [(dataset1, source1), (dataset2, source2)]
    if others:
        all_pairs.extend(others)
    all_ids,  word_pos_lookup= set(), {}
    for dataset, _ in all_pairs:
    # add every id string into the set
      all_ids.update(item['id'] for item in dataset)
                                  #open the dataset and add the ids to all ids
                                  #    [D2 Item 1] FULL ITEM: {'id': '2677109430.jpg#1r1n:church:NN:5:11:4:10:roof:h:roberta', 'premise': 'This roof choir sings to the masses as they sing joyous songs from the book at a roof.', 'hypothesis': 'The roof has cracks in the ceiling.', 'label': 'neutral'}
                                  # -> Added ID: 2677109430.jpg#1r1n:church:NN:5:11:4:10:roof:h:roberta
                                  # -> Unique ID count so far: 3
    for n, full_id in enumerate(sorted(all_ids), start=1):
        parts = full_id.split(':') #split id
        word_pos_id = ':'.join(parts[:3] + [parts[-3]])
        origin, model = parts[-2],  parts[-1]
        word_pos_lookup.setdefault(word_pos_id, {}).setdefault(origin, set()).add(model) #word_pos_lookup structure {'2677109430.jpg#1r1n:church:NN:apartment': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:attic': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:basement': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:bathroom': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:bedroom': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:building': {'h': {'roberta', 'bert'}}, '2677109430.jpg#1r1n:church:NN:cabin': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:cellar': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:flat': {'h': {'roberta'}}}
    renamed, seen_ids = [], set()


    for d_idx, (dataset, src) in enumerate(all_pairs, start=1):
        for k, item in enumerate(dataset, start=1): # item structure   Raw item: {'id': '2677109430.jpg#1r1n:church:NN:5:11:4:10:house:h:bert', 'premise': 'This house choir sings to the masses as they sing joyous songs from the book at a house.', 'hypothesis': 'The house has cracks in the ceiling.', 'label': 'neutral'}
            parts = item['id'].split(':')
            word_pos_id = ':'.join(parts[:3] + [parts[-3]])
            origin, model, base_id = parts[-2], parts[-1], without_last_2(item['id'])  ###word_pos_id: 2677109430.jpg#1r1n:church:NN:house
                                                                                         #origin: h,  #model: bert
            new_id = None

            if origin in ['h', 'p']: # if the origin is p or h meaning the suggestion was not common for p/h from one of the models
                opposite_origin = 'p' if origin == 'h' else 'h'  #what is the opposite

                has_opposite = (
                    word_pos_id in word_pos_lookup and
                    opposite_origin in word_pos_lookup[word_pos_id] #check if that exists as a key in the dicitonary
                )

                token = "ph" if has_opposite else origin

                if opposite:
                    if not has_opposite:
                        continue                   # skip the sentence if it does not have an opposite and the argument of opposite is True
                    opposite_models = word_pos_lookup[word_pos_id][opposite_origin] #check the opposite modle

                    new_id = (
                        f"{base_id}:ph:{shared_label(len(set(word_pos_lookup[word_pos_id].get(origin, set())) | set(opposite_models)))}"
                        if any(m != model for m in opposite_models)
                        else f"{base_id}:ph:{model}"
                        )
                else:
                    new_id = f"{base_id}:{token}:{model}"
                                                    #word_pos_lookup structure {'2677109430.jpg#1r1n:church:NN:apartment': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:attic': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:basement': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:bathroom': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:bedroom': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:building': {'h': {'roberta', 'bert'}}, '2677109430.jpg#1r1n:church:NN:cabin': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:cellar': {'h': {'roberta'}}, '2677109430.jpg#1r1n:church:NN:flat': {'h': {'roberta'}}}

            elif origin == 'ph':
                                                  #if ph, just check if another model exists
                                                  #so if there is any one of these entries with another model
                origins = word_pos_lookup.get(word_pos_id, {})
                other_origins_exist = (any(origins.get(o, set()) - {model} for o in ('p', 'h', 'ph')))
                if other_origins_exist:
                  involved_models = set()
                  for o in ('p', 'h', 'ph'):
                      involved_models |= origins.get(o, set())
                  label = shared_label(len(involved_models))
                  new_id = f"{base_id}:ph:{label}"
                else:
                  new_id = f"{base_id}:ph:{model}"
            else:
                continue


            if word_pos_id not in seen_ids:        # if the word is not in seen ids, copy element, replace id, add to renamed, and add to seen ids
                new_item = item.copy()
                new_item['id'] = new_id
                renamed.append(new_item)
                seen_ids.add(word_pos_id)
            else:
                continue


    outfile = f"potential_variants_{pos_name_file_name}_{type}_stats.txt"  # change path/name if you like

    with open(outfile, "w", encoding="utf-8") as f:
    # Save the original stdout
      original_stdout = sys.stdout

      # Redirect stdout to both file and console
      def dual_print(*args, **kwargs):
          print(*args, **kwargs, file=original_stdout)
          print(*args, **kwargs, file=f)

      # now use dual_print instead of print
      final_counts = Counter(id_0(item['id']) for item in renamed)
      qualified_base_ids = {bid for bid, count in final_counts.items() if count >= min_count}
      dual_print(f"Qualified base IDs (count >= {min_count}): {len(qualified_base_ids)}")

      final_dataset = [item for item in sorted(renamed, key=lambda x: x['id']) if id_0(item['id']) in qualified_base_ids]
      dual_print(f"length of final dataset {len(final_dataset)}")

      source_counts = Counter(item['id'].split(':')[-1] for item in final_dataset)
      total_instances = sum(source_counts.values())
      dual_print(f"Total instances in final dataset: {total_instances}")
      dual_print(f"Source counts: {dict(source_counts)}")

      average_per_source = ({src: count / total_instances for src, count in source_counts.items()} if total_instances else {})
      label_counts = Counter(item['label'] for item in final_dataset)

      dual_print("\n=== Unique  base_ids (initial SNLI ids) that have enough suggestions", len(qualified_base_ids))
      dual_print("=== Label Counts in Final Dataset ===")
      for label, count in label_counts.items():
          dual_print(f"{label}: {count}")

      dual_print("\n=== Average Instances per Source ===")
      for src, avg in average_per_source.items():
          dual_print(f"{src}: {avg:.2f}")
    return final_dataset, average_per_source

def process_matching_keys(data, sentence, word_with_pos, all_matching_keys,
          allowed_pos_tags,  pos_tag_filtering, prob, nlp, singles, batch_nlp_classification_no,
          save_suggestions_in_file=False, data_with_suggestions=None, outside_function_counter_for_count_for_most_common_words_their_pos_tag=None,
          outside_function_counter_for_count_for_pos=None, avrg=None, type_pos_filtering=None, total_words=None, total_remaining_suggestions=None,
          total_words_with_10_plus=None, no_prob_counter=None, average_prob_suggestions=None, diff_total=None, num_sentences=None, num_sentences_binary=None,
          words_replaced=None, average_prob_replaced=None,
          save_cleaned_only=False, cleaned_data_file=None):
                                                                                      # average_prob_suggestions = probability of suggestions in a list for a replacement word
                                                                                    # no_prob_counter = how many occurances marked by positions do not have probabilities bc they were not in the first 200 words
                                                                                    # total_words_with_10_plus = no. replaced words that have more than 10 words after probability filtering
                                                                                    # total_remaining_suggestions = no. of remaning suggestions after all filtering
                                                                                    # count = no that is add if pos tag filtering is yes, for each replaced occurance
                                                                                    # diff total = difference between before and after pos tag filtering, added outside loop
                                                                                    # total_Words = count increasing with each processed occurance
                                                                                    #outputsȘ intersected_suggestions, added_pos_counts

  cleaned_suggestions, added_pos_counts, per_key_cleaned_lists, intersected_suggestions =  [], False, [], []
  if save_cleaned_only and cleaned_data_file is not None:
    if sentence not in cleaned_data_file:
      cleaned_data_file[sentence] = {}
    if word_with_pos not in cleaned_data_file[sentence]:
      cleaned_data_file[sentence][word_with_pos] = {}
  if len(all_matching_keys) == 0:
    print("No matching keys found", sentence)
  for k in all_matching_keys: #
      count_for_most_common_words_their_pos_tag, count_for_pos=Counter(), Counter()
      data_key = data[word_with_pos][k]; has_p = has_pos_tags(data_key); temporary_suggestions = []   #get suggestions for that key  #check if everything has POS tag
      if has_p == False and pos_tag_filtering == 'yes':                       #if not everything has POS tag and the argument for pos_tag_filtering  is yes
          p_start, p_end, _ = k.split(":")
          suggestions, pos_counts = filter_suggestions_by_contextual_pos(
              data_key, sentence, int(p_start), int(p_end), allowed_pos_tags, nlp, batch_nlp_classification_no
          )
          if save_suggestions_in_file and data_with_suggestions is not None: #if save filtered data is yes (which would mean replace suggestions with the new suggestion:pos) and if we have a file with suggestions
              data_with_suggestions[sentence][word_with_pos][k] = suggestions #replace the suggestions we have now in the file with the tagged suggestions
          temporary_suggestions.extend(suggestions)                           # add suggestion to temporary list
      else:
          temporary_suggestions.extend(data[word_with_pos][k])                #same here
      if pos_tag_filtering == 'yes' and avrg == 'yes':
        for w, _, p, *_ in (s.split(":") for s in temporary_suggestions):
            count_for_most_common_words_their_pos_tag[f"{w}:{p}"] += 1 ## for each sugggestion add count of pos tag

        count_for_pos.update(count_pos(temporary_suggestions))
        added_pos_counts = True
        if outside_function_counter_for_count_for_most_common_words_their_pos_tag is not None:
            outside_function_counter_for_count_for_most_common_words_their_pos_tag.update(count_for_most_common_words_their_pos_tag)
        if outside_function_counter_for_count_for_pos is not None:
            outside_function_counter_for_count_for_pos.update(count_for_pos)

      if type_pos_filtering in ('all_pos_tags_of_class_of_replaced_word', 'pos_tag_of_replaced_word'):
        o = allowed_pos_tags if type_pos_filtering == 'all_pos_tags_of_class_of_replaced_word' else [word_with_pos.split(':')[1]]

      cleaned_list = filter_candidates(temporary_suggestions, singles)

      if pos_tag_filtering == 'yes':                                           #THIS IS WHERE POS TAG FILTERING IS DONE #####

          len_before = len(cleaned_list)
          cleaned_list = [s for s in cleaned_list if s.split(":")[-1] in o]
          len_after = len(cleaned_list)
          diff_total += (len_before - len_after)                                          # for each occurance of a word replaced, calculate how big is the diff
          if len(cleaned_list) == 0:                                                      #if no suggestions left, continue
              continue
          average_prob_suggestions += sum(float(c.split(":")[1]) for c in cleaned_list) / len(cleaned_list) #sum the probabilityies and divide them by the length of the list
      if prob == "yes":
          try:
              original_prob = float(k.split(":")[2])
              average_prob_replaced+=original_prob
          except (IndexError, ValueError):
              no_prob_counter += 1 # continu eif the word did not have the prob
              continue

          words_replaced+=1 if original_prob !=None else 0
          # print(cleaned_list)
          # cleaned_list_str = [c for c in cleaned_list if type(c.split(":")[1]) == 'str']
          # print(cleaned_list_str)
          cleaned_list = [c for c in cleaned_list if c.split(":")[1] != '' and float(c.split(":")[1]) >= original_prob]
          total_words_with_10_plus += 1 if len(cleaned_list) > 10 else 0

          for c in cleaned_list:
            if (prob1 := float(c.split(":")[1])) < original_prob:
                print(f"DEBUG: Found problematic entry: {c}, prob={prob1}, threshold={original_prob}")
      if save_cleaned_only and cleaned_data_file is not None:
        cleaned_data_file[sentence][word_with_pos][k] = cleaned_list
      total_remaining_suggestions += len(cleaned_list); total_words += 1; per_key_cleaned_lists.append(cleaned_list); cleaned_suggestions.extend(cleaned_list)

  if len(all_matching_keys) == 1:
    intersected_suggestions = cleaned_suggestions
  elif len(all_matching_keys) > 1:
    presence = defaultdict(list)
    for lst in per_key_cleaned_lists:
      for s in set(lst):  #                                                       use set() to avoid duplicates in the same list
        if pos_tag_filtering =='yes':
          word, val, pos = s.split(":")
          presence[word].append((float(val), pos))
        else:
          parts = s.split(":")
          word, val = parts[0], parts[1]
          pos1=None   # first two pieces
          presence[word].append((float(val), pos1))
    keep = {
          w: f"{w}:{sum(v for v, _ in vals) / len(vals)}:{vals[0][1]}"            # assumming POS tag is the same for the 2 options
          for w, vals in presence.items() if len(vals) >= 2                     # appears in ≥2 keys
      }
    intersected_suggestions = [
          keep[s.split(":")[0]] for s in cleaned_suggestions if s.split(":")[0] in keep
      ]
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
                    model_ids_dictionary,
                    svae_cleaned_value,
                    name_file_cleaned_,
                    number_batch_for_pos_tagging,
                    mock_test: bool = False,
                    id_mock_test: str=None,
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

    if if_ids_exist!=None:
      SNLI_filtered_2={key: value for key, value in SNLI_filtered_2.items() if key in if_ids_exist} # if id is in if_ids_exist
      print(f"Filtered length: {len(SNLI_filtered_2)}") #sub-sample
    processed_second_data = pos_toks_extract_from_dataset(SNLI_filtered_2, mapping)
    seed_dataset, labels_sample = process_unmasked_dataset(processed_second_data, neutral_number, entailment_number, contradiction_number, id=True)
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
    ids_multiple_keys, replacement_summary, quality_filtered_suggestions=[], {}, {}
    for entry in tqdm(processed_second_data):

        id, premise, hypothesis, tok_p, pos_p, tok_h, pos_h, label = (entry['id'], entry['premise'], entry['hypothesis'], entry['p_t'],entry['p_p'], entry['h_t'], entry['h_p'], entry['label'] )


        if mock_test and id != id_mock_test:
          continue
        # print(id)
        # print(premise)
        # print(hypothesis)
        premise_id = premise
        hypothesis_id = hypothesis
        word2fillers, word2probabilities, word2pos, _, _, _, positions = [defaultdict(list), defaultdict(list), defaultdict(int), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
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


            all_matching_keys_p = get_all_matching_keys(premise_data, word_with_pos, premise_pos_list)
                                                                                        ## result get_all_matching_keys the matching keys ['48:51:7.56e-01'] all matching keys ['20:23:9.83e-01', '48:51:7.56e-01'], or single element if single apperance
            all_matching_keys_h = get_all_matching_keys(hypothesis_data, word_with_pos, hypothesis_pos_list)
            p_starrt, p_end, _= all_matching_keys_p[0].split(':')
            h_starrt, h_end, _= all_matching_keys_h[0].split(':')
            key=f"{word}:{pos}:{p_starrt}:{p_end}:{h_starrt}:{h_end}"                   #this will be indexed only with the first position

            premise_suggestions, premise_pos_filter_applied = process_matching_keys(
                premise_data, premise, word_with_pos, all_matching_keys_p, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, data_with_suggestions, count_for_most_common_words_their_pos_tag_p, overall_count_for_pos_p, avrg='yes', type_pos_filtering=type_pos_filtering, total_words=total_words_p,
                total_remaining_suggestions=number_premise_suggestions_remaining_all_filtering, total_words_with_10_plus=no_premise_words_with_10more_suggestions_higher_probability, no_prob_counter=premise_replaced_words_had_no_probability, average_prob_suggestions=premise_avg_suggestion_prob, diff_total=premise_diff_after_pos_filter, num_sentences=num_sentences_both, num_sentences_binary='yes', words_replaced= words_replaced_p,
                average_prob_replaced=average_prob_replaced_premise, save_cleaned_only=svae_cleaned_value, cleaned_data_file=quality_filtered_suggestions
              )

            hypothesis_suggestions, hypothesis_pos_filter_applied = process_matching_keys(
                hypothesis_data, hypothesis, word_with_pos, all_matching_keys_h, allowed_pos_tags, pos_tag_filtering, prob, nlp, singles, number_batch_for_pos_tagging, save_suggestions_in_file, data_with_suggestions, count_for_most_common_words_their_pos_tag_h, overall_count_for_pos_h, avrg='yes', type_pos_filtering=type_pos_filtering, total_words=total_words_h,
                total_remaining_suggestions=number_hypothesis_suggestions_remaining_all_filtering, total_words_with_10_plus=no_hypothesis_words_with_10more_suggestions_higher_probability, no_prob_counter=hypothesis_replaced_words_had_no_probability,
                average_prob_suggestions=hypothesis_avg_suggestion_prob, diff_total=hypothesis_diff_after_pos_filter, words_replaced= words_replaced_h, average_prob_replaced=average_prob_replaced_hypothesis, save_cleaned_only=svae_cleaned_value, cleaned_data_file=quality_filtered_suggestions
              )
            # print('pos tag filtering', pos_tag_filtering)
            if (
                  (pos_tag_filtering == 'yes' and not has_pos_tags(premise_suggestions))
                  or (pos_tag_filtering == 'yes' and not has_pos_tags(hypothesis_suggestions))
                  or (pos_tag_filtering == 'yes'  and not premise_pos_filter_applied)
                  or (pos_tag_filtering == 'yes'  and not hypothesis_pos_filter_applied)
              ):
                  print("Some of the suggestions for premise or hypothesis are not tagged for POS tag")
            # print('the length of premise', len(premise_suggestions))
            # print('the length of hypothesis', len(hypothesis_suggestions))
            premise_fillers= [c.split(":")[0] for c in premise_suggestions] #suggestions
            hypothesis_fillers= [c.split(":")[0] for c in hypothesis_suggestions]
            if len(premise_fillers)==0 and len(hypothesis_fillers)==0: #I concluded this does not affect the code
              continue
            common_suggestions = set(premise_fillers) & set(hypothesis_fillers)
            if len(common_suggestions) < number_of_minimal_suggestions_common_bt_p_h: #this is useful only when setting a number of minnimal suggestions
                global_words_without_replacements += 1

                if len(words) == 1 or (len(words) >= 1 and words_with_not_enough_replacements_inside_loop_utilitary == len(words) - 1):
                    seed_dataset = [item for item in seed_dataset if item['id'] != id]
                    problems_removed_due_to_low_suggestions += 1
                elif len(words) >= 1:
                    words_with_not_enough_replacements_inside_loop_utilitary += 1
                continue

            premise_probabilities = [float(c.split(":")[1]) for c in premise_suggestions  if c.split(":")[1] != '']
            hypothesis_probabilities = [float(c.split(":")[1]) for c in hypothesis_suggestions  if c.split(":")[1] != '']

            word2fillers[key] = [premise_fillers, hypothesis_fillers]
            word2probabilities[key] = [premise_probabilities, hypothesis_probabilities]
            word2pos[key] = [pos, pos]
            positions[key]= [all_matching_keys_p, all_matching_keys_h]

                                                                          ######word2fillers outside loop defaultdict(<class 'list'>, {'church:NN:5:11:4:10': [[], ['building', 'house']]}),
                                                                          #########[word] = {
                                                                              # 'average_rank': sum(ranks)/n,
                                                                              # 'ranks' :ranks,
                                                                              # 'average_prob': f"{avg_prob:.2e}",
                                                                              # "individual_probs": [f"{p:.2e}" for p in probs1]}
        if word2fillers and skip_entry==False:
          words = {}
          for w in word2fillers:

              words[w] = ranked_overlap(word2fillers[w], word2probabilities[w], type_rank_operation).items()
              words[w] = sorted(words[w], key=lambda x: x[1][type_rank])      #get a certain type of ranking -: average rank or prob, and sort by that
          assigned_pos_tags = set()
          for w, ranked_fillers in words.items():
                                                                              #Ranked fillers for church:NN:5:11:4:10: [('house', {'average_rank': 0.5, 'ranks': [1], 'average_prob': '1.08e-01', 'individual_probs': ['1.08e-01']}), ('building', {'average_rank': 0.0, 'ranks': [0], 'average_prob': '1.52e-01', 'individual_probs': ['1.52e-01']})]
              parts = w.split(':')                                    # split the origin.word
              if len(parts) != 6:
                  print(f"Unexpected key format: {w}")
                  continue
              word_only, pos, premise_start, premise_end, hypothesis_start, hypothesis_end = parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
              positions_to_replace_premise = [':'.join(i.split(':')[:2]) for i in positions[w][0]] # Positions to replace (premise): ['5:11', '83:89']
              positions_to_replace_hypothesis = [':'.join(i.split(':')[:2]) for i in positions[w][1]] #Positions to replace (hypothesis): ['4:10']
              premise_suggestions_fillers=[]
              if id not in replacement_summary:
                replacement_summary[id] = {}                                  ##### {'2677109430.jpg#1r1n': {'church:NN': [['5:11', '83:89'], ['4:10']]}}

              key = f"{word_only}:{pos}"
              if key not in replacement_summary[id]:
                replacement_summary[id][key] = [positions_to_replace_premise, positions_to_replace_hypothesis]
              else:
                print('RED FLAG — duplicate key, skipping:', key)
              for key, value in word2fillers.items():                                    #Processing word2fillers entry: key church:NN:5:11:4:10 -> value [[], ['building', 'house']] key _> value
                parts = key.split(':')
                to_look= ':'.join(parts[:2])
                if to_look == word_only+':'+pos:
                  premise_suggestion_for_id_indexing, hypothesis_suggestion_for_id_indexing = value[0], value[1]

              expected_variants = 0
              if isinstance(rank_option, int):                                              # if there are not enough fillers to choose one from, we expect 0 variants created
                  expected_variants = 1 if rank_option < len(ranked_fillers) else 0
              elif isinstance(rank_option, slice):
                  expected_variants = len(range(*rank_option.indices(len(ranked_fillers))))  #but here we expacted 2, bc the keys in ranked_fillers are 2, if intersection would be done at this stage it should be 0
              sentence_variants = []
              if label in expected_generation:
                expected_generation[label] += expected_variants


              indices = [rank_option] if isinstance(rank_option, int) else range(*rank_option.indices(len(ranked_fillers))) #modifies rank/_option to be applicable to ranekd fillers

              for i in indices:
                  if i >= len(ranked_fillers):
                      continue

                  best_ = ranked_fillers[i][0].strip()

                  p_variant = premise_id
                  h_variant = hypothesis_id

                  def sort_positions(positions):
                   return sorted(positions, key=lambda s: int(s.split(':')[0]), reverse=True)

                  p_positions_sorted = sort_positions(positions_to_replace_premise)
                  h_positions_sorted = sort_positions(positions_to_replace_hypothesis)

                  for i in p_positions_sorted:
                      start, end = i.split(':')

                      p_variant = p_variant[:int(start)] + best_ + p_variant[int(end):]  #this iterates until positions are done


                  for i in h_positions_sorted:
                      start, end = i.split(":")

                      h_variant = h_variant[:int(start)] + best_ + h_variant[int(end):]
                  oiringacr = (
                    'h' if best_ not in premise_suggestion_for_id_indexing else
                    'p' if best_ not in hypothesis_suggestion_for_id_indexing else
                    'ph'
                    )

                  if p_positions_sorted and h_positions_sorted:
                      sentence_variants.append((p_variant, h_variant, best_, oiringacr)) #SENTENCE VARIANT STRUCTURE [('This building choir sings to the masses as they sing joyous songs from the book at a building.', 'The building has cracks in the ceiling.', 'building', 'h')]
                  else:
                      print('❌ Skipped: One of the position lists is empty.')

              assigned_pos_tags.update(word2pos[w])
              for idx, (p_variant, h_variant, best_, oiringacr) in enumerate(sentence_variants):
                numeric_label = None

                if label in {'neutral', 'entailment', 'contradiction'}:
                  numeric_label = label
                  actual_generation[label] += 1
                processed_entry = {
                    'id': f"{id}:{word_only}:{pos}:{premise_start}:{premise_end}:{hypothesis_start}:{hypothesis_end}:{best_}:{oiringacr}:{model_ids_dictionary.get(model_name)}",
                    'premise': p_variant,
                    'hypothesis': h_variant,
                    'label': numeric_label
                    } #processed_entry: {'id': '2677109430.jpg#1r1n:church:NN:5:11:4:10:building:h:bert', 'premise': 'This building choir sings to the masses as they sing joyous songs from the book at a building.', 'hypothesis': 'The building has cracks in the ceiling.', 'label': 'neutral'}

                processed_data.append(processed_entry)

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
    premise_count, hypothesis_count, word_count_freq_premise, word_count_freq_hypothesis = None, None, None, None
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
    if svae_cleaned_value:
      with open(name_file_cleaned_, 'w') as f_out:
          json.dump(quality_filtered_suggestions, f_out)
    return processed_data, seed_dataset, file_counts, premise_count, hypothesis_count, word_count_freq_premise, word_count_freq_hypothesis, replacement_summary


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

    pairs_models_datasets = []
    missing = []
    for model in models:
        key = f"{model}_{pos_tag}"
        # print('the key for the model and the pos tag', key)
        data = processed_results.get(key, [])
        # print('data', data)
        if data:
            # print('the model that will be printed as well', model)
            pairs_models_datasets.append((data, model))  # (dataset, source label = model)
        else:
            missing.append(key)

    if len(pairs_models_datasets) < 2:
        # print(len(pairs_models_datasets))
        raise ValueError(
            f"Need at least two models' suggestions  for pos_tag='{pos_tag}'. ")

    (dataset1, source1), (dataset2, source2), *others = pairs_models_datasets
    return merge_and_analyze_datasets(
        dataset1, source1,
        dataset2, source2,
        pos_tag,
        type_evaluation,
        min_count,
        name=name,
        opposite=opposite,
        others=others,
        label_for_shared_suggestions=label_for_shared_suggestions)

def generate_output_filenames(suggestion_file, models_dictionary, pos_dicitonary, number_inflation, type_dataset):
    #not double-checked
    """
    IN:
      /.../robert-base-cased.1.noun.200.test.json
    extract parts of the name and automatically generate the output file names required for the processed dataset

    out: output_processed_dataset, output_initial, output_all_inflated, output_all_sample, pos_to_mask
    """
    basename = os.path.basename(suggestion_file)
    parts = basename.split('.')

    if len(parts) < 6:
        raise ValueError("Filename does not follow expected naming convention.")

    model_tested, model_number, pos_full, size, split_str = parts[0], parts[1], parts[2], parts[3], parts[4][:2]
    model_name=models_dictionary.get(model_tested)
    pos_abbrev = pos_dicitonary.get(pos_full.lower(), pos_full.lower())

    output_processed_dataset = f"{model_name}.{model_number}.{pos_abbrev}.{size}.{split_str}{type_dataset}.inf.{number_inflation}.json"
    output_initial = f"{model_name}.{model_number}.{pos_abbrev}.{size}.{split_str}{type_dataset}.samp.{number_inflation}.json"
    output_all_inflated = f"{model_name}.{model_number}.all.{size}.{split_str}{type_dataset}.inf.{number_inflation}.json"
    output_all_sample = f"{model_name}.{model_number}.all.{size}.{split_str}{type_dataset}.samp.{number_inflation}.json"

    # Use the full pos tag as pos_to_mask if that's what you need.
    pos_to_mask = pos_full

    return output_processed_dataset, output_initial, output_all_inflated, output_all_sample, pos_to_mask

def scramble_word(word, rng=None, max_tries=10):
    """
    Return a scrambled version of `word` by shuffling its characters.
    Ensures it's different from the original when possible.
    """
    rng = rng or random.Random()
    if len(word) < 2:
        return word  # nothing to scramble

    chars = list(word)
    # try a few times to get a different permutation (if possible)
    for _ in range(max_tries):
        rng.shuffle(chars)
        mixed = ''.join(chars)
        if mixed != word:
            return mixed
    return ''.join(chars)  # fallback (might be same if all chars identical)

def scramble_per_item(item, rng=None):
    """Scrambles the replacement word in item['id'] and substitutes it in premise/hypothesis."""
    rng = rng or random.Random()

    id_parts = item["id"].split(":")
    word_to_change = id_parts[-3]

    mixed_word = scramble_word(word_to_change, rng=rng)

    pattern = re.compile(rf'\b{re.escape(word_to_change)}\b')

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


