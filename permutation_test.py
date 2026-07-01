from utils import *
def re_arrage_LLM_data_for_accuracy(file_name_predictions_, file_name_gold):
  '''{
  "seed": {
    "2938747424.jpg#3r1e": "entailment" -> {'id': '"2938747424.jpg#3r1e", 'gold_label': 'entailment', 'label_index': '...'}
    '''
  gold_data=load_data(file_name_gold)
  id_lookup_gold=build_id_lookup(gold_data, 'full')
  (flat_predictions,) = file_name_predictions_.values()
  re_arranged=[]
  for id, prediction in flat_predictions.items():
      re_arranged.append({'id': id, 'gold_label': id_lookup_gold[id]['label'], 'label_index': prediction})
  return re_arranged

def arrange_for_permutation_test(group_a_file, group_b_file, group_a_gold_file, group_b_gold_file, plot=False, LLM=False):
  '''load data
  re arrange LLM data to contain gold label and index
  convert values in 1/0 lists depening on the corectness
  calculate permutation'''
  group_a_data, group_b_data = load_two_files(group_a_file, group_b_file)
  if LLM:
    group_a_data, group_b_data = re_arrage_LLM_data_for_accuracy(group_a_data, group_a_gold_file), re_arrage_LLM_data_for_accuracy(group_b_data, group_b_gold_file)
  is_correct_a, is_correct_b = correctness_scores(group_a_data), correctness_scores(group_b_data)
  p_value_= permutation_test(is_correct_a, is_correct_b, plot=plot)
  return p_value_

def plot_pemrutation(permuted_diffs, observed_diff):

  '''claude generfated plot'''
  plt.hist(permuted_diffs, bins=40)
  plt.axvline(observed_diff, color='red', linestyle='--', label=f'observed diff = {observed_diff:.3f}')
  plt.xlabel('permuted accuracy diff')
  plt.ylabel('count')
  plt.legend()
  plt.show()

def permutation_test(group_a, group_b, num_permutations=10000, plot=False):
    '''pemutation test from https://www.geeksforgeeks.org/machine-learning/permutation-tests-in-machine-learning/ but with mean changed'''

    group_a, group_b = np.array(group_a), np.array(group_b)
    n_a = len(group_a)

    observed_diff = accuracy_calculation(group_a) - accuracy_calculation(group_b)

    combined = np.concatenate([group_a, group_b])
    permuted_diffs = np.empty(num_permutations)
    for i in range(num_permutations):
        np.random.shuffle(combined)
        permuted_diffs[i] = accuracy_calculation(combined[:n_a]) - accuracy_calculation(combined[n_a:])

    if plot:
      plot_pemrutation(permuted_diffs, observed_diff)

    return accuracy_calculation(np.abs(permuted_diffs) >= np.abs(observed_diff))

from itertools import product

def config_permutation_test(list_data_types_scr, list_data_types_n_scr, list_prompt_types):
  '''all types of combinations'''
  yield from product(list_data_types_scr, list_data_types_n_scr, list_prompt_types)
