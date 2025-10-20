from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_pattern_accuracy(dictionary_results, model, tags, color_map_r, style_map_leg,
                         color_dict, legend_dict, name_plot, title_replacement,
                         color_legend_title, style_legend_title, style_legened, style_type,
                         min_threshold=0):
    """

    """
    plt.figure(figsize=(20, 16))

    color_map, legend_names = color_dict, legend_dict
    plotted_colors, plotted_styles = set(), set()
    for model_key, results in dictionary_results.items():

        if model and model_key not in model:
            continue

        for i in results:
            name_file = i["input_file"]

            if color_map_r == 'model':
                to_look_for = model_key
            elif color_map_r == 'pos_tag':
                matching_tags = [tag for tag in tags if tag in name_file.lower()]
                # print(matching_tags)
                if not matching_tags:
                    continue
                to_look_for = matching_tags[0]
            else:
                to_look_for = model_key

            nested_accuracies = i["pattern_accuracy"]
            thresholds = sorted([float(k) for k in nested_accuracies.keys() if float(k) >= min_threshold])

            if not thresholds:
                continue

            scaled_thresholds = [int(round(t * 100)) for t in thresholds]
            accuracies = [nested_accuracies[str(t)] for t in thresholds]


            color = color_map.get(to_look_for, 'black')
            to_look_for_style = name_file if style_type == 'per_gen_model' else to_look_for
            style = style_map_leg.get(to_look_for_style, '-') if style_map_leg is not None else '-'
            plt.plot(scaled_thresholds, accuracies, linestyle=style, color=color, linewidth=4.5)
            plotted_colors.add((color, to_look_for))
            if style_map_leg:
                plotted_styles.add((style, to_look_for_style))

    color_legend, style_legend = [], []
    for color, label in plotted_colors:
        display_label = legend_names.get(label, label)
        color_legend.append(Line2D([0], [0], color=color, lw=2, label=display_label))

    if style_map_leg!=None:
      for style, model_key in plotted_styles:
          # if style_type=='per_nvn_model':
          #   model_key=to_look_for
          display_label = style_legened.get(model_key, model_key)
          style_legend.append(Line2D([0], [0], color='black', linestyle=style, lw=2, label=display_label))

    all_handles = color_legend + style_legend

    header_color = Line2D([0], [0], color='none', label=color_legend_title)
    header_style = Line2D([0], [0], color='none', label=style_legend_title)
    combined_handles = [header_color] + color_legend + [header_style] + style_legend

    plt.legend(handles=combined_handles, loc='lower left', fontsize=25)
    plt.xlabel("Accuracy threshold (%)", fontsize=40, fontname='Liberation Serif')
    plt.ylabel("Pattern accuracy", fontsize=40, fontname='Liberation Serif')
    plt.title(f"Pattern Accuracy {title_replacement}.", fontsize=45, fontname='Liberation Serif')
    plt.tick_params(axis='both', labelsize=30)
    thresholds_int = int(min_threshold * 100)
    print(thresholds_int)
    plt.xlim(left=thresholds_int, right=100)
    plt.grid(True, which='major', color='pink', linestyle='--', linewidth=2, alpha=0.7)
    plt.savefig(name_plot, dpi=300, bbox_inches='tight')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    plt.show()


def load_data_matirx(file_paths, pos_tag):
    """Load data from multiple model files"""
    models_data = {}
    for file_path in file_paths:
        model_name = file_path.split('_')[-1].split('.')[0]
        with open(file_path, 'r') as f:
            models_data[model_name] = json.load(f)
    return models_data

def load_seed_dataset(seed_file_path):
    """Load the seed dataset with premise-hypothesis pairs"""
    with open(seed_file_path, 'r') as f:
        return json.load(f)

def get_tokens_for_position(sentence_data, token_pos):
    """Get all suggestion tokens for a specific token position"""
    tokens = set()
    if token_pos in sentence_data:
        for score_key in sentence_data[token_pos]:
            for suggestion in sentence_data[token_pos][score_key]:
                token = suggestion.split(':')[0]
                tokens.add(token)
    return tokens

def calculate_token_level_overlap(data1, data2):
    """Calculate average overlap per token position between two sentence data structures"""
    common_positions = set(data1.keys()) & set(data2.keys())
    if not common_positions:
        return 0.0

    overlaps = []
    for pos in common_positions:
        tokens1 = get_tokens_for_position(data1, pos)
        tokens2 = get_tokens_for_position(data2, pos) #averages the overlaps of all token positions for premise and hypotehsis
        overlaps.append(len(tokens1 & tokens2))

    return np.mean(overlaps)

def calculate_self_overlap(sentence_data):
    """Calculate average number of suggestions per token position"""
    if not sentence_data:
        return 0.0

    counts = []
    for pos in sentence_data:
        tokens = get_tokens_for_position(sentence_data, pos)
        counts.append(len(tokens))

    return np.mean(counts)

def calculate_premise_hypothesis_overlap_token_level(models_data, seed_data):
    """Calculate token-level premise-hypothesis overlap for each model"""
    model_names = list(models_data.keys())
    results = {}

    for model in model_names:
        problem_overlaps = []

        for problem in seed_data:
            premise, hypothesis = problem['premise'], problem['hypothesis']

            if premise in models_data[model] and hypothesis in models_data[model]:
                overlap = calculate_token_level_overlap( #mean of average of token positions across premise and hypothesos
                    models_data[model][premise],
                    models_data[model][hypothesis]
                )
                problem_overlaps.append(overlap)

        results[model] = np.mean(problem_overlaps) if problem_overlaps else 0.0

    return results
def get_common_tokens_between_sentences(data1, data2):
    """Get tokens that appear in suggestions for both sentences"""
    all_positions = set(data1.keys()) | set(data2.keys())
    common_tokens = set() #and updated

    for pos in all_positions:
        tokens1 = get_tokens_for_position(data1, pos) #for x postions in P
        tokens2 = get_tokens_for_position(data2, pos) #and H
        common_tokens.update(tokens1 & tokens2) #intresection number

    return common_tokens

def get_position_wise_common_tokens(data1, data2):
    """Get common tokens per position between two sentences, return dict of position -> common_tokens"""
    all_positions = set(data1.keys()) | set(data2.keys())
    position_common = {}

    for pos in all_positions:
        tokens1 = get_tokens_for_position(data1, pos)
        tokens2 = get_tokens_for_position(data2, pos)
        common_tokens = tokens1 & tokens2
        position_common[pos] = common_tokens

    return position_common
def calculate_cross_model_overlaps(models_data, seed_data, verbose):
    """Calculate cross-model premise-hypothesis overlaps"""
    model_names = list(models_data.keys())
    n_models = len(model_names)
    overlap_matrix = np.zeros((n_models, n_models))

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                problem_unique_counts = []

                for problem in seed_data:
                    premise, hypothesis = problem['premise'], problem['hypothesis']

                    if all(s in models_data[m] for m in [model1, model2] for s in [premise, hypothesis]):
                        common1 = get_position_wise_common_tokens(models_data[model1][premise], models_data[model1][hypothesis])

                        # Get all common tokens from other models
                        other_models_tokens = set()
                        for other_model in model_names:
                            if other_model != model1:
                                common_other = get_position_wise_common_tokens(models_data[other_model][premise], models_data[other_model][hypothesis])
                                for pos_tokens in common_other.values():
                                    other_models_tokens.update(pos_tokens)

                        # Get unique tokens from current model
                        model_tokens = set()
                        for pos_tokens in common1.values():
                            model_tokens.update(pos_tokens)
                        if verbose:
                          print('Length other tokens',len(other_models_tokens))
                          print('The other tokens linked to lenigh',other_models_tokens)
                          print('the number of tokens in this model',len(model_tokens))
                        unique_tokens = model_tokens - other_models_tokens
                        if verbose:
                          intersection = model_tokens & other_models_tokens
                          print('the intersection', len(intersection))
                          print('the number of unique tokens', len(unique_tokens))
                        problem_unique_counts.append(len(unique_tokens))

                overlap_matrix[i][j] = np.mean(problem_unique_counts) if problem_unique_counts else 0.0
            else:
                problem_overlaps = []

                for problem in seed_data:
                    premise, hypothesis = problem['premise'], problem['hypothesis']

                    if all(s in models_data[m] for m in [model1, model2] for s in [premise, hypothesis]):
                        common1 = get_position_wise_common_tokens(models_data[model1][premise], models_data[model1][hypothesis])
                        common2 = get_position_wise_common_tokens(models_data[model2][premise], models_data[model2][hypothesis])

                        all_positions = set(common1.keys()) | set(common2.keys())
                        position_intersections = []

                        for pos in all_positions:
                            common_tokens_m1 = common1.get(pos, set())
                            common_tokens_m2 = common2.get(pos, set())
                            intersection = len(common_tokens_m1 & common_tokens_m2)
                            position_intersections.append(intersection)

                        avg_intersection = np.mean(position_intersections) if position_intersections else 0.0
                        problem_overlaps.append(avg_intersection)

                overlap_matrix[i][j] = np.mean(problem_overlaps) if problem_overlaps else 0.0

    return overlap_matrix




def analyze_model_overlap_enhanced(file_paths, seed_file_path, pos_tag_name, name_map, verbose):
    """Enhanced token-level analysis"""
    models_data = load_data_matirx(file_paths, pos_tag_name)
    seed_data = load_seed_dataset(seed_file_path)
    model_names = list(models_data.keys())

    print(f"Loaded models: {model_names}")
    print(f"Loaded {len(seed_data)} problems from seed dataset")

    common_sentences = set.intersection(*[set(models_data[m].keys()) for m in model_names])
    print(f"Common sentences across all models: {len(common_sentences)}")

    n_models = len(model_names)
    sentence_overlap_matrix = np.zeros((n_models, n_models))

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                averages = [calculate_self_overlap(models_data[model1][s]) for s in common_sentences]
                sentence_overlap_matrix[i][j] = np.mean(averages) if averages else 0.0
            else:
                overlaps = [calculate_token_level_overlap(models_data[model1][s], models_data[model2][s])
                           for s in common_sentences]
                sentence_overlap_matrix[i][j] = np.mean(overlaps) if overlaps else 0.0

    ph_cross_matrix = calculate_cross_model_overlaps(models_data, seed_data, verbose)
    display_names = [name_map.get(m, m) for m in model_names]
    # First matrix (Sentence)
    fig1, ax1 = plt.subplots(figsize=(10, 10))  # much bigger figure
    disp1 = ConfusionMatrixDisplay(confusion_matrix=sentence_overlap_matrix,
                                  display_labels=display_names)
    im1 = disp1.plot(ax=ax1, cmap='Blues', values_format='.1f', colorbar=True)

    # Make ticks and labels bigger
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_xlabel(ax1.get_xlabel(), fontsize=18)
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=18)

    ax1.set_title(
        'Cross-Model Shared Suggestions Per Sentence',
        fontsize=22, fontweight='bold', pad=30
    )
    fig1.savefig(f'confusion_matrix_{pos_tag_name}_per_sentence.pdf',
                dpi=300, bbox_inches='tight')

    # Second matrix (Premise-Hypothesis)
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    disp2 = ConfusionMatrixDisplay(confusion_matrix=ph_cross_matrix,
                                  display_labels=display_names)
    im2 = disp2.plot(ax=ax2, cmap='Greens', values_format='.1f', colorbar=False)

    # Manual colorbar with bigger ticks
    cbar = fig2.colorbar(im2.im_, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)

    # Make ticks and labels bigger
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_xlabel("", fontsize=22)
    ax2.set_ylabel("", fontsize=22)

    ax2.set_title(
        'Cross-Model Intersection of ALL Suggestions \n for Premise-Hypothesis Per Problem',
        fontsize=22, fontweight='bold', pad=30
    )
    fig2.savefig(f'confusion_matrix_{pos_tag_name}_per_problem.pdf',
                dpi=300, bbox_inches='tight')

    return sentence_overlap_matrix, model_names, ph_cross_matrix
# def calculate_cross_model_overlaps(models_data, seed_data):
#     """Calculate cross-model premise-hypothesis overlaps"""
#     model_names = list(models_data.keys())
#     n_models = len(model_names)
#     overlap_matrix = np.zeros((n_models, n_models))

#     for i, model1 in enumerate(model_names):
#         for j, model2 in enumerate(model_names):
#             if i == j:
#                 overlap_matrix[i][j] = calculate_premise_hypothesis_overlap_token_level(models_data, seed_data)[model1]
#             else:
#                 problem_overlaps = []

#                 for problem in seed_data:
#                     premise, hypothesis = problem['premise'], problem['hypothesis']

#                     if all(s in models_data[m] for m in [model1, model2] for s in [premise, hypothesis]):
#                         common1 = get_position_wise_common_tokens(models_data[model1][premise], models_data[model1][hypothesis]) #get common suggestions for each position bt P and H
#                         common2 = get_position_wise_common_tokens(models_data[model2][premise], models_data[model2][hypothesis]) #for two models

#                         all_positions = set(common1.keys()) | set(common2.keys()) #get poitions common
#                         position_intersections = []

#                         for pos in all_positions:
#                             common_tokens_m1 = common1.get(pos, set()) #get common tokens PH for this position from first model
#                             common_tokens_m2 = common2.get(pos, set()) #same for the second
#                             intersection = len(common_tokens_m1 & common_tokens_m2)#intersection
#                             position_intersections.append(intersection) #append number


#                         avg_intersection = np.mean(position_intersections) if position_intersections else 0.0 #average all the numbers
#                         problem_overlaps.append(avg_intersection)

#                 overlap_matrix[i][j] = np.mean(problem_overlaps) if problem_overlaps else 0.0

#     return overlap_matrix



# def analyze_model_overlap_enhanced(file_paths, seed_file_path, pos_tag_name, name_map):
#     """Enhanced token-level analysis"""
#     models_data = load_data_matirx(file_paths, pos_tag_name)
#     seed_data = load_seed_dataset(seed_file_path)
#     model_names = list(models_data.keys())

#     print(f"Loaded models: {model_names}")
#     print(f"Loaded {len(seed_data)} problems from seed dataset")

#     common_sentences = set.intersection(*[set(models_data[m].keys()) for m in model_names])
#     print(f"Common sentences across all models: {len(common_sentences)}")

#     n_models = len(model_names)
#     sentence_overlap_matrix = np.zeros((n_models, n_models))

#     for i, model1 in enumerate(model_names):
#         for j, model2 in enumerate(model_names):
#             if i == j:
#                 averages = [calculate_self_overlap(models_data[model1][s]) for s in common_sentences]
#                 sentence_overlap_matrix[i][j] = np.mean(averages) if averages else 0.0
#             else:
#                 overlaps = [calculate_token_level_overlap(models_data[model1][s], models_data[model2][s])
#                            for s in common_sentences]
#                 sentence_overlap_matrix[i][j] = np.mean(overlaps) if overlaps else 0.0

#     ph_cross_matrix = calculate_cross_model_overlaps(models_data, seed_data)

#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))
#     display_names = [name_map.get(m, m) for m in model_names]

#     disp1 = ConfusionMatrixDisplay(confusion_matrix=sentence_overlap_matrix, display_labels=display_names)
#     disp1.plot(ax=ax1, cmap='Blues', values_format='.1f')
#     ax1.set_title(f'Cross-Model Shared Suggestions for {pos_tag_name}s Across Models Per Sentence')

#     disp2 = ConfusionMatrixDisplay(confusion_matrix=ph_cross_matrix, display_labels=display_names)
#     disp2.plot(ax=ax2, cmap='Greens', values_format='.1f')
#     ax2.set_title(f'Cross-Model Premise-Hypothesis Common Suggestions for {pos_tag_name}s Per Problem')
#     plt.savefig(f'confusion_matrix_{pos_tag_name}.pdf', dpi=300, bbox_inches='tight')
#     plt.show()

#     return sentence_overlap_matrix, model_names, ph_cross_matrix

