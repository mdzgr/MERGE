
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


