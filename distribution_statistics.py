import json, glob, os, re
from collections import defaultdict
def parse_id(id_str):
    after_hash = id_str.split("#", 1)[1]
    parts = after_hash.split(":")
    runner_word = parts[1]
    pos = parts[2]
    prem_pos = f"{parts[3]}:{parts[4]}"
    hyp_pos  = f"{parts[5]}:{parts[6]}"
    target_word = parts[7]
    return runner_word, pos, prem_pos, hyp_pos, target_word

def word_replace(sentence, old_word, new_word):
    pattern = r'(?<!\w)' + re.escape(old_word) + r'(?!\w)'
    return re.sub(pattern, new_word, sentence)

def model_from_path(path):
    base = os.path.basename(path)
    prefix = "conffusion_matrix_cleaned_suggesrions_all_"
    if base.startswith(prefix) and base.endswith(".json"):
        return base[len(prefix):-5]
    return base

def check_sentence(sent, needed_pos):
    entry = data.get(sent)
    runner_key = f"{runner_word}:{pos}"
    slot = entry.get(runner_key)
    position_prefix = f"{needed_pos}:"
    for pos_key, suggestions in slot.items():
        if pos_key.startswith(position_prefix):
            for s in suggestions:
                first = s.split(":")[0]
                if first == target_word:
                    return True
    return False



def summarize_models_to_file(score_groups, results, label, file_handle):
    for total, id_list in sorted(score_groups.items()):
        counts = Counter()
        for iid in id_list:
            for m in results.get(iid, []):
                counts[m] += 1

        if counts:
            line = f"Score ({label}) {total}: " + "  ".join(
                f"{m}({n})" for m, n in counts.most_common()
            )
        else:
            line = f"Score ({label}) {total}: (no hits)"

        file_handle.write(line + "\n")
