#FIXME no explicit punctuaion definition but use the sent and tokens to get chr offsets


#do not replace negation in adverb
def extract_nouns(pos_tags, tokens, source, pos_type):
    noun_tags = {'NN', 'NNS'}
    verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    tags = [
        ".", ",", "?", "'", ":", ";", "-", "–", "—", "'", "\"", "(", ")", "[", "]", "{", "}",
        "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "_", "~", "`", "<", ">", "="
    ]

    special_tok = ["'s", "'t", "n't"]
    dictionary_positions = {}
    current_position = 0

    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):

        if pos_type == 'noun' and pos in noun_tags:
            # print('hey')
            start = current_position
            end = current_position + len(token)
            offset = (start, end)

            if token not in dictionary_positions:
                dictionary_positions[token] = {'positions': [offset], 'pos': pos, 'source': source}
            else:
                dictionary_positions[token]['positions'].append(offset)

        elif pos_type == 'verb' and (pos in noun_tags or pos in verb_tags):
            start = current_position
            end = current_position + len(token)
            offset = (start, end)

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
