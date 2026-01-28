from transformers import pipeline
import torch
def classify_nli_batch(batch, model, tokenizer, prompts_2, max_new_tokens=20):
    prompts = []

    for e in batch:
        messages = [
            {
                "role": "user",
                "content": f"""You will be given a premise and a hypothesis about that premise.
You need to decide whether the hypothesis is entailed by the premise by choosing one of the following answers: ’e’: The hypothesis follows logically from the premise. ’c’: The hypothesis does not logically follow from the premise.
’n’: It is not possible to determine whether the hypothesis is loggically followed by the premise.
Read the premise and hypothesis and select the correct answer from the three
answer labels (e, n, c). Also provide a single
line of explanation in a new line. End your response with [END] and output nothing after.

{prompts_2}

Premise: "{e['premise']}"
Hypothesis: "{e['hypothesis']}"
Answer:"
"""
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    decoded = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    labels = []
    for text in decoded:
        label = text.strip().split()[0].lower().strip('.,!?')
        labels.append(label)

    return labels

def run_nli_inference(
    input_path,
    output_path,
    type_test,
    pipe,
    prompts,
    model_name='llama',
    dataset_name='name',
    batch_size=8,
    hf_model="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=20
):

    BATCH_SIZE = batch_size
    model_name = model_name
    dataset_name = dataset_name

    with open(input_path, 'r') as f:
        filtered_datasets_sample = json.load(f)


    filtered_datasets_sample1 = {}
    if type_test in ['samp', 'test']:
      filtered_datasets_sample1[type_test] = filtered_datasets_sample
    else:
      filtered_datasets_sample1= filtered_datasets_sample

    results, filtered = {}, defaultdict(list)

    IDS_TO_KEEP = {
        '112010c:alternative:JJ:39:50:60:71:precarious:ph:albert'
    }

    for dataset, items in filtered_datasets_sample1.items():
        for example in items:
            if example["id"] in IDS_TO_KEEP:
                filtered[dataset].append(example)
    for dataset, items in filtered_datasets_sample1.items():
        print('for dataset', dataset)
        results[dataset] = {}
        for i in tqdm(range(0, len(items), BATCH_SIZE)):
            batch = items[i:i+BATCH_SIZE]

            preds = classify_nli_batch(batch, pipe.model, pipe.tokenizer, prompts)

            for e, p in zip(batch, preds):
                results[dataset][e["id"]] = p

    with open(output_path, 'w') as f:
        json.dump(results, f)
    return results
