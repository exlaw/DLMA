# Copyright 2024 Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
from data_utils.prompt_set import prompt_dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_log_likelihood(model, tokenizer, text, start_text):
    tokens = tokenizer(text, return_tensors='pt').to(device)
    start_tokens = tokenizer(start_text, return_tensors='pt').to(device)
    labels = tokens.input_ids.clone()
    labels[:, :start_tokens.input_ids.size(1)] = -100

    with torch.no_grad():
        logits = model(**tokens).logits.to(torch.float32)
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)
        labels[labels == -100] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1).item()


def get_log_likelihood_batch(model, tokenizer, texts, start_texts):
    batch_size = len(texts)
    encoded_texts = tokenizer(texts, padding=True, return_tensors='pt').to(device)
    encoded_start_texts = [tokenizer(st, return_tensors='pt').to(device) for st in start_texts]

    labels = encoded_texts.input_ids.clone()
    
    # Set the labels for start_texts to -100 for each text in the batch
    for i in range(batch_size):
        start_length = encoded_start_texts[i].input_ids.size(1)
        labels[i, :start_length] = tokenizer.pad_token_id
    
    with torch.no_grad():
        logits = model(**encoded_texts).logits.to(torch.float64)
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != tokenizer.pad_token_id)
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        total_result = (per_token_logps * loss_mask).sum(-1)
    total_result = total_result.tolist()
        
    return total_result


def get_two_likelihood_for_model(model, tokenzier, prompt_type, data_items):
    prompts = [prompt_dict[prompt_type].format(data_item['prompt']) for data_item in data_items]
    result_safe = [prompt + data_item['chosen'] for prompt, data_item in zip(prompts, data_items)]
    result_not_safe = [prompt + data_item['rejected'] for prompt, data_item in zip(prompts, data_items)]
    results1 = get_log_likelihood_batch(model, tokenzier, result_safe, prompts)
    results2 = get_log_likelihood_batch(model, tokenzier, result_not_safe, prompts)
    return results1, results2


def get_reward_with_different_prompt(model, tokenzier, prompt_type1, prompt_type2, data_items):
    likelihood1_safes, likelihood1_not_safes = get_two_likelihood_for_model(model, tokenzier, prompt_type1, data_items)
    likelihood2_safes, likelihood2_not_safes = get_two_likelihood_for_model(model, tokenzier, prompt_type2, data_items)
    results = []
    for likelihood1_safe, likelihood1_not_safe, likelihood2_safe, likelihood2_not_safe, data_item in zip(likelihood1_safes, likelihood1_not_safes, likelihood2_safes, likelihood2_not_safes, data_items):
        reward_safe = likelihood1_safe - likelihood2_safe
        reward_not_safe = likelihood1_not_safe - likelihood2_not_safe
        margin = reward_safe - reward_not_safe
        results.append({
            "prompt": data_item['prompt'],
            "chosen": data_item['chosen'],
            "rejected": data_item['rejected'],
            "reward_safe": reward_safe,
            "reward_not_safe": reward_not_safe,
            "margin": margin
        })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--split_number", type=int, default=4)
    parser.add_argument("--current_part", type=int, default=1)
    parser.add_argument("--prompt_type1", type=str, default="system_prompt_safe_hh2")
    parser.add_argument("--prompt_type2", type=str, default="system_prompt_not_safe_hh2")
    parser.add_argument("--model_path", type=str, default="/root/models/llama-sft-13b")
    parser.add_argument("--input_data_path", type=str, default="/mnt/task_wrapper/user_output/artifacts/generated_data/hh_all_data_iter1.jsonl")
    parser.add_argument("--output_data_path", type=str, default=None)

    base_dir = "/mnt/task_wrapper/user_output/artifacts/generated_data/"

    args = parser.parse_args()

    input_data_path = args.input_data_path

    with open(input_data_path, "r") as f:
        data = f.readlines()

    data = data[(args.current_part - 1) * len(data) // args.split_number: args.current_part * len(data) // args.split_number]

    if args.output_data_path is None:
        output_data_path = base_dir + "{}_with_reward_total_{}_part_{}_new1.jsonl".format(input_data_path.split("/")[-1].split(".")[0], args.split_number, args.current_part)
    else:
        output_data_path = args.output_data_path

    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    batch_size = args.batch_size
    with open(output_data_path, "w") as f:
        for i in tqdm(range(0, len(data), batch_size)):
            data_items = [json.loads(line) for line in data[i:i+batch_size]]
            result = get_reward_with_different_prompt(model, tokenizer, args.prompt_type1, args.prompt_type2, data_items)
            for item in result:
                f.write(json.dumps(item) + '\n')
    
    