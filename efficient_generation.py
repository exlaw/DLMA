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

import argparse
import json
import os
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils.prompt_set import prompt_dict

def generate_data_with_prompt_type(batch: List[str], prompt_type: str, tokenizer, model, device):
    batch = [prompt_dict[prompt_type].format(item) for item in batch]
    inputs = tokenizer(batch, padding=True, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts, batch

def process_dataset(dataset) -> List[str]:
    prompts = []
    for item in dataset:
        context = item['context']
        prompt = context[0]['text']
        for index in range(1, len(context)):
            if index % 2 == 1:
                prompt += "\nASSISTANT: " + context[index]['text']
            else:
                prompt += "\nUSER: " + context[index]['text']
        prompts.append(prompt)
    return prompts

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    tokenizer.padding_side = "left"

    result = []

    if args.data_path == "PKU-Alignment/PKU-SafeRLHF":
        dataset = load_dataset(args.data_path)
        data = dataset['train']['prompt']
    elif args.data_path == "hh-helpful":
        dataset = load_dataset("PKU-Alignment/processed-hh-rlhf", data_dir="helpful-base")
        data = process_dataset(dataset['train'])
    elif args.data_path == "hh-harmless":
        dataset = load_dataset("PKU-Alignment/processed-hh-rlhf", data_dir="harmless-base")
        data = process_dataset(dataset['train'])

    filtered_data = [item for item in data if len(item.split()) <= args.max_data_length]
    data = filtered_data[len(data) // args.split_number * (args.current_part - 1): len(data) // args.split_number * args.current_part]

    max_number = args.max_number if args.max_number != -1 else len(data)
    for i in tqdm(range(0, max_number, args.batch_size)):
        batch = data[i:i + args.batch_size]
        pos_generated_texts, pos_input = generate_data_with_prompt_type(batch, args.prompt_type_pos, tokenizer, model, device)
        neg_generated_texts, neg_input = generate_data_with_prompt_type(batch, args.prompt_type_neg, tokenizer, model, device)
        for j in range(len(batch)):
            result.append({
                "prompt": batch[j],
                "pos_prompt_type": args.prompt_type_pos,
                "neg_prompt_type": args.prompt_type_neg,
                "chosen": pos_generated_texts[j].split(pos_input[j])[-1].strip(),
                "rejected": neg_generated_texts[j].split(neg_input[j])[-1].strip()
            })

    output_path = args.output_path or f"/mnt/task_wrapper/user_output/artifacts/generated_data/all_data_{args.data_path.split('/')[-1]}_output_model_{args.model_path.split('/')[-1]}_batchsize_{args.batch_size}_prompt_pos_{args.prompt_type_pos}_prompt_neg_{args.prompt_type_neg}_split_{args.split_number}_part_{args.current_part}_new.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in result:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--prompt_type_pos", type=str, default="system_prompt_safe")
    parser.add_argument("--prompt_type_neg", type=str, default="system_prompt_safe")
    parser.add_argument("--model_path", type=str, default="/root/models/llama-7b-sft-new")
    parser.add_argument("--data_path", type=str, default="PKU-Alignment/PKU-SafeRLHF")
    parser.add_argument("--split_number", type=int, default=4)
    parser.add_argument("--current_part", type=int, default=1)
    parser.add_argument("--max_data_length", type=int, default=600)
    parser.add_argument("--max_number", type=int, default=-1)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    main(args)