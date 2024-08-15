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

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import argparse


def split_model(args):
    model_path = args.input_model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path)

    new_path = args.output_model_path

    if not os.path.exists(new_path):
        model.save_pretrained(new_path, max_shard_size="10GB")
        tokenizer.save_pretrained(new_path)

def split_model_from_pt(args):
    model_path = args.base_model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path)

    model.load_state_dict(torch.load(args.input_model_path)['state'])

    new_path = args.output_model_path

    if not os.path.exists(new_path):
        model.save_pretrained(new_path, max_shard_size="10GB")
        tokenizer.save_pretrained(new_path)

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--input_model_path", type=str, default="")
    parser.add_argument("--output_model_path", type=str, default="")

    args = parser.parse_args()
    if args.base_model_path is not None:
        split_model_from_pt(args)
    else:
        split_model(args)



