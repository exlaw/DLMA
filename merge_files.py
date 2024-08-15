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
import os
import argparse
import random

def ensure_directory_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def merge_jsonl_files(data_name, model_name, batch_size, prompt_type_pos, prompt_type_neg, split_number, output_dir):
    data_name = os.path.basename(data_name)
    model_name = os.path.basename(model_name)

    input_dir = f"generated_data/all_data_{data_name}_output_model_{model_name}_batchsize_{batch_size}_prompt_pos_{prompt_type_pos}_prompt_neg_{prompt_type_neg}/raw_with_reward"
    
    output_train = os.path.join(output_dir, "train.jsonl")
    output_test = os.path.join(output_dir, "test.jsonl")

    ensure_directory_exists(output_dir)
    
    entries = []
    for part in range(1, split_number + 1):
        input_file_path = os.path.join(input_dir, f"split_{split_number}_part_{part}.jsonl")
        if os.path.exists(input_file_path):
            with open(input_file_path, 'r') as infile:
                for line in infile:
                    entries.append(line)
        else:
            print(f"Warning: The file {input_file_path} does not exist.")

    random.shuffle(entries)
    num_test_samples = len(entries) // 20 

    with open(output_train, 'w') as train_file, open(output_test, 'w') as test_file:
        for i, entry in enumerate(entries):
            if i < num_test_samples:
                test_file.write(entry)
            else:
                train_file.write(entry)


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL files from a multi-part processing.")
    parser.add_argument("--data_name", type=str, required=True, help="Data name to specify in paths.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to specify in paths.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size used in processing.")
    parser.add_argument("--prompt_type_pos", type=str, required=True, help="Positive prompt type.")
    parser.add_argument("--prompt_type_neg", type=str, required=True, help="Negative prompt type.")
    parser.add_argument("--split_number", type=int, required=True, help="Number of parts to merge.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output data storage.")
    
    args = parser.parse_args()

    # Call the function with arguments parsed from command line
    merge_jsonl_files(
        args.data_name,
        args.model_name,
        args.batch_size,
        args.prompt_type_pos,
        args.prompt_type_neg,
        args.split_number,
        args.output_dir
    )

if __name__ == "__main__":
    main()