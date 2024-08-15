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

#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu_id) gpu_id="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --prompt_type_pos) prompt_type_pos="$2"; shift ;;
        --prompt_type_neg) prompt_type_neg="$2"; shift ;;
        --model_path) model_path="$2"; shift ;;
        --data_path) data_path="$2"; shift ;;
        --split_number) split_number="$2"; shift ;;
        --part) part="$2"; shift ;;
        *) echo "Unkonwn: $1"; exit 1 ;;
    esac
    shift
done

data_base_dir=generated_data
model_name=$(basename $model_path)  
data_name=$(basename $data_path)  

output_dir="${data_base_dir}/all_data_${data_name}_output_model_${model_name}_batchsize_${batch_size}_prompt_pos_${prompt_type_pos}_prompt_neg_${prompt_type_neg}/raw"
output_file="${output_dir}/split_${split_number}_part_${part}.jsonl"

output_dir_with_reward="${data_base_dir}/all_data_${data_name}_output_model_${model_name}_batchsize_${batch_size}_prompt_pos_${prompt_type_pos}_prompt_neg_${prompt_type_neg}/raw_with_reward"
output_file_with_reward="${output_dir_with_reward}/split_${split_number}_part_${part}.jsonl"

mkdir -p "${output_dir}"
mkdir -p "${output_dir_with_reward}"

cmd="CUDA_VISIBLE_DEVICES=$gpu_id python efficient_generation.py \
    --batch_size $batch_size \
    --prompt_type_pos $prompt_type_pos \
    --prompt_type_neg $prompt_type_neg \
    --model_path $model_path \
    --data_path $data_path \
    --split_number $split_number \
    --current_part $part \
    --output_path $output_file"

echo "Command: $cmd"

eval $cmd

cmd="CUDA_VISIBLE_DEVICES=$gpu_id python calculate_margin_and_filter.py \
    --batch_size $batch_size \
    --split_number 1 \
    --current_part 1 \
    --input_data_path $output_file \
    --prompt_type1 harmless_positive_prompt \
    --prompt_type2 harmless_negative_prompt \
    --model_path $model_path \
    --output_data_path $output_file_with_reward"

echo "Command: $cmd"

eval $cmd
