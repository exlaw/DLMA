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


prompt_type_pos="${1:-harmless_positive_prompt}"
prompt_type_neg="${2:-harmless_negative_prompt}"
model_path="${3:-/root/models/Mistral-7B-sft}"
data_path="${4:-hh-harmless}"
split_number="${5:-8}"
batch_size="${6:-4}"

mkdir -p "logs"

for ((gpu_id=0; gpu_id < split_number; gpu_id++)); do
  part=$((gpu_id + 1))

  cmd="nohup bash scripts/single_pipline.sh \
    --gpu_id $gpu_id \
    --batch_size $batch_size \
    --prompt_type_pos \"$prompt_type_pos\"\
    --prompt_type_neg \"$prompt_type_neg\"\
    --model_path \"$model_path\" \
    --data_path \"$data_path\" \
    --split_number $split_number \
    --part $part \
    > logs/all_output$part.log 2>&1 &"

  echo "Executing command: $cmd"

  eval $cmd
done
