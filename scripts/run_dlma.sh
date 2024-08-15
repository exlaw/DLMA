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


default_model="llama7b"
default_model_path="/root/models/llama-7b-sft-new"
default_datasets="[llama2-pku-safety]"
default_exp_name="dpo_llama2_weight_margin_02_40"
default_margin_loss_weight=0.2
default_min_range=-40.0
default_max_range=40.0

while getopts model:model_path:datasets:exp_name:margin_loss_weight:min_range:max_range: flag
do
    case "${flag}" in
        model) model=${OPTARG};;                    
        model_path) model_path=${OPTARG};;          
        datasets) datasets=${OPTARG};;              
        exp_name) exp_name=${OPTARG};;              
        margin_loss_weight) margin_loss_weight=${OPTARG};;       
        min_range) min_range=${OPTARG};;            
        max_range) max_range=${OPTARG};;          
    esac
done


model=${model:-$default_model}
model_path=${model_path:-$default_model_path}
datasets=${datasets:-$default_datasets}
exp_name=${exp_name:-$default_exp_name}
margin_loss_weight=${margin_loss_weight:-$default_margin_loss_weight}
min_range=${min_range:-$default_min_range}
max_range=${max_range:-$default_max_range}


python -u train.py \
    model=$model \
    model.name_or_path=$model_path \
    datasets=$datasets \
    loss=dpo \
    loss.beta=0.1 \
    exp_name=$exp_name \
    gradient_accumulation_steps=2 \
    batch_size=64 \
    eval_batch_size=32 \
    trainer=FSDPTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16 \
    loss.use_weighted_loss=true \
    loss.margin_loss_weight=$margin_loss_weight \
    loss.min_range=$min_range \
    loss.max_range=$max_range