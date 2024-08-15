# DLMA (Direct Large Language Model Alignment Through Self-Rewarding Contrastive Prompt Distillation)

[![License][]](https://opensource.org/licenses/Apache-2.0)[![Python][]](https://www.python.org/)[![PyTorch][]](https://pytorch.org/)

DLMA is a novel approach for aligning large language models through self-rewarding contrastive prompt distillation. This repository contains the source code and instructions for setting up the environment, performing supervised instruct tuning, generating preference data, and training DLMA models.

## Table of Contents

- [Installation](#installation)
- [Supervised Instruct Tuning](#supervised-instruct-tuning)
- [Preference Data Generation](#preference-data-generation)
- [Training DLMA Models](#training-dlma-models)
- [License](#license)

## Installation

To set up the environment, use the following command:

```bash
conda env create --file conda-recipe.yaml
```

## Supervised Instruct Tuning

We recommend using the [safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf) library for performing supervised fine-tuning (SFT). You can also choose to use other libraries such as [LLamaFactory](https://github.com/hiyouga/LLaMA-Factory).

1. Clone the `safe-rlhf` repository:   
```bash
git clone git@github.com:PKU-Alignment/safe-rlhf.git
cd safe-rlhf
```
2. Train the SFT model:   
```bash   
bash scripts/sft.sh \
--model_name_or_path /root/models/meta-llama/Llama-2-7b-hf \
--output_dir output/llama2-sft  
 ```

3. Split and resave the model:  
```bash   
cd ..   
python split_and_resave_model.py \   
--input_model_path safe-rlhf/output/llama2-sft \  
--output_model_path models/sft   
```

## Preference Data Generation

Generate preference data using the following scripts:

```bash
prompt_type_pos="harmless_positive_prompt"
prompt_type_neg="harmless_negative_prompt"
model_output_path="models/sft"
data_path="PKU-Alignment/PKU-SafeRLHF"
gpu_number=8
batch_size=4

bash scripts/generate_data_scripts.sh \
$prompt_type_pos \
$prompt_type_neg \
$model_output_path \
$data_path \
$gpu_number \
$batch_size
```

Combine different parts:

```python
python merge_files.py \
--data_name "$data_path" \
--model_name "$model_output_path" \
--batch_size "$batch_size" \
--prompt_type_pos "$prompt_type_pos" \
--prompt_type_neg "$prompt_type_neg" \
--split_number "$gpu_number" \
--output_dir "generated_data/llama2-pku-safety"
```

## Training DLMA Models

1. Train the DLMA model:   
```bash   
bash scripts/run_dlma.sh \   
-model llama7b \   
-model_path models/sft \   
-datasets \[llama2-pku-safety\] \   
-exp_name dpo_llama2_weight_margin_02_40 \   
-margin_loss_weight 0.2 \   
-min_range -40 \   
-max_range 40    
```

2. Split and resave the model:   

```python   
python split_and_resave_model.py \   
--base_model_path safe-rlhf/output/llama2-sft \   
--input_model_path /root/DLMA/.cache/root/dpo_llama2_weight_margin_02_40/LATEST/policy.pt \ 
--output_model_path models/dlma 
```

The trained DLMA model will be available in the `models/dlma` directory.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Citation

```
@inproceedings{liu-etal-2024-direct,
    title = "Direct Large Language Model Alignment Through Self-Rewarding Contrastive Prompt Distillation",
    author = "Liu, Aiwei  and
      Bai, Haoping  and
      Lu, Zhiyun  and
      Kong, Xiang  and
      Wang, Xiaoming  and
      Shan, Jiulong  and
      Cao, Meng  and
      Wen, Lijie",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.523",
    pages = "9688--9712",
}
```