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

import contextlib
from torch import nn

def init_new_embeddings(
        embeddings: nn.Embedding | nn.Linear | None,
        new_num_embeddings: int,
        num_new_embeddings: int,
    ) -> None:
    if embeddings is None:
        return

    params = [embeddings.weight, getattr(embeddings, 'bias', None)]
    context = contextlib.nullcontext()
    with context:
        for param in params:
            if param is None:
                continue
            assert param.size(0) == new_num_embeddings
            param_data = param.data
            param_mean = param_data[:-num_new_embeddings].mean(dim=0, keepdim=True)
            param_data[-num_new_embeddings:] = param_mean

def add_special_tokens_for_model(tokenizer, model):
    if tokenizer.pad_token is None:
        special_tokens_dict = {'pad_token': '<pad>'}
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        new_num_embeddings = len(tokenizer)
        model.config.pad_token_id = tokenizer.pad_token_id
        if num_new_tokens > 0:
            model.resize_token_embeddings(new_num_embeddings)
            init_new_embeddings(
                model.get_input_embeddings(),
                new_num_embeddings=new_num_embeddings,
                num_new_embeddings=num_new_tokens,
            )
            init_new_embeddings(
                model.get_output_embeddings(),
                new_num_embeddings=new_num_embeddings,
                num_new_embeddings=num_new_tokens,
            )