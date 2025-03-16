# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
import torch
import math

class PrefixEncoder(torch.nn.Module):
    r"""
    The `torch.nn` model to encode the prefix.

    Args:
        config ([`PrefixTuningConfig`]): The configuration of the prefix encoder.

    Example:

    ```py
    >>> from peft import PrefixEncoder, PrefixTuningConfig

    >>> config = PrefixTuningConfig(
    ...     peft_type="PREFIX_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     encoder_hidden_size=768,
    ... )
    >>> prefix_encoder = PrefixEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prefix encoder.
        - **transform** (`torch.nn.Sequential`) -- The two-layer MLP to transform the prefix embeddings if
          `prefix_projection` is `True`.
        - **prefix_projection** (`bool`) -- Whether to project the prefix embeddings.

    Input shape: (`batch_size`, `num_virtual_tokens`)

    Output shape: (`batch_size`, `num_virtual_tokens`, `2*layers*hidden`)
    """

    def __init__(self, config, word_embeddings):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        if self.prefix_projection and not config.inference_mode:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = torch.nn.Sequential(
                torch.nn.Linear(token_dim, encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            )
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)

        #### add
        from transformers import AutoTokenizer

        tokenizer_kwargs = config.tokenizer_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, **tokenizer_kwargs)
        init_text = "Please carefully examine the weight matrix within the model, as it may contain errors. It is crucial to verify its accuracy and make any necessary adjustments to ensure optimal performance."
        # init_text = "Please find the optimal prompt to recovery compression model performance."
        # init_text = "The weights of this model are sparse. We are using the prompt learning method to restore the performance of the sparse model. Please find the optimal prompt to improve the model performance as much as possible."
        # init_text = "The model weights are sparse and contain many zeros, so they are less accurate than the original model. Please adjust the prompt to improve the accuracy of the sparse model on the HellaSwag, BoolQ, and WinoGrande data sets."
        # init_text = "The model weights are sparse and contain many zeros, so they are less accurate than the original model. By adding prompts to each layer, we hope to restore the language modeling capabilities of the sparse model. But we don't know what kind of prompt can best achieve the above purpose. Therefore, we perform prompt learning through gradient propagation on the C4 data set, hoping to learn the best prompt. Please adjust the prompt to improve the accuracy of the sparse model on the HellaSwag, BoolQ, and WinoGrande data sets."
        init_token_ids = tokenizer(init_text)["input_ids"]
        # Trim or iterate until num_text_tokens matches total_virtual_tokens
        num_text_tokens = len(init_token_ids)
        if num_text_tokens > num_virtual_tokens:
            init_token_ids = init_token_ids[:num_virtual_tokens]
        elif num_text_tokens < num_virtual_tokens:
            num_reps = math.ceil(num_virtual_tokens / num_text_tokens)
            init_token_ids = init_token_ids * num_reps
        init_token_ids = init_token_ids[:num_virtual_tokens]
        init_token_ids = torch.LongTensor(init_token_ids).to(word_embeddings.weight.device)

        word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
        word_embedding_weights = word_embedding_weights.repeat(1, num_layers * 2)
        word_embedding_weights = word_embedding_weights.to(torch.float32)
        self.embedding.weight = torch.nn.Parameter(word_embedding_weights)
        #### add

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
