# coding=utf-8
#
# Modified by Hang Le (hangtp.le@gmail.com)
# Original copyrights by the fairseq authors and the HuggingFace team
# 
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Data2Vec2 multi configuration"""

import math
from typing import Optional
from collections import namedtuple
from enum import Enum, auto
from dataclasses import dataclass, field

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Modality(Enum):
    AUDIO = auto()
    TEXT = auto()


class D2v2ModalityConfig:
    def __init__(
        self,
        type=Modality,
        prenet_depth=4,
        prenet_layerdrop=0,
        prenet_dropout=0.0,
        start_drop_path_rate=0.0,
        end_drop_path_rate=0.0,
        num_extra_tokens=0,
        init_extra_token_zero=True,
        mask_noise_std=0.01,
        mask_prob_min=None,
        mask_prob=0.7,
        inverse_mask=False,
        mask_prob_adjust=0.0,
        keep_masked_pct=0.0,
        mask_length=5,
        add_masks=False,
        remove_masks=False,
        mask_dropout=0.0,
        encoder_zero_mask=True,
        mask_channel_prob=0.0,
        mask_channel_length=64,
        local_grad_mult=1.0,
        use_alibi_encoder=False,
        alibi_scale=1.0,
        learned_alibi=False,
        alibi_max_pos=None,
        learned_alibi_scale=False,
        learned_alibi_scale_per_head=False,
        learned_alibi_scale_per_layer=False,
        num_alibi_heads=12,
        model_depth=12,
        ema_local_encoder=False,
        decoder=None,
    ):
        self.type = type
        self.prenet_depth = prenet_depth
        self.prenet_layerdrop = prenet_layerdrop
        self.prenet_dropout = prenet_dropout
        self.start_drop_path_rate = start_drop_path_rate
        self.end_drop_path_rate = end_drop_path_rate
        self.num_extra_tokens = num_extra_tokens
        self.init_extra_token_zero = init_extra_token_zero
        self.mask_noise_std = mask_noise_std
        self.mask_prob_min = mask_prob_min
        self.mask_prob = mask_prob
        self.inverse_mask = inverse_mask
        self.mask_prob_adjust = mask_prob_adjust
        self.keep_masked_pct = keep_masked_pct
        self.mask_length = mask_length
        self.add_masks = add_masks
        self.remove_masks = remove_masks
        self.mask_dropout = mask_dropout
        self.encoder_zero_mask = encoder_zero_mask
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_length = mask_channel_length
        self.local_grad_mult = local_grad_mult
        self.use_alibi_encoder = use_alibi_encoder
        self.alibi_scale = alibi_scale
        self.learned_alibi = learned_alibi
        self.alibi_max_pos = alibi_max_pos
        self.learned_alibi_scale = learned_alibi_scale
        self.learned_alibi_scale_per_head = learned_alibi_scale_per_head
        self.learned_alibi_scale_per_layer = learned_alibi_scale_per_layer
        self.num_alibi_heads = num_alibi_heads
        self.model_depth = model_depth


class D2v2AudioConfig(D2v2ModalityConfig):
    def __init__(
        self, 
        type=Modality.AUDIO,
        extractor_mode="layer_norm",
        feature_encoder_spec="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        conv_pos_width=95,
        conv_pos_groups=16,
        conv_pos_depth=5,
        conv_pos_pre_ln=False,
        **kwargs,
    ):
        super().__init__(type=type, **kwargs)
        self.extractor_mode = extractor_mode
        self.feature_encoder_spec = feature_encoder_spec
        self.conv_pos_width = conv_pos_width
        self.conv_pos_groups = conv_pos_groups
        self.conv_pos_depth = conv_pos_depth
        self.conv_pos_pre_ln = conv_pos_pre_ln


class D2v2TextConfig(D2v2ModalityConfig):
    def __init__(
        self, 
        type=Modality.TEXT,
        max_source_positions=512,
        learned_pos=True,
        dropout=0.1,
        no_scale_embedding=True,
        layernorm_embedding=True,
        no_token_positional_embeddings=False,
        **kwargs,
    ):
        super().__init__(type=type, **kwargs)
        self.max_source_positions = max_source_positions
        self.learned_pos = learned_pos
        self.dropout = dropout
        self.no_scale_embedding = no_scale_embedding
        self.layernorm_embedding = layernorm_embedding
        self.no_token_positional_embeddings = no_token_positional_embeddings


class D2v2ModalitiesConfig:
    def __init__(self, audio_args, text_args):
        self.audio = D2v2AudioConfig(**audio_args)
        self.text = D2v2TextConfig(**text_args)


class Data2Vec2MultiConfig(PretrainedConfig):

    model_type = "data2vec2-multi"

    def __init__(
        self,
        depth=12,
        start_drop_path_rate=0.0,
        end_drop_path_rate=0.0,
        num_heads=12,
        norm_eps=1e-5,
        norm_affine=True,
        encoder_dropout=0.1,
        post_mlp_drop=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        dropout_input=0.0,
        layerdrop=0.0,
        embed_dim=768,
        mlp_ratio=4.0,
        layer_norm_first=False,
        end_of_block_targets=False,
        clone_batch=1,
        log_norms=True,
        modalities=None,
        supported_modality=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.start_drop_path_rate = start_drop_path_rate
        self.end_drop_path_rate = end_drop_path_rate

        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.norm_affine = norm_affine
        self.post_mlp_drop = post_mlp_drop
        self.encoder_dropout = encoder_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout_input = dropout_input
        self.layerdrop = layerdrop
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.layer_norm_first = layer_norm_first
        self.end_of_block_targets = end_of_block_targets
        self.clone_batch = clone_batch
        self.log_norms = log_norms

        self.modalities = modalities
        self.supported_modality = supported_modality