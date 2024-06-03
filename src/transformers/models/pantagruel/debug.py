# coding=utf-8
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert data2vec 2 checkpoint."""

import os
import argparse
import json

import torch

from transformers import (
    Data2Vec2MultiConfig,
    Data2Vec2MultiModel,
)

def compare_tensors(tensor_a, tensor_b):
    max_absolute_diff = torch.max(torch.abs(tensor_a - tensor_b)).item()
    # print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(tensor_a, tensor_b, atol=1e-3)
    if not success:
        raise ValueError("!!!Something went wrong!!!")
    

def main():
    # with open('config.json', 'r') as fp:
    #     model_config = json.load(fp)

    checkpoint_path = "/lus/home/CT10/c1615074/tphle/pantagruel/pretrained_models/Text_Base_fr_4GB_v0/checkpoint_best.pt"
    pytorch_dump_folder_path = "/lus/home/CT10/c1615074/tphle/pantagruel/pretrained_models/Text_Base_fr_4GB_v0/HuggingFace"

    fairseq_ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = fairseq_ckpt["model"]
    # fairseq_model_config = fairseq_ckpt["cfg"]["model"]
    # model_config = {k: v for k, v in fairseq_model_config.items() if not "ema" in k and not "decoder" in k and not "loss" in k}
    # model_config["modalities"]["text"]["vocab_size"] = 49937

    # configuration = Data2Vec2MultiConfig()
    # configuration.update(model_config)

    # print(configuration.supported_modality)
    # # print(type(configuration.modalities.audio))
    # print(configuration.modalities.text)
    # print(type(configuration.modalities))

    # hf_model = Data2Vec2MultiModel(configuration)
    # keys = list(state_dict.keys())
    # for k in keys:
    #     if "ema" in k or "decoder" in k:
    #         del state_dict[k]

    # # saving pretrained model and configuration
    # hf_model.load_state_dict(state_dict, strict=True)
    # hf_model.save_pretrained(pytorch_dump_folder_path, safe_serialization=False)

    # # comparing with fairseq state dict
    # print("Loading from pretrain model...")
    # configuration = Data2Vec2MultiConfig.from_pretrained(pytorch_dump_folder_path)
    # # print(f"*** HuggingFace configuration ***\n{configuration}")
    # print(f"supported_modality: {configuration.supported_modality}")
    # print(f"type(configuration.modalities): {type(configuration.modalities)}")
    # print(type(configuration.modalities.text))

    test_model = Data2Vec2MultiModel.from_pretrained(pytorch_dump_folder_path)
    test_model.eval()
    test_model.freeze_feature_encoder()
    for n, p in test_model.named_parameters():
        compare_tensors(p, state_dict[n])

if __name__ == "__main__":
    main()
