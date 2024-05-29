# coding=utf-8
# Modified by Hang Le (hangtp.le@gmail.com)
# Original copyrights below
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

import argparse

import torch
from transformers import Data2Vec2MultiConfig, Data2Vec2MultiModel
from transformers.utils import WEIGHTS_NAME


def compare_tensors(tensor_a, tensor_b):
    max_absolute_diff = torch.max(torch.abs(tensor_a - tensor_b)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(tensor_a, tensor_b, atol=1e-3)
    if success:
        return "Two tensors are the same."
    else:
        raise ValueError("!!!Something went wrong!!!")


@torch.no_grad()
def convert_data2vec2_checkpoint(
    checkpoint_path, pytorch_dump_folder_path,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    fairseq_ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = fairseq_ckpt["model"]
    fairseq_model_config = fairseq_ckpt["cfg"]["model"]
    model_config = {k: v for k, v in fairseq_model_config.items() if not "ema" in k and not "decoder" in k and not "loss" in k}
    configuration = Data2Vec2MultiConfig(**model_config)
    print(f"*** HuggingFace configuration ***\n{configuration}")

    hf_model = Data2Vec2MultiModel(configuration)
    keys = list(state_dict.keys())
    for k in keys:
        if "ema" in k or "decoder" in k:
            del state_dict[k]
    hf_model.load_state_dict(state_dict, strict=True)
    hf_model.save_pretrained(pytorch_dump_folder_path, safe_serialization=False)

    test_model = Data2Vec2MultiModel.from_pretrained(pytorch_dump_folder_path)
    test_model.eval()

    for n, p in test_model.named_parameters():
        print(f'- Comparing {n}: {compare_tensors(p, state_dict[n])}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    args = parser.parse_args()

    convert_data2vec2_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path
    )