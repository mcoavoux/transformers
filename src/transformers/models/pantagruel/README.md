# Pantagruel

This folder contains codebase for loading and using the speech-only and text-only pre-trained models, which are based on [data2vec 2.0 architecture](https://arxiv.org/abs/2212.07525). The pre-trained models were trained using `fairseq` v1 library.

## Current models
Current available pre-trained models are saved under the follwing directory: `/lus/work/CT10/lig3801/SHARED/pretrained_models`, including:
- `Speech_Base_fr_1K`: trained on around 1K hours of the French subset of Multilingual LibriSpeech corpus 
- `Text_Base_fr_4GB_v0`: trained on around 5GB of text from French Wikipedia 2019 dump
- `camembert-base-wikipedia-4gb`: trained on similar pre-training corpus as `Text_Base_fr_4GB_v0`.

The converted HuggingFace models are saved under sub-folder named `HuggingFace` in corresponding model-specific folders.


## Feature extraction
To extract representations for a given audio or textual input, the pre-trained speech-only or text-only models can be used as follows:
```python
from pathlib import Path
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    Data2Vec2MultiConfig,
    Data2Vec2MultiModel,
)
BOS_TOKEN = "<s>"

pretrained_dir = Path("/lus/work/CT10/lig3801/SHARED/pretrained_models")
audio_model_dir = pretrained_dir / "Speech_Base_fr_1K" / "HuggingFace"
text_model_dir = pretrained_dir / "Text_Base_fr_4GB_v0" / "HuggingFace"

mode = "AUDIO"
model_dir = audio_model_dir if mode == "AUDIO" else text_model_dir

# load speech-only or text-only pretrained model
hf_model = Data2Vec2MultiModel.from_pretrained(model_dir)
hf_model.eval()
hf_model.freeze_feature_encoder()

# audio input
input_values = torch.randn((3, 320000), dtype=torch.float32)
hf_output = hf_model(input_values, mode="AUDIO")
extracted_features = hf_output.last_hidden_state

# text input
SAMPLE_TEXT = "Bonjour le monde !!"

tokenizer = ByteLevelBPETokenizer(
    (text_model_dir / "encoder.json").as_posix(),
    (text_model_dir / "vocab.bpe").as_posix(),
    add_prefix_space=True)
encoded = tokenizer.encode(SAMPLE_TEXT)
# prepend BOS token <s>
encoded_ids = [tokenizer.token_to_id(BOS_TOKEN)] + encoded.ids
input_values = torch.tensor(encoded_ids, dtype=torch.int64).unsqueeze(0)
hf_output = hf_model(input_values, mode="TEXT")
extracted_features = hf_output.last_hidden_state
```
