Learning ASR-Robust Contextualized Embeddings
---
[Paper](https://ieeexplore.ieee.org/abstract/document/9054689)
| [Presentation](https://2020.ieeeicassp-virtual.org/presentation/poster/learning-asr-robust-contextualized-embeddings-spoken-language-understanding/)

Implementation of our ICASSP 2020 paper *Learning ASR-Robust Contextualized Embeddings for Spoken Language Understanding*.

**Note: the dataset and model configs will be uploaded very soon**

## Requirements
* Python >= 3.6
* Install the required Python packages with `pip3 install -r requirements.txt`

## How to run
We provide a transcribed and processed dataset of the SNIPS NLU benchmark, where the audio files were generated with a TTS system, for training and evaluation.

The training configs are located in [models](models).

### Steps
For training baseline models with or without ELMo embeddings:

```
# For static word embeddings
python3 main.py ../models/snips_tts/1

# For pre-trained ELMo embeddings
python3 main.py ../models/snips_tts/2
```

For fine-tuning ELMo with our method and using it to train SLU classifier
```
# Fine-tuning
python3 main_lm.py ../models/lm/snips_tts/1

# Training SLU classifier with the fine-tuned LM, you might want to modify the specific checkpoint in the config.
python3 main.py ../models/snips_tts/3
```
