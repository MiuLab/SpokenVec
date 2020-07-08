Learning ASR-Robust Contextualized Embeddings
---
[Paper](https://www.csie.ntu.edu.tw/~yvchen/doc/ICASSP20_SpokenVec.pdf)
| [Slides](https://www.csie.ntu.edu.tw/~yvchen/doc/ICASSP20_SpokenVec_slide.pdf)
| [Presentation](https://2020.ieeeicassp-virtual.org/presentation/poster/learning-asr-robust-contextualized-embeddings-spoken-language-understanding/)

Implementation of our ICASSP 2020 paper *Learning ASR-Robust Contextualized Embeddings for Spoken Language Understanding*.

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

For fine-tuning ELMo with only LM objective (ULMFit) and using it to train SLU classifier
```
# Fine-tuning LM
python3 main_lm.py ../models/lm/snips_tts/1

# Training SLU classifier with the fine-tuned LM, you might want to modify the specific checkpoint in the config.
python3 main.py ../models/snips_tts/3
```

For fine-tuning ELMo with our method and using it to train SLU classifier
```
# Fine-tuning LM with unsupervised extracted confusions
python3 main_lm.py ../models/lm/snips_tts/2

# Fine-tuning LM with supervised extracted confusions
python3 main_lm.py ../models/lm/snips_tts/3

# Training SLU classifier with the fine-tuned LM, you might want to modify the specific checkpoint in the config.
# with lm/snips_tts/2, which uses unsupervised extraction
python3 main.py ../models/snips_tts/4

# with lm/snips_tts/3, which uses supervised extraction
python3 main.py ../models/snips_tts/5
```
