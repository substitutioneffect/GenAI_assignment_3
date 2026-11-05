# GenAI_assignment_3

Assignment 3 (Group 17): Text-to-Voice (single-speaker TTS) on LJSpeech 1.1. Each model has its own folder with its notebook, requirements, result CSVs and WAV examples, and every notebook runs on Google Colab with a GPU.

| Folder | Model | Experiments | Code | Fine-tuned weights |
|---|---|---|---|---|
| [`Qwen3TTS/`](Qwen3TTS/) | Qwen3-TTS 12Hz 0.6B Base, full fine-tuning of the Talker | Base + four controlled runs (learning rate 2e-7 / 5e-7 × 1k / 2k data) chosen by a nine-candidate learning-rate search; PyTorch eager vs vLLM-Omni inference | `17_A3_Qwen.ipynb` | [Google Drive](https://drive.google.com/drive/folders/1gMHQJ8O9UIm3ZoDJEVk9b9s0HpJrW1Gh) (anyone with the link) |
| [`SpeechT5/`](SpeechT5/) | Microsoft SpeechT5 (`microsoft/speecht5_tts`), full fine-tuning and LoRA | Exp0 (no training) + ExpA–ExpF (learning rate 1e-5 / 1e-4, 1k / 2k data, LoRA) | `Group17_A3_T5model.ipynb`, `generate_wav_T5A.py` … `generate_wav_T5F.py` | [GitHub Releases](https://github.com/substitutioneffect/GenAI_assignment_3/releases) (`T5_ExpA.zip` … `T5_ExpF.zip`; needs `GH_TOKEN` because the repository is private) |
| [`XTTS/`](XTTS/) | Coqui XTTS-v2, GPT fine-tuning | Exp 0 (zero-shot) + Exp A / Exp B (learning rate 1e-6 / 1e-7, 1k data) | `XTTS.ipynb` | not shared; the notebook retrains them |

- **Qwen3-TTS** and **SpeechT5** are the two systems of the main report; see [`Qwen3TTS/README.md`](Qwen3TTS/README.md) and [`SpeechT5/README.md`](SpeechT5/README.md) for running instructions.
- **XTTS-v2** is a supplementary study kept outside the main report; the last section of `XTTS.ipynb` explains why. Its setup cell expects `LJSpeech-1.1.tar.bz2` in the Drive folder set as `PROJECT_ROOT`.


**Not in Git:** the LJSpeech dataset (downloaded or read from Drive as described in each folder), model checkpoints and generated evaluation caches (see `.gitignore`).
