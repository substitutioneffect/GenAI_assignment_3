# SpeechT5 Text-to-Speech (LJSpeech)

Fine-tuning Microsoft SpeechT5 to speak in the LJSpeech voice.

## Files

| File | What it is |
|---|---|
| `Group17_A3_T5model.ipynb` | Full code: data, training, evaluation |
| `generate_wav_T5A.py` … `generate_wav_T5F.py` | One script per model: type text, hear the speech |
| `requirements.txt` | Packages to install |
| `results/` | Test results for every model (CSV) |
| `wav_examples/` | 12 test sentences spoken by every model |

## Models

All models start from `microsoft/speecht5_tts`, train for 20 epochs (seed 42), and keep the epoch with the lowest validation loss.

| Model | Training data | Learning rate | Method | WER (%) | RTF |
|---|---|---|---|---|---|
| Exp0 | none | none | Original model, no training | 9.27 | 0.246 |
| ExpA | 1,000 clips | 1e-5 | Full fine-tuning | 6.86 | 0.245 |
| ExpB | 1,000 clips | 1e-4 | Full fine-tuning | 7.11 | 0.248 |
| **ExpC** | **2,000 clips** | **1e-5** | **Full fine-tuning (best)** | **4.36** | **0.245** |
| ExpD | 2,000 clips | 1e-4 | Full fine-tuning | 8.61 | 0.248 |
| ExpE | 1,000 clips | 1e-5 | LoRA (trains 1.6% of weights) | 5.24 | 0.504 |
| ExpF | 1,000 clips | 1e-4 | LoRA (trains 1.6% of weights) | 13.80 | 0.507 |

- **WER**: word error rate from Whisper; lower is better.
- **RTF**: generation time ÷ audio length; lower is faster.
- Exp0, ExpA, ExpB, ExpE and ExpF use the 100-sentence test set; ExpC and ExpD use the 200-sentence test set.

## Generate speech

In Colab or Jupyter, from this folder:

```python
!pip install -q -r requirements.txt
%env GH_TOKEN=your_github_token
%run generate_wav_T5C.py "Adelaide has great weather in spring."
```

An audio player appears under the cell (22,050 Hz); no file is saved. Change the letter (`T5A` … `T5F`) to hear another model.
On the first run, each script downloads its model zip from the [Releases](https://github.com/substitutioneffect/GenAI_assignment_3/releases) page and unzips it into `checkpoints/`. The repo is private, so set your GitHub token once per session with `%env GH_TOKEN=...`; it is never written into the scripts.
