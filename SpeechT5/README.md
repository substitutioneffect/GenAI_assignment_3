# SpeechT5 Text-to-Speech (LJSpeech)

Fine-tuning Microsoft SpeechT5 to speak in the LJSpeech voice.

## Files

| File | What it is |
|---|---|
| `Group17_A3_T5model.ipynb` | Full code: data, training, evaluation |
| `generate_wav.py` | Turns any text into a WAV file |
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

## Generate a WAV

1. Download a model zip (for example `T5_ExpC.zip`) from the [Releases](https://github.com/substitutioneffect/GenAI_assignment_3/releases) page.
2. Unzip it into `checkpoints/`, so you have `checkpoints/ExpC/`.
3. Run:

```bash
pip install -r requirements.txt
python generate_wav.py "Adelaide has great weather in spring."                   # uses ExpC
python generate_wav.py "Adelaide has great weather in spring." checkpoints/ExpE  # any other model
```

The audio is saved in `wav_examples/`, for example `T5_Git_ExpC_01.wav`.
