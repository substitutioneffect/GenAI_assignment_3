# Qwen3-TTS

`17_A3_Qwen.ipynb` contains the Qwen3-TTS experiments, evaluation tables and listening examples. Every path in the notebook is relative to this folder.

**Reported model checkpoints (Google Drive, 14.5 GB):** <https://drive.google.com/drive/folders/1gMHQJ8O9UIm3ZoDJEVk9b9s0HpJrW1Gh>

### Folder contents

- `17_A3_Qwen.ipynb` and `requirements.txt`
- `wav_examples/`: 121 labelled WAVs from the saved run, `qwen_<experiment>_<pytorch|vllm>_<prompt>.wav` (5 models × 2 backends × 12 fixed prompts) and the demo sentence `qwen_expC_demo.wav`
- `results/`: result tables (CSV) written by the saved run
- `qwen_report_data/`: seed-42 split manifests, the vLLM-Omni package list (`runtime/engine_requirements.txt`) and the records of the reported runs, including the SHA-256 of the four reported checkpoints (`results/formal_registry.json`)
- `LJSpeech-1.1.tar.bz2` (optional, not in Git): downloaded automatically when absent

### Reported checkpoints (Google Drive, not in Git)

The fine-tuned checkpoints are too large for Git (3.6 GB each). They are shared on Google Drive (anyone with the link can view) in the folder `GenAI_3_Checkpoints`:

**<https://drive.google.com/drive/folders/1gMHQJ8O9UIm3ZoDJEVk9b9s0HpJrW1Gh>**

| Folder | Experiment | Model name | Selected epoch | SHA-256 (start) |
|---|---|---|---:|---|
| `checkpoints/expA/` | expA | `qwen_lr2e-7_1k` | 12 | `5c39fc23a0e1` |
| `checkpoints/expB/` | expB | `qwen_lr5e-7_1k` | 6 | `00a46f48248d` |
| `checkpoints/expC/` | expC | `qwen_lr2e-7_2k` | 10 | `537e58a6c8b0` |
| `checkpoints/expD/` | expD | `qwen_lr5e-7_2k` | 4 | `483978765c38` |

The same folder holds `evaluation/` (0.9 GB): the saved vLLM-Omni items of the reported run (WAV, timing, transcript and scores; 5 models × 600 items). The Base model is downloaded from Hugging Face at the pinned revision.

### Run in Google Colab

1. Copy this folder to Google Drive, open the notebook in Google Colab and select an **NVIDIA A100 GPU (40 GB)** runtime. The recorded environment uses **Python 3.12**.
2. In **Step 1**, set `NOTEBOOK_FOLDER` to the Drive copy of this folder (for example `/content/drive/MyDrive/Qwen3TTS`; leave it empty when the notebook runs from this folder). Step 1 mounts Drive and installs the dependencies; the vLLM-Omni engine uses its separately pinned environment.
3. Choose how to run:
   - **Reported checkpoints (no training):** open <https://drive.google.com/drive/folders/1gMHQJ8O9UIm3ZoDJEVk9b9s0HpJrW1Gh> and add the folder to My Drive (**Organise → Add shortcut → My Drive**; a shortcut uses none of your storage), so that it appears at `/content/drive/MyDrive/GenAI_3_Checkpoints` (otherwise change `SHARED_FOLDER` in Step 1). Set `RUN_SEARCH = False` and `RUN_TRAINING = False`. Step 1 copies `evaluation/` next to the notebook; the search and training cells replay the recorded histories; every later step loads the shared checkpoints and stops if one differs from the recorded SHA-256. Saved vLLM-Omni items are reused (same checkpoint, text and seed); PyTorch eager audio for the backend comparison is generated again (about 4–5 A100 hours), and the demo synthesises new speech with the reported models.
   - **Whole study from scratch:** keep the default switches (`RUN_SEARCH`, `RUN_TRAINING`, `RUN_EVALUATION`, `RUN_DEMO` = `True`): learning-rate search, four training runs, evaluation with vLLM-Omni and PyTorch eager, and the demo (about a day of A100 time). Checkpoints are large (`latest.pt` about 11 GB, `best.pt` about 3.6 GB per run); the search needs about 130 GB of free space at its peak. Finished epochs and generated items are saved, so an interrupted session continues when the cells are run again.
4. Execute the cells in notebook order, from setup through final verification. To preserve the supplied execution evidence, use a working copy for a fresh execution.

A fresh run rewrites `results/` and `wav_examples/`; `evaluation/` and `checkpoints/` hold generated audio and model weights and are not meant for Git. The outputs saved in the notebook, `results/` and `wav_examples/` come from the 30–31 Oct 2026 run with the reported checkpoints (see the note at the top of the notebook). Keep the recorded seeds, model versions and generation settings for reproducible comparisons.
