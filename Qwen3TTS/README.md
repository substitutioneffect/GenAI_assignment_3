# Qwen3-TTS

`Qwen_A3_submission.ipynb` contains the Qwen3-TTS experiments, evaluation tables and listening examples. Every path in the notebook is relative to this folder.

### Folder contents

- `Qwen_A3_submission.ipynb` and `requirements.txt`
- `wav_examples/`: 121 labelled WAVs from the saved run, `qwen_<experiment>_<pytorch|vllm>_<prompt>.wav` (5 models × 2 backends × 12 fixed prompts) and the demo sentence `qwen_expC_demo.wav`
- `results/`: result tables (CSV) written by the saved run
- `qwen_report_data/`: seed-42 split manifests, the vLLM-Omni package list (`runtime/engine_requirements.txt`) and the records of the reported runs
- `LJSpeech-1.1.tar.bz2` (optional, not in Git): downloaded automatically when absent

### Run in Google Colab

1. Copy this folder to Google Drive, open the notebook in Google Colab and select an **NVIDIA A100 GPU (40 GB)** runtime. The recorded environment uses **Python 3.12**.
2. In **Step 1**, set `NOTEBOOK_FOLDER` to the Drive copy of this folder (for example `/content/drive/MyDrive/Qwen3TTS`; leave it empty when the notebook runs from this folder). Step 1 mounts Drive and installs the dependencies; the vLLM-Omni engine uses its separately pinned environment.
3. Execute the cells in notebook order, from setup through final verification. To preserve the supplied execution evidence, use a working copy for a fresh execution.
4. The default switches (`RUN_SEARCH`, `RUN_TRAINING`, `RUN_EVALUATION`, `RUN_DEMO` = `True`) repeat the whole study: learning-rate search, four training runs, evaluation with vLLM-Omni and PyTorch eager, and the demo (about a day of A100 time). Finished epochs and generated items are saved, so an interrupted session continues when the cells are run again.
5. Checkpoints are large (`latest.pt` about 11 GB, `best.pt` about 3.6 GB per run); the search needs about 130 GB of free space at its peak.

A fresh run rewrites `results/` and `wav_examples/`; `evaluation/` and `checkpoints/` hold generated audio and model weights and are not meant for Git. The outputs saved in the notebook, `results/` and `wav_examples/` come from the 30–31 Oct 2026 run with the recorded checkpoints (see the note at the top of the notebook). Keep the recorded seeds, model versions and generation settings for reproducible comparisons.
