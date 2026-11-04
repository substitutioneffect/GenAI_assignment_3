"""
SpeechT5 ExpD: 2,000 clips, lr 1e-4, full fine-tuning.

Usage (Colab / Jupyter):
    %env GH_TOKEN=your_github_token                  # once per session (the repo is private)
    %run generate_wav_T5D.py "Your sentence here."
Shows an audio player. No file is saved.
First run: downloads T5_ExpD.zip from the GitHub Release and unzips it into checkpoints/ExpD/.
"""
import io
import os
import sys
import zipfile
from pathlib import Path

import requests
import torch
from unidecode import unidecode
from nemo_text_processing.text_normalization.normalize import Normalizer
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

GH_TOKEN = os.environ.get("GH_TOKEN", "")     # set in the notebook: %env GH_TOKEN=...

RUN_NAME = "ExpD"
REPO = "substitutioneffect/GenAI_assignment_3"
RELEASE_TAG = "V1.1"
SEED = 42
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_ID = "microsoft/speecht5_tts"
LORA_ADAPTER_NAME = "ljspeech"

text = sys.argv[1] if len(sys.argv) > 1 else input("Text: ")
run_checkpoint_dir = Path("checkpoints") / RUN_NAME


# ---- download and unzip the model from the GitHub Release (first run only) ----
if not run_checkpoint_dir.exists():
    headers = {"Authorization": f"Bearer {GH_TOKEN}"} if GH_TOKEN else {}
    assets = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{RELEASE_TAG}", headers=headers).json()["assets"]
    url = [a["url"] for a in assets if a["name"] == f"T5_{RUN_NAME}.zip"][0]
    print(f"Downloading T5_{RUN_NAME}.zip ...")
    data = requests.get(url, headers={**headers, "Accept": "application/octet-stream"}).content
    zipfile.ZipFile(io.BytesIO(data)).extractall("checkpoints")         # -> checkpoints/{RUN_NAME}/


# ---- notebook: text preprocessing ----
normalizer = Normalizer(lang="en", input_case="cased", cache_dir="nemo_cache")   # cached: only the first run is slow

def clean_text(text, num_handle=1):
    text = unidecode(text)                       # 1. non-ASCII -> ASCII
    if num_handle == 1:
        text = normalizer.normalize(text)        # 2. numbers -> words (new text only)
    for short, full in {"No.": "Number", "i.e." : "i e", "e.g.": "e g" }.items() :       # 3. expand abbreviations
        text = text.replace(short, full)
    text = text.lower()                          # 4. lowercase
    for symbol in '"()[]':                       # 5. remove quotes and brackets (not spoken)
        text = text.replace(symbol, "")
    text = text.replace(";", ",")                # 6. semicolon -> comma (keeps the pause)
    for symbol in "-:":                          # 7. hyphens and colons -> space
        text = text.replace(symbol, " ")
    text = " ".join(text.split())                # 8. single spaces
    if text.endswith(","):                       # 9. final comma -> full stop
        text = text[:-1]
        if not text.endswith("."):               #  "p.m.," becomes "p.m.", not "p.m.."
            text += "."
    return text


# ---- notebook: load best checkpoint (run_experiment, full fine-tune) ----
model = SpeechT5ForTextToSpeech.from_pretrained(str(run_checkpoint_dir))
model = model.to(DEVICE).eval()

# ---- notebook: processor, vocoder and speaker embedding ----
processor = SpeechT5Processor.from_pretrained(BASE_MODEL_ID)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE).eval()
speaker_embedding = torch.load(run_checkpoint_dir / "speaker_embedding.pt")     # (512,)
speaker = speaker_embedding.unsqueeze(0).to(DEVICE)                              # (1, 512)


# ---- notebook: generate_examples ----
input_ids = processor(text=clean_text(text, num_handle=1), return_tensors="pt")["input_ids"].to(DEVICE)   # (1, L)
torch.manual_seed(SEED)
with torch.no_grad():
    speech = model.generate_speech(input_ids, speaker_embeddings=speaker, vocoder=vocoder, maxlenratio=8.0)  # (T,)
audio = speech.cpu().numpy().reshape(-1)


# ---- play (no file saved) ----
print(f"{RUN_NAME}: {len(audio) / SAMPLE_RATE:.2f} s of audio")
from IPython.display import Audio, display
display(Audio(audio, rate=SAMPLE_RATE))
