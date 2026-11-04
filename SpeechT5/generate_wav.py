"""
Text -> WAV with a trained SpeechT5 checkpoint, using the same functions as the notebook.

Usage:  python generate_wav.py "Your sentence here."                       (uses checkpoints/ExpC)
        python generate_wav.py "Your sentence here." checkpoints/ExpE      (any other run)
"""
import sys
from pathlib import Path

import soundfile as sf
import torch
from unidecode import unidecode
from nemo_text_processing.text_normalization.normalize import Normalizer
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

SEED = 42
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_ID = "microsoft/speecht5_tts"
LORA_ADAPTER_NAME = "ljspeech"

text = sys.argv[1] if len(sys.argv) > 1 else input("Text: ")
run_checkpoint_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("checkpoints/ExpC")
OUTPUT_DIR = Path("wav_examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- notebook: text preprocessing ----
normalizer = Normalizer(lang="en", input_case="cased")

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


# ---- notebook: load best checkpoint (run_experiment) ----
if (run_checkpoint_dir / "adapter_config.json").exists():      # LoRA (ExpE, ExpF)
    model = SpeechT5ForTextToSpeech.from_pretrained(BASE_MODEL_ID)
    model.load_adapter(str(run_checkpoint_dir), adapter_name=LORA_ADAPTER_NAME)
    model.set_adapter(LORA_ADAPTER_NAME)
else:                                                          # full fine-tune (ExpA-ExpD)
    model = SpeechT5ForTextToSpeech.from_pretrained(str(run_checkpoint_dir))
model = model.to(DEVICE).eval()

# ---- notebook: processor and vocoder ----
processor = SpeechT5Processor.from_pretrained(BASE_MODEL_ID)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE).eval()

# ---- notebook: speaker embedding (saved with torch.save) ----
speaker_embedding = torch.load(run_checkpoint_dir / "speaker_embedding.pt")     # (512,)
speaker = speaker_embedding.unsqueeze(0).to(DEVICE)                              # (1, 512)

# ---- notebook: generate_examples ----
input_ids = processor(text=clean_text(text, num_handle=1), return_tensors="pt")["input_ids"].to(DEVICE)   # (1, L)
torch.manual_seed(SEED)
with torch.no_grad():
    speech = model.generate_speech(input_ids, speaker_embeddings=speaker, vocoder=vocoder, maxlenratio=8.0)  # (T,)

number = len(list(OUTPUT_DIR.glob(f"T5_Git_{run_checkpoint_dir.name}_*.wav"))) + 1
output_path = OUTPUT_DIR / f"T5_Git_{run_checkpoint_dir.name}_{number:02d}.wav"
sf.write(str(output_path), speech.cpu().numpy().reshape(-1), SAMPLE_RATE)
print(f"Saved {output_path} ({len(speech) / SAMPLE_RATE:.2f} s)")
