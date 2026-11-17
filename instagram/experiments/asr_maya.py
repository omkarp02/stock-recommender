"""ASR comparison, step 2 (run with .venv): transcribe the WAVs from step 1
with Maya Transcribe. Maya takes at most 30 s per clip, so each reel is split at
the quietest moment between 18 s and 28 s into each chunk.
Writes experiments/out/maya.json.

  .venv/bin/python experiments/asr_maya.py CODE [CODE ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
MODEL = HERE.parent / "models" / "maya-transcribe"
SR = 16000
STYLE = "mixed"  # Hindi in Devanagari, English words in Latin script


def split(audio, min_s=18, max_s=28, win_s=0.3):
    chunks, start = [], 0
    win = int(win_s * SR)
    while len(audio) - start > max_s * SR:
        lo, hi = start + min_s * SR, start + max_s * SR
        seg = audio[lo:hi]
        # RMS energy per sliding window; cut in the middle of the quietest one
        energy = np.convolve(seg**2, np.ones(win) / win, mode="valid")
        cut = lo + int(np.argmin(energy)) + win // 2
        chunks.append(audio[start:cut])
        start = cut
    chunks.append(audio[start:])
    return chunks


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL, dtype=torch.float32, local_files_only=True
    ).to(device).eval()
    print(f"loaded on {device}")

    out_file = OUT / "maya.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    for code in sys.argv[1:]:
        audio, sr = sf.read(OUT / "wav" / f"{code}.wav", dtype="float32")
        assert sr == SR
        chunks = split(audio)
        start = time.time()
        texts = []
        for c in chunks:
            inputs = processor.apply_transcription_request(
                audio=c, prompt=f"style: {STYLE}", language="Hindi"
            ).to(model.device, model.dtype)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            texts.append(processor.decode(ids[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")[0].strip())
        results[code] = {
            "style": STYLE,
            "duration": round(len(audio) / SR, 2),
            "chunks": [round(len(c) / SR, 1) for c in chunks],
            "seconds": round(time.time() - start, 1),
            "text": " ".join(texts),
        }
        r = results[code]
        print(f"{code} chunks={r['chunks']} {r['seconds']}s: {r['text'][:80]}")
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
