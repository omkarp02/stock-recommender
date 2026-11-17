"""Speech-to-text with language routing.

Whisper (large-v3-turbo) detects the language of every clip. Hindi goes to
Maya Transcribe, which handles Hindi/Hinglish far better (see experiments/);
everything else stays on Whisper.

Maya takes at most 30 s per clip, so longer audio is split at the quietest
moment between 18 s and 28 s into each chunk. Its "mixed" style keeps English
words (stock names, "stop loss", ...) in Latin script.

Model download (once, ~1.6 GB):
  .venv/bin/hf download maya-research/maya-transcribe --local-dir models/maya-transcribe
"""
from pathlib import Path

import av
import numpy as np

# faster-whisper 1.2.1 passes metadata_errors= to av.open, which PyAV >= 15 removed.
# Patch before anything decodes audio (decode_audio as well as the model).
_av_open = av.open
av.open = lambda *a, metadata_errors=None, **k: _av_open(*a, **k)

SR = 16000
WHISPER_MODEL = "large-v3-turbo"
MAYA_DIR = Path(__file__).parent / "models" / "maya-transcribe"
MAYA_STYLE = "mixed"

_whisper = None
_maya = None


def _load_whisper(name, device):
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        print(f"loading whisper '{name}' (first run downloads it)...")
        _whisper = WhisperModel(name, device=device, compute_type="int8" if device == "cpu" else "default")
    return _whisper


def _load_maya():
    global _maya
    if _maya is None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not MAYA_DIR.exists():
            raise FileNotFoundError(f"Maya model missing; download it first (see asr.py): {MAYA_DIR}")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"loading Maya Transcribe on {device}...")
        processor = AutoProcessor.from_pretrained(MAYA_DIR, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            MAYA_DIR, dtype=torch.float32, local_files_only=True
        ).to(device).eval()
        _maya = (processor, model)
    return _maya


def split_audio(audio, min_s=18, max_s=28, win_s=0.3):
    """Split into chunks of at most max_s seconds, cutting at the quietest point."""
    chunks, start, win = [], 0, int(win_s * SR)
    while len(audio) - start > max_s * SR:
        lo, hi = start + min_s * SR, start + max_s * SR
        energy = np.convolve(audio[lo:hi] ** 2, np.ones(win) / win, mode="valid")
        cut = lo + int(np.argmin(energy)) + win // 2
        chunks.append(audio[start:cut])
        start = cut
    chunks.append(audio[start:])
    return chunks


def _maya_transcribe(audio):
    import torch

    processor, model = _load_maya()
    texts = []
    for chunk in split_audio(audio):
        inputs = processor.apply_transcription_request(
            audio=chunk, prompt=f"style: {MAYA_STYLE}", language="Hindi"
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        texts.append(processor.decode(ids[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")[0].strip())
    return " ".join(t for t in texts if t)


def transcribe(path, whisper_model=WHISPER_MODEL, translate=False, device="cpu", engine="auto"):
    """Returns {language, language_probability, duration, text, model}.
    engine: "auto" (route by language), "whisper" or "maya" to force one.
    Returns None for files without a sound track (e.g. silent carousel clips)."""
    from faster_whisper.audio import decode_audio

    with av.open(str(path)) as c:
        if not c.streams.audio:
            return None
    audio = decode_audio(str(path), sampling_rate=SR)
    if len(audio) < SR // 2:  # under half a second: nothing to transcribe
        return None
    whisper = _load_whisper(whisper_model, device)
    # transcribe() detects the language up front; segments are only decoded
    # if we iterate them, so routing to Maya costs just the detection pass.
    segments, info = whisper.transcribe(
        audio, task="translate" if translate else "transcribe", vad_filter=True, beam_size=5
    )
    result = {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(len(audio) / SR, 2),
    }
    use_maya = engine == "maya" or (engine == "auto" and info.language == "hi" and not translate)
    if use_maya:
        result["text"] = _maya_transcribe(audio)
        result["model"] = f"maya-transcribe ({MAYA_STYLE})"
    else:
        result["text"] = " ".join(s.text.strip() for s in segments)
        result["model"] = f"whisper-{whisper_model}"
    return result
