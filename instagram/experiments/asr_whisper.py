"""ASR comparison, step 1 (run with .venv): download reels, make 16 kHz mono WAVs,
transcribe with Whisper. Writes experiments/out/whisper.json.

  .venv/bin/python experiments/asr_whisper.py CODE [CODE ...]
"""
import json
import sys
import time
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import asr  # noqa: E402
import transcribe_latest as t  # noqa: E402

OUT = Path(__file__).parent / "out"
WAV = OUT / "wav"


def to_wav(src, dst):
    """Decode any audio file to 16 kHz mono float32 WAV with PyAV."""
    import wave

    c = av.open(str(src))
    res = av.AudioResampler(format="s16", layout="mono", rate=16000)
    pcm = []
    for frame in c.decode(audio=0):
        for f in res.resample(frame):
            pcm.append(f.to_ndarray().reshape(-1))
    for f in res.resample(None):
        pcm.append(f.to_ndarray().reshape(-1))
    data = np.concatenate(pcm).astype(np.int16)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(data.tobytes())
    return len(data) / 16000


def main():
    codes = sys.argv[1:]
    WAV.mkdir(parents=True, exist_ok=True)
    out_file = OUT / "whisper.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    for code in codes:
        wav = WAV / f"{code}.wav"
        if not wav.exists():
            files, _ = t.download_audio(f"https://www.instagram.com/reel/{code}/", f"asr_{code}")
            if not files:
                print(f"{code}: download failed")
                continue
            dur = to_wav(files[0], wav)
            for f in files:
                Path(f).unlink(missing_ok=True)
            print(f"{code}: {dur:.1f}s audio")
        start = time.time()
        r = asr.transcribe(str(wav), engine="whisper")
        r["seconds"] = round(time.time() - start, 1)
        results[code] = r
        print(f"{code} [{r['language']} {r['language_probability']}] {r['seconds']}s: {r['text'][:80]}")
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
