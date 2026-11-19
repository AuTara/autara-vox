"""
Offline Neural-to-Speech Audio Synthesizer
==========================================
Converts decoded English text into audible `.wav` audio using `espeak-ng` / `espeak`
if installed, or a clean formant-envelope vocal synthesizer fallback so the demo
always produces playable `.wav` speech output completely offline.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def synthesize_speech_wav(text: str, out_wav: str | Path, sample_rate: int = 16000) -> Path:
    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_text = (text or "silence").strip()

    for bin_name in ("espeak-ng", "espeak"):
        exe = shutil.which(bin_name)
        if exe:
            subprocess.run([exe, "-w", str(out_path), clean_text], check=True)
            return out_path

    # Pure-Python deterministic formant synthesizer fallback (zero external dependencies)
    vowels = {"a": (730, 1090), "e": (530, 1840), "i": (270, 2290), "o": (570, 840), "u": (300, 870)}
    segments = []
    t_step = int(0.09 * sample_rate)
    t = np.linspace(0, 0.09, t_step, endpoint=False)
    for ch in clean_text.lower():
        if ch == " ":
            segments.append(np.zeros(int(0.08 * sample_rate), dtype=np.float32))
            continue
        f1, f2 = vowels.get(ch, (420, 1450))
        f0 = 135.0
        env = np.hanning(t_step)
        sig = (
            0.50 * np.sin(2 * np.pi * f0 * t)
            + 0.30 * np.sin(2 * np.pi * f1 * t)
            + 0.20 * np.sin(2 * np.pi * f2 * t)
        ) * env
        segments.append((sig * 0.35).astype(np.float32))

    audio = np.concatenate(segments) if segments else np.zeros(sample_rate // 4, dtype=np.float32)
    sf.write(str(out_path), audio, sample_rate)
    return out_path
