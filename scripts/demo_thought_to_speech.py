#!/usr/bin/env python3
"""
Autara-Vox Live EEG-to-Speech Demo (Text + Phonemes + Spoken .wav Audio)
========================================================================
Demonstrates end-to-end decoding from real held-out 128-ch EEG + subvocal
high-frequency articulatory stream -> open-loop English text, zero-shot phrase
retrieval (including custom user phrase bank like "i am the one"), and offline
speech audio synthesis (.wav).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from autara_vox import AutaraVoxSpeechDecoder, synthesize_speech_wav

DATA_ROOT = Path("/media/beast/DATA1/autara_vox_datasets/ds004408_proc")
CKPT_PATH = Path("checkpoints/autara_vox_speech_best.pt")
DEMO_OUT = Path("paper/results/demo_audio")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((DATA_ROOT / "meta.json").read_text())
    records = json.loads((DATA_ROOT / "records.json").read_text())
    phoneme_vocab = meta["phoneme_vocab"]

    pure_arr = np.load(DATA_ROOT / "pure_eeg_fp16.npy", mmap_mode="r")
    jaw_arr = np.load(DATA_ROOT / "jaw_emg_fp16.npy", mmap_mode="r")

    model = AutaraVoxSpeechDecoder(
        num_phonemes=len(phoneme_vocab), jaw_ch=jaw_arr.shape[1], d_model=256
    ).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()

    # Select 5 held-out unseen-run test windows
    max_run = max(r["run_id"] for r in records)
    unseen = [i for i, r in enumerate(records) if r["run_id"] == max_run][:5]

    # Build a semantic phrase bank combining held-out phrases + custom target phrase "i am the one"
    gallery_phrases = ["i am the one"] + list(dict.fromkeys([records[i]["text"] for i in unseen]))
    tok = model.tokenizer(gallery_phrases, padding=True, truncation=True, max_length=24, return_tensors="pt").to(device)
    with torch.no_grad():
        g_txt = model.encode_text_frozen(tok["input_ids"], tok["attention_mask"])
        g_emb = F.normalize(model.text_proj(g_txt), dim=-1)

    print("\n=================================================================")
    print("       AUTARA-VOX: LIVE EEG-TO-SPEECH INFERENCE DEMO             ")
    print("=================================================================\n")

    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    for k, idx in enumerate(unseen, start=1):
        pe = torch.from_numpy(np.array(pure_arr[idx], dtype=np.float32)).unsqueeze(0).to(device)
        je = torch.from_numpy(np.array(jaw_arr[idx], dtype=np.float32)).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        gen_txts, ctc_ids, z_n, cw, sw = model.generate_open_loop(pe, je, stream_mode="dual")
        lat_ms = (time.perf_counter() - t0) * 1000.0

        sims = (z_n @ g_emb.T).squeeze(0)
        best_j = int(sims.argmax().item())
        wav_path = synthesize_speech_wav(gen_txts[0], DEMO_OUT / f"decoded_{k}.wav")

        print(f">>> Unseen Neural Window [{k}/{len(unseen)}]")
        print(f"    ├─ Ground Truth Text : {records[idx]['text']}")
        print(f"    ├─ Open-Loop Decoded : {gen_txts[0]}")
        print(f"    ├─ Top-1 Retrieved   : {gallery_phrases[best_j]} (cos={sims[best_j].item():.3f})")
        print(
            f"    ├─ 4-Center Gate     : Speech={cw[0,0]*100:.1f}% | Aud={cw[0,1]*100:.1f}% | "
            f"Vis={cw[0,2]*100:.1f}% | Mem={cw[0,3]*100:.1f}%"
        )
        print(f"    ├─ Dual-Stream Gate  : Pure-EEG={sw[0,0]*100:.1f}% | Jaw/Artic={sw[0,1]*100:.1f}%")
        print(f"    └─ Audio Synthesized : {wav_path} ({lat_ms:.1f} ms)\n")

    # Also synthesize custom reference phrase "i am the one"
    ref_wav = synthesize_speech_wav("i am the one", DEMO_OUT / "target_i_am_the_one.wav")
    print(f"Synthesized target speech demo -> {ref_wav}")


if __name__ == "__main__":
    main()
