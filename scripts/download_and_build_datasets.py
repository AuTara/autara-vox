#!/usr/bin/env python3
"""
Autara-Vox 4-Center OpenNeuro Dataset Builder
=============================================
Downloads & preprocesses all three open-access OpenNeuro BIDS EEG datasets onto
`/media/beast/DATA1/autara_vox_datasets/`:

  1. ds003626 (Thinking Out Loud — Speech + Visual + 8-ch EXG Jaw/Facial)
     - 10 subjects x 3 sessions (derivatives eeg-epo.fif + exg-epo.fif + events.dat)
     - Pure cortical EEG (1-45 Hz) partitioned into 4 centers [4, 16, 256] @ 128 Hz
     - Subvocal jaw/facial high-freq EMG (50-120 Hz) from EXG1..EXG8 [8, 512] @ 256 Hz
     - Session-level baseline normalization (preserves task ERD/ERS & articulatory bursts!)

  2. ds004408 (Continuous Naturalistic English Speech + Millisecond .TextGrid)
     - 10 subjects x 20 audiobook runs ("The Old Man and the Sea")
     - 128-ch BioSemi EEG @ 512 Hz -> 4-center Pure EEG (1-45 Hz @ 128 Hz) [4, 16, 384]
       AND High-Freq Jaw/Temporal Articulatory Ring (50-200 Hz envelope @ 128 Hz) [16, 384]
     - Exact millisecond-aligned English phrases (3.0s windows) + ARPAbet phoneme sequences!

  3. ds003655 (Verbal Working Memory — Modified Sternberg 5/6/7 Letters)
     - 12 subjects mapped to 4 centers [4, 16, 256] @ 128 Hz for Memory-Center pretraining
       and working-memory transfer validation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import boto3
import mne
import numpy as np
from botocore import UNSIGNED
from botocore.client import Config
from scipy import signal

from autara_vox.montage import (
    CORTICAL_CENTERS,
    build_biosemi128_parcellation,
    map_1020_to_4centers,
)

mne.set_log_level("ERROR")

BUCKET = "openneuro.org"
DEFAULT_OUT = Path("/media/beast/DATA1/autara_vox_datasets")


def make_s3():
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED, max_pool_connections=32),
    )


def download_file(s3, key: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        s3.download_file(BUCKET, key, str(tmp))
        tmp.replace(dst)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


def sos_bandpass(data: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    hi = min(high, nyq - 1.0)
    lo = max(low, 0.1)
    if lo >= hi:
        return data.astype(np.float32)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def sos_notch(data: np.ndarray, freq: float, fs: float, q: float = 30.0) -> np.ndarray:
    if freq >= 0.5 * fs:
        return data
    b, a = signal.iirnotch(freq, Q=q, fs=fs)
    return signal.filtfilt(b, a, data, axis=-1).astype(np.float32)


def session_zscore(arr: np.ndarray) -> np.ndarray:
    """Normalize across an entire session/run per channel (preserves task-level power shifts!)."""
    # arr shape: [..., C, T]
    axes = tuple(i for i in range(arr.ndim) if i != arr.ndim - 2)
    mean = np.mean(arr, axis=axes, keepdims=True)
    std = np.std(arr, axis=axes, keepdims=True)
    return np.clip((arr - mean) / (std + 1e-6), -6.0, 6.0).astype(np.float32)


# =============================================================================
# 1. ds003626: Thinking Out Loud (Inner Speech + Pronounced/Jaw + Visualized)
# =============================================================================
def build_ds003626(out_dir: Path, workers: int = 6):
    print("\n=== [1/3] Building ds003626 (Inner / Pronounced+Jaw / Visualized Speech) ===", flush=True)
    parc = build_biosemi128_parcellation()
    raw_dir = out_dir / "ds003626_raw"
    proc_dir = out_dir / "ds003626_proc"
    proc_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(f"sub-{s:02d}", f"ses-{se:02d}") for s in range(1, 11) for se in range(1, 4)]

    def fetch_and_proc(job: Tuple[str, str]):
        sub, ses = job
        s3 = make_s3()
        base = f"ds003626/derivatives/{sub}/{ses}/{sub}_{ses}"
        eeg_p = raw_dir / sub / ses / f"{sub}_{ses}_eeg-epo.fif"
        exg_p = raw_dir / sub / ses / f"{sub}_{ses}_exg-epo.fif"
        ev_p = raw_dir / sub / ses / f"{sub}_{ses}_events.dat"

        ok = (
            download_file(s3, f"{base}_eeg-epo.fif", eeg_p)
            and download_file(s3, f"{base}_exg-epo.fif", exg_p)
            and download_file(s3, f"{base}_events.dat", ev_p)
        )
        if not ok:
            return None

        try:
            eeg_epo = mne.read_epochs(str(eeg_p), preload=True, verbose=False)
            exg_epo = mne.read_epochs(str(exg_p), preload=True, verbose=False)
            ev = np.load(str(ev_p), allow_pickle=True)

            sfreq = float(eeg_epo.info["sfreq"])  # 256 Hz
            eeg_np = eeg_epo.get_data().astype(np.float32)  # [N, 128, 1153]
            exg_np = exg_epo.get_data().astype(np.float32)  # [N, 8, 1153]

            n = min(len(eeg_np), len(exg_np), len(ev))
            eeg_np, exg_np, ev = eeg_np[:n], exg_np[:n], ev[:n]

            # Active speech execution interval: [0.5s .. 2.5s] after cue (sample 128..640 @ 256 Hz)
            # Note: epochs in ds003626 start at t=0.0s (cue onset) -> action interval is [0.5s, 2.5s]
            i0 = int(0.5 * sfreq)
            i1 = i0 + int(2.0 * sfreq)  # 512 samples @ 256 Hz
            if i1 > eeg_np.shape[-1]:
                i0, i1 = 0, int(2.0 * sfreq)

            eeg_win = eeg_np[:, :, i0:i1]  # [N, 128, 512]
            exg_win = exg_np[:, :, i0:i1]  # [N, 8, 512]

            # Pure cortical EEG: 1-45 Hz + 50 Hz notch + CAR -> decimate 256->128 Hz (256 samples)
            eeg_filt = sos_notch(eeg_win, 50.0, sfreq)
            eeg_filt = sos_bandpass(eeg_filt, 1.0, 45.0, sfreq)
            eeg_filt = eeg_filt - eeg_filt.mean(axis=1, keepdims=True)
            eeg_128 = eeg_filt[:, :, ::2]  # [N, 128, 256]

            # Map 128 BioSemi channels to 4 canonical centers [N, 4, 16, 256]
            ch_map = {name: i for i, name in enumerate(eeg_epo.ch_names)}
            center_stack = []
            for c in CORTICAL_CENTERS:
                idxs = [ch_map.get(parc.channel_names[k], k) for k in parc.center_indices[c]]
                center_stack.append(eeg_128[:, idxs, :])
            pure_4c = np.stack(center_stack, axis=1)  # [N, 4, 16, 256]
            pure_4c = session_zscore(pure_4c)

            # Subvocal jaw/facial EXG: 50-120 Hz high-frequency myogenic band @ 256 Hz (512 samples)
            exg_hf = sos_notch(exg_win, 50.0, sfreq)
            exg_hf = sos_notch(exg_hf, 100.0, sfreq)
            exg_hf = sos_bandpass(exg_hf, 48.0, 120.0, sfreq)
            exg_hf = session_zscore(exg_hf)  # [N, 8, 512]

            word_y = ev[:, 1].astype(np.int64)
            cond_y = ev[:, 2].astype(np.int64)
            sub_id = np.full(n, int(sub.split("-")[1]), dtype=np.int64)
            ses_id = np.full(n, int(ses.split("-")[1]), dtype=np.int64)
            return pure_4c, exg_hf, word_y, cond_y, sub_id, ses_id
        except Exception as e:
            print(f"  [WARN] ds003626 {sub} {ses}: {e}", flush=True)
            return None

    all_pure, all_jaw, all_word, all_cond, all_sub, all_ses = [], [], [], [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(fetch_and_proc, jobs):
            if res is None:
                continue
            p, j, w, c, s, se = res
            all_pure.append(p)
            all_jaw.append(j)
            all_word.append(w)
            all_cond.append(c)
            all_sub.append(s)
            all_ses.append(se)
            print(f"  Loaded ds003626 sub-{s[0]:02d} ses-{se[0]:02d}: {len(w)} trials", flush=True)

    pure_arr = np.concatenate(all_pure, axis=0)
    jaw_arr = np.concatenate(all_jaw, axis=0)
    word_arr = np.concatenate(all_word, axis=0)
    cond_arr = np.concatenate(all_cond, axis=0)
    sub_arr = np.concatenate(all_sub, axis=0)
    ses_arr = np.concatenate(all_ses, axis=0)

    np.save(proc_dir / "pure_eeg.npy", pure_arr)
    np.save(proc_dir / "jaw_emg.npy", jaw_arr)
    np.save(proc_dir / "word_y.npy", word_arr)
    np.save(proc_dir / "cond_y.npy", cond_arr)
    np.save(proc_dir / "sub_id.npy", sub_arr)
    np.save(proc_dir / "ses_id.npy", ses_arr)

    meta = {
        "total_trials": int(len(word_arr)),
        "pure_eeg_shape": list(pure_arr.shape),
        "jaw_emg_shape": list(jaw_arr.shape),
        "subjects": sorted(list(set(sub_arr.tolist()))),
        "conditions": {"0": "Pronounced", "1": "Inner", "2": "Visualized"},
        "words": {"0": "Up", "1": "Down", "2": "Right", "3": "Left"},
    }
    (proc_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"ds003626 complete: {meta}", flush=True)


# =============================================================================
# 2. ds004408: Continuous Naturalistic English Speech + .TextGrid Words/Phonemes
# =============================================================================
def parse_textgrid(tg_text: str) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
    """Parse Praat TextGrid into (phones, words) lists of (xmin, xmax, label)."""
    tiers = tg_text.split('class = "IntervalTier"')[1:]
    phones: List[Tuple[float, float, str]] = []
    words: List[Tuple[float, float, str]] = []

    pat = re.compile(
        r"xmin\s*=\s*([0-9.eE+-]+)\s+xmax\s*=\s*([0-9.eE+-]+)\s+text\s*=\s*\"([^\"]*)\""
    )
    for tier in tiers:
        name_m = re.search(r'name\s*=\s*"([^"]+)"', tier)
        tname = name_m.group(1).lower() if name_m else ""
        items = [
            (float(a), float(b), txt.strip())
            for a, b, txt in pat.findall(tier)
            if txt.strip() and txt.strip().lower() not in ("sil", "sp", "")
        ]
        if "phone" in tname:
            phones = items
        elif "word" in tname:
            words = items
    return phones, words


def segment_textgrid_phrases(
    phones: List[Tuple[float, float, str]],
    words: List[Tuple[float, float, str]],
    window_sec: float = 3.0,
    stride_sec: float = 2.0,
) -> List[Dict]:
    """Group millisecond-aligned words & phonemes into natural English phrase windows."""
    if not words:
        return []
    max_t = words[-1][1]
    segments = []
    t = 0.0
    while t + window_sec <= max_t:
        t_end = t + window_sec
        w_in = [w[2].lower() for w in words if (0.5 * (w[0] + w[1])) >= t and (0.5 * (w[0] + w[1])) < t_end]
        p_in = [p[2].upper() for p in phones if (0.5 * (p[0] + p[1])) >= t and (0.5 * (p[0] + p[1])) < t_end]
        if len(w_in) >= 3 and len(p_in) >= 5:
            segments.append(
                {
                    "t_start": float(t),
                    "t_end": float(t_end),
                    "text": " ".join(w_in),
                    "phonemes": p_in,
                }
            )
        t += stride_sec
    return segments


def build_ds004408(out_dir: Path, num_subjects: int = 8, num_runs: int = 12, workers: int = 6):
    print(
        f"\n=== [2/3] Building ds004408 (Continuous Speech EEG + TextGrid | {num_subjects} subs x {num_runs} runs) ===",
        flush=True,
    )
    parc = build_biosemi128_parcellation()
    raw_dir = out_dir / "ds004408_raw"
    proc_dir = out_dir / "ds004408_proc"
    proc_dir.mkdir(parents=True, exist_ok=True)

    s3 = make_s3()
    # Download TextGrids first
    run_phrases: Dict[int, List[Dict]] = {}
    phoneme_vocab_set = set()
    for run in range(1, num_runs + 1):
        tg_p = raw_dir / "stimuli" / f"audio{run:02d}.TextGrid"
        if download_file(s3, f"ds004408/stimuli/audio{run:02d}.TextGrid", tg_p):
            phones, words = parse_textgrid(tg_p.read_text(errors="ignore"))
            segs = segment_textgrid_phrases(phones, words, window_sec=3.0, stride_sec=2.0)
            run_phrases[run] = segs
            for seg in segs:
                phoneme_vocab_set.update(seg["phonemes"])

    phoneme_list = ["<blank>"] + sorted(phoneme_vocab_set)
    phoneme_to_id = {p: i for i, p in enumerate(phoneme_list)}
    print(f"  Parsed {sum(len(v) for v in run_phrases.values())} unique phrase slots | Phoneme vocab={len(phoneme_list)}", flush=True)

    # Assign stable global phrase_id per (run, seg_idx)
    phrase_id_map: Dict[Tuple[int, int], int] = {}
    next_pid = 0
    for run in sorted(run_phrases.keys()):
        for si in range(len(run_phrases[run])):
            phrase_id_map[(run, si)] = next_pid
            next_pid += 1

    jobs = [(s, r) for s in range(1, num_subjects + 1) for r in range(1, num_runs + 1) if r in run_phrases]

    def fetch_and_proc_run(job: Tuple[int, int]):
        sub_num, run = job
        sub = f"sub-{sub_num:03d}"
        s3_loc = make_s3()
        base = f"ds004408/{sub}/eeg/{sub}_task-listening_run-{run:02d}_eeg"
        vhdr_p = raw_dir / sub / f"{sub}_run-{run:02d}_eeg.vhdr"
        vmrk_p = raw_dir / sub / f"{sub}_run-{run:02d}_eeg.vmrk"
        eeg_p = raw_dir / sub / f"{sub}_run-{run:02d}_eeg.eeg"

        ok = (
            download_file(s3_loc, f"{base}.vhdr", vhdr_p)
            and download_file(s3_loc, f"{base}.vmrk", vmrk_p)
            and download_file(s3_loc, f"{base}.eeg", eeg_p)
        )
        if not ok:
            return None

        # Fix internal filenames inside .vhdr if renamed
        vhdr_txt = vhdr_p.read_text(errors="ignore")
        vhdr_txt = re.sub(r"DataFile=.*", f"DataFile={eeg_p.name}", vhdr_txt)
        vhdr_txt = re.sub(r"MarkerFile=.*", f"MarkerFile={vmrk_p.name}", vhdr_txt)
        vhdr_p.write_text(vhdr_txt)

        try:
            raw = mne.io.read_raw_brainvision(str(vhdr_p), preload=True, verbose=False)
            sfreq = float(raw.info["sfreq"])  # 512 Hz
            data = raw.get_data()[:128].astype(np.float32)  # [128, T_512]
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # Stream A: Pure cortical EEG (1-45 Hz + 50 Hz notch + CAR -> decimate 512->128 Hz)
            pure = sos_notch(data, 50.0, sfreq)
            pure = sos_bandpass(pure, 1.0, 45.0, sfreq)
            pure = pure - pure.mean(axis=0, keepdims=True)
            pure_128 = pure[:, ::4]  # [128, T_128]
            ch_map = {name: i for i, name in enumerate(raw.ch_names[:128])}
            pure_4c = np.stack(
                [
                    pure_128[[ch_map.get(parc.channel_names[k], k) for k in parc.center_indices[c]]]
                    for c in CORTICAL_CENTERS
                ],
                axis=0,
            )  # [4, 16, T_128]
            pure_4c = session_zscore(pure_4c)

            # Stream B: High-Freq Jaw/Temporal Articulatory Ring (50-200 Hz Hilbert envelope -> 128 Hz)
            jaw_idxs = [ch_map.get(parc.channel_names[k], k) for k in parc.jaw_scalp_indices]
            jaw_raw = data[jaw_idxs]  # [16, T_512]
            jaw_hf = sos_notch(jaw_raw, 50.0, sfreq)
            jaw_hf = sos_notch(jaw_hf, 100.0, sfreq)
            jaw_hf = sos_notch(jaw_hf, 150.0, sfreq)
            jaw_hf = sos_bandpass(jaw_hf, 50.0, 200.0, sfreq)
            # Smooth rectified power envelope -> decimate 512->128 Hz (zero aliasing of 50-200 Hz EMG!)
            jaw_env = np.abs(jaw_hf)
            sos_lp = signal.butter(4, 40.0, btype="lowpass", fs=sfreq, output="sos")
            jaw_env = signal.sosfiltfilt(sos_lp, jaw_env, axis=-1).astype(np.float32)
            jaw_128 = session_zscore(jaw_env[:, ::4])  # [16, T_128]

            # Slice synchronized 3.0s phrase windows (384 samples @ 128 Hz)
            win_len = 384
            pure_wins, jaw_wins, records = [], [], []
            t_len = min(pure_4c.shape[-1], jaw_128.shape[-1])
            for si, seg in enumerate(run_phrases[run]):
                i0 = int(round(seg["t_start"] * 128.0))
                i1 = i0 + win_len
                if i1 > t_len:
                    break
                pure_wins.append(pure_4c[:, :, i0:i1])
                jaw_wins.append(jaw_128[:, i0:i1])
                records.append(
                    {
                        "sub_id": sub_num,
                        "run_id": run,
                        "phrase_id": phrase_id_map[(run, si)],
                        "text": seg["text"],
                        "phoneme_ids": [phoneme_to_id[p] for p in seg["phonemes"]],
                        "phonemes": seg["phonemes"],
                    }
                )
            if not pure_wins:
                return None
            return (
                sub_num,
                run,
                np.stack(pure_wins, axis=0).astype(np.float16),
                np.stack(jaw_wins, axis=0).astype(np.float16),
                records,
            )
        except Exception as e:
            print(f"  [WARN] ds004408 {sub} run-{run:02d}: {e}", flush=True)
            return None

    all_pure, all_jaw, all_records = [], [], []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(fetch_and_proc_run, jobs):
            done += 1
            if res is None:
                continue
            s_num, r_num, pw, jw, recs = res
            all_pure.append(pw)
            all_jaw.append(jw)
            all_records.extend(recs)
            if done % 10 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] ds004408 total phrase windows: {len(all_records)}", flush=True)

    pure_arr = np.concatenate(all_pure, axis=0)
    jaw_arr = np.concatenate(all_jaw, axis=0)
    np.save(proc_dir / "pure_eeg_fp16.npy", pure_arr)
    np.save(proc_dir / "jaw_emg_fp16.npy", jaw_arr)
    (proc_dir / "records.json").write_text(json.dumps(all_records))
    meta = {
        "total_windows": len(all_records),
        "pure_eeg_shape": list(pure_arr.shape),
        "jaw_emg_shape": list(jaw_arr.shape),
        "phoneme_vocab": phoneme_list,
        "num_subjects": num_subjects,
        "num_runs": num_runs,
    }
    (proc_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"ds004408 complete: windows={len(all_records)}, pure={pure_arr.shape}, jaw={jaw_arr.shape}", flush=True)


# =============================================================================
# 3. ds003655: Verbal Working Memory (Modified Sternberg 5/6/7 Letters)
# =============================================================================
def build_ds003655(out_dir: Path, num_subjects: int = 12, workers: int = 6):
    print(f"\n=== [3/3] Building ds003655 (Verbal Working Memory Center | {num_subjects} subs) ===", flush=True)
    raw_dir = out_dir / "ds003655_raw"
    proc_dir = out_dir / "ds003655_proc"
    proc_dir.mkdir(parents=True, exist_ok=True)

    def fetch_and_proc_sub(sub_num: int):
        sub = f"sub-{sub_num:03d}"
        s3 = make_s3()
        base = f"ds003655/{sub}/eeg/{sub}_task-VerbalWorkingMemory"
        set_p = raw_dir / sub / f"{sub}_task-VerbalWorkingMemory_eeg.set"
        fdt_p = raw_dir / sub / f"{sub}_task-VerbalWorkingMemory_eeg.fdt"
        ev_p = raw_dir / sub / f"{sub}_task-VerbalWorkingMemory_events.tsv"

        ok = (
            download_file(s3, f"{base}_eeg.set", set_p)
            and download_file(s3, f"{base}_eeg.fdt", fdt_p)
            and download_file(s3, f"{base}_events.tsv", ev_p)
        )
        if not ok:
            return None

        try:
            raw = mne.io.read_raw_eeglab(str(set_p), preload=True, verbose=False)
            sfreq = float(raw.info["sfreq"])
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            ch_names = [raw.ch_names[i] for i in eeg_picks]
            data = raw.get_data(picks=eeg_picks).astype(np.float32)

            data = sos_notch(data, 50.0, sfreq)
            data = sos_bandpass(data, 1.0, 45.0, sfreq)
            data = data - data.mean(axis=0, keepdims=True)

            # Resample to 128 Hz
            n_new = int(round(data.shape[1] * 128.0 / sfreq))
            t_old = np.linspace(0.0, 1.0, data.shape[1], endpoint=False)
            t_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
            data_128 = np.stack([np.interp(t_new, t_old, ch) for ch in data], axis=0).astype(np.float32)
            data_128 = session_zscore(data_128)

            c_map = map_1020_to_4centers(ch_names)
            # Tile each center's channels cleanly to 16 channels so it fits exact 4-center [4, 16, T]
            center_blocks = []
            for c in CORTICAL_CENTERS:
                idxs = c_map[c] if c_map[c] else [0]
                block = data_128[idxs]
                reps = int(np.ceil(16 / block.shape[0]))
                block16 = np.tile(block, (reps, 1))[:16]
                center_blocks.append(block16)
            data_4c = np.stack(center_blocks, axis=0)  # [4, 16, T_128]

            # Parse delay-period events: e.g. "delay period: 6M" or "delay period: 5R"
            wins, task_y, load_y = [], [], []
            lines = ev_p.read_text(errors="ignore").strip().splitlines()
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                ttype = parts[3]
                if "delay period:" not in ttype:
                    continue
                m = re.search(r"([567])([MR])", ttype)
                if not m:
                    continue
                load_cls = int(m.group(1)) - 5  # 0=5 letters, 1=6 letters, 2=7 letters
                manip_cls = 1 if m.group(2) == "M" else 0  # 0=Retention, 1=Manipulation
                onset_s = float(parts[0])
                i0 = int(round(onset_s * 128.0))
                i1 = i0 + 256  # 2.0s active memory delay
                if i1 <= data_4c.shape[-1]:
                    wins.append(data_4c[:, :, i0:i1])
                    task_y.append(manip_cls)
                    load_y.append(load_cls)

            if not wins:
                return None
            return (
                np.stack(wins, axis=0).astype(np.float32),
                np.array(task_y, dtype=np.int64),
                np.array(load_y, dtype=np.int64),
                np.full(len(wins), sub_num, dtype=np.int64),
            )
        except Exception as e:
            print(f"  [WARN] ds003655 {sub}: {e}", flush=True)
            return None

    all_w, all_t, all_l, all_s = [], [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(fetch_and_proc_sub, range(1, num_subjects + 1)):
            if res is None:
                continue
            w, t, l, s = res
            all_w.append(w)
            all_t.append(t)
            all_l.append(l)
            all_s.append(s)
            print(f"  Loaded ds003655 sub-{s[0]:03d}: {len(t)} memory trials", flush=True)

    w_arr = np.concatenate(all_w, axis=0)
    t_arr = np.concatenate(all_t, axis=0)
    l_arr = np.concatenate(all_l, axis=0)
    s_arr = np.concatenate(all_s, axis=0)

    np.save(proc_dir / "pure_eeg.npy", w_arr)
    np.save(proc_dir / "task_y.npy", t_arr)
    np.save(proc_dir / "load_y.npy", l_arr)
    np.save(proc_dir / "sub_id.npy", s_arr)
    meta = {
        "total_trials": int(len(t_arr)),
        "shape": list(w_arr.shape),
        "task_classes": {"0": "Retention", "1": "Manipulation (Alphabetization)"},
        "load_classes": {"0": "5 letters", "1": "6 letters", "2": "7 letters"},
    }
    (proc_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"ds003655 complete: {meta}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ds4408-subs", type=int, default=8)
    parser.add_argument("--ds4408-runs", type=int, default=12)
    parser.add_argument("--ds3655-subs", type=int, default=12)
    parser.add_argument("--only", type=str, default="all", choices=["all", "ds003626", "ds004408", "ds003655"])
    args = parser.parse_args()

    t0 = time.time()
    if args.only in ("all", "ds003626"):
        build_ds003626(args.out_dir, workers=args.workers)
    if args.only in ("all", "ds004408"):
        build_ds004408(args.out_dir, num_subjects=args.ds4408_subs, num_runs=args.ds4408_runs, workers=args.workers)
    if args.only in ("all", "ds003655"):
        build_ds003655(args.out_dir, num_subjects=args.ds3655_subs, workers=args.workers)
    print(f"\nALL REQUESTED DATASETS BUILT IN {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
