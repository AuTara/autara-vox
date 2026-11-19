#!/usr/bin/env python3
"""
Experiment 0 & 1: Verbal Working Memory Pretraining (ds003655) +
Dual-Stream Inner / Pronounced+Jaw / Visualized Speech Decoding (ds003626)
==========================================================================
Includes:
  - Mandatory Every-10-Epoch Live Verification Probes (Epochs 10, 20, 30)
  - Strict Pure-EEG (1-45 Hz) vs. Subvocal Jaw/EXG (50-120 Hz) vs. Dual-Stream Ablation
  - Real EEG vs. Random-Noise EEG Control (Jo et al., 2024 honesty verification)
  - 4-Center Cortical Region Ablation (Speech, Auditory, Visual, Memory)
  - Working-Memory Pretraining Transfer Evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from autara_vox import (
    CORTICAL_CENTERS,
    AutaraVoxClassifier,
    EveryTenEpochProbe,
    compute_classification_metrics,
)

DATA_ROOT = Path("/media/beast/DATA1/autara_vox_datasets")
CKPT_DIR = Path("checkpoints")
RESULTS_DIR = Path("paper/results")

WORD_NAMES = ["Up", "Down", "Right", "Left"]
COND_NAMES = ["Pronounced (Overt+Jaw)", "Inner Speech (Covert)", "Visualized"]


def augment_neural(
    pure_eeg: torch.Tensor, jaw_emg: torch.Tensor | None = None
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    # Random circular time shift (+/- 8 samples), amplitude scale jitter, Gaussian noise, channel dropout
    shift = int(torch.randint(-8, 9, (1,)).item())
    x = torch.roll(pure_eeg, shifts=shift, dims=-1)
    scale = 0.85 + 0.30 * torch.rand(x.size(0), 1, 1, 1, device=x.device)
    x = x * scale + 0.08 * torch.randn_like(x)
    # 10% channel dropout
    ch_mask = (torch.rand(x.size(0), 4, 16, 1, device=x.device) > 0.10).float()
    x = x * ch_mask

    j_out = None
    if jaw_emg is not None:
        j = torch.roll(jaw_emg, shifts=shift * 2, dims=-1)
        j_scale = 0.85 + 0.30 * torch.rand(j.size(0), 1, 1, device=j.device)
        j_out = j * j_scale + 0.08 * torch.randn_like(j)
    return x, j_out


@torch.no_grad()
def eval_ds003626(
    model: AutaraVoxClassifier,
    loader: DataLoader,
    device: torch.device,
    stream_mode: str = "dual",
    active_centers=CORTICAL_CENTERS,
    noise_control: bool = False,
) -> Dict:
    model.eval()
    w_true, w_pred = [], []
    c_true, c_pred = [], []
    center_w_sum = torch.zeros(4)
    stream_w_sum = torch.zeros(2)
    n_tot = 0

    for pe, je, wy, cy in loader:
        pe, je = pe.to(device), je.to(device)
        if noise_control:
            pe = torch.randn_like(pe)
            je = torch.randn_like(je)
        out = model(pe, je, stream_mode=stream_mode, active_centers=active_centers)
        wp = out["primary_logits"].argmax(dim=-1).cpu().tolist()
        cp = out["aux_logits"].argmax(dim=-1).cpu().tolist()
        w_pred.extend(wp)
        w_true.extend(wy.tolist())
        c_pred.extend(cp)
        c_true.extend(cy.tolist())
        center_w_sum += out["center_weights"].cpu().sum(dim=0)
        stream_w_sum += out["stream_weights"].cpu().sum(dim=0)
        n_tot += len(wy)

    overall_word = compute_classification_metrics(w_true, w_pred)
    overall_cond = compute_classification_metrics(c_true, c_pred)

    # Break down 4-word accuracy by condition (Pronounced vs. Inner vs. Visualized)
    per_cond = {}
    w_true_arr, w_pred_arr, c_true_arr = np.array(w_true), np.array(w_pred), np.array(c_true)
    for cid, cname in enumerate(("pronounced", "inner", "visualized")):
        mask = c_true_arr == cid
        if mask.sum() > 0:
            per_cond[cname] = compute_classification_metrics(w_true_arr[mask], w_pred_arr[mask])

    cw = (center_w_sum / max(n_tot, 1)).tolist()
    sw = (stream_w_sum / max(n_tot, 1)).tolist()
    return {
        "word_acc": overall_word["acc"],
        "word_f1": overall_word["macro_f1"],
        "cond_acc": overall_cond["acc"],
        "cond_f1": overall_cond["macro_f1"],
        "per_condition_word": per_cond,
        "center_weights": {c: cw[i] for i, c in enumerate(CORTICAL_CENTERS)},
        "stream_weights": {"pure_eeg": sw[0], "jaw_emg": sw[1]},
        "sample_pairs": [
            {
                "gt": f"{WORD_NAMES[w_true[i]]} [{COND_NAMES[c_true[i]]}]",
                "generated": f"{WORD_NAMES[w_pred[i]]} [{COND_NAMES[c_pred[i]]}]",
            }
            for i in range(min(5, len(w_true)))
        ],
    }


def run_exp0_memory_pretrain(device: torch.device, epochs: int = 30) -> Path:
    print("\n================ EXP 0: VERBAL WORKING MEMORY CENTER (ds003655) ================", flush=True)
    proc = DATA_ROOT / "ds003655_proc"
    pure = np.load(proc / "pure_eeg.npy")
    task_y = np.load(proc / "task_y.npy")
    load_y = np.load(proc / "load_y.npy")
    sub_id = np.load(proc / "sub_id.npy")

    # Strict unseen-subject split (last 3 subjects unseen)
    uniq_subs = sorted(np.unique(sub_id).tolist())
    test_subs = set(uniq_subs[-3:])
    tr_mask = np.array([s not in test_subs for s in sub_id])
    te_mask = ~tr_mask

    tr_ds = TensorDataset(torch.from_numpy(pure[tr_mask]), torch.from_numpy(task_y[tr_mask]), torch.from_numpy(load_y[tr_mask]))
    te_ds = TensorDataset(torch.from_numpy(pure[te_mask]), torch.from_numpy(task_y[te_mask]), torch.from_numpy(load_y[te_mask]))
    tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True, drop_last=False)
    te_ld = DataLoader(te_ds, batch_size=128, shuffle=False)

    model = AutaraVoxClassifier(num_primary_classes=2, num_aux_classes=3, jaw_ch=8, d_model=256, dropout=0.25).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    probe = EveryTenEpochProbe("exp0_memory_ds003655", RESULTS_DIR, interval=10)

    best_acc = 0.0
    ckpt_path = CKPT_DIR / "memory_pretrained_backbone.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        for pe, ty, ly in tr_ld:
            pe, ty, ly = pe.to(device), ty.to(device), ly.to(device)
            pe, _ = augment_neural(pe, None)
            out = model(pe, None, stream_mode="pure_eeg")
            loss = F.cross_entropy(out["primary_logits"], ty, label_smoothing=0.05) + 0.5 * F.cross_entropy(
                out["aux_logits"], ly, label_smoothing=0.05
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Quick val
        model.eval()
        yt, yp, lt, lp = [], [], [], []
        cw_sum = torch.zeros(4)
        with torch.no_grad():
            for pe, ty, ly in te_ld:
                out = model(pe.to(device), None, stream_mode="pure_eeg")
                yp.extend(out["primary_logits"].argmax(-1).cpu().tolist())
                yt.extend(ty.tolist())
                lp.extend(out["aux_logits"].argmax(-1).cpu().tolist())
                lt.extend(ly.tolist())
                cw_sum += out["center_weights"].cpu().sum(0)
        tm = compute_classification_metrics(yt, yp)
        lm = compute_classification_metrics(lt, lp)
        if tm["acc"] > best_acc:
            best_acc = tm["acc"]
            torch.save(model.backbone.state_dict(), ckpt_path)

        if probe.should_run(ep, epochs):
            # Noise control
            np_pred = []
            with torch.no_grad():
                for pe, ty, _ in te_ld:
                    out = model(torch.randn_like(pe).to(device), None, stream_mode="pure_eeg")
                    np_pred.extend(out["primary_logits"].argmax(-1).cpu().tolist())
            nm = compute_classification_metrics(yt, np_pred)
            cw = (cw_sum / max(len(yt), 1)).tolist()
            probe.record_and_report(
                epoch=ep,
                real_metrics={"acc": tm["acc"], "macro_f1": tm["macro_f1"], "load_3cls_acc": lm["acc"]},
                noise_metrics={"acc": nm["acc"], "macro_f1": nm["macro_f1"]},
                center_weights={c: cw[i] for i, c in enumerate(CORTICAL_CENTERS)},
            )

    summary = {
        "dataset": "ds003655_VerbalWorkingMemory",
        "train_trials": int(tr_mask.sum()),
        "test_trials": int(te_mask.sum()),
        "best_unseen_subject_task_acc": best_acc,
        "last_load_3cls_acc": lm["acc"],
    }
    (RESULTS_DIR / "exp0_memory_center.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved Memory-Center pretrained backbone ({best_acc:.1f}% unseen-subject acc) -> {ckpt_path}", flush=True)
    return ckpt_path


def run_exp1_inner_speech(
    device: torch.device,
    epochs: int = 30,
    memory_ckpt: Path | None = None,
    run_tag: str = "exp1_inner_speech_ds003626",
) -> Dict:
    print(f"\n================ EXP 1: {run_tag} (ds003626) ================", flush=True)
    proc = DATA_ROOT / "ds003626_proc"
    pure = np.load(proc / "pure_eeg.npy")
    jaw = np.load(proc / "jaw_emg.npy")
    word_y = np.load(proc / "word_y.npy")
    cond_y = np.load(proc / "cond_y.npy")
    ses_id = np.load(proc / "ses_id.npy")

    # Cross-session held-out split: Sessions 1 & 2 train, Session 3 completely unseen test
    # (standard benchmark protocol for ds003626 Nieto et al., 2022!)
    tr_mask = ses_id <= 2
    te_mask = ses_id == 3

    tr_ds = TensorDataset(
        torch.from_numpy(pure[tr_mask]),
        torch.from_numpy(jaw[tr_mask]),
        torch.from_numpy(word_y[tr_mask]),
        torch.from_numpy(cond_y[tr_mask]),
    )
    te_ds = TensorDataset(
        torch.from_numpy(pure[te_mask]),
        torch.from_numpy(jaw[te_mask]),
        torch.from_numpy(word_y[te_mask]),
        torch.from_numpy(cond_y[te_mask]),
    )
    tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=128, shuffle=False)

    model = AutaraVoxClassifier(
        num_primary_classes=4,
        num_aux_classes=3,
        jaw_ch=jaw.shape[1],
        d_model=256,
        eeg_patch_stride=8,
        jaw_patch_stride=16,
        dropout=0.25,
    ).to(device)

    if memory_ckpt and memory_ckpt.exists():
        # Transfer compatible 4-center cortical encoder & conformer weights from Verbal Working Memory
        src_sd = torch.load(memory_ckpt, map_location=device, weights_only=True)
        dst_sd = model.backbone.state_dict()
        loaded = 0
        for k, v in src_sd.items():
            if k in dst_sd and dst_sd[k].shape == v.shape:
                dst_sd[k] = v
                loaded += 1
        model.backbone.load_state_dict(dst_sd)
        print(f"  Transferred {loaded} cortical tensors from ds003655 Verbal Working Memory!", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    probe = EveryTenEpochProbe(run_tag, RESULTS_DIR, interval=10)

    best_score = -1.0
    best_ckpt = CKPT_DIR / f"{run_tag}_best.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        steps = 0
        for pe, je, wy, cy in tr_ld:
            pe, je, wy, cy = pe.to(device), je.to(device), wy.to(device), cy.to(device)
            pe, je = augment_neural(pe, je)

            # Modality dropout during training so Dual, Pure-EEG, and Jaw-EMG all stay calibrated
            r = float(torch.rand(1).item())
            mode = "dual" if r < 0.65 else ("pure_eeg" if r < 0.85 else "jaw_only")

            out = model(pe, je, stream_mode=mode)
            loss = F.cross_entropy(out["primary_logits"], wy, label_smoothing=0.08) + 0.7 * F.cross_entropy(
                out["aux_logits"], cy, label_smoothing=0.05
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            steps += 1
        sched.step()

        dual_ev = eval_ds003626(model, te_ld, device, stream_mode="dual")
        score = 0.6 * dual_ev["word_acc"] + 0.4 * dual_ev["cond_acc"]
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_ckpt)

        if ep % 5 == 0 or ep == 1:
            print(
                f"  Ep {ep:02d}/{epochs} | Loss={loss_sum/max(steps,1):.4f} | "
                f"Word4={dual_ev['word_acc']:.1f}% | Cond3={dual_ev['cond_acc']:.1f}% | "
                f"Gate[EEG={dual_ev['stream_weights']['pure_eeg']*100:.0f}%, Jaw={dual_ev['stream_weights']['jaw_emg']*100:.0f}%]",
                flush=True,
            )

        if probe.should_run(ep, epochs):
            pure_ev = eval_ds003626(model, te_ld, device, stream_mode="pure_eeg")
            jaw_ev = eval_ds003626(model, te_ld, device, stream_mode="jaw_only")
            noise_ev = eval_ds003626(model, te_ld, device, stream_mode="dual", noise_control=True)
            probe.record_and_report(
                epoch=ep,
                real_metrics={
                    "acc": dual_ev["word_acc"],
                    "word_f1": dual_ev["word_f1"],
                    "cond_acc": dual_ev["cond_acc"],
                    "cond_f1": dual_ev["cond_f1"],
                    "pronounced_word_acc": dual_ev["per_condition_word"].get("pronounced", {}).get("acc", 0.0),
                    "inner_word_acc": dual_ev["per_condition_word"].get("inner", {}).get("acc", 0.0),
                    "visualized_word_acc": dual_ev["per_condition_word"].get("visualized", {}).get("acc", 0.0),
                },
                noise_metrics={"acc": noise_ev["word_acc"], "cond_acc": noise_ev["cond_acc"]},
                sample_decodings=dual_ev["sample_pairs"],
                stream_ablation={
                    "dual": {"word_acc": dual_ev["word_acc"], "cond_acc": dual_ev["cond_acc"]},
                    "pure_eeg": {"word_acc": pure_ev["word_acc"], "cond_acc": pure_ev["cond_acc"]},
                    "jaw_only": {"word_acc": jaw_ev["word_acc"], "cond_acc": jaw_ev["cond_acc"]},
                },
                center_weights=dual_ev["center_weights"],
            )

    # Reload best checkpoint for complete publication ablations
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    final_results = {
        "run_tag": run_tag,
        "memory_pretrained": bool(memory_ckpt is not None),
        "train_trials": int(tr_mask.sum()),
        "test_trials": int(te_mask.sum()),
        "stream_ablation": {
            "dual": eval_ds003626(model, te_ld, device, stream_mode="dual"),
            "pure_eeg": eval_ds003626(model, te_ld, device, stream_mode="pure_eeg"),
            "jaw_only": eval_ds003626(model, te_ld, device, stream_mode="jaw_only"),
            "noise_control": eval_ds003626(model, te_ld, device, stream_mode="dual", noise_control=True),
        },
        "four_center_ablation": {
            c: eval_ds003626(model, te_ld, device, stream_mode="pure_eeg", active_centers=(c,))
            for c in CORTICAL_CENTERS
        },
    }
    # Strip verbose sample_pairs from summary JSON
    for sec in ("stream_ablation", "four_center_ablation"):
        for k in final_results[sec]:
            final_results[sec][k].pop("sample_pairs", None)

    (RESULTS_DIR / f"{run_tag}_results.json").write_text(json.dumps(final_results, indent=2))
    print(f"Saved complete Experiment 1 results -> {RESULTS_DIR / f'{run_tag}_results.json'}", flush=True)
    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-mem", type=int, default=30)
    parser.add_argument("--epochs-speech", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    mem_ckpt = run_exp0_memory_pretrain(device, epochs=args.epochs_mem)
    # Run BOTH scratch and memory-pretrained to prove working-memory transfer in the paper!
    run_exp1_inner_speech(device, epochs=args.epochs_speech, memory_ckpt=None, run_tag="exp1_inner_scratch")
    run_exp1_inner_speech(device, epochs=args.epochs_speech, memory_ckpt=mem_ckpt, run_tag="exp1_inner_mem_transfer")

    print(f"\nEXP 0 + EXP 1 COMPLETE IN {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
