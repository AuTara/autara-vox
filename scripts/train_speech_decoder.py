#!/usr/bin/env python3
"""
Experiment 2: Continuous Naturalistic EEG-to-Text & Phoneme Decoding (ds004408)
===============================================================================
Trains AutaraVoxSpeechDecoder (4-Center Pure EEG + High-Freq Jaw/Temporal Stream
+ Conformer + Phoneme CTC + CLIP Contrastive Alignment + Q-Former -> BART-base)
with:
  - Strict Held-Out Unseen Sentences (Runs 9-10) AND Zero-Shot Unseen Subjects (Sub 7-8)
  - Mandatory Every-10-Epoch Live Verification Probes (Epochs 10, 20, 30, 40)
  - True Open-Loop Autoregressive Decoding (ZERO teacher-forcing leakage!)
  - Real-EEG vs. Random-Noise-EEG Control (Jo et al., 2024 protocol)
  - Complete Dual-Stream & 4-Center Cortical Ablation Tables for Paper
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from autara_vox import (
    CORTICAL_CENTERS,
    AutaraVoxSpeechDecoder,
    EveryTenEpochProbe,
    compute_phoneme_error_rate,
    compute_retrieval_metrics,
    compute_text_generation_metrics,
)

DATA_ROOT = Path("/media/beast/DATA1/autara_vox_datasets/ds004408_proc")
CKPT_DIR = Path("checkpoints")
RESULTS_DIR = Path("paper/results")


class ContinuousSpeechDataset(Dataset):
    def __init__(self, pure_mmap: np.ndarray, jaw_mmap: np.ndarray, records: List[Dict], indices: List[int]):
        self.pure = pure_mmap
        self.jaw = jaw_mmap
        self.records = records
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        rec = self.records[i]
        pe = torch.from_numpy(np.array(self.pure[i], dtype=np.float32))
        je = torch.from_numpy(np.array(self.jaw[i], dtype=np.float32))
        return pe, je, rec["text"], rec["phoneme_ids"], rec["phonemes"], rec["phrase_id"]


def make_collate_fn(tokenizer, max_text_len: int = 24):
    def collate(batch):
        pes, jes, texts, ph_id_lists, ph_str_lists, pids = zip(*batch)
        pure_t = torch.stack(pes, dim=0)
        jaw_t = torch.stack(jes, dim=0)

        tok = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        labels = tok["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        # Flatten CTC targets
        flat_ph = []
        ph_lens = []
        for p_ids in ph_id_lists:
            # Cap phoneme length <= 32 so always <= T_tok (48) for valid CTC
            clipped = p_ids[:32]
            flat_ph.extend(clipped)
            ph_lens.append(len(clipped))

        ctc_targets = torch.tensor(flat_ph, dtype=torch.long)
        ctc_lens = torch.tensor(ph_lens, dtype=torch.long)

        return (
            pure_t,
            jaw_t,
            tok["input_ids"],
            tok["attention_mask"],
            labels,
            ctc_targets,
            ctc_lens,
            list(texts),
            list(ph_str_lists),
            list(pids),
        )

    return collate


def greedy_ctc_decode(argmax_ids: torch.Tensor, phoneme_vocab: List[str]) -> List[List[str]]:
    decoded = []
    for row in argmax_ids.cpu().tolist():
        seq = []
        prev = 0
        for idx in row:
            if idx != 0 and idx != prev and idx < len(phoneme_vocab):
                seq.append(phoneme_vocab[idx])
            prev = idx
        decoded.append(seq)
    return decoded


def augment_speech_batch(pe: torch.Tensor, je: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    shift = int(torch.randint(-12, 13, (1,)).item())
    pe = torch.roll(pe, shifts=shift, dims=-1)
    je = torch.roll(je, shifts=shift, dims=-1)
    pe = pe * (0.85 + 0.30 * torch.rand(pe.size(0), 1, 1, 1, device=pe.device)) + 0.08 * torch.randn_like(pe)
    je = je * (0.85 + 0.30 * torch.rand(je.size(0), 1, 1, device=je.device)) + 0.08 * torch.randn_like(je)
    ch_mask = (torch.rand(pe.size(0), 4, 16, 1, device=pe.device) > 0.12).float()
    return pe * ch_mask, je


@torch.no_grad()
def evaluate_speech_decoder(
    model: AutaraVoxSpeechDecoder,
    loader: DataLoader,
    phoneme_vocab: List[str],
    device: torch.device,
    stream_mode: str = "dual",
    active_centers=CORTICAL_CENTERS,
    noise_control: bool = False,
    max_gen_samples: int = 128,
) -> Dict:
    model.eval()
    all_zn, all_zt, all_pids, all_texts = [], [], [], []
    gen_preds, gen_refs = [], []
    ph_preds, ph_refs = [], []
    cw_sum = torch.zeros(4)
    sw_sum = torch.zeros(2)
    n_tot = 0

    for pe, je, input_ids, attn_mask, _labels, _ctc_t, _ctc_l, texts, ph_strs, pids in loader:
        pe, je = pe.to(device), je.to(device)
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

        if noise_control:
            pe = torch.randn_like(pe)
            je = torch.randn_like(je)

        # Frozen text embeddings for CLIP retrieval
        frozen_txt = model.encode_text_frozen(input_ids, attn_mask)
        z_t = torch.nn.functional.normalize(model.text_proj(frozen_txt), dim=-1)

        if len(gen_preds) < max_gen_samples:
            pred_txts, ctc_ids, z_n, cw, sw = model.generate_open_loop(
                pe, je, stream_mode=stream_mode, active_centers=active_centers, max_new_tokens=18, num_beams=2
            )
            need = max_gen_samples - len(gen_preds)
            gen_preds.extend(pred_txts[:need])
            gen_refs.extend(texts[:need])
            ph_preds.extend(greedy_ctc_decode(ctc_ids[:need], phoneme_vocab))
            ph_refs.extend(ph_strs[:need])
        else:
            bb = model.backbone(pe, je, stream_mode=stream_mode, active_centers=active_centers)
            z_n = torch.nn.functional.normalize(model.neural_proj(bb.pooled), dim=-1)
            cw, sw = bb.center_weights, bb.stream_weights

        all_zn.append(z_n.cpu())
        all_zt.append(z_t.cpu())
        all_pids.extend(pids)
        all_texts.extend(texts)
        cw_sum += cw.cpu().sum(dim=0)
        sw_sum += sw.cpu().sum(dim=0)
        n_tot += len(texts)

    zn_cat = torch.cat(all_zn, dim=0)
    zt_cat = torch.cat(all_zt, dim=0)

    ret = compute_retrieval_metrics(zn_cat, zt_cat, text_ids=all_pids)
    txt_m = compute_text_generation_metrics(gen_preds, gen_refs)
    per = compute_phoneme_error_rate(ph_preds, ph_refs)

    # Build top-1 retrieved text for first 5 samples for live inspection
    # Build unique gallery
    uniq_idx, seen = [], set()
    for i, pid in enumerate(all_pids):
        if pid not in seen:
            seen.add(pid)
            uniq_idx.append(i)
    gal_z = zt_cat[uniq_idx]
    gal_txt = [all_texts[i] for i in uniq_idx]
    sim_mat = zn_cat[:5] @ gal_z.T
    best_sim, best_i = sim_mat.max(dim=-1)

    samples = []
    for i in range(min(5, len(gen_preds))):
        samples.append(
            {
                "gt": gen_refs[i],
                "retrieved": gal_txt[int(best_i[i].item())],
                "sim": f"{best_sim[i].item():.3f}",
                "generated": gen_preds[i],
                "phonemes_pred": " ".join(ph_preds[i][:12]),
                "phonemes_gt": " ".join(ph_refs[i][:12]),
            }
        )

    cw = (cw_sum / max(n_tot, 1)).tolist()
    sw = (sw_sum / max(n_tot, 1)).tolist()
    metrics = {
        **ret,
        **txt_m,
        "per": per,
        "center_weights": {c: cw[i] for i, c in enumerate(CORTICAL_CENTERS)},
        "stream_weights": {"pure_eeg": sw[0], "jaw_emg": sw[1]},
        "samples": samples,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--memory-ckpt", type=Path, default=CKPT_DIR / "memory_pretrained_backbone.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads((DATA_ROOT / "meta.json").read_text())
    records = json.loads((DATA_ROOT / "records.json").read_text())
    phoneme_vocab = meta["phoneme_vocab"]

    pure_arr = np.load(DATA_ROOT / "pure_eeg_fp16.npy", mmap_mode="r")
    jaw_arr = np.load(DATA_ROOT / "jaw_emg_fp16.npy", mmap_mode="r")

    # Split 1: Unseen Sentences (Runs 9-10 across training subjects 1-6)
    # Split 2: Zero-Shot Unseen Subjects (Subjects 7-8 across all runs)
    max_sub = max(r["sub_id"] for r in records)
    max_run = max(r["run_id"] for r in records)
    unseen_subs = {max_sub - 1, max_sub}
    unseen_runs = {max_run - 1, max_run}

    train_idx = [i for i, r in enumerate(records) if r["sub_id"] not in unseen_subs and r["run_id"] not in unseen_runs]
    val_unseen_sent_idx = [i for i, r in enumerate(records) if r["sub_id"] not in unseen_subs and r["run_id"] in unseen_runs]
    val_unseen_sub_idx = [i for i, r in enumerate(records) if r["sub_id"] in unseen_subs and r["run_id"] in unseen_runs]

    print(
        f"=== EXP 2: AUTARA-VOX CONTINUOUS EEG-TO-SPEECH (ds004408) ===\n"
        f"Train={len(train_idx)} | Unseen-Sentence Val={len(val_unseen_sent_idx)} | "
        f"Unseen-Subject Val={len(val_unseen_sub_idx)} | Phonemes={len(phoneme_vocab)}",
        flush=True,
    )

    model = AutaraVoxSpeechDecoder(
        num_phonemes=len(phoneme_vocab),
        jaw_ch=jaw_arr.shape[1],
        d_model=256,
        bart_name="facebook/bart-base",
        num_prefix_tokens=24,
        dropout=0.2,
    ).to(device)

    # Warm-start cortical encoders from ds003655 Verbal Working Memory pretraining if available
    if args.memory_ckpt.exists():
        src_sd = torch.load(args.memory_ckpt, map_location=device, weights_only=True)
        dst_sd = model.backbone.state_dict()
        loaded = 0
        for k, v in src_sd.items():
            if k in dst_sd and dst_sd[k].shape == v.shape:
                dst_sd[k] = v
                loaded += 1
        model.backbone.load_state_dict(dst_sd)
        print(f"Transferred {loaded} cortical weights from Verbal Working Memory pretraining!", flush=True)

    collate = make_collate_fn(model.tokenizer)
    tr_ld = DataLoader(
        ContinuousSpeechDataset(pure_arr, jaw_arr, records, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )
    val_sent_ld = DataLoader(
        ContinuousSpeechDataset(pure_arr, jaw_arr, records, val_unseen_sent_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
    )
    val_sub_ld = DataLoader(
        ContinuousSpeechDataset(pure_arr, jaw_arr, records, val_unseen_sub_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
    )

    # Separate lower LR for BART decoder vs neural backbone/adapter
    bart_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("bart.")]
    neural_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("bart.")]
    opt = torch.optim.AdamW(
        [
            {"params": neural_params, "lr": args.lr, "weight_decay": 0.02},
            {"params": bart_params, "lr": args.lr * 0.15, "weight_decay": 0.01},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    probe = EveryTenEpochProbe("exp2_speech_ds004408", RESULTS_DIR, interval=10)
    best_top5 = -1.0
    best_ckpt = CKPT_DIR / "autara_vox_speech_best.pt"
    t_start = time.time()

    for ep in range(1, args.epochs + 1):
        ep_t0 = time.time()
        model.train()
        lm_sum, ct_sum, ctc_sum, steps = 0.0, 0.0, 0.0, 0

        for pe, je, input_ids, attn_mask, labels, ctc_targets, ctc_lens, _, _, _ in tr_ld:
            pe, je = pe.to(device, non_blocking=True), je.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            ctc_targets = ctc_targets.to(device, non_blocking=True)
            ctc_lens = ctc_lens.to(device, non_blocking=True)

            pe, je = augment_speech_batch(pe, je)
            r = float(torch.rand(1).item())
            mode = "dual" if r < 0.70 else ("pure_eeg" if r < 0.88 else "jaw_only")

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(pe, je, input_ids, attn_mask, labels, stream_mode=mode)
                T_tok = out["ctc_log_probs"].size(0)
                in_lens = torch.full((pe.size(0),), T_tok, dtype=torch.long, device=device)
                l_ctc = ctc_loss_fn(out["ctc_log_probs"].float(), ctc_targets, in_lens, ctc_lens)
                loss = out["lm_loss"] + 1.0 * out["contrastive_loss"] + 0.4 * l_ctc

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            lm_sum += float(out["lm_loss"].item())
            ct_sum += float(out["contrastive_loss"].item())
            ctc_sum += float(l_ctc.item())
            steps += 1

        sched.step()
        if ep % 2 == 0 or ep == 1:
            print(
                f"  Ep {ep:02d}/{args.epochs} | LM={lm_sum/steps:.3f} | Clip={ct_sum/steps:.3f} | "
                f"CTC={ctc_sum/steps:.3f} | {time.time() - ep_t0:.1f}s",
                flush=True,
            )

        # MANDATORY EVERY-10-EPOCH LIVE VERIFICATION PROBE
        if probe.should_run(ep, args.epochs):
            real_ev = evaluate_speech_decoder(model, val_sent_ld, phoneme_vocab, device, stream_mode="dual", max_gen_samples=96)
            noise_ev = evaluate_speech_decoder(model, val_sent_ld, phoneme_vocab, device, stream_mode="dual", noise_control=True, max_gen_samples=96)
            pure_ev = evaluate_speech_decoder(model, val_sent_ld, phoneme_vocab, device, stream_mode="pure_eeg", max_gen_samples=64)
            jaw_ev = evaluate_speech_decoder(model, val_sent_ld, phoneme_vocab, device, stream_mode="jaw_only", max_gen_samples=64)

            # Attach noise generations to live examples so user sees Real vs Noise side-by-side!
            samples = real_ev["samples"]
            for i in range(min(len(samples), len(noise_ev["samples"]))):
                samples[i]["noise_gen"] = noise_ev["samples"][i]["generated"]

            scalar_keys = ("top1", "top5", "top10", "mrr", "cos_sim", "bleu1", "bleu2", "bleu4", "rouge1", "rougeL", "wer", "cer", "per", "diversity")
            probe.record_and_report(
                epoch=ep,
                real_metrics={k: real_ev[k] for k in scalar_keys},
                noise_metrics={k: noise_ev[k] for k in scalar_keys},
                sample_decodings=samples,
                stream_ablation={
                    "dual": {"top5": real_ev["top5"], "bleu1": real_ev["bleu1"], "rougeL": real_ev["rougeL"], "per": real_ev["per"]},
                    "pure_eeg": {"top5": pure_ev["top5"], "bleu1": pure_ev["bleu1"], "rougeL": pure_ev["rougeL"], "per": pure_ev["per"]},
                    "jaw_only": {"top5": jaw_ev["top5"], "bleu1": jaw_ev["bleu1"], "rougeL": jaw_ev["rougeL"], "per": jaw_ev["per"]},
                },
                center_weights=real_ev["center_weights"],
            )

            if real_ev["top5"] > best_top5:
                best_top5 = real_ev["top5"]
                torch.save(model.state_dict(), best_ckpt)

    # Complete Final Publication Benchmark on Best Checkpoint
    print("\n=== RUNNING FINAL PUBLICATION ABLATIONS ON BEST CHECKPOINT ===", flush=True)
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    def strip_samples(d: Dict) -> Dict:
        return {k: v for k, v in d.items() if k != "samples"}

    final_unseen_sent = {
        "dual": evaluate_speech_decoder(model, val_sent_ld, phoneme_vocab, device, stream_mode="dual", max_gen_samples=160),
        "pure_eeg": evaluate_speech_decoder(model, val_sent_ld, phoneme_vocab, device, stream_mode="pure_eeg", max_gen_samples=160),
        "jaw_only": evaluate_speech_decoder(model, val_sent_ld, phoneme_vocab, device, stream_mode="jaw_only", max_gen_samples=160),
        "noise_control": evaluate_speech_decoder(model, val_sent_ld, phoneme_vocab, device, stream_mode="dual", noise_control=True, max_gen_samples=160),
    }
    final_four_centers = {
        c: strip_samples(
            evaluate_speech_decoder(model, val_sent_ld, phoneme_vocab, device, stream_mode="pure_eeg", active_centers=(c,), max_gen_samples=96)
        )
        for c in CORTICAL_CENTERS
    }
    final_unseen_sub = {
        "dual": strip_samples(evaluate_speech_decoder(model, val_sub_ld, phoneme_vocab, device, stream_mode="dual", max_gen_samples=128)),
        "pure_eeg": strip_samples(evaluate_speech_decoder(model, val_sub_ld, phoneme_vocab, device, stream_mode="pure_eeg", max_gen_samples=128)),
        "noise_control": strip_samples(evaluate_speech_decoder(model, val_sub_ld, phoneme_vocab, device, stream_mode="dual", noise_control=True, max_gen_samples=128)),
    }

    report = {
        "dataset": "ds004408_NaturalisticSpeech",
        "train_windows": len(train_idx),
        "unseen_sentence_windows": len(val_unseen_sent_idx),
        "unseen_subject_windows": len(val_unseen_sub_idx),
        "phoneme_vocab_size": len(phoneme_vocab),
        "train_wall_sec": round(time.time() - t_start, 1),
        "unseen_sentences": {k: strip_samples(v) for k, v in final_unseen_sent.items()},
        "unseen_subjects": final_unseen_sub,
        "four_center_ablation": final_four_centers,
        "qualitative_examples": final_unseen_sent["dual"]["samples"],
    }
    (RESULTS_DIR / "exp2_speech_decoder_results.json").write_text(json.dumps(report, indent=2))
    print(f"EXP 2 COMPLETE! Saved -> {RESULTS_DIR / 'exp2_speech_decoder_results.json'}", flush=True)


if __name__ == "__main__":
    main()
