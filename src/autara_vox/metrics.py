"""
Publication-Grade Evaluation Metrics for EEG-to-Speech & Inner-Speech Decoding
==============================================================================
Computes:
  - Open-loop BLEU-1, BLEU-2, BLEU-4 (smoothed NLTK)
  - ROUGE-1, ROUGE-2, ROUGE-L F1
  - Word Error Rate (WER) and Character Error Rate (CER)
  - Phoneme Error Rate (PER) via exact Levenshtein distance
  - Contrastive Top-1, Top-5, Top-10 Retrieval Accuracy & MRR
  - Output Diversity Ratio (Mode-Collapse Detector)
  - Macro-F1 & Classification Accuracy for Inner/Overt/Visual & Memory benchmarks
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import jiwer
import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score


def levenshtein_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    m, n = len(seq_a), len(seq_b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        a_i = seq_a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if a_i == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def compute_phoneme_error_rate(preds: List[List[str]], refs: List[List[str]]) -> float:
    total_edits = 0
    total_len = 0
    for p, r in zip(preds, refs):
        total_edits += levenshtein_distance(p, r)
        total_len += max(len(r), 1)
    return float(total_edits / max(total_len, 1)) * 100.0


def compute_text_generation_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute open-loop BLEU-1/2/4, ROUGE-1/2/L, WER, CER, and diversity ratio."""
    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    b1, b2, b4 = [], [], []
    r1, r2, rl = [], [], []
    clean_preds, clean_refs = [], []

    for pred, ref in zip(preds, refs):
        p_txt = (pred or "").strip().lower()
        r_txt = (ref or "").strip().lower()
        if not r_txt:
            continue
        clean_preds.append(p_txt if p_txt else "<empty>")
        clean_refs.append(r_txt)

        p_tok = p_txt.split()
        r_tok = r_txt.split()
        b1.append(sentence_bleu([r_tok], p_tok, weights=(1, 0, 0, 0), smoothing_function=smooth))
        b2.append(sentence_bleu([r_tok], p_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        b4.append(sentence_bleu([r_tok], p_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))

        rs = scorer.score(r_txt, p_txt)
        r1.append(rs["rouge1"].fmeasure)
        r2.append(rs["rouge2"].fmeasure)
        rl.append(rs["rougeL"].fmeasure)

    if not clean_refs:
        return {
            "bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0,
            "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
            "wer": 100.0, "cer": 100.0, "diversity": 0.0,
        }

    wer_val = jiwer.wer(clean_refs, clean_preds) * 100.0
    cer_val = jiwer.cer(clean_refs, clean_preds) * 100.0
    diversity = len(set(clean_preds)) / max(len(clean_preds), 1)

    return {
        "bleu1": float(np.mean(b1) * 100.0),
        "bleu2": float(np.mean(b2) * 100.0),
        "bleu4": float(np.mean(b4) * 100.0),
        "rouge1": float(np.mean(r1) * 100.0),
        "rouge2": float(np.mean(r2) * 100.0),
        "rougeL": float(np.mean(rl) * 100.0),
        "wer": float(wer_val),
        "cer": float(cer_val),
        "diversity": float(diversity * 100.0),
    }


def compute_retrieval_metrics(
    neural_emb: torch.Tensor,
    text_emb: torch.Tensor,
    text_ids: Sequence[int] | None = None,
) -> Dict[str, float]:
    """
    Compute CLIP-style neural-to-text retrieval metrics (Top-1, Top-5, Top-10, MRR, CosSim).
    Deduplicates candidate text gallery if `text_ids` is provided so identical phrases
    don't artificially split rank mass.
    """
    z_n = F.normalize(neural_emb.float(), dim=-1)
    z_t = F.normalize(text_emb.float(), dim=-1)

    if text_ids is not None:
        # Build unique phrase gallery
        unique_map: Dict[int, int] = {}
        gallery_vecs = []
        gt_gallery_idx = []
        for idx, tid in enumerate(text_ids):
            if tid not in unique_map:
                unique_map[tid] = len(gallery_vecs)
                gallery_vecs.append(z_t[idx])
            gt_gallery_idx.append(unique_map[tid])
        gallery = torch.stack(gallery_vecs, dim=0)  # [U, D]
        gt = torch.tensor(gt_gallery_idx, device=z_n.device, dtype=torch.long)
    else:
        gallery = z_t
        gt = torch.arange(z_n.size(0), device=z_n.device, dtype=torch.long)

    sim = z_n @ gallery.T  # [N, U]
    diag_sim = sim[torch.arange(sim.size(0), device=sim.device), gt].mean().item()

    # Exact rank of ground truth for each query (1-indexed)
    gt_scores = sim[torch.arange(sim.size(0), device=sim.device), gt].unsqueeze(1)
    ranks = (sim > gt_scores).sum(dim=1) + 1  # 1 = best

    n_gallery = gallery.size(0)
    top1 = (ranks <= 1).float().mean().item() * 100.0
    top5 = (ranks <= min(5, n_gallery)).float().mean().item() * 100.0
    top10 = (ranks <= min(10, n_gallery)).float().mean().item() * 100.0
    mrr = (1.0 / ranks.float()).mean().item() * 100.0

    return {
        "top1": float(top1),
        "top5": float(top5),
        "top10": float(top10),
        "mrr": float(mrr),
        "cos_sim": float(diag_sim),
        "gallery_size": int(n_gallery),
    }


def compute_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred) * 100.0),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0),
    }
