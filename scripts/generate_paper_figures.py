#!/usr/bin/env python3
"""
Generate Publication-Quality IEEE Vector Figures (PDF + PNG, 300 DPI)
=====================================================================
Strictly reads verified empirical metrics from `paper/results/*.json` (zero
hardcoded or fabricated numbers!) and exports 4 camera-ready figures to
`paper/figures/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from autara_vox.montage import CORTICAL_CENTERS, build_biosemi128_parcellation

RESULTS = Path("paper/results")
FIGURES = Path("paper/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc, ec="#1e293b"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12", facecolor=fc, edgecolor=ec, lw=1.1
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontweight="bold", fontsize=8.5, color="#0f172a")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.0, color="#334155")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1, color="#334155"))

    # Inputs -> 4 cortical centers + Jaw EMG
    box(0.15, 3.35, 2.0, 0.78, "Speech Center", "Broca / vSMC (1-45 Hz)", "#dbeafe")
    box(0.15, 2.40, 2.0, 0.78, "Auditory Center", "Wernicke / STG (1-45 Hz)", "#e0e7ff")
    box(0.15, 1.45, 2.0, 0.78, "Visual Center", "Occipital / PO (1-45 Hz)", "#dcfce7")
    box(0.15, 0.50, 2.0, 0.78, "Memory Center", "Prefrontal / DLPFC", "#fef3c7")

    box(2.75, 2.15, 1.95, 1.75, "4-Center Gate", "Multi-Scale Convs\n+ Region Attention\n[w_S, w_A, w_V, w_M]", "#eff6ff")
    box(2.75, 0.45, 1.95, 1.35, "Subvocal Jaw/EXG", "50-200 Hz Envelope\n+ Rectified Micro-EMG", "#fee2e2")

    box(5.25, 1.25, 1.95, 2.05, "Dual-Stream\nConformer", "Reliability Gate\n[Pure-EEG vs Jaw]\n4x Macaron Blocks", "#f3e8ff")

    box(7.75, 3.15, 2.1, 0.90, "Phoneme CTC", "ARPAbet Alignment", "#f1f5f9")
    box(7.75, 1.95, 2.1, 0.95, "CLIP Retrieval", "Zero-Shot Phrase Bank\n(e.g. 'I am the one')", "#ecfeff")
    box(7.75, 0.65, 2.1, 1.05, "Q-Former -> BART", "Open-Loop Text\n+ TTS Speech (.wav)", "#f0fdf4")

    for y in (3.74, 2.79, 1.84, 0.89):
        arrow(2.22, y, 2.68, 3.00)
    arrow(4.78, 3.00, 5.18, 2.55)
    arrow(4.78, 1.12, 5.18, 1.95)
    arrow(7.28, 2.70, 7.68, 3.60)
    arrow(7.28, 2.30, 7.68, 2.42)
    arrow(7.28, 1.85, 7.68, 1.18)

    ax.set_title("Autara-Vox: Dual-Stream 4-Center Cortical-Articulatory EEG-to-Speech Architecture", fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_architecture.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_dual_stream_ablation():
    e1 = json.loads((RESULTS / "exp1_inner_mem_transfer_results.json").read_text())
    e2 = json.loads((RESULTS / "exp2_speech_decoder_results.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # Left: ds003626 (Word4 & Cond3 across Dual / Pure EEG / Jaw Only / Noise)
    modes = ["dual", "pure_eeg", "jaw_only", "noise_control"]
    labels = ["Dual\nStream", "Pure EEG\n(1-45Hz)", "Jaw/EXG\n(50-120Hz)", "Noise\nControl"]
    w_acc = [e1["stream_ablation"][m]["word_acc"] for m in modes]
    c_acc = [e1["stream_ablation"][m]["cond_acc"] for m in modes]

    x = np.arange(len(modes))
    width = 0.36
    ax1.bar(x - width / 2, w_acc, width, label="4-Word Acc (%)", color="#2563eb")
    ax1.bar(x + width / 2, c_acc, width, label="3-Modality Acc (%)", color="#10b981")
    ax1.axhline(25.0, ls="--", lw=0.9, color="#64748b", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a) Inner/Overt/Visual Speech (ds003626)")
    ax1.legend(frameon=False, loc="upper right")
    ax1.set_ylim(0, 105)

    # Right: ds004408 Continuous Speech (Top-5 Retrieval & BLEU-1 across Dual / Pure EEG / Jaw / Noise)
    top5 = [e2["unseen_sentences"][m]["top5"] for m in modes]
    b1 = [e2["unseen_sentences"][m]["bleu1"] for m in modes]
    ax2.bar(x - width / 2, top5, width, label="Top-5 Retrieval (%)", color="#7c3aed")
    ax2.bar(x + width / 2, b1, width, label="Open-Loop BLEU-1", color="#f59e0b")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("(b) Continuous Speech Decoding (ds004408)")
    ax2.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_dual_stream_ablation.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_dual_stream_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_four_center_cortical_map():
    parc = build_biosemi128_parcellation()
    e1 = json.loads((RESULTS / "exp1_inner_mem_transfer_results.json").read_text())
    e2 = json.loads((RESULTS / "exp2_speech_decoder_results.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Left: 2D Topographic Projection of the 4 Cortical Centers + Jaw Ring
    pos = parc.positions_3d
    colors = {"speech": "#2563eb", "auditory": "#7c3aed", "visual": "#16a34a", "memory": "#ea580c"}
    # Draw head outline
    theta = np.linspace(0, 2 * np.pi, 200)
    r_head = float(np.max(np.hypot(pos[:, 0], pos[:, 1]))) * 1.06
    ax1.plot(r_head * np.cos(theta), r_head * np.sin(theta), color="#334155", lw=1.2)
    # Nose indicator
    ax1.plot([-0.012, 0.0, 0.012], [r_head * 0.99, r_head * 1.10, r_head * 0.99], color="#334155", lw=1.2)

    for c in CORTICAL_CENTERS:
        idxs = parc.center_indices[c]
        ax1.scatter(pos[idxs, 0], pos[idxs, 1], s=26, color=colors[c], label=c.capitalize(), edgecolors="white", lw=0.4)
    j_idxs = parc.jaw_scalp_indices
    ax1.scatter(pos[j_idxs, 0], pos[j_idxs, 1], s=42, facecolors="none", edgecolors="#dc2626", lw=1.1, label="Jaw Ring")

    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("(a) BioSemi-128 4-Center Parcellation")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=7.5)

    # Right: Single-Center Ablation Performance on ds003626 (Modality Acc) & ds004408 (Top-10 Retrieval)
    c_labels = [c.capitalize() for c in CORTICAL_CENTERS]
    s1_vals = [e1["four_center_ablation"][c]["cond_acc"] for c in CORTICAL_CENTERS]
    s2_vals = [e2["four_center_ablation"][c]["top10"] for c in CORTICAL_CENTERS]

    x = np.arange(len(CORTICAL_CENTERS))
    w = 0.36
    ax2.bar(x - w / 2, s1_vals, w, label="ds003626 Modality Acc (%)", color="#0ea5e9")
    ax2.bar(x + w / 2, s2_vals, w, label="ds004408 Top-10 Ret (%)", color="#8b5cf6")
    ax2.set_xticks(x)
    ax2.set_xticklabels(c_labels)
    ax2.set_ylabel("Performance (%)")
    ax2.set_title("(b) Single-Center Cortical Ablation")
    ax2.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_four_center_cortical_map.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig3_four_center_cortical_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_10epoch_probes():
    h1 = json.loads((RESULTS / "exp1_inner_mem_transfer_probe_history.json").read_text())
    h2 = json.loads((RESULTS / "exp2_speech_ds004408_probe_history.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    ep1 = [p["epoch"] for p in h1]
    ax1.plot(ep1, [p["real_metrics"]["cond_acc"] for p in h1], "o-", color="#10b981", lw=2, label="Real EEG+Jaw (Modality)")
    ax1.plot(ep1, [p["real_metrics"]["acc"] for p in h1], "s-", color="#2563eb", lw=2, label="Real EEG+Jaw (4-Word)")
    ax1.plot(ep1, [p["noise_control_metrics"]["acc"] for p in h1], "--^", color="#ef4444", lw=1.4, label="Noise Control (4-Word)")
    ax1.set_xlabel("Training Epoch (10-Epoch Probes)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a) ds003626 Every-10-Epoch Health Probe")
    ax1.set_xticks(ep1)
    ax1.legend(frameon=False, fontsize=7.5)
    ax1.grid(alpha=0.25)

    ep2 = [p["epoch"] for p in h2]
    ax2.plot(ep2, [p["real_metrics"]["top5"] for p in h2], "o-", color="#7c3aed", lw=2, label="Real EEG Top-5 Ret (%)")
    ax2.plot(ep2, [p["noise_control_metrics"]["top5"] for p in h2], "--^", color="#ef4444", lw=1.4, label="Noise Control Top-5 (%)")
    ax2.plot(ep2, [p["real_metrics"]["bleu1"] for p in h2], "s-", color="#f59e0b", lw=2, label="Real Open-Loop BLEU-1")
    ax2.set_xlabel("Training Epoch (10-Epoch Probes)")
    ax2.set_ylabel("Score (%)")
    ax2.set_title("(b) ds004408 Every-10-Epoch Probe vs. Noise")
    ax2.set_xticks(ep2)
    ax2.legend(frameon=False, fontsize=7.5)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_10epoch_probes.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig4_10epoch_probes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    fig1_architecture()
    fig2_dual_stream_ablation()
    fig3_four_center_cortical_map()
    fig4_10epoch_probes()
    print(f"Generated all 4 IEEE publication figures in {FIGURES}/")


if __name__ == "__main__":
    main()
