---
language: en
tags:
- bci
- eeg
- eeg-to-speech
- inner-speech
- subvocal-emg
- conformer
- bart
- neuroscience
- medical
license: mit
---

# Autara-Vox: Dual-Stream 4-Center Cortical-Articulatory EEG-to-Speech Model

**Autara Research Labs**

**Autara-Vox** is a standalone non-invasive EEG-to-Speech neural architecture and IEEE Transactions manuscript ([`paper/autara_vox_paper.pdf`](paper/autara_vox_paper.pdf)) that explicitly disentangles:
1. **Pure Cortical EEG Oscillations (`1–45 Hz`)** parcellated across 4 neuroanatomical brain centers (**Speech/Broca's**, **Auditory/Wernicke's**, **Visual/Occipital**, **Memory/Prefrontal**).
2. **Subvocal Jaw & Facial Articulatory EMG (`50–200 Hz`)** with a leak-free gating switch (`dual | pure_eeg | jaw_only`).
3. **Mandatory Every-10-Epoch Live Verification Probes** paired with Gaussian Random-Noise EEG Controls (`paper/results/*_probe_ep*.json`).

## Headline Verified Results (14,208 Synchronized OpenNeuro Trials)
- **Exp 0 — Verbal Working Memory (`ds003655`, Unseen Subjects):** `63.1%` accuracy (`+16.8%` Macro-F1 over Random-Noise Control).
- **Exp 1 — Inner / Pronounced+Jaw / Visualized Speech (`ds003626`, Unseen Session 3):**
  - **Dual-Stream + Memory Transfer:** **`69.1%`** 3-modality accuracy (**`73.7%`** Macro-F1, `+42.9%` over noise control), beating both Pure EEG (`55.3%`) and Jaw/EXG (`66.6%`).
  - **Pronounced+Jaw 4-Word Accuracy:** **`68.3%`** (`2.7×` chance).
  - **4-Center Cortical Synergy (Pure EEG Only):** All 4 centers (`55.3%`) outperform every individual center alone (`Visual 45.5%`, `Auditory 45.1%`, `Speech 41.0%`, `Memory 37.6%`).
- **Exp 2 — Open-Loop Continuous Speech Decoding (`ds004408`, Zero Teacher Forcing):**
  - Learned 4-center gate autonomously routed **`96.4%–99.7%`** to the Auditory Center during continuous speech listening.
  - Best retrieval checkpoint: `Top-5 = 3.66%` (`+1.10%` vs. noise), `BLEU-1 = 9.92`, `ROUGE-L = 11.14`; Epoch 30 Pure-EEG: `BLEU-1 = 11.48`, `ROUGE-L = 11.77`, `PER = 88.74%`, diversity **`18.75%`** (`3.6×` noise control).

## Model Weights (Hugging Face)
Trained checkpoints are hosted on Hugging Face at [`autara-ai/Autara-Vox`](https://huggingface.co/autara-ai/Autara-Vox):
- `checkpoints/autara_vox_speech_best.pt` — Continuous Speech Decoder (`AutaraVoxSpeechDecoder` + BART-base adapter)
- `checkpoints/exp1_inner_mem_transfer_best.pt` — Dual-Stream Inner/Pronounced/Visualized Classifier (Memory-pretrained)
- `checkpoints/exp1_inner_scratch_best.pt` — Dual-Stream Inner/Pronounced/Visualized Classifier (Scratch)
- `checkpoints/memory_pretrained_backbone.pt` — Verbal Working Memory 4-Center Pretrained Backbone

## Quick Start

```bash
# Install dependencies via uv
uv venv --python 3.12
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install -e . pytest

# 1. Run unit tests
.venv/bin/pytest -v tests/test_autara_vox.py

# 2. Build 4-center OpenNeuro datasets (ds003626, ds004408, ds003655)
.venv/bin/python scripts/download_and_build_datasets.py --workers 8 --ds4408-subs 8 --ds4408-runs 10 --ds3655-subs 12

# 3. Train Exp 0 (Memory) + Exp 1 (Inner/Jaw Speech) with 10-epoch live probes
.venv/bin/python scripts/train_inner_speech.py --epochs-mem 30 --epochs-speech 30

# 4. Train Exp 2 (Continuous Speech Decoder) with 10-epoch live probes
.venv/bin/python scripts/train_speech_decoder.py --epochs 30 --batch-size 32

# 5. Live End-to-End EEG-to-Speech Demo & Publication Figures
.venv/bin/python scripts/demo_thought_to_speech.py
.venv/bin/python scripts/generate_paper_figures.py
```
