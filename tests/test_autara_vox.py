import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from autara_vox import (
    CORTICAL_CENTERS,
    AutaraVoxClassifier,
    build_biosemi128_parcellation,
    compute_phoneme_error_rate,
    compute_retrieval_metrics,
    compute_text_generation_metrics,
)


def test_biosemi128_4center_parcellation():
    parc = build_biosemi128_parcellation()
    assert len(parc.channel_names) == 128
    assert len(parc.jaw_scalp_indices) > 0
    for c in CORTICAL_CENTERS:
        assert len(parc.center_indices[c]) == 16, f"Center {c} must have 16 balanced channels"


def test_dual_stream_and_4center_ablations():
    model = AutaraVoxClassifier(num_primary_classes=4, num_aux_classes=3, jaw_ch=8, d_model=64)
    model.eval()

    pure_eeg = torch.randn(2, 4, 16, 256)
    jaw_emg = torch.randn(2, 8, 512)

    # 1. Dual stream
    out_dual = model(pure_eeg, jaw_emg, stream_mode="dual")
    assert out_dual["primary_logits"].shape == (2, 4)
    assert out_dual["aux_logits"].shape == (2, 3)
    assert torch.allclose(out_dual["stream_weights"].sum(dim=1), torch.ones(2), atol=1e-4)
    assert torch.allclose(out_dual["center_weights"].sum(dim=1), torch.ones(2), atol=1e-4)

    # 2. Pure EEG only -> jaw weight must be ~0
    out_eeg = model(pure_eeg, jaw_emg, stream_mode="pure_eeg")
    assert (out_eeg["stream_weights"][:, 1] < 1e-5).all()

    # 3. Jaw EMG only -> pure_eeg weight must be ~0
    out_jaw = model(pure_eeg, jaw_emg, stream_mode="jaw_only")
    assert (out_jaw["stream_weights"][:, 0] < 1e-5).all()

    # 4. Single cortical center ablation (e.g., speech only)
    out_sp = model(pure_eeg, None, stream_mode="pure_eeg", active_centers=("speech",))
    assert (out_sp["center_weights"][:, 0] > 0.999).all()
    assert (out_sp["center_weights"][:, 1:] < 1e-5).all()


def test_metrics_sanity():
    tm = compute_text_generation_metrics(["i am the one", "he was old"], ["i am the one", "he was an old man"])
    assert 0.0 <= tm["bleu1"] <= 100.0
    assert 0.0 <= tm["rougeL"] <= 100.0
    assert tm["diversity"] == 100.0

    per = compute_phoneme_error_rate([["AY1", "AE1", "M"]], [["AY1", "AE1", "M"]])
    assert per == 0.0

    z = torch.eye(4)
    rm = compute_retrieval_metrics(z, z)
    assert rm["top1"] == 100.0
