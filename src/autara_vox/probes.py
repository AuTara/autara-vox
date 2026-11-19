"""
Mandatory Every-10-Epoch Live Verification Probe & Health Monitor
=================================================================
Automatically executes at Epoch 10, 20, 30, ... during training to verify:
  1. Actual open-loop decoded text & phonemes vs. ground truth on held-out trials
  2. Real-EEG vs. Random-Noise-EEG sanity control (Jo et al., 2024 honesty protocol)
  3. Mode-collapse diversity check & Dual-Stream (Pure EEG vs. Jaw EMG) health
  4. Persists structured JSON probe snapshots to `paper/results/` for paper plots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class EveryTenEpochProbe:
    def __init__(self, run_name: str, results_dir: str | Path = "paper/results", interval: int = 10):
        self.run_name = run_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.history: List[Dict[str, Any]] = []

    def should_run(self, epoch: int, total_epochs: int) -> bool:
        return (epoch % self.interval == 0) or (epoch == total_epochs)

    def record_and_report(
        self,
        epoch: int,
        real_metrics: Dict[str, Any],
        noise_metrics: Dict[str, Any] | None = None,
        sample_decodings: List[Dict[str, str]] | None = None,
        stream_ablation: Dict[str, Any] | None = None,
        center_weights: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        alerts: List[str] = []

        # 1. Check Mode Collapse on open-loop text diversity
        if "diversity" in real_metrics and real_metrics["diversity"] < 15.0:
            alerts.append(f"LOW_DIVERSITY ({real_metrics['diversity']:.1f}% unique outputs — possible LM collapse)")

        # 2. Check Real-EEG vs Random-Noise Control Delta
        neural_gain: Dict[str, float] = {}
        if noise_metrics:
            for key in ("top1", "top5", "top10", "bleu1", "rougeL", "acc"):
                if key in real_metrics and key in noise_metrics:
                    delta = float(real_metrics[key]) - float(noise_metrics[key])
                    neural_gain[key] = round(delta, 3)
            # Flag if primary neural metric does not exceed random noise control
            primary = "top5" if "top5" in neural_gain else ("acc" if "acc" in neural_gain else None)
            if primary is not None and neural_gain[primary] <= 0.5:
                alerts.append(
                    f"NOISE_CONTROL_WARNING ({primary} gain vs noise = {neural_gain[primary]:+.2f}%)"
                )

        status = "HEALTHY" if not alerts else "ATTENTION_NEEDED"
        snapshot = {
            "epoch": epoch,
            "run_name": self.run_name,
            "status": status,
            "alerts": alerts,
            "real_metrics": real_metrics,
            "noise_control_metrics": noise_metrics or {},
            "neural_gain_over_noise": neural_gain,
            "stream_ablation": stream_ablation or {},
            "center_weights": center_weights or {},
            "sample_decodings": sample_decodings or [],
        }
        self.history.append(snapshot)

        out_path = self.results_dir / f"{self.run_name}_probe_ep{epoch:02d}.json"
        out_path.write_text(json.dumps(snapshot, indent=2))
        (self.results_dir / f"{self.run_name}_probe_history.json").write_text(
            json.dumps(self.history, indent=2)
        )

        # Crisp live terminal report
        print("\n" + "=" * 74)
        print(f"  [LIVE 10-EPOCH PROBE] {self.run_name} @ Epoch {epoch}  |  STATUS: {status}")
        print("=" * 74)
        for k, v in real_metrics.items():
            if isinstance(v, float):
                noise_str = ""
                if noise_metrics and k in noise_metrics and isinstance(noise_metrics[k], (int, float)):
                    delta = v - float(noise_metrics[k])
                    noise_str = f"  (Noise Ctrl: {float(noise_metrics[k]):6.2f} | Gain: {delta:+6.2f})"
                print(f"  {k:14s}: {v:6.2f}{noise_str}")

        if stream_ablation:
            print("  --- Dual-Stream Ablation (Pure EEG vs. Jaw/EXG vs. Dual) ---")
            for mode, vals in stream_ablation.items():
                summary = " | ".join(
                    f"{mk}={mv:.2f}" if isinstance(mv, float) else f"{mk}={mv}"
                    for mk, mv in vals.items()
                )
                print(f"    {mode:12s}: {summary}")

        if center_weights:
            cw = " | ".join(f"{k}={v*100:.1f}%" for k, v in center_weights.items())
            print(f"  4-Center Gate : {cw}")

        if sample_decodings:
            print("  --- Live Unseen-Trial Decodings ---")
            for idx, ex in enumerate(sample_decodings[:5], start=1):
                print(f"    [{idx}] GT   : {ex.get('gt', '')}")
                if "retrieved" in ex:
                    print(f"        RET  : {ex['retrieved']} (sim={ex.get('sim', '')})")
                if "generated" in ex:
                    print(f"        GEN  : {ex['generated']}")
                if "noise_gen" in ex:
                    print(f"        NOISE: {ex['noise_gen']}")
                if "phonemes_pred" in ex:
                    print(f"        PHON : {ex['phonemes_pred']}  [GT: {ex.get('phonemes_gt', '')}]")

        if alerts:
            for a in alerts:
                print(f"  [!] ALERT: {a}")
        print("=" * 74 + "\n", flush=True)
        return snapshot
