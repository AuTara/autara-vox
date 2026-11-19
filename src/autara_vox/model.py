"""
Autara-Vox: Dual-Stream 4-Center Cortical-Articulatory EEG-to-Speech Architecture
=================================================================================
Novel architectural components:
  1. Four Neuroanatomical Cortical Center Encoders (Speech/Broca's, Auditory/Wernicke's,
     Visual/Occipital, Memory/Prefrontal) operating strictly on Pure Cortical EEG (1-45 Hz).
  2. Dedicated Subvocal Jaw & Articulatory High-Frequency Encoder (50-200 Hz myogenic
     envelope + temporal morphology) with explicit leak-free gating switch:
     `stream_mode in ("dual", "pure_eeg", "jaw_only")`.
  3. Dynamic 4-Center Cortical Attention Gate + Dual-Stream Gate yielding interpretable
     per-sample region & pathway attributions for publication ablations.
  4. Macaron Conformer Backbone + Joint Multi-Objective Heads:
       - AutaraVoxClassifier   : Inner/Pronounced/Visualized Speech & Verbal Memory
       - AutaraVoxSpeechDecoder: Phoneme CTC + CLIP Contrastive Retrieval + BART Seq2Seq
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput

from .montage import CORTICAL_CENTERS


class ConformerConvModule(nn.Module):
    """Depthwise-separable gated convolution module from Conformer (Gulati et al., 2020)."""

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = self.norm(x).transpose(1, 2)  # [B, D, T]
        h = F.glu(self.pw1(h), dim=1)     # [B, D, T]
        h = F.silu(self.bn(self.dw(h)))
        h = self.drop(self.pw2(h)).transpose(1, 2)
        return x + h


class ConformerBlock(nn.Module):
    """Macaron-style Conformer block (FFN/2 -> MHSA -> Conv -> FFN/2)."""

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.15, kernel_size: int = 15):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_drop(attn_out)
        x = self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class CorticalRegionTokenizer(nn.Module):
    """
    Multi-scale oscillatory encoder for one neuroanatomical cortical center (16 channels).
    Extracts Beta/Gamma (k=7), Alpha/Mu (k=15), and Delta/Theta (k=31) bands @ 128 Hz,
    applies spatial mixing across the region's 16 electrodes, and downsamples via strided
    patch convolution to temporal tokens `[B, T_tok, D]`.
    """

    def __init__(self, in_ch: int = 16, d_model: int = 256, patch_stride: int = 8, dropout: float = 0.15):
        super().__init__()
        fb = 24
        self.k_fast = nn.Conv2d(1, fb, kernel_size=(1, 7), padding=(0, 3))
        self.k_mid = nn.Conv2d(1, fb, kernel_size=(1, 15), padding=(0, 7))
        self.k_slow = nn.Conv2d(1, fb, kernel_size=(1, 31), padding=(0, 15))
        self.spatial = nn.Conv2d(fb * 3, d_model, kernel_size=(in_ch, 1))
        self.bn = nn.BatchNorm1d(d_model)
        self.patch = nn.Conv1d(d_model, d_model, kernel_size=patch_stride, stride=patch_stride)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_center (16), T]
        x4 = x.unsqueeze(1)  # [B, 1, C, T]
        ms = torch.cat(
            [F.gelu(self.k_fast(x4)), F.gelu(self.k_mid(x4)), F.gelu(self.k_slow(x4))],
            dim=1,
        )
        sp = F.gelu(self.spatial(ms)).squeeze(2)  # [B, D, T]
        sp = self.bn(sp)
        tok = F.gelu(self.patch(sp)).transpose(1, 2)  # [B, T_tok, D]
        return self.drop(self.norm(tok))


class SubvocalJawArticulatoryEncoder(nn.Module):
    """
    Dedicated Subvocal Jaw & Facial Articulatory Encoder (50-200 Hz stream).
    Processes both raw high-frequency articulatory signals and their rectified
    amplitude envelope (`|x|`) to capture silent lip/jaw/temporalis micro-potentials.
    """

    def __init__(self, in_ch: int = 16, d_model: int = 256, patch_stride: int = 8, dropout: float = 0.15):
        super().__init__()
        # Concatenate raw + rectified envelope -> 2 * in_ch
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch * 2, d_model // 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=15, padding=7),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.patch = nn.Conv1d(d_model, d_model, kernel_size=patch_stride, stride=patch_stride)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_jaw, T_jaw]
        dual_feat = torch.cat([x, torch.abs(x)], dim=1)
        h = self.stem(dual_feat)
        tok = F.gelu(self.patch(h)).transpose(1, 2)
        return self.drop(self.norm(tok))


@dataclass
class BackboneOutput:
    seq_tokens: torch.Tensor       # [B, T_tok, D]
    pooled: torch.Tensor           # [B, D]
    center_weights: torch.Tensor   # [B, 4] (speech, auditory, visual, memory)
    stream_weights: torch.Tensor   # [B, 2] (pure_eeg, jaw_emg)


class AutaraVoxBackbone(nn.Module):
    """
    Shared Dual-Stream 4-Center Cortical-Articulatory Conformer Backbone.
    Supports exact ablation switches:
      - `stream_mode`: "dual" | "pure_eeg" | "jaw_only"
      - `active_centers`: any non-empty subset of ("speech", "auditory", "visual", "memory")
    """

    def __init__(
        self,
        center_ch: int = 16,
        jaw_ch: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        eeg_patch_stride: int = 8,
        jaw_patch_stride: int = 8,
        max_tokens: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model

        # Stream A: 4 independent neuroanatomical cortical center tokenizers
        self.center_encoders = nn.ModuleDict(
            {
                c: CorticalRegionTokenizer(center_ch, d_model, eeg_patch_stride, dropout)
                for c in CORTICAL_CENTERS
            }
        )
        self.center_embs = nn.Parameter(torch.randn(4, 1, 1, d_model) * 0.02)
        self.center_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        # Stream B: Subvocal jaw & articulatory encoder
        self.jaw_encoder = SubvocalJawArticulatoryEncoder(jaw_ch, d_model, jaw_patch_stride, dropout)
        self.stream_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        # Temporal positional encoding + Conformer fusion stack
        self.pos_emb = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        self.conformer = nn.ModuleList(
            [ConformerBlock(d_model, nhead=nhead, dropout=dropout) for _ in range(num_layers)]
        )
        self.pool_attn = nn.Linear(d_model, 1)

    def forward(
        self,
        pure_eeg: Optional[torch.Tensor],          # [B, 4, 16, T_eeg]
        jaw_emg: Optional[torch.Tensor] = None,    # [B, C_jaw, T_jaw]
        stream_mode: str = "dual",
        active_centers: Sequence[str] = CORTICAL_CENTERS,
    ) -> BackboneOutput:
        use_eeg = stream_mode in ("dual", "pure_eeg") and pure_eeg is not None
        use_jaw = stream_mode in ("dual", "jaw_only") and jaw_emg is not None
        if not use_eeg and not use_jaw:
            raise ValueError("At least one stream (pure_eeg or jaw_emg) must be active.")

        ref = pure_eeg if use_eeg else jaw_emg
        B = ref.size(0)
        device = ref.device

        # --- Stream A: 4-Center Pure Cortical EEG ---
        eeg_seq: Optional[torch.Tensor] = None
        center_weights = torch.zeros(B, 4, device=device)

        if use_eeg:
            active_set = set(active_centers)
            center_seqs = []
            center_logits = []
            for idx, c in enumerate(CORTICAL_CENTERS):
                if c in active_set:
                    tok = self.center_encoders[c](pure_eeg[:, idx]) + self.center_embs[idx]
                    logit = self.center_gate(tok.mean(dim=1))  # [B, 1]
                    center_seqs.append(tok)
                    center_logits.append(logit)
                else:
                    center_seqs.append(None)
                    center_logits.append(torch.full((B, 1), -1e9, device=device))

            center_weights = F.softmax(torch.cat(center_logits, dim=1), dim=1)  # [B, 4]

            # Weighted cortical combination preserving temporal resolution [B, T_tok, D]
            for idx, tok in enumerate(center_seqs):
                if tok is not None:
                    w = center_weights[:, idx].view(B, 1, 1)
                    contrib = tok * (0.5 + w)
                    eeg_seq = contrib if eeg_seq is None else (eeg_seq + contrib)

        # --- Stream B: Subvocal Jaw / Articulatory High-Frequency Stream ---
        jaw_seq: Optional[torch.Tensor] = None
        if use_jaw:
            jaw_seq = self.jaw_encoder(jaw_emg)
            # Align temporal token length to eeg_seq if both streams present
            if eeg_seq is not None and jaw_seq.size(1) != eeg_seq.size(1):
                jaw_seq = F.interpolate(
                    jaw_seq.transpose(1, 2), size=eeg_seq.size(1), mode="linear", align_corners=False
                ).transpose(1, 2)

        # --- Dynamic Dual-Stream Reliability Gate ---
        g_eeg = (
            self.stream_gate(eeg_seq.mean(dim=1))
            if eeg_seq is not None
            else torch.full((B, 1), -1e9, device=device)
        )
        g_jaw = (
            self.stream_gate(jaw_seq.mean(dim=1))
            if jaw_seq is not None
            else torch.full((B, 1), -1e9, device=device)
        )
        stream_weights = F.softmax(torch.cat([g_eeg, g_jaw], dim=1), dim=1)  # [B, 2]

        if eeg_seq is not None and jaw_seq is not None:
            fused = eeg_seq * (0.5 + stream_weights[:, 0:1].unsqueeze(-1)) + jaw_seq * (
                0.5 + stream_weights[:, 1:2].unsqueeze(-1)
            )
        elif eeg_seq is not None:
            fused = eeg_seq
        else:
            fused = jaw_seq

        T_tok = fused.size(1)
        x = fused + self.pos_emb[:, :T_tok, :]
        for blk in self.conformer:
            x = blk(x)

        # Attentive temporal pooling
        pw = F.softmax(self.pool_attn(x), dim=1)  # [B, T_tok, 1]
        pooled = (x * pw).sum(dim=1)              # [B, D]

        return BackboneOutput(
            seq_tokens=x,
            pooled=pooled,
            center_weights=center_weights,
            stream_weights=stream_weights,
        )


class AutaraVoxClassifier(nn.Module):
    """
    Multi-task classification wrapper for:
      - Experiment 1 (ds003626): Inner Speech / Pronounced Speech / Visualized Condition
        (4 directional words + 3 speech modality states)
      - Pretraining / Transfer (ds003655): Verbal Working Memory (Retention vs. Manipulation + Load)
    """

    def __init__(
        self,
        num_primary_classes: int = 4,
        num_aux_classes: int = 3,
        jaw_ch: int = 8,
        d_model: int = 256,
        eeg_patch_stride: int = 8,
        jaw_patch_stride: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.backbone = AutaraVoxBackbone(
            center_ch=16,
            jaw_ch=jaw_ch,
            d_model=d_model,
            eeg_patch_stride=eeg_patch_stride,
            jaw_patch_stride=jaw_patch_stride,
            dropout=dropout,
        )
        self.primary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_primary_classes),
        )
        self.aux_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_aux_classes),
        )

    def forward(
        self,
        pure_eeg: Optional[torch.Tensor],
        jaw_emg: Optional[torch.Tensor] = None,
        stream_mode: str = "dual",
        active_centers: Sequence[str] = CORTICAL_CENTERS,
    ) -> Dict[str, torch.Tensor]:
        out = self.backbone(pure_eeg, jaw_emg, stream_mode=stream_mode, active_centers=active_centers)
        return {
            "primary_logits": self.primary_head(out.pooled),
            "aux_logits": self.aux_head(out.pooled),
            "embedding": out.pooled,
            "center_weights": out.center_weights,
            "stream_weights": out.stream_weights,
        }


class AutaraVoxSpeechDecoder(nn.Module):
    """
    Open-Vocabulary EEG-to-Speech & Phoneme Decoder (Experiment 2: ds004408).
    Combines:
      1. AutaraVoxBackbone (4-Center Pure EEG + High-Freq Jaw/Articulatory Conformer)
      2. Phoneme CTC Head (ARPAbet phoneme sequence decoding)
      3. CLIP-Style Contrastive Neural-Text Alignment Head (zero-shot phrase retrieval)
      4. Q-Former Cross-Attention Bridge -> Pretrained `facebook/bart-base` Seq2Seq Decoder
         for true open-loop autoregressive English text generation.
    """

    def __init__(
        self,
        num_phonemes: int = 44,
        jaw_ch: int = 16,
        d_model: int = 256,
        bart_name: str = "facebook/bart-base",
        num_prefix_tokens: int = 24,
        dropout: float = 0.2,
        freeze_bart_encoder: bool = True,
    ):
        super().__init__()
        self.backbone = AutaraVoxBackbone(
            center_ch=16,
            jaw_ch=jaw_ch,
            d_model=d_model,
            eeg_patch_stride=8,
            jaw_patch_stride=8,
            dropout=dropout,
        )

        # 1. Phoneme CTC Head
        self.ctc_head = nn.Linear(d_model, num_phonemes)

        # 2. Pretrained BART Seq2Seq LM
        self.tokenizer = BartTokenizer.from_pretrained(bart_name)
        self.bart = BartForConditionalGeneration.from_pretrained(bart_name)
        bart_dim = self.bart.config.d_model  # 768

        if freeze_bart_encoder:
            for p in self.bart.model.encoder.parameters():
                p.requires_grad = False
        # Freeze token embeddings to prevent overfitting to small vocab
        for p in self.bart.model.shared.parameters():
            p.requires_grad = False

        # 3. Contrastive Alignment Projection (maps neural pooled vector -> BART text space)
        self.neural_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bart_dim),
            nn.GELU(),
            nn.Linear(bart_dim, 256),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(bart_dim),
            nn.Linear(bart_dim, 256),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # ln(1/0.07)

        # 4. Q-Former Cross-Attention Neural-to-BART Prefix Adapter
        self.prefix_queries = nn.Parameter(torch.randn(1, num_prefix_tokens, bart_dim) * 0.02)
        self.kv_proj = nn.Linear(d_model, bart_dim)
        self.qformer_norm_q = nn.LayerNorm(bart_dim)
        self.qformer_norm_kv = nn.LayerNorm(bart_dim)
        self.qformer_attn = nn.MultiheadAttention(bart_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.qformer_ffn = nn.Sequential(
            nn.LayerNorm(bart_dim),
            nn.Linear(bart_dim, bart_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bart_dim * 2, bart_dim),
        )
        self.prefix_out_norm = nn.LayerNorm(bart_dim)

    @torch.no_grad()
    def encode_text_frozen(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        enc_out = self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (enc_out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled

    def neural_to_bart_prefix(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        B = seq_tokens.size(0)
        q = self.prefix_queries.expand(B, -1, -1)
        kv = self.qformer_norm_kv(self.kv_proj(seq_tokens))
        attn_out, _ = self.qformer_attn(self.qformer_norm_q(q), kv, kv, need_weights=False)
        h = q + attn_out
        h = h + self.qformer_ffn(h)
        return self.prefix_out_norm(h)  # [B, num_prefix_tokens, 768]

    def forward(
        self,
        pure_eeg: Optional[torch.Tensor],
        jaw_emg: Optional[torch.Tensor],
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        stream_mode: str = "dual",
        active_centers: Sequence[str] = CORTICAL_CENTERS,
    ) -> Dict[str, torch.Tensor]:
        bb = self.backbone(pure_eeg, jaw_emg, stream_mode=stream_mode, active_centers=active_centers)

        # 1. Phoneme CTC log-probs: [T_tok, B, V]
        ctc_log_probs = F.log_softmax(self.ctc_head(bb.seq_tokens), dim=-1).transpose(0, 1)

        # 2. Contrastive EEG <-> Text embeddings
        z_neural = F.normalize(self.neural_proj(bb.pooled), dim=-1)
        frozen_txt = self.encode_text_frozen(text_input_ids, text_attention_mask)
        z_text = F.normalize(self.text_proj(frozen_txt), dim=-1)

        scale = self.logit_scale.exp().clamp(max=50.0)
        logits_nt = scale * (z_neural @ z_text.T)
        targets = torch.arange(z_neural.size(0), device=z_neural.device)
        contrastive_loss = 0.5 * (
            F.cross_entropy(logits_nt, targets) + F.cross_entropy(logits_nt.T, targets)
        )

        # 3. Seq2Seq LM Loss conditioned ONLY on neural prefix tokens
        prefix = self.neural_to_bart_prefix(bb.seq_tokens)
        enc_out = BaseModelOutput(last_hidden_state=prefix)
        lm_out = self.bart(encoder_outputs=enc_out, labels=labels)

        return {
            "lm_loss": lm_out.loss,
            "contrastive_loss": contrastive_loss,
            "ctc_log_probs": ctc_log_probs,
            "z_neural": z_neural,
            "z_text": z_text,
            "center_weights": bb.center_weights,
            "stream_weights": bb.stream_weights,
        }

    @torch.no_grad()
    def generate_open_loop(
        self,
        pure_eeg: Optional[torch.Tensor],
        jaw_emg: Optional[torch.Tensor] = None,
        stream_mode: str = "dual",
        active_centers: Sequence[str] = CORTICAL_CENTERS,
        max_new_tokens: int = 20,
        num_beams: int = 3,
    ) -> Tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        True Open-Loop Autoregressive Decoding (ZERO ground-truth token leakage!).
        Returns: (decoded_texts, ctc_argmax_ids, z_neural, center_weights, stream_weights)
        """
        bb = self.backbone(pure_eeg, jaw_emg, stream_mode=stream_mode, active_centers=active_centers)
        prefix = self.neural_to_bart_prefix(bb.seq_tokens)
        enc_out = BaseModelOutput(last_hidden_state=prefix)

        gen_ids = self.bart.generate(
            encoder_outputs=enc_out,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        ctc_preds = self.ctc_head(bb.seq_tokens).argmax(dim=-1)  # [B, T_tok]
        z_neural = F.normalize(self.neural_proj(bb.pooled), dim=-1)
        return texts, ctc_preds, z_neural, bb.center_weights, bb.stream_weights
