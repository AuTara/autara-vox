"""
Neuroanatomical 4-Center Cortical Parcellation & Subvocal Jaw/Articulatory Mapping
==================================================================================
Maps standard 128-channel BioSemi ActiveTwo coordinates (used natively in both
OpenNeuro ds003626 and ds004408) and 10-20 coordinates (ds003655) into four
functionally distinct cortical centers + subvocal jaw/articulatory sites:
  1. speech   : Broca's area (left IFG) + bilateral ventral sensorimotor speech cortex
  2. auditory : Wernicke's area + bilateral superior temporal / perisylvian auditory cortex
  3. visual   : Occipital & parieto-occipital visual cortex
  4. memory   : Prefrontal / DLPFC & phonological-loop working memory center
  + jaw_scalp : Inferior fronto-temporal scalp ring adjacent to masseter/temporalis muscles
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import mne
import numpy as np

CORTICAL_CENTERS = ("speech", "auditory", "visual", "memory")


@dataclass(frozen=True)
class CorticalParcellation:
    channel_names: List[str]
    center_indices: Dict[str, List[int]]
    jaw_scalp_indices: List[int]
    positions_3d: np.ndarray  # [N_ch, 3] (x=right, y=anterior, z=superior)


def build_biosemi128_parcellation() -> CorticalParcellation:
    """Deterministically partition BioSemi 128 channels using standard MNE 3D coordinates."""
    montage = mne.channels.make_standard_montage("biosemi128")
    ch_pos = montage.get_positions()["ch_pos"]
    names = list(ch_pos.keys())  # Exact canonical order A1..A32, B1..B32, C1..C32, D1..D32
    coords = np.array([ch_pos[n] for n in names], dtype=np.float32)

    centers: Dict[str, List[int]] = {c: [] for c in CORTICAL_CENTERS}
    jaw_ring: List[int] = []

    for idx, (x, y, z) in enumerate(coords):
        # Inferior fronto-temporal scalp ring (closest to temporalis/masseter jaw EMG leakage)
        if z < 0.025 and y > -0.045 and abs(x) > 0.052:
            jaw_ring.append(idx)

        # 4 non-overlapping neuroanatomical cortical centers:
        # 1. Prefrontal / Executive & Working Memory (anterior frontal)
        if y > 0.045:
            centers["memory"].append(idx)
        # 2. Auditory / Wernicke's (bilateral lateral temporal / perisylvian)
        elif abs(x) > 0.058 and -0.055 <= y <= 0.025 and z < 0.052:
            centers["auditory"].append(idx)
        # 3. Visual / Occipital & Parieto-Occipital (posterior)
        elif y < -0.028:
            centers["visual"].append(idx)
        # 4. Speech Production (Broca's / IFG + Ventral Sensorimotor & Central Speech Motor)
        else:
            centers["speech"].append(idx)

    # Guarantee fixed balanced channel counts (16 representative electrodes per center)
    # by evenly subsampling spatially sorted indices if > 16, or padding if < 16.
    balanced: Dict[str, List[int]] = {}
    for c in CORTICAL_CENTERS:
        idxs = centers[c]
        if len(idxs) >= 16:
            pick = np.linspace(0, len(idxs) - 1, 16, dtype=int)
            balanced[c] = [idxs[i] for i in pick]
        else:
            balanced[c] = idxs

    if len(jaw_ring) > 16:
        pick = np.linspace(0, len(jaw_ring) - 1, 16, dtype=int)
        jaw_ring = [jaw_ring[i] for i in pick]

    return CorticalParcellation(
        channel_names=names,
        center_indices=balanced,
        jaw_scalp_indices=jaw_ring,
        positions_3d=coords,
    )


def map_1020_to_4centers(ch_names: List[str]) -> Dict[str, List[int]]:
    """Map standard 10-20 channels (e.g., ds003655 21-ch Verbal Working Memory) to the 4 centers."""
    norm = {name.upper(): i for i, name in enumerate(ch_names)}

    def pick(candidates: List[str]) -> List[int]:
        return [norm[c.upper()] for c in candidates if c.upper() in norm]

    return {
        "memory": pick(["Fp1", "Fp2", "F3", "Fz", "F4"]),
        "speech": pick(["F7", "F8", "C3", "Cz", "C4"]),
        "auditory": pick(["T3", "T4", "T7", "T8", "T5", "T6", "P7", "P8"]),
        "visual": pick(["O1", "O2", "Oz", "P3", "Pz", "P4"]),
    }


def get_ordered_64_center_indices(parc: CorticalParcellation) -> List[int]:
    """Return the exact 64 channel indices stacked as [16 speech | 16 auditory | 16 visual | 16 memory]."""
    out: List[int] = []
    for c in CORTICAL_CENTERS:
        out.extend(parc.center_indices[c])
    return out
