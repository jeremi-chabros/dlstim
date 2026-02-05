#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.1",
#     "numpy>=1.24",
#     "pandas>=2.0",
#     "h5py>=3.9",
#     "tqdm>=4.65",
#     "scikit-learn>=1.3",
#     "scipy>=1.10",
# ]
# ///
"""
Stimulation Artifact Detector for NeuroPace RNS ECoG
====================================================

Multi-Scale Conditional 1D U-Net with FiLM conditioning on stimulation
parameters, electrode montage, and lead geometry.

Architecture:
    - 4-channel ECoG input (250 Hz)
    - FiLM conditioning on: B1/B2 stim params, montage, lead type
    - Encoder: 4 levels with depthwise-separable convolutions
    - Bottleneck with dilated convolutions for wider receptive field
    - Decoder: 4 levels with skip connections
    - Output: per-sample artifact probability mask

Optimized for Apple Silicon (MPS) - M4 Pro 48GB
"""

from __future__ import annotations

import re
import math
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import scipy.ndimage

# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class Config:
    """Full configuration for model, data, and training."""
    # ---------- Paths ----------
    data_dir: Path = Path("./processed_data")
    annotations_path: Path = Path("./consolidated_stim_detection_data.csv")
    checkpoint_dir: Path = Path("./checkpoints")

    # ---------- Signal ----------
    sampling_rate: int = 250
    n_channels: int = 4
    window_samples: int = 2048       # ~8.2 s at 250 Hz
    stride_samples: int = 512        # ~2 s stride (75% overlap for training)
    inference_stride_ratio: float = 0.25  # 75% overlap at inference

    # ---------- Conditioning dimensions ----------
    # B1: 5 stim params + 9 montage = 14
    # B2: 5 stim params + 9 montage = 14
    # Lead 1: 2 features  (is_depth, spacing_mm_norm)
    # Lead 2: 2 features
    # Total: 32
    n_cond_features: int = 32

    # ---------- Model architecture ----------
    base_channels: int = 32
    channel_mult: tuple = (1, 2, 4, 8)       # → 32, 64, 128, 256
    bottleneck_channels: int = 512
    film_hidden_dim: int = 128
    dropout: float = 0.05

    # ---------- Training ----------
    batch_size: int = 64
    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-3
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_weight: float = 1.0
    focal_weight: float = 0.5
    patience: int = 15
    min_lr: float = 1e-6
    preload_to_ram: bool = True

    # ---------- Post-processing ----------
    min_artifact_samples: int = 250  # 1 second at 250 Hz
    merge_gap_ms: float = 300.0      # merge detections within 300 ms

    # ---------- Device ----------
    device: str = field(default_factory=lambda: (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    ))


# =============================================================================
# 2. Conditioning: Montage & Lead Parsers
# =============================================================================

class MontageParser:
    """Parse RNS electrode montage string to numeric vector.

    Montage string format: "0-00|----|+"
        - Characters before first '|': Lead 1 contacts (4 contacts)
        - Characters between '|': Lead 2 contacts (4 contacts)
        - Character after last '|':  Case electrode

    Mapping: '0' → 0 (off), '-' → -1 (cathode), '+' → +1 (anode)

    Output: length-9 vector [L1_c1, L1_c2, L1_c3, L1_c4, L2_c1..c4, case]
    """
    _MAP = {'0': 0.0, '-': -1.0, '+': 1.0}

    @classmethod
    def parse_single(cls, s: str) -> np.ndarray:
        """Parse one montage string → (9,) float32 array."""
        if pd.isna(s) or s == '' or s == 'Disabled':
            return np.zeros(9, dtype=np.float32)
        clean = s.replace('|', '')
        try:
            vec = [cls._MAP[c] for c in clean]
        except KeyError:
            return np.zeros(9, dtype=np.float32)
        if len(vec) != 9:
            return np.zeros(9, dtype=np.float32)
        return np.array(vec, dtype=np.float32)

    @classmethod
    def parse_batch(cls, montage_list: list[str]) -> torch.Tensor:
        """Parse list of montage strings → (B, 9) tensor."""
        return torch.tensor(
            np.stack([cls.parse_single(s) for s in montage_list]),
            dtype=torch.float32
        )


class LeadParser:
    """Parse RNS lead type string to numeric features.

    Lead types:  "D3.5" (depth, 3.5mm spacing), "C10" (cortical, 10mm spacing)

    Output: [is_depth, spacing_mm_normalized]
        - is_depth: 1.0 for depth, 0.0 for cortical
        - spacing_mm_normalized: spacing / 10.0 (so 3.5mm→0.35, 10mm→1.0)
    """
    _PATTERN = re.compile(r'([DC])(\d+\.?\d*)')

    @classmethod
    def parse_single(cls, s: str) -> np.ndarray:
        """Parse one lead string → (2,) float32 array."""
        if pd.isna(s) or s == '':
            return np.zeros(2, dtype=np.float32)
        m = cls._PATTERN.match(str(s).strip())
        if not m:
            return np.zeros(2, dtype=np.float32)
        is_depth = 1.0 if m.group(1) == 'D' else 0.0
        spacing = float(m.group(2)) / 10.0  # normalize
        return np.array([is_depth, spacing], dtype=np.float32)


# =============================================================================
# 3. Stim Parameter Extractor
# =============================================================================

# Normalization ranges for stim parameters (based on RNS specs + data)
PARAM_RANGES = {
    'stim_current_mA':        (0.0, 12.0),
    'stim_pulse_width_uS':    (40.0, 400.0),
    'stim_charge_density_uC': (0.0, 50.0),
    'stim_frequency_Hz':      (1.0, 333.0),
    'stim_duration_ms':       (10.0, 5000.0),
}

STIM_PARAM_NAMES = [
    'stim_current_mA',
    'stim_pulse_width_uS',
    'stim_charge_density_uC',
    'stim_frequency_Hz',
    'stim_duration_ms',
]


def extract_conditioning_vector(row: pd.Series) -> np.ndarray:
    """Extract full conditioning vector from a CSV row.

    Returns (32,) float32 array:
        [0:5]   B1 stim params (normalized)
        [5:14]  B1 montage (raw ±1/0)
        [14:19] B2 stim params (normalized)
        [19:28] B2 montage (raw ±1/0)
        [28:30] Lead 1 features
        [30:32] Lead 2 features
    """
    parts = []

    # --- Bank 1 stim params ---
    b1_params = []
    for name in STIM_PARAM_NAMES:
        val = row.get(f'T1_B1_{name}', 0.0)
        if pd.isna(val):
            val = 0.0
        lo, hi = PARAM_RANGES[name]
        b1_params.append(np.clip((val - lo) / (hi - lo + 1e-8), 0.0, 1.0))
    parts.append(np.array(b1_params, dtype=np.float32))

    # --- Bank 1 montage ---
    b1_cfg = row.get('T1_B1_electrode_config', '')
    parts.append(MontageParser.parse_single(b1_cfg))

    # --- Bank 2 stim params ---
    b2_params = []
    for name in STIM_PARAM_NAMES:
        val = row.get(f'T1_B2_{name}', 0.0)
        if pd.isna(val):
            val = 0.0
        lo, hi = PARAM_RANGES[name]
        b2_params.append(np.clip((val - lo) / (hi - lo + 1e-8), 0.0, 1.0))
    parts.append(np.array(b2_params, dtype=np.float32))

    # --- Bank 2 montage ---
    b2_cfg = row.get('T1_B2_electrode_config', '')
    parts.append(MontageParser.parse_single(b2_cfg))

    # --- Lead geometry ---
    parts.append(LeadParser.parse_single(row.get('LEAD_1', '')))
    parts.append(LeadParser.parse_single(row.get('LEAD_2', '')))

    return np.concatenate(parts)  # (32,)


# =============================================================================
# 4. FiLM Generator
# =============================================================================

class FiLMGenerator(nn.Module):
    """Generate γ (scale) and β (shift) for all conditioned layers.

    Single shared MLP → split into per-layer (γ, β) pairs.
    Initialized so γ≈1, β≈0 (identity conditioning at init).
    """

    def __init__(self, n_cond: int, channel_dims: list[int], hidden_dim: int = 128):
        super().__init__()
        self.channel_dims = channel_dims
        total_out = 2 * sum(channel_dims)  # γ + β per layer

        self.mlp = nn.Sequential(
            nn.Linear(n_cond, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, total_out),
        )
        self._init_identity()

    def _init_identity(self):
        """Initialize last layer so γ=1, β=0 (no effect at start)."""
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        total_gamma = sum(self.channel_dims)
        with torch.no_grad():
            last.bias[:total_gamma] = 1.0
            last.bias[total_gamma:] = 0.0

    def forward(self, cond: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            cond: (B, n_cond) conditioning vector
        Returns:
            List of (γ, β) tuples, one per layer. Each γ,β has shape (B, C_layer).
        """
        out = self.mlp(cond)
        total_gamma = sum(self.channel_dims)

        pairs = []
        g_off, b_off = 0, total_gamma
        for dim in self.channel_dims:
            gamma = out[:, g_off:g_off + dim]
            beta = out[:, b_off:b_off + dim]
            pairs.append((gamma, beta))
            g_off += dim
            b_off += dim
        return pairs


# =============================================================================
# 5. Model Blocks
# =============================================================================

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution (faster on M-series chips)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7,
                 dilation: int = 1, bias: bool = True):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size, padding=pad,
            dilation=dilation, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ConditionalBlock(nn.Module):
    """Conv → BatchNorm → FiLM → Dropout → SiLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7,
                 dilation: int = 1, dropout: float = 0.05):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor,
                beta: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        # FiLM: γ ⊙ x + β  (broadcast over time dimension)
        x = gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
        x = self.drop(x)
        x = self.act(x)
        return x


class EncoderLevel(nn.Module):
    """Two conditional blocks + max-pool downsampling."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.05):
        super().__init__()
        self.block1 = ConditionalBlock(in_ch, out_ch, dropout=dropout)
        self.block2 = ConditionalBlock(out_ch, out_ch, dropout=dropout)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, film1: tuple, film2: tuple):
        x = self.block1(x, *film1)
        x = self.block2(x, *film2)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderLevel(nn.Module):
    """Upsample + concat skip + two conditional blocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.05):
        super().__init__()
        self.block1 = ConditionalBlock(in_ch + skip_ch, out_ch, dropout=dropout)
        self.block2 = ConditionalBlock(out_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                film1: tuple, film2: tuple):
        x = F.interpolate(x, size=skip.shape[-1], mode='linear',
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, *film1)
        x = self.block2(x, *film2)
        return x


class Bottleneck(nn.Module):
    """Two dilated conditional blocks at deepest level for wider receptive field."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.05):
        super().__init__()
        self.block1 = ConditionalBlock(in_ch, out_ch, dilation=1, dropout=dropout)
        self.block2 = ConditionalBlock(out_ch, out_ch, dilation=2, dropout=dropout)

    def forward(self, x: torch.Tensor, film1: tuple, film2: tuple):
        x = self.block1(x, *film1)
        x = self.block2(x, *film2)
        return x


# =============================================================================
# 6. Conditional U-Net
# =============================================================================

class StimArtifactUNet(nn.Module):
    """Multi-Scale Conditional 1D U-Net with FiLM for stim artifact detection.

    Architecture overview:
        Input: (B, 4, L) ECoG signal
        Conditioning: (B, 32) stim/montage/lead features → FiLM

        Encoder:  4→32→64→128→256  (4 levels, each 2 conditioned blocks + pool)
        Bottleneck: 256→512          (2 dilated conditioned blocks)
        Decoder:  512→256→128→64→32  (4 levels, skip + 2 conditioned blocks)
        Head:     32→1               (1×1 conv + sigmoid)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        base = cfg.base_channels
        mult = cfg.channel_mult

        enc_channels = [base * m for m in mult]   # [32, 64, 128, 256]
        dec_channels = enc_channels[::-1]          # [256, 128, 64, 32]

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        in_ch = cfg.n_channels
        for out_ch in enc_channels:
            self.encoders.append(EncoderLevel(in_ch, out_ch, cfg.dropout))
            in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = Bottleneck(enc_channels[-1], cfg.bottleneck_channels,
                                     cfg.dropout)

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        in_ch = cfg.bottleneck_channels
        for i, out_ch in enumerate(dec_channels):
            skip_ch = enc_channels[-(i + 1)]
            self.decoders.append(DecoderLevel(in_ch, skip_ch, out_ch, cfg.dropout))
            in_ch = out_ch

        # --- Output head ---
        self.head = nn.Conv1d(dec_channels[-1], 1, kernel_size=1)

        # --- FiLM generator ---
        # 2 blocks per encoder level + 2 bottleneck + 2 per decoder level
        film_dims = []
        for ch in enc_channels:
            film_dims.extend([ch, ch])
        film_dims.extend([cfg.bottleneck_channels, cfg.bottleneck_channels])
        for ch in dec_channels:
            film_dims.extend([ch, ch])

        self.film_gen = FiLMGenerator(cfg.n_cond_features, film_dims,
                                       cfg.film_hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, 4, L)  ECoG robust-scaled with file median & IQR
            cond: (B, 32)    conditioning vector
        Returns:
            (B, 1, L) artifact probability mask
        """
        films = self.film_gen(cond)
        fi = 0  # film index counter

        # Encoder
        skips = []
        for enc in self.encoders:
            x, skip = enc(x, films[fi], films[fi + 1])
            skips.append(skip)
            fi += 2

        # Bottleneck
        x = self.bottleneck(x, films[fi], films[fi + 1])
        fi += 2

        # Decoder
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)], films[fi], films[fi + 1])
            fi += 2

        return torch.sigmoid(self.head(x))


# =============================================================================
# 7. Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance (artifact vs background)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation overlap."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.view(-1)
        t = target.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """Weighted sum of Focal and Dice losses."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.focal = FocalLoss(cfg.focal_alpha, cfg.focal_gamma)
        self.dice = DiceLoss()
        self.fw = cfg.focal_weight
        self.dw = cfg.dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        fl = self.focal(pred, target)
        dl = self.dice(pred, target)
        return {'loss': self.fw * fl + self.dw * dl, 'focal': fl, 'dice': dl}


# =============================================================================
# 8. Dataset
# =============================================================================

def parse_onset_vector(onset_str: str) -> np.ndarray:
    """Parse Julia-formatted onset vector from CSV.

    Examples:
        "Union{Missing, Float64}[17.776, 66.056]" → array([17.776, 66.056])
        "[0.208]" → array([0.208])
    """
    if pd.isna(onset_str) or onset_str == '':
        return np.array([], dtype=np.float64)
    m = re.search(r'\[(.*?)\]', str(onset_str))
    if not m:
        return np.array([], dtype=np.float64)
    vals = []
    for v in m.group(1).split(','):
        v = v.strip()
        if v and v.lower() != 'missing':
            try:
                vals.append(float(v))
            except ValueError:
                continue
    return np.array(vals, dtype=np.float64)


def _robust_scale_stats(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute median and IQR per channel for robust global scaling.

    data: (n_channels, n_samples) float32
    Returns: (median, iqr) each (n_channels,) for use as (x - median) / (iqr + eps).
    """
    median = np.median(data, axis=1).astype(np.float32)  # (4,)
    q1 = np.percentile(data, 25, axis=1).astype(np.float32)
    q3 = np.percentile(data, 75, axis=1).astype(np.float32)
    iqr = (q3 - q1).astype(np.float32)
    return median, iqr


class StimArtifactDataset(Dataset):
    """Sliding-window dataset for stim artifact detection.

    Each sample is a (window_samples,) slice of one recording with:
        - signal: (4, window_samples) robust-scaled ECoG (using file median & IQR)
        - mask:   (1, window_samples) binary artifact mask
        - cond:   (32,) conditioning features

    Normalization uses the entire recording's median and IQR per channel,
    not the window's statistics (robust global scaling).
    """

    def __init__(self, df: pd.DataFrame, file_map: dict[str, Path],
                 cfg: Config, augment: bool = True):
        self.cfg = cfg
        self.augment = augment
        self.data_cache: dict[str, np.ndarray] = {}
        self.file_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # path -> (median, iqr)
        self.samples: list[dict] = []

        # Pre-load H5 files into RAM and compute file-level stats (median, IQR)
        if cfg.preload_to_ram:
            unique_paths = set()
            for _, row in df.iterrows():
                fid = row['file_id']
                if fid in file_map:
                    unique_paths.add(file_map[fid])
            print(f"Pre-loading {len(unique_paths)} H5 files into RAM...")
            for p in tqdm(unique_paths, desc="Caching"):
                self._cache_file(p)
            mem_gb = sum(a.nbytes for a in self.data_cache.values()) / 1e9
            print(f"Cached {len(self.data_cache)} files ({mem_gb:.2f} GB)")

        # Build sample index
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building samples"):
            fid = row['file_id']
            if fid not in file_map:
                continue
            h5_path = file_map[fid]
            onsets = row['onset_times']
            if len(onsets) == 0:
                continue

            # Recording length
            if cfg.preload_to_ram and str(h5_path) in self.data_cache:
                n_samples = self.data_cache[str(h5_path)].shape[1]
            else:
                with h5py.File(h5_path, 'r') as f:
                    n_samples = f['channel_1'].shape[0]

            # Mask duration (from CSV, minimum 250 samples = 1s)
            mask_dur_ms = row.get('mask_duration_ms', 1000.0)
            if pd.isna(mask_dur_ms):
                mask_dur_ms = 1000.0
            mask_dur_samples = max(
                int(mask_dur_ms / 1000.0 * cfg.sampling_rate),
                cfg.min_artifact_samples
            )

            # Conditioning vector
            cond = extract_conditioning_vector(row)

            # Generate windows
            for start in range(0, max(1, n_samples - cfg.window_samples),
                               cfg.stride_samples):
                end = start + cfg.window_samples
                if end > n_samples:
                    break
                self.samples.append({
                    'h5_path': h5_path,
                    'file_id': fid,
                    'rec_len': n_samples,
                    'start': start,
                    'end': end,
                    'onsets': onsets,
                    'mask_dur': mask_dur_samples,
                    'cond': cond,
                })

        print(f"Total samples: {len(self.samples)}")

        # When not preloading, compute file-level stats (median, IQR) for each unique file
        if not cfg.preload_to_ram:
            unique_paths = {s['h5_path'] for s in self.samples}
            for p in tqdm(unique_paths, desc="File stats"):
                key = str(p)
                if key not in self.file_stats:
                    with h5py.File(p, 'r') as f:
                        data = np.stack([
                            f[f'channel_{i}'][:].astype(np.float32)
                            for i in range(1, 5)
                        ], axis=0)
                    self.file_stats[key] = _robust_scale_stats(data)

    def _cache_file(self, h5_path: Path) -> None:
        with h5py.File(h5_path, 'r') as f:
            data = np.stack([
                f[f'channel_{i}'][:].astype(np.float32)
                for i in range(1, 5)
            ], axis=0)  # (4, T)
        self.data_cache[str(h5_path)] = data
        self.file_stats[str(h5_path)] = _robust_scale_stats(data)

    def _make_mask(self, onsets: np.ndarray, start: int, end: int,
                   mask_dur: int) -> np.ndarray:
        """Create binary artifact mask for a window."""
        length = end - start
        mask = np.zeros(length, dtype=np.float32)
        fs = self.cfg.sampling_rate
        for t_sec in onsets:
            onset_sample = int(t_sec * fs)
            a = max(0, onset_sample - start)
            b = min(length, onset_sample - start + mask_dur)
            if a < length and b > 0:
                mask[a:b] = 1.0
        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        key = str(s['h5_path'])

        # Load signal
        if key in self.data_cache:
            sig = self.data_cache[key][:, s['start']:s['end']].copy()
        else:
            with h5py.File(s['h5_path'], 'r') as f:
                sig = np.stack([
                    f[f'channel_{i}'][s['start']:s['end']].astype(np.float32)
                    for i in range(1, 5)
                ], axis=0)

        # Robust global scaling: normalize using file's median and IQR (not window stats)
        median, iqr = self.file_stats[key]
        sig = (sig - median[:, np.newaxis]) / (iqr[:, np.newaxis] + 1e-8)

        # Mask
        mask = self._make_mask(s['onsets'], s['start'], s['end'], s['mask_dur'])

        # Augmentation (training only)
        if self.augment:
            sig, mask = self._augment(sig, mask)

        return {
            'signal': torch.from_numpy(sig),
            'mask': torch.from_numpy(mask).unsqueeze(0),
            'cond': torch.from_numpy(s['cond']),
            'file_id': s['file_id'],
            'start': torch.tensor(s['start'], dtype=torch.long),
            'end': torch.tensor(s['end'], dtype=torch.long),
            'rec_len': torch.tensor(s['rec_len'], dtype=torch.long),
        }

    def _augment(self, sig: np.ndarray, mask: np.ndarray):
        """Augmentations that preserve temporal alignment."""
        # Amplitude jitter
        if np.random.random() < 0.5:
            sig *= np.random.uniform(0.8, 1.2)
        # Additive Gaussian noise
        if np.random.random() < 0.3:
            sig += np.random.normal(0, 0.05, sig.shape).astype(np.float32)
        # Channel dropout (zero one channel)
        if np.random.random() < 0.1:
            sig[np.random.randint(0, 4)] = 0.0
        # Time flip (flip both signal and mask)
        if np.random.random() < 0.2:
            sig = sig[:, ::-1].copy()
            mask = mask[::-1].copy()
        return sig, mask


# =============================================================================
# 9. Data Loading Utilities
# =============================================================================

def load_annotations(cfg: Config) -> pd.DataFrame:
    """Load CSV, parse onsets, add file_id."""
    df = pd.read_csv(cfg.annotations_path)
    df['file_id'] = df['LAY_FILENAME'].str.replace('.lay', '', regex=False)
    df['onset_times'] = df['START_AT_VECTOR'].apply(parse_onset_vector)
    df = df[df['onset_times'].apply(len) > 0].reset_index(drop=True)
    return df


def find_h5_files(cfg: Config) -> dict[str, Path]:
    """Map file_id → H5 path. Checks both filename stem and source_file attr."""
    h5_map: dict[str, Path] = {}
    for p in cfg.data_dir.glob("*.h5"):
        h5_map[p.stem] = p
        try:
            with h5py.File(p, 'r') as f:
                src = f.attrs.get('source_file', '')
                if isinstance(src, bytes):
                    src = src.decode()
                if src:
                    clean = src.replace('.dat', '').replace('.lay', '')
                    h5_map[clean] = p
        except Exception:
            pass
    return h5_map


# =============================================================================
# 10. Post-Processing
# =============================================================================

def post_process_mask(pred_mask: np.ndarray, cfg: Config) -> np.ndarray:
    """Post-process predicted binary mask.

    1. Morphological closing to merge nearby detections (within merge_gap_ms).
    2. Remove detections shorter than min_artifact_samples (250 frames).
    """
    gap_samples = int(cfg.merge_gap_ms / 1000.0 * cfg.sampling_rate)
    struct = np.ones(max(gap_samples, 1))
    closed = scipy.ndimage.binary_closing(pred_mask, structure=struct).astype(int)

    labeled, n_feat = scipy.ndimage.label(closed)
    result = np.zeros_like(closed)
    for i in range(1, n_feat + 1):
        component = (labeled == i)
        if component.sum() >= cfg.min_artifact_samples:
            result[component] = 1
    return result


# =============================================================================
# 11. Metrics
# =============================================================================

def compute_sample_metrics(pred: torch.Tensor, target: torch.Tensor,
                           threshold: float = 0.5) -> dict[str, float]:
    """Per-sample precision / recall / F1 / IoU."""
    pred_bin = (pred > threshold).float()
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {'precision': prec, 'recall': rec, 'f1': f1, 'iou': iou}


def compute_event_metrics(pred_mask: np.ndarray, true_mask: np.ndarray,
                          min_iou: float = 0.3) -> dict[str, float]:
    """Event-level precision / recall / F1 using IoU overlap."""
    from scipy.ndimage import label, find_objects

    pred_lab, n_pred = label(pred_mask)
    true_lab, n_true = label(true_mask)

    if n_pred == 0 and n_true == 0:
        return {'event_precision': 1.0, 'event_recall': 1.0, 'event_f1': 1.0}
    if n_pred == 0:
        return {'event_precision': 1.0, 'event_recall': 0.0, 'event_f1': 0.0}
    if n_true == 0:
        return {'event_precision': 0.0, 'event_recall': 0.0, 'event_f1': 0.0}

    pred_slices = find_objects(pred_lab)
    true_slices = find_objects(true_lab)

    tp = 0
    matched = set()
    for i, ps in enumerate(pred_slices):
        best_iou, best_j = 0.0, -1
        pm = (pred_lab == (i + 1))
        for j, ts in enumerate(true_slices):
            if j in matched:
                continue
            # Quick bounding-box check
            if ps[0].start >= ts[0].stop or ps[0].stop <= ts[0].start:
                continue
            tm = (true_lab == (j + 1))
            inter = np.logical_and(pm, tm).sum()
            union = np.logical_or(pm, tm).sum()
            iou = inter / (union + 1e-8)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= min_iou:
            tp += 1
            matched.add(best_j)

    fp = n_pred - tp
    fn = n_true - len(matched)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {'event_precision': prec, 'event_recall': rec, 'event_f1': f1}


# =============================================================================
# 12. Trainer
# =============================================================================

class Trainer:
    """Training loop with validation stitching, early stopping, and LR scheduling."""

    def __init__(self, model: StimArtifactUNet, cfg: Config):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.device = cfg.device

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5,
            min_lr=cfg.min_lr
        )
        self.criterion = CombinedLoss(cfg)
        self.best_f1 = 0.0

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # torch.compile not yet stable on MPS
        if hasattr(torch, 'compile') and cfg.device == 'cuda':
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        all_pred, all_tgt = [], []

        pbar = tqdm(loader, desc="  Train")
        for i, batch in enumerate(pbar):
            sig = batch['signal'].to(self.device, non_blocking=True)
            msk = batch['mask'].to(self.device, non_blocking=True)
            cnd = batch['cond'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(sig, cnd)
            losses = self.criterion(pred, msk)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += losses['loss'].detach()
            all_pred.append(pred.detach().cpu())
            all_tgt.append(msk.detach().cpu())

            if i % 20 == 0:
                pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")

        metrics = compute_sample_metrics(torch.cat(all_pred), torch.cat(all_tgt))
        metrics['loss'] = total_loss.item() / len(loader)
        return metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        """Validate with whole-file prediction stitching + post-processing."""
        self.model.eval()
        total_loss = 0.0

        # Collect per-file windows
        file_wins: dict[str, list[dict]] = {}
        file_lens: dict[str, int] = {}

        for batch in tqdm(loader, desc="  Val  "):
            sig = batch['signal'].to(self.device)
            msk = batch['mask'].to(self.device)
            cnd = batch['cond'].to(self.device)

            pred = self.model(sig, cnd)
            total_loss += self.criterion(pred, msk)['loss'].item()

            pred_np = pred.cpu().numpy().squeeze(1)  # (B, L)
            mask_np = msk.cpu().numpy().squeeze(1)

            for i in range(len(pred_np)):
                fid = batch['file_id'][i]
                if fid not in file_wins:
                    file_wins[fid] = []
                    file_lens[fid] = batch['rec_len'][i].item()
                file_wins[fid].append({
                    'pred': pred_np[i], 'mask': mask_np[i],
                    'start': batch['start'][i].item(),
                    'end': batch['end'][i].item(),
                })

        # Stitch predictions per file
        sample_metrics_list = []
        event_metrics_list = []

        for fid, wins in file_wins.items():
            rlen = file_lens[fid]
            pred_acc = np.zeros(rlen, dtype=np.float32)
            mask_acc = np.zeros(rlen, dtype=np.float32)
            wgt = np.zeros(rlen, dtype=np.float32)

            for w in wins:
                s, e = w['start'], w['end']
                pred_acc[s:e] += w['pred']
                mask_acc[s:e] = w['mask']  # binary, overwrite is fine
                wgt[s:e] += 1.0

            pred_avg = np.divide(pred_acc, wgt, out=np.zeros_like(pred_acc),
                                 where=wgt > 0)

            # Sample-level metrics (raw)
            sm = compute_sample_metrics(
                torch.from_numpy(pred_avg).unsqueeze(0),
                torch.from_numpy(mask_acc).unsqueeze(0),
            )
            sample_metrics_list.append(sm)

            # Event-level metrics (post-processed)
            pred_bin = post_process_mask((pred_avg > 0.5).astype(int), self.cfg)
            em = compute_event_metrics(pred_bin, mask_acc.astype(int))
            event_metrics_list.append(em)

        # Aggregate
        agg = {}
        for key in sample_metrics_list[0]:
            agg[key] = np.mean([m[key] for m in sample_metrics_list])
        for key in event_metrics_list[0]:
            agg[key] = np.mean([m[key] for m in event_metrics_list])
        agg['loss'] = total_loss / max(len(loader), 1)
        return agg

    @torch.no_grad()
    def predict_file(self, h5_path: Path, cond: np.ndarray) -> np.ndarray:
        """Run inference on a whole file with overlapping windows.

        Uses robust global scaling: file median and IQR to normalize each window.
        Returns full-length probability array.
        """
        self.model.eval()
        with h5py.File(h5_path, 'r') as f:
            data = np.stack([
                f[f'channel_{i}'][:].astype(np.float32)
                for i in range(1, 5)
            ], axis=0)

        median, iqr = _robust_scale_stats(data)
        rlen = data.shape[1]
        ws = self.cfg.window_samples
        stride = max(1, int(ws * self.cfg.inference_stride_ratio))

        pred_acc = np.zeros(rlen, dtype=np.float32)
        wgt = np.zeros(rlen, dtype=np.float32)
        cond_t = torch.from_numpy(cond).unsqueeze(0).to(self.device)

        for s in range(0, rlen - ws + 1, stride):
            e = s + ws
            win = data[:, s:e].copy()
            win = (win - median[:, np.newaxis]) / (iqr[:, np.newaxis] + 1e-8)

            sig_t = torch.from_numpy(win).unsqueeze(0).to(self.device)
            p = self.model(sig_t, cond_t).cpu().numpy().squeeze()
            pred_acc[s:e] += p
            wgt[s:e] += 1.0

        return np.divide(pred_acc, wgt, out=np.zeros_like(pred_acc),
                         where=wgt > 0)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Device: {self.device}")
        print(f"Model parameters: {n_params:,}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        patience_ctr = 0

        for epoch in range(self.cfg.epochs):
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.cfg.epochs}  (lr={lr:.2e})")
            print(f"{'='*60}")

            train_m = self.train_epoch(train_loader)
            val_m = self.validate(val_loader)

            self.scheduler.step(val_m['f1'])

            print(f"  Train │ loss={train_m['loss']:.4f}  F1={train_m['f1']:.4f}"
                  f"  P={train_m['precision']:.4f}  R={train_m['recall']:.4f}")
            print(f"  Val   │ loss={val_m['loss']:.4f}  F1={val_m['f1']:.4f}"
                  f"  P={val_m['precision']:.4f}  R={val_m['recall']:.4f}")
            print(f"  Event │ F1={val_m['event_f1']:.4f}"
                  f"  P={val_m['event_precision']:.4f}"
                  f"  R={val_m['event_recall']:.4f}")

            if val_m['f1'] > self.best_f1:
                self.best_f1 = val_m['f1']
                patience_ctr = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.cfg,
                    'f1': self.best_f1,
                }, self.cfg.checkpoint_dir / 'best_model.pt')
                print(f"  ✓ New best (F1={self.best_f1:.4f})")
            else:
                patience_ctr += 1
                print(f"  ✗ No improvement ({patience_ctr}/{self.cfg.patience})")
                if patience_ctr >= self.cfg.patience:
                    print(f"\nEarly stopping. Best F1: {self.best_f1:.4f}")
                    break


# =============================================================================
# 13. Main
# =============================================================================

def main():
    cfg = Config()

    # --- Load annotations ---
    print("Loading annotations...")
    df = load_annotations(cfg)
    print(f"  {len(df)} annotated recordings")

    # --- Find H5 files ---
    print("Finding H5 files...")
    file_map = find_h5_files(cfg)
    print(f"  {len(file_map)} H5 files found")

    # --- Split by subject (prevent leakage) ---
    # Use SUBJECT_ID_LR as group key for GroupShuffleSplit
    groups = df['SUBJECT_ID_LR'].values if 'SUBJECT_ID_LR' in df.columns \
        else df['file_id'].values

    # For single-subject datasets, split by file_id instead
    unique_groups = np.unique(groups)
    if len(unique_groups) == 1:
        print("  Single subject detected → splitting by file_id")
        groups = df['file_id'].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(f"  Train: {len(train_df)} recordings, Val: {len(val_df)} recordings")

    # --- Build datasets ---
    print("Building datasets...")
    train_ds = StimArtifactDataset(train_df, file_map, cfg, augment=True)
    val_ds = StimArtifactDataset(val_df, file_map, cfg, augment=False)

    # Dataloader kwargs (pin_memory only for CUDA)
    loader_kw = {'num_workers': 0}  # RAM-cached → main thread is fastest
    if cfg.device == 'cuda':
        loader_kw['pin_memory'] = True

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2,
                            shuffle=False, **loader_kw)

    # --- Create model & train ---
    model = StimArtifactUNet(cfg)
    trainer = Trainer(model, cfg)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()