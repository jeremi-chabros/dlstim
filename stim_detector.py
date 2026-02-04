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
# ]
# ///
"""
Multi-Scale Conditional 1D U-Net with FiLM for Stimulation Artifact Detection
Optimized for Apple Silicon (MPS) - M4 Pro
"""

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
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class Config:
    """Full configuration for model, data, and training."""
    # Paths
    data_dir: Path = Path("/Users/jch/Postdoc/Projects/stim detection/dlstim/processed_data")
    annotations_path: Path = Path("/Users/jch/Postdoc/Projects/stim detection/dlstim/consolidated_stim_data.csv")
    checkpoint_dir: Path = Path("./checkpoints")
    
    # Signal parameters
    sampling_rate: int = 250
    n_channels: int = 4
    window_samples: int = 512  # ~2 seconds at 250 Hz
    stride_samples: int = 128  # ~0.5 second stride for training
    
    # Stimulation parameter columns (must match CSV)
    stim_param_cols: tuple = (
        'B1_stim_current_mA',
        'B1_stim_pulse_width_uS',
        'B1_stim_charge_density_uC',
        'B1_stim_frequency_Hz',
        'B1_stim_duration_ms',
    )
    n_stim_params: int = 5
    
    # Model architecture
    base_channels: int = 32
    channel_mult: tuple = (1, 2, 4, 8)  # 32, 64, 128, 256
    bottleneck_channels: int = 512
    film_hidden_dim: int = 128
    dropout: float = 0.3
    
    # Training — optimized for M4 Pro 48GB
    batch_size: int = 128          # Larger batch for GPU utilization
    epochs: int = 100
    lr: float = 1e-3

    weight_decay: float = 1e-3
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    num_workers: int = 8           # Use performance cores
    prefetch_factor: int = 4       # Pre-load batches
    preload_to_ram: bool = True    # Cache all H5 data in RAM
    patience: int = 10
    
    # Artifact labeling
    artifact_duration_ms: float = 200.0  # Default if not specified
    
    # Device
    device: str = field(default_factory=lambda: (
        "mps" if torch.backends.mps.is_available() 
        else "cuda" if torch.cuda.is_available() 
        else "cpu"
    ))


# =============================================================================
# 2. FiLM Generator
# =============================================================================

class FiLMGenerator(nn.Module):
    """
    Generates γ (scale) and β (shift) parameters for all FiLM layers.
    Single MLP → split output to match each block's channel count.
    """
    def __init__(self, n_params: int, channel_dims: list[int], hidden_dim: int = 128):
        super().__init__()
        self.channel_dims = channel_dims
        total_film_params = 2 * sum(channel_dims)  # γ + β for each
        
        self.mlp = nn.Sequential(
            nn.Linear(n_params, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, total_film_params),
        )
        
        # Initialize final layer for γ≈1, β≈0
        self._init_weights()
    
    def _init_weights(self):
        final_layer = self.mlp[-1]
        nn.init.zeros_(final_layer.weight)
        # Set bias: first half (γ) to 1, second half (β) to 0
        total_gamma = sum(self.channel_dims)
        with torch.no_grad():
            final_layer.bias[:total_gamma] = 1.0
            final_layer.bias[total_gamma:] = 0.0
    
    def forward(self, params: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            params: (B, n_params) normalized stimulation parameters
        Returns:
            List of (γ, β) tuples, one per FiLM layer
        """
        out = self.mlp(params)  # (B, total_film_params)
        
        film_params = []
        total_gamma = sum(self.channel_dims)
        gamma_offset, beta_offset = 0, total_gamma
        
        for dim in self.channel_dims:
            gamma = out[:, gamma_offset:gamma_offset + dim]
            beta = out[:, beta_offset:beta_offset + dim]
            film_params.append((gamma, beta))
            gamma_offset += dim
            beta_offset += dim
        
        return film_params


# =============================================================================
# 3. Conditional Block (Conv + BN + FiLM + Activation)
# =============================================================================

class ConditionalBlock(nn.Module):
    """Single convolutional block with FiLM conditioning."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)  # Smoother than LeakyReLU
    
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L)
            gamma: (B, C_out) - scale
            beta: (B, C_out) - shift
        """
        x = self.conv(x)
        x = self.bn(x)
        # FiLM: reshape gamma/beta to (B, C, 1) for broadcasting
        x = gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
        x = self.dropout(x)
        x = self.act(x)
        return x


# =============================================================================
# 4. Encoder & Decoder Levels
# =============================================================================

class EncoderLevel(nn.Module):
    """Two conditional blocks + downsampling."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
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
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block1 = ConditionalBlock(in_ch + skip_ch, out_ch, dropout=dropout)
        self.block2 = ConditionalBlock(out_ch, out_ch, dropout=dropout)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, film1: tuple, film2: tuple):
        x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, *film1)
        x = self.block2(x, *film2)
        return x


class Bottleneck(nn.Module):
    """Two conditional blocks at the deepest level."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block1 = ConditionalBlock(in_ch, out_ch, dropout=dropout)
        self.block2 = ConditionalBlock(out_ch, out_ch, dropout=dropout)
    
    def forward(self, x: torch.Tensor, film1: tuple, film2: tuple):
        x = self.block1(x, *film1)
        x = self.block2(x, *film2)
        return x


# =============================================================================
# 5. Multi-Scale Conditional U-Net
# =============================================================================

class ConditionalUNet(nn.Module):
    """
    Multi-Scale Conditional 1D U-Net with FiLM.
    
    Architecture:
        Encoder: 4 levels (4→32→64→128→256)
        Bottleneck: 256→512
        Decoder: 4 levels (512→256→128→64→32)
        Head: 32→1
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        base = cfg.base_channels
        mult = cfg.channel_mult
        
        # Channel dimensions at each level
        enc_channels = [base * m for m in mult]  # [32, 64, 128, 256]
        dec_channels = enc_channels[::-1]        # [256, 128, 64, 32]
        
        # Build encoder
        self.encoders = nn.ModuleList()
        in_ch = cfg.n_channels
        for out_ch in enc_channels:
            self.encoders.append(EncoderLevel(in_ch, out_ch, cfg.dropout))
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = Bottleneck(enc_channels[-1], cfg.bottleneck_channels, cfg.dropout)
        
        # Build decoder
        self.decoders = nn.ModuleList()
        in_ch = cfg.bottleneck_channels
        for i, out_ch in enumerate(dec_channels):
            skip_ch = enc_channels[-(i+1)]
            self.decoders.append(DecoderLevel(in_ch, skip_ch, out_ch, cfg.dropout))
            in_ch = out_ch
        
        # Output head
        self.head = nn.Conv1d(dec_channels[-1], 1, kernel_size=1)
        
        # Compute FiLM channel dimensions (2 blocks per level)
        film_dims = []
        # Encoder: 2 blocks per level
        for ch in enc_channels:
            film_dims.extend([ch, ch])
        # Bottleneck: 2 blocks
        film_dims.extend([cfg.bottleneck_channels, cfg.bottleneck_channels])
        # Decoder: 2 blocks per level
        for ch in dec_channels:
            film_dims.extend([ch, ch])
        
        self.film_generator = FiLMGenerator(cfg.n_stim_params, film_dims, cfg.film_hidden_dim)
        self.n_film_layers = len(film_dims)
    
    def forward(self, x: torch.Tensor, stim_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, L) - raw signal, Z-scored
            stim_params: (B, n_stim_params) - normalized stim parameters
        Returns:
            (B, 1, L) - artifact probability mask
        """
        # Generate all FiLM parameters
        film_params = self.film_generator(stim_params)
        film_idx = 0
        
        # Encoder path
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x, film_params[film_idx], film_params[film_idx + 1])
            skips.append(skip)
            film_idx += 2
        
        # Bottleneck
        x = self.bottleneck(x, film_params[film_idx], film_params[film_idx + 1])
        film_idx += 2
        
        # Decoder path
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = decoder(x, skip, film_params[film_idx], film_params[film_idx + 1])
            film_idx += 2
        
        # Output
        x = self.head(x)
        x = torch.sigmoid(x)
        return x


# =============================================================================
# 6. Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceLoss(nn.Module):
    """Soft Dice Loss for segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class CombinedLoss(nn.Module):
    """Focal + Dice combined loss."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.focal = FocalLoss(cfg.focal_alpha, cfg.focal_gamma)
        self.dice = DiceLoss()
        self.focal_weight = cfg.focal_weight
        self.dice_weight = cfg.dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        focal = self.focal(pred, target)
        dice = self.dice(pred, target)
        total = self.focal_weight * focal + self.dice_weight * dice
        return {'loss': total, 'focal': focal, 'dice': dice}


# =============================================================================
# 7. Dataset
# =============================================================================

class StimArtifactDataset(Dataset):
    """Dataset for stimulation artifact detection with optional RAM caching."""
    
    # Normalization constants for stim parameters (adjust based on your data)
    PARAM_RANGES = {
        'B1_stim_current_mA': (0.0, 12.0),
        'B1_stim_pulse_width_uS': (40.0, 400.0),
        'B1_stim_charge_density_uC': (0.0, 50.0),
        'B1_stim_frequency_Hz': (1.0, 333.0),
        'B1_stim_duration_ms': (10.0, 5000.0),
    }
    
    def __init__(self, df: pd.DataFrame, file_map: dict[str, Path], cfg: Config, augment: bool = True):
        self.cfg = cfg
        self.augment = augment
        self.samples = []
        self.data_cache: dict[str, np.ndarray] = {}  # RAM cache for H5 data
        
        # Pre-load all H5 files into RAM if enabled
        if cfg.preload_to_ram:
            print("Pre-loading H5 files into RAM...")
            unique_paths = set()
            for _, row in df.iterrows():
                if row['file_id'] in file_map:
                    unique_paths.add(file_map[row['file_id']])
            
            for h5_path in tqdm(unique_paths, desc="Caching data"):
                self._cache_file(h5_path)
            
            mem_gb = sum(arr.nbytes for arr in self.data_cache.values()) / 1e9
            print(f"Cached {len(self.data_cache)} files ({mem_gb:.2f} GB)")
        
        # Build sample index
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset"):
            file_id = row['file_id']
            if file_id not in file_map:
                continue
            
            h5_path = file_map[file_id]
            onset_times = row['onset_times']
            
            if len(onset_times) == 0:
                continue
            
            # Get recording length
            if cfg.preload_to_ram and str(h5_path) in self.data_cache:
                n_samples = self.data_cache[str(h5_path)].shape[1]
            else:
                with h5py.File(h5_path, 'r') as f:
                    n_samples = f['channel_1'].shape[0]
            
            # Extract stim parameters
            stim_params = self._extract_stim_params(row)
            
            # Generate windows
            artifact_duration_samples = int(
                row.get('stim_duration_ms', cfg.artifact_duration_ms) / 1000 * cfg.sampling_rate
            )
            
            for start in range(0, n_samples - cfg.window_samples, cfg.stride_samples):
                end = start + cfg.window_samples
                self.samples.append({
                    'h5_path': h5_path,
                    'start': start,
                    'end': end,
                    'onset_times': onset_times,
                    'artifact_duration_samples': artifact_duration_samples,
                    'stim_params': stim_params,
                })
    
    def _cache_file(self, h5_path: Path) -> None:
        """Load entire H5 file into RAM cache."""
        with h5py.File(h5_path, 'r') as f:
            data = np.stack([
                f[f'channel_{i}'][:].astype(np.float32)
                for i in range(1, 5)
            ], axis=0)
        self.data_cache[str(h5_path)] = data
    
    def _extract_stim_params(self, row: pd.Series) -> np.ndarray:
        """Extract and normalize stimulation parameters."""
        params = []
        for col in self.cfg.stim_param_cols:
            val = row.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            # Normalize to [0, 1]
            lo, hi = self.PARAM_RANGES.get(col, (0.0, 1.0))
            norm_val = (val - lo) / (hi - lo + 1e-8)
            params.append(np.clip(norm_val, 0.0, 1.0))
        return np.array(params, dtype=np.float32)
    
    def _create_mask(self, onset_times: np.ndarray, start: int, end: int, 
                     artifact_duration: int) -> np.ndarray:
        """Create binary artifact mask for a window."""
        mask = np.zeros(end - start, dtype=np.float32)
        fs = self.cfg.sampling_rate
        
        for onset_sec in onset_times:
            onset_sample = int(onset_sec * fs)
            # Mark artifact region
            art_start = max(0, onset_sample - start)
            art_end = min(end - start, onset_sample - start + artifact_duration)
            if art_start < end - start and art_end > 0:
                mask[art_start:art_end] = 1.0
        
        return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load signal from cache or disk
        h5_key = str(sample['h5_path'])
        if h5_key in self.data_cache:
            data = self.data_cache[h5_key][:, sample['start']:sample['end']].copy()
        else:
            with h5py.File(sample['h5_path'], 'r') as f:
                data = np.stack([
                    f[f'channel_{i}'][sample['start']:sample['end']].astype(np.float32)
                    for i in range(1, 5)
                ], axis=0)
        
        # Z-score normalization per channel
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True) + 1e-8
        data = (data - mean) / std
        
        # Create mask
        mask = self._create_mask(
            sample['onset_times'],
            sample['start'],
            sample['end'],
            sample['artifact_duration_samples']
        )
        
        # Augmentation
        if self.augment:
            data, mask = self._augment(data, mask)
        
        return {
            'signal': torch.from_numpy(data),
            'mask': torch.from_numpy(mask).unsqueeze(0),
            'stim_params': torch.from_numpy(sample['stim_params']),
        }
    
    def _augment(self, data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply augmentations."""
        # Amplitude scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            data = data * scale
        
        # Gaussian noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.05, data.shape).astype(np.float32)
            data = data + noise
        
        # Channel dropout (zero one channel)
        if np.random.random() < 0.1:
            ch = np.random.randint(0, 4)
            data[ch] = 0.0
        
        return data, mask


# =============================================================================
# 8. Data Loading Utilities (from user's code)
# =============================================================================

def parse_onset_vector(onset_str: str) -> np.ndarray:
    """Parses onset vector string from CSV."""
    if pd.isna(onset_str) or onset_str == "":
        return np.array([])
    match = re.search(r'\[(.*?)\]', str(onset_str))
    if match:
        values = []
        for val in match.group(1).split(','):
            val = val.strip()
            if val and val.lower() != 'missing':
                try:
                    values.append(float(val))
                except ValueError:
                    continue
        return np.array(values)
    return np.array([])


def load_annotations(cfg: Config) -> pd.DataFrame:
    """Load and preprocess annotations CSV."""
    df = pd.read_csv(cfg.annotations_path)
    df['file_id'] = df['LAY_FILENAME'].str.replace('.lay', '', regex=False)
    df['onset_times'] = df['START_AT_VECTOR'].apply(parse_onset_vector)
    df['stim_duration_ms'] = df['B1_stim_duration_ms'].fillna(cfg.artifact_duration_ms)
    df = df[df['onset_times'].apply(len) > 0].reset_index(drop=True)
    return df


def find_h5_files(cfg: Config) -> dict[str, Path]:
    """Find and map H5 files."""
    h5_files = {p.stem: p for p in cfg.data_dir.glob("*.h5")}
    for h5_path in cfg.data_dir.glob("*.h5"):
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'source_file' in f.attrs:
                    src = f.attrs['source_file']
                    if isinstance(src, bytes):
                        src = src.decode()
                    clean_id = src.replace('.dat', '').replace('.lay', '')
                    h5_files[clean_id] = h5_path
        except Exception:
            pass
    return h5_files


# =============================================================================
# 9. Metrics
# =============================================================================

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, 
                    threshold: float = 0.5) -> dict[str, float]:
    """Compute detection metrics."""
    pred_bin = (pred > threshold).float()
    
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()
    tn = ((1 - pred_bin) * (1 - target)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
    }


# =============================================================================
# 10. Training Loop
# =============================================================================

class Trainer:
    """Training manager with MPS optimization, Early Stopping, and Plateau Scheduler."""
    
    def __init__(self, model: nn.Module, cfg: Config):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.device = cfg.device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        
        # CHANGE 1: Switch to ReduceLROnPlateau
        # This lowers LR when the validation F1 score stops improving
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',      # We want to MAXIMIZE F1
            factor=0.5,      # Cut LR by half when stuck
            patience=3,      # Wait 3 epochs before cutting LR
            verbose=True
        )
        
        self.criterion = CombinedLoss(cfg)
        self.best_f1 = 0.0
        
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(torch, 'compile') and cfg.device != "mps":
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
    
    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss, total_focal, total_dice = 0.0, 0.0, 0.0
        all_preds, all_targets = [], []
        
        pbar = tqdm(loader, desc="Training")
        for batch in pbar:
            signal = batch['signal'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)
            stim_params = batch['stim_params'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(signal, stim_params)
            losses = self.criterion(pred, mask)
            losses['loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += losses['loss'].item()
            total_focal += losses['focal'].item()
            total_dice += losses['dice'].item()
            
            # Detach to save memory
            all_preds.append(pred.detach().cpu())
            all_targets.append(mask.detach().cpu())
            
            pbar.set_postfix({'loss': f"{losses['loss'].item():.4f}"})
        
        n = len(loader)
        # Compute metrics on CPU to avoid MPS sync overhead during loop
        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
        metrics.update({
            'loss': total_loss / n,
            'focal': total_focal / n,
            'dice': total_dice / n,
        })
        return metrics
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        
        for batch in tqdm(loader, desc="Validating"):
            signal = batch['signal'].to(self.device)
            mask = batch['mask'].to(self.device)
            stim_params = batch['stim_params'].to(self.device)
            
            pred = self.model(signal, stim_params)
            losses = self.criterion(pred, mask)
            total_loss += losses['loss'].item()
            
            all_preds.append(pred.cpu())
            all_targets.append(mask.cpu())
        
        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
        metrics['loss'] = total_loss / len(loader)
        return metrics
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # CHANGE 2: Early Stopping Counter
        patience_counter = 0
        
        for epoch in range(self.cfg.epochs):
            print(f"\nEpoch {epoch + 1}/{self.cfg.epochs}")
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # CHANGE 3: Step the scheduler based on VAL F1
            # Note: ReduceLROnPlateau takes the metric as an argument
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['f1'])
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            print(f"LR    - {current_lr:.2e}")

            # CHANGE 4: Early Stopping Logic
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                patience_counter = 0  # Reset counter
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'f1': self.best_f1,
                }, self.cfg.checkpoint_dir / 'best_model.pt')
                print(f"  → Saved best model (F1: {self.best_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  → No improvement. Patience: {patience_counter}/{self.cfg.patience}")
                
                if patience_counter >= self.cfg.patience:
                    print(f"\nEarly stopping triggered! Best F1: {self.best_f1:.4f}")
                    break


# =============================================================================
# 11. Main
# =============================================================================

def main():
    cfg = Config()
    print(f"Device: {cfg.device}")
    
    # Load data
    print("Loading annotations...")
    df = load_annotations(cfg)
    print(f"Found {len(df)} annotated recordings")
    
    print("Finding H5 files...")
    file_map = find_h5_files(cfg)
    print(f"Found {len(file_map)} H5 files")
    
    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Create datasets
    print("Building datasets...")
    train_dataset = StimArtifactDataset(train_df, file_map, cfg, augment=True)
    val_dataset = StimArtifactDataset(val_df, file_map, cfg, augment=False)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Data loaders - optimized for M4 Pro with RAM caching
    loader_kwargs = {
        'num_workers': cfg.num_workers,
        'persistent_workers': True,
        'prefetch_factor': cfg.prefetch_factor,
    }
    # pin_memory only helps CUDA, not MPS
    if cfg.device == "cuda":
        loader_kwargs['pin_memory'] = True
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        **loader_kwargs,
    )
    
    # Create model
    model = ConditionalUNet(cfg)
    
    # Train
    trainer = Trainer(model, cfg)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()