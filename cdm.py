"""One-dimensional conditional diffusion model with a conditional U-Net, BGL/HR/BMI cross-attention, DDPM sampling, and a hybrid MSE-SSIM loss."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ConditionStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, conditions: np.ndarray) -> "ConditionStandardizer":
        conditions = np.asarray(conditions, dtype=np.float32)
        mean = conditions.mean(axis=0)
        std = conditions.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, conditions: np.ndarray) -> np.ndarray:
        return ((np.asarray(conditions, dtype=np.float32) - self.mean) / self.std).astype(np.float32)


class PPGConditionDataset(Dataset):
    def __init__(self, signals: np.ndarray, conditions_std: np.ndarray):
        signals = np.asarray(signals, dtype=np.float32)
        conditions_std = np.asarray(conditions_std, dtype=np.float32)
        if signals.ndim == 2:
            signals = signals[:, None, :]
        if signals.ndim != 3 or signals.shape[1] != 1:
            raise ValueError("signals must have shape [N,L] or [N,1,L]")
        if conditions_std.shape != (len(signals), 3):
            raise ValueError("conditions must have shape [N,3] = [BGL, HR, BMI]")
        self.x = torch.from_numpy(signals)
        self.c = torch.from_numpy(conditions_std)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.c[idx]


def sinusoidal_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    device = t.device
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    angles = t.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 128, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class PhysiologicalCrossAttention(nn.Module):
    """Use the PPG feature sequence as queries and the BGL, HR, and BMI condition tokens as keys and values for cross-attention."""
    def __init__(self, channels: int, attn_dim: int = 128, heads: int = 4):
        super().__init__()
        self.q_proj = nn.Conv1d(channels, attn_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(attn_dim, heads, batch_first=True)
        self.out_proj = nn.Conv1d(attn_dim, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x).transpose(1, 2)
        y, _ = self.attn(q, cond_tokens, cond_tokens, need_weights=False)
        y = self.out_proj(y.transpose(1, 2))
        return x + y


class ConditionTokenizer(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(1, dim) for _ in range(3)])

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.proj[i](cond[:, i:i+1]) for i in range(3)], dim=1)


class Downsample1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2)
    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
    def forward(self, x, target_len: int):
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return self.conv(x)


class ConditionalUNet1D(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), time_dim=128, cond_dim=128):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.time_dim = time_dim
        self.cond_tokens = ConditionTokenizer(cond_dim)

        self.in_conv = nn.Conv1d(1, c1, kernel_size=5, padding=2)

        self.enc1 = ResidualBlock1D(c1, c1, time_dim)
        self.ca1 = PhysiologicalCrossAttention(c1, cond_dim, 4)
        self.down1 = Downsample1D(c1, c2)

        self.enc2 = ResidualBlock1D(c2, c2, time_dim)
        self.ca2 = PhysiologicalCrossAttention(c2, cond_dim, 4)
        self.down2 = Downsample1D(c2, c3)

        self.enc3 = ResidualBlock1D(c3, c3, time_dim)
        self.ca3 = PhysiologicalCrossAttention(c3, cond_dim, 4)
        self.down3 = Downsample1D(c3, c4)

        self.mid = ResidualBlock1D(c4, c4, time_dim)
        self.ca_mid = PhysiologicalCrossAttention(c4, cond_dim, 4)

        self.up3 = Upsample1D(c4, c3)
        self.dec3 = ResidualBlock1D(c3 + c3, c3, time_dim)
        self.ca_dec3 = PhysiologicalCrossAttention(c3, cond_dim, 4)

        self.up2 = Upsample1D(c3, c2)
        self.dec2 = ResidualBlock1D(c2 + c2, c2, time_dim)
        self.ca_dec2 = PhysiologicalCrossAttention(c2, cond_dim, 4)

        self.up1 = Upsample1D(c2, c1)
        self.dec1 = ResidualBlock1D(c1 + c1, c1, time_dim)
        self.ca_dec1 = PhysiologicalCrossAttention(c1, cond_dim, 4)

        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv1d(c1, 1, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond_std: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_dim)
        tokens = self.cond_tokens(cond_std)

        x = self.in_conv(x)
        s1 = self.ca1(self.enc1(x, t_emb), tokens)
        x = self.down1(s1)

        s2 = self.ca2(self.enc2(x, t_emb), tokens)
        x = self.down2(s2)

        s3 = self.ca3(self.enc3(x, t_emb), tokens)
        x = self.down3(s3)

        x = self.ca_mid(self.mid(x, t_emb), tokens)

        x = self.up3(x, s3.shape[-1])
        x = torch.cat([x, s3], dim=1)
        x = self.ca_dec3(self.dec3(x, t_emb), tokens)

        x = self.up2(x, s2.shape[-1])
        x = torch.cat([x, s2], dim=1)
        x = self.ca_dec2(self.dec2(x, t_emb), tokens)

        x = self.up1(x, s1.shape[-1])
        x = torch.cat([x, s1], dim=1)
        x = self.ca_dec1(self.dec1(x, t_emb), tokens)

        return self.out_conv(F.silu(self.out_norm(x)))


class GaussianDiffusion1D:
    def __init__(
        self,
        steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ):
        self.steps = steps
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alpha_bars[:-1]], dim=0
        )

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return arr.gather(0, t).view(-1, 1, 1).to(x.device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self._extract(self.alpha_bars, t, x0)
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
        return xt, noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        ab = self._extract(self.alpha_bars, t, xt)
        return (xt - torch.sqrt(1.0 - ab) * eps_pred) / torch.sqrt(ab)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        cond_std: torch.Tensor,
        length: int = 10875,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        model.eval()
        b = cond_std.shape[0]
        x = torch.randn((b, 1, length), device=cond_std.device, generator=generator)

        for step in reversed(range(self.steps)):
            t = torch.full((b,), step, device=cond_std.device, dtype=torch.long)
            eps = model(x, t, cond_std)
            beta = self._extract(self.betas, t, x)
            alpha = self._extract(self.alphas, t, x)
            abar = self._extract(self.alpha_bars, t, x)
            abar_prev = self._extract(self.alpha_bars_prev, t, x)

            mean = (1.0 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1.0 - abar) * eps)
            if step > 0:
                posterior_var = beta * (1.0 - abar_prev) / (1.0 - abar)
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
                x = mean + torch.sqrt(torch.clamp(posterior_var, min=1e-20)) * noise
            else:
                x = mean
        return x


def _gaussian_window_1d(window_size=11, sigma=1.5, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, -1)


def ssim_1d(x: torch.Tensor, y: torch.Tensor, window_size=11, sigma=1.5) -> torch.Tensor:
    # Inputs are expected in the [0, 1] range; return the mean SSIM over the current batch.
    w = _gaussian_window_1d(window_size, sigma, x.device, x.dtype)
    pad = window_size // 2
    mu_x = F.conv1d(x, w, padding=pad)
    mu_y = F.conv1d(y, w, padding=pad)
    sigma_x2 = F.conv1d(x * x, w, padding=pad) - mu_x * mu_x
    sigma_y2 = F.conv1d(y * y, w, padding=pad) - mu_y * mu_y
    sigma_xy = F.conv1d(x * y, w, padding=pad) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x2 + sigma_y2 + c2)
    score = num / (den + 1e-12)
    return score.mean()


def hybrid_diffusion_loss(
    model: nn.Module,
    diffusion: GaussianDiffusion1D,
    x0: torch.Tensor,
    cond_std: torch.Tensor,
    lambda_ssim: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    b = x0.shape[0]
    t = torch.randint(0, diffusion.steps, (b,), device=x0.device)
    xt, noise = diffusion.q_sample(x0, t)
    eps_pred = model(xt, t, cond_std)
    mse = F.mse_loss(eps_pred, noise)

    x0_hat = diffusion.predict_x0(xt, t, eps_pred)
    # Before SSIM calculation, clip z-score-normalized signals to [-3, 3] and linearly map them to [0, 1].
    x_ref = (torch.clamp(x0, -3.0, 3.0) + 3.0) / 6.0
    x_hat = (torch.clamp(x0_hat, -3.0, 3.0) + 3.0) / 6.0
    ssim_loss = 1.0 - ssim_1d(x_ref, x_hat, window_size=11, sigma=1.5)

    loss = (1.0 - lambda_ssim) * mse + lambda_ssim * ssim_loss
    return loss, {
        "total": float(loss.detach()),
        "mse": float(mse.detach()),
        "ssim_loss": float(ssim_loss.detach()),
    }


def _epoch_loss(model, loader, diffusion, optimizer, device, train: bool):
    model.train(train)
    total, n = 0.0, 0
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        loss, _ = hybrid_diffusion_loss(model, diffusion, x, c, lambda_ssim=0.3)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * len(x)
        n += len(x)
    return total / max(n, 1)


def train_cdm_provisional(
    train_signals: np.ndarray,
    train_conditions: np.ndarray,
    val_signals: np.ndarray,
    val_conditions: np.ndarray,
    device: str = "cuda",
    seed: int = 42,
    max_epochs: int = 2000,
    patience: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> Tuple[int, ConditionStandardizer]:
    """Train a provisional CDM on the inner training/validation split to determine the optimal epoch E*."""
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() or "cuda" not in device else "cpu")

    scaler = ConditionStandardizer.fit(train_conditions)
    tr_ds = PPGConditionDataset(train_signals, scaler.transform(train_conditions))
    va_ds = PPGConditionDataset(val_signals, scaler.transform(val_conditions))
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = ConditionalUNet1D().to(device)
    diffusion = GaussianDiffusion1D(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    best, best_epoch, wait = float("inf"), 1, 0
    for epoch in range(1, max_epochs + 1):
        _epoch_loss(model, tr, diffusion, opt, device, train=True)
        with torch.no_grad():
            val_loss = _epoch_loss(model, va, diffusion, opt, device, train=False)
        if val_loss < best - 1e-12:
            best = val_loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_epoch, scaler


def train_cdm_fixed_epochs(
    all_train_signals: np.ndarray,
    all_train_conditions: np.ndarray,
    epochs: int,
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> Tuple[ConditionalUNet1D, GaussianDiffusion1D, ConditionStandardizer]:
    """Reinitialize the CDM and train it for exactly E* epochs on all real data from the current outer-training partition."""
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() or "cuda" not in device else "cpu")
    scaler = ConditionStandardizer.fit(all_train_conditions)
    ds = PPGConditionDataset(all_train_signals, scaler.transform(all_train_conditions))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = ConditionalUNet1D().to(device)
    diffusion = GaussianDiffusion1D(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    for _ in range(int(epochs)):
        _epoch_loss(model, loader, diffusion, opt, device, train=True)
    return model, diffusion, scaler
