"""Core layers for SetPeriod-MTCMD.

SetPeriod-MTCMD is a forecasting architecture derived from the period-based 2D
modeling principle of TimesNet and the multi-period decomposition motivation of
MTCMD.  This implementation contributes three research modules:

1. PeriodSetMixer
   Treats FFT-discovered periods as an unordered set and updates every period
   representation through a shared, permutation-equivariant set context.
2. SemanticDualAxisDilatedConv
   Uses separate depthwise dilated convolutions for the intraperiod axis
   (within a cycle) and the interperiod axis (across repeated cycles).
3. PeriodicConfidenceResidualGate
   Uses spectral concentration and residual statistics to control a lightweight
   irregular-residual correction branch.

Tensor conventions
------------------
Raw time series:        [batch, time, variables]
Embedded sequence:      [batch, time, hidden]
Period map:             [batch, hidden, cycles, period]

The module is deliberately self-contained except for PyTorch.  The model wrapper
in ``model/SetPeriodMTCMD.py`` imports THUML's ``layers.Embed.DataEmbedding``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _validate_3d(x: torch.Tensor, name: str = "x") -> None:
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape [B,T,C], received {tuple(x.shape)}")


def _ensure_odd_positive(values: Iterable[int], name: str) -> Tuple[int, ...]:
    parsed = tuple(int(v) for v in values)
    if not parsed:
        raise ValueError(f"{name} cannot be empty")
    if any(v <= 0 or v % 2 == 0 for v in parsed):
        raise ValueError(f"Every value in {name} must be a positive odd integer: {parsed}")
    return parsed


def _ensure_positive(values: Iterable[int], name: str) -> Tuple[int, ...]:
    parsed = tuple(int(v) for v in values)
    if not parsed:
        raise ValueError(f"{name} cannot be empty")
    if any(v <= 0 for v in parsed):
        raise ValueError(f"Every value in {name} must be positive: {parsed}")
    return parsed


def pad_reshape_period(x: torch.Tensor, period: int) -> Tuple[torch.Tensor, int]:
    """Convert [B,T,D] to [B,D,cycles,period] with zero padding."""
    _validate_3d(x)
    if period <= 0:
        raise ValueError(f"period must be positive, received {period}")
    batch, length, channels = x.shape
    padded_length = int(math.ceil(length / period) * period)
    if padded_length > length:
        padding = x.new_zeros(batch, padded_length - length, channels)
        x = torch.cat([x, padding], dim=1)
    period_map = x.reshape(batch, padded_length // period, period, channels)
    period_map = period_map.permute(0, 3, 1, 2).contiguous()
    return period_map, padded_length


def restore_period_map(period_map: torch.Tensor, target_length: int) -> torch.Tensor:
    """Convert [B,D,cycles,period] back to [B,T,D] and crop to target length."""
    if period_map.ndim != 4:
        raise ValueError(
            f"period_map must have shape [B,D,cycles,period], got {tuple(period_map.shape)}"
        )
    batch, channels, cycles, period = period_map.shape
    restored = period_map.permute(0, 2, 3, 1).reshape(batch, cycles * period, channels)
    return restored[:, :target_length, :]


@dataclass
class SpectralDecompositionOutput:
    """Output of the seasonal/residual Fourier decomposition."""

    seasonal: torch.Tensor
    residual: torch.Tensor
    periods: Tuple[int, ...]
    frequency_indices: torch.Tensor
    selected_amplitudes: torch.Tensor
    period_weights: torch.Tensor
    full_amplitude: torch.Tensor
    selected_energy_ratio: torch.Tensor
    spectral_entropy: torch.Tensor


# -----------------------------------------------------------------------------
# Adaptive trend + Fourier seasonal/residual decomposition
# -----------------------------------------------------------------------------


class AdaptiveMultiScaleTrend(nn.Module):
    """Per-variable adaptive mixture of moving-average trend estimates.

    Each variable obtains its own routing weights from four simple statistics:
    mean absolute level, standard deviation, first-difference magnitude, and
    second-difference magnitude.  The operation preserves [B,T,C].
    """

    def __init__(
        self,
        kernel_sizes: Sequence[int] = (5, 13, 25),
        router_hidden: int = 16,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.kernel_sizes = _ensure_odd_positive(kernel_sizes, "trend kernel_sizes")
        self.temperature = max(float(temperature), 1e-4)
        hidden = max(4, int(router_hidden))
        self.router = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(self.kernel_sizes)),
        )

    @staticmethod
    def _moving_average(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        # [B,T,C] -> [B,C,T]
        channels_first = x.transpose(1, 2)
        padding = (kernel_size - 1) // 2
        padded = F.pad(channels_first, (padding, padding), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
        return trend.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_3d(x)
        first = x[:, 1:, :] - x[:, :-1, :]
        if x.size(1) >= 3:
            second = first[:, 1:, :] - first[:, :-1, :]
            second_energy = second.abs().mean(dim=1)
        else:
            second_energy = x.new_zeros(x.size(0), x.size(2))

        stats = torch.stack(
            [
                x.abs().mean(dim=1),
                x.std(dim=1, unbiased=False),
                first.abs().mean(dim=1) if first.numel() else x.new_zeros(x.size(0), x.size(2)),
                second_energy,
            ],
            dim=-1,
        )  # [B,C,4]
        logits = self.router(stats.float()).to(dtype=x.dtype)
        weights = F.softmax(logits / self.temperature, dim=-1)  # [B,C,M]

        candidates = torch.stack(
            [self._moving_average(x, kernel) for kernel in self.kernel_sizes], dim=-1
        )  # [B,T,C,M]
        trend = torch.sum(candidates * weights[:, None, :, :], dim=-1)
        return trend, weights


class FourierSeasonalResidualDecomposition(nn.Module):
    """Hard Top-K Fourier seasonal reconstruction plus exact residual.

    The selected frequencies are global within a mini-batch, matching the common
    TimesNet practice.  Sample-specific amplitudes are retained for aggregation
    and confidence estimation.  ``band_radius`` optionally retains neighboring
    bins around every selected frequency.
    """

    def __init__(
        self,
        top_k: int = 5,
        period_temperature: float = 1.0,
        band_radius: int = 0,
    ) -> None:
        super().__init__()
        self.top_k = max(1, int(top_k))
        self.period_temperature = max(float(period_temperature), 1e-4)
        self.band_radius = max(0, int(band_radius))

    @staticmethod
    def _deduplicate_periods(
        indices: torch.Tensor, sequence_length: int
    ) -> Tuple[Tuple[int, ...], torch.Tensor]:
        periods: List[int] = []
        kept_indices: List[int] = []
        for index in indices.detach().cpu().tolist():
            index_int = max(1, int(index))
            period = max(1, int(round(sequence_length / index_int)))
            if period not in periods:
                periods.append(period)
                kept_indices.append(index_int)
        if not kept_indices:
            periods = [sequence_length]
            kept_indices = [1]
        kept = indices.new_tensor(kept_indices, dtype=torch.long)
        return tuple(periods), kept

    def forward(self, detrended: torch.Tensor) -> SpectralDecompositionOutput:
        _validate_3d(detrended, "detrended")
        batch, length, variables = detrended.shape
        spectrum = torch.fft.rfft(detrended.float(), dim=1)
        amplitude_bfc = spectrum.abs()  # [B,F,C]
        amplitude_bf = amplitude_bfc.mean(dim=-1)  # [B,F]
        global_amplitude = amplitude_bf.mean(dim=0)

        if global_amplitude.numel() <= 1:
            frequency_indices = torch.zeros(1, device=detrended.device, dtype=torch.long)
            periods = (1,)
        else:
            global_for_topk = global_amplitude.clone()
            global_for_topk[0] = -torch.inf
            k = min(self.top_k, global_for_topk.numel() - 1)
            raw_indices = torch.topk(global_for_topk, k=k, dim=0).indices
            periods, frequency_indices = self._deduplicate_periods(raw_indices, length)

        selected_amplitudes = amplitude_bf.index_select(1, frequency_indices)
        period_weights = F.softmax(
            selected_amplitudes.to(dtype=detrended.dtype) / self.period_temperature,
            dim=-1,
        )

        frequency_mask = torch.zeros(
            spectrum.size(1), device=detrended.device, dtype=spectrum.real.dtype
        )
        for index in frequency_indices.detach().cpu().tolist():
            left = max(1, int(index) - self.band_radius)
            right = min(spectrum.size(1) - 1, int(index) + self.band_radius)
            frequency_mask[left : right + 1] = 1.0
        seasonal_spectrum = spectrum * frequency_mask[None, :, None]
        seasonal = torch.fft.irfft(seasonal_spectrum, n=length, dim=1).to(detrended.dtype)
        residual = detrended - seasonal

        power = amplitude_bfc.square()
        non_dc_power = power[:, 1:, :]
        total_energy = non_dc_power.sum(dim=1).clamp_min(1e-8)  # [B,C]
        selected_energy = power.index_select(1, frequency_indices).sum(dim=1)
        selected_energy_ratio = (selected_energy / total_energy).clamp(0.0, 1.0)

        probability = non_dc_power / total_energy[:, None, :]
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / math.log(max(non_dc_power.size(1), 2))
        entropy = entropy.clamp(0.0, 1.0).to(detrended.dtype)

        return SpectralDecompositionOutput(
            seasonal=seasonal,
            residual=residual,
            periods=periods,
            frequency_indices=frequency_indices,
            selected_amplitudes=selected_amplitudes.to(detrended.dtype),
            period_weights=period_weights,
            full_amplitude=amplitude_bfc.to(detrended.dtype),
            selected_energy_ratio=selected_energy_ratio.to(detrended.dtype),
            spectral_entropy=entropy,
        )


class TrendSeasonResidualDecomposition(nn.Module):
    """Complete normalized-input decomposition: x = trend + seasonal + residual."""

    def __init__(
        self,
        trend_kernel_sizes: Sequence[int] = (5, 13, 25),
        trend_router_hidden: int = 16,
        trend_temperature: float = 1.0,
        top_k: int = 5,
        period_temperature: float = 1.0,
        seasonal_band_radius: int = 0,
    ) -> None:
        super().__init__()
        self.trend = AdaptiveMultiScaleTrend(
            kernel_sizes=trend_kernel_sizes,
            router_hidden=trend_router_hidden,
            temperature=trend_temperature,
        )
        self.fourier = FourierSeasonalResidualDecomposition(
            top_k=top_k,
            period_temperature=period_temperature,
            band_radius=seasonal_band_radius,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, SpectralDecompositionOutput, torch.Tensor]:
        trend, trend_weights = self.trend(x)
        detrended = x - trend
        spectral = self.fourier(detrended)
        return trend, spectral, trend_weights


# -----------------------------------------------------------------------------
# Period-set interaction
# -----------------------------------------------------------------------------


class PeriodMapDescriptor(nn.Module):
    """Shared descriptor encoder for variable-size 2D period maps."""

    def __init__(self, channels: int, set_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        channels = int(channels)
        set_dim = int(set_dim)
        descriptor_channels = max(8, min(set_dim, channels))
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, descriptor_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.project = nn.Linear(2 * descriptor_channels, set_dim)

    def forward(self, period_map: torch.Tensor) -> torch.Tensor:
        features = self.encoder(period_map)
        mean = features.mean(dim=(2, 3))
        std = features.var(dim=(2, 3), unbiased=False).add(1e-6).sqrt()
        return self.project(torch.cat([mean, std], dim=-1))


class PeriodSetMixer(nn.Module):
    """Permutation-equivariant all-period interaction.

    Every period uses shared encoders.  The global context is a weighted sum over
    the period set, hence invariant to input ordering.  Each output remains tied
    to its corresponding input map, hence the overall mapping is equivariant.
    """

    def __init__(
        self,
        channels: int,
        set_dim: int = 64,
        update_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.set_dim = int(set_dim)
        hidden = max(self.set_dim, int(update_hidden))

        self.map_descriptor = PeriodMapDescriptor(channels, set_dim, dropout)
        self.meta_encoder = nn.Sequential(
            nn.Linear(4, set_dim),
            nn.GELU(),
            nn.Linear(set_dim, set_dim),
        )
        self.descriptor_norm = nn.LayerNorm(set_dim)
        self.context_score = nn.Sequential(
            nn.Linear(set_dim, set_dim), nn.Tanh(), nn.Linear(set_dim, 1)
        )
        self.context_value = nn.Sequential(
            nn.Linear(set_dim, set_dim), nn.GELU(), nn.Linear(set_dim, set_dim)
        )
        self.update = nn.Sequential(
            nn.Linear(4 * set_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, set_dim),
        )
        self.update_norm = nn.LayerNorm(set_dim)
        self.film = nn.Sequential(
            nn.Linear(set_dim, 2 * channels),
            nn.Tanh(),
        )
        self.period_score = nn.Sequential(
            nn.Linear(set_dim, set_dim), nn.GELU(), nn.Linear(set_dim, 1)
        )

    def forward(
        self,
        period_maps: Sequence[torch.Tensor],
        periods: Sequence[int],
        amplitudes: torch.Tensor,
        initial_weights: torch.Tensor,
        total_length: int,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if len(period_maps) == 0:
            raise ValueError("period_maps cannot be empty")
        if not (len(period_maps) == len(periods) == amplitudes.size(1) == initial_weights.size(1)):
            raise ValueError("period maps, periods, amplitudes, and weights must have matching K")

        batch = period_maps[0].size(0)
        descriptors: List[torch.Tensor] = []
        max_amplitude = amplitudes.max(dim=1, keepdim=True).values.clamp_min(1e-6)
        normalized_amplitudes = amplitudes / max_amplitude

        for index, (period_map, period) in enumerate(zip(period_maps, periods)):
            map_descriptor = self.map_descriptor(period_map)
            period_value = float(period)
            meta = amplitudes.new_tensor(
                [
                    math.log1p(period_value) / math.log1p(max(float(total_length), 1.0)),
                    period_value / max(float(total_length), 1.0),
                    0.0,
                    0.0,
                ]
            ).view(1, 4).repeat(batch, 1)
            meta[:, 2] = normalized_amplitudes[:, index]
            meta[:, 3] = initial_weights[:, index]
            descriptor = self.descriptor_norm(map_descriptor + self.meta_encoder(meta.float()))
            descriptors.append(descriptor)

        descriptor_tensor = torch.stack(descriptors, dim=1)  # [B,K,S]
        context_logits = self.context_score(descriptor_tensor).squeeze(-1)
        context_weights = F.softmax(context_logits, dim=1)
        context_values = self.context_value(descriptor_tensor)
        context = torch.sum(context_values * context_weights.unsqueeze(-1), dim=1)

        context_expanded = context.unsqueeze(1).expand_as(descriptor_tensor)
        update_input = torch.cat(
            [
                descriptor_tensor,
                context_expanded,
                descriptor_tensor * context_expanded,
                (descriptor_tensor - context_expanded).abs(),
            ],
            dim=-1,
        )
        updated = self.update_norm(descriptor_tensor + self.update(update_input))

        film = self.film(updated)  # [B,K,2D]
        gamma, beta = film.chunk(2, dim=-1)
        modulated_maps: List[torch.Tensor] = []
        for index, period_map in enumerate(period_maps):
            scale = 1.0 + gamma[:, index, :].unsqueeze(-1).unsqueeze(-1)
            shift = beta[:, index, :].unsqueeze(-1).unsqueeze(-1)
            modulated_maps.append(scale * period_map + shift)

        period_quality_logits = self.period_score(updated).squeeze(-1)
        diagnostics = {
            "set_context_weights": context_weights,
            "period_descriptors": descriptor_tensor,
            "updated_period_descriptors": updated,
            "film_gamma": gamma,
            "film_beta": beta,
        }
        return modulated_maps, updated, period_quality_logits, diagnostics


# -----------------------------------------------------------------------------
# Semantic dual-axis period convolution
# -----------------------------------------------------------------------------


class _DepthwiseAxisExpert(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, axis: str) -> None:
        super().__init__()
        if axis not in {"intra", "inter"}:
            raise ValueError("axis must be 'intra' or 'inter'")
        radius = dilation * (kernel_size - 1) // 2
        if axis == "intra":
            kernel = (1, kernel_size)
            dilation_pair = (1, dilation)
            padding = (0, radius)
        else:
            kernel = (kernel_size, 1)
            dilation_pair = (dilation, 1)
            padding = (radius, 0)
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel,
            dilation=dilation_pair,
            padding=padding,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SemanticDualAxisDilatedConv(nn.Module):
    """Period-conditioned depthwise convolution on the two semantic 2D axes."""

    def __init__(
        self,
        channels: int,
        set_dim: int,
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 3,
        expansion: int = 2,
        router_temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dilations = _ensure_positive(dilations, "dilations")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = int(kernel_size)
        self.router_temperature = max(float(router_temperature), 1e-4)
        experts = len(self.dilations)

        self.intra_router = nn.Linear(set_dim, experts)
        self.inter_router = nn.Linear(set_dim, experts)
        self.intra_experts = nn.ModuleList(
            [
                _DepthwiseAxisExpert(channels, self.kernel_size, dilation, "intra")
                for dilation in self.dilations
            ]
        )
        self.inter_experts = nn.ModuleList(
            [
                _DepthwiseAxisExpert(channels, self.kernel_size, dilation, "inter")
                for dilation in self.dilations
            ]
        )

        self.spatial_gate = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        self.descriptor_gate = nn.Linear(set_dim, channels)
        hidden = max(channels, int(expansion) * channels)
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Dropout2d(dropout),
        )
        self.norm = nn.GroupNorm(1, channels)

    @staticmethod
    def _mix_experts(
        x: torch.Tensor, experts: nn.ModuleList, weights: torch.Tensor
    ) -> torch.Tensor:
        outputs = torch.stack([expert(x) for expert in experts], dim=-1)
        return torch.sum(outputs * weights[:, None, None, None, :], dim=-1)

    def forward(
        self, period_map: torch.Tensor, descriptor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        intra_weights = F.softmax(
            self.intra_router(descriptor) / self.router_temperature, dim=-1
        )
        inter_weights = F.softmax(
            self.inter_router(descriptor) / self.router_temperature, dim=-1
        )
        intra = self._mix_experts(period_map, self.intra_experts, intra_weights)
        inter = self._mix_experts(period_map, self.inter_experts, inter_weights)

        gate = self.spatial_gate(torch.cat([intra, inter], dim=1))
        gate = gate + self.descriptor_gate(descriptor).unsqueeze(-1).unsqueeze(-1)
        gate = torch.sigmoid(gate)
        fused = gate * intra + (1.0 - gate) * inter
        output = self.norm(period_map + self.channel_mixer(fused))
        diagnostics = {
            "intra_router_weights": intra_weights,
            "inter_router_weights": inter_weights,
            "axis_gate_mean": gate.mean(dim=(1, 2, 3)),
        }
        return output, diagnostics


# -----------------------------------------------------------------------------
# Complete period-set block
# -----------------------------------------------------------------------------


class SetPeriodBlock(nn.Module):
    """Set interaction + semantic dual-axis convolution + period aggregation."""

    def __init__(
        self,
        channels: int,
        set_dim: int = 64,
        set_update_hidden: int = 128,
        dilations: Sequence[int] = (1, 2, 4, 8),
        axis_kernel_size: int = 3,
        axis_expansion: int = 2,
        axis_temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.set_mixer = PeriodSetMixer(
            channels=channels,
            set_dim=set_dim,
            update_hidden=set_update_hidden,
            dropout=dropout,
        )
        self.axis_block = SemanticDualAxisDilatedConv(
            channels=channels,
            set_dim=set_dim,
            dilations=dilations,
            kernel_size=axis_kernel_size,
            expansion=axis_expansion,
            router_temperature=axis_temperature,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(channels)
        self._last_balance_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        periods: Sequence[int],
        amplitudes: torch.Tensor,
        initial_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        _validate_3d(x)
        batch, length, channels = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected hidden size {self.channels}, received {channels}")

        period_maps: List[torch.Tensor] = []
        padded_lengths: List[int] = []
        for period in periods:
            period_map, padded_length = pad_reshape_period(x, int(period))
            period_maps.append(period_map)
            padded_lengths.append(padded_length)

        modulated_maps, descriptors, quality_logits, set_info = self.set_mixer(
            period_maps=period_maps,
            periods=periods,
            amplitudes=amplitudes,
            initial_weights=initial_weights,
            total_length=length,
        )

        restored_outputs: List[torch.Tensor] = []
        axis_infos: List[Dict[str, torch.Tensor]] = []
        all_intra: List[torch.Tensor] = []
        all_inter: List[torch.Tensor] = []
        for index, period_map in enumerate(modulated_maps):
            processed, axis_info = self.axis_block(period_map, descriptors[:, index, :])
            restored_outputs.append(restore_period_map(processed, length))
            axis_infos.append(axis_info)
            all_intra.append(axis_info["intra_router_weights"])
            all_inter.append(axis_info["inter_router_weights"])

        stacked = torch.stack(restored_outputs, dim=-1)  # [B,T,D,K]
        aggregation_logits = initial_weights.clamp_min(1e-8).log() + quality_logits
        aggregation_weights = F.softmax(aggregation_logits, dim=-1)
        aggregated = torch.sum(
            stacked * aggregation_weights[:, None, None, :], dim=-1
        )
        output = self.output_norm(x + aggregated)

        # Switch/load-balance style auxiliary loss across batch and periods.
        intra_tensor = torch.stack(all_intra, dim=1)  # [B,K,M]
        inter_tensor = torch.stack(all_inter, dim=1)
        expert_count = intra_tensor.size(-1)
        target = 1.0 / expert_count
        mean_intra = intra_tensor.mean(dim=(0, 1))
        mean_inter = inter_tensor.mean(dim=(0, 1))
        self._last_balance_loss = (
            (mean_intra - target).square().mean()
            + (mean_inter - target).square().mean()
        )

        diagnostics: Dict[str, object] = {
            "periods": tuple(int(p) for p in periods),
            "initial_period_weights": initial_weights,
            "aggregation_weights": aggregation_weights,
            "set": set_info,
            "axis": axis_infos,
            "balance_loss": self._last_balance_loss,
        }
        return output, diagnostics

    def auxiliary_balance_loss(self) -> torch.Tensor:
        if self._last_balance_loss is None:
            return next(self.parameters()).new_zeros(())
        return self._last_balance_loss


# -----------------------------------------------------------------------------
# Trend and irregular residual forecasting heads
# -----------------------------------------------------------------------------


class SharedTemporalProjection(nn.Module):
    """Apply the same temporal Linear(input_length -> output_length) to every channel."""

    def __init__(self, input_length: int, output_length: int) -> None:
        super().__init__()
        self.projection = nn.Linear(int(input_length), int(output_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_3d(x)
        return self.projection(x.transpose(1, 2)).transpose(1, 2)


class CausalDepthwiseResidualBlock(nn.Module):
    """Lightweight causal depthwise-separable Conv1d residual block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, 2 * channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(2 * channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.left_padding, 0), mode="replicate")
        features = self.depthwise(padded)
        return self.norm(x + self.pointwise(features))


class IrregularResidualExpert(nn.Module):
    """Variable-shared causal residual expert.

    Variables are folded into the batch dimension, so the temporal filters are
    shared across variables and the parameter count does not scale with C.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        hidden: int = 32,
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = max(4, int(hidden))
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.input_projection = nn.Conv1d(1, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                CausalDepthwiseResidualBlock(hidden, kernel_size, int(d), dropout)
                for d in _ensure_positive(dilations, "residual dilations")
            ]
        )
        self.temporal_projection = nn.Linear(self.seq_len, self.pred_len)
        self.output_projection = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        _validate_3d(residual, "residual")
        batch, length, variables = residual.shape
        if length != self.seq_len:
            raise ValueError(f"Expected residual length {self.seq_len}, got {length}")
        x = residual.permute(0, 2, 1).reshape(batch * variables, 1, length)
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        x = self.temporal_projection(x)  # Linear over final temporal dimension
        x = self.output_projection(x).squeeze(1)
        return x.reshape(batch, variables, self.pred_len).transpose(1, 2)


class PeriodicConfidenceResidualGate(nn.Module):
    """Per-variable confidence and residual correction gate."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        hidden = max(4, int(hidden))
        self.confidence = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.correction_gate = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(
        self,
        normalized_input: torch.Tensor,
        residual: torch.Tensor,
        selected_energy_ratio: torch.Tensor,
        spectral_entropy: torch.Tensor,
        period_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        _validate_3d(normalized_input, "normalized_input")
        max_period_weight = period_weights.max(dim=-1).values[:, None]
        max_period_weight = max_period_weight.expand_as(selected_energy_ratio)
        confidence_input = torch.stack(
            [
                selected_energy_ratio,
                1.0 - spectral_entropy,
                max_period_weight,
            ],
            dim=-1,
        )
        periodic_confidence = self.confidence(confidence_input.float()).squeeze(-1)
        periodic_confidence = periodic_confidence.to(normalized_input.dtype)

        residual_energy = residual.abs().mean(dim=1)
        if normalized_input.size(1) > 1:
            roughness = (
                normalized_input[:, 1:, :] - normalized_input[:, :-1, :]
            ).abs().mean(dim=1)
        else:
            roughness = normalized_input.new_zeros(normalized_input.size(0), normalized_input.size(2))
        gate_input = torch.stack(
            [1.0 - periodic_confidence, residual_energy, roughness], dim=-1
        )
        correction_gate = self.correction_gate(gate_input.float()).squeeze(-1)
        correction_gate = correction_gate.to(normalized_input.dtype)
        diagnostics = {
            "periodic_confidence": periodic_confidence,
            "residual_energy": residual_energy,
            "roughness": roughness,
            "residual_gate": correction_gate,
        }
        return correction_gate, periodic_confidence, diagnostics
