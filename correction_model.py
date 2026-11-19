"""Residual-aware global warm-start corrections for the fixed P1 Darcy system.

This module is independent of the original local coarse-space network. A
prediction is an additive solution correction, expressed in the caller's
positive, input-derived solution scale. No true solution may enter features
or normalization. The network enforces homogeneous Dirichlet boundaries;
the subsequent classical solver remains responsible for final accuracy.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _cell_logs_to_nodes(cells: np.ndarray) -> np.ndarray:
    """Average incident cell log values; physical edges use existing cells."""
    ny, nx = cells.shape
    values = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    counts = np.zeros_like(values)
    for sy, sx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        values[sy:sy + ny, sx:sx + nx] += cells
        counts[sy:sy + ny, sx:sx + nx] += 1
    return values / counts


def _as_nodal_array(values, n: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (n + 1, n + 1):
        result = values.copy()
    elif values.size == (n - 1) ** 2:
        result = np.zeros((n + 1, n + 1), dtype=np.float64)
        result[1:-1, 1:-1] = values.reshape(n - 1, n - 1)
    else:
        raise ValueError(f"{name} must be a full nodal grid or an interior free vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    if np.any(result[[0, -1], :]) or np.any(result[:, [0, -1]]):
        raise ValueError(f"{name} must have zero physical boundary entries")
    return result


def make_correction_features(current_k, previous_k, residual, previous_solution,
                             solution_scale: float,
                             residual_scale: float | None = None,
                             log_reference: float = 1e4) -> tuple[np.ndarray, dict]:
    """Construct the six prescribed nodal channels, without label leakage.

    Channels are incident-cell average log(k)/log(log_reference), incident
    average log(k/k_previous)/log(log_reference), residual/residual_scale,
    previous_solution/solution_scale, and x,y coordinates in [0,1]. The last
    two fields use y-major storage, matching numerics.py. Coefficients remain
    defined on cells; residual and previous solution can be nodal arrays or
    free-node vectors. Their physical boundary values must already be zero.

    ``solution_scale`` is required and must be computed only from information
    available before solving the current system, e.g. an available iterate or
    a coarse prediction. The label must be (u_current-u_base)/solution_scale.
    The caller must use this same scale to recover the physical prediction.
    The base iterate supplied here as previous_solution must match that label.

    Default residual_scale is max(interior RMS residual, 1e-12). Both scales
    are recorded. Normalization performs no clipping; no target is consulted.
    """
    k = np.asarray(current_k, dtype=np.float64)
    old = np.asarray(previous_k, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] < 2 or old.shape != k.shape:
        raise ValueError("current_k and previous_k must be equally sized square cell arrays")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(old)) or np.any(k <= 0) or np.any(old <= 0):
        raise ValueError("coefficients must be positive and finite")
    if not math.isfinite(solution_scale) or solution_scale <= 0:
        raise ValueError("solution_scale must be positive and finite, determined without labels")
    if not math.isfinite(log_reference) or log_reference <= 1:
        raise ValueError("log_reference must be finite and greater than one")
    n = k.shape[0]
    r = _as_nodal_array(residual, n, "residual")
    u = _as_nodal_array(previous_solution, n, "previous_solution")
    if residual_scale is None:
        residual_scale = max(float(np.sqrt(np.mean(r[1:-1, 1:-1] ** 2))), 1e-12)
    if not math.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("residual_scale must be positive and finite")
    log_k, log_old = np.log(k), np.log(old)
    yy, xx = np.meshgrid(np.linspace(0., 1., n + 1), np.linspace(0., 1., n + 1), indexing="ij")
    features = np.stack((_cell_logs_to_nodes(log_k) / math.log(log_reference),
                         _cell_logs_to_nodes(log_k - log_old) / math.log(log_reference),
                         r / residual_scale, u / solution_scale, xx, yy)).astype(np.float32)
    return features, {
        "solution_scale": float(solution_scale),
        "residual_scale": float(residual_scale),
        "log_reference": float(log_reference),
        "channel_order": ["nodal_mean_log_k", "nodal_mean_log_ratio",
                          "normalized_residual", "normalized_base_solution", "x", "y"],
        "target_convention": "(current_solution-base_solution)/solution_scale",
        "label_used_for_normalization": False,
    }


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        groups = math.gcd(4, cout)
        self.layers = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                    nn.GroupNorm(groups, cout), nn.SiLU(),
                                    nn.Conv2d(cout, cout, 3, padding=1),
                                    nn.GroupNorm(groups, cout), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class CorrectionUNet(nn.Module):
    """Compact two-level U-Net with exact zero physical boundary output.

    Default widths are 8/16/32. ``width=16`` provides the larger option.
    Bilinear interpolation targets skip-tensor sizes, including 129x129 grids.
    GroupNorm is independent of batch statistics. The zero-initialized head
    starts from a zero correction; there is no clipping or learned acceptance
    heuristic in the forward pass.
    """
    def __init__(self, width: int = 8, zero_init_output: bool = True):
        super().__init__()
        if not isinstance(width, int) or width < 2:
            raise ValueError("width must be an integer >= 2")
        self.width = width
        self.encoder1 = _ConvBlock(6, width)
        self.encoder2 = _ConvBlock(width, 2 * width)
        self.middle = _ConvBlock(2 * width, 4 * width)
        self.decoder2 = _ConvBlock(6 * width, 2 * width)
        self.decoder1 = _ConvBlock(3 * width, width)
        self.head = nn.Conv2d(width, 1, 1)
        if zero_init_output:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 6 or min(x.shape[-2:]) < 5:
            raise ValueError("input must have shape (batch,6,height,width), spatial size >= 5")
        one = self.encoder1(x)
        two = self.encoder2(F.avg_pool2d(one, 2))
        mid = self.middle(F.avg_pool2d(two, 2))
        up_two = F.interpolate(mid, size=two.shape[-2:], mode="bilinear", align_corners=False)
        two_out = self.decoder2(torch.cat((two, up_two), dim=1))
        up_one = F.interpolate(two_out, size=one.shape[-2:], mode="bilinear", align_corners=False)
        raw = self.head(self.decoder1(torch.cat((one, up_one), dim=1)))
        # Padding, rather than multiplying a mask, gives exact boundary zeros.
        return F.pad(raw[..., 1:-1, 1:-1], (1, 1, 1, 1))


def _node_batch(u: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not isinstance(u, torch.Tensor) or not u.is_floating_point():
        raise ValueError("nodal values must be a floating-point Torch tensor")
    original_ndim = u.ndim
    if u.ndim == 2:
        u = u.unsqueeze(0)
    elif u.ndim == 4 and u.shape[1] == 1:
        u = u[:, 0]
    elif u.ndim != 3:
        raise ValueError("nodal values must have shape (H,W), (B,H,W), or (B,1,H,W)")
    if min(u.shape[-2:]) < 3:
        raise ValueError("a nodal grid needs at least one interior node")
    return u, original_ndim


def _coefficient_batch(k, u: torch.Tensor) -> torch.Tensor:
    k = torch.as_tensor(k, dtype=u.dtype, device=u.device)
    if k.ndim == 2:
        k = k.unsqueeze(0)
    elif k.ndim == 4 and k.shape[1] == 1:
        k = k[:, 0]
    if k.ndim != 3 or k.shape[-2:] != (u.shape[-2] - 1, u.shape[-1] - 1):
        raise ValueError("cell coefficients must match the nodal grid")
    if k.shape[0] not in (1, u.shape[0]):
        raise ValueError("coefficient batch must equal the nodal batch or be one")
    return k


def _restore_nodes(u: torch.Tensor, ndim: int) -> torch.Tensor:
    return u[0] if ndim == 2 else u.unsqueeze(1) if ndim == 4 else u


def p1_energy(u: torch.Tensor, k) -> torch.Tensor:
    """Return one exact discrete u'A(k)u energy per batch item (no h factor).

    Each square is split SW-SE-NE and SW-NE-NW, exactly as in numerics.py.
    With zero physical boundary this equals the assembled free-node matrix
    energy. Nonzero boundary inputs instead give the full nodal FE energy.
    The usual 1/2 potential-energy factor is deliberately not included.
    """
    u, _ = _node_batch(u)
    k = _coefficient_batch(k, u)
    sw, se, nw, ne = u[:, :-1, :-1], u[:, :-1, 1:], u[:, 1:, :-1], u[:, 1:, 1:]
    cell = .5 * k * ((sw - se).square() + (se - ne).square()
                     + (sw - nw).square() + (nw - ne).square())
    return cell.sum(dim=(-2, -1))


def p1_apply(u: torch.Tensor, k) -> torch.Tensor:
    """Apply the exact P1 stiffness and return interior action, boundary zero.

    Shape is preserved. For a Dirichlet solve the input must have zero
    boundary; the returned array is the free-node A u padded by zeros.
    All operations are differentiable and execute on the input device.
    """
    u, ndim = _node_batch(u)
    k = _coefficient_batch(k, u)
    sw, se, nw, ne = u[:, :-1, :-1], u[:, :-1, 1:], u[:, 1:, :-1], u[:, 1:, 1:]
    out = (F.pad(.5 * k * (2 * sw - se - nw), (0, 1, 0, 1))
           + F.pad(.5 * k * (2 * se - sw - ne), (1, 0, 0, 1))
           + F.pad(.5 * k * (2 * nw - sw - ne), (0, 1, 1, 0))
           + F.pad(.5 * k * (2 * ne - se - nw), (1, 0, 1, 0)))
    return _restore_nodes(F.pad(out[..., 1:-1, 1:-1], (1, 1, 1, 1)), ndim)


def p1_diagonal(k: torch.Tensor) -> torch.Tensor:
    """Return the assembled full nodal diagonal; shape mirrors coefficient rank.

    Each cell contributes its coefficient to each corner diagonal. Physical
    boundary entries are positive full-matrix diagonals; residual functions
    have zero boundary and therefore contribute nothing there.
    """
    if not isinstance(k, torch.Tensor) or not k.is_floating_point() or k.ndim not in (2, 3, 4):
        raise ValueError("k must be a floating-point tensor of rank 2, 3, or 4")
    if k.ndim == 4 and k.shape[1] != 1:
        raise ValueError("four-dimensional coefficients require a singleton channel")
    return (F.pad(k, (0, 1, 0, 1)) + F.pad(k, (1, 0, 0, 1))
            + F.pad(k, (0, 1, 1, 0)) + F.pad(k, (1, 0, 1, 0)))


def _loss_reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def relative_energy_loss(prediction: torch.Tensor, target: torch.Tensor, k,
                         epsilon: float = 1e-12, l2_weight: float = 0.,
                         reduction: str = "mean") -> torch.Tensor:
    """Per-sample E(pred-target)/max(E(target),epsilon), optionally plus L2.

    This is a squared relative energy error, not its square root. Both inputs
    must use the same input-derived solution scale. The finite denominator
    floor is explicit; training code should record how zero/tiny targets are
    handled, rather than interpreting their ratios as ordinary relative errors.
    """
    if epsilon <= 0 or l2_weight < 0:
        raise ValueError("epsilon must be positive and l2_weight nonnegative")
    pred, _ = _node_batch(prediction)
    truth, _ = _node_batch(target)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have the same nodal batch shape")
    error = pred - truth
    losses = p1_energy(error, k) / p1_energy(truth, k).clamp_min(epsilon)
    if l2_weight:
        losses = losses + l2_weight * error.square().sum((-2, -1)) / truth.square().sum((-2, -1)).clamp_min(epsilon)
    return _loss_reduce(losses, reduction)


def weighted_residual_loss(prediction: torch.Tensor, target: torch.Tensor, k,
                           diagonal: torch.Tensor | None = None,
                           epsilon: float = 1e-12,
                           reduction: str = "mean") -> torch.Tensor:
    """Squared relative diagonal-weighted residual error, samplewise.

    Loss = ||A(pred-target)||^2_(D^-1) / max(||A target||^2_(D^-1),epsilon),
    where D is the exact current P1 stiffness diagonal. Boundary residuals are
    zero. ``diagonal`` may supply a cached p1_diagonal(k). This complements
    energy loss by penalizing residuals that can burden subsequent iteration;
    validation still must use complete safeguarded solver time and accuracy.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    pred, _ = _node_batch(prediction)
    truth, _ = _node_batch(target)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have the same nodal batch shape")
    coefficients = _coefficient_batch(k, pred)
    if diagonal is None:
        diag = p1_diagonal(coefficients)
    else:
        diag, _ = _node_batch(torch.as_tensor(diagonal, device=pred.device, dtype=pred.dtype))
        if diag.shape[-2:] != pred.shape[-2:] or diag.shape[0] not in (1, pred.shape[0]):
            raise ValueError("cached diagonal does not match the nodal grid or batch")
    weights = diag.clamp_min(epsilon).reciprocal()
    error_residual = p1_apply(pred - truth, coefficients)
    target_residual = p1_apply(truth, coefficients)
    numerator = (error_residual.square() * weights).sum((-2, -1))
    denominator = (target_residual.square() * weights).sum((-2, -1)).clamp_min(epsilon)
    return _loss_reduce(numerator / denominator, reduction)
