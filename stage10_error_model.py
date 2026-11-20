"""Coefficient-conditioned residual operators; no claim of a new architecture.

The symmetric variant uses exact adjoints of its analysis convolutions.  Its
positive diagonal term makes H strictly positive definite for finite weights.
No activation or bias is applied to the residual stream in either linear model.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ErrorOperator(nn.Module):
    def __init__(self, kind="symmetric", width=8, levels=3):
        super().__init__()
        if kind not in ("symmetric", "untied", "nonlinear"):
            raise ValueError(kind)
        self.kind, self.levels = kind, levels
        self.analysis = nn.ModuleList([
            nn.Conv2d(1 if i == 0 else width, width, 3,
                      stride=1 if i == 0 else 2, padding=1, bias=False)
            for i in range(levels)])
        if kind != "symmetric":
            self.synthesis = nn.ParameterList([
                nn.Parameter(layer.weight.detach().clone()) for layer in self.analysis])
        self.coefficient_net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, 1 + levels * width, 3, padding=1))
        self.log_scale = nn.Parameter(torch.arange(levels, dtype=torch.float32) * 1.38629436112)
        self.width = width

    def forward(self, features, residual, diagonal, grid_cells):
        """features depend only on k/mesh; residual/diagonal have (B,1,H,W)."""
        invsqrt = diagonal.rsqrt()
        # This fixed mesh scaling is linear, including for negative/mixed goals.
        scale = float(grid_cells ** 2)
        v = residual * invsqrt * scale
        gates = F.softplus(self.coefficient_net(features)) + 1e-4
        result = gates[:, :1] * v
        states, shapes = [], []
        x = v
        for i, layer in enumerate(self.analysis):
            shapes.append(x.shape[-2:])
            x = layer(x)
            if self.kind == "nonlinear":
                x = F.silu(x)
            states.append(x)
        for level, state in enumerate(states):
            gate = gates[:, 1+level*self.width:1+(level+1)*self.width]
            gate = F.interpolate(gate, size=state.shape[-2:], mode="bilinear", align_corners=False)
            x = state * gate * self.log_scale[level].exp()
            for j in range(level, -1, -1):
                stride = 1 if j == 0 else 2
                wanted = shapes[j]
                natural = tuple((n-1)*stride+1 for n in x.shape[-2:])
                op = tuple(a-b for a,b in zip(wanted,natural))
                weight = (self.analysis[j].weight if self.kind == "symmetric"
                          else self.synthesis[j])
                x = F.conv_transpose2d(x, weight, stride=stride, padding=1, output_padding=op)
                if self.kind == "nonlinear" and j > 0:
                    x = F.silu(x)
            result = result + x
        return result * invsqrt / scale


def energy(v, diagonal, east, south):
    """Five-point SPD P1 matrix quadratic form, independently tested against CSR."""
    out = (diagonal * v.square()).sum(dim=(-3,-2,-1))
    out += 2 * (east * v[..., :, :-1] * v[..., :, 1:]).sum(dim=(-3,-2,-1))
    out += 2 * (south * v[..., :-1, :] * v[..., 1:, :]).sum(dim=(-3,-2,-1))
    return out
