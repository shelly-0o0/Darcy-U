"""Reproducible local coefficient changes for the Darcy solver pilot.

Arrays use ``k[iy, ix]`` and represent cellwise constant coefficients on the
unit square. Geometry is sampled on a fixed 32 by 32 design lattice, so N=64,
128, and 256 describe exactly the same physical geometry for a given seed.
There is no global random state or implicit training/test split: callers must
assign disjoint seeds to complete sequences before extracting local patches.
"""

from __future__ import annotations

import numpy as np


DESIGN_N = 32
FAMILIES = ("drift", "moving", "topology", "mixed")


def _design(seed: int) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    n = DESIGN_N
    mask = np.zeros((n, n), dtype=bool)
    width = int(rng.integers(1, 3))
    left = int(rng.integers(4, 8))
    right = int(rng.integers(24, 28))
    lane = int(rng.integers(13, 19))
    top = int(rng.integers(3, 6))
    bottom = int(rng.integers(26, 30))
    gap_left = int(rng.integers(12, 15))
    gap_right = int(rng.integers(18, 21))
    branch_y = top + int(rng.integers(2, 5))
    # Two separate connected channel systems, with a guaranteed open gap.
    mask[top:bottom, left:left + width] = True
    mask[top:bottom, right:right + width] = True
    mask[lane:lane + width, left:gap_left] = True
    mask[lane:lane + width, gap_right:right + width] = True
    mask[branch_y:branch_y + width, left:gap_left - 1] = True
    return mask, {
        "width": width,
        "left": left,
        "right": right,
        "lane": lane,
        "gap_left": gap_left,
        "gap_right": gap_right,
        "drift_y": top + 1,
        "inclusion_y": bottom - 3,
        "inclusion_x": gap_left,
    }


def coefficient_sequence(
    N: int,
    steps: int = 20,
    seed: int = 0,
    contrast: float = 1e4,
    family: str = "mixed",
) -> tuple[np.ndarray, list[dict]]:
    """Return ``(coefficients[step, iy, ix], step_metadata)``.

    ``topology`` connects two channel systems halfway through the sequence
    and blocks their bridge at three quarters. ``moving`` translates a small
    high-permeability inclusion every third step. ``drift`` changes only a
    short section of one channel. ``mixed`` combines these changes.

    N must be a positive multiple of 32, steps >= 4, and contrast > 1.
    Topological events refer to the high-permeability channel geometry; the
    elliptic coefficient remains strictly positive everywhere.
    """
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < DESIGN_N or N % DESIGN_N:
        raise ValueError("N must be an integer positive multiple of 32")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 4:
        raise ValueError("steps must be an integer >= 4")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if not np.isfinite(contrast) or contrast <= 1:
        raise ValueError("contrast must be finite and greater than one")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    base_mask, design = _design(int(seed))
    factor = N // DESIGN_N
    output = np.empty((steps, N, N), dtype=np.float64)
    metadata = []
    connect_step = steps // 2
    block_step = 3 * steps // 4
    width = design["width"]
    lane = design["lane"]
    gl, gr = design["gap_left"], design["gap_right"]
    prev_coarse = None
    prev_position = None

    for step in range(steps):
        mask = base_mask.copy()
        events = []
        bridge_state = "open_gap"
        if family in ("topology", "mixed") and step >= connect_step:
            mask[lane:lane + width, gl:gr] = True
            bridge_state = "connected"
            if step == connect_step:
                events.append("bridge_connected")
            if step >= block_step:
                middle = (gl + gr) // 2
                mask[lane:lane + width, middle:middle + 1] = False
                bridge_state = "blocked"
                if step == block_step:
                    events.append("bridge_blocked")

        inclusion_position = None
        if family in ("moving", "mixed"):
            # Reflect along a short, isolated track, never jumping end-to-end.
            phase = (step // 3) % 6
            offset = phase if phase <= 3 else 6 - phase
            inclusion_position = design["inclusion_x"] + offset
            iy = design["inclusion_y"]
            mask[iy:iy + 2, inclusion_position:inclusion_position + 2] = True
            if step and inclusion_position != prev_position:
                events.append("inclusion_moved")

        coarse = np.where(mask, float(contrast), 1.0)
        if family in ("drift", "mixed"):
            # A small positive drift; maxima remain unchanged elsewhere.
            exponent = 0.85 + 0.15 * np.cos(np.pi * step / (steps - 1))
            iy, ix = design["drift_y"], design["right"]
            coarse[iy:iy + 3, ix:ix + width] = contrast ** exponent
            if step:
                events.append("local_coefficient_drift")

        if step == 0:
            events = ["initial"]
            changed_coarse = DESIGN_N * DESIGN_N
        else:
            changed_coarse = int(np.count_nonzero(coarse != prev_coarse))
            if not events:
                events = ["unchanged"]
        fine = np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)
        output[step] = fine
        metadata.append({
            "step": step,
            "event": "+".join(events),
            "events": events,
            "family": family,
            "seed": int(seed),
            "contrast": float(contrast),
            "design_n": DESIGN_N,
            "changed_cells": changed_coarse * factor * factor,
            "change_fraction": changed_coarse / (DESIGN_N * DESIGN_N),
            "bridge_state": bridge_state,
        })
        prev_coarse = coarse
        prev_position = inclusion_position
    return output, metadata


def smooth_rhs(x, y, kind: str = "one") -> np.ndarray:
    """Evaluate fixed smooth loads at coordinates x,y (NumPy broadcasting).

    ``sine`` is the manufactured forcing for k=1 and exact solution
    sin(pi*x)sin(pi*y). ``source`` is a smooth, balanced source-sink pair.
    """
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if kind == "one":
        return np.ones_like(x)
    if kind == "sine":
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    if kind == "source":
        sigma = 0.08
        source = np.exp(-((x - 0.3)**2 + (y - 0.5)**2) / (2 * sigma**2))
        sink = np.exp(-((x - 0.7)**2 + (y - 0.5)**2) / (2 * sigma**2))
        return source - sink
    raise ValueError("kind must be one, sine, or source")
