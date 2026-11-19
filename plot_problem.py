#!/usr/bin/env python3
"""Offline scientific visualization of a saved Darcy benchmark sequence.

Usage: python plot_problem.py results/pilot_cpu

The four reference pressures are recomputed using sparse direct solves.
These solves are exclusively for this figure and are NOT online inference,
training labels, or part of any reported online benchmark timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
import numpy as np
import scipy.sparse.linalg as sla
from threadpoolctl import threadpool_limits

from fields import smooth_rhs
from numerics import assemble_system, free_to_grid


def plot_problem(run_path: Path) -> dict:
    run_path = Path(run_path).resolve()
    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    configuration = manifest["configuration"]
    candidates = sorted(run_path.glob("fields_seed*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No saved coefficient sequence in {run_path}")
    first_seed = configuration.get("test_seeds", [None])[0]
    preferred = run_path / f"fields_seed{first_seed}.npz"
    coefficient_path = preferred if preferred.exists() else candidates[0]
    seed = int(coefficient_path.stem.removeprefix("fields_seed"))
    with np.load(coefficient_path, allow_pickle=False) as saved:
        fields = np.asarray(saved["k"], dtype=np.float64)
    if fields.ndim != 3 or fields.shape[1] != fields.shape[2]:
        raise ValueError("Saved coefficients must have shape (steps,n,n)")
    steps, n, _ = fields.shape
    if steps < 4 or np.any(fields <= 0):
        raise ValueError("At least four frames with positive diffusion are required")
    selected = [0, steps // 2, 3 * steps // 4, steps - 1]
    event_path = run_path / f"events_seed{seed}.json"
    events = json.loads(event_path.read_text()) if event_path.exists() else None
    rhs_kind = configuration.get("rhs", "one")
    forcing = None if rhs_kind == "one" else lambda x, y: smooth_rhs(x, y, rhs_kind)
    solutions = []
    audits = []
    total_start = perf_counter()
    # This context keeps offline figure generation predictable and prevents
    # a BLAS implementation from launching a large pool of worker threads.
    with threadpool_limits(limits=1, user_api="blas"):
        for step in selected:
            start = perf_counter()
            A, b, _ = assemble_system(fields[step], forcing)
            solution = sla.spsolve(A, b)
            residual = float(np.linalg.norm(b - A @ solution) / np.linalg.norm(b))
            grid = free_to_grid(solution, n)
            solutions.append(grid)
            audits.append({
                "step": step,
                "event": None if events is None else events[step].get("event"),
                "coefficient_min": float(fields[step].min()),
                "coefficient_max": float(fields[step].max()),
                "pressure_min": float(grid.min()),
                "pressure_max": float(grid.max()),
                "direct_relative_residual": residual,
                "offline_assembly_and_solve_seconds": perf_counter() - start,
            })

    kmin = float(min(fields[t].min() for t in selected))
    kmax = float(max(fields[t].max() for t in selected))
    knorm = LogNorm(vmin=kmin, vmax=kmax if kmax > kmin else kmin * 1.01)
    pmin = float(min(p.min() for p in solutions))
    pmax = float(max(p.max() for p in solutions))
    if pmin < 0 < pmax:
        pnorm = TwoSlopeNorm(vmin=pmin, vcenter=0.0, vmax=pmax)
        pressure_cmap = "RdBu_r"
    else:
        pnorm = Normalize(vmin=pmin, vmax=pmax if pmax > pmin else pmin + 1.)
        pressure_cmap = "viridis"
    xy = np.linspace(0., 1., n + 1)
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.6),
                                 layout="constrained", sharex=True, sharey=True)
        kimage = pimage = None
        for j, (step, pressure) in enumerate(zip(selected, solutions)):
            kimage = axes[0, j].imshow(
                fields[step], origin="lower", extent=(0, 1, 0, 1),
                interpolation="nearest", cmap="magma", norm=knorm,
                rasterized=True,
            )
            pimage = axes[1, j].pcolormesh(
                xy, xy, pressure, shading="gouraud",
                cmap=pressure_cmap, norm=pnorm, rasterized=True,
            )
            event_label = ""
            if events is not None:
                bridge = events[step].get("bridge_state", "")
                event_label = {"open_gap": "open gap", "connected": "connected",
                               "blocked": "blocked"}.get(bridge, "")
            axes[0, j].set_title(f"Step {step}" + (f" · {event_label}" if event_label else ""))
            axes[1, j].set_xlabel("x")
            for ax in axes[:, j]:
                ax.set_aspect("equal")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([0, .5, 1])
                ax.set_yticks([0, .5, 1])
        axes[0, 0].set_ylabel("Permeability $k$\n$y$")
        axes[1, 0].set_ylabel("Reference pressure $p$\n$y$")
        kcbar = fig.colorbar(kimage, ax=axes[0, :], shrink=.9, pad=.02,
                             aspect=28)
        kcbar.set_label("$k$ (log scale)")
        pcbar = fig.colorbar(pimage, ax=axes[1, :], shrink=.9, pad=.02,
                             aspect=28)
        pcbar.set_label("$p$")
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        pcbar.formatter = formatter
        pcbar.update_ticks()
        fig.suptitle(
            f"Darcy coefficient sequence and pressure response\n"
            f"seed {seed} · {n} × {n} cells · zero Dirichlet boundary · "
            f"offline direct reference solves",
            fontsize=12,
        )
        destination = run_path / "figures"
        destination.mkdir(parents=True, exist_ok=True)
        png_path = destination / "problem_snapshots.png"
        pdf_path = destination / "problem_snapshots.pdf"
        fig.savefig(png_path, dpi=220, facecolor="white")
        fig.savefig(pdf_path, dpi=220, facecolor="white",
                    metadata={"Title": "Darcy coefficient sequence and pressure response",
                              "Subject": "Offline direct-reference visualization; excluded from online timings"})
        plt.close(fig)

    audit = {
        "purpose": "offline visualization only",
        "online_timing_included": False,
        "reference_solver": "scipy.sparse.linalg.spsolve on the original P1 FE system",
        "coefficient_file": str(coefficient_path.relative_to(run_path)),
        "coefficient_file_sha256": hashlib.sha256(coefficient_path.read_bytes()).hexdigest(),
        "seed": seed,
        "n": n,
        "rhs": rhs_kind,
        "selected_steps": selected,
        "coefficient_color_range": [kmin, kmax],
        "pressure_color_range": [pmin, pmax],
        "snapshots": audits,
        "offline_total_seconds": perf_counter() - total_start,
        "outputs": [str(png_path.relative_to(run_path)), str(pdf_path.relative_to(run_path))],
    }
    (destination / "problem_snapshots.audit.json").write_text(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path, help="saved benchmark directory, e.g. results/pilot_cpu")
    args = parser.parse_args()
    print(json.dumps(plot_problem(args.run_path), indent=2))
