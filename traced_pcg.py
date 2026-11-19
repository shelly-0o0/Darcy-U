"""Optional, cost-accounted PCG observations for sequence recycling.

The solver follows ``numerics.solve_pcg``: SciPy CG with a fixed
preconditioner, a true-residual acceptance check, and at most three
residual-refresh continuations within one total iteration budget.
Captured vectors are completed iterate increments, never reference errors.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.sparse.linalg as sla


def solve_pcg_traced(A, b, M=None, x0=None, rtol=1e-8, maxiter=500,
                     max_vectors=0, record_residuals=False):
    """Return ``(x, metrics, basis, trace)`` with optional observations.

    ``basis`` contains the first ``max_vectors`` finite, nonzero increments
    ``x_k - x_{k-1}``, including increments from residual-refresh runs.
    All copies, differences and diagnostic residual applications are timed.
    Setting ``max_vectors=0`` disables vector differences and copies in the
    callback. ``record_residuals`` is an offline diagnostic option: each
    completed iteration then records its true residual and ``r.T @ M @ r``
    (``r.T @ r`` when M is absent), including their cost in ``seconds``.

    Iterations are numbered from one over the entire solve; ``restart=0``
    is the initial CG invocation and subsequent restarts are 1, 2, and 3.
    No initial or final-only observation is added to the per-iteration trace.
    """
    if not isinstance(max_vectors, (int, np.integer)) or max_vectors < 0:
        raise ValueError("max_vectors must be a nonnegative integer")
    iterations = 0
    restarts = 0
    basis = []
    trace = []
    previous_x = None
    trace_bnorm = None

    def callback(current_x):
        nonlocal iterations, previous_x
        iterations += 1
        if len(basis) < max_vectors:
            increment = current_x - previous_x
            if np.all(np.isfinite(increment)) and np.any(increment != 0.):
                # Subtraction already owns its buffer; no extra copy needed.
                basis.append(increment)
            # SciPy mutates its iterate in place on the following iteration.
            previous_x = (current_x.copy()
                          if len(basis) < max_vectors else None)
        if record_residuals:
            r = np.asarray(b).reshape(-1) - A @ current_x
            norm = float(np.linalg.norm(r))
            mr = r if M is None else M @ r
            trace.append({
                "iteration": iterations,
                "restart": restarts,
                "residual_norm": norm,
                "relative_residual": float(norm / trace_bnorm),
                "preconditioned_residual_squared": float(r @ mr),
            })

    start = perf_counter()
    if max_vectors:
        previous_x = (np.zeros(np.asarray(b).size, dtype=np.asarray(b).dtype)
                      if x0 is None else np.asarray(x0).reshape(-1).copy())
    if record_residuals:
        trace_bnorm = max(np.linalg.norm(b), np.finfo(float).tiny)
    x, info = sla.cg(A, b, M=M, x0=x0, rtol=rtol, atol=0.,
                     maxiter=maxiter, callback=callback)
    initial_info = int(info)
    bnorm = max(np.linalg.norm(b), np.finfo(float).tiny)
    residual = float(np.linalg.norm(b - A @ x) / bnorm)
    initial_residual = residual
    while (residual > rtol and restarts < 3 and iterations < maxiter
           and info >= 0):
        restarts += 1
        if len(basis) < max_vectors:
            # A continuation starts from the returned iterate, including
            # when the preceding invocation completed without callbacks.
            previous_x = x.copy()
        x, info = sla.cg(A, b, M=M, x0=x, rtol=rtol * (.2 ** restarts),
                         atol=0., maxiter=maxiter - iterations,
                         callback=callback)
        residual = float(np.linalg.norm(b - A @ x) / bnorm)
    elapsed = perf_counter() - start
    return x, {
        "iterations": iterations,
        "info": int(info),
        "seconds": elapsed,
        "initial_info": initial_info,
        "initial_relative_residual": initial_residual,
        "restarts": restarts,
        "relative_residual": residual,
        "converged": bool(residual <= rtol),
        "captured_vectors": len(basis),
        "record_residuals": bool(record_residuals),
    }, basis, trace
