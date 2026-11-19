"""Stage-9 semilinear convection/diffusion and inexact Newton core.

Uniform interior-node finite differences on (0,1)^2 use centered diffusion
and conservative first-order upwind convection with zero Dirichlet data.
All solver costs, including true linear/nonlinear residual checks, are timed.
Problem assembly and optional audit-snapshot copying are reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from time import perf_counter
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

ETA_INITIAL, ETA_MIN, ETA_MAX = .5, 1e-10, .9
EW2_GAMMA, EW2_POWER = .9, 2.
GOLDEN_POWER = (1. + math.sqrt(5.)) / 2.
ARMIJO, MAX_BACKTRACKS = 1e-4, 20
GMRES_RESTART, GMRES_CYCLES = 50, 20
ILU_DROP_TOL, ILU_FILL_FACTOR = 1e-4, 10.
METHODS = ('ew1', 'ew2', 'fixed_0.01', 'fixed_0.1', 'fixed_0.5', 'fixed_0.9', 'direct')
COMPONENTS = ('jacobian', 'preconditioner', 'linear', 'line_search', 'residual', 'forcing')


def _vector(value, dimension):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (dimension,):
        raise ValueError(f'Expected a vector of shape ({dimension},)')
    return value


@dataclass
class Problem:
    n: int
    h: float
    dimension: int
    params: dict
    rhs: np.ndarray
    exact: np.ndarray | None
    linear_operator: sp.csr_matrix
    coordinates: tuple[np.ndarray, np.ndarray]
    manufactured: bool
    signature: str
    assembly_seconds: float = 0.

    def F(self, u):
        u = _vector(u, self.dimension)
        with np.errstate(over='ignore', invalid='ignore'):
            return self.linear_operator @ u + self.params['gamma'] * u**3 - self.rhs

    def jacobian(self, u):
        u = _vector(u, self.dimension)
        with np.errstate(over='ignore', invalid='ignore'):
            diagonal = 3. * self.params['gamma'] * u**2
        return (self.linear_operator + sp.diags(diagonal, format='csr')).tocsr()


def make_problem(n, params, manufactured=False):
    """Build an n-by-n interior-node problem (dimension n**2).

    Parameters: eps>0, gamma>=0, beta=(bx,by), amplitude, cx, cy. Defaults
    are eps=.01, gamma=1, beta=(1,.5), amplitude=1, center=(.5,.5).
    Manufactured data use the *analytic* PDE forcing for amplitude*sin(pi*x)
    *sin(pi*y), not the discrete operator applied to the exact grid values.
    """
    start = perf_counter()
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive number of interior nodes')
    p = dict(eps=.01, gamma=1., beta=(1., .5), amplitude=1., cx=.5, cy=.5)
    if set(params) - set(p):
        raise ValueError(f'Unknown parameters: {sorted(set(params) - set(p))}')
    p.update(params)
    try:
        beta = tuple(float(v) for v in p['beta'])
    except (TypeError, ValueError) as error:
        raise ValueError('beta must contain two finite components') from error
    if len(beta) != 2:
        raise ValueError('beta must have two components')
    p = {key: float(value) for key, value in p.items() if key != 'beta'}
    p['beta'] = beta
    if not all(np.isfinite(v) for v in [*beta, *[p[k] for k in p if k != 'beta']]):
        raise ValueError('All problem parameters must be finite')
    if p['eps'] <= 0. or p['gamma'] < 0.:
        raise ValueError('eps must be positive and gamma nonnegative')
    n, h = int(n), 1. / (int(n) + 1)

    def axis_operator(velocity):
        diffusion = p['eps'] / h**2
        return sp.diags((np.full(n-1, -diffusion-max(velocity, 0.)/h),
                         np.full(n, 2.*diffusion+abs(velocity)/h),
                         np.full(n-1, -diffusion-max(-velocity, 0.)/h)),
                        (-1, 0, 1), shape=(n, n), format='csr')

    identity = sp.eye(n, format='csr')
    # C-order flattening: x is the fast coordinate; tensor products prevent
    # any artificial coupling between the end and start of adjacent rows.
    matrix = (sp.kron(identity, axis_operator(beta[0]), format='csr') +
              sp.kron(axis_operator(beta[1]), identity, format='csr') +
              sp.eye(n*n, format='csr'))
    coordinates = np.arange(1, n+1, dtype=np.float64) * h
    x, y = np.meshgrid(coordinates, coordinates, indexing='xy')
    sine = np.sin(np.pi*x) * np.sin(np.pi*y)
    exact = None
    if manufactured:
        exact = p['amplitude'] * sine
        dx = p['amplitude'] * np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)
        dy = p['amplitude'] * np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)
        rhs = (2.*p['eps']*np.pi**2+1.)*exact + beta[0]*dx + beta[1]*dy + p['gamma']*exact**3
        exact = exact.ravel()
        exact.flags.writeable = False
    else:
        rhs = p['amplitude'] * np.exp(-((x-p['cx'])**2+(y-p['cy'])**2)/(2.*.15**2))
        rhs += .1*p['amplitude']*sine
    rhs = np.asarray(rhs, dtype=np.float64).ravel()
    rhs.flags.writeable = False
    x.flags.writeable = y.flags.writeable = False
    signature = hashlib.sha256(json.dumps(dict(n=n, params=p, manufactured=bool(manufactured)),
                                          sort_keys=True).encode()).hexdigest()
    problem = Problem(n, h, n*n, p, rhs, exact, matrix, (x, y), bool(manufactured), signature)
    problem.assembly_seconds = perf_counter() - start
    return problem


def clip_eta(eta):
    return float(np.clip(eta, ETA_MIN, ETA_MAX))


def forcing_details(method, norm, prevnorm=None, prevlinearizednorm=None, preveta=None):
    """Native forcing term; EW2 uses the prespecified power 2, gamma .9."""
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    if method == 'direct':
        return dict(native_eta=None, raw_eta=None, safeguard_eta=None, safeguard_applied=False)
    if method.startswith('fixed_'):
        raw = float(method.removeprefix('fixed_'))
        return dict(native_eta=clip_eta(raw), raw_eta=raw,
                    safeguard_eta=None, safeguard_applied=False)
    if prevnorm is None:
        return dict(native_eta=ETA_INITIAL, raw_eta=ETA_INITIAL,
                    safeguard_eta=None, safeguard_applied=False)
    if not (np.isfinite(norm) and norm >= 0. and np.isfinite(prevnorm) and prevnorm > 0.
            and preveta is not None and np.isfinite(preveta) and ETA_MIN <= preveta < 1.):
        raise ValueError('Invalid forcing-controller history')
    if method == 'ew1':
        if prevlinearizednorm is None or not np.isfinite(prevlinearizednorm) or prevlinearizednorm < 0.:
            raise ValueError('EW1 requires the accepted previous linearized residual norm')
        raw = abs(norm-prevlinearizednorm)/prevnorm
        safeguard = preveta**GOLDEN_POWER
    else:
        raw = EW2_GAMMA*(norm/prevnorm)**EW2_POWER
        safeguard = EW2_GAMMA*preveta**EW2_POWER
    active = safeguard > .1
    native = clip_eta(max(raw, safeguard) if active else raw)
    return dict(native_eta=native, raw_eta=float(raw), safeguard_eta=float(safeguard),
                safeguard_applied=bool(active))


def forcing_eta(method, norm, prevnorm=None, prevlinearizednorm=None, preveta=None):
    return forcing_details(method, norm, prevnorm, prevlinearizednorm, preveta)['native_eta']


def solve(problem, method, initial_state=None, capture=False, first_multiplier=1.,
          rtol=1e-8, atol=1e-10, max_newton=50):
    """Solve F(u)=0 with a fixed ILU per Newton linear solve.

    ``total_seconds`` excludes Problem assembly and optional snapshot-copy
    time only. ``line_search`` includes trial F evaluations; ``residual``
    includes initial/final F, true linear verification, and the accepted
    old-Jacobian linearization. Components do not overlap; overhead_seconds
    accounts for remaining control/trace work.

    A restart retains the original F0 norm and global iteration cap. The
    multiplier scales only the first requested eta of this call, after
    the native controller and before clipping. Later EW safeguards use
    the previous accepted-step bound eta_eff=(1-alpha)+alpha*requested_eta,
    which can exceed the requested cap when damping is needed. No direct
    recovery is permitted.
    """
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    if not (np.isfinite(first_multiplier) and first_multiplier > 0.):
        raise ValueError('first_multiplier must be finite and positive')
    if not (np.isfinite(rtol) and rtol > 0. and np.isfinite(atol) and atol >= 0.):
        raise ValueError('Invalid nonlinear tolerances')
    if isinstance(max_newton, (bool, np.bool_)) or not isinstance(max_newton, (int, np.integer)) or max_newton < 0:
        raise ValueError('max_newton must be a nonnegative integer')
    start = perf_counter()
    components = dict.fromkeys(COMPONENTS, 0.)
    snapshot_seconds = 0.
    snapshots, trace = [], []
    if initial_state is None:
        u = np.zeros(problem.dimension, dtype=np.float64)
        iteration, prevnorm, prevlinearizednorm, preveta = 0, None, None, None
        F0norm = None
    else:
        if initial_state.get('problem_signature') != problem.signature:
            raise ValueError('Restart state belongs to a different problem')
        u = _vector(initial_state['u'], problem.dimension).copy()
        if not np.all(np.isfinite(u)):
            raise ValueError('Restart iterate must be finite')
        iteration = initial_state['iteration']
        if isinstance(iteration, (bool, np.bool_)) or not isinstance(iteration, (int, np.integer)) or iteration < 0:
            raise ValueError('Invalid restart iteration')
        F0norm = float(initial_state['F0norm'])
        if not np.isfinite(F0norm) or F0norm < 0.:
            raise ValueError('Invalid original nonlinear residual norm')
        prevnorm, prevlinearizednorm, preveta = [initial_state[key]
            for key in ('prevnorm', 'prevlinearizednorm', 'preveta')]
        if iteration > 0 and (prevnorm is None or prevlinearizednorm is None or preveta is None):
            raise ValueError('Incomplete restart forcing history')
    initial_iteration = int(iteration)
    t = perf_counter()
    F = problem.F(u)
    norm = float(np.linalg.norm(F))
    if F0norm is None:
        F0norm = norm
    tolerance = max(atol, rtol*F0norm)
    components['residual'] += perf_counter()-t
    linear_iterations = linear_failures = rejections = accepted = 0
    status = 'max_newton'

    def snapshot():
        nonlocal snapshot_seconds
        if not capture:
            return
        tcopy = perf_counter()
        value = u.copy()
        value.flags.writeable = False
        native = forcing_eta(method, norm, prevnorm, prevlinearizednorm, preveta) \
            if np.isfinite(norm) else None
        snapshots.append(dict(u=value, F0norm=F0norm, prevnorm=prevnorm,
            prevlinearizednorm=prevlinearizednorm, preveta=preveta, iteration=int(iteration),
            residual_norm=norm, absolute_residual=norm, native_eta=native,
            converged=bool(np.isfinite(norm) and norm <= tolerance), method=method,
            problem_signature=problem.signature))
        snapshot_seconds += perf_counter()-tcopy

    for _ in range(initial_iteration, int(max_newton)):
        snapshot()
        if not np.isfinite(norm):
            status = 'nonfinite_nonlinear_residual'
            break
        if norm <= tolerance:
            status = 'converged'
            break
        attempt_start = perf_counter()
        row = dict(iteration=int(iteration), residual_before=norm, residual_after=None,
            accepted_alpha=None, accepted=False, line_search_trials=0,
            line_search_rejections=0, linear_iterations=0, linear_info=None,
            linear_converged=False, actual_eta=None,
            achieved_linear_relative_residual=None,
            accepted_linearized_residual_norm=None, accepted_actual_eta=None,
            effective_eta=None, error=None)
        t = perf_counter()
        details = forcing_details(method, norm, prevnorm, prevlinearizednorm, preveta)
        multiplier = first_multiplier if not trace else 1.
        eta = None if method == 'direct' else clip_eta(details['native_eta']*multiplier)
        target = ETA_MIN if method == 'direct' else eta
        row.update(details, requested_eta=eta, eta=eta, first_multiplier=multiplier,
                   linear_relative_target=target)
        components['forcing'] += perf_counter()-t
        t = perf_counter()
        J = problem.jacobian(u)
        components['jacobian'] += perf_counter()-t
        M = None
        if method != 'direct':
            t = perf_counter()
            try:
                ilu = sla.spilu(J.tocsc(), drop_tol=ILU_DROP_TOL, fill_factor=ILU_FILL_FACTOR)
                M = sla.LinearOperator(J.shape, matvec=ilu.solve, dtype=np.float64)
            except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                row['error'] = f'{type(error).__name__}: {error}'
                status = 'preconditioner_failure'
            finally:
                components['preconditioner'] += perf_counter()-t
            if row['error'] is not None:
                linear_failures += 1
                row['failure'] = status
                row['seconds'] = perf_counter()-attempt_start
                trace.append(row)
                break
        count = [0]

        def count_iteration(_residual):
            count[0] += 1

        step, info = None, None
        t = perf_counter()
        try:
            if method == 'direct':
                with warnings.catch_warnings():
                    warnings.simplefilter('error', sla.MatrixRankWarning)
                    step = sla.spsolve(J.tocsc(), -F)
                info = 0
            else:
                step, info = sla.gmres(J, -F, M=M, restart=GMRES_RESTART,
                    maxiter=GMRES_CYCLES, rtol=eta, atol=0.,
                    callback=count_iteration, callback_type='pr_norm')
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError,
                sla.MatrixRankWarning) as error:
            row['error'] = f'{type(error).__name__}: {error}'
        finally:
            components['linear'] += perf_counter()-t
        row['linear_iterations'] = count[0]
        linear_iterations += count[0]
        row['linear_info'] = int(info) if info is not None else None
        t = perf_counter()
        Jstep = None
        if step is not None and np.shape(step) == u.shape and np.all(np.isfinite(step)):
            Jstep = J @ step
            actual_linear_norm = float(np.linalg.norm(F+Jstep))
            actual_eta = actual_linear_norm/norm
            linear_ok = bool(info == 0 and np.isfinite(actual_eta) and actual_eta <= target)
            row.update(actual_eta=actual_eta, achieved_linear_relative_residual=actual_eta,
                       linear_absolute_residual=actual_linear_norm)
        else:
            linear_ok = False
        row['linear_converged'] = linear_ok
        components['residual'] += perf_counter()-t
        if not linear_ok:
            linear_failures += 1
            status = 'linear_failure'
            row['failure'] = status
            row['seconds'] = perf_counter()-attempt_start
            trace.append(row)
            break
        t = perf_counter()
        alpha = 1.
        trial_u, trial_F, trial_norm = None, None, None
        for backtrack in range(MAX_BACKTRACKS+1):
            candidate = u+alpha*step
            candidate_F = problem.F(candidate)
            candidate_norm = float(np.linalg.norm(candidate_F))
            row['line_search_trials'] += 1
            if (np.isfinite(candidate_norm) and candidate_norm < norm and
                    candidate_norm <= (1.-ARMIJO*alpha)*norm):
                trial_u, trial_F, trial_norm = candidate, candidate_F, candidate_norm
                break
            row['line_search_rejections'] += 1
            if backtrack < MAX_BACKTRACKS:
                alpha *= .5
        components['line_search'] += perf_counter()-t
        rejections += row['line_search_rejections']
        if trial_u is None:
            status = 'line_search_failure'
            row['failure'] = status
            row['seconds'] = perf_counter()-attempt_start
            trace.append(row)
            break
        t = perf_counter()
        accepted_linearized_norm = float(np.linalg.norm(F+alpha*Jstep))
        components['residual'] += perf_counter()-t
        # The EW1 history uses the accepted damped displacement, not the
        # rejected full Newton step or its undamped linear residual.
        effective_eta = (1.-alpha)+alpha*target
        accepted_actual_eta = accepted_linearized_norm/norm
        prevnorm, prevlinearizednorm = norm, accepted_linearized_norm
        preveta = effective_eta
        u, F, norm = trial_u, trial_F, trial_norm
        iteration += 1
        accepted += 1
        row.update(accepted=True, accepted_alpha=alpha, residual_after=norm,
                   accepted_linearized_residual_norm=accepted_linearized_norm,
                   accepted_actual_eta=accepted_actual_eta, effective_eta=effective_eta,
                   step_norm=float(np.linalg.norm(alpha*step)),
                   seconds=perf_counter()-attempt_start)
        trace.append(row)
    # Always independently recompute the final nonlinear residual.
    t = perf_counter()
    F = problem.F(u)
    norm = float(np.linalg.norm(F))
    converged = bool(np.isfinite(norm) and norm <= tolerance and linear_failures == 0)
    components['residual'] += perf_counter()-t
    if converged:
        status = 'converged'
    if capture and (not snapshots or snapshots[-1]['iteration'] != iteration):
        snapshot()
    total_seconds = perf_counter()-start-snapshot_seconds
    relative = norm/F0norm if F0norm > 0. else (0. if norm == 0. else float('inf'))
    return dict(u=u, converged=converged, status=status,
        absolute_residual=norm, relative_residual=relative, F0norm=F0norm,
        tolerance=tolerance, total_seconds=total_seconds, components=components,
        overhead_seconds=total_seconds-sum(components.values()),
        snapshot_seconds_excluded=snapshot_seconds, snapshot_timing='Audit-state copies and their derived native eta are outside solver timing.',
        assembly_seconds_excluded=problem.assembly_seconds,
        newton_iterations=accepted, newton_attempts=len(trace), initial_iteration=initial_iteration,
        final_iteration=int(iteration), linear_iterations=linear_iterations,
        line_search_rejections=rejections, linear_failures=linear_failures,
        trace=trace, **({'snapshots': snapshots} if capture else {}))
