"""Analytic problems for numerical verification, not a representative dataset.

All three goals are normalized compact regional averages.  Their exact values
are evaluated by independent one-dimensional Gauss integration of the analytic
solution, never by a finite-element reference solve.  The layered manufactured
source changes with the material; these cases cannot support an AI speed claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss


GOAL_CENTERS = ((.25, .25), (.75, .25), (.50, .75))
GOAL_RADIUS = .22
BUMP_INTEGRAL = 256. / 315.


def bump_1d(x, center, radius=GOAL_RADIUS):
    if not np.isfinite(radius) or radius <= 0. or not np.isfinite(center):
        raise ValueError("radius must be finite and positive; center must be finite")
    s = (np.asarray(x) - center) / radius
    return np.maximum(1. - s * s, 0.) ** 4 / (radius * BUMP_INTEGRAL)


def regional_goals():
    return tuple((lambda x, y, cx=cx, cy=cy:
                  bump_1d(x, cx) * bump_1d(y, cy)) for cx, cy in GOAL_CENTERS)


def _integrate(function, cuts, order):
    nodes, weights = leggauss(order)
    return sum(.5 * (b-a) * np.dot(weights, function(.5*(a+b)+.5*(b-a)*nodes))
               for a, b in zip(cuts[:-1], cuts[1:]) if b > a)


@dataclass(frozen=True)
class DarcyCase:
    name: str
    coefficient: Callable
    coefficient_grad: Callable
    source: Callable
    exact: Callable
    exact_grad: Callable
    goals: tuple
    exact_qoi: np.ndarray
    metadata: dict
    interfaces: tuple
    x_factor: Callable
    y_factor: Callable

    def analytic_goals(self, order=64):
        answers = []
        for cx, cy in GOAL_CENTERS:
            cuts = sorted({cx-GOAL_RADIUS, cx+GOAL_RADIUS,
                           *(a for a in self.interfaces if cx-GOAL_RADIUS < a < cx+GOAL_RADIUS)})
            ix = _integrate(lambda x: self.x_factor(x)*bump_1d(x, cx), cuts, order)
            iy = _integrate(lambda y: self.y_factor(y)*bump_1d(y, cy),
                            [cy-GOAL_RADIUS, cy+GOAL_RADIUS], order)
            answers.append(ix*iy)
        return np.array(answers)


def manufactured_case(kind="layered", contrast=10., barrier=False):
    if kind not in ("constant", "smooth", "layered"):
        raise ValueError("kind must be constant, smooth or layered")
    contrast = float(contrast)
    if not np.isfinite(contrast) or contrast < 1.:
        raise ValueError("contrast must be finite and at least one")
    if kind != "layered" and barrier:
        raise ValueError("barrier only applies to layered cases")
    pi = np.pi
    sy = lambda y: np.sin(pi*np.asarray(y))
    if kind == "smooth":
        k = lambda x, y: np.exp(.5*np.asarray(x)+.25*np.asarray(y))
        kg = lambda x, y: np.stack((.5*k(x,y), .25*k(x,y)))
        vx = lambda x: np.sin(pi*np.asarray(x))
        exact = lambda x, y: vx(x)*sy(y)
        grad = lambda x, y: np.stack(np.broadcast_arrays(
            pi*np.cos(pi*np.asarray(x))*sy(y),
            pi*vx(x)*np.cos(pi*np.asarray(y))))
        source = lambda x, y: k(x,y)*(2*pi*pi*exact(x,y)
            -.5*pi*np.cos(pi*np.asarray(x))*sy(y)
            -.25*pi*vx(x)*np.cos(pi*np.asarray(y)))
        interfaces = ()
        name = "smooth_exp"
        metadata = dict(kind=kind, coefficient_alignment="smooth",
                        k_min=1., k_max=float(np.exp(.75)), interfaces=[])
    else:
        breaks = np.array([0., .25, .5, .75, 1.]) if kind == "layered" else np.array([0., 1.])
        value = 1./contrast if barrier else contrast
        levels = np.array([1., value, 1., value]) if kind == "layered" else np.ones(1)
        c = float(np.sum((breaks[1:]**2-breaks[:-1]**2)/(2*levels)) /
                  np.sum(np.diff(breaks)/levels))

        def k(x, y):
            xx, _ = np.broadcast_arrays(np.asarray(x), np.asarray(y))
            ix = np.clip(np.searchsorted(breaks, xx, side="right")-1, 0, len(levels)-1)
            return levels[ix]

        def vx(x):
            xx = np.asarray(x)
            result = np.zeros_like(xx, dtype=float)
            for a, b, kval in zip(breaks[:-1], breaks[1:], levels):
                t = np.clip(xx, a, b)
                result += (c*(t-a)-.5*(t*t-a*a))/kval
            return result

        kg = lambda x, y: np.zeros((2,)+np.broadcast_shapes(np.shape(x), np.shape(y)))
        exact = lambda x, y: vx(x)*sy(y)
        grad = lambda x, y: np.stack(np.broadcast_arrays(
            (c-np.asarray(x))/k(x,y)*sy(y),
            pi*vx(x)*np.cos(pi*np.asarray(y))))
        source = lambda x, y: (1.+pi*pi*k(x,y)*vx(x))*sy(y)
        interfaces = tuple(float(x) for x in breaks[1:-1])
        name = "constant" if kind == "constant" else f"layered_{'barrier' if barrier else 'conductive'}_{contrast:g}"
        metadata = dict(kind=kind, coefficient_alignment="element_aligned",
                        k_min=float(levels.min()), k_max=float(levels.max()),
                        interfaces=list(interfaces), layer_values=levels.tolist(),
                        c=c, contrast=float(levels.max()/levels.min()), barrier=bool(barrier))
    metadata.update(goal_centers=[list(c) for c in GOAL_CENTERS], goal_radius=GOAL_RADIUS,
                    goal_kind="normalized compact tensor polynomial C3 regional average",
                    manufactured_source=True, scale_from_input=1./metadata["k_min"])
    case = DarcyCase(name, k, kg, source, exact, grad, regional_goals(),
                     np.empty(3), metadata, interfaces, vx, sy)
    q = case.analytic_goals(64)
    q.flags.writeable = False
    return DarcyCase(name, k, kg, source, exact, grad, case.goals, q, metadata, interfaces, vx, sy)


def verification_cases():
    return [manufactured_case("constant"), manufactured_case("smooth"),
            *(manufactured_case("layered", c) for c in (10., 100., 1000.)),
            manufactured_case("layered", 1000., barrier=True)]
