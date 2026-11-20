"""Six fixed, interface-aligned, non-manufactured single-goal problems.

The same unit-integral source and observation are used for every material.
No exact solution or reference value belongs to the online problem interface.
"""
from dataclasses import dataclass
import numpy as np
from stage15_cases import bump_1d


def source(x, y):
    return bump_1d(x, .25, .18) * bump_1d(y, .5, .18)


def goal(x, y):
    return bump_1d(x, .75, .18) * bump_1d(y, .5, .18)


@dataclass(frozen=True)
class Case:
    name: str
    coefficient: object
    source: object
    goal: object
    metadata: dict
    interface_x: tuple
    interface_y: tuple


def cases():
    specifications = [
        ('constant', lambda x, y: np.ones_like(x), (), (), 1.),
        ('layers_10', lambda x, y: np.where((np.floor(x*4).astype(int) % 2) == 1, .1, 1.), (.25,.5,.75), (), .1),
        ('layers_1000', lambda x, y: np.where((np.floor(x*4).astype(int) % 2) == 1, .001, 1.), (.25,.5,.75), (), .001),
        ('barrier_1000', lambda x, y: np.where((x > .5) & (x < .625), .001, 1.), (.5,.625), (), .001),
        ('channel_1000', lambda x, y: np.where((y > .375) & (y < .625), 1., .001), (), (.375,.625), .001),
        ('inclusion_1000', lambda x, y: np.where((x > .5) & (x < .75) & (y > .25) & (y < .75), .001, 1.), (.5,.75), (.25,.75), .001)]
    return [Case(name, coefficient, source, goal,
                 dict(k_min=kmin, k_max=1., contrast=1./kmin, source_center=[.25,.5],
                      goal_center=[.75,.5], radius=.18, source_integral=1., goal_integral=1.,
                      manufactured=False, coefficient_alignment='element_aligned',
                      interface_x=list(ix), interface_y=list(iy)), ix, iy)
            for name,coefficient,ix,iy,kmin in specifications]


def mesh_audit(mesh, case):
    vertices = mesh.p[:, mesh.t]
    a, b = vertices[:,1]-vertices[:,0], vertices[:,2]-vertices[:,0]
    areas = abs(a[0]*b[1]-a[1]*b[0])/2
    crossing = sum(int(np.sum((vertices[axis].min(axis=0)<cut-1e-13) &
                             (vertices[axis].max(axis=0)>cut+1e-13)))
                   for axis,cuts in enumerate((case.interface_x,case.interface_y)) for cut in cuts)
    boundary=mesh.p[:,mesh.boundary_nodes()]
    square=bool(np.all(np.any((abs(boundary)<1e-13)|(abs(boundary-1)<1e-13),axis=0)))
    return dict(accepted=bool(np.all(areas>0) and abs(areas.sum()-1)<1e-12 and crossing==0 and square),
                crossing_elements=crossing, area=float(areas.sum()), square_boundary=square)
