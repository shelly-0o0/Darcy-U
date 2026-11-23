"""Predeclared stratified, without-replacement aligned geometry sample.

Only geometry metadata may be inspected before the Stage15C freeze.  No
solution, indicator, or performance value influences this generator.
"""
from itertools import combinations, product
import numpy as np
from stage15b_cases import Case, source, goal

SEED = 1520261123
FAMILIES = ('layers', 'barrier', 'channel', 'inclusion')


def specifications():
    pools = {
        'layers': [v for v in combinations(range(1,8),3) if v != (2,4,6)],
        'barrier': [(a,b) for a,b in combinations(range(1,8),2)
                    if b-a in (1,2) and (a,b)!=(4,5)],
        'channel': [(a,b) for a,b in combinations(range(1,8),2)
                    if b-a in (1,2,3) and (a,b)!=(3,5)],
        'inclusion': [x+y for x,y in product(combinations(range(1,8),2), repeat=2)
                      if 1<=x[1]-x[0]<=3 and 1<=y[1]-y[0]<=4 and x+y!=(4,6,2,6)],
    }
    result=[]
    for fi,family in enumerate(FAMILIES):
        rng=np.random.default_rng(SEED+fi)
        selected=rng.choice(len(pools[family]),size=6,replace=False)
        for j,idx in enumerate(selected):
            result.append(dict(name=f'{family}_{j:02d}',family=family,
                               geometry_eighths=list(pools[family][int(idx)]),
                               contrast=10 if j%2==0 else 1000,
                               seed=SEED+fi,pool_size=len(pools[family])))
    return result


def from_spec(spec):
    family=spec['family']; cuts=tuple(v/8 for v in spec['geometry_eighths'])
    low=1./spec['contrast']
    if family=='layers':
        def coefficient(x,y):
            stripe=sum((x>v).astype(int) for v in cuts)
            return np.where(stripe%2==1,low,1.)
        ix,iy=cuts,()
    elif family=='barrier':
        def coefficient(x,y): return np.where((x>cuts[0]) & (x<cuts[1]),low,1.)
        ix,iy=cuts,()
    elif family=='channel':
        def coefficient(x,y): return np.where((y>cuts[0]) & (y<cuts[1]),1.,low)
        ix,iy=(),cuts
    elif family=='inclusion':
        def coefficient(x,y):
            return np.where((x>cuts[0]) & (x<cuts[1]) & (y>cuts[2]) & (y<cuts[3]),low,1.)
        ix,iy=cuts[:2],cuts[2:]
    else: raise ValueError('unknown material family')
    metadata=dict(spec,k_min=low,k_max=1.,source_center=[.25,.5],goal_center=[.75,.5],
                  radius=.18,source_integral=1.,goal_integral=1.,manufactured=False,
                  coefficient_alignment='element_aligned',interface_x=list(ix),interface_y=list(iy),
                  geometry_sampling='seeded stratified without replacement; finite aligned pool')
    return Case(spec['name'],coefficient,source,goal,metadata,ix,iy)


def cases():
    return [from_spec(s) for s in specifications()]
