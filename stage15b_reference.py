"""Fixed reference hierarchy: empirical stability, never an error certificate."""
from time import perf_counter
import numpy as np
from stage15_fem import make_mesh, assemble
from stage15b_adaptive import solve_columns, accepted
from stage15b_cases import mesh_audit

REFERENCE_CONFIG = dict(p2_n=[32,64,128,256], intorder=10, high_order=18,
                        p1_cross_n=512, min_tau=3e-4,
                        finest_difference_fraction=.05, quadrature_fraction=.01,
                        independent_order_fraction=.1, growth_allowance=1.1)


def reference_solve(case,n,degree,intorder):
    start=perf_counter()
    mesh=make_mesh(n)
    alignment=mesh_audit(mesh,case)
    p=assemble(mesh,case.coefficient,(case.source,case.goal),degree=degree,
               intorder=intorder,coefficient_alignment='element_aligned')
    u,metrics=solve_columns(p,(0,),'direct')
    qoi=float(p.loads[:,1]@u[:,0])
    return dict(n=n,degree=degree,intorder=intorder,qoi=qoi,free_dofs=len(p.free),
                assembly_seconds=p.timings['assembly_seconds'],solver=metrics,
                total_seconds=perf_counter()-start,mesh_check=alignment,
                accepted=bool(alignment['accepted'] and accepted(metrics)))


def assess_reference(rows,config):
    if len(rows)!=len(config['p2_n'])+2:
        raise ValueError('Unexpected reference row count')
    p2=[r for r in rows if r['degree']==2 and r['intorder']==config['intorder']]
    if [r['n'] for r in p2]!=config['p2_n']:
        raise ValueError('Incomplete reference hierarchy')
    high=[r for r in rows if r['degree']==2 and r['intorder']==config['high_order']]
    cross=[r for r in rows if r['degree']==1]
    if len(high)!=1 or high[0]['n']!=max(config['p2_n']) or len(cross)!=1 or cross[0]['n']!=config['p1_cross_n'] or cross[0]['intorder']!=config['intorder']:
        raise ValueError('Incomplete independent reference checks')
    q=np.array([r['qoi'] for r in p2]); delta=np.diff(q)
    tolerance=config['min_tau']
    high_q=high[0]['qoi']
    tiny=100*np.finfo(float).eps*max(1.,float(abs(q).max()))
    checks=dict(all_linear_and_mesh_checks=all(r['accepted'] for r in rows),
                finest_difference=abs(delta[-1])<=config['finest_difference_fraction']*tolerance,
                quadrature=abs(high_q-q[-1])<=config['quadrature_fraction']*tolerance,
                independent_order=abs(high_q-cross[0]['qoi'])<=config['independent_order_fraction']*tolerance,
                differences_not_growing=bool(np.all(abs(delta[1:])<=config['growth_allowance']*abs(delta[:-1])+tiny)),
                stable_difference_sign=bool(np.all(delta[abs(delta)>tiny]>0) or np.all(delta[abs(delta)>tiny]<0)))
    checks={name:bool(value) for name,value in checks.items()}
    return dict(accepted=all(checks.values()),checks=checks,qoi=high_q,
                signed_hierarchy_differences=delta.tolist(),
                last_ratio=None if abs(delta[-2])<=tiny else float(delta[-1]/delta[-2]),
                finest_difference_over_tau=float(abs(delta[-1])/tolerance),
                quadrature_difference_over_tau=float(abs(high_q-q[-1])/tolerance),
                independent_order_difference_over_tau=float(abs(high_q-cross[0]['qoi'])/tolerance),
                no_continuous_error_bound=True)


def build_reference(case,config=None):
    cfg=dict(REFERENCE_CONFIG if config is None else config)
    start=perf_counter()
    rows=[reference_solve(case,n,2,cfg['intorder']) for n in cfg['p2_n']]
    rows.append(reference_solve(case,max(cfg['p2_n']),2,cfg['high_order']))
    rows.append(reference_solve(case,cfg['p1_cross_n'],1,cfg['intorder']))
    return dict(case=case.name,config=cfg,rows=rows,assessment=assess_reference(rows,cfg),
                total_seconds=perf_counter()-start,offline_only=True)
