"""Reference-free, interface-preserving finite candidate bank.

Only given coefficient and four source/known-goal functions enter this file.
No PDE solution or held-out goal is used to construct the bank. Choosing a
candidate using its fine-reference error is a separate offline experiment.
"""
from __future__ import annotations
import numpy as np
from stage11_adaptive import budget_mark, count_added_dofs, _neighbors
from stage13_experiment import coefficient_scores
from stage13_selection_diagnostic import interior_interface_mask

BANK_NAMES=("coefficient", "seed_coefficient", "source_weighted",
            "goal_weighted", "source_goal_weighted", "seed_uniform",
            *tuple(f"random_{seed}_beta_{beta}" for seed in range(4) for beta in (0.5,1.5)))


def seeded_mark(scores, nc, budget, mandatory, forbidden=None):
    """Greedy exact-DOF completion, never removing mandatory interface cells.

    If greedy stalls, a checkerboard independent of the seed gives a
    constructive completion, each extra cell costing exactly one DOF.
    Capacity failure is explicit; the mandatory set is never relaxed.
    """
    values=np.asarray(scores,dtype=float).reshape(-1)
    seed=np.asarray(mandatory,dtype=bool).reshape(-1)
    forbidden=np.zeros(nc*nc,dtype=bool) if forbidden is None else np.asarray(forbidden,dtype=bool).reshape(-1)
    if nc<2 or values.size!=nc*nc or seed.size!=nc*nc or not np.all(np.isfinite(values)):
        raise ValueError("Invalid score/seed grid")
    if forbidden.size!=nc*nc or np.any(seed & forbidden):raise ValueError("Invalid forbidden cells")
    if int(budget)!=budget or budget<0:raise ValueError("Invalid integer budget")
    budget=int(budget);initial=count_added_dofs(np.flatnonzero(seed),nc)
    if initial>budget:raise ValueError("Mandatory seed exceeds budget")
    selected=seed.copy();spent=initial
    order=np.lexsort((np.arange(values.size),-values))
    for cell in order:
        if spent==budget:break
        if selected[cell] or forbidden[cell]:continue
        cost=1+sum(bool(selected[n]) for n in _neighbors(cell,nc))
        if spent+cost<=budget:selected[cell]=True;spent+=cost
    fallback=spent!=budget;parity_value=None
    if fallback:
        safe=~(seed | forbidden)
        for cell in np.flatnonzero(seed):
            for n in _neighbors(cell,nc):safe[n]=False
        yy,xx=np.mgrid[:nc,:nc];parity=((yy+xx)%2).ravel();options=[]
        for p in (0,1):
            available=order[safe[order] & (parity[order]==p)]
            if len(available)>=budget-initial:
                chosen=available[:budget-initial]
                options.append((float(values[chosen].sum()),p,chosen))
        if not options:raise ValueError("No exact independent-set completion with this mandatory seed")
        _,parity_value,chosen=max(options,key=lambda item:(item[0],-item[1]))
        selected=seed.copy();selected[chosen]=True
    cells=np.flatnonzero(selected)
    actual=count_added_dofs(cells,nc)
    if actual!=budget or not np.all(selected[seed]) or np.any(selected[forbidden]):
        raise RuntimeError("Seed/budget contract failed")
    return cells,dict(rule="mandatory_seed_greedy_independent_checkerboard_fallback",
        mandatory_dofs=initial,requested_budget=budget,added_dofs=actual,
        greedy_spent=spent,fallback=fallback,fallback_parity=parity_value)


def build_bank(k, functions, geometry, budget, mode="locked_ring"):
    """Four functions are source + known goals; audit goals are excluded."""
    if len(functions)!=4:raise ValueError("Exactly source plus three known goals required")
    nc=len(k)//2;mask=interior_interface_mask(k).ravel();base=coefficient_scores(k)
    if mode not in ("locked_ring","coverage"):raise ValueError("Unknown intervention")
    baseline,baseline_marking=budget_mark(base,nc,budget)
    baseline_mask=np.zeros(nc*nc,dtype=bool);baseline_mask[baseline]=True
    ring=mask.copy()
    for cell in np.flatnonzero(mask):
        for other in _neighbors(cell,nc):ring[other]=True
    mandatory=baseline_mask & ring if mode=="locked_ring" else mask
    forbidden=~baseline_mask & ring if mode=="locked_ring" else np.zeros_like(mask)
    yy,xx=np.mgrid[:nc,:nc];x=(xx+.5)/nc;y=(yy+.5)/nc
    fields=np.stack([f(x,y) for f in functions]).reshape(4,-1)
    fields=fields/np.maximum(np.max(np.abs(fields),axis=1,keepdims=True),1e-300)
    source=fields[0];goals=fields[1:].mean(axis=0)
    scores=[base,base,base*(.25+source),base*(.25+goals),
            base*(.25+.5*(source+goals)),np.zeros(nc*nc)]
    basis=np.stack((np.sin(2*np.pi*x),np.cos(2*np.pi*y),
                    np.sin(2*np.pi*(x+y)),np.cos(4*np.pi*(x-y))))
    for seed in range(4):
        rng=np.random.default_rng(55600+17*int(geometry)+seed)
        field=np.einsum("i,ijk->jk",rng.normal(size=4),basis)
        field=np.clip((field-field.mean())/max(field.std(),1e-300),-3,3).ravel()
        for beta in (.5,1.5):scores.append(base*np.exp(beta*field))
    bank=[]
    for index,(name,score) in enumerate(zip(BANK_NAMES,scores)):
        if index==0:cells,marking=baseline.copy(),baseline_marking.copy()
        else:cells,marking=seeded_mark(score,nc,budget,mandatory,forbidden)
        selected=np.zeros(nc*nc,dtype=bool);selected[cells]=True
        bank.append(dict(index=index,name=name,mode=mode,selected_cells=cells.tolist(),
            marking=marking,scores=score.tolist(),interface_covered=bool(np.all(selected[mask])),
            covered_interface_cells=int(selected[mask].sum()),interface_cells=int(mask.sum()),
            ring_state_matched=bool(np.array_equal(selected[ring],baseline_mask[ring]))))
    if len(bank)!=len(BANK_NAMES):raise RuntimeError("Incomplete bank")
    return bank,mask


def error_metrics(q, fine, coarse):
    """Separate normalization for known and audit triples; never sum signs."""
    q=np.asarray(q);fine=np.asarray(fine);coarse=np.asarray(coarse)
    scales=np.maximum(abs(coarse),max(float(abs(coarse).max())*1e-12,1e-30))
    signed=(fine-q)/scales;coarse_error=(fine-coarse)/scales
    denominator=float(np.linalg.norm(coarse_error))
    norm=float(np.linalg.norm(signed))
    return dict(qoi=q.tolist(),fine_qoi=fine.tolist(),coarse_qoi=coarse.tolist(),
        scales=scales.tolist(),signed_scaled_error=signed.tolist(),
        coarse_scaled_norm=denominator,scaled_norm=norm,
        maximum_scaled_error=float(np.max(abs(signed))),
        E=None if denominator<1e-12 else norm/denominator)
