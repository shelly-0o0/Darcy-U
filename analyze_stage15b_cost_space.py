"""Post-hoc cost ledger, not a simulated learned solver or speed claim."""
from pathlib import Path
import argparse,gzip,json
import numpy as np
from stage15b_experiment import sha,dump

ROOT=Path(__file__).resolve().parent
KEYS=('p2_assembly_seconds','p2_dual_solve_seconds','dwr_evaluation_seconds','dwr_localization_seconds')


def run(directory,output):
    directory,output=Path(directory),Path(output)
    if output.exists(): raise FileExistsError('Do not overwrite analysis')
    done=json.loads((directory/'complete.json').read_text())
    for name,digest in done['file_sha256'].items():
        if sha(directory/name)!=digest: raise ValueError('Changed probe artifact: '+name)
    with gzip.open(directory/'rows.json.gz','rt') as stream: rows=json.load(stream)
    groups=[]
    names=sorted({(r['case'],r['tau']) for r in rows})
    for name,tau in names:
        values=[]
        for r in rows:
            if (r['case'],r['tau'],r['variant'])!=(name,tau,'deferred'): continue
            budget=sum(sum(row['timings'].get(k,0.) for k in KEYS) for row in r['rows'] if row['status']=='continue')
            time=r['summary']['algorithm_seconds']
            values.append(dict(repeat=r['repeat'],online_seconds=time,
                               nonterminal_estimation_seconds=budget,fraction=budget/time,
                               hypothetical_remaining_seconds=time-budget,
                               assumed_same_trajectory_and_final_verification=True))
        groups.append(dict(case=name,tau=tau,measurements=values,
                           median_nonterminal_fraction=float(np.median([v['fraction'] for v in values])),
                           median_nonterminal_ms=float(np.median([1000*v['nonterminal_estimation_seconds'] for v in values]))))
    result=dict(post_hoc=True,probe_complete_sha256=sha(directory/'complete.json'),
                analysis_source_sha256=sha(__file__),groups=groups,
                groups_at_least_30_percent=sum(g['median_nonterminal_fraction']>=.3 for g in groups),
                interpretation='Fixed-trajectory arithmetic only. Final verification remains paid. No model, inference, fallback or changed trajectory was measured; no AI speed claim or G1 approval.')
    dump(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='results/stage15b_deferred_probe')
    parser.add_argument('--output',default='results/stage15b_cost_space.json')
    args=parser.parse_args();run(ROOT/args.input,ROOT/args.output)
