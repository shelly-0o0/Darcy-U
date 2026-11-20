"""Post-hoc paired engineering probe; old formal data are never edited."""
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
import argparse
import json
import numpy as np
from stage15b_experiment import CONFIG,sha,dump,assess_output
from stage15b_cases import cases
from stage15_fem import make_mesh
import stage15b_adaptive as original
import stage15b_deferred as deferred

ROOT=Path(__file__).resolve().parent
SOURCES=['stage15b_deferred.py','stage15b_deferred_probe.py','test_stage15b_deferred.py',
         'stage15b_adaptive.py','stage15b_experiment.py','stage15b_cases.py','stage15b_reference.py',
         'docs/stage15b_deferred_probe_zh.md','stage15_fem.py','stage14_solver.py']


def trajectory_signature(record):
    return [(r['status'],r['mesh_coordinates_sha256'],r['mesh_connectivity_sha256'],
             r.get('solution_sha256'),r.get('selected_elements',[])) for r in record['rows']]


def execute(case,tau,variant,config):
    start=perf_counter(); mesh=make_mesh(config['initial_n']); creation=perf_counter()-start
    kernel=original if variant=='original' else deferred
    result=kernel.run_adaptive(case.coefficient,case.source,case.goal,initial_mesh=mesh,tau=tau,marker='dwr',
                              config=dict(config['adaptive'],solver='direct'))
    summary=result.summary(); summary['initial_mesh_seconds']=creation
    summary['kernel_algorithm_seconds']=summary['algorithm_seconds']
    summary['algorithm_seconds']+=creation; summary['total_seconds']+=creation
    return dict(case=case.name,tau=tau,variant=variant,summary=summary,rows=result.rows)


def summarize(records,references):
    refs={r['case']:r for r in references}
    pairs=[]
    for case in cases():
        for tau in CONFIG['taus']:
            for repeat in range(3):
                selected={r['variant']:r for r in records if (r['case'],r['tau'],r['repeat'])==(case.name,tau,repeat)}
                if set(selected)!= {'original','deferred'}: raise ValueError('Incomplete paired matrix')
                a,b=selected['original'],selected['deferred']
                same=trajectory_signature(a)==trajectory_signature(b)
                aq=[r['qoi'] for r in a['rows']]; bq=[r['qoi'] for r in b['rows']]
                difference=float(max(abs(np.asarray(aq)-np.asarray(bq)))) if len(aq)==len(bq) else None
                values_same=difference is not None and difference<=1e-12*max(1.,max(abs(np.asarray(aq))))
                offline=[assess_output(r['summary'],refs[case.name],tau) for r in (a,b)]
                pairs.append(dict(case=case.name,tau=tau,repeat=repeat,
                    equivalent=bool(same and values_same),maximum_qoi_difference=difference,
                    both_success=all(r['success'] for r in offline),offline=offline,
                    old_seconds=a['summary']['algorithm_seconds'],new_seconds=b['summary']['algorithm_seconds'],
                    new_over_old=b['summary']['algorithm_seconds']/a['summary']['algorithm_seconds']))
    groups=[]
    for case in cases():
        for tau in CONFIG['taus']:
            p=[r for r in pairs if (r['case'],r['tau'])==(case.name,tau)]
            groups.append(dict(case=case.name,tau=tau,
                old_median_seconds=float(np.median([r['old_seconds'] for r in p])),
                new_median_seconds=float(np.median([r['new_seconds'] for r in p])),
                median_paired_ratio=float(np.median([r['new_over_old'] for r in p])),
                min_paired_ratio=min(r['new_over_old'] for r in p),max_paired_ratio=max(r['new_over_old'] for r in p)))
    return dict(post_hoc_engineering_only=True,paths=len(records),paired_comparisons=len(pairs),
                equivalent_pairs=sum(r['equivalent'] for r in pairs),
                successful_pairs=sum(r['both_success'] for r in pairs),pairs=pairs,groups=groups,
                median_of_group_paired_ratios=float(np.median([r['median_paired_ratio'] for r in groups])),
                no_ai=True,no_independent_confirmation=True)


def report(summary):
    lines=['# 15B事后工程优化：停止层延后局部化','',
       f"{summary['paths']}条新计时路径，{summary['equivalent_pairs']}/{summary['paired_comparisons']}配对轨迹一致，{summary['successful_pairs']}/{summary['paired_comparisons']}配对两版均通过原参考精度验收。",'',
       '原288条正式路径保持不变。本表原/新版均在同一新批次测量，逐组合三次配对；不是跨批次拼接。', '',
       '| 材料 | τ | 原版中位ms | 延后版中位ms | 新/旧配对比中位 | 配对比范围 |',
       '|---|---:|---:|---:|---:|---|']
    for g in summary['groups']:
        lines.append(f"| {g['case']} | {g['tau']:g} | {g['old_median_seconds']*1000:.3f} | {g['new_median_seconds']*1000:.3f} | {g['median_paired_ratio']:.4f} | {g['min_paired_ratio']:.4f}–{g['max_paired_ratio']:.4f} |")
    lines+=['',f"12组合配对比中位数的中位：{summary['median_of_group_paired_ratios']:.4f}。",'',
             '只改变是否在已经决定停止的层计算局部指标；仍支付P2伴随和全局核验。收益归于经典程序优化，没有训练或推理。相同固定案例的事后探针不构成新几何确认，也不能据小样本毫秒计时宣称总体显著性。','']
    return '\n'.join(lines)


def run(output):
    output=Path(output)
    if output.exists(): raise FileExistsError('Choose a new output directory')
    raw=ROOT/'results/stage15b_classical'
    completion=json.loads((raw/'complete.json').read_text())
    if sha(raw/'references.json')!=completion['file_sha256']['references.json']:
        raise ValueError('Original reference hash mismatch')
    references=json.loads((raw/'references.json').read_text())
    source={name:sha(ROOT/name) for name in SOURCES}
    output.mkdir(parents=True)
    dump(output/'freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source,
         original_complete_sha256=sha(raw/'complete.json'),config=CONFIG,repeats=3,
         purpose='Post-hoc deferred-localization engineering comparison; not AI'))
    records=[]; start=perf_counter(); materials=cases(); warmups=[]
    for variant in ('original','deferred'):
        config=dict(CONFIG,initial_n=4,adaptive=dict(CONFIG['adaptive'],max_steps=2))
        warmups.append(execute(materials[0],1e-6,variant,config))
    dump(output/'warmups.json.gz',warmups)
    for ci,case in enumerate(materials):
        for ti,tau in enumerate(CONFIG['taus']):
            for repeat in range(3):
                order=('original','deferred') if (ci+ti+repeat)%2==0 else ('deferred','original')
                for variant in order:
                    r=execute(case,tau,variant,CONFIG); r['repeat']=repeat
                    records.append(r)
    dump(output/'rows.json.gz',records)
    summary=summarize(records,references)
    dump(output/'summary.json',summary)
    (output/'REPORT_zh.md').write_text(report(summary))
    for name,digest in source.items():
        if sha(ROOT/name)!=digest: raise RuntimeError('Probe source changed: '+name)
    dump(output/'complete.json',dict(complete=True,wall_seconds=perf_counter()-start,
         file_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(json.dumps({k:summary[k] for k in ['paths','equivalent_pairs','successful_pairs','median_of_group_paired_ratios']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/stage15b_deferred_probe')
    args=parser.parse_args();run(ROOT/args.output)
