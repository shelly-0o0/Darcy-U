#!/usr/bin/env python3
"""Summarize the final retrospective-space budget screen, without solving PDEs."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
METHODS = ['baseline_scipy', 'baseline_event_custom', 'frozen_all_balanced4',
           'current_reference4', 'current_reference8']
COMPONENTS = ['assembly_seconds', 'guess_seconds', 'setup_seconds',
              'correction_seconds', 'solve_seconds', 'maintenance_seconds',
              'recovery_seconds']


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def distribution(values):
    values = list(values)
    return dict(median=median(values), min=min(values), max=max(values), count=len(values))


def row_valid(row):
    finite=lambda v:isinstance(v,(int,float)) and math.isfinite(v)
    if not (row['converged'] and row['final_converged'] and not row['solver_fallback'] and
            not row['cache_rejected'] and finite(row['final_relative_residual']) and row['final_relative_residual']<=1e-8):
        return False
    if row['method'].startswith('current_reference'):
        if not row['reference_space_valid'] or row['rank']!=row['requested_rank']:return False
    elif row['method']=='frozen_all_balanced4':
        if row['stored_space_rank']!=4 or (row['step']>0 and row['rank']!=4):return False
    return finite(row['orthogonality_error']) and row['orthogonality_error']<=1e-7


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['method'], row['seed'], row['repeat'])].append(row)
    sequences = []
    for (method, seed, repeat), group in sorted(grouped.items()):
        assert sorted(r['step'] for r in group) == list(range(20))
        values = {key: sum(r[key] for r in group)
                  for key in ['total_seconds', 'iterations', *COMPONENTS]}
        sequences.append(dict(method=method, seed=seed, repeat=repeat, **values))
    methods = {}
    for method in METHODS:
        selected = [s for s in sequences if s['method'] == method]
        by_geometry = {}
        for seed in sorted({s['seed'] for s in selected}):
            local = [s for s in selected if s['seed'] == seed]
            by_geometry[str(seed)] = {key: median(s[key] for s in local)
                for key in ['total_seconds', 'iterations', *COMPONENTS]}
        methods[method] = dict(by_geometry=by_geometry,
            **{key: distribution(g[key] for g in by_geometry.values())
               for key in ['total_seconds', 'iterations', *COMPONENTS]})
    comparisons = {}
    for method in METHODS[1:]:
        pairs = []
        for seed, values in methods[method]['by_geometry'].items():
            baseline = methods['baseline_scipy']['by_geometry'][seed]
            budget = .8 * baseline['total_seconds'] - values['total_seconds']
            pairs.append(dict(seed=int(seed),
                time_ratio=values['total_seconds']/baseline['total_seconds'],
                iterations_ratio=values['iterations']/baseline['iterations'],
                time_difference=values['total_seconds']-baseline['total_seconds'],
                construction_budget_seconds=budget,
                average_budget_seconds_per_system=budget/20))
        comparisons[method] = dict(by_geometry=pairs,
            **{key: distribution(p[key] for p in pairs) for key in
               ['time_ratio', 'iterations_ratio', 'time_difference',
                'construction_budget_seconds', 'average_budget_seconds_per_system']},
            all_geometries_at_most_point8=all(p['time_ratio'] <= .8 for p in pairs),
            faster_geometries=sum(p['time_ratio'] < 1 for p in pairs))
    return dict(methods=methods, versus_prespecified_scipy=comparisons,
                sequences=sequences, rows=len(rows),
                causal_control_rows=sum(not r['method'].startswith('current_reference') for r in rows),
                retrospective_reference_rows=sum(r['method'].startswith('current_reference') for r in rows),
                max_true_residual=max(r['final_relative_residual'] for r in rows),
                pcg_failures=sum(not r['converged'] for r in rows),
                recoveries=sum(r['solver_fallback'] for r in rows),
                final_failures=sum(not r['final_converged'] for r in rows))


def quality_summary(metadata):
    systems=[r for s in metadata['sequences'] for r in s['systems']]
    result=dict(systems=len(systems), invalid_systems=sum(not r['numerically_valid'] for r in systems),
        source_pcg_failures=sum(not r['source_pcg']['converged'] for r in systems),
        direct_failures=sum(r['direct_relative_residual'] is None or r['direct_relative_residual']>1e-8 for r in systems),
        acquisition_seconds=sum(r['acquisition_seconds'] for r in systems),
        direct_seconds=sum(r['direct_seconds'] for r in systems),
        quality_seconds=sum(r['quality_seconds'] for r in systems),
        storage_seconds=sum(s['storage_seconds'] for s in metadata['sequences']),
        captured_vectors=distribution(r['captured_vectors'] for r in systems),
        cache_checks=sum(r['cache_audit'] is not None for r in systems),
        cache_failures=sum(r['cache_audit'] is not None and not r['cache_audit']['valid'] for r in systems),
        ranks={})
    for q in ['4','8']:
        by_geometry=[]
        for sequence in metadata['sequences']:
            local=sequence['systems']
            energy=[r['quality'][q]['projected_energy_remaining_ratio'] for r in local]
            frozen=[r['frozen4_projection']['projected_energy_remaining_ratio'] for r in local]
            residuals=[r['quality'][q]['max_a_relative_ritz_residual'] for r in local]
            valid=lambda values:[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
            by_geometry.append(dict(seed=sequence['seed'],
                actual_rank=distribution(r['quality'][q]['rank'] for r in local),
                invalid_systems=sum(not r['quality'][q]['valid'] for r in local),
                energy_remaining=distribution(valid(energy)) if valid(energy) else None,
                frozen_energy_remaining=distribution(valid(frozen)) if valid(frozen) else None,
                max_column_a_relative_ritz_residual=distribution(valid(residuals)) if valid(residuals) else None))
        result['ranks'][q]=dict(by_geometry=by_geometry,
            invalid_systems=sum(v['invalid_systems'] for v in by_geometry),
            energy_remaining_geometry_median=median(v['energy_remaining']['median'] for v in by_geometry if v['energy_remaining']) if any(v['energy_remaining'] for v in by_geometry) else None,
            frozen_energy_remaining_geometry_median=median(v['frozen_energy_remaining']['median'] for v in by_geometry if v['frozen_energy_remaining']) if any(v['frozen_energy_remaining'] for v in by_geometry) else None)
    return result


def validate_stage(folder, frozen, splits, freeze_hash, decision_hash):
    manifest=read(folder/'manifest.json');config=manifest['configuration']
    n=config['n'];role=config['role']
    seeds=splits['groups']['development' if role=='development' else 'confirmation']
    if n==256:seeds=seeds[:2]
    assert config['seeds']==seeds and config['repeats']==3
    assert config['specs']==frozen['specs']
    assert config['freeze_sha256']==freeze_hash
    assert config['decision_sha256']==(None if role=='development' else decision_hash)
    for key in ['source_sha256','protocol_sha256','splits_sha256']:
        assert manifest[key]==frozen[key]
    assert manifest['reference_metadata_sha256']==sha(folder/'reference_metadata.json')
    preparation=ROOT/'results'/f'stage8_{role}_preparation.json'
    assert manifest['phase_preparation_sha256']==sha(preparation)
    assert read(preparation)['all_grids_prepared_and_loaded_before_measurement'] is True
    rows=read(folder/'measurements.json')
    expected={(m,seed,r,t) for m in METHODS for seed in seeds for r in range(3) for t in range(20)}
    actual=[(r['method'],r['seed'],r['repeat'],r['step']) for r in rows]
    assert len(actual)==len(set(actual))==len(expected) and set(actual)==expected
    complete=read(folder/'complete.json')
    assert complete['complete'] and complete['rows']==len(rows)
    for row in rows:
        assert row['n']==n
        assert all(isinstance(row[k],(int,float)) and math.isfinite(row[k]) and row[k]>=0 for k in ['total_seconds',*COMPONENTS])
        assert sum(row[k] for k in COMPONENTS)<=row['total_seconds']+1e-8
        assert row['extraction_seconds']<=row['maintenance_seconds']+1e-8
        assert row['final_converged']==(row['final_relative_residual'] is not None and row['final_relative_residual']<=1e-8)
        assert row['solver_fallback']==(not row['converged'])
        assert 0<=row['iterations']<=500 and 0<=row['restarts']<=3
        if row['method'].startswith('current_reference'):
            assert not row['capture_requested'] and not row['extraction_requested']
            assert not row['extraction'] and row['extraction_seconds']==0
            assert row['retrospective_reference'] is True
            assert row['solver_operator']==('balanced' if row['reference_space_valid'] else 'base')
            assert row['cache_rejected']==(not row['reference_space_valid'])
            if row['capture_stats']:assert row['capture_stats']['max_vectors']==0
    metadata=read(folder/'reference_metadata.json')
    assert [s['seed'] for s in metadata['sequences']]==seeds
    for sequence in metadata['sequences']:
        assert [r['step'] for r in sequence['systems']]==list(range(20))
        assert [r['step'] for r in sequence['systems'] if r['cache_audit'] is not None]==[0,10]
        for row in rows:
            if row['seed']==sequence['seed'] and row['method'].startswith('current_reference'):
                assert row['reference_space_sha256']==sequence['systems'][row['step']]['space_sha256'][str(row['requested_rank'])]
    result=summarize(rows)
    result['numerically_valid']=bool(all(row_valid(r) for r in rows)
        and all(r['numerically_valid'] for s in metadata['sequences'] for r in s['systems']))
    assert result['numerically_valid']==read(folder/'summary.json')['numerically_valid']
    result['quality']=quality_summary(metadata)
    result['input_sha256']={str(p.relative_to(ROOT)):sha(p) for p in sorted(folder.glob('*.json'))}
    result['reference_metadata']=metadata
    return result


def plots(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    groups = list(summary['stages'].items())
    labels = ['SciPy AMG', 'Custom events', 'Frozen rank 4',
              'Current ref. 4', 'Current ref. 8']
    fig, axes = plt.subplots(2, len(groups), figsize=(5.3*len(groups), 7),
                             squeeze=False, constrained_layout=True)
    for column, (name, stage) in enumerate(groups):
        methods = stage['methods']
        for row, key, ylabel in [(0, 'total_seconds', '20-step path time (s)'),
                                 (1, 'iterations', '20-step PCG iterations')]:
            ax = axes[row, column]
            values = np.array([methods[m][key]['median'] for m in METHODS])
            low = np.array([methods[m][key]['min'] for m in METHODS])
            high = np.array([methods[m][key]['max'] for m in METHODS])
            ax.bar(range(5), values, color=['#929aa0', '#b7bec3', '#d1a05c', '#3c819b', '#417a6a'])
            ax.errorbar(range(5), values, yerr=[values-low, high-values],
                        fmt='none', color='#333333', capsize=3)
            ax.set_xticks(range(5), labels, rotation=25, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(name.replace('stage8_', ''))
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.15)
    fig.suptitle('Retrospective directions: path cost EXCLUDES direction acquisition\nGeometry medians and ranges; references are not online algorithms', fontsize=12)
    folder=out/'figures';folder.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(folder/f'budget_screen.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def markdown(summary):
    lines=['# 第八阶段：当前轨迹参考空间的最后预算筛查', '',
        f"生成时间：{summary['created_utc']}。", '',
        '**当前参考方向来自已经完成的当前系统轨迹，不是因果在线算法，也不是AI模型或最优方向上界。** 参考路径时间没有包含方向获取；实际组装、初值、AMG、正交化、投影、B构造/应用、PCG、真残差和维护仍计入。', '',
        '开发规则在运行前固定：同一个q必须在两个网格全部开发几何中均达到SciPy完整时间的0.8倍以内，才允许一次独立确认；若两个q都通过预定选择q4，不在确认后改选。', '',
        '完整20步求和→同几何重复中位→几何配对和汇总。误差条为几何范围，不是置信区间。构造预算=0.8×SciPy完整时间−参考部署路径时间，负值不支持本参考的训练投入。', '',
        '参考方向与离线真解先统一加载到内存，再开始计时，采用数据已驻留口径；不包含磁盘读取与方向获取费用。所有方法的部署路径仍独立组装矩阵并构建自己的AMG和历史。']
    for phase, preparation in summary['phase_preparation'].items():
        lines.append(f"{phase}整批加载费用为{preparation['loading_seconds']:.3f}秒（每阶段一次，未重复计入两个网格）。")
    lines+=['']
    for name, stage in summary['stages'].items():
        lines += [f'## {name}', '',
            f"共{stage['rows']}条路径记录，其中因果对照{stage['causal_control_rows']}条、事后参考{stage['retrospective_reference_rows']}条。PCG失败{stage['pcg_failures']}、恢复{stage['recoveries']}、最终精度失败{stage['final_failures']}；最大真残差{stage['max_true_residual']:.6e}。", '',
            '| 方法 | 完整路径秒数中位数 | 迭代中位数 | 对SciPy配对时间比 |',
            '|---|---:|---:|---:|']
        for method, value in stage['methods'].items():
            ratio=1. if method=='baseline_scipy' else stage['versus_prespecified_scipy'][method]['time_ratio']['median']
            lines.append(f"| {method} | {value['total_seconds']['median']:.6f} | {value['iterations']['median']:.1f} | {ratio:.6f} |")
        lines += ['', '| 参考方法/几何 | 配对时间比 | 20步构造预算/秒 | 平均每系统预算/毫秒 |', '|---|---:|---:|---:|']
        for method in METHODS[3:]:
            for pair in stage['versus_prespecified_scipy'][method]['by_geometry']:
                lines.append(f"| {method}/{pair['seed']} | {pair['time_ratio']:.6f} | {pair['construction_budget_seconds']:.6f} | {1000*pair['average_budget_seconds_per_system']:.3f} |")
        quality=stage['quality']
        lines += ['', f"本组离线准备另含{quality['systems']}次基础M参考求解与{quality['systems']}次直接参考。共同方向获取总费{quality['acquisition_seconds']:.3f}秒、直接解{quality['direct_seconds']:.3f}秒、质量核验{quality['quality_seconds']:.3f}秒、存储{quality['storage_seconds']:.3f}秒；共同获取费包含两种q与首步冻结参考的提取，不是某一种q的独立算法费用。",
            f"参考结构无效系统{quality['invalid_systems']}个，基础PCG失败{quality['source_pcg_failures']}个；预定缓存核验{quality['cache_checks']}次，失败{quality['cache_failures']}次。", '',
            '| q | 当前参考投影后剩余误差能量 | 首步冻结4维剩余误差能量 | 当前参考结构无效系统 |',
            '|---|---:|---:|---:|']
        for q,value in quality['ranks'].items():
            lines.append(f"| {q} | {value['energy_remaining_geometry_median']} | {value['frozen_energy_remaining_geometry_median']} | {value['invalid_systems']} |")
        lines += ['', '误差能量比在同一基础历史初值下测量，先取每几何20系统中位，再取几何中位；不是网络预测误差。独立实际A/M作用下的特征残差、实际秩与逐系统质量记录完整保留在JSON；结构检查通过不代表精确最慢空间。']
        lines += ['']
    decision_fields={k:summary['decision'][k] for k in ['numerically_valid','selected_rank','proceed_to_confirmation','training_authorized']}
    lines += ['## 预定决策及证据边界', '',
        '运行器决定的关键字段如下，完整记录见[原始决定](../stage8_decision.json)；报告不会根据汇总中位数替代“所有几何过线”的预定门槛。', '',
        '```json', json.dumps(decision_fields, ensure_ascii=False, indent=2), '```', '',
        f"最终状态：`{summary['completion']['status']}`；是否运行确认：`{summary['completion']['confirmation_run']}`。",
        (f"开发预选q={summary['completion']['selected_rank']}；同一q确认是否过线：{summary['completion'].get('selected_rank_confirmed')}。" if summary['completion']['confirmation_run'] else '按开发停止规则结束；没有运行确认或学习训练。'), '',
        '参考质量可能有限，尤其当前右端激发的有限轨迹不等于全系统最慢空间。本次未过门槛可支持资源止损，不能证明所有学习空间都没有加速可能。即便参考过线，仍需用实际可执行构造费用和独立确认检验收益。', '',
        '源码、划分、参考生成诊断、费用及原始文件哈希见[summary.json](summary.json)。研究判断见[第八阶段结论](../../docs/stage8_findings_zh.md)，历史复盘见[路线复盘](../../docs/route_retrospective_zh.md)。', '',
        '![部署路径与迭代](figures/budget_screen.png)', '']
    return '\n'.join(lines).rstrip()


def main():
    frozen=read(ROOT/'results/stage8_freeze.json')
    decision=read(ROOT/'results/stage8_decision.json')
    complete=read(ROOT/'results/stage8_complete.json')
    split_path=ROOT/'data/stage8_splits.json';splits=read(split_path)
    for name,digest in frozen['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'docs/stage8_protocol_zh.md')==frozen['protocol_sha256']
    assert sha(split_path)==frozen['splits_sha256']
    freeze_hash=sha(ROOT/'results/stage8_freeze.json')
    decision_hash=sha(ROOT/'results/stage8_decision.json')
    assert decision['freeze_sha256']==complete['freeze_sha256']==freeze_hash
    assert complete['decision_sha256']==decision_hash and complete['complete']
    for name,digest in decision['development_sha256'].items():assert sha(ROOT/name)==digest,name
    previous=read(ROOT/'data/stage7_splits.json')
    assert set(previous['excluded_seeds']+sum(previous['groups'].values(),[])+[75004])<=set(splits['excluded_seeds'])
    seeds=sum(splits['groups'].values(),[])
    assert len(seeds)==len(set(seeds))==8 and not set(seeds)&set(splits['excluded_seeds'])
    assert len(set(splits['geometry_sha256'].values()))==8
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        stages={},decision=decision,completion=complete,phase_preparation={},
        reporter_sha256=sha(Path(__file__)),freeze_sha256=freeze_hash,
        decision_sha256=decision_hash,
        scope='Retrospective reference paths exclude direction acquisition; no AI or causal speedup claim.')
    roles=['development']+(['confirm'] if complete['confirmation_run'] else [])
    for role in roles:
        summary['phase_preparation'][role]=read(ROOT/'results'/f'stage8_{role}_preparation.json')
        for n in [128,256]:
            name=f'stage8_{role}{n}'
            summary['stages'][name]=validate_stage(ROOT/'results'/name,frozen,splits,freeze_hash,decision_hash)
    valid=all(summary['stages'][f'stage8_development{n}']['numerically_valid'] for n in [128,256])
    assert valid==decision['numerically_valid']
    qualified=[]
    for q in [4,8]:
        recorded=decision['candidates'][str(q)]
        points=[]
        for n in [128,256]:
            comparison=summary['stages'][f'stage8_development{n}']['versus_prespecified_scipy'][f'current_reference{q}']
            for pair in comparison['by_geometry']:
                item=next(v for v in recorded['cases'] if v['n']==n and v['seed']==pair['seed'])
                for key in ['time_ratio','construction_budget_seconds']:
                    assert abs(item[key]-pair[key])<1e-12
                points.append(pair['time_ratio']<=.8)
        passed=valid and all(points)
        assert recorded['passes_all_geometries']==passed
        if passed:qualified.append(q)
    assert decision['selected_rank']==(qualified[0] if qualified else None)
    assert decision['proceed_to_confirmation']==bool(qualified)
    if not qualified:
        assert complete['status']=='stopped_after_development' and not complete['confirmation_run']
        assert not any((ROOT/'results'/f'stage8_confirm{n}').exists() for n in [128,256])
    else:
        for name,digest in complete['confirmation_sha256'].items():assert sha(ROOT/name)==digest
        assessment=complete['assessment']
        confirmed_valid=all(summary['stages'][f'stage8_confirm{n}']['numerically_valid'] for n in [128,256])
        assert assessment['numerically_valid']==confirmed_valid
        confirmed_qualified=[]
        for q in [4,8]:
            passed=confirmed_valid
            for n in [128,256]:
                pairs=summary['stages'][f'stage8_confirm{n}']['versus_prespecified_scipy'][f'current_reference{q}']['by_geometry']
                for pair in pairs:
                    record=next(v for v in assessment['candidates'][str(q)]['cases'] if v['n']==n and v['seed']==pair['seed'])
                    for key in ['time_ratio','construction_budget_seconds']:assert abs(record[key]-pair[key])<1e-12
                    passed=passed and pair['time_ratio']<=.8
            assert assessment['candidates'][str(q)]['passes_all_geometries']==passed
            if passed:confirmed_qualified.append(q)
        assert complete['selected_rank']==decision['selected_rank']
        assert complete['selected_rank_confirmed']==(decision['selected_rank'] in confirmed_qualified)
    summary['totals']={key:sum(s[key] for s in summary['stages'].values()) for key in ['rows','causal_control_rows','retrospective_reference_rows','pcg_failures','recoveries','final_failures']}
    summary['totals']['offline_reference_solves']=sum(s['quality']['systems'] for s in summary['stages'].values())
    summary['totals']['direct_reference_solves']=summary['totals']['offline_reference_solves']
    verification=ROOT/'results/stage8_verification.json'
    if verification.exists():summary['verification']=dict(record=read(verification),sha256=sha(verification))
    out=ROOT/'results/stage8_report';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'REPORT_zh.md').write_text(markdown(summary)+'\n')
    plots(summary,out)
    print(json.dumps(dict(status=complete['status'],**summary['totals']),ensure_ascii=False))


if __name__=='__main__':main()
