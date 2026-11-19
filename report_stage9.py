#!/usr/bin/env python3
"""Read-only aggregation of stage 9; never reruns a nonlinear solve."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'results/stage9_screen'
OUT=ROOT/'results/stage9_report'
METHODS=['ew1','ew2','fixed_0.01','fixed_0.1','fixed_0.5','fixed_0.9','direct']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(value,digits=6):return '不可用' if value is None else f'{value:.{digits}f}'


def trajectory_audit(branches,decision):
    fields=['residual_before','residual_after','accepted_alpha','actual_eta','linear_iterations']
    signatures=defaultdict(set)
    for row in branches:
        signature=json.dumps(dict(final_u=row['solution_sha256'],
            trace=[[step[k] for k in fields] for step in row['trace']]),sort_keys=True)
        signatures[(row['case'],row['iteration'],row['multiplier'])].add(signature)
    assert all(len(v)==1 for v in signatures.values()),'Repeated numerical trajectories differ'
    by_state=defaultdict(set)
    for (case,k,m),values in signatures.items():by_state[(case,k)].update(values)
    best_same=best_non_native_same=0
    for point in decision['states']:
        best=point['best']
        if best is None:continue
        prefix=(point['case'],point['iteration'])
        if signatures[(*prefix,best['multiplier'])]==signatures[(*prefix,1.)]:
            best_same+=1
            best_non_native_same+=best['multiplier']!=1.
    return dict(states=len(by_state),actions=len(signatures),
        distinct_logged_trajectories_summed_within_state=sum(len(v) for v in by_state.values()),
        states_all_actions_identical=sum(len(v)==1 for v in by_state.values()),
        best_action_identical_to_native=best_same,non_native_winner_identical_to_native=best_non_native_same,
        signature_fields=fields,includes_final_solution_sha256=True,
        scope='Equality of recorded numerical scalars and final solution bytes, not proof of equality of all intermediate vectors')


def main():
    freeze=read(SOURCE/'freeze.json');complete=read(SOURCE/'complete.json')
    assert complete['complete']
    for name,digest in freeze['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'docs/stage9_protocol_zh.md')==freeze['protocol_sha256']
    assert sha(ROOT/'data/stage9_cases.json')==freeze['cases_sha256']
    if complete['status'] in ['stopped_numerical_verification','stopped_invalid_baselines']:
        OUT.mkdir(parents=True,exist_ok=True)
        summary=dict(completion=complete,freeze=freeze,reporter_sha256=sha(__file__),
            input_sha256={p.name:sha(p) for p in SOURCE.glob('*.json')})
        for name in ['verification','baseline_selection']:
            if (SOURCE/f'{name}.json').exists():summary[name]=read(SOURCE/f'{name}.json')
        (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        (OUT/'REPORT_zh.md').write_text('# 第九阶段：提前停止\n\n状态：`'+complete['status']+'`。\n\n数值验证或经典基线未满足预定条件，未完成后续分支诊断，未训练模型。失败记录与输入哈希见[summary.json](summary.json)。\n')
        print(json.dumps(dict(status=complete['status'])))
        return
    decision=read(SOURCE/'decision.json')
    assert sha(SOURCE/'freeze.json')==decision['freeze_sha256']
    assert sha(SOURCE/'decision.json')==complete['decision_sha256']
    for name,digest in decision['input_sha256'].items():assert sha(SOURCE/name)==digest,name
    baseline=read(SOURCE/'baselines.json');branches=read(SOURCE/'branches.json')
    selection=read(SOURCE/'baseline_selection.json');prep=read(SOURCE/'snapshot_preparation.json')
    verification=read(SOURCE/'verification.json')
    assert verification['valid']
    assert len(baseline)==504==complete['baseline_rows']
    assert len(branches)==complete['branch_rows']
    expected={(n,c,m,r) for n in [32,64] for c in range(12) for m in METHODS for r in range(3)}
    actual=[(r['n'],r['case'],r['method'],r['repeat']) for r in baseline]
    assert len(set(actual))==len(actual) and set(actual)==expected
    expected_branches={(s['case'],p['iteration'],a['multiplier'],r)
        for s in prep['sequences'] for p in s['snapshots'] for a in p['actions'] for r in range(3)}
    actual_branches=[(r['case'],r['iteration'],r['multiplier'],r['repeat']) for r in branches]
    assert len(set(actual_branches))==len(actual_branches) and set(actual_branches)==expected_branches
    for row in baseline+branches:
        assert row['total_seconds']>=row['solve_seconds']>=sum(row['components'].values())-1e-8
        assert row['total_seconds']>=row['assembly_seconds']+row['solve_seconds']-1e-8
        assert row['snapshot_seconds_excluded']==0
        assert row['accurate']==bool(row['converged'] and row['absolute_residual'] is not None
            and row['absolute_residual']<=row['tolerance'] and row['relative_solution_error'] is not None
            and row['relative_solution_error']<=1e-6)
        if 'expected_eta' in row and row['trace']:
            assert abs(row['trace'][0]['eta']-row['expected_eta'])<1e-14
    snapshots={(s['case'],p['iteration']):p for s in prep['sequences'] for p in s['snapshots']}
    for p in snapshots.values():
        if (ROOT/p['archive']).exists():assert sha(ROOT/p['archive'])==p['archive_sha256']
    for row in branches:
        assert row['source_state_sha256']==snapshots[(row['case'],row['iteration'])]['state_u_sha256']
    groups=defaultdict(list)
    for row in baseline:groups[(row['n'],row['method'],row['case'])].append(row)
    data={}
    points={}
    for n in [32,64]:
        data[str(n)]={}
        for m in METHODS:
            vals=[]
            for c in range(12):
                group=groups[(n,m,c)]
                p=dict(case=c,seconds=median(r['total_seconds'] for r in group),
                    newton=median(r['newton_iterations'] for r in group),linear=median(r['linear_iterations'] for r in group),
                    valid=all(r['accurate'] for r in group),
                    linear_fraction=median(r['components']['linear']/r['total_seconds'] for r in group),
                    preconditioner_fraction=median(r['components']['preconditioner']/r['total_seconds'] for r in group))
                points[(n,m,c)]=p;vals.append(p)
            data[str(n)][m]=dict(points=vals,seconds=median(p['seconds'] for p in vals),
                newton=median(p['newton'] for p in vals),linear=median(p['linear'] for p in vals),
                linear_fraction=median(p['linear_fraction'] for p in vals),
                preconditioner_fraction=median(p['preconditioner_fraction'] for p in vals),
                failed_rows=sum(not r['accurate'] for r in baseline if r['n']==n and r['method']==m))
    scores={m:median(points[(n,m,c)]['seconds']/points[(n,'ew2',c)]['seconds']
                    for n in [32,64] for c in range(12)) for m in METHODS}
    valid_methods=[m for m in METHODS if all(points[(n,m,c)]['valid'] for n in [32,64] for c in range(12))]
    assert selection['primary']==min(valid_methods,key=lambda m:scores[m])
    assert selection['continuation']==min([m for m in ['ew1','ew2'] if m in valid_methods],key=lambda m:scores[m])
    by_action=defaultdict(list)
    for r in branches:by_action[(r['case'],r['iteration'],r['multiplier'])].append(r)
    ratios=[]
    for c in range(12):
        action_times=[median(r['total_seconds'] for r in group) for (case,k,m),group in by_action.items()
            if case==c and k==0 and all(r['accurate'] for r in group)]
        ratio=min(action_times)/points[(64,selection['primary'],c)]['seconds'] if action_times else None
        recorded=next(p for p in decision['step0_pairs'] if p['case']==c)
        assert ratio==recorded['ratio']
        if ratio is not None:ratios.append(ratio)
    numerical=bool(len(ratios)==12 and all(r['accurate'] for r in branches) and all(s['source_accurate'] for s in prep['sequences']))
    assert numerical==decision['numerically_valid']
    passed=bool(numerical and median(ratios)<=.85 and sum(r<1 for r in ratios)>=9)
    assert passed==decision['opportunity_gate_passed']
    best_fixed=min([m for m in valid_methods if m.startswith('fixed_')],key=lambda m:scores[m])
    secondary=dict(best_fixed=best_fixed,
        ew_to_direct={str(n):median(points[(n,selection['continuation'],c)]['seconds']/points[(n,'direct',c)]['seconds'] for c in range(12)) for n in [32,64]},
        ew_to_fixed={str(n):median(points[(n,selection['continuation'],c)]['seconds']/points[(n,best_fixed,c)]['seconds'] for c in range(12)) for n in [32,64]},
        trajectory=trajectory_audit(branches,decision))
    totals=dict(baseline_rows=len(baseline),branch_rows=len(branches),
        baseline_failures=sum(not r['accurate'] for r in baseline),branch_failures=sum(not r['accurate'] for r in branches),
        max_relative_residual=max((r['relative_residual'] for r in baseline+branches if r['relative_residual'] is not None),default=None),
        max_solution_error=max((r['relative_solution_error'] for r in baseline+branches if r['relative_solution_error'] is not None),default=None))
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),reporter_sha256=sha(__file__),
        completion=complete,selection=selection,decision=decision,verification=verification,
        snapshot_preparation=prep,by_grid=data,totals=totals,secondary_audit=secondary,
        input_sha256={p.name:sha(p) for p in sorted(SOURCE.glob('*.json'))})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 第九阶段：非线性内层精度机会筛查','',
        '**本轮为经典基线与事后单动作分支诊断，没有AI训练、学习策略或独立确认。**', '',
        f"固定主基线：`{selection['primary']}`；固定分支续接：`{selection['continuation']}`。主基线在全部24个网格–实例上按预定规则选择，没有按每例挑选。",'',
        f"基线{len(baseline)}次、分支{len(branches)}次求解；精度失败分别{totals['baseline_failures']}/{totals['branch_failures']}。最大相对非线性真残差{totals['max_relative_residual']}，对高精度离散参考最大相对L2误差{totals['max_solution_error']}。",'',
        '另有8次制造解验证、24次高精度离散参考、12次来源轨迹、7次预热求解。32/64复用相同12个参数实例，不构成24个独立样本。', '',
        '## 离散验证', '', '| 方程 | 最后一对网格收敛阶 | 预定区间 |', '|---|---:|---|']
    for item in verification['manufactured']:
        lines.append(f"| {item['label']} | {item['orders'][-1]:.6f} | {item['required_last_order']} |")
    for n,methods in data.items():
        lines+=['',f'## {n}网格经典完整成本','',
            '| 方法 | 完整中位秒 | Newton步 | GMRES步 | 线性解成本占比 | ILU成本占比 | 失败次数 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for m,v in methods.items():
            lines.append(f"| {m} | {v['seconds']:.6f} | {v['newton']:.1f} | {v['linear']:.1f} | {v['linear_fraction']:.1%} | {v['preconditioner_fraction']:.1%} | {v['failed_rows']} |")
    lines+=['','先按同一实例三次完整计时取中位，再汇总实例。各成本占比也是先按重复、再按实例取中位，不把汇总中位数的和当作一个真实轨迹。直接Newton的线性解费用包含直接分解；GMRES步数为零不代表没有求解工作。', '',
        '## 步0有限动作机会与停止决定','',
        f"事后选出的最佳首动作/固定主基线，配对时间比中位数为 **{fmt(decision['step0_median_ratio'])}**，更快实例 **{decision['step0_faster_instances']}/12**；预定条件为不超过0.85且至少9/12更快。",'',
        '| 实例 | 最佳首倍率 | 完整时间比 | 主基线三次时间MAD/毫秒 | 分支三次时间MAD/毫秒 |',
        '|---|---:|---:|---:|---:|']
    for p in decision['step0_pairs']:
        a=p['best_action']
        lines.append(f"| {p['case']} | {a['multiplier'] if a else None} | {fmt(p['ratio'])} | {1000*p['baseline_mad']:.4f} | {fmt(1000*a['time_mad'] if a else None,4)} |")
    lines+=['',f"状态：`{complete['status']}`。机会门槛通过：`{passed}`；训练：`False`。",'',
        '这是每实例事后选择首动作的有限参考，排除了选择所需的多次试算费用；不能作为在线或AI加速。后续步骤只报告独立分支的剩余成本，不与其他状态拼接成新轨迹。','',
        '| 固定来源Newton步 | 状态数 | 最佳分支/原生EW剩余时间比中位数 |', '|---|---:|---:|']
    for k in [0,1,2,3]:
        ps=[p for p in decision['states'] if p['iteration']==k and p['remaining_ratio_to_continuation'] is not None]
        lines.append(f"| {k} | {len(ps)} | {fmt(median(p['remaining_ratio_to_continuation'] for p in ps) if ps else None)} |")
    lines+=['',f"来源轨迹、状态组织与存储共{prep['preparation_seconds']:.3f}秒，单列且不作为免费部署过程。全部快照先准备完，再进行分支计时。",'',
        f"数值轨迹审计：{secondary['trajectory']['actions']}个去重动作在各状态内共{secondary['trajectory']['distinct_logged_trajectories_summed_within_state']}种记录轨迹/最终解签名；{secondary['trajectory']['states_all_actions_identical']}/{secondary['trajectory']['states']}个状态所有动作签名相同。事后获选的非原生动作中，{secondary['trajectory']['non_native_winner_identical_to_native']}个与原生动作的记录数值轨迹和最终解字节一致，其微小计时优势不能归因为求解路径改善。并非所有状态都对动作不敏感。",'',
        'Botti2015公式与Oladokun2020全文仍未完整核实，经典对照覆盖与新颖性尚未成立。未过门槛可支持停止当前动作方案；过线也只允许补齐对照和近邻核查。未测其他预条件更新策略、更多动作或跨参数/网格独立确认，不能把此结果推广为所有非线性控制的结论。', '',
        '[研究结论](../../docs/stage9_findings_zh.md) · [运行前协议](../../docs/stage9_protocol_zh.md) · [原文核查](../../docs/stage9_related_work_zh.md) · [完整聚合](summary.json)', '',
        '![成本与有限动作](figures/screen.png)']
    (OUT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    plot(data,decision,selection)
    print(json.dumps(dict(status=complete['status'],**totals)))


def plot(data,decision,selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    labels=['EW1','EW2','Fixed .01','Fixed .1','Fixed .5','Fixed .9','Direct']
    for n,color in [('32','#417a6a'),('64','#3c819b')]:
        axes[0].plot(range(7),[data[n][m]['seconds'] for m in METHODS],'-o',label=f'n={n}',color=color)
    axes[0].set_xticks(range(7),labels,rotation=30,ha='right');axes[0].set_ylabel('Complete solve time (s)');axes[0].legend()
    pairs=decision['step0_pairs'];ratios=[p['ratio'] if p['ratio'] is not None else float('nan') for p in pairs]
    axes[1].bar([p['case'] for p in pairs],ratios,color='#3c819b')
    axes[1].axhline(1,color='#666',linestyle=':');axes[1].axhline(.85,color='#bb6450',linestyle='--',label='Prespecified .85 gate')
    axes[1].set_xlabel('Parameter instance');axes[1].set_ylabel(f'Best first action / fixed {selection["primary"]}');axes[1].legend()
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Classical nonlinear baselines and retrospective first-action screen\nNo learned policy; action selection cost excluded from reference paths')
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/f'screen.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
