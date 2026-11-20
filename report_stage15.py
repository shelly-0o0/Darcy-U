"""Recompute all stage-15A decisions from saved records and adaptive actions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_checked(directory, check_sources=True):
    directory = Path(directory)
    done = json.loads((directory/"complete.json").read_text())
    if not done.get("complete"):
        raise ValueError("Incomplete experiment")
    for name, expected in done["file_sha256"].items():
        if digest(directory/name) != expected:
            raise ValueError("Raw file digest mismatch: "+name)
    freeze = json.loads((directory/"freeze.json").read_text())
    if check_sources:
        for name, expected in freeze["source_sha256"].items():
            if digest(ROOT/name) != expected:
                raise ValueError("Frozen source mismatch: "+name)
        for name, expected in freeze["library_sha256"].items():
            if digest(ROOT/"vendor"/name) != expected:
                raise ValueError("Library source mismatch: "+name)
    return freeze, json.loads((directory/"rows.json").read_text()), json.loads((directory/"quadrature.json").read_text()), done


def summarize(freeze, rows, quadrature, done):
    # This module does not import the experiment or use its marking helper.
    import sequence_baselines
    from skfem import MeshTri

    checks=[]
    def check(name, value, limit=None):
        ok = bool(value) if limit is None else bool(np.isfinite(value) and value <= limit)
        checks.append(dict(name=name,passed=ok,value=value,limit=limit))
    config=freeze["config"]
    names=[c["name"] for c in freeze["cases"]]
    expected={(n,"uniform",k) for n in names for k in config["uniform_n"]} | {
        (n,"adaptive",k) for n in names for k in range(config["adaptive_refinements"]+1)}
    keys=[(r["case"],r["kind"],r["key"]) for r in rows]
    if set(keys)!=expected or len(keys)!=len(expected):
        raise ValueError("Unexpected, missing or duplicate context")
    if len(quadrature)!=len(names) or {r["case"] for r in quadrature}!=set(names):
        raise ValueError("Missing or duplicate quadrature checks")
    if done["rows"]!=len(rows) or done["quadrature_rows"]!=len(quadrature):
        raise ValueError("Completion counts mismatch")
    solver_records=[]
    identity_max=strong_max=nested_max=0.
    decisions=0
    for r in rows:
        prefix=f'{r["case"]}/{r["kind"]}/{r["key"]}'
        scale=r["scale_from_input"]
        check(prefix+"/mesh",r["mesh_check"]["accepted"])
        check(prefix+"/analytic_integral",r["analytic_goals_64_96_max_difference"]/scale,1e-12)
        d=r["dwr"]
        for field in ("identity_errors","element_sum_errors","algebraic_decomposition_errors"):
            value=float(np.max(abs(np.array(d[field]))))/scale
            identity_max=max(identity_max,value)
            check(prefix+"/"+field,value,1e-10)
        strong=float(np.max(abs(np.array(d["strong"]["weak_global_difference"]))))/scale
        strong_max=max(strong_max,strong)
        check(prefix+"/body_face_identity",strong,1e-10)
        for field in ("nested_matrix_relative_error","nested_load_relative_error"):
            nested_max=max(nested_max,d[field])
            check(prefix+"/"+field,d[field],1e-11)
        check(prefix+"/enriched_dual_accepted",d["enriched_dual_metrics"]["accepted"])
        for degree in ("p1","p2"):
            rec=r[degree]
            solver_records.append((prefix+"/"+degree,rec["solver"]))
            true_error=np.array(r["exact_qoi"])-np.array(rec["qoi"])
            check(prefix+"/"+degree+"/stored_error",float(np.max(abs(true_error-rec["signed_exact_qoi_error"]))) / scale,1e-14)
        for rec in r.get("amg",[]):
            solver_records.append((prefix+"/amg"+str(rec["degree"]),rec["solver"]))
        if r["kind"]=="adaptive":
            p=np.asarray(r["mesh_coordinates"],dtype=float)
            t=np.asarray(r["mesh_connectivity"],dtype=np.int32)
            local=np.asarray(r["strong_element_contributions"],dtype=float)
            if t.shape!=(3,r["elements"]) or local.shape!=(r["elements"],3):
                raise ValueError("Invalid adaptive payload")
            check(prefix+"/stored_local_sum",float(np.max(abs(local.sum(axis=0)-np.array(d["strong"]["global_estimates"]))))/scale,1e-10)
            if "marking" in r:
                info=r["marking"]
                scores=np.sum(abs(local),axis=1)
                order=np.lexsort((np.arange(len(scores)),-scores))
                count=int(np.searchsorted(np.cumsum(scores[order]),info["theta"]*scores.sum(),side="left"))+1
                selected=np.sort(order[:count])
                check(prefix+"/mark_scores",np.allclose(scores,info["scores"],rtol=1e-14,atol=0.))
                check(prefix+"/mark_selected",np.array_equal(selected,info["selected_ids"]))
                check(prefix+"/mark_theta",info["theta"]==config["marking_theta"])
                rebuilt=MeshTri(p,t,sort_t=r["mesh_sort_t"]).refined(selected)
                next_row=next(x for x in rows if x["case"]==r["case"] and x["kind"]=="adaptive" and x["key"]==r["key"]+1)
                check(prefix+"/next_mesh_coordinates",np.array_equal(rebuilt.p,np.array(next_row["mesh_coordinates"])))
                check(prefix+"/next_mesh_connectivity",np.array_equal(rebuilt.t,np.array(next_row["mesh_connectivity"])))
                decisions+=1
    check("adaptive_decision_count",decisions==len(names)*config["adaptive_refinements"])
    case_rows=[]
    for name in names:
        uniform=sorted((r for r in rows if r["case"]==name and r["kind"]=="uniform"),key=lambda x:x["key"])
        previous,finest=uniform[-2:]
        scale=finest["scale_from_input"]
        rates={}
        for degree, thresholds in (("p1",{"l2":1.8,"energy":.85}),("p2",{"l2":2.7,"energy":1.7})):
            for norm, threshold in thresholds.items():
                e0=previous[degree]["error_norms"][norm]
                e1=finest[degree]["error_norms"][norm]
                resolved=e1>100*np.finfo(float).eps*scale
                rate=float(np.log(e0/e1)/np.log(finest["key"]/previous["key"])) if resolved else None
                rates[degree+"_"+norm]=rate
                check(name+"/rate_"+degree+"_"+norm,False if rate is None else rate>=threshold)
        tau=config["tau_scale"]*scale
        maxerr=float(np.max(abs(np.array(finest["p2"]["signed_exact_qoi_error"]))))
        check(name+"/reference_error",maxerr/tau,.1)
        true=np.array(finest["p1"]["signed_exact_qoi_error"])
        total=np.array(finest["dwr"]["total_estimates"])
        effectivity=[]
        for j in range(3):
            valid=abs(true[j])>100*np.finfo(float).eps*scale
            effect=float(total[j]/true[j]) if valid else None
            effectivity.append(effect)
            check(name+"/effectivity"+str(j),False if effect is None else .9<=effect<=1.1)
        qr=next(q for q in quadrature if q["case"]==name)
        solver_records.append((name+"/quadrature18",qr["solver"]))
        qdiff=float(np.max(abs(np.array(qr["qoi_high"])-np.array(qr["qoi_low"])) ))
        check(name+"/quadrature_sensitivity",qdiff/tau,.01)
        case_rows.append(dict(case=name,finest_n=finest["key"],rates=rates,
            finest_p2_absolute_qoi_error=maxerr,finest_p2_error_over_tau=maxerr/tau,
            quadrature_difference_over_tau=qdiff/tau,effectivity=effectivity,
            adaptive_elements=[r["elements"] for r in rows if r["case"]==name and r["kind"]=="adaptive"]))
    worst_residual=worst_backward=0.
    for name,m in solver_records:
        check(name+"/solver_state",m["converged"] and m["all_info_zero"])
        if len(m["per_rhs"])!=4:
            raise ValueError("Expected four RHS columns")
        for j, rhs in enumerate(m["per_rhs"]):
            worst_residual=max(worst_residual,rhs["relative_residual"])
            worst_backward=max(worst_backward,rhs["componentwise_backward_error"])
            check(name+f"/rhs{j}_residual",rhs["relative_residual"],1e-9)
            check(name+f"/rhs{j}_backward",rhs["componentwise_backward_error"],1e-12)
    return dict(stage="15A",numerical_audit_pass=all(x["passed"] for x in checks),
        checks=checks,failed_checks=[x for x in checks if not x["passed"]],
        cases=case_rows,uniform_contexts=sum(r["kind"]=="uniform" for r in rows),
        adaptive_contexts=sum(r["kind"]=="adaptive" for r in rows),adaptive_decisions_rebuilt=decisions,
        linear_solver_calls=len(solver_records),rhs_solutions=4*len(solver_records),
        amg_calls=sum(len(r.get("amg",[])) for r in rows),quadrature_checks=len(quadrature),
        worst_relative_residual=worst_residual,worst_componentwise_backward_error=worst_backward,
        worst_normalized_dwr_identity_error=identity_max,worst_normalized_strong_weak_difference=strong_max,
        worst_nested_relative_error=nested_max,wall_seconds=done["wall_seconds"],
        G0_scope="aligned/smooth manufactured solutions only; general qualification remains incomplete",
        G1="not_evaluated",G2="not_evaluated",G3="not_evaluated",models_trained=0)


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=list(dict.fromkeys(r["case"] for r in rows))
    fig,axes=plt.subplots(2,3,figsize=(13,7.8),layout="constrained")
    for ax,name in zip(axes.flat,names):
        selected=sorted((r for r in rows if r["case"]==name and r["kind"]=="uniform"),key=lambda r:r["key"])
        for degree,color in (("p1","#20639b"),("p2","#d86a28")):
            for norm,style in (("l2","-"),("energy","--")):
                ax.loglog([r["key"] for r in selected],
                          [r[degree]["error_norms"][norm]/r["scale_from_input"] for r in selected],
                          style+"o",color=color,label=degree.upper()+" "+norm,markersize=3)
        ax.set_title(name.replace("layered_", ""),fontsize=11)
        ax.set_xlabel("n (cells per axis)");ax.set_ylabel("Error / input scale")
        ax.grid(True,which="both",alpha=.18)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Stage 15A: manufactured-solution convergence (no AI)",fontsize=15)
    fig.savefig(output/"convergence.png",dpi=180)
    fig.savefig(output/"convergence.pdf")
    plt.close(fig)


def run(directory, output, check_sources=True):
    directory,output=Path(directory),Path(output)
    if output.exists():
        raise FileExistsError("Use a new report directory")
    freeze,rows,quad,done=read_checked(directory,check_sources)
    summary=summarize(freeze,rows,quad,done)
    output.mkdir(parents=True)
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    plot(rows,output)
    table=["| 案例 | P1 L2阶 | P1能量阶 | P2 L2阶 | P2能量阶 | 最细P2目标误差/τ | DWR效能范围 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    fmt=lambda x:"不可判定" if x is None else f"{x:.4f}"
    for c in summary["cases"]:
        rr=c["rates"];e=[x for x in c["effectivity"] if x is not None]
        es=f"{min(e):.6f}–{max(e):.6f}" if e else "不可判定"
        table.append(f'| {c["case"]} | '+" | ".join(fmt(rr[k]) for k in ("p1_l2","p1_energy","p2_l2","p2_energy"))+
                     f' | {c["finest_p2_error_over_tau"]:.3e} | {es} |')
    verdict="通过" if summary["numerical_audit_pass"] else "未通过"
    text=f'''# 第十五阶段A数值核验报告

本轮预定数值门槛：**{verdict}**。这是六个确定性制造解上的实现与精度核验，不是总体统计结论、连续误差证书或学习收益证明。

{summary["uniform_contexts"]}个均匀网格上下文、{summary["adaptive_contexts"]}个局部细化状态；重建{summary["adaptive_decisions_rebuilt"]}次实际标记及下一网格。共{summary["linear_solver_calls"]}次求解器调用、{summary["rhs_solutions"]}个右端解（每次4右端共用分解/层级）；包括{summary["amg_calls"]}次AMG调用和{summary["quadrature_checks"]}次18阶求积对照。完整审查耗时{summary["wall_seconds"]:.2f}秒，包含离线解析评价与重复核验，不能作为部署求解耗时。

{chr(10).join(table)}

最坏实际相对残差：{summary["worst_relative_residual"]:.3e}；最坏分量后向误差：{summary["worst_componentwise_backward_error"]:.3e}。归一化DWR恒等式/局部总和最坏差：{summary["worst_normalized_dwr_identity_error"]:.3e}；强/弱式最坏全局差：{summary["worst_normalized_strong_weak_difference"]:.3e}。

失败检查数：{len(summary["failed_checks"])}。完整逐项结果在summary.json，原始输入/记录在实验目录，不删失败样本。

![制造解收敛](convergence.png)

局部标记使用强式体/面DWR贡献的绝对值聚合与固定θ=.5的Dörfler规则；解析真值只参与离线评价。三轮固定细化不从真误差决定提前停止。不同合法局部化不必产生相同排名。

**边界：** 参考检查有已知解析解支撑，仅适用于本轮固定材料与制造载荷。未对齐双包含体、一般屏障/通道、平衡通量认证、NGO-AFEM与同精度成本尚未完成。G1/G2/G3均未评价，训练模型数0。下一项应完成代表性问题的强基线，而不是直接训练。
'''
    (output/"REPORT_zh.md").write_text(text)
    (output/"complete.json").write_text(json.dumps(dict(complete=True,source_directory=str(directory),
        file_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()},
        source_complete_sha256=digest(directory/"complete.json")),indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("checks","cases")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="results/stage15_numerical")
    p.add_argument("--output",default="results/stage15_report")
    args=p.parse_args()
    run(ROOT/args.input,ROOT/args.output)
