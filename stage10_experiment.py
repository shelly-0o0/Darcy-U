"""Frozen representation pilot for a residual-pair discretization error operator.

Not an adaptive mesh algorithm, certified estimator, or end-to-end speed study.
All references and classical baselines use the very same nested Galerkin system.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from stage10_error_model import ErrorOperator, energy
from stage10_goal_numerics import assemble_nested, solve_primal, solve_goal, element_indicators
from sequence_baselines import FixedAMGHierarchy

ROOT=Path(__file__).resolve().parent
KINDS=("symmetric","untied","nonlinear")
SEEDS=(51001,51002,51003)
MIXTURE_WEIGHTS=np.array([[1.,-1.,0.,0.],[.5,.5,-.5,-.5],[-3.,0.,0.,0.],[0.,10.,-5.,1.]])
CONFIG=dict(train_geometries=64,validation_geometries=12,test_geometries=16,
            goals=4,coarse_cells=16,epochs=60,batch_geometries=4,
            learning_rate=.002,weight_decay=1e-5,width=8,levels=3,
            gram_loss_weight=.25,seeds=SEEDS,kinds=KINDS)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def assert_frozen(frozen,checkpoints=(),root=ROOT):
    for name,digest in frozen["source_sha256"].items():
        if sha(root/name)!=digest:raise RuntimeError(f"Frozen source changed: {name}")
    if sha(root/"docs/stage10_protocol_zh.md")!=frozen["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")
    for item in checkpoints:
        if sha(root/item["checkpoint"])!=item["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen checkpoint changed: {item['checkpoint']}")


def coefficient(seed,n,ood=False):
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[:n,:n];x=(xx+.5)/n;y=(yy+.5)/n
    center=rng.uniform(.25,.75,2);radius=rng.uniform(.12,.25)
    origin=rng.uniform(.15,.6,2);sizes=rng.uniform(.15,.3,2)
    contrast=float(rng.choice([100.,1000.] if ood else [5.,20.]))
    mask=((x-center[0])**2+(y-center[1])**2<radius**2)
    mask|=(abs(x-origin[0])<sizes[0]/2)&(abs(y-origin[1])<sizes[1]/2)
    smooth=np.exp(.25*np.sin(2*np.pi*x+rng.uniform(0,6))*np.cos(2*np.pi*y+rng.uniform(0,6)))
    return smooth*np.where(mask,contrast,1.)


def rhs_functions(seed,narrow=False):
    rng=np.random.default_rng(seed+700000)
    centers=rng.uniform(.2,.8,(4,2));widths=rng.uniform(.025,.045,4) if narrow else rng.uniform(.08,.18,4)
    return [lambda x,y,c=c.copy(),w=float(w): np.exp(-((x-c[0])**2+(y-c[1])**2)/(2*w*w))
            for c,w in zip(centers,widths)]


def features(k,diag,n):
    # Geometry/coefficient only: source/goal cannot influence any gates.
    lk=np.log(k)
    nodal=(lk[:-1,:-1]+lk[1:,:-1]+lk[:-1,1:]+lk[1:,1:])/4
    yy,xx=np.mgrid[1:n,1:n];x=xx/n;y=yy/n
    distance=np.minimum.reduce((x,y,1-x,1-y))
    return np.stack((nodal/np.log(20.), np.log(diag)-np.log(diag).mean(),distance))


def make_case(seed,nc=16,ood=False,narrow=False):
    start=time.perf_counter(); nf=2*nc;m=nf-1
    k=coefficient(seed,nf,ood); funcs=rhs_functions(seed,narrow)
    case=assemble_nested(nc,k,funcs[0])
    primals=[]; goals=[]
    # Every right-hand side can play either the source or the goal role.
    for f in funcs:
        p=solve_primal(case,source=f,fine_reference=True)
        g=solve_goal(case,f,primal=p)
        primals.append(p); goals.append(g)
    r=np.stack([g["r_z"] for g in goals]);target=np.stack([g["delta_z"] for g in goals])
    a=case.Af;d=a.diagonal().reshape(m,m)
    e=r@target.T
    tensors=dict(features=features(k,d,nf).astype("float32"),
        residual=r.reshape(4,1,m,m).astype("float32"),
        target=target.reshape(4,1,m,m).astype("float32"),
        diagonal=d[None].astype("float32"),
        east=np.pad(a.diagonal(1),(0,1)).reshape(m,m)[:,:-1][None].astype("float32"),
        south=a.diagonal(m).reshape(m-1,m)[None].astype("float32"),
        gram=e.astype("float32"))
    diagnostics=dict(seed=seed,nc=nc,ood=ood,narrow=narrow,
        coefficient_sha256=hashlib.sha256(k.tobytes()).hexdigest(),
        reference_symmetry=float(np.linalg.norm(e-e.T)/np.linalg.norm(e)),
        minimum_reference_eigenvalue=float(np.linalg.eigvalsh((e+e.T)/2).min()),
        max_galerkin_residual=float(max(np.linalg.norm(case.P.T@z)/np.linalg.norm(z) for z in r)),
        max_reference_relative_residual=float(max(np.linalg.norm(a@v-z)/np.linalg.norm(z) for v,z in zip(target,r))),
        generation_seconds=time.perf_counter()-start)
    if (not all(np.isfinite(x).all() for x in (r,target,e)) or np.any(e.diagonal()<=0)
        or diagnostics["minimum_reference_eigenvalue"] < -1e-10*np.linalg.norm(e,2)
        or diagnostics["max_galerkin_residual"]>1e-8
        or diagnostics["reference_symmetry"]>1e-9 or diagnostics["max_reference_relative_residual"]>1e-8):
        raise RuntimeError(f"Invalid reference {diagnostics}")
    return dict(case=case,primals=primals,goals=goals,tensors=tensors,reference=e,
                r=r,target=target,diagnostics=diagnostics)


def dataset(seeds,nc=16,ood=False,narrow=False):
    return [make_case(s,nc,ood,narrow) for s in seeds]


def stacked(data,device):
    return {k:torch.tensor(np.stack([x["tensors"][k] for x in data]),device=device)
            for k in data[0]["tensors"]}


def prediction(model,batch,nf):
    b,q,_,h,w=batch["residual"].shape
    feat=batch["features"][:,None].expand(-1,q,-1,-1,-1).reshape(b*q,3,h,w)
    diag=batch["diagonal"][:,None].expand(-1,q,-1,-1,-1).reshape(b*q,1,h,w)
    return model(feat,batch["residual"].reshape(b*q,1,h,w),diag,nf).reshape(b,q,1,h,w)


def objective(out,batch):
    err=out-batch["target"]
    d=batch["diagonal"][:,None];east=batch["east"][:,None];south=batch["south"][:,None]
    denom=energy(batch["target"],d,east,south).clamp_min(1e-25)
    defect=(energy(err,d,east,south)/denom).mean()
    gram=torch.einsum("bqi,bsi->bqs",batch["residual"].flatten(2),out.flatten(2))
    scale=(batch["gram"].diagonal(dim1=-2,dim2=-1)[:,:,None]*
           batch["gram"].diagonal(dim1=-2,dim2=-1)[:,None,:]).clamp_min(1e-30)
    gl=((gram-batch["gram"]).square()/scale).mean()
    return defect+.25*gl,defect,gl


def fit(kind,seed,train,val,device,out,epochs):
    torch.manual_seed(seed);np.random.seed(seed)
    model=ErrorOperator(kind,width=CONFIG["width"]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=CONFIG["learning_rate"],weight_decay=CONFIG["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=1e-4)
    generator=np.random.default_rng(seed)
    history=[];best=float("inf");best_epoch=None
    target=out/f"{kind}_{seed}.pt";start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        order=generator.permutation(train["features"].shape[0])
        for i in range(0,len(order),CONFIG["batch_geometries"]):
            ids=order[i:i+CONFIG["batch_geometries"]]
            batch={k:v[ids] for k,v in train.items()}
            opt.zero_grad(set_to_none=True)
            loss,_,_=objective(prediction(model,batch,32),batch)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
            losses.append(float(loss.detach()))
        scheduler.step();model.eval()
        with torch.no_grad():vl,ve,vg=objective(prediction(model,val,32),val)
        score=float(vl);history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),
            validation_loss=score,validation_energy=float(ve),validation_gram=float(vg)))
        if score<best:
            best=score;best_epoch=epoch
            torch.save(dict(state_dict=model.state_dict(),kind=kind,seed=seed,epoch=epoch,
                            validation_loss=best,config=CONFIG),target)
        if epoch%10==0: print(json.dumps(dict(event="training",kind=kind,seed=seed,epoch=epoch,val=score)),flush=True)
    seconds=time.perf_counter()-start
    model.load_state_dict(torch.load(target,map_location=device,weights_only=False)["state_dict"])
    model.eval()
    record=dict(kind=kind,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
        best_epoch=best_epoch,validation_loss=best,training_seconds=seconds,
        checkpoint=str(target.relative_to(ROOT)),checkpoint_sha256=sha(target),history=history)
    dump(out/f"train_{kind}_{seed}.json",record)
    return model,record


def metrics(item,pred):
    r=item["r"];e=item["reference"];actual=r@pred.T
    target=item["target"];a=item["case"].Af
    energy_ratios=[float((v-t)@(a@(v-t))/(t@(a@t))) for v,t in zip(pred,target)]
    normalized_symmetric=(actual+actual.T)/2
    eig=np.linalg.eigvalsh(normalized_symmetric)
    recalls=[]; indicator_l1=[]
    for j in range(4):
        true=element_indicators(item["case"],item["primals"][0],target[j])["coarse_cell"].ravel()
        estimated=element_indicators(item["case"],item["primals"][0],pred[j])["coarse_cell"].ravel()
        number=int(np.ceil(.25*len(true)))
        best=np.argsort(-abs(true),kind="stable")[:number]
        selected=np.argsort(-abs(estimated),kind="stable")[:number]
        recalls.append(float(abs(true[selected]).sum()/max(abs(true[best]).sum(),1e-30)))
        indicator_l1.append(float(abs(estimated-true).sum()/max(abs(true).sum(),1e-30)))
    return dict(matrix_error_relative=float(np.linalg.norm(actual-e)/np.linalg.norm(e)),
        normalized_matrix_rmse=float(np.sqrt(np.mean((actual-e)**2/np.outer(e.diagonal(),e.diagonal())))),
        defect_energy_error_mean=float(np.mean(energy_ratios)),
        reciprocity_defect=float(np.linalg.norm(actual-actual.T)/max(np.linalg.norm(actual),1e-30)),
        minimum_symmetric_eigenvalue=float(eig.min()),
        relative_minimum_symmetric_eigenvalue=float(eig.min()/max(np.linalg.norm(actual,2),1e-30)),
        diagonal_positive=bool(np.all(actual.diagonal()>0)),
        marker_mass_ratio=float(np.mean(recalls)),indicator_relative_l1=float(np.mean(indicator_l1)),
        reference_matrix=e.tolist(),estimated_matrix=actual.tolist())


def mixture_metrics(item,pred,mixed):
    weights=MIXTURE_WEIGHTS;expected=weights@pred
    emix=(weights@item["r"])@mixed.T;exact=weights@item["reference"]@weights.T
    cross=item["r"]@mixed.T;cross_exact=item["reference"]@weights.T
    scale=np.outer(item["reference"].diagonal(),exact.diagonal())
    return dict(mixture_weights=weights.tolist(),mixture_reference_matrix=exact.tolist(),
        mixture_estimated_matrix=emix.tolist(),mixture_cross_reference=cross_exact.tolist(),
        mixture_cross_estimated=cross.tolist(),
        mixture_linearity_defect=float(np.linalg.norm(mixed-expected)/max(np.linalg.norm(expected),1e-30)),
        mixture_linearity_by_combination=(np.linalg.norm(mixed-expected,axis=1)/np.maximum(np.linalg.norm(expected,axis=1),1e-30)).tolist(),
        mixture_cross_normalized_rmse=float(np.sqrt(np.mean((cross-cross_exact)**2/scale))),
        mixture_cross_normalized_rmse_by_combination=np.sqrt(np.mean((cross-cross_exact)**2/scale,axis=0)).tolist(),
        mixture_diagonal_relative_error=(abs(emix.diagonal()-exact.diagonal())/exact.diagonal()).tolist(),
        mixture_matrix_error_relative=float(np.linalg.norm(emix-exact)/np.linalg.norm(exact)),
        mixed_symmetric_eigenvalue_relative=float(np.linalg.eigvalsh((emix+emix.T)/2).min()/max(np.linalg.norm(emix,2),1e-30)))


def evaluate_model(item,model,device):
    b=stacked([item],device);nf=item["case"].nf
    with torch.no_grad():
        pred=prediction(model,b,nf).cpu().numpy().reshape(4,-1).astype("float64")
        # Fixed signed and amplified mixtures never appear in training.
        weights=MIXTURE_WEIGHTS
        mix=dict(b)
        mix["residual"]=torch.tensor((weights@item["r"]).reshape(1,4,1,nf-1,nf-1),dtype=torch.float32,device=device)
        mixed=prediction(model,mix,nf).cpu().numpy().reshape(4,-1).astype("float64")
    result=metrics(item,pred)
    result.update(mixture_metrics(item,pred,mixed))
    return result


def classical(item):
    a=item["case"].Af;r=np.concatenate((item["r"],MIXTURE_WEIGHTS@item["r"]));d=a.diagonal()
    hierarchy=FixedAMGHierarchy(a,seed=51700)
    result={}
    jac=r/d
    result["jacobi1"]={**metrics(item,jac[:4]),**mixture_metrics(item,jac[:4],jac[4:])}
    jac=np.zeros_like(r)
    for _ in range(8):jac+=(2./3)*(r-(a@jac.T).T)/d
    result["jacobi8"]={**metrics(item,jac[:4]),**mixture_metrics(item,jac[:4],jac[4:])}
    amg=np.stack([hierarchy.preconditioner@v for v in r])
    result["amg_v1"]={**metrics(item,amg[:4]),**mixture_metrics(item,amg[:4],amg[4:])}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",default="results/stage10_pilot")
    args=parser.parse_args();out=ROOT/args.output
    if (out/"freeze.json").exists():raise FileExistsError("Use a fresh output directory; never overwrite a frozen run")
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_names=["stage10_experiment.py","stage10_error_model.py","stage10_goal_numerics.py",
                  "test_stage10_error_model.py","test_stage10_goal_numerics.py","test_stage10_experiment.py",
                  "numerics.py","sequence_baselines.py"]
    frozen=dict(config=CONFIG,smoke=args.smoke,source_sha256={n:sha(ROOT/n) for n in source_names},
        protocol_sha256=sha(ROOT/"docs/stage10_protocol_zh.md"),
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
            device=str(device),device_name=torch.cuda.get_device_name(0) if device.type=="cuda" else platform.processor()))
    dump(out/"freeze.json",frozen)
    count=4 if args.smoke else CONFIG["train_geometries"]
    train=dataset(range(50000,50000+count));val=dataset(range(50100,50100+(2 if args.smoke else 12)))
    train_t=stacked(train,device);val_t=stacked(val,device)
    dump(out/"training_data.json",dict(train=[x["diagnostics"] for x in train],validation=[x["diagnostics"] for x in val]))
    models={};training=[]
    for seed in (SEEDS[:1] if args.smoke else SEEDS):
        for kind in KINDS:
            model,rec=fit(kind,seed,train_t,val_t,device,out,2 if args.smoke else CONFIG["epochs"])
            models[(kind,seed)]=model;training.append(rec)
    dump(out/"training_complete.json",dict(checkpoints=[dict(kind=x["kind"],seed=x["seed"],sha256=x["checkpoint_sha256"]) for x in training]))
    assert_frozen(frozen,training)
    nt=2 if args.smoke else 16
    sets=[("id",range(50200,50200+nt),16,False,False),
          ("contrast",range(50300,50300+nt),16,True,False),
          ("narrow_load",range(50200,50200+nt),16,False,True),
          ("resolution",range(50200,50200+nt),32,False,False)]
    rows=[];manifest=[]
    for split,seeds,nc,ood,narrow in sets:
        for geometry in seeds:
            item=make_case(geometry,nc,ood,narrow);manifest.append(dict(split=split,**item["diagnostics"]))
            for method,record in classical(item).items():
                rows.append(dict(split=split,geometry=geometry,method=method,training_seed=None,**record))
            for (kind,seed),model in models.items():
                rows.append(dict(split=split,geometry=geometry,method=kind,training_seed=seed,**evaluate_model(item,model,device)))
        print(json.dumps(dict(event="evaluated",split=split,geometries=nt)),flush=True)
        dump(out/"evaluation.json",rows);dump(out/"test_data.json",manifest)
    hashes={x["coefficient_sha256"] for x in [i["diagnostics"] for i in train+val]}
    if any(x["coefficient_sha256"] in hashes for x in manifest):raise RuntimeError("Train/test coefficient collision")
    assert_frozen(frozen,training)
    dump(out/"complete.json",dict(complete=True,freeze_sha256=sha(out/"freeze.json"),
        rows=len(rows),models=len(training),training_seconds=sum(x["training_seconds"] for x in training),
        source_sha256={n:sha(ROOT/n) for n in source_names},
        result_sha256={n:sha(out/n) for n in ["evaluation.json","test_data.json","training_data.json","training_complete.json"]},
        scope="Representation pilot only; no runtime, adaptivity, continuum accuracy or novelty claim"))
    print(json.dumps(dict(event="complete",rows=len(rows),output=str(out))),flush=True)


if __name__=="__main__":main()
