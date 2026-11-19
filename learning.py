"""Small U-Net and basis-invariant incremental subspace training.

The network is trained on local spectral targets, never on test pressures.
Full B matrices are stored sparsely; training uses sparse matrix products.
"""
from __future__ import annotations
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from fields import coefficient_sequence
from numerics import assemble_system, build_patches, patch_spectral_basis, b_orthonormalize


def field_hash(fields):
    return hashlib.sha256(np.ascontiguousarray(fields).tobytes()).hexdigest()


def nodal_values(cells):
    pad = np.pad(np.asarray(cells), 1, mode='edge')
    return (pad[:-1, :-1]+pad[1:, :-1]+pad[:-1, 1:]+pad[1:, 1:])*.25


def make_features(patch, old_cells, old_space, pad=32):
    iy, ix = patch.local_positions
    ny, nx = patch.grid_shape
    if max(ny, nx) > pad:
        raise ValueError('Increase pad for a larger patch')
    features = np.zeros((6, pad, pad), dtype=np.float32)
    lk = nodal_values(np.log(patch.coefficient))
    lo = nodal_values(np.log(old_cells))
    scale = np.log(1e6)  # fixed before training; OOD inputs are not clipped
    features[0, iy, ix] = lk[iy, ix]/scale
    features[1, iy, ix] = (lk-lo)[iy, ix]/scale
    if old_space.shape[1]:
        q, _ = la.qr(old_space, mode='economic', check_finite=False)
        features[2, iy, ix] = np.sum(q*q, axis=1)
    features[3, iy, ix] = 2*ix/max(nx-1, 1)-1
    features[4, iy, ix] = 2*iy/max(ny-1, 1)-1
    features[5, iy, ix] = 1.
    return features


class ConvBlock(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(ni,no,3,padding=1), nn.GroupNorm(4,no),
                                    nn.SiLU(), nn.Conv2d(no,no,3,padding=1),
                                    nn.GroupNorm(4,no), nn.SiLU())
    def forward(self,x): return self.layers(x)


class TinyUNet(nn.Module):
    def __init__(self, width=16, directions=4):
        super().__init__()
        self.width, self.directions = width, directions
        self.e1=ConvBlock(6,width)
        self.e2=ConvBlock(width,2*width)
        self.mid=ConvBlock(2*width,4*width)
        self.d2=ConvBlock(6*width,2*width)
        self.d1=ConvBlock(3*width,width)
        self.head=nn.Conv2d(width,directions,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(F.avg_pool2d(a,2)); c=self.mid(F.avg_pool2d(b,2))
        d=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        e=self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.head(e)


def build_dataset(path, train_sequences=32, val_sequences=8, n=128, steps=20,
                  gamma=.03, core=16, overlap=2, max_patches_per_pair=3):
    """Disjoint geometry seeds; labels computed only for selected changed patches."""
    start=time.perf_counter()
    train_seeds=list(range(100,100+train_sequences))
    val_seeds=list(range(10000,10000+val_sequences))
    data={'train':[],'val':[], 'manifest':{'train_seeds':train_seeds,'val_seeds':val_seeds,
          'n':n,'steps':steps,'gamma':gamma,'core':core,'overlap':overlap,'hashes':{}}}
    seen=set()
    for split,seeds in [('train',train_seeds),('val',val_seeds)]:
        for count,seed in enumerate(seeds):
            contrast=float(10**(2+(seed%3)))
            fields,_=coefficient_sequence(n,steps,seed,contrast,'mixed')
            digest=field_hash(fields)
            if digest in seen:
                raise RuntimeError('Duplicate full coefficient sequence across data seeds')
            seen.add(digest)
            data['manifest']['hashes'][str(seed)]=digest
            rng=np.random.default_rng(seed+700000)
            times=sorted(set([steps//4,steps//2,3*steps//4,steps-1]))
            for t in times:
                s=max(0,t-int(rng.choice([1,3,5])))
                ao,_,_=assemble_system(fields[s]); an,_,_=assemble_system(fields[t])
                po=build_patches(fields[s],ao,core,overlap)
                pn=build_patches(fields[t],an,core,overlap)
                changed=[i for i,(p,q) in enumerate(zip(po,pn)) if not np.array_equal(p.B,q.B)]
                rng.shuffle(changed)
                for i in changed[:max_patches_per_pair]:
                    old,new=po[i],pn[i]
                    zo=patch_spectral_basis(old,gamma=gamma,anchor=2*gamma)
                    zo=b_orthonormalize(zo,new.B)
                    target=patch_spectral_basis(new,gamma=gamma,anchor=2*gamma)
                    bt=new.B@target
                    residual=target-zo@(zo.T@bt) if zo.shape[1] else target.copy()
                    missing=float(np.sum(residual*(new.B@residual)))
                    data[split].append({'features':make_features(new,old.coefficient,zo),
                        'B':sp.csr_matrix(new.B), 'Z':zo,'target':target,
                        'iy':new.local_positions[0], 'ix':new.local_positions[1],
                        'invdiag':1/np.sqrt(np.diag(new.B)), 'missing':missing,
                        'seed':seed,'step':t,'anchor_step':s,'patch':i,'contrast':contrast})
                del ao,an,po,pn
            print(json.dumps({'stage':'data','split':split,'sequences':count+1,
                              'samples':len(data[split])}),flush=True)
    data['manifest']['generation_seconds']=time.perf_counter()-start
    data['manifest']['samples']={s:len(data[s]) for s in ['train','val']}
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    path.with_suffix('.manifest.json').write_text(json.dumps(data['manifest'],indent=2))
    return data


def tensor_sample(sample,device):
    coo=sample['B'].tocoo()
    b=torch.sparse_coo_tensor(torch.from_numpy(np.vstack([coo.row,coo.col])),
        torch.from_numpy(coo.data),coo.shape,dtype=torch.float64).coalesce().to(device)
    return {'B':b,'Z':torch.as_tensor(sample['Z'],device=device,dtype=torch.float64),
        'target':torch.as_tensor(sample['target'],device=device,dtype=torch.float64),
        'iy':torch.as_tensor(sample['iy'],device=device,dtype=torch.long),
        'ix':torch.as_tensor(sample['ix'],device=device,dtype=torch.long),
        'invdiag':torch.as_tensor(sample['invdiag'],device=device,dtype=torch.float64)}


def coverage_loss(output,sample):
    z,b,target=sample['Z'],sample['B'],sample['target']
    if target.shape[1]==0:
        return output.sum()*0.
    y=output[:,sample['iy'],sample['ix']].T.to(torch.float64)*sample['invdiag'][:,None]
    by=torch.sparse.mm(b,y)
    if z.shape[1]: y=y-z@(z.T@by)
    by=torch.sparse.mm(b,y)
    gram=y.T@by
    scale=torch.diagonal(gram).clamp_min(1e-20).rsqrt()
    y=y*scale[None,:]; by=by*scale[None,:]
    gram=y.T@by
    target_remaining=target-z@(z.T@torch.sparse.mm(b,target)) if z.shape[1] else target
    bt=torch.sparse.mm(b,target_remaining)
    cross=y.T@bt
    # Tiny ridge is training regularization only. Inference removes dependent columns.
    gain=torch.sum(cross*torch.linalg.solve(gram+1e-6*torch.eye(gram.shape[0],device=gram.device,dtype=gram.dtype),cross))
    initial=torch.sum(target_remaining*bt)
    return (initial-gain)/max(target.shape[1],1)


def train_model(data, checkpoint, epochs=50, batch_size=16, seed=7, width=16, directions=4,
                device='cpu',threads=2):
    torch.set_num_threads(threads)
    torch.manual_seed(seed); np.random.seed(seed)
    model=TinyUNet(width,directions).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-5)
    x={s:torch.as_tensor(np.stack([d['features'] for d in data[s]]),device=device) for s in ['train','val']}
    st={s:[tensor_sample(d,device) for d in data[s]] for s in ['train','val']}
    zero_update_baseline={s:float(np.mean([
        d['missing']/max(d['target'].shape[1],1) for d in data[s]
    ])) for s in ['train','val']}
    history=[]; best=float('inf'); best_epoch=-1
    checkpoint=Path(checkpoint); checkpoint.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    for epoch in range(epochs):
        model.train(); order=np.random.permutation(len(st['train'])); losses=[]
        for offset in range(0,len(order),batch_size):
            ids=order[offset:offset+batch_size]
            optimizer.zero_grad(set_to_none=True)
            outputs=model(x['train'][ids])
            loss=torch.stack([coverage_loss(outputs[j],st['train'][int(i)]) for j,i in enumerate(ids)]).mean()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite subspace loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        scheduler.step(); model.eval(); vals=[]
        with torch.no_grad():
            for offset in range(0,len(st['val']),batch_size):
                outputs=model(x['val'][offset:offset+batch_size])
                vals.extend(float(coverage_loss(outputs[j],st['val'][i])) for j,i in enumerate(range(offset,min(offset+batch_size,len(st['val'])))))
        record={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_loss':float(np.mean(vals)),
                'elapsed_seconds':time.perf_counter()-start}
        history.append(record)
        if record['val_loss']<best:
            best=record['val_loss']; best_epoch=epoch+1
            torch.save({'state_dict':model.state_dict(),'width':width,'directions':directions,
                        'seed':seed,'epoch':best_epoch,'validation_loss':best},checkpoint)
        if epoch==0 or (epoch+1)%5==0: print(json.dumps({'stage':'train',**record}),flush=True)
    result={'history':history,'best_epoch':best_epoch,'best_validation_loss':best,
            'train_seconds':time.perf_counter()-start,'seed':seed,'device':device,
            'parameters':sum(p.numel() for p in model.parameters()),'epochs':epochs,'batch_size':batch_size,
            'zero_update_baseline_loss':zero_update_baseline,
            'zero_update_baseline_description':'mean B-weighted missing target energy / target dimension before adding any learned directions',
            'selection':'minimum validation subspace loss; no test measurements used'}
    checkpoint.with_suffix('.training.json').write_text(json.dumps(result,indent=2))
    return load_model(checkpoint,device),result


def load_model(path,device='cpu'):
    state=torch.load(path,map_location=device,weights_only=True)
    model=TinyUNet(state['width'],state['directions']).to(device)
    model.load_state_dict(state['state_dict']); model.eval()
    return model


def predict_directions(model,patches,old_coeffs,old_spaces,device='cpu'):
    if not patches: return []
    features=np.stack([make_features(p,c,z) for p,c,z in zip(patches,old_coeffs,old_spaces)])
    with torch.no_grad():
        output=model(torch.as_tensor(features,device=device)).cpu().numpy()
    result=[]
    for out,p,z in zip(output,patches,old_spaces):
        y=out[:,p.local_positions[0],p.local_positions[1]].T.astype(np.float64)
        y/=np.sqrt(np.diag(p.B))[:,None]
        delta=b_orthonormalize(y,p.B,existing=z)
        result.append(np.column_stack([z,delta]))
    return result
