"""Algebraic checks needed for the complete-T ordered-area experiment."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse
import datetime
import json
from pathlib import Path
import numpy as np
import torch
import brainhive_ordered_area_20261028_v1 as M


def run(output):
    torch.set_num_threads(2)
    rng=np.random.default_rng(1729);x=rng.normal(size=(12,5,6));a,s=M.area_and_scatter(x)
    direct=np.zeros((12,6,6))
    for t in range(5):
        for u in range(t+1,5):
            v=np.einsum('ni,nj->nij',x[:,t],x[:,u]);direct+=.5*(v-v.swapaxes(1,2))
    ids=np.triu_indices(6,1);assert np.allclose(a,direct[:,ids[0],ids[1]],atol=1e-14)
    ar,sr=M.area_and_scatter(x[:,::-1]);assert np.allclose(ar,-a,atol=1e-14) and np.allclose(sr,s,atol=1e-14)
    permutation=[0,2,1,3,4];ap,sp=M.area_and_scatter(x[:,permutation]);assert not np.allclose(ap,a)
    assert np.allclose(x.sum(1),x[:,permutation].sum(1)) and np.allclose(sp,s)
    endpoint=x.sum(1);s2=np.einsum('nti,ntj->nij',np.cumsum(x,axis=1)-x,x)+.5*np.einsum('nti,ntj->nij',x,x)
    assert np.allclose(s2+s2.swapaxes(1,2),np.einsum('ni,nj->nij',endpoint,endpoint),atol=1e-13)
    # A global orientation reversal is an invertible feature gauge, not a trained control.
    n=60;X=torch.from_numpy(rng.normal(size=(n,9)));Z=torch.from_numpy(rng.normal(size=(n,5)));Y=torch.from_numpy(rng.normal(size=(n,3)));penalty=.6
    L,base=M.solve(torch,X,Y,penalty);linear,extra=M.extend(torch,L,base,X,Z,Y,penalty)
    joint=torch.cat((X,Z),dim=1);_,truth=M.solve(torch,joint,Y,penalty)
    assert torch.max(torch.abs(torch.cat((linear,extra))-truth))<1e-12
    revlinear,revextra=M.extend(torch,L,base,X,-Z,Y,penalty)
    assert torch.max(torch.abs(X@linear+Z@extra-(X@revlinear-Z@revextra)))<1e-12
    # FIT-only projection and scaling must not change when held EEG changes.
    small=rng.normal(size=(30,5,12));fit=np.arange(20)
    p,note=M.fit_basis(small,fit,rank=6);small[20:]+=100
    p2,note2=M.fit_basis(small,fit,rank=6)
    assert np.array_equal(note['basis'],note2['basis']) and np.array_equal(note['mean'],note2['mean'])
    assert np.allclose(p[fit],p2[fit])
    result=dict(status='D38_ORDERED_AREA_REVERSAL_GAUGE_FIT_ONLY_AND_SCHUR_CHECKS_PASSED',
       UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tests=7,
       global_reversal_retraining_is_not_valid_control=True,CUDA_initialized=torch.cuda.is_initialized())
    assert not result['CUDA_initialized']
    with Path(output).open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
