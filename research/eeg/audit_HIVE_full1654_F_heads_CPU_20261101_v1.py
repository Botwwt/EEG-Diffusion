"""Actual fullscope state, ordered coverage and all165400 CPU64 forwards."""
import argparse,gc,hashlib,time
from pathlib import Path
import brainhive_full1654_F_study_contract_20261101_v1 as G

def run(path,arm):
    start=time.monotonic();s=G.setup(path)
    import torch,numpy as np
    import torch.nn.functional as F
    import brainhive_exposed_fullfit_component_inputs_CPU_20261101_v1 as I
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    out=Path(s['output_root'])/arm;done=G.read(out/'fit_completion.json');assert done['status']=='COMPLETE_FULL1654_F_HEAD_THREE_PASSES'
    assert done['study']==G.ref(path) and done['arm']==arm and (done['updates'],done['exposures'])==(31014,496200)
    inputs=I.load_verified(s['input_bank'],s['asset_protocol']);orders=[torch.randperm(165400,generator=torch.Generator().manual_seed(G.seed('pass'+str(i)))) for i in range(3)]
    hashes=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in orders]
    provenance=G.read(G.check(done['provenance']));assert hashes==provenance['orders_sha256']
    expected_steps=[128,8192,10338,16384,20676,24576,31014];assert [r['updates'] for r in done['checkpoints']]==expected_steps
    final=torch.load(G.check(done['component']),weights_only=True,map_location='cpu');assert final['seed']==1729 and final['strength']==1.
    cls=G.model_class(arm);model=cls().eval();model.load_state_dict(final['model'],strict=True)
    assert sum(v.numel() for v in model.parameters())==525568
    states=[]
    for commit in done['checkpoints']:
        v=torch.load(G.check(commit['checkpoint']),weights_only=True,map_location='cpu');n=v['updates']
        assert v['study']==G.ref(path) and v['orders_sha256']==hashes and v['arm']==arm
        assert n==commit['updates'] and v['exposures']==commit['exposures']
        coverage=torch.zeros(3,165400,dtype=torch.bool);remaining=n
        for epoch in range(3):
            take=min(remaining,10338);count=min(165400,take*16);coverage[epoch,orders[epoch][:count]]=True;remaining-=take
        assert remaining==0 and torch.equal(coverage,v['coverage']) and int(coverage.sum())==v['exposures']
        opt=v['optimizer'];assert len(opt['state'])==4 and len(opt['param_groups'])==1
        group=opt['param_groups'][0];assert (group['lr'],group['weight_decay'],tuple(group['betas']),group['eps'])==(1e-4,.01,(.9,.999),1e-8)
        for index,p in zip(group['params'],model.parameters()):
            state=opt['state'][index];assert set(state)=={'step','exp_avg','exp_avg_sq'} and int(state['step'])==n
            assert all(state[k].shape==p.shape and torch.isfinite(state[k]).all() for k in ('exp_avg','exp_avg_sq')) and (state['exp_avg_sq']>=0).all()
        if n==31014:assert all(torch.equal(v['model'][k],value) for k,value in final['model'].items())
        states.append(dict(updates=n,exposures=v['exposures'],all4_Adam_counters_moments_and_coverage_verified=True));del v;gc.collect()
    actual=np.load(G.check(done['actual_train_outputs']),allow_pickle=False,mmap_mode='r');assert actual.shape==(165400,1024) and actual.dtype==np.float32
    w={k:v.double() for k,v in final['model'].items()};maximum=relative_num=relative_den=0.
    with torch.no_grad():
        for lo in range(0,165400,256):
            hi=min(lo+256,165400);aid=inputs['affine_ids'][lo:hi];c=torch.from_numpy(inputs['condition'][lo:hi].copy()).double();gamma=torch.from_numpy(inputs['weight'][aid]).double();beta=torch.from_numpy(inputs['bias'][aid]).double()
            u=(c-beta)/gamma;h=F.gelu(F.linear(u,w['trunk.0.weight'],w['trunk.0.bias']));d=F.linear(h,w['direction.weight'],w['direction.bias'])
            if arm=='bounded_F':
                d=d-d.mean(-1,keepdim=True);r2=u.square().sum(-1,keepdim=True);d=d-(d*u).sum(-1,keepdim=True)/r2*u
                delta=.25*r2.sqrt()*d/(r2+d.square().sum(-1,keepdim=True)).sqrt();scale=r2.sqrt()/(u+delta).norm(dim=-1,keepdim=True)
                predicted=c+gamma*((scale-1)*u+scale*delta)
            else:predicted=c+gamma*.25*d
            value=torch.from_numpy(actual[lo:hi].copy()).double();error=value-predicted
            assert torch.allclose(value,predicted,**s['numerical_guard'])
            maximum=max(maximum,float(error.abs().max()));relative_num+=float(error.square().sum());relative_den+=float(value.square().sum())
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_FULL1654_F_HEAD_STATE_AND_ALL165400_FORWARD_AUDIT',study=G.ref(path),arm=arm,component=done['component'],updates=31014,exposures=496200,
        checkpoints=states,all165400_CPU64_forwards_verified=True,maximum_absolute_error=maximum,relative_L2_error=(relative_num/relative_den)**.5,
        guard=s['numerical_guard'],all20_own_affines_and_raw_OOF_bank_required=True,not_all31014_gradient_transitions_replayed=True,
        no_R80_payloads_opened=True,CUDA_initialized=False,seconds=time.monotonic()-start)
    G.write(out/'fit_CPU_audit.json',result);print(__import__('json').dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--arm',choices=G.ARMS,required=True);a=p.parse_args();run(a.study,a.arm)
