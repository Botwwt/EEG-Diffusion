"""Actual fullfit state/coverage/log prefix and all165400 explicit CPU64 forwards."""
import argparse
import hashlib
from pathlib import Path
import time
import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F
E=F.E;K=F.K


def prefix_check(pin):
    path=Path(pin['path']);assert path.is_file() and not path.is_symlink() and path.stat().st_size>=pin['bytes']
    h=hashlib.sha256();remaining=pin['bytes']
    with path.open('rb') as stream:
        while remaining:
            value=stream.read(min(1<<20,remaining));assert value;h.update(value);remaining-=len(value)
    assert h.hexdigest()==pin['sha256']


def run(protocol,latest_only=False):
    started=time.monotonic();s=F.setup_fit(protocol)
    import numpy as np
    import torch
    import torch.nn.functional as fn
    import brainhive_exposed_fullfit_component_inputs_CPU_20261101_v1 as I
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    root=Path(s['output_root'])/'ordinary_full1654'
    if latest_only:
        value,pointer=F.load_checkpoint(root,torch);order=E.orders(torch)
        assert torch.equal(value['coverage'],F.expected_coverage(value['updates'],order,torch))
        assert value['provenance']==K.ref(root/'input_provenance.json')
        assert value['orders_sha256']==[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in order]
        for pin in value['log_prefixes']:prefix_check(pin)
        assert len(value['optimizer']['state'])==4
        for state in value['optimizer']['state'].values():
            assert int(state['step'])==value['updates']
            assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all() and (state['exp_avg_sq']>=0).all()
        result=dict(status='PASS_LATEST_ACTUAL_ORDINARY_CHECKPOINT_CPU_STATE',checkpoint=pointer,
                    updates=value['updates'],exposures=value['exposures'],CUDA_initialized=False)
        K.write(root/'segments'/f"{value['segment']:03d}"/'latest_CPU_audit.json',result)
        print(__import__('json').dumps(result));return
    done=K.read(root/'fit_completion.json')
    assert done['status']=='COMPLETE_FULL1654_ORIGINAL_ORDINARY_THREE_PASSES_FIXED_FINAL'
    assert (done['updates'],done['exposures'],done['seed'],done['parameters'])==(31014,496200,1729,525568)
    assert done['strength']==1. and done['radius']==.25 and done['old_Dfit_calibration_performed'] is False
    final=torch.load(K.check(done['component']),weights_only=True,map_location='cpu')
    assert final['arm']=='ordinary_full1654' and final['method']==E.METHOD and final['protocol']==K.ref(protocol)
    provenance=K.read(root/'input_provenance.json');order=E.orders(torch)
    hashes=[hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in order]
    assert hashes==provenance['orders_sha256']
    latest,pointer=F.load_checkpoint(root,torch)
    assert pointer==done['latest_checkpoint'] and latest['updates']==31014
    history=latest['checkpoint_history']+[pointer]
    paths=[item['checkpoint']['path'] for item in history];assert len(paths)==len(set(paths))
    states=[];last_updates=-1
    for item in history:
        value=torch.load(K.check(item['checkpoint']),weights_only=True,map_location='cpu')
        assert value['updates']>last_updates;last_updates=value['updates']
        assert value['orders_sha256']==hashes and value['provenance']==K.ref(root/'input_provenance.json')
        assert value['exposures']==F.prefix_counts(value['updates'])[2]
        assert torch.equal(value['coverage'],F.expected_coverage(value['updates'],order,torch))
        for pin in value['log_prefixes']:prefix_check(pin)
        opt=value['optimizer'];assert len(opt['state'])==4 and len(opt['param_groups'])==1
        group=opt['param_groups'][0]
        assert (group['lr'],group['weight_decay'],tuple(group['betas']),group['eps'])==(1e-4,.01,(.9,.999),1e-8)
        for index,parameter in zip(group['params'],value['model'].values()):
            parts=opt['state'][index];assert int(parts['step'])==value['updates']
            for name in ('exp_avg','exp_avg_sq'):assert parts[name].shape==parameter.shape and torch.isfinite(parts[name]).all()
            assert (parts['exp_avg_sq']>=0).all()
        states.append(dict(updates=value['updates'],exposures=value['exposures'],segment=value['segment'],
                           all4_Adam_counters_moments_coverage_and_log_prefixes_verified=True))
        del value
    assert len(list((root/'checkpoints').glob('step-*/commit.json')))==len(history)
    assert all(torch.equal(latest['model'][key],value) for key,value in final['model'].items())
    assert latest['coverage'].all()
    inputs=I.load_verified(s['input_bank'],s['asset_protocol'])
    actual=np.load(K.check(done['actual_train_outputs']),mmap_mode='r',allow_pickle=False)
    assert actual.shape==(165400,1024) and actual.dtype==np.float32
    state={key:value.double() for key,value in final['model'].items()}
    maximum=numerator=denominator=0.
    with torch.no_grad():
        for lo in range(0,165400,256):
            hi=min(lo+256,165400);aid=inputs['affine_ids'][lo:hi]
            c=torch.from_numpy(np.array(inputs['condition'][lo:hi],copy=True)).double()
            weight=torch.from_numpy(inputs['weight'][aid]).double();bias=torch.from_numpy(inputs['bias'][aid]).double()
            u=(c-bias)/weight;h=fn.gelu(fn.linear(u,state['trunk.0.weight'],state['trunk.0.bias']))
            d=fn.linear(h,state['direction.weight'],state['direction.bias']);d=d-d.mean(-1,keepdim=True)
            r2=u.square().sum(-1,keepdim=True);d=d-(d*u).sum(-1,keepdim=True)/r2*u
            delta=.25*r2.sqrt()*d/(r2+d.square().sum(-1,keepdim=True)).sqrt();scale=r2.sqrt()/(u+delta).norm(dim=-1,keepdim=True)
            prediction=c+weight*((scale-1)*u+scale*delta)
            value=torch.from_numpy(np.array(actual[lo:hi],copy=True)).double();error=value-prediction
            assert torch.allclose(value,prediction,**s['numerical_guard'])
            maximum=max(maximum,float(error.abs().max()));numerator+=float(error.square().sum());denominator+=float(value.square().sum())
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_FULL1654_ORIGINAL_ORDINARY_FINAL_STATE_AND_ALL165400_CPU_FORWARD_AUDIT',
                protocol=K.ref(protocol),component=done['component'],updates=31014,exposures=496200,
                checkpoint_states=states,all165400_explicit_CPU64_forwards_verified=True,
                maximum_absolute_error=maximum,relative_L2_error=(numerator/denominator)**.5,guard=s['numerical_guard'],
                not_all31014_gradient_transitions_replayed=True,future_GPU_bit_exactness_NOT_proved=True,
                no_R80_payloads_opened=True,seconds=time.monotonic()-started,CUDA_initialized=False)
    K.write(root/'fit_CPU_audit.json',result)
    print(__import__('json').dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--latest-only',action='store_true')
    args=parser.parse_args();run(args.protocol,args.latest_only)
