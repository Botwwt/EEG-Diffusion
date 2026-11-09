"""D37 real T/ordinary/native-KV engineering, without fitting or UNet execution."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import brainhive_temporal_conditional_attention_20261028_v1 as M

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(x):
    p=Path(x['path']);assert p.is_file() and sha(p)==x['sha256']
    if 'bytes' in x:assert p.stat().st_size==x['bytes']
    return p
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def save(p,x):
    with p.open('xb') as f:np.save(f,x,allow_pickle=False)

def run(spec_path):
    start=time.monotonic();base=rss();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    spec=read(spec_path);assert spec['status']=='D37_TEMPORAL_CONDITIONAL_ATTENTION_CPU_ENGINEERING_LOCKED'
    for item in spec['source_pins']:check(item)
    folder=Path(spec['output']);assert not folder.exists();folder.mkdir(parents=True)
    def budget():assert time.monotonic()-start<850 and rss()-base<10<<30
    sys.path[:0]=['/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages','/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages',
                 '/root/mced_representation_runtime_v1/lib/python3.12/site-packages','/root/mced_eeg_brainhive_parquet_packages_v1']
    from safetensors.torch import load_file
    from brainhive_reversible_eeg_20261028_v1 import author_brain_class
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    R=Path(__file__).resolve().parent
    d33=read(check(spec['D33_spec']));physical=np.load(check(d33['T_trials']),mmap_mode='r',allow_pickle=False)
    index=[json.loads(x) for x in check(d33['T_index']).read_text().splitlines()]
    rows=np.linspace(0,13229,32,dtype=np.int64);assert len(set(rows))==32
    assert all(index[int(i)]['row']==int(i) and index[int(i)]['partition']=='T' for i in rows)
    x=torch.tensor(np.asarray(physical[rows]).mean(axis=1,dtype=np.float32).reshape(32,4250))
    parent=torch.load(check(spec['parent_checkpoint']),weights_only=True,map_location='cpu');state=parent['model']
    brain=author_brain_class(R/'reproduction_sources/BrainHIVE_f06db69f/main/models_brain.py',R/'reproduction_sources/BrainHIVE_f06db69f/main/layers.py')(4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);brain.load_state_dict(state,strict=True)
    with torch.no_grad():native=brain(x);computed,tangent,note=M.temporal_tangents(x,state)
    forward_error=float((native-computed).abs().max());assert forward_error<2e-5
    state64={k:v.double() for k,v in state.items()};x64=x.double()
    c64,t64,n64=M.temporal_tangents(x64,state64)
    def f(h):
        z=h+F.linear(F.gelu(F.linear(h,state64['block.layers.0.ff.w1.weight'])),state64['block.layers.0.ff.w2.weight'])
        return F.layer_norm(z,(1024,),state64['block.layers.0.norm.weight'],state64['block.layers.0.norm.bias'],1e-6)
    jvp_error=0.
    for t in range(5):
        _,truth=torch.autograd.functional.jvp(f,n64['hidden'],n64['hidden_residuals'][:,t]);jvp_error=max(jvp_error,float((truth-t64[:,t]).abs().max()))
    assert jvp_error<1e-10 and float(t64.sum(1).abs().max())<1e-10
    # Deliberately synthetic W-kernel perturbation: not a physiological EEG trial.
    torch.manual_seed(1729);Q,_=torch.linalg.qr(state64['proj_in.weight'].T,mode='reduced')
    v=torch.randn(8,4250,dtype=torch.float64);v=v-(v@Q)@Q.T;v*=.1*x64[:8].norm(dim=1,keepdim=True)/v.norm(dim=1,keepdim=True)
    changed,changed_time,_=M.temporal_tangents(x64[:8]+v,state64)
    native_null_change=float((changed-c64[:8]).norm()/c64[:8].norm())
    temporal_null_change=float((changed_time-t64[:8]).norm()/t64[:8].norm())
    assert native_null_change<1e-10 and temporal_null_change>1e-5
    frozen=read(check(spec['ordinary_frozen_spec']));trained=torch.load(check(frozen['checkpoint']),map_location='cpu',weights_only=True)
    assert trained['calibration']['selected_strength']==frozen['selected_strength']==1
    assert sha(check(frozen['source']))=='2b93c0644bde268ae876f0e2b722106852b25910fedc1cffc97820e96cadf5d7'
    ordinary=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);ordinary.load_state_dict(trained['model'],strict=True)
    gamma=state['block.layers.0.norm.weight'][None].expand(32,-1);beta=state['block.layers.0.norm.bias'][None].expand(32,-1)
    with torch.no_grad():corrected=ordinary(native,gamma,beta,gate_mode='fixed')[0]
    assets=read(check(spec['native_asset_verification']))['files'];W=Path('/root/brainhive_reproduction_20261020_v1')
    def asset(kind):
        path=W/'assets/prior_H14'/kind/'diffusion_pytorch_model.safetensors';assert sha(path)==assets[str(path)]['sha256'];return load_file(str(path))
    fusion=asset('fusion_encoder');projection=asset('proj');ip=asset('attn_adapter')
    def prior(condition):
        g=state['block.layers.0.norm.weight'][None].expand(len(condition),-1)
        b=state['block.layers.0.norm.bias'][None].expand(len(condition),-1)
        ordinary_c=ordinary(condition,g,b,gate_mode='fixed')[0]
        adapted=ordinary_c+F.linear(F.gelu(F.linear(ordinary_c,fusion['adapter.ff.w1.weight'])),fusion['adapter.ff.w2.weight'])
        return F.layer_norm(F.linear(adapted,projection['proj.weight'],projection['proj.bias']).reshape(len(condition),4,2048),
                            (2048,),projection['norm.weight'],projection['norm.bias'],1e-5)
    tokens,detail,bound=M.frozen_prior_temporal_tokens(native.detach(),tangent.detach(),prior)
    zero=torch.zeros_like(detail);zkey=zvalue=zero;detail_key=detail_value=detail
    relative=detail.square().sum(-1).mean(-1).sqrt()/tokens.norm(dim=-1)
    assert relative.max()<=.250002 and detail.sum(2).abs().max()<2e-4
    _,changed_prior_detail,_=M.frozen_prior_temporal_tokens(native[:8].detach(),changed_time.float(),prior)
    prior_temporal_null_change=float((changed_prior_detail-detail[:8]).norm()/detail[:8].norm())
    assert prior_temporal_null_change>1e-5
    pairs=sorted(key for key in ip if key.endswith('.to_k_ip.weight'));assert len(pairs)==70 and len(ip)==140
    records=[]
    for number,key_name in enumerate(pairs):
        wk=ip[key_name];wv=ip[key_name.replace('.to_k_ip.weight','.to_v_ip.weight')]
        assert wk.shape==wv.shape and wk.shape[1]==2048
        dim=wk.shape[0];assert dim%64==0;heads=dim//64
        def pack(value):return value.reshape(32,4,5,heads,64).permute(0,3,1,2,4)
        generator=torch.Generator().manual_seed(1800+number)
        query=torch.randn((32,heads,8,64),generator=generator)
        with torch.no_grad():
            base_key=F.linear(tokens,wk).reshape(32,4,heads,64).permute(0,2,1,3)
            zk=pack(F.linear(zkey,wk));zv=pack(F.linear(zvalue,wv))
            zero_c=M.attention_delta(query,base_key,zk,zv)
            zero_f=M.attention_delta(query,base_key,zk,zv,mode='flat_residual')
            assert zero_c.count_nonzero()==0 and zero_f.count_nonzero()==0
            dk=pack(F.linear(detail_key,wk));dv=pack(F.linear(detail_value,wv))
            delta,p=M.attention_delta(query,base_key,dk,dv,return_probabilities=True)
            flat,pflat=M.attention_delta(query,base_key,dk,dv,mode='flat_residual',return_probabilities=True)
            root=(query@base_key.transpose(-1,-2)/8).softmax(-1)
            marginal_error=float((p['joint'].sum(-1)-root).abs().max());assert marginal_error<2e-6
            record=dict(layer=key_name,hidden_size=dim,heads=heads,synthetic_query_positions=8,
                        zero_value_native_return_error=0.,conditional_marginal_max_error=marginal_error,
                        flat_residual_marginal_RMS=float((pflat['marginal']-root).square().mean().sqrt()),
                        conditional_native_tangent_delta_RMS=float(delta.square().mean().sqrt()),flat_native_tangent_delta_RMS=float(flat.square().mean().sqrt()))
        records.append(record);budget()
    save(folder/'uniform32_T_rows.npy',rows);save(folder/'native_conditions32.npy',native.numpy());save(folder/'ordinary_conditions32.npy',corrected.numpy())
    save(folder/'native_temporal_tangents32.npy',tangent.numpy());save(folder/'ordinary_native_tokens32.npy',tokens.numpy())
    save(folder/'frozen_prior_temporal_token_tangents32.npy',detail.numpy())
    with (folder/'all70_native_IP_checks.json').open('x') as f:json.dump(records,f,indent=2);f.write('\n')
    value=dict(status='D37_REAL_T_AND_ALL70_NATIVE_IP_CPU_ENGINEERING_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(spec_path),
        engineering_T_rows=32,not_original_oracle_T32=True,subject='sub-01',parent_role='full',FIT_only_not_OOF=True,
        native_forward_max_error=forward_error,all160_real_output_tangent_JVP_max_error=jvp_error,
        synthetic_Wkernel_native_relative_change=native_null_change,synthetic_Wkernel_temporal_relative_change=temporal_null_change,
        synthetic_kernel_challenge_is_not_physiological_EEG=True,original_ordinary_SHA=sha(check(frozen['checkpoint'])),old_ordinary_strength=1,
        native_IP_layers_checked=70,all_native_KV_tensors_checked=140,queries_are_synthetic_not_UNet_trajectory=True,
        ordinary_A_P_and_IP_CPU_FP32_interface=True,new_readout_parameters=0,additional_learned_parameters=0,
        actual_frozen_ordinary_A_P_JVPs=160,maximum_token_RMS_relative_trust_ratio=float(relative.max()),fixed_trust_radius=.25,
        synthetic_Wkernel_frozen_prior_temporal_relative_change=prior_temporal_null_change,synthetic_readout_fixture_used=False,
        host_UNet_or_GPU_or_pixel_profile_completed=False,
        attention_operator_is_not_bound_to_UNet_yet=True,new_neural_models_fit=0,SGD_updates=0,new_GPU_jobs=0,new_images=0,
        original_IP_file_sha256=sha(W/'assets/prior_H14/attn_adapter/diffusion_pytorch_model.safetensors'),
        original_ordinary_unchanged=True,no_automatic_scientific_training_admission=True,seconds=time.monotonic()-start,peak_extra_CPU_bytes=rss()-base,
        files=[ref(p) for p in folder.iterdir() if p.is_file()])
    assert not torch.cuda.is_initialized();budget()
    with (folder/'completion.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();run(a.spec)
