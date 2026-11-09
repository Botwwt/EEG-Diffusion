"""One OLD-only microbenchmark AFTER original study/review/owner completion.

Real T EEG, existing frozen checkpoints and each host's own visual coordinates.
No optimizer, new training, query images, metric selection or source replacement.
Only this isolated benchmark enables encoder gradients to compare Jacobians.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse, fcntl, gc, hashlib, json, statistics, subprocess, sys, time, warnings
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);a=ap.parse_args()
policy=json.loads(Path(a.policy).read_bytes());root=Path(a.policy).parent
repo=Path(policy['repo']);sys.path[:0]=[str(root), str(repo), str(Path(policy['original_policy']).parent)]
import atm_continuous_margin_contract_20261106_v204 as C
c=C.setup(policy['original_policy']);q,K,tc=c['q'],c['K'],c['train']
assert c['out'].joinpath('completion.json').is_file()
assert K.read(c['out']/'completion.json')['status']=='COMPLETE_CONTINUOUS_MARGIN_TRAINING_AND_IMAGES'
assert K.read(Path(q['project_root'])/'terminal_review/completion.json')['status']=='PASS_ALL2400_IDENTITIES_PIXELS_NOISES_11200_RANK_VALUES_7200_LOWLEVEL_138_CIS'
launch=K.read(Path(policy['original_policy']).parent/'actual_launch.json')
assert not K.process(launch['PID'],launch['start_ticks'])['live']
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==tc['physical']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sys.path[:0]=tc['physical']['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from torch.nn import functional as F
import eeg_cross_host_gpu_runtime_20261106_v231 as A
from check_eeg_cross_host_gpu_runtime_CPU_20261106_v231 import serial

started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(1729)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory)
def guard():
    assert time.monotonic()-started<800
    assert torch.cuda.max_memory_allocated()<20<<30
    C.resources(c)

def compute_guard():
    # Keep disk/SHA walks outside ALL timed variants. GPU watchdog still bounds chunks.
    assert time.monotonic()-started<800
    assert torch.cuda.max_memory_allocated()<20<<30

def direct_batched(raw,visual,ids,groups,params):
    scores=F.normalize(raw,dim=-1)@visual.T
    ii=torch.arange(len(raw),device=raw.device);own=scores[ii,ids]
    negative=scores.clone();negative[ii,ids]=-torch.inf
    same=groups[ids,None]==groups[None,:]
    wi=negative.masked_fill(~same,-torch.inf).argmax(-1);gi=negative.argmax(-1)
    m=torch.cat((own-scores[ii,wi],own-scores[ii,gi]))
    rows=[]
    for lo in range(0,len(m),8):
        compute_guard();hi=min(lo+8,len(m))
        seeds=m.new_zeros((hi-lo,len(m)))
        seeds[torch.arange(hi-lo,device=raw.device),torch.arange(lo,hi,device=raw.device)]=1
        gs=torch.autograd.grad(m,params,grad_outputs=seeds,is_grads_batched=True,retain_graph=True)
        rows.append(torch.cat([g.reshape(hi-lo,-1) for g in gs],dim=1).double())
    return m.detach(),torch.cat(rows)

def run_encoder(label, net, inputs, visual, model_pins, prepare=None):
    net=net.cuda().eval();net.requires_grad_(True)
    if prepare is None:
        params=tuple(net.parameters())
    else:
        candidates=tuple(prepare(net));raw=net(inputs)
        grads=torch.autograd.grad(raw.sum(),candidates,allow_unused=True)
        params=tuple(p for p,g in zip(candidates,grads) if g is not None)
        del raw,grads
    original={n:v.detach().cpu().clone() for n,v in net.state_dict().items()}
    ids=torch.arange(len(inputs),device='cuda');groups=torch.arange(len(visual),device='cuda')//10
    visual=F.normalize(visual.cuda(),dim=-1)
    def calculate(mode):
        raw=net(inputs)
        if mode=='serial':return serial(raw,visual,ids,groups,params)[:2]
        if mode=='generic_batched8':return direct_batched(raw,visual,ids,groups,params)
        return A.factorized_margin_jacobian(raw,visual,ids,groups,params,
                                          chunk_size=1 if mode=='factored1' else 8,guard=compute_guard)[:2]
    modes=('serial','generic_batched8','factored1','factored8')
    guard();base_m,base_j=calculate('serial');errors={}
    for mode in modes[1:]:
        fast_m,fast_j=calculate(mode)
        error=float((base_j-fast_j).abs().max());relative=float((base_j-fast_j).norm()/base_j.norm().clamp_min(1e-30))
        agree=bool(torch.equal(base_m,fast_m))
        errors[mode]=dict(margins_equal=agree,J_max_abs_error=error,J_relative_l2=relative)
        assert agree and error<=1e-4 and relative<=2e-5,(label,mode,error,relative)
        del fast_m,fast_j
    del base_m,base_j;gc.collect();torch.cuda.empty_cache()
    timings={mode:[] for mode in modes}
    for order in (modes,tuple(reversed(modes)),modes[1:]+modes[:1]):
        for mode in order:
            guard();torch.cuda.synchronize();t=time.perf_counter();value=calculate(mode);torch.cuda.synchronize()
            timings[mode].append(time.perf_counter()-t);del value
    assert all(torch.equal(v.detach().cpu(),original[n]) for n,v in net.state_dict().items())
    for pin in model_pins:K.checked(pin)
    return dict(host=label,queries=len(inputs),gallery=len(visual),parameters=sum(p.numel() for p in params),
                constraint_rows=2*len(inputs),gallery_coordinate='own_native_visual',numerical_checks=errors,seconds=timings,
                median_ratios={mode:statistics.median(timings['serial'])/statistics.median(timings[mode]) for mode in modes},
                disk_and_SHA_guards_outside_all_timed_variants=True,
                weights_and_buffers_unchanged=True,model_pins=model_pins,
                full640_rows_or_QP_or_training_trajectory_tested=False)

results=[]
try:
    # Fixed first eight T rows; no D/V/R80 scores or data consulted.
    rows=tc['T'][:8];assert len(rows)==8 and len(tc['T'])==13230
    from models.atms import ATMS
    import atm_response_contrast_repair_core_20261105_v195 as H
    nn_state=torch.load(K.checked(q['models']['NN']),map_location='cpu',weights_only=True,mmap=True)['model']
    cc_state=torch.load(K.checked(q['models']['CC']),map_location='cpu',weights_only=True,mmap=True)['model']
    initial={n:nn_state[n] if n.startswith('proj_eeg.') else v for n,v in cc_state.items()}
    atms=ATMS();atms.load_state_dict(initial,strict=True)
    class ATM(torch.nn.Module):
        def __init__(self,net):super().__init__();self.net=net
        def forward(self,x):return self.net(x,torch.ones(len(x),device=x.device,dtype=torch.long))
    xall=torch.load(K.checked(tc['physical']['EEG']),map_location='cpu',weights_only=True,mmap=True)
    vall=torch.load(K.checked(tc['physical']['visual_target']),map_location='cpu',weights_only=True,mmap=True)
    atm=ATM(atms)
    results.append(run_encoder('ATMS_direct_encoder',atm,xall[rows].clone().cuda(),vall[tc['T']].clone(),
                               list(q['models'].values()),lambda net:H.prepare_student(net.net)))
    del atms,atm,xall,vall,initial,nn_state,cc_state;gc.collect();torch.cuda.empty_cache()
    from brainhive_native_models_v1 import classes, WORK
    contract=K.read(WORK/'data/component_input_contract_v1.json')
    subject=next(s for s in contract['subjects'] if s['subject']=='sub-01')
    source=next(s for s in subject['sources'] if s['role']=='full')
    model_pin={'path':source['checkpoint_path'],'sha256':source['checkpoint_sha256'],
               'bytes':Path(source['checkpoint_path']).stat().st_size}
    checkpoint=torch.load(K.checked(model_pin),map_location='cpu',weights_only=True,mmap=True)
    hive=classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1)
    hive.load_state_dict(checkpoint['model'],strict=True)
    binding=K.read(WORK/'data/actual_input_bindings.json')
    xb=torch.load(binding['subjects']['sub-01']['train_path'],map_location='cpu',weights_only=True,mmap=True)
    x=xb[rows][:,binding['selected_indices'],:].reshape(8,4250).clone().cuda()
    vpin={'path':contract['target_cache_path'],'sha256':contract['target_cache_sha256'],
          'bytes':Path(contract['target_cache_path']).stat().st_size}
    vf=torch.load(K.checked(vpin),map_location='cpu',weights_only=True,mmap=True)[tc['T']].clone()
    results.append(run_encoder('HIVE_BrainMLP_encoder',hive,x,vf,[model_pin,vpin]))
    guard()
    result=dict(status='COMPLETE_REAL_TWO_ENCODER_GPU_MICROBENCHMARK_ONLY',results=results,
                seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
                new_parameter_updates=0,new_images=0,condition_cache_full_UNet_benchmark=False,
                novelty_or_end_to_end_acceleration_established=False)
    K.write(root/'completion.json',result);print(json.dumps(result),flush=True)
except BaseException as exc:
    import traceback
    K.write(root/'failure.json',dict(status='FAILED_PRESERVED_NO_RETRY',results=results,error=repr(exc),traceback=traceback.format_exc()))
    raise
