"""Finite real-T, clinical-backbone engineering profile; temporary updates only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,hashlib,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from eeg_labram_pretrained_20261028_v1 import build_encoder,transfer_clinical
from eeg_labram_interface_20261028_v1 import global_and_patch_conditions

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def checked(item):
    path=Path(item['path']);assert sha(path)==item['sha256'],str(path)
    if 'bytes' in item:assert path.stat().st_size==item['bytes']
    return path
def peak_CPU():
    return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def objective(c,target,logit_scale):
    similarity=logit_scale.float().exp().clamp(max=100)*F.normalize(c.float(),dim=-1)@F.normalize(target.float(),dim=-1).T
    labels=torch.arange(len(c),device=c.device)
    nce=(F.cross_entropy(similarity,labels)+F.cross_entropy(similarity.T,labels))/2
    mse=F.mse_loss(c.float(),target.float())
    return .8*nce+.2*mse,dict(NCE=float(nce.detach()),MSE=float(mse.detach()))

def run(spec_path):
    start=time.monotonic();spec=json.loads(spec_path.read_bytes());out=Path(spec['output']);assert not out.exists();out.mkdir()
    phase='REGISTERED_INPUTS'
    try:
        for item in spec['source_pins']:checked(item)
        completion_path=checked(spec['input_completion']);inputs=json.loads(completion_path.read_bytes())
        assert inputs['status']=='COMPLETE_SUB01_T13230_TRUE_ROLE_NORMALIZATION_AND_200HZ_EEG_INPUTS' and inputs['T_stimuli']==13230
        for role in ('full','oof0','oof1'):
            checked(inputs['roles'][role]['EEG']);checked(inputs['roles'][role]['normalization'])
        target_path=checked(inputs['visual_targets']);checked(inputs['identities']);checked(inputs['original_array_rows'])
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        assert gpu==spec['GPU_UUID']
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not active,active
        ordinary=Path('/root/brainhive_reproduction_20261020_v1/components/bounded_v1/ordinary/component_final.pt')
        assert sha(ordinary)=='a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'
        torch.set_num_threads(2);torch.manual_seed(1729)
        model=build_encoder(spec['author_source'],spec['vendor'],seed=1729)
        transfer=transfer_clinical(model,checked(spec['clinical_checkpoint']))
        from models.labram import get_input_chans
        stats=json.loads(Path(inputs['roles']['full']['normalization']['path']).read_bytes())
        positions=get_input_chans([x.upper() for x in stats['channel_names']]);assert len(positions)==64
        x=np.load(inputs['roles']['full']['EEG']['path'],mmap_mode='r',allow_pickle=False)
        y=np.load(target_path,mmap_mode='r',allow_pickle=False)
        assert x.shape==(13230,63,1,200) and y.shape==(13230,1024)
        indices=np.arange(128);eeg=torch.tensor(x[indices],device='cuda');target=torch.tensor(y[indices],device='cuda')
        model=model.cuda();torch.cuda.reset_peak_memory_stats();base_CPU=peak_CPU();checks={};timings={}
        def guard():
            assert torch.cuda.max_memory_allocated()<20<<30 and peak_CPU()-base_CPU<10<<30
            assert time.monotonic()-start<800
        phase='REAL_T_INTERFACE'
        model.eval()
        with torch.no_grad():
            c,memory=global_and_patch_conditions(model,eeg[:4],input_chans=positions)
            reference=model(eeg[:4],input_chans=positions)[0]
        checks['real_clinical_global_exact_author_forward']=torch.equal(c,reference)
        checks['real_T_63_local_memories']=memory.shape==(4,63,1024)
        checks['same_global_preserved']=torch.allclose(memory.mean(1),c,atol=1e-6,rtol=1e-5)
        checks['local_deviation_nonzero_finite']=bool(torch.isfinite(memory).all() and (memory-c[:,None]).norm()>0)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05)
        phase='TEMPORARY_REAL_T_UPDATES';events=[];torch.cuda.synchronize();began=time.monotonic()
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                c=model(eeg,input_chans=positions)[0]
                loss,parts=objective(c,target,model.logit_scale)
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);assert torch.isfinite(norm)
            checks['temporal_conv_gradient_nonzero']=bool(next(model.patch_embed.parameters()).grad.norm()>0)
            optimizer.step()
            with torch.no_grad():model.logit_scale.clamp_(max=float(np.log(100)))
            guard();events.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm),**parts))
        torch.cuda.synchronize();timings['three_b128_update_seconds']=time.monotonic()-began
        model.load_state_dict(initial,strict=True);optimizer=None;model.eval()
        checks['temporary_updates_fully_reset']=all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in initial.items())
        assert all(checks.values()),checks;guard()
        write(out/'completion.json',dict(status='PASS_REAL_T_CLINICAL_FULL_ENCODER_GPU_ENGINEERING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec_sha256=sha(spec_path),input_completion=spec['input_completion'],transfer=transfer,checks=checks,temporary_updates=3,formal_trained_models=0,new_images=0,batch=128,events=events,timings=timings,peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),extra_CPU_peak=peak_CPU()-base_CPU,seconds=time.monotonic()-start,GPU_UUID=gpu,ordinary_unchanged=sha(ordinary)=='a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'))
        print(json.dumps(json.loads((out/'completion.json').read_bytes())))
    except BaseException as error:
        write(out/'failure.json',dict(status='EEG_REAL_T_GPU_ENGINEERING_FAILURE_PRESERVED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
