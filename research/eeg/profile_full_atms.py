"""S0 official full-model throughput probe; not a formal training result."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/atm_official_20261016')
sys.path[:0]=['/root/mced_representation_runtime_v1/lib/python3.12/site-packages',
    '/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages',
    '/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages',str(SOURCE)]
import torch
from torch import nn
import torch.nn.functional as F
from models.atms import ATMS


def main():
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE).decode().strip()
    assert commit=='efb6b1d5d7dd14bea44ee5a952b72a951fa6da54'
    tree=ast.parse((SOURCE/'encoder_utils.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_loss')
    scope=dict(nn=nn,F=F);exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE/'encoder_utils.py'),'exec'),scope)
    model=ATMS().cuda().train()
    x=torch.load('/root/mced_eeg_full_montage_cache_v3/sub-01_train_mean4.pt',weights_only=True,mmap=True)[:64].clone().cuda()
    y=torch.load('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2/train_image_embeds.pt',weights_only=True,mmap=True)[:64].clone().cuda()
    subjects=torch.ones(64,dtype=torch.long,device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    records=[]
    for step in range(25):
        torch.cuda.synchronize();tick=time.monotonic()
        output=model(x,subjects)
        loss=scope['_compute_loss'](output,y,model.logit_scale,model.loss_func,'generation',.9)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        records.append(dict(step=step+1,seconds=time.monotonic()-tick,loss=float(loss),gradient_norm_preclip=float(norm)))
    steady=sum(r['seconds'] for r in records[5:])/20
    active=sum(p.numel() for p in model.parameters() if p.grad is not None)
    report=dict(status='PROFILE_COMPLETED',formal_training=False,scope='S0 real64-query repeated-batch timing only; discard weights',
        source_commit=commit,source_file=str(SOURCE/'models/atms.py'),source_sha256=hashlib.sha256((SOURCE/'models/atms.py').read_bytes()).hexdigest(),
        seed=1729,parameters=sum(p.numel() for p in model.parameters()),gradient_active_parameters=active,
        input_shape=list(x.shape),output_shape=list(output.shape),precision='float32',batch=64,diagnostic_updates=25,
        steady_step_seconds=steady,images_per_second=64/steady,
        full_subject_16540_epoch_seconds=((16540+63)//64)*steady,
        ten_subject_100_encoder_epochs_hours=10*100*((16540+63)//64)*steady/3600,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),records=records,seconds=time.monotonic()-started)
    report.update(model_kwargs='ATMS() official per-subject defaults; num_subjects=2; joint_train=False',
                  source_role='official pinned develop checkout, not the local Codex history commit',
                  conditional_subject_id=1,
                  inactive_parameter_names={n:p.numel() for n,p in model.named_parameters() if p.grad is None})
    (ROOT/'audit/full_atms_official_default_profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
