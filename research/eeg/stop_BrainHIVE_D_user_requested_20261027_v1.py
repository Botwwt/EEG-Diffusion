"""User-requested D withdrawal: stop controller, retain exact atomic checkpoint."""
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'audit/continuous_iteration_20261027_v1'
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/paired_effect_20261025_v1')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_text())


def main():
    from run_BrainHIVE_trialset_paused_screen_20261027_v1 import proc,live,same,gpu
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'D_withdrawal_closed.json').exists()
    state_path=ROOT/'audit/efficacy_focus_20261026_v1/queue_state.json'
    state=read(state_path);assert state['status']=='D_ONLY_FULL_T_FULL_V_EFFICACY_RUNNING'
    assert state['current_stage']=='BrainHIVE_paired_effect_v1_D_full_T_exact_resume_20261025'
    receipt_path=ROOT/'audit'/(state['current_stage']+'_job.json');job=read(receipt_path)
    assert job['status']=='RUNNING' and job['wrapper_pid']==state['child_pid']
    controller,wrapper,trainer=[proc(pid) for pid in (state['pid'],job['wrapper_pid'],job['pid'])]
    assert str(ROOT/'queue_BrainHIVE_D_efficacy_focus_20261026_v1.py') in controller['argv']
    assert trainer['argv']==job['command'] and trainer['ppid']==wrapper['pid'] and wrapper['ppid']==controller['pid']
    assert '--arm' in trainer['argv'] and trainer['argv'][trainer['argv'].index('--arm')+1]=='D'
    assert gpu()=={trainer['pid']}
    bound=dict(controller=controller,wrapper=wrapper,trainer=trainer)
    (OUT/'D_stop_request.json').write_text(json.dumps(dict(user_instruction='先停止运行d吧',bound_processes=bound,queue_before=state),indent=2)+'\n')
    same(controller);os.kill(controller['pid'],signal.SIGTERM)
    deadline=time.monotonic()+55
    while any(live(proc(p['pid'])) for p in bound.values()):
        assert time.monotonic()<deadline,'Controller cleanup not closed; preserve unresolved evidence'
        time.sleep(.25)
    assert not gpu()
    for path in (ROOT/'audit/GPU_JOB.lock',ROOT/'audit/functional_full_lifecycle_20261022_v1/lifecycle_controller.lock'):
        with path.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
    import torch
    from train_BrainHIVE_bounded_component_v1 import seed
    torch.set_num_threads(2)
    train=DATA/'arms/D/train';payload=(train/'resume.pt').read_bytes()
    saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    progress=read(train/'progress.json')
    assert progress['status']=='PAIRED_TRAIN_FINITE_STOP_EXACT_RESUME'
    assert saved['updates']==progress['updates'] and saved['cursor']==saved['seen']==progress['cursor']
    assert tuple(saved['coverage'].shape)==(3,132300) and int(saved['coverage'].sum())==saved['seen']
    for i in range(3):
        order=torch.randperm(132300,generator=torch.Generator().manual_seed(seed('pass'+str(i))))
        expected=torch.zeros(132300,dtype=torch.bool)
        expected[order[:max(0,min(132300,saved['cursor']-i*132300))]]=True
        assert torch.equal(expected,saved['coverage'][i])
    assert sum(v.numel() for v in saved['model'].values())==525568 and saved['optimizer']['state']
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    assert all(k in saved for k in ('torch_rng','cuda_rng','noise_rng')) and len(saved['cuda_rng'])==1
    assert saved['provenance_sha256']==sha(train/'input_provenance.json')
    logs=(train/'training.jsonl').read_bytes()
    assert len(logs)==saved['log_bytes'] and hashlib.sha256(logs).hexdigest()==saved['log_sha256']
    closed=read(receipt_path)
    assert closed['status']=='FAILED_OR_JOB_TIMEOUT_RESUMABLE' and closed['exit_code']!=0
    ledger=[json.loads(s) for s in (ROOT/'audit/compute_ledger.jsonl').read_text().splitlines()]
    assert any(x==closed for x in ledger)
    assert not (train/'completion.json').exists() and not (DATA/'arms/D/images').exists()
    assert not torch.cuda.is_initialized()
    now=datetime.datetime.now(datetime.timezone.utc)
    result=dict(status='USER_STOPPED_D_EXACT_CHECKPOINT_AND_GPU_QUIESCENCE_VERIFIED',
                UTC=now.isoformat(),America_Los_Angeles=now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),
                updates=saved['updates'],primary_exposures=saved['seen'],checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
                coverage_optimizer_RNG_log_prefix_CPU_verified=True,bound_processes=bound,
                all_processes_dead=True,GPU_empty=True,both_locks_free=True,CUDA_initialized=False,
                wrapper_closed=closed,controller_terminal_state=read(state_path),
                automatic_D_resume_authorized=False,scientific_negative_result=False)
    (OUT/'D_withdrawal_closed.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','America_Los_Angeles','updates','primary_exposures','all_processes_dead','GPU_empty','both_locks_free')}),flush=True)


if __name__=='__main__':main()
