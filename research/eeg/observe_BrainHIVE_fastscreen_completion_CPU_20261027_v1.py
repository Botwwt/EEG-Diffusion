"""Read-only science-state and resumed-D audit; writes only its new receipt."""
from pathlib import Path
import datetime
import hashlib
import io
import json
import subprocess
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
A=ROOT/'audit/trialset_fastscreen_20261027_v1'
B=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/trialset_fastscreen_20261027_v1')
D=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/paired_effect_20261025_v1/arms/D/train')


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    import torch
    import brainhive_paired_study_contract_20261025_v1 as C
    from train_BrainHIVE_bounded_component_v1 import seed
    from run_BrainHIVE_trialset_paused_screen_20261027_v1 import proc
    torch.set_num_threads(2)
    plan=read(A/'execution_plan_runtimefix.json')
    for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h
    assert read(A/'pause_v2/results.json')['status']=='FASTSCREEN_FIT_IMAGES_SCORE_COMPLETE'
    proof=read(A/'pause_v2/proof.json')
    identities={}
    for role,bound in proof['processes'].items():
        p=proc(bound['pid'])
        assert p and p['start_ticks']==bound['start_ticks'] and p['argv']==bound['argv'] and p['state'] not in ('T','t','Z')
        identities[role]=p
    q=read(ROOT/'audit/efficacy_focus_20261026_v1/queue_state.json')
    assert q['status']=='D_ONLY_FULL_T_FULL_V_EFFICACY_RUNNING' and not q.get('error')
    assert q['pid']==identities['controller']['pid'] and q['child_pid']==identities['wrapper']['pid']
    progress=read(D/'progress.json')
    checkpoint_bytes=(D/'resume.pt').read_bytes()
    checkpoint_sha=hashlib.sha256(checkpoint_bytes).hexdigest()
    saved=torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=False)
    initial=read(A/'D_checkpoint_before_screen.json')
    assert saved['updates']>initial['updates'] and saved['updates']<=24807
    assert saved['cursor']==saved['seen'] and saved['seen']>initial['primary_exposures']
    assert tuple(saved['coverage'].shape)==(3,132300) and int(saved['coverage'].sum())==saved['seen']
    provenance=read(D/'input_provenance.json')
    assert saved['provenance_sha256']==sha(D/'input_provenance.json')
    for i in range(3):
        order=torch.randperm(132300,generator=torch.Generator().manual_seed(seed('pass'+str(i))))
        assert hashlib.sha256(order.numpy().tobytes()).hexdigest()==provenance['orders_sha256'][i]
        expected=torch.zeros(132300,dtype=torch.bool)
        expected[order[:max(0,min(132300,saved['cursor']-i*132300))]]=True
        assert torch.equal(expected,saved['coverage'][i])
    assert sum(v.numel() for v in saved['model'].values())==525568 and saved['optimizer']['state']
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for item in saved['optimizer']['state'].values() for v in item.values())
    assert all(k in saved for k in ('torch_rng','cuda_rng','noise_rng')) and len(saved['cuda_rng'])==1
    payload=(D/'training.jsonl').read_bytes()[:saved['log_bytes']]
    assert hashlib.sha256(payload).hexdigest()==saved['log_sha256']
    config=read(ROOT/'configs/paired_effect_20261025_v1/study_review_v3.json')
    core=C.validate(config)
    assert core=='8ccb781afc2a4afe23d951bf0057387246ebd42917fcc2912b097634357c7f18'
    ledger=[json.loads(x) for x in (ROOT/'audit/compute_ledger.jsonl').read_text().splitlines()]
    receipts=[]
    for j in plan['jobs']:
        r=read(ROOT/'audit'/(j['name']+'_job.json'))
        assert r['status']=='COMPLETED' and r['exit_code']==0 and any(x==r for x in ledger)
        assert r['peak_total_GPU_used_bytes']<=30<<30
        assert not (proc(r['pid']) and proc(r['pid'])['state']!='Z')
        receipts.append(r)
    fit=read(B/'fit_runtimefix/summary.json');gen=read(B/'images_runtimefix/completion.json');score=read(B/'score_runtimefix/summary.json')
    assert fit['rows_per_arm']==13230 and gen['images']==160 and score['generated_images']==160
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    assert {int(line.split(',')[0]) for line in gpu.splitlines()}=={identities['trainer']['pid']}
    now=datetime.datetime.now(datetime.timezone.utc)
    result=dict(status='FASTSCREEN_COMPLETE_AND_ORIGINAL_D_ACTUALLY_ADVANCING',UTC=now.isoformat(),
                America_Los_Angeles=now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),
                D_processes=identities,D_progress=progress,
                D_CPU_checkpoint=dict(updates=saved['updates'],primary_exposures=saved['seen'],
                    checkpoint_sha256=checkpoint_sha,coverage_prefix_optimizer_rng_log_verified=True),
                original_scientific_core=core,new_jobs=receipts,
                successful_screen_GPU_job_charged_seconds=sum(r['charged_seconds'] for r in receipts),
                fit_seconds=fit['elapsed_seconds'],generation_seconds=gen['seconds'],score_seconds=score['seconds'],
                peak_combined_GPU_used_bytes=max(r['peak_total_GPU_used_bytes'] for r in receipts),
                fit_means=fit['arms'],fit_paired=fit['paired_vs_mean'],image_means=score['means'],
                image_paired=score['paired_each_mean_minus_mean'],
                GPU_processes=gpu,CUDA_initialized=torch.cuda.is_initialized(),
                claims=dict(development_screen_only=True,full_V=False,SOTA=False,trialset_general_rejection=False))
    assert not result['CUDA_initialized']
    p=A/'actual_completion_and_D_resumed.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','America_Los_Angeles','D_CPU_checkpoint','successful_screen_GPU_job_charged_seconds','fit_seconds','generation_seconds','score_seconds','peak_combined_GPU_used_bytes')},indent=2))


if __name__=='__main__':main()
