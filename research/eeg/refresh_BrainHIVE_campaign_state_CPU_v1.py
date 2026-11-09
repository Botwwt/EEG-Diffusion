"""Refresh mutable campaign status only from live processes and native receipts."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')


def main():
    state_path=ROOT/'campaign_state.json'
    snapshot=ROOT/'audit/before_BrainHIVE_unlimited_20261020_v1/campaign_state.json'
    if not snapshot.exists():shutil.copy2(state_path,snapshot)
    state=json.loads(state_path.read_text())
    authorization=json.loads((ROOT/'configs/resource_authorization_20261020_v3.json').read_text())
    assert authorization['total_GPU_job_hours_cap'] is None
    launch=json.loads((WORK/'audit/native_benchmark_v2_launch.json').read_text())
    queue=json.loads((WORK/'audit/native_benchmark_queue_v2.json').read_text())
    proc=Path('/proc')/str(launch['pid'])/'cmdline'
    queue_live=proc.exists() and b'queue_BrainHIVE_native_benchmark_v2.py' in proc.read_bytes()
    current_job=ROOT/'audit'/((queue.get('job') or '')+'_job.json')
    job=json.loads(current_job.read_text()) if current_job.exists() else {}
    proc=Path('/proc')/str(job.get('pid',0))/'cmdline'
    GPU_live=proc.exists() and any(x in proc.read_bytes() for x in [b'generate_BrainHIVE_native_images_v2.py',b'extract_image_features.py'])
    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    state.update(stage='BRAINHIVE_NATIVE_REFERENCE_RUNNING' if queue_live else queue['status'],
        primary_parent='Brain-HIVE H14+B32+VAE f06db69f',last_update_utc=utc,
        resource_budget='User removed total time cap; existing single GPU, no purchase/expansion, finite jobs',
        resource_authority='configs/resource_authorization_20261020_v3.json',
        confirmed_total_gpu_hours=None,remaining_GPU_cap_hours=None,remaining_authorized_hours=None,
        active_controller_pid=launch['pid'] if queue_live else None,
        active_state_path=str(WORK/'audit/native_benchmark_queue_v2.json'),
        active_GPU_pid=job.get('pid') if GPU_live else None,
        author_decoder_status='BrainHIVE actual native images; old ATM decoder deferred',
        last_goal_turn_classification='progress',previous_goal_turn_classification='progress',
        scientific_goal_complete=False,new_EEG_image_gain_established=False,
        component_implemented=False,candidate_limit=12,candidate_limit_not_reset=True,
        verified_wait=[dict(UTC=utc,queue_pid=launch['pid'],queue_process_live=queue_live,
                           GPU_pid=job.get('pid'),GPU_process_live=GPU_live,job=queue.get('job'))],
        BrainHIVE_parent_models_CPU_verified=40,BrainHIVE_oracle_images_completed=32,
        BrainHIVE_native_images={arm:len(p.read_text().splitlines()) if p.exists() else 0
            for arm in ['correct','permuted','missing']
            for p in [WORK/'images'/('native_R80_'+arm+'_v1')/'inference_manifest.jsonl']})
    temporary=state_path.with_suffix('.json.pending')
    temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
    ledger=ROOT/'experiment_ledger.jsonl';event='BrainHIVE_complete40_parent_and_oracle32_native_running_v1'
    entries=[json.loads(x) for x in ledger.read_text().splitlines()]
    if not any(x.get('event_id')==event for x in entries):
        with ledger.open('a') as file:file.write(json.dumps(dict(event_id=event,UTC=utc,
            type='actual_pinned_parent_and_oracle_completion_with_live_native_generation',
            source_commit='f06db69fcc4a029439e369c5c3dba6099e57560e',training_seed=1729,
            subjects=10,parent_fits=40,parent_proof=str(WORK/'audit/all40_parent_training_CPU_verification_v1.json'),
            oracle_images=32,oracle_pairing_proof=str(WORK/'images/oracle_Tdiag32_v1/pairing_audit.json'),
            native_queue=str(WORK/'audit/native_benchmark_queue_v2.json'),
            new_component_implemented=False,scientific_goal_complete=False,
            declaration_status='Retrospective evidence index; original source/config/receipts determine registration'))+'\n')
    print(json.dumps(dict(status=state['stage'],queue_live=queue_live,GPU_live=GPU_live,images=state['BrainHIVE_native_images'])),flush=True)


if __name__=='__main__':main()
