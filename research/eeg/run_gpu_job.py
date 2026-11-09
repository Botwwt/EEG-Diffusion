"""One-GPU job lock, wall-time ledger, and hard cap for this private campaign."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    command=args.command[1:] if args.command and args.command[0]=='--' else args.command
    assert command and '/' not in args.name and args.max_seconds>0
    if args.name.startswith('S1_V_generate_REF_PRIOR'):
        # Finite values alone do not validate an EEG-to-image condition chain.
        # A real T-image diagnostic must resolve the observed sampled norm shift.
        gate=ROOT/'audit/prior_sampling_prerequisite_check.json'
        assert gate.exists(), 'Prior T-image/condition-space diagnostic pending; preserve REF-REG and do not launch the full prior image run'
        decision=json.loads(gate.read_text())
        assert decision['full_image_run_authorized_after_diagnostic'],decision
        assert args.name.startswith('S1_V_generate_'+decision['permitted_method']), 'Readiness does not authorize a different prior version'
    budget=json.loads((ROOT/'audit/locked_budget_allocation.json').read_text())
    assert budget['total_limit_GPU_hours']==40
    lock=(ROOT/'audit/GPU_JOB.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=ROOT/'audit/compute_ledger.jsonl'
    if not ledger.exists():
        ledger.write_text(json.dumps(dict(type='initial_prior_compute_allowance',charged_seconds=(1.4351579818408937+.25)*3600,
            basis='Historical recorded job compute plus conservative quarter-hour S0 GPU allowance; provider rental billing not independently available'))+'\n')
    used=sum(float(row.get('charged_seconds',0)) for row in map(json.loads,ledger.read_text().splitlines()))
    assert used+args.max_seconds+45<=40*3600,'Job reservation exceeds confirmed total cap'
    time_bounds=ROOT/'configs/compute_time_bounds.json'
    if time_bounds.exists():
        deadline=json.loads(time_bounds.read_text())['conservative_occupied_time_deadline_epoch']
        assert time.time()+args.max_seconds+45<deadline,'Job reservation exceeds conservative occupied-GPU wall-time bound'
    receipt=ROOT/'audit'/f'{args.name}_job.json';assert not receipt.exists(),'Use a separately named retry receipt'
    start=time.monotonic();p=subprocess.Popen(command,cwd=ROOT)
    record=dict(name=args.name,pid=p.pid,wrapper_pid=os.getpid(),command=command,
        start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),max_seconds=args.max_seconds,
        charged_seconds_before_job=used,remaining_seconds_at_start=40*3600-used,status='RUNNING')
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    try:
        code=p.wait(timeout=args.max_seconds+10)
    except subprocess.TimeoutExpired:
        p.terminate()
        try:code=p.wait(timeout=20)
        except subprocess.TimeoutExpired:p.kill();code=p.wait()
        record['termination_reason']='Registered per-job wall-time limit; preserve existing epoch checkpoints'
    elapsed=time.monotonic()-start
    record.update(status='COMPLETED' if code==0 else 'FAILED_OR_BUDGET_STOPPED_RESUMABLE',exit_code=code,
        charged_seconds=elapsed,end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    with ledger.open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True);sys.exit(0 if code==0 else 1)


if __name__=='__main__':main()
