"""Execute the actual frozen six-arm benchmark once, within the same40h cap.

No training/selection or automatic retry follows benchmark access. All original
images, receipts, failures and single query random streams remain authoritative.
"""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import time

from run_native_oof_stage import ROOT,PY,load,sha,stop_group,recover_gpu_receipt
from prepare_final_benchmark import check_sources,case_definitions

DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
STATE=ROOT/'audit/final_benchmark_state_v1.json'
STOP=ROOT/'STOP_FINAL_BENCHMARK'


def remaining():
    subprocess.run([PY,str(ROOT/'refresh_status.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    p=load(ROOT/'audit/current_progress.json')
    assert not any(j['alive'] for j in p['live_GPU_jobs']),'Another registered GPU job is still live'
    return p['remaining_authorized_hours']*3600


def write(state):
    state['last_update_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,STATE)
    campaign=load(ROOT/'campaign_state.json');campaign.update(stage='S6_FROZEN_BENCHMARK',final_benchmark_state=str(STATE),
        last_update_utc=state['last_update_utc'],final_test_unlocked=True)
    (ROOT/'campaign_state.json').write_text(json.dumps(campaign,indent=2)+'\n')
    (ROOT/'STATUS.md').write_text('Frozen official benchmark: '+state['status']+'\n\n'
        +'Actual state and owned process/log/command: audit/final_benchmark_state_v1.json. '
        +'All six preregistered2000-query arms require completion; no result is inferred from job exit alone.\n\n'
        +'Total40GPUh,seed1729,all10subjects,one image/query; no purchases or large C/D copies. '
        +'Official200 was exposed historically; independent H and SOTA are not established. '
        +'R1/R80 are cross-R generalization of the R4-trained selected reference, with distinct fixed image streams.\n\n'
        +'No training/checkpoint selection/seed changes after unlock. All current and historical failures are retained. '
        +'Final metrics: server evaluation/benchmark; unified report: reports/current_comparison.md.\n')


def coverage(cases):
    records=[]
    for case in cases:
        images=DATA/'images/S1_full_native_batchsafe_v2/benchmark'/case['arm']
        evaluation=DATA/'evaluation/benchmark'/case['arm']
        def read(path):
            try:return load(path)
            except (FileNotFoundError,json.JSONDecodeError,OSError):return {}
        identity=read(evaluation/'identity_v1/summary.json');metrics=read(evaluation/'metrics/summary.json');low=read(evaluation/'low_level/summary.json')
        records.append(dict(arm=case['arm'],expected_images=2000,saved_image_files=len(list(images.glob('sub-*/*.jpg'))),
            saved_query_receipts=len(list((images/'receipts').glob('sub-*/*.json'))),
            identity_verified_queries=identity.get('verified_queries',0),
            complete_primary_scores=(metrics.get('status')=='COMPLETED' and metrics.get('queries')==2000 and metrics.get('subjects')==10),
            complete_low_level_audit=(low.get('status')=='COMPLETED' and low.get('verified_and_decoded')==2000
                and low.get('missing_count')==low.get('corrupt_or_mismatched_count')==0)))
    return records


def run_phase(state,policy,name,spec,script,*arguments,gpu=True,expected_outstanding=0):
    check_sources(policy);assert not STOP.exists(),'Explicit final benchmark stop file'
    for path,digest in state['official_Python_source_sha256'].items():assert sha(path)==digest,('Frozen official source changed',path)
    left=remaining()
    assert left>=expected_outstanding+policy['cleanup_reserve_seconds'],'Complete remaining final scope no longer fits; preserve state'
    cap=min(spec['max_seconds'],left-policy['cleanup_reserve_seconds'])
    assert cap>spec['expected_seconds'],'No adequate whole-query phase reservation'
    log=ROOT/'logs'/f'{name}.log';assert not log.exists(),'No automatic repeat of final evaluation'
    command=[PY,str(ROOT/script),*map(str,arguments)]
    if gpu:command=[PY,str(ROOT/'run_gpu_job.py'),'--name',name,'--max-seconds',str(cap),'--',*command]
    env=os.environ.copy()
    if not gpu:env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    tick=time.monotonic();process=None;interrupted=False
    try:
        with log.open('xb') as file:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='RUNNING_FROZEN_FINAL_PHASE',phase=name,pid=process.pid,command=command,log=str(log),GPU_used=gpu,
                actual_remaining_seconds=left,expected_outstanding_seconds=expected_outstanding,actual_phase_cap_seconds=cap)
            write(state)
            while process.poll() is None:
                if STOP.exists() or time.monotonic()-tick>=min(cap+30,left-policy['cleanup_reserve_seconds']):
                    interrupted=True;stop_group(process);break
                time.sleep(5)
            code=process.wait()
    finally:
        if process is not None:
            if process.poll() is None:interrupted=True;stop_group(process)
            elapsed=time.monotonic()-tick;code=process.returncode
            if gpu and interrupted:recover_gpu_receipt(name,code,elapsed)
            if not gpu:
                with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
                    file.write(json.dumps(dict(stage=name,command=command,seconds=elapsed,exit_code=code,
                        GPU_used=False,controller_interrupted=interrupted,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))+'\n')
    assert code==0 and not interrupted,('Final phase incomplete; preserve outputs, never change random streams',name,code)
    state['completed_phases'].append(name);write(state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resolved',required=True);args=parser.parse_args()
    path=Path(args.resolved).resolve();assert path==ROOT/'configs/final_benchmark_resolved_v1.json'
    r=load(path);assert r['status']=='FINAL_METHODS_AND_PROTOCOL_FROZEN_NOT_UNLOCKED'
    policy=load(r['policy_path']);assert sha(r['policy_path'])==r['policy_sha256'];check_sources(policy)
    assert [(c['method'],c['condition'],c['R']) for c in r['cases']]==case_definitions(r['selected_reference'])
    for key in ['gate','selection']:
        assert sha(r[key+'_path'])==r[key+'_sha256']
    for key in ['control_state_proof','gallery_proof']:
        proof=r[key];assert sha(proof['path'])==proof['sha256']
    for proof in r['postprocessing_cost_proofs']:assert sha(proof['path'])==proof['sha256']
    for measurement in r['measurements'].values():
        for proof in measurement['proofs'].values():assert sha(proof['path'])==proof['sha256']
        for subject,checkpoints in measurement['checkpoints'].items():
            for filename,digest in [('encoder_checkpoint','encoder_sha256'),('prior_checkpoint','prior_sha256')]:
                if checkpoints[filename] is not None:assert sha(checkpoints[filename])==checkpoints[digest]
    for name in ['split','data_manifest','query_manifest']:
        filename={'split':'configs/split_manifest.json','data_manifest':'audit/data_manifest.csv','query_manifest':'audit/query_manifest_R4.csv'}[name]
        assert sha(ROOT/filename)==r[name+'_sha256']
    for row in r['test_data_identity_rows']:assert sha(row['path'])==row['sha256']
    for filename,digest in r['official_Python_source_sha256'].items():assert sha(filename)==digest
    assert not STATE.exists() and not STOP.exists()
    unlock=ROOT/'configs/FINAL_EVAL_UNLOCK.json';assert not unlock.exists(),'Do not reopen final selection'
    gate=load(r['gate_path']);assert gate['passed']
    assert gate['six_arms']==r['cases']
    for case in r['cases']:
        assert not (DATA/'images/S1_full_native_batchsafe_v2/benchmark'/case['arm']).exists()
        assert not (DATA/'evaluation/benchmark'/case['arm']).exists()
    assert not (DATA/'evaluation/gallery/benchmark').exists()
    assert remaining()>=gate['required_seconds_with_allowance'],'Fresh complete final budget no longer fits'
    # Only now authorize the existing generator/gallery benchmark branches.
    value=dict(status='UNLOCKED_ONCE_AFTER_ACTUAL_FULL_STAGE_AND_BUDGET_CHECKS',
        method_and_checkpoint_selection_frozen=True,split_sha256=r['split_sha256'],
        frozen_resolved_path=str(path),frozen_resolved_sha256=sha(path),policy_sha256=r['policy_sha256'],
        cases=r['cases'],model_updates_allowed=False,selection_after_unlock_allowed=False,
        seed=1729,total_expected_images=12000,official200_previously_exposed=True,independent_H_established=False,
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    with unlock.open('x') as file:json.dump(value,file,indent=2);file.write('\n')
    state=dict(status='FINAL_UNLOCKED_NOT_STARTED',controller_pid=os.getpid(),resolved_path=str(path),resolved_sha256=sha(path),
        expected_images=12000,completed_phases=[],final_test_unlocked=True,automatic_retry=False,model_updates_after_unlock=False,
        independent_confirmation=False,SOTA_established=False,official_Python_source_sha256=r['official_Python_source_sha256'])
    write(state)
    image_base=DATA/'images/S1_full_native_batchsafe_v2/benchmark';evaluation=DATA/'evaluation/benchmark'
    gallery=DATA/'evaluation/gallery/benchmark';work=[]
    for i,case in enumerate(r['cases']):
        arm=case['arm'];images=image_base/arm;out=evaluation/arm
        assert not images.exists() and not out.exists(),'No silent reuse of previous benchmark files'
        work.append((f'S6_generate_{arm}',case['generation'],'generate_reference_images.py',
            ['--method',case['method'],'--condition',case['condition'],'--partition','benchmark','--R',case['R']],True))
        if i==0:
            from prepare_final_benchmark import phase_estimate
            work.append(('S6_prepare_original_gallery',phase_estimate(policy['gallery_materialize_expected_seconds'],policy),
                'prepare_real_gallery.py',['--generation-manifest',images/'manifest.jsonl','--partition','benchmark'],False))
            work.append(('S6_original_gallery_features',phase_estimate(r['gallery_expected_seconds'],policy),
                'extract_image_features.py',['--manifest',gallery/'manifest.jsonl','--output',gallery/'features','--real-gallery'],True))
        work.extend([
            (f'S6_features_{arm}',case['features'],'extract_image_features.py',['--manifest',images/'manifest.jsonl','--output',out/'features'],True),
            (f'S6_scores_{arm}',case['score'],'score_full_manifest.py',['--manifest',images/'manifest.jsonl','--features',out/'features',
                '--gallery-manifest',gallery/'manifest.jsonl','--gallery-features',gallery/'features','--output',out/'metrics'],False),
            (f'S6_low_level_{arm}',case['low_level'],'score_low_level.py',['--manifest',images/'manifest.jsonl',
                '--gallery-manifest',gallery/'manifest.jsonl','--output',out/'low_level'],False),
            (f'S6_identity_{arm}',case['identity'],'audit_complete_image_pairing.py',['--manifest',images/'manifest.jsonl',
                '--scores',out/'metrics/per_query.csv','--output',out/'identity_v1'],False)])
    try:
        from prepare_final_benchmark import phase_estimate
        for i,(name,spec,script,arguments,gpu) in enumerate(work):
            expected=sum(w[1]['expected_seconds'] for w in work[i:])+r['paired_expected_seconds']+r['offline_gallery_expected_seconds']+policy['report_expected_seconds']
            run_phase(state,policy,name,spec,script,*arguments,gpu=gpu,expected_outstanding=expected)
            if name.startswith('S6_identity_'):
                arm=name[len('S6_identity_'):];case=next(c for c in r['cases'] if c['arm']==arm)
                frozen=r['measurements'][f'{case["method"]}_{case["condition"]}_R4']
                records=[json.loads(line) for line in (image_base/arm/'manifest.jsonl').read_text().splitlines()]
                assert len(records)==2000
                for row in records:
                    cp=frozen['checkpoints'][row['subject']]
                    assert row['encoder_sha256']==cp['encoder_sha256'] and row['prior_sha256']==cp['prior_sha256']
                    assert all(row[k]==v for k,v in frozen['image_interface'].items())
        selected=r['selected_reference'];correct=f'{selected}_correct_R4'
        paired=[('REF_REG_correct_R4','REF_PRIOR_V_correct_R4'),(correct,f'{selected}_permuted_R4'),(correct,f'{selected}_missing_R4')]
        paired_spec=phase_estimate(r['paired_expected_seconds']/3,policy)
        for j,(a,b) in enumerate(paired):
            run_phase(state,policy,f'S6_paired_{b}_minus_{a}',paired_spec,'paired_analysis.py',
                '--reference-manifest',image_base/a/'manifest.jsonl','--candidate-manifest',image_base/b/'manifest.jsonl',
                '--reference-metrics',evaluation/a/'metrics/per_query.csv','--candidate-metrics',evaluation/b/'metrics/per_query.csv',
                '--output',DATA/'evaluation/paired'/f'benchmark_{b}_minus_{a}_v1',gpu=False,
                expected_outstanding=r['offline_gallery_expected_seconds']+(3-j)*paired_spec['expected_seconds']+policy['report_expected_seconds'])
        gallery_root=DATA/'galleries/benchmark_frozen_v1';gallery_spec=phase_estimate(r['offline_gallery_expected_seconds']/3,policy)
        for j,R in enumerate(['R4','R1','R80']):
            arguments=[]
            for case in r['cases']:
                if case['R']==R:
                    arm=case['arm'];arguments+=['--arm',arm,image_base/arm/'manifest.jsonl',evaluation/arm/'metrics/per_query.csv']
            run_phase(state,policy,f'S6_full_offline_gallery_{R}',gallery_spec,'build_offline_gallery.py',
                *arguments,'--output',gallery_root/R,gpu=False,expected_outstanding=(3-j)*gallery_spec['expected_seconds']+policy['report_expected_seconds'])
        (gallery_root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Complete frozen benchmark</title>'
            '<h1>Complete frozen benchmark</h1><p>Official200 was previously exposed. No independent H or human ratings.</p>'
            '<p>R4: both references and matched EEG interventions. R1/R80: selected R4-trained reference, distinct fixed noise streams.</p>'
            '<ul>'+''.join(f'<li><a href="{R}/index.html">{R}: all2000 queries</a></li>' for R in ['R4','R1','R80'])+'</ul>')
        audit=coverage(r['cases'])
        assert sum(a['saved_image_files'] for a in audit)==sum(a['identity_verified_queries'] for a in audit)==12000
        assert all(a['complete_primary_scores'] and a['complete_low_level_audit'] for a in audit)
        run_phase(state,policy,'S6_unified_comparison_report',phase_estimate(policy['report_expected_seconds'],policy),
            'write_comparison_report.py',gpu=False,expected_outstanding=policy['report_expected_seconds'])
        state.update(status='COMPLETED_ALL_SIX_FROZEN_BENCHMARK_ARMS',coverage=audit,gallery=str(gallery_root/'index.html'),
            independent_confirmation=False,SOTA_established=False,EEG_interface_mechanism_gain_established=False)
        write(state)
    except BaseException as error:
        state.update(status='FAILED_OR_BUDGET_STOPPED_FINAL_OUTPUTS_PRESERVED',error=repr(error),coverage=coverage(r['cases']),
            no_post_test_model_or_seed_change=True,automatic_retry=False)
        write(state);raise


if __name__=='__main__':
    def terminate(signum,frame):raise SystemExit('Explicit final supervisor signal '+str(signum))
    signal.signal(signal.SIGTERM,terminate)
    main()
