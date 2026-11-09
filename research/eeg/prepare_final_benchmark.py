"""Freeze the already-declared six-arm benchmark only after real stage closure.

Does not unlock/generate/score the benchmark. Previously exposed official200 is
explicitly not an independent H; potential unfinished S3/S4 work takes priority.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import subprocess
import time

from run_native_oof_stage import ROOT,PY,load,sha,fresh_remaining

DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


class Pending(RuntimeError):pass


def check_sources(policy):
    for name,digest in policy['source_sha256'].items():assert sha(ROOT/name)==digest,('Frozen final source changed',name)


def case_definitions(selected):
    assert selected in ['REF_REG','REF_PRIOR_V']
    return [('REF_REG','correct','R4'),('REF_PRIOR_V','correct','R4'),
            (selected,'permuted','R4'),(selected,'missing','R4'),
            (selected,'correct','R1'),(selected,'correct','R80')]


def phase_estimate(seconds,policy,fixed=0):
    assert seconds>=0
    expected=seconds+fixed
    return dict(expected_seconds=expected,max_seconds=math.ceil((expected*policy['per_phase_cap_multiplier']+30)/30)*30)


def scientific_closeout(selected):
    paths=sorted((ROOT/'audit').glob('S3_OOF_budget_check_*.json'))
    if not paths:raise Pending('Actual S3 budget decision missing')
    decisions=[load(p) for p in paths];funded=[d for d in decisions if d['full_two_fold_training_and_condition_sampling_funded']]
    if not funded:
        # The predeclared second review can become affordable after measured
        # controls release their forecast margin. Do not consume its budget.
        if len(decisions)<2 or len(decisions[-1]['completed_full_controls'])!=2:
            raise Pending('Complete the permitted second S3 budget review after both actual full controls before freezing final evaluation')
        assert decisions[-1]['selected_reference']==selected
        return dict(status='S3_NOT_RUN_AFTER_TWO_ACTUAL_COMPLETE_SCOPE_BUDGET_DECISIONS',
            budget_proofs=[dict(path=str(p),sha256=sha(p)) for p in paths],CPU_gate_failed=False)
    found=[]
    for path in (ROOT/'configs').glob('S3_native_oof_*_resolved.json'):
        r=load(path)
        if r['selected_reference']!=selected:continue
        state_path=ROOT/'audit'/f'{r["run_id"]}_state.json'
        if state_path.exists():found.append((path,r,state_path,load(state_path)))
    if len(found)!=1:raise Pending('A unique actual selected-chain S3 stage must be inspected before final freeze')
    path,r,state_path,state=found[0]
    if state['status']!='COMPLETED_ALL10_OOF_CACHE_AND_CPU_DCHECK':
        raise Pending('S3 is incomplete: inspect saved checkpoints/cache, remaining budget and possible continuation before final closeout')
    assert state['resolved_sha256']==sha(path)
    report_path=Path(state['CPU_validation_summary_path']);assert sha(report_path)==state['CPU_validation_summary_sha256']
    report=load(report_path);assert report['subjects']==10 and report['Dfit_queries']==8200 and report['Dcheck_queries']==8300
    gate=report['gate'];failed=not (gate['native_energy_CI_upper_below_zero_vs_Iso_and_Moment']
        and gate['projected_energy_CI_upper_below_zero_vs_Iso_and_Moment'])
    return dict(status='ACTUAL_COMPLETE_OOF_AND_CPU_CHANNEL_VALIDATION',resolved_proof=dict(path=str(path),sha256=sha(path)),
        state_proof=dict(path=str(state_path),sha256=sha(state_path)),validation_proof=dict(path=str(report_path),sha256=sha(report_path)),
        CPU_gate_failed=failed,full_UNet_response_validated=False,image_mechanism_gain_established=False)


def full_V_arm(method,condition):
    arm=f'{method}_{condition}_R4';images=DATA/'images/S1_full_native_batchsafe_v2/V'/arm
    evaluation=DATA/'evaluation/V'/arm
    generation=images/'completion.json';features=evaluation/'features/completion.json';metrics=evaluation/'metrics/summary.json'
    low=evaluation/'low_level/summary.json';g=load(generation);f=load(features);m=load(metrics);l=load(low)
    assert g['status']==f['status']==m['status']==l['status']=='COMPLETED'
    assert g['queries']==f['queries']==m['queries']==l['verified_and_decoded']==16600
    assert l['missing_count']==l['corrupt_or_mismatched_count']==0
    assert m['subjects']==10 and m['method']==method and m['intervention']==condition and m['repeat_protocol']=='R4'
    manifest=images/'manifest.jsonl';assert sha(manifest)==g['image_manifest_sha256']==f['manifest_sha256']
    assert g['counts']['sample_evaluations']==16600*40,'Resumed generation requires actual cumulative cost reconciliation before final projection'
    jobs=[]
    for path in (ROOT/'audit').glob('*_job.json'):
        job=load(path);command=job.get('command',[])
        if str(manifest) in command and str(ROOT/'extract_image_features.py') in command:jobs.append((path,job))
    if len(jobs)!=1 or jobs[0][1]['status']!='COMPLETED':
        raise Pending('A resumed/failed feature extraction needs actual cumulative full-run cost reconciliation')
    feature_seconds=max(f['seconds'],jobs[0][1]['charged_seconds'])
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows)==16600 and len({q['query_id'] for q in rows})==16600
    interface_keys=['generator_revision','image_interface_config_sha256','steps','CFG','IP_scale','UNet_NFE','image_interface_version']
    assert all(len({q[k] for q in rows})==1 for k in interface_keys)
    checkpoints={}
    for subject in sorted({q['subject'] for q in rows}):
        batch=[q for q in rows if q['subject']==subject];assert len(batch)==1660
        fields=['encoder_checkpoint','encoder_sha256','prior_checkpoint','prior_sha256']
        assert all(len({q[k] for q in batch})==1 for k in fields)
        checkpoints[subject]={k:batch[0][k] for k in fields}
        for key,digest in [('encoder_checkpoint','encoder_sha256'),('prior_checkpoint','prior_sha256')]:
            if batch[0][key] is not None:assert sha(batch[0][key])==batch[0][digest]
    proof={name:dict(path=str(path),sha256=sha(path)) for name,path in
        [('generation',generation),('features',features),('metrics',metrics),('low_level',low),('manifest',manifest)]}
    return dict(arm=arm,generation_seconds=g['seconds'],feature_seconds=feature_seconds,
        scoring_seconds=m['seconds'],low_level_seconds=l['seconds'],proofs=proof,checkpoints=checkpoints,
        image_interface={k:rows[0][k] for k in interface_keys})


def prepare(policy_path):
    policy_path=Path(policy_path);policy=load(policy_path);check_sources(policy)
    assert policy['version']=='frozen_six_arm_benchmark_policy_v1' and policy['total_GPU_hour_cap']==40
    assert not (ROOT/'configs/FINAL_EVAL_UNLOCK.json').exists(),'Final evaluation already unlocked; do not freeze/select again'
    state_path=ROOT/'audit/S1_EEG_control_queue_v2_state.json'
    if not state_path.exists() or load(state_path)['status']!='COMPLETED_FULL_V_EEG_CONTROLS':
        raise Pending('Both complete16600-query EEG controls, primary/low-level metrics, paired analysis and full gallery are required')
    control=load(state_path);assert control['EEG_control_arms']==2 and control['full_query_count_per_arm']==16600
    assert control['policy_sha256']==sha(ROOT/'configs/full_EEG_controls_stage_policy_v2.json')
    assert sha(control['release_path'])==control['release_sha256']
    selection_path=ROOT/'audit/reference_control_selection.json';selection=load(selection_path)
    selected=selection['selected_reference'];assert selected==control['selected_reference']
    assert selection['status']=='SELECTED_ON_COMPLETE_V_BY_PREDECLARED_RULE'
    closeout=scientific_closeout(selected)
    definitions=case_definitions(selected);assert len(set(definitions))==6
    for method,condition,R in definitions:
        arm=f'{method}_{condition}_{R}'
        assert not (DATA/'images/S1_full_native_batchsafe_v2/benchmark'/arm).exists(),'Previous final files need technical inspection before any unlock'
        assert not (DATA/'evaluation/benchmark'/arm).exists()
    assert not (DATA/'evaluation/gallery/benchmark').exists()
    measurements={}
    for method,condition,R in definitions:
        key=(method,condition)
        if key not in measurements:measurements[key]=full_V_arm(method,condition)
    gallery=DATA/'evaluation/gallery/V/features/completion.json';gallery_cost=load(gallery)
    assert gallery_cost['status']=='COMPLETED' and gallery_cost['queries']==1660
    ratio=2000/16600;cases=[]
    for method,condition,R in definitions:
        measured=measurements[(method,condition)]
        cases.append(dict(method=method,condition=condition,R=R,arm=f'{method}_{condition}_{R}',queries=2000,
            generation=phase_estimate(measured['generation_seconds']*ratio,policy,policy['generation_fixed_allowance_seconds']),
            features=phase_estimate(measured['feature_seconds']*ratio,policy,policy['feature_fixed_allowance_seconds']),
            score=phase_estimate(measured['scoring_seconds']*ratio,policy,policy['score_fixed_allowance_seconds']),
            low_level=phase_estimate(measured['low_level_seconds']*ratio,policy,policy['low_level_fixed_allowance_seconds']),
            identity=phase_estimate(policy['identity_expected_seconds_per_case'],policy)))
    gallery_expected=gallery_cost['seconds']*(200/1660)+policy['gallery_fixed_allowance_seconds']
    primary_expected=sum(c[p]['expected_seconds'] for c in cases for p in ['generation','features','score','low_level','identity'])
    paired_paths=[DATA/'evaluation/paired/V_PRIOR_V_vs_REG_v1/summary.json']
    paired_paths += [DATA/'evaluation/paired'/f'V_{selected}_{condition}_minus_correct_v2/summary.json' for condition in ['permuted','missing']]
    paired_expected=0
    for path in paired_paths:
        p=load(path);assert p['status']=='COMPLETED_PAIRED_FULL_PARTITION' and p['queries']==16600
        paired_expected+=p['seconds']*ratio+policy['paired_fixed_allowance_seconds']
    gallery_V=Path(control['gallery']).parent/'completion.json';gv=load(gallery_V)
    assert gv['status']=='COMPLETED_FULL_OFFLINE_GALLERY' and gv['expected_generated_images']==66400
    offline_expected=gv['seconds']*((12000+3*200)/gv['assets'])+3*policy['offline_gallery_fixed_allowance_seconds']
    extra_expected=gallery_expected+paired_expected+offline_expected+policy['gallery_materialize_expected_seconds']+policy['report_expected_seconds']
    expected=primary_expected+extra_expected
    remaining=fresh_remaining()
    selected_measurement=measurements[(selected,'correct')]
    S4_expected=5*(selected_measurement['generation_seconds']+selected_measurement['feature_seconds'])
    if not closeout['CPU_gate_failed'] and S4_expected<=remaining-expected-policy['cleanup_reserve_seconds']:
        raise Pending('A complete matched S4 image scope may still be fundable; finish its UNet/training prerequisite review before final freeze')
    required=expected*policy['whole_stage_runtime_multiplier']+policy['cleanup_reserve_seconds']
    gate=dict(status='COMPLETE_SIX_ARM_FINAL_BUDGET_REVIEW',UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        expected_seconds=expected,required_seconds_with_allowance=required,actual_remaining_seconds=remaining,
        passed=remaining>=required,policy_sha256=sha(policy_path),six_arms=cases,total_expected_images=12000,
        S3_closeout=closeout,S4_matched_image_expected_seconds=S4_expected,
        S4_closed_by_CPU_gate=closeout['CPU_gate_failed'],S4_image_gain_established=False,
        estimates_are_not_mathematical_runtime_lower_bounds=True,final_test_unlocked=False)
    gate_path=ROOT/'audit/final_benchmark_complete_scope_gate_v1.json'
    with gate_path.open('x') as file:json.dump(gate,file,indent=2);file.write('\n')
    if not gate['passed']:return dict(status='NOT_RUN_COMPLETE_FINAL_SCOPE_UNFUNDED',gate_path=str(gate_path),final_test_unlocked=False)
    # Read existing verified input identities only; do not run the encoder or
    # open final images/features while constructing this frozen configuration.
    with (ROOT/'audit/data_manifest.csv').open() as file:
        data=[d for d in csv.DictReader(file) if d['dataset_split']=='test']
    assert len(data)==30 and {d['repeat_protocol'] for d in data}=={'R1','R4','R80'}
    for d in data:assert sha(d['path'])==d['sha256']
    official=Path('/root/atm_official_20261016');base=load(ROOT/'configs/full_reference_resolved.json')
    assert sha(official/'models/atms.py')==base['official_atms_source_sha256']
    prior_paths=list(official.rglob('diffusion_prior.py'));assert len(prior_paths)==1
    assert sha(prior_paths[0])==base['official_prior_source_sha256']
    tracked=subprocess.check_output(['git','ls-files','*.py'],cwd=official,text=True).splitlines()
    external_sources={str(official/name):sha(official/name) for name in tracked}
    resolved=dict(status='FINAL_METHODS_AND_PROTOCOL_FROZEN_NOT_UNLOCKED',policy_path=str(policy_path),policy_sha256=sha(policy_path),
        gate_path=str(gate_path),gate_sha256=sha(gate_path),selection_path=str(selection_path),selection_sha256=sha(selection_path),
        control_state_proof=dict(path=str(state_path),sha256=sha(state_path)),S3_closeout=closeout,
        selected_reference=selected,cases=cases,measurements={m['arm']:m for m in measurements.values()},
        gallery_proof=dict(path=str(gallery),sha256=sha(gallery)),gallery_expected_seconds=gallery_expected,
        extra_expected_seconds=extra_expected,paired_expected_seconds=paired_expected,offline_gallery_expected_seconds=offline_expected,
        postprocessing_cost_proofs=[dict(path=str(p),sha256=sha(p)) for p in paired_paths+[gallery_V]],
        expected_total_images=12000,subjects=10,seed=1729,
        split_sha256=sha(ROOT/'configs/split_manifest.json'),data_manifest_sha256=sha(ROOT/'audit/data_manifest.csv'),
        query_manifest_sha256=sha(ROOT/'audit/query_manifest_R4.csv'),test_data_identity_rows=data,
        official_Python_source_sha256=external_sources,
        model_updates_after_unlock_allowed=False,selection_after_unlock_allowed=False,
        official200_previously_exposed=True,independent_H_established=False,
        limitations=['Six previously declared baseline/intervention/cross-R arms; no unrun adapter is represented.',
            'R1/R80 use the R4-trained selected encoder and different predetermined image streams; not R-matched trained SOTA or same-noise R causal effects.',
            'No independent H, new collection, best-of-N or post-test tuning is authorized.'])
    path=ROOT/'configs/final_benchmark_resolved_v1.json'
    with path.open('x') as file:json.dump(resolved,file,indent=2);file.write('\n')
    return resolved|dict(resolved_path=str(path),resolved_sha256=sha(path))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',required=True);args=parser.parse_args()
    try:result=prepare(args.policy)
    except Pending as error:
        print(json.dumps(dict(status='PREREQUISITES_PENDING_FINAL_STAYS_LOCKED',reason=str(error))),flush=True)
        raise SystemExit(2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
