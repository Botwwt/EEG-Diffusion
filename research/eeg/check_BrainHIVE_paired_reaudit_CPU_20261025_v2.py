"""Regression tests for v2 closures/resources and limits of the scientific idea."""
from pathlib import Path
import copy
import json
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import brainhive_paired_study_contract_20261025_v1 as C
from evaluate_BrainHIVE_paired_effect_CPU_20261025_v2 import stratum_rows,functional_readout
from run_BrainHIVE_paired_feature_guard_20261025_v2 import PeakGuard
from queue_BrainHIVE_paired_effect_20261025_v2 import command_spec


def rejected(fn,contains=None):
    try:fn()
    except AssertionError as error:
        if contains:assert contains in str(error)
        return
    raise AssertionError('Invalid state accepted')


def main():
    checks=[]
    values=[100,10,1000]
    guard=PeakGuard(dict(CPU=20,GPU=30,minimum_headroom=40,seconds=60),lambda:values[0],lambda:values[1],lambda:values[2])
    guard.check();values[0]=121;rejected(guard.check,'CPU peak');checks.append('feature_CPU_peak_enforced')
    values[:]=[100,31,1000]
    guard=PeakGuard(dict(CPU=20,GPU=30,minimum_headroom=40,seconds=60),lambda:values[0],lambda:values[1],lambda:values[2])
    rejected(guard.check,'GPU allocated');checks.append('feature_GPU_peak_enforced')
    values[:]=[100,10,39]
    fresh=PeakGuard(dict(CPU=20,GPU=30,minimum_headroom=40,seconds=60),lambda:values[0],lambda:values[1],lambda:values[2])
    rejected(fresh.check,'headroom');checks.append('feature_actual_headroom_enforced')
    values[:]=[100,10,1000];fresh.started-=61;rejected(fresh.check,'deadline');checks.append('feature_finite_deadline_enforced')
    matrix=np.arange(12,dtype=float).reshape(2,3,2)
    rows=list(stratum_rows('D_minus_O',matrix,['s1','s2'],['c1','c2','c3'],10,['m1','m2']))
    assert len(rows)==2+3+6
    assert rows[0]['queries']==30 and rows[2]['queries']==20 and rows[-1]['queries']==10
    assert rows[0]['m1']==float(matrix[0,:,0].mean()) and rows[-1]['m2']==matrix[-1,-1,-1]
    checks.append('all_subject_concept_and_joint_strata_correct')
    with tempfile.TemporaryDirectory(prefix='hive_review_v2_') as directory:
        root=Path(directory);(root/'audit').mkdir();(root/'functional_T80').mkdir()
        config=dict(functional_job_name='probe',functional_queries=dict(path='fixed',sha256='fixed'))
        summary=root/'functional_T80/summary.json';job=root/'audit/probe_job.json'
        with patch.object(C,'ROOT',root),patch.object(C,'DATA',root),patch.object(C,'core',return_value='core'):
            assert functional_readout(config)['result']=='UNKNOWN';checks.append('missing_functional_is_UNKNOWN')
            summary.write_text('{"status":"partially_written"}')
            job.write_text('{"status":"FAILED","exit_code":1}')
            assert functional_readout(config)['result']=='UNKNOWN';checks.append('failed_job_with_partial_summary_is_UNKNOWN')
            summary.write_text('{')
            assert functional_readout(config)['result']=='UNKNOWN';checks.append('malformed_functional_summary_is_UNKNOWN')
            files={}
            for arm in C.ARMS:
                folder=root/'arms'/arm/'train';folder.mkdir(parents=True)
                (folder/'head.pt').write_bytes(arm.encode());files[arm]=C.ref(folder/'head.pt')
                C.atomic(folder/'completion.json',dict(checkpoint=files[arm]))
            (root/'raw.gz').write_bytes(b'fixture');(root/'queries.json').write_text('{}')
            good=dict(status='COMPLETE_FIXED_POSTFIT_PAIRED_T80',study_core_sha256='core',
                queries=80,unique_stimuli=8,subjects=10,state_domains=2,timesteps=4,layers=70,missing_states=0,
                raw_query_layer_records=358400,minimum_valid_fraction=1.,trained_checkpoints=files,
                fixed_selection=config['functional_queries'],raw_states=C.ref(root/'raw.gz'),query_manifest=C.ref(root/'queries.json'),
                paired_projection_increment=dict(result='SUPPORTED'),D_pair_gap_minus_controls={d:{r:{a:.1 for a in ('O','A','U')}
                    for r in ('training','deployment')} for d in ('DDPM_real_T','native_parent_trajectory')})
            summary.write_text(json.dumps(good));job.write_text('{"status":"COMPLETED","exit_code":0}')
            assert functional_readout(config)['result']=='SUPPORTED';checks.append('complete_functional_provenance_and_all_fixed_contrasts_required')
            bad=copy.deepcopy(good);bad['raw_query_layer_records']-=1;summary.write_text(json.dumps(bad))
            assert functional_readout(config)['result']=='UNKNOWN';checks.append('missing_raw_state_cannot_pass')
            bad=copy.deepcopy(good);bad['D_pair_gap_minus_controls']['DDPM_real_T']['training']['A']=-.1
            summary.write_text(json.dumps(bad));assert functional_readout(config)['result']=='UNKNOWN';checks.append('contradictory_positive_claim_cannot_pass')
            summary.write_text(json.dumps(good));(root/'raw.gz').write_bytes(b'corrupt')
            assert functional_readout(config)['result']=='UNKNOWN';checks.append('mutated_functional_raw_file_cannot_pass')
    config=dict(registration_path='/locked/review.json',functional_job_name='probe',
        training_job_names={a:[a+'_train',a+'_resume'] for a in C.ARMS},
        generation_job_names={a:dict(T32=[a+'_T32'],V=[a+'_V',a+'_V_resume']) for a in C.ARMS})
    spec=command_spec(config,'O','train',1);assert spec['command'][-1]=='--resume'
    assert 'feature_guard' in command_spec(config,'D','features')['command'][3]
    assert command_spec(config,None,'functional')['command'][3].endswith('_v2.py')
    checks.append('review_resume_guarded_features_and_v2_functional_are_actual_commands')
    # An independent teacher cannot become EEG-predictable by subtracting pairs.
    # Balanced orthogonal EEG/teacher signs give minimum pair MSE at zero gain.
    eeg=torch.tensor([-1.,-1.,1.,1.]);teacher=torch.tensor([-1.,1.,-1.,1.])
    def pair_error(gain):
        error=gain*eeg-teacher
        return (error[:,None]-error[None,:]).square().mean()
    assert pair_error(0)<pair_error(1) and pair_error(0)==pair_error(-0.)
    checks.append('difference_MSE_does_not_guarantee_anti_shrinkage_counterexample')
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PAIRED_REAUDIT_V2_CPU_REGRESSIONS_PASSED',checks=checks,
        GPU_used=False,new_fit=False,scientific_core_changed=False)))


if __name__=='__main__':main()
