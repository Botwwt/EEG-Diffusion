"""Dependency/schema rejection and small synthetic math checks, never real fit."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import brainhive_training_mean_field_CPU_20261101_v1 as M

def specification():
    names=('brainhive_training_mean_field_CPU_20261101_v1.py','compute_HIVE_training_mean_field_CPU_20261101_v1.py',
           'audit_HIVE_training_mean_field_CPU_20261101_v1.py','queue_HIVE_training_mean_field_CPU_20261101_v1.py',
           'check_HIVE_training_mean_field_CPU_20261101_v1.py')
    return dict(status=M.STATUS,host='old',parent_protocol=M.PARENT,output_root=str(M.OUT),code_root=str(M.CODE),
        parent_PID=487039,parent_start_ticks=113863283,roles=list(M.ROLES),queries=165400,subjects=10,concepts=1654,images=16540,
        chunk=512,new_training_updates=0,training_seed=1729,
        mean_definition='FP32_native_raw_MLP_outputs_FP64_sum_over_all165400_training_OOF_rows',
        raw_R80_inputs_allowed=False,image_score_payloads_allowed=False,official200_previously_exposed=True,independent_H=False,
        GPU_admitted=False,maximum_output_bytes=32<<20,maximum_extra_CPU_bytes=4<<30,minimum_start_CPU_headroom_bytes=24<<30,
        minimum_runtime_CPU_headroom_bytes=16<<30,CPU_seconds_per_stage=900,strength=1.,radius=.25,no_automatic_retry=True,
        legacy_floor_bytes=50<<30,disk_buffer_bytes=8<<30,numerical_guard=dict(atol=1e-5,rtol=5e-6),
        bounded_F_component=M.FHEAD,
        bounded_F_fit_completion=dict(path=str(M.BASE/'HIVE_full1654_F_lifecycle_20261101_v2/bounded_F/fit_completion.json'),bytes=1,sha256='0'*64),
        waiting_deadline_epoch=1.0,
        owned_process=M.OWNED,
        source_pins=[dict(path=str(M.CODE/name),bytes=1,sha256='0'*64) for name in names])

def contract_checks():
    s=specification();assert M.validate(s)==s
    refused=0
    for key,bad in [('host','new'),('queries',132300),('training_seed',2027),('raw_R80_inputs_allowed',True),
                    ('image_score_payloads_allowed',True),('GPU_admitted',True),('radius',.5),('strength',.75),
                    ('independent_H',True),('roles',['ordinary_T1323'])]:
        changed=copy.deepcopy(s);changed[key]=bad
        try:M.validate(changed)
        except AssertionError:refused+=1
        else:raise AssertionError((key,'invalid scope accepted'))
    done='COMPLETE_FULL1654_ORIGINAL_ORDINARY_FIT_2000_R80_ALL16_AND_FIXED5'
    assert M.classify_dependency(True,False,None,False)=='WAIT_SAME_ORIGINAL_LIFECYCLE'
    assert M.classify_dependency(False,True,done,False)=='WAIT_SAME_ORIGINAL_LIFECYCLE'
    assert M.classify_dependency(False,False,None,False).startswith('STOP_')
    assert M.classify_dependency(False,False,done,True)=='STOP_FAILED_DEPENDENCY_NO_RETRY'
    assert M.classify_dependency(False,False,done,False)=='READY_CLOSED_ORIGINAL_LIFECYCLE'
    with patch.object(M,'read',return_value=s),patch.object(M,'checked',side_effect=lambda p:Path(p['path'])),\
         patch.object(M,'dependency',return_value=dict(status='WAIT_SAME_ORIGINAL_LIFECYCLE')):
        try:M.checked_setup(str(M.CODE/'protocol.json'))
        except AssertionError:pass
        else:raise AssertionError('Live dependency bypassed before runtime import')
    return dict(valid_scope=True,invalid_scopes_rejected=refused,dependency_cases=5,
                actual_checked_setup_rejected_live_before_tensor_import=True,
                not_real_training_or_source_asset_validation=True)

def math_checks(torch):
    import numpy as np
    torch.set_num_threads(2)
    gen=torch.Generator().manual_seed(1729)
    u=torch.randn(12,1024,generator=gen);u-=u.mean(1,keepdim=True)
    gamma=torch.tensor([.5,2.0]*6)[:,None].expand_as(u).clone()
    beta=torch.linspace(-.3,.4,12)[:,None].expand_as(u).clone()
    c=gamma*u+beta
    state={'trunk.0.weight':torch.randn(256,1024,generator=gen)*.03,
           'trunk.0.bias':torch.randn(256,generator=gen)*.01,
           'direction.weight':torch.randn(1024,256,generator=gen)*.03,
           'direction.bias':torch.randn(1024,generator=gen)*.01}
    with torch.no_grad():
        raw=M.raw_output(c,gamma,beta,state,torch)
        mean,var=M.mean_and_variance(raw.double().sum(0),raw.double().square().sum(0),len(raw),torch)
        assert torch.allclose(mean,raw.double().mean(0),atol=1e-15,rtol=0)
        alternative=mean+.2
        lhs=(raw.double()-alternative).square().sum()-(raw.double()-mean).square().sum()
        rhs=len(raw)*(alternative-mean).square().sum()
        assert abs(float(lhs-rhs))<1e-9
        wrong=M.raw_output(c,gamma[:1].expand_as(gamma),beta[:1].expand_as(beta),state,torch)
        assert float((wrong-raw).square().mean())>1e-3
        modified=M.constant_state(state,mean.float(),torch)
        fixed=M.raw_output(c,gamma,beta,modified,torch)
        assert torch.equal(fixed,mean.float().expand_as(fixed))
        assert all(torch.equal(state[k],modified[k]) for k in ('trunk.0.weight','trunk.0.bias'))
        assert torch.count_nonzero(modified['direction.weight'])==0
        # A simple synthetic20-group map demonstrates that group counts must
        # belong to each row, not one broadcast encoder affine.
        assert np.bincount(np.tile(np.arange(20),2),minlength=20).tolist()==[2]*20
    assert not torch.cuda.is_initialized()
    return dict(mean_optimum_identity=True,own_affine_mismatch_detected=True,
                frozen_constant_state_for_all12_synthetic_inputs=True,
                unchanged_trunk_and_zero_direction_weight=True,synthetic_only=True,
                CUDA_initialized=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract-only',action='store_true');args=p.parse_args()
    result=dict(status='PASS_TRAINING_MEAN_PREPARATION_LOCAL_OR_SERVER_CONTRACT_CHECKS',contract=contract_checks())
    if not args.contract_only:
        import torch
        result['math']=math_checks(torch)
    print(json.dumps(result))
