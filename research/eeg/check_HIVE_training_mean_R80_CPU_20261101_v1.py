"""Wrong-scope/dependency refusal plus independent synthetic geometry/fullpanel stats."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
import brainhive_training_mean_R80_contract_CPU_20261101_v1 as M


def specification():
    return dict(status=M.STATUS,host='old',GPU_UUID=M.GPU,code_root=M.CODE.as_posix(),output_root=M.OUT.as_posix(),
        parent_protocol=M.PARENT,mean_protocol=M.MEAN,owned_process=M.OWNED,parent_PID=487039,parent_start_ticks=113863283,
        mean_PID=492474,mean_start_ticks=114699744,roles=M.ROLES,comparisons=[list(x) for x in M.CONTRASTS],
        queries_per_role=2000,new_research_images=4000,new_training_updates=0,training_seed=1729,source_training_cost_inherited=True,
        resolution=512,UNet_steps=4,CFG=0,IP_scale=1,one_image_per_query=True,batch_queries_per_segment=200,
        strength=1.,radius=.25,profile_indices=list(range(0,2000,200)),fixed_preview_indices=[0,399,600,999,1400,1999],
        maximum_output_bytes=4<<30,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,
        minimum_start_CPU_headroom_bytes=24<<30,minimum_runtime_CPU_headroom_bytes=10<<30,
        legacy_floor_bytes=50<<30,disk_buffer_bytes=8<<30,per_GPU_seconds=900,execution_seconds=21600,
        feature_attempts=3,author_attempts=3,numerical_guard=dict(atol=1e-5,rtol=5e-6),
        official200_previously_exposed=True,independent_H=False,mean_uses_R80_inputs_or_scores=False,
        no_automatic_scientific_retry=True,images_require_actual_per_role_GPU_profile=True,waiting_deadline_epoch=1793696143.411037,
        source_pins=[dict(path=(M.CODE/name).as_posix(),bytes=0,sha256='0'*64) for name in M.ENTRIES],inherited_source_pins=[])


def contract_checks():
    s=specification();assert M.validate(s)==s
    mutations=[('host','new'),('GPU_UUID','other'),('training_seed',2027),('new_training_updates',1),
        ('mean_uses_R80_inputs_or_scores',True),('queries_per_role',80),('UNet_steps',8),('radius',.5),
        ('source_training_cost_inherited',False),('images_require_actual_per_role_GPU_profile',False),
        ('comparisons',[['mean_bounded_F','parent']]),('maximum_output_bytes',3<<30),('waiting_deadline_epoch',1793696144.411037),
        ('official200_previously_exposed',False),('no_automatic_scientific_retry',False)]
    for key,value in mutations:
        changed=copy.deepcopy(s);changed[key]=value
        try:M.validate(changed)
        except AssertionError:pass
        else:raise AssertionError('Accepted wrong scope '+key)
    cases=[((True,True,True,False,False,False),'WAIT_SAME_ORIGINAL_AND_CPU_MEAN_QUEUES'),
        ((False,True,False,True,False,False),'WAIT_SAME_ORIGINAL_AND_CPU_MEAN_QUEUES'),
        ((False,False,True,True,True,False),'WAIT_SAME_ORIGINAL_AND_CPU_MEAN_QUEUES'),
        ((False,False,False,True,False,False),'STOP_INCOMPLETE_DEPENDENCY_NO_RETRY'),
        ((False,False,False,True,True,True),'STOP_FAILED_DEPENDENCY_NO_RETRY'),
        ((False,False,False,True,True,False),'READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS')]
    for args,expected in cases:assert M.classify_dependency(*args)==expected
    pin=dict(path=str(M.CODE/'protocol.json'),bytes=12,sha256='a'*64)
    review=dict(status='PASS_SOURCE_BOUND_CONDITIONAL_MEAN_CONTROL_LIFECYCLE_REVIEW',protocol=pin,source_pins=s['source_pins'])
    def fake_read(path):return review if Path(path).name=='actual_conditional_execution_review.json' else s
    before=set(sys.modules)
    with patch.object(M,'read',fake_read),patch.object(M,'checked',lambda p:Path(p['path'])),patch.object(M,'ref',lambda p:pin),\
         patch.object(M,'dependency',lambda unused:dict(status='WAIT_SAME_ORIGINAL_AND_CPU_MEAN_QUEUES')):
        try:M.setup(str(M.CODE/'protocol.json'))
        except AssertionError:pass
        else:raise AssertionError('Live source reached runtime import')
    assert not any(name.startswith(('torch','diffusers')) for name in set(sys.modules)-before)
    from brainhive_gpu_exit_contract_CPU_20261101_v1 import verify
    good=dict(exit_code=0,status='COMPLETED',wrapper_pid=17)
    assert verify(0,good,17)['success']
    timed=dict(exit_code=-15,status='FAILED_OR_JOB_TIMEOUT_RESUMABLE',wrapper_pid=17,
               termination_reason='Finite registered job timeout; preserve resumable checkpoints')
    assert verify(1,timed,17)['timed_out'] and not verify(1,timed,17)['success']
    for rc,receipt,expected in [(0,timed,17),(1,good,17),(0,good,18)]:
        try:verify(rc,receipt,expected)
        except AssertionError:pass
        else:raise AssertionError('Accepted wrapper/child/owner mismatch')
    return dict(invalid_scopes_rejected=len(mutations),dependency_cases=len(cases),wrapper_child_owner_cases=5,
                actual_setup_rejected_live_before_tensor_import=True)


def mathematical_checks():
    import numpy as np
    import torch
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from audit_HIVE_training_mean_R80_CPU_20261101_v1 import constant_geometry_numpy
    torch.set_num_threads(2);torch.manual_seed(1729)
    rng=np.random.default_rng(1729);u=rng.normal(size=(31,1024));u-=u.mean(-1,keepdims=True)
    w=rng.uniform(.6,1.4,size=(31,1024));b=rng.normal(0,.1,size=(31,1024));c=w*u+b;raw=rng.normal(size=1024)
    model=BoundedConditionAdapter().double().eval().requires_grad_(False)
    with torch.no_grad():model.direction.bias.copy_(torch.from_numpy(raw))
    ct,wt,bt=[torch.from_numpy(value) for value in (c,w,b)]
    with torch.inference_mode():
        native=model(ct,wt,bt)[0].numpy();off=model(ct,wt,bt,enabled=False)[0].numpy()
        zero=BoundedConditionAdapter().double().eval().requires_grad_(False)(ct,wt,bt)[0].numpy()
    explicit=constant_geometry_numpy(c,w,b,raw,np)
    error=float(np.abs(native-explicit).max());assert error<1e-12
    assert np.array_equal(off,c) and np.array_equal(zero,c)
    changed=(native-b)/w
    assert np.allclose(np.linalg.norm(changed,axis=-1),np.linalg.norm(u,axis=-1),atol=2e-13,rtol=2e-14)
    cosine=(changed*u).sum(-1)/(np.linalg.norm(changed,axis=-1)*np.linalg.norm(u,axis=-1))
    assert np.all(np.arccos(np.clip(cosine,-1,1))<=np.arctan(.25)+1e-12)
    assert not np.array_equal(native[0]-c[0],native[1]-c[1])
    return dict(synthetic_queries=31,independent_NumPy_vs_native_FP64_maxerror=error,
                zero_and_disabled_bit_exact=True,norm_and_classical_angular_bound=True,
                constant_raw_field_can_still_make_input_dependent_bounded_action=True,real_R80_or_GPU_test=False)


def statistics_checks():
    import numpy as np
    from brainhive_loss_control_R80_statistics_CPU_20261030_v1 import COMMON,PAPER
    from compare_HIVE_training_mean_R80_CPU_20261101_v1 import comparison_values
    subjects=[f'sub-{i+1:02d}' for i in range(10)];concepts=[f'c{i:03d}' for i in range(200)]
    cells={(sub,con):sub+'/'+con for sub in subjects for con in concepts}
    shifts=dict(parent=0.,bounded_F=.02,ordinary_full1654=.03,mean_bounded_F=.005,mean_ordinary_full1654=.005)
    data={role:{kind:{key:{metric:.4+shift+(index%17)*.0001 for metric in metrics}
                            for index,key in enumerate(cells.values())} for kind,metrics in (('common',COMMON),('paper',PAPER))}
          for role,shift in shifts.items()}
    results,differences,undefined,support=comparison_values(data,subjects,concepts,cells)
    assert not undefined and len(differences)==64 and sum(value.size for value in differences.values())==128000
    for left,right in M.CONTRASTS:
        name=left+'_minus_'+right;expected=shifts[left]-shifts[right]
        for metric,value in results[name].items():
            assert abs(value['paired_left_minus_right']-expected)<1e-14
            assert np.allclose(differences[name,metric],expected,atol=1e-14,rtol=0)
        assert support[name]==(expected>=.01)
    data['mean_bounded_F']['common'][next(iter(cells.values()))]['DINO2AFC']=None
    unknown,limited,undefined,unused=comparison_values(data,subjects,concepts,cells)
    assert len(undefined)==2 and sum(value.size for value in limited.values())==124000
    assert all(unknown[name][metric]['expected']==2000 for name,metric in undefined)
    return dict(fullpanel_cells=128000,contrasts=4,metrics=16,undefined_metrics_keep_full2000=True,
                source_mean_and_mean_parent_signs_verified=True,synthetic_only=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract-only',action='store_true');a=p.parse_args()
    result=dict(status='PASS_TRAIN_MEAN_R80_LOCAL_OR_SERVER_CONTRACT_CHECKS',contract=contract_checks())
    if not a.contract_only:result.update(math=mathematical_checks(),statistics=statistics_checks())
    print(json.dumps(result,allow_nan=False))
