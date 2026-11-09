"""CPU checks of scope, exact dependency gating and common-offset diagnostics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import VAR_multiscale_profile_contract_CPU_20261101_v1 as C


def fixture():
    return dict(status='REVIEWED_CONDITIONAL_T8_ALL1000_FULL680_NATIVE_VAR_ENGINEERING',host='old',GPU_UUID=C.GPU,
        code_root=C.CODE.as_posix(),output_root=C.OUT.as_posix(),existing_control_protocol=C.MEAN_PROTOCOL,
        existing_control_contract=C.MEAN_CONTRACT,existing_control_PID=493907,existing_control_start_ticks=114931939,
        fixed_T_positions=C.POSITIONS,numerical_probe_classes=C.CLASSES,all_native_classes=list(range(1000)),class_batch=8,
        scales=[1,2,3,4,5,6,8,10,13,16],tokens=680,seed=1729,top_k=600,CFG=0,
        maximum_output_bytes=128<<20,maximum_GPU_bytes=20<<30,maximum_extra_CPU_bytes=10<<30,
        minimum_start_CPU_headroom_bytes=24<<30,minimum_runtime_CPU_headroom_bytes=10<<30,
        per_GPU_seconds=900,soft_GPU_seconds=780,execution_seconds=1800,waiting_deadline_epoch=1793699743.411037,
        numerical_guard=dict(atol=1e-4,rtol=2e-5),constant_class_sampled_tokens_and_pixels_must_match=True,
        no_EEG_fit_or_research_images=True,new_optimizer_updates=0,pretrained_precision='FP32_no_autocast_no_TF32',
        no_automatic_retry=True,full_training_bank_or_EEG_training_admitted=False,
        native_module_roots=[(C.BASE/'EEG_native_conditional_variational_20261029_v1/code').as_posix(),(C.BASE/'AR_anchored_training_code_v1').as_posix()],
        source_pins=[dict(path=(C.CODE/n).as_posix(),bytes=1,sha256='0'*64) for n in C.ENTRIES],
        assets=dict(author_inventory=dict(sha256='c0d6b113c99f69a96036e95520fe64ccae7507bafd2674bfcb9b38024fe0f8e3'),
            public_VQ=dict(sha256='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c'),
            public_VAR=dict(sha256='4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb'),
            data=dict(tokens={},token_identities={},token_completion={})))


def rejected(call):
    try:call()
    except (AssertionError,ValueError):return
    raise AssertionError('Expected refusal was not observed')


def run(output=None,protocol=None):
    assert 'torch' not in sys.modules
    checks=[];s=fixture();assert C.validate(s)==s;checks.append('complete_T8_all1000_all680_fixed_pretrained_scope_accepted')
    changes=dict(host='new',tokens=5,all_native_classes=list(range(16)),class_batch=16,
        fixed_T_positions=list(range(8)),seed=2027,CFG=3,top_k=5,maximum_output_bytes=64<<20,
        maximum_GPU_bytes=24<<30,minimum_start_CPU_headroom_bytes=16<<30,new_optimizer_updates=1,
        no_automatic_retry=False,pretrained_precision='BF16',full_training_bank_or_EEG_training_admitted=True,
        constant_class_sampled_tokens_and_pixels_must_match=False)
    for key,value in changes.items():
        wrong=copy.deepcopy(s);wrong[key]=value;rejected(lambda:C.validate(wrong))
    checks.append('16_scope_precision_seed_resource_or_retry_mutations_rejected')
    for live,worker,up,complete,failed,expected in [
        (True,False,'READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS',True,False,'WAIT_SAME_THREE_EXISTING_QUEUES'),
        (False,True,'READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS',True,False,'WAIT_SAME_THREE_EXISTING_QUEUES'),
        (False,False,'WAIT_SAME_ORIGINAL_AND_CPU_MEAN_QUEUES',False,False,'WAIT_SAME_THREE_EXISTING_QUEUES'),
        (False,False,'READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS',True,False,'READY_ALL_THREE_COMPLETE_AND_EXITED'),
        (False,False,'READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS',False,False,'STOP_EXISTING_CONTROL_INCOMPLETE_NO_RETRY'),
        (False,False,'READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS',True,True,'STOP_EXISTING_DEPENDENCY_FAILED_NO_RETRY'),
        (False,False,'STOP_INCOMPLETE_DEPENDENCY_NO_RETRY',False,False,'STOP_EXISTING_DEPENDENCY_FAILED_NO_RETRY')]:
        assert C.classify_dependency(live,worker,up,complete,failed)==expected
    checks.append('exact_live_controller_or_worker_waits_terminal_failure_or_missing_completion_stops')
    import numpy as np
    from audit_VAR_multiscale_native_profile_CPU_20261101_v1 import summarize
    common=np.zeros((8,1000,10),np.float64)
    zero=summarize(common)
    assert np.max(np.abs(zero['uniform_prior_per_scale_KL_to_uniform']))<1e-12
    rng=np.random.default_rng(1729);values=rng.normal(size=common.shape)
    offsets=rng.normal(size=(8,1,10))*20
    a=summarize(values);b=summarize(values+offsets)
    for key in ('uniform_prior_per_scale_entropy','coarse5_vs_full680_global_class_posterior_total_variation','cumulative_uniform_prior_entropy'):
        assert np.allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
    checks.append('class_independent_scale_NLL_offset_is_not_mistaken_for_more_class_evidence')
    values=np.zeros((8,1000,10),np.float64);values[:,0,:2]=2;values[:,1,2:]=2
    result=summarize(values)
    assert all(result['coarse5_vs_full680_global_class_argmax_changed'])
    assert min(result['coarse5_vs_full680_global_class_posterior_total_variation'])>0
    checks.append('different_late_scale_class_evidence_changes_full_posterior_without_asserting_EEG_gain')
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'source.py';p.write_bytes(b'actual bytes\n')
        pin=C.ref(p);assert C.checked(pin)==p
        wrong=dict(pin,sha256='1'*64);rejected(lambda:C.checked(wrong))
        p.write_bytes(b'changed bytes\n');rejected(lambda:C.checked(pin))
    checks.append('real_file_SHA_or_bytes_mutation_rejected')
    prior=dict(required_remaining_bytes=100<<20,effective_CPU_headroom_bytes=30<<30,
        used_output_bytes=1,free_bytes=1,parent_remaining_budget_retained=8<<30,physical_reservation=False)
    fake=types.SimpleNamespace(validate=lambda x:x,resources=lambda *x,**kw:prior,allocated=lambda p:1<<20)
    with (patch.object(C,'mean_contract',return_value=fake),patch.object(C,'checked',return_value=Path('unused')),
          patch.object(C,'read',return_value={}),patch.object(C.shutil,'disk_usage',return_value=types.SimpleNamespace(free=512<<20))):
        result=C.resources(s,starting=True);assert result['required_remaining_bytes']==226<<20
        with patch.object(C.shutil,'disk_usage',return_value=types.SimpleNamespace(free=(226<<20)-1)):
            rejected(lambda:C.resources(s))
        prior['effective_CPU_headroom_bytes']=23<<30;rejected(lambda:C.resources(s,starting=True))
        assert C.resources(s,starting=False)['effective_CPU_headroom_bytes']==23<<30
    checks.append('new128MiB_budget_added_to_all_existing_promises_with_unchanged_start_and_runtime_CPU_floors')
    if protocol is not None:
        registered=json.loads(Path(protocol).read_bytes());C.validate(registered)
        local=Path(__file__).resolve().parent
        for pin in registered['source_pins']:
            f=local/Path(pin['path']).name;assert f.stat().st_size==pin['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==pin['sha256']
        checks.append('actual_new_protocol_matches_all_seven_local_sources')
        if Path(protocol)==C.CODE/'protocol.json':
            gate=C.dependency(registered)
            if gate['status'].startswith('WAIT_'):
                original_read=C.read
                def read_with_review(path):
                    if Path(path)==C.CODE/'actual_conditional_engineering_review.json':
                        return dict(status='PASS_SOURCE_BOUND_CONDITIONAL_NATIVE_VAR_ENGINEERING_REVIEW',protocol=C.ref(protocol),source_pins=registered['source_pins'])
                    return original_read(path)
                with patch.object(C,'read',side_effect=read_with_review):rejected(lambda:C.setup(protocol))
                checks.append('actual_OLD_source_complete_setup_refuses_live_dependency_before_Torch_or_CUDA')
    assert 'torch' not in sys.modules
    result=dict(status='PASS_NATIVE_VAR_ENGINEERING_CONTRACT_AND_INDEPENDENT_DIAGNOSTIC_CPU_ONLY',checks=checks,
        actual_pretrained_model_or_EEG_data_tested=False,new_GPU_jobs=0,new_optimizer_updates=0,new_research_images=0)
    if output:
        with Path(output).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--protocol');a=p.parse_args();run(a.output,a.protocol)
