"""Meaningful small transaction corruption and scientific decision tests."""
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
import torch
import brainhive_paired_semantic_study_contract_CPU_20261030_v1 as C

def main():
    assert os.name=='posix';torch.set_num_threads(2);checks=[]
    def rejects(fn):
        try:fn()
        except (AssertionError,ValueError,KeyError):return
        raise AssertionError('Expected fail-closed branch accepted')
    with tempfile.TemporaryDirectory(prefix='paired_semantic_contract_CPU_') as tmp:
        root=Path(tmp);protocol=root/'protocol.json';C.atomic(protocol,dict(engineering_test_only=True))
        arm=C.ARMS[0];folder=root/arm;folder.mkdir();order=np.arange(4,dtype=np.int64)
        pairs=[dict(array_rows=[2*i,2*i+1],T_indices=[2*i,2*i+1]) for i in range(4)]
        def row(i):return dict(update=i+1,pair_index=i,array_rows=pairs[i]['array_rows'],T_indices=pairs[i]['T_indices'],primary_EEG_exposures=2*(i+1),proof=dict(UNet_NFE=4,ancestral=['0'*64]*4,steps=[dict(timestep=t) for t in [999.,749.,499.,249.]]),loss=.3,absolute=.2,pairing=.1,effect=.01,gradients={'EEG':dict(gradient_norm_before_clip=1.),'P':dict(gradient_norm_before_clip=1.)})
        def state(n,rows):return dict(order=order.copy(),updates=n,arm=arm,registration_sha256=C.ref(protocol)['sha256'],log_sha256=hashlib.sha256(b''.join((json.dumps(r,allow_nan=False)+'\n').encode() for r in rows)).hexdigest(),test_tensor=torch.ones(3))
        rows=[row(0),row(1)];p=C.publish_checkpoint(folder,state(2,rows),rows,np.ones((2,2,768),np.float32),None,protocol,pairs,torch,np)
        found=C.scan_checkpoints(folder,protocol,arm,order,pairs,all_state_hashes=True);assert found[-1]['end_update']==2;checks.append('actual_immutable_torch_checkpoint_transaction')
        pending=folder/'checkpoints/step-000003.pending.interrupted';pending.mkdir();(pending/'uncommitted').write_text('preserve')
        assert C.scan_checkpoints(folder,protocol,arm,order,pairs)[-1]['end_update']==2;checks.append('uncommitted_prefix_not_consumed')
        rejects(lambda:C.publish_checkpoint(folder,state(2,rows),rows,np.ones((2,2,768),np.float32),None,protocol,pairs,torch,np));checks.append('committed_state_never_overwritten')
        bad=[row(0),row(1)];bad[1]['pair_index']=0;rejects(lambda:C.validate_rows(bad,order,0,2,pairs));checks.append('duplicate_pair_coverage_rejected')
        log=Path(p['directory'])/'training_log.jsonl';original=log.read_bytes();log.write_bytes(original+b' ')
        rejects(lambda:C.scan_checkpoints(folder,protocol,arm,order,pairs));log.write_bytes(original);checks.append('changed_log_rejected')
        newer=[row(2),row(3)];second=C.publish_checkpoint(folder,state(4,newer),newer,np.ones((2,2,768),np.float32),p,protocol,pairs,torch,np)
        assert C.scan_checkpoints(folder,protocol,arm,order,pairs,all_state_hashes=True)[-1]['end_update']==4;checks.append('complete_chain_joins_exact_order')
        path=Path(p['directory'])/'state.pt';original=path.read_bytes();path.write_bytes(original+b'x')
        rejects(lambda:C.scan_checkpoints(folder,protocol,arm,order,pairs,all_state_hashes=True));path.write_bytes(original);checks.append('older_state_corruption_full_audit_rejects')
        altered=root/'other_protocol.json';C.atomic(altered,dict(engineering_test_only=False))
        rejects(lambda:C.scan_checkpoints(folder,altered,arm,order,pairs));checks.append('wrong_protocol_rejected')
    def contrasts(d,lo,hi,top=.01):
        m={'DINO2AFC':dict(difference=d,descriptive_concept_bootstrap95=[lo,hi]),'within_concept10_top1':dict(difference=top)}
        return {'joint_correct':{a:m for a in ('ordinary',)+C.ARMS[1:]}}
    assert C.image_decision(contrasts(-.03,-.05,-.01),80)['action']=='STOP_INSTANCE';checks.append('fixed80_clear_harm_stops')
    assert C.image_decision(contrasts(.001,-.03,.03),80)['action']=='EXPAND_ALL_TO400';checks.append('fixed80_ambiguity_expands_all')
    assert C.image_decision(contrasts(.02,.01,.03),400)['action']=='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION';checks.append('fixed400_joint_support_gate')
    c=contrasts(.02,.01,.03);c['joint_correct']['joint_swapped_targets']={'DINO2AFC':dict(difference=0,descriptive_concept_bootstrap95=[-.01,.01])};assert C.image_decision(c,400)['action']=='STOP_INSTANCE';checks.append('wrong_target_control_not_exempt')
    assert C.image_decision(contrasts(.02,.01,.03,top=-.01),400)['action']=='STOP_INSTANCE';checks.append('fine_stimulus_loss_not_hidden')
    from audit_HIVE_paired_semantic_study_CPU_20261030_v1 import loss_values,metric_values
    from brainhive_paired_semantic_effect_20261030_v1 import paired_objective
    rng=np.random.default_rng(1729);g=rng.normal(size=(3,2,768)).astype(np.float32);v=rng.normal(size=(3,2,768)).astype(np.float32)
    for arm in C.ARMS:
        values=loss_values(g,v,arm,np)
        for i in range(3):
            target=torch.from_numpy(v[i].copy());target=target.flip(0) if arm=='joint_swapped_targets' else target
            loss,terms=paired_objective(torch.from_numpy(g[i].copy()),target,effect_enabled=arm!='joint_absolute_only')
            assert abs(float(loss)-values['loss'][i])<2e-6 and abs(float(terms['effect'])-values['effect'][i])<2e-6
    checks.append('independent_all5_loss_formulas_match_real_objective')
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    gallery=[dict(stimulus_id=str(i),concept_id=str(i//10)) for i in range(20)];targets=np.eye(20);features=np.stack([targets[0],np.ones(20),targets[17]])
    q=[gallery[0],gallery[12],gallery[17]];actual=metric_values(features,targets,q,gallery,np);expected=S.metric_arrays(features,targets,q,gallery)
    assert np.max(np.abs(actual-np.stack([expected[k] for k in S.METRICS],axis=1)))<2e-12;checks.append('independent_gallery_rank_ties_and_sameconcept10_match')
    assert not torch.cuda.is_initialized();checks.append('all_checks_CPU_only')
    print(json.dumps(dict(status='PASS_SMALL_TRANSACTION_AND_FIXED_SCIENTIFIC_POLICY_CHECKS',checks=checks,new_GPU_jobs=0,real_training_or_images=False)))
if __name__=='__main__':main()
