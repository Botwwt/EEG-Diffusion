"""Meaningful pre-fit checks of checkpoint/log transactions and score joins."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import torch
from train_BrainHIVE_paired_effect_20261025_v1 import save_tensor,restore_committed_log
from generate_BrainHIVE_paired_effect_20261025_v1 import verify_prefix
from evaluate_BrainHIVE_paired_effect_CPU_20261025_v1 import join_scores
from queue_BrainHIVE_paired_effect_20261025_v1 import command_spec
from compare_BrainHIVE_functional_V_CPU_v2 import METRICS,crossed_statistics


def rejected(fn):
    try:fn()
    except (AssertionError,ValueError):return
    raise AssertionError('Corruption accepted')


def main():
    torch.set_num_threads(2);torch.manual_seed(1729);checks=[]
    with tempfile.TemporaryDirectory(prefix='hive_paired_contract_') as directory:
        root=Path(directory);log=root/'training.jsonl';committed=b'{"updates":1}\n';tail=b'{"updates":2}\npartial'
        log.write_bytes(committed+tail);saved=dict(log_bytes=len(committed),log_sha256=hashlib.sha256(committed).hexdigest())
        restore_committed_log(log,saved);assert log.read_bytes()==committed
        assert next(root.glob('uncommitted_log_tail_*')).read_bytes()==tail
        log.write_bytes(b'corrupt prefix');rejected(lambda:restore_committed_log(log,saved));checks.append('committed_log_rollback_preserves_suffix_and_rejects_corrupt_prefix')
        first=torch.nn.Linear(5,3);opt=torch.optim.AdamW(first.parameters(),lr=1e-4)
        def update(model,optimizer):
            optimizer.zero_grad();loss=model(torch.randn(4,5)).square().mean();loss.backward();optimizer.step()
        update(first,opt)
        save_tensor(torch,root/'resume.pt',dict(model=copy.deepcopy(first.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=torch.get_rng_state()))
        update(first,opt);expected=copy.deepcopy(first.state_dict())
        second=torch.nn.Linear(5,3);other=torch.optim.AdamW(second.parameters(),lr=1e-4)
        payload=torch.load(root/'resume.pt',weights_only=False);second.load_state_dict(payload['model']);other.load_state_dict(payload['optimizer']);torch.set_rng_state(payload['rng'])
        update(second,other);assert all(torch.equal(expected[k],v) for k,v in second.state_dict().items());checks.append('atomic_checkpoint_optimizer_and_RNG_exact_CPU_roundtrip')
        from PIL import Image
        folder=root/'images';pixel=folder/'pixels/sub-01/test.webp';pixel.parent.mkdir(parents=True);Image.new('RGB',(512,512),(40,50,60)).save(pixel,lossless=True)
        query=dict(query_id='sub-01/test.jpg/R4',subject='sub-01',stimulus_id='test.jpg',array_row=5,image_seed=123)
        record=query|dict(arm='O',global_query_index=0,study_core_sha256='core',generated_path=str(pixel),generated_sha256=hashlib.sha256(pixel.read_bytes()).hexdigest(),component_checkpoint_sha256='head')
        manifest=folder/'inference_manifest.jsonl';manifest.write_text(json.dumps(record)+'\n')
        assert len(verify_prefix(folder,[query],'O',dict(checkpoint=dict(sha256='head')),'core'))==1
        rejected(lambda:verify_prefix(folder,[query|dict(image_seed=124)],'O',dict(checkpoint=dict(sha256='head')),'core'));checks.append('own_arm_prefix_seed_and_file_identity')
        bad=record|dict(generated_sha256='wrong');manifest.write_text(json.dumps(bad)+'\n')
        rejected(lambda:verify_prefix(folder,[query],'O',dict(checkpoint=dict(sha256='head')),'core'));checks.append('changed_saved_image_hash_rejected')
    manifests=[dict(query_id='q',subject='sub-01',stimulus_id='x.jpg',generated_path='/out/x',score_target_path='/target/x',concept_id='c')]
    semantic=[dict(query_id='q',subject='sub-01',stimulus_id='x.jpg',generated_path='/out/x',score_target_path='/target/x',**{k:'0.5' for k in METRICS[:8]})]
    low=[dict(query_id='q',subject='sub-01',stimulus_id='x.jpg',generated_path='/out/x',scoring_target_path='/target/x',**{k:'0.2' for k in METRICS[8:]})]
    joined=join_scores(manifests,semantic,low,METRICS);assert len(joined)==1 and joined[0]['DINO2AFC']==.5
    rejected(lambda:join_scores(manifests,semantic,[low[0]|dict(scoring_target_path='/wrong')],METRICS))
    rejected(lambda:join_scores(manifests,[semantic[0]|dict(DINO2AFC='nan')],low,METRICS))
    rejected(lambda:join_scores(manifests,semantic,[],METRICS));checks.append('score_join_wrong_target_nonfinite_and_missing_rejected')
    import time
    values=[{k:0. for k in METRICS}];effects,_=crossed_statistics(values,manifests,['sub-01'],['c'],deadline=time.monotonic()+30)
    assert all(v['mean']==0 and v['CI95_subject_and_concept_bootstrap']==[0.,0.] for v in effects.values());checks.append('fixed_crossed_statistics_zero_pair_difference')
    config=dict(registration_path='/locked/study.json',functional_job_name='locked_functional_T80',training_job_names={a:[a+'_train',a+'_resume'] for a in ('O','A','D','U')},generation_job_names={a:dict(T32=[a+'_T32'],V=[a+'_V',a+'_V_resume']) for a in ('O','A','D','U')})
    stages=[command_spec(config,a,s) for a in ('O','A','D','U') for s in ('train','postfit','generate','audit','features','score')]+[command_spec(config,None,'functional'),command_spec(config,None,'compare')]
    assert len(stages)==len({s['name'] for s in stages})==26
    assert all(s['maximum_seconds']>0 for s in stages)
    assert command_spec(config,'D','train',1)['command'][-1]=='--resume';checks.append('finite_26_stage_matrix_and_declared_exact_resume')
    from diagnose_BrainHIVE_paired_postfit_T80_20261025_v1 import response_measures
    ti=torch.randn(2,2,4,8);tj=torch.randn_like(ti);si=torch.randn_like(ti);sj=torch.randn_like(ti)
    metrics,valid=response_measures(si,sj,ti,tj,torch);shift=torch.randn_like(si)
    changed,_=response_measures(si+shift,sj+shift,ti,tj,torch)
    assert valid.all() and torch.allclose(metrics['pair_gap'],changed['pair_gap'],atol=1e-5)
    assert torch.allclose(metrics['absolute'],metrics['common']+metrics['contrast']);checks.append('fixed_postfit_pair_projection_and_MSE_decomposition')
    assert not torch.cuda.is_initialized();print(json.dumps(dict(status='PAIRED_PIPELINE_CPU_CHECKS_PASSED',checks=checks,new_fit=False,GPU_used=False)))


if __name__=='__main__':main()
