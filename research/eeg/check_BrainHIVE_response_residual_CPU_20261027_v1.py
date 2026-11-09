"""Actual native response reuse plus differentiable sphere and grouped-loss checks."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_response_residual_20261027_v1 as V
from brainhive_bounded_component_v1 import BoundedConditionAdapter


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(1729);checks=[]
    profile=U.read(Path(__file__).resolve().parent/'audit/cross_encoder_20261027_v1/next_encoder_response_profile/completion.json')
    inputs=U.read(U.checked(profile['inputs']));reference=np.load(U.checked(profile['responses']),allow_pickle=False)
    sys.path[:0]=['/root/atm_official_20261016','/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages','/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages']
    from models.atms import ATMS
    for name,role,ordinal in [('sub-01','oof0',0),('sub-10','full',29)]:
        subject=inputs['subjects'][name];model=ATMS().eval().requires_grad_(False)
        model.load_state_dict(torch.load(U.checked(subject['models'][role]['checkpoint']),weights_only=True,map_location='cpu',mmap=True),strict=True)
        eeg=torch.load(U.checked(subject['input']),weights_only=True,map_location='cpu',mmap=True)
        item=profile['rows'][ordinal];counts=[];handle=model.proj_eeg.register_forward_pre_hook(lambda module,args:counts.append(len(args[0])))
        try:
            with torch.inference_mode():native,response=V.observe_responses(model,eeg[item['array_rows']],torch.full((2,),int(name[-2:]),dtype=torch.long))
        finally:handle.remove()
        assert np.array_equal(response.numpy(),reference[ordinal]) and counts==[2,8] and len(model.enc_eeg[0]._forward_hooks)==0
    checks.append('two original models/four actual EEG inputs: all four responses bitwise equal frozen old CPU profile; only4 extra native head sample evaluations')
    u=torch.nn.functional.layer_norm(torch.randn(20,1024),(1024,));d=torch.randn_like(u)*20
    correction,stats=V.bounded_direction(u,d);v=u+correction
    assert torch.allclose(v.norm(dim=-1),u.norm(dim=-1),atol=1e-5,rtol=1e-6) and stats['used_radius'].max()<=1
    checks.append('native mean-zero tangent, radius<=.25 and original norm preserved')
    old=BoundedConditionAdapter();old.trunk=torch.nn.Identity();old.direction=torch.nn.Identity()
    expected,_=old(u,torch.ones_like(u),torch.zeros_like(u));delta,_=V.bounded_direction(u,u)
    assert torch.equal(expected,u+delta);checks.append('same exact residual sphere arithmetic as frozen ordinary adapter')
    model=V.ResponseAdapter();context=torch.randn(20,5,1024);c=torch.randn_like(u)
    zero,stats=model(c,context,u);assert torch.equal(zero,torch.zeros_like(zero))
    target=torch.randn_like(u);B=torch.randn(1024,1024)*.01
    prediction=(u+zero)@B;loss=(prediction-target).square().mean()+.1*V.local_loss(prediction,target)
    loss.backward();assert model.out.weight.grad.norm()>0 and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    checks.append('zero-init preserves native bytes and actual grouped MSE/within10 loss gives finite nonzero output gradient')
    permutation=torch.roll(torch.arange(20),10);slots=V.contexts(c,c+1,torch.randn(20,4,1024),permutation)
    assert slots['c_only'].shape==slots['paired_responses'].shape==(20,5,1024) and torch.equal(slots['paired_responses'][:,0],slots['shuffled_responses'][:,0])
    checks.append('capacity-matched context shapes; shuffled arm keeps correct final-ATMS token')
    # Grouped loss must agree with the evaluator under identical targets.
    groups=[list(range(10)),list(range(10,20))]
    m=V.OLD.metrics(prediction.detach(),target,groups)
    assert abs(float(V.local_loss(prediction.detach(),target))-float(m['within10_log_loss'].mean()))<1e-6
    checks.append('training within10 CE matches frozen evaluation convention')
    subjects=np.repeat(['s1','s2'],20);concepts=np.tile(np.repeat(['a','b'],10),2)
    values={arm:{key:np.full(40,value) for key,value in [('MSE',1.),('within10_log_loss',2.),('within10_top1',.1),('target_cosine',.2),('within10_cosine_margin',0.)]} for arm in V.ARMS}
    linear={'linear_c':copy.deepcopy(values['c_only']),'linear_joint':copy.deepcopy(values['c_only'])}
    assert not V.readout(values,linear,subjects,concepts)['prediction_gate_passed']
    values['paired_responses']['MSE'][:]=.8;values['paired_responses']['within10_log_loss'][:]=1.8;values['paired_responses']['within10_top1'][:]=.2
    assert V.readout(values,linear,subjects,concepts)['prediction_gate_passed']
    linear['linear_joint']['MSE'][:]=.7;assert not V.readout(values,linear,subjects,concepts)['prediction_gate_passed']
    checks.append('null rejects, true increment passes, stronger linear baseline blocks false promotion')
    assert not torch.cuda.is_initialized()
    result=dict(status='D16_ALL7_NATIVE_RESPONSE_GEOMETRY_LOSS_AND_GATE_CPU_CHECKS_PASSED',source=U.ref(__file__),checks=len(checks),passed=checks,
        parameters_per_head=sum(p.numel() for p in model.parameters()),CUDA_initialized=False,new_real_fit=False,new_images=0)
    U.atomic(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
