"""D17 real native encoder loss/gradient/dropout and grouped metric interface checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_crossbrain_supervision_20261027_v1 as V
import fit_BrainHIVE_crossbrain_supervision_GPU_20261027_v1 as FIT
import brainhive_cross_encoder_20261027_v1 as OLD

def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);assert not torch.cuda.is_initialized();done=U.read(a.teacher);mapping=U.read(U.checked(done['mapping']))
    inputs=U.read(U.checked(done['raw_T_inputs']));selected=np.array(mapping['groups'][:2]).reshape(-1)
    raw=np.load(inputs['subjects']['sub-01']['path'],mmap_mode='r',allow_pickle=False);x=torch.from_numpy(np.array(raw[selected],copy=True))
    contract=U.read('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json');source=next(x for x in contract['subjects'][0]['sources'] if x['role']=='full')
    saved=torch.load(U.checked(dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256'])),weights_only=True,map_location='cpu')
    from brainhive_native_models_v1 import classes
    model=classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1);model.load_state_dict(saved['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==6452224
    model.train();torch.manual_seed(31337);first=model(x);torch.manual_seed(31337);second=model(x);assert torch.equal(first,second)
    torch.manual_seed(31338);assert not torch.equal(first,model(x))
    fbank=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
    visual=fbank[[mapping['T_array_rows'][i] for i in selected]];prob=torch.from_numpy(np.load(U.checked(done['probabilities']),allow_pickle=False)[0,:2].copy())
    wrong=torch.stack([prob[j,V.within_concept_permutation(mapping['concepts'][j])] for j in range(2)])
    c=first.detach().requires_grad_();nce,kl,fine=V.student_losses(c,visual,prob,saved['logit_scale']);g_correct=torch.autograd.grad(nce+.1*kl,c)[0]
    c=first.detach().requires_grad_();n0,k0,_=V.student_losses(c,visual,wrong,saved['logit_scale']);g_wrong=torch.autograd.grad(n0+.1*k0,c)[0]
    assert torch.equal(nce,n0) and torch.isfinite(g_correct).all() and torch.isfinite(g_wrong).all() and (g_correct-g_wrong).norm()>1e-8
    assert torch.allclose(prob.sum(1),wrong.sum(1),atol=1e-6)
    actual=FIT.feature_metrics(first.detach(),visual);expected=OLD.metrics(first.detach(),visual,[list(range(10)),list(range(10,20))])
    differences={k:float(np.abs(actual[k]-expected[k]).max()) for k in actual};assert max(differences.values())<2e-6
    generator=torch.Generator().manual_seed(1729);order=torch.randperm(1323,generator=generator);groups=torch.tensor(mapping['groups']);coverage=np.zeros(13230,np.int8);sizes=[]
    for lo in range(0,1323,100):idx=groups[order[lo:lo+100]].reshape(-1);coverage[idx.numpy()]+=1;sizes.append(len(idx))
    assert np.all(coverage==1) and len(sizes)==14 and sizes[-1]==230
    assert not torch.cuda.is_initialized();result=dict(status='D17_NATIVE_ENCODER_MATCHED_DROPOUT_LOSS_GRADIENT_METRIC_AND_FULL_PASS_CPU_CHECKS_PASSED',checks=6,
        real_raw_T_queries=20,real_native_checkpoint_loaded=True,parameters_per_encoder=6452224,matched_dropout_masks_same_seed=True,
        wrong_teacher_preserves_NCE_and_column_mass=True,correct_wrong_teacher_gradient_difference=float((g_correct-g_wrong).norm()),
        vectorized_metric_max_errors=differences,all13230_once_per_pass=True,updates_per_pass=14,last_batch=230,
        new_parameter_updates=0,new_GPU_jobs=0,CUDA_initialized=False,source=U.ref(__file__),teacher=U.ref(a.teacher))
    U.atomic(a.output,result);print(json.dumps(result))

if __name__=='__main__':main()
