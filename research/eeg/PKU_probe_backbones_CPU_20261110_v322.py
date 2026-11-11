"""Actual rawEEG forward/gradient, full-batch GradCache and paired-architecture checks."""
import copy,json,os,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
import torch
from PKU_backbone_core_20261110_v322 import *
torch.set_num_threads(4);ROOT.mkdir(exist_ok=True);out=ROOT/'engineering';out.mkdir();start=time.monotonic()
split,T,folds,post=input_contract();raw=torch.load(R/'data/recovery_20261108_v236/sub-03_train_mean4.pt',map_location='cpu',weights_only=True,mmap=True);x=raw[T[:4]].clone();del raw
fullF=torch.load(R/'runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt',map_location='cpu',weights_only=True);target=fullF[T[:4]].clone();del fullF
proofs=[]
for case in CASES:
    torch.manual_seed(1729);a=make(case).train();b=copy.deepcopy(a);s1=nn.Parameter(torch.tensor(2.6592));s2=nn.Parameter(s1.detach().clone());begin=rng_state(x.device)
    values=[encode(a,case,x[i:i+2],post) for i in [0,2]];loss=nce(torch.cat(values),target,s1);loss.backward();final_rng=rng_state(x.device);final_buffers=buffers(a)
    set_rng(begin,x.device);got=cached_backward(b,lambda v:encode(b,case,v,post),x,s2,lambda lo,hi:target[lo:hi],micro=2,mixed=False)
    errors={}
    for (n,p),(n2,q) in zip(a.named_parameters(),b.named_parameters()):
        assert n==n2 and (p.grad is None)==(q.grad is None)
        if p.grad is not None:errors[n]=float((p.grad-q.grad).abs().max()/(1+p.grad.abs().max()))
    assert abs(got-float(loss))<1e-6 and max(errors.values())<5e-5 and abs(float(s1.grad-s2.grad))<1e-6
    assert torch.equal(final_rng['CPU'],rng_state(x.device)['CPU']) and all(torch.equal(v,dict(b.named_buffers())[n]) for n,v in final_buffers.items())
    af=affine(b,case);assert af['gamma'].shape==af['beta'].shape==(1024,)
    proofs.append(dict(case=case,parameters=sum(p.numel() for p in a.parameters()),gradient_parameters=sum(p.numel() for p in a.parameters() if p.grad is not None),GradCache_gradient_max_scaled_error=max(errors.values()),loss_error=abs(got-float(loss)),RNG_and_BN_final_exact=True,real_EEG_queries=4));del a,b,s1,s2
torch.manual_seed(1729);joint=make('Joint63').eval();torch.manual_seed(1729);separated=make('Separated63').eval();assert all(torch.equal(a,b) for a,b in zip(joint.state_dict().values(),separated.state_dict().values()))
assert sum(p.numel() for p in joint.parameters())==sum(p.numel() for p in separated.parameters())
assert joint.space_allowed.sum()==10*63*63 and joint.time_allowed.sum()==63*10*10
done=dict(status='PASS_ALL5_REAL_EEG_GRADCACHE_GRADIENTS_BN_RNG_AND_MATCHED_JOINT_SEPARATED_ARCHITECTURES',CPU_job=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,models=proofs,real_GPU_training_pending=True,new_updates=0,new_images=0,full_effective_contrastive_batch=1024,microbatch=128)
write(out/'completion.json',done);print(json.dumps(done),flush=True)
