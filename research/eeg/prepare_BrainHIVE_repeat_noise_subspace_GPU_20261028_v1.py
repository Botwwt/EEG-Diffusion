"""D23 role-private4250dim repeat metrics and actual native-gradient profile."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_repeat_noise_subspace_20261027_v1 as V

def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    return x

def main():
    p=argparse.ArgumentParser()
    for key in ('spec','native-registration','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    assert a.max_seconds==180 and not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();stop=[False];stage='INPUTS'
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    try:
        s=U.read(a.spec)
        for ref in s['source_pins']:U.checked(ref)
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(a.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;cfg=C.read(a.native_registration)
        doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=180,maximum_output_bytes=2048<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stop[0] and time.monotonic()-start<160;budget.check()
        def log(v):
            v=v|dict(seconds=time.monotonic()-start)
            with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
            print(json.dumps(v),flush=True)
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        from brainhive_native_models_v1 import classes
        BrainMLP=classes()['BrainMLP'];trials=np.load(U.checked(s['trials']),mmap_mode='r',allow_pickle=False);raw=np.load(U.checked(s['T_mean4']),mmap_mode='r',allow_pickle=False).reshape(13230,4250)
        targets=np.load(U.checked(s['targets']),mmap_mode='r',allow_pickle=False);assert trials.shape==(13230,4,17,250) and targets.shape==(13230,1024)
        # Physical means must reproduce the exact original native input source.
        for lo in range(0,13230,512):
            check();value=torch.from_numpy(np.array(trials[lo:lo+512],copy=True)).mean(1).reshape(-1,4250).numpy();assert np.array_equal(value,raw[lo:lo+512])
        geometries={};profiles=[];perm=torch.randperm(4250,generator=torch.Generator().manual_seed(1729),device='cpu').cuda()
        for role,item in s['roles'].items():
            check();stage='ROLE_'+role;ix=np.asarray(item['fit_positions']);n=len(ix);outer=torch.zeros((4250,4250),device='cuda',dtype=torch.float64)
            for lo in range(0,n,256):
                check();t=torch.from_numpy(np.array(trials[ix[lo:lo+256]],copy=True)).cuda().double().reshape(-1,4,4250);r=t-t.mean(1,keepdim=True);r=r.flatten(0,1);outer.add_(r.T@r)
            noise=V.shrink_mean4_noise(outer,n);del outer,t,r
            old=torch.load(U.checked(item['checkpoint']),weights_only=True,map_location='cpu');model=BrainMLP(4250,1024,num_layers=1,dropout=.1).cuda().eval().requires_grad_(False);model.load_state_dict(old['model'],strict=True)
            w=model.proj_in.weight.detach().clone();geos={}
            for name,metric in [('repeat',noise),('permuted',noise[perm][:,perm])]:
                check();g=V.geometry(w,metric);path=a.output/(role+'_'+name+'_geometry.pt');torch.save(cpu(g),path);geos[name]=U.ref(path)
                assert sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())<(2048<<20)-(1<<20)
                chosen=ix[:1024];rawbatch=torch.from_numpy(np.array(raw[chosen],copy=True)).cuda();target=torch.from_numpy(np.array(targets[chosen],copy=True)).cuda();white=V.whiten(rawbatch,g)
                for constrained in ([True,False] if name=='repeat' else [True]):
                    arm=('repeat_noise_constrained' if constrained else 'repeat_noise_unconstrained') if name=='repeat' else 'permuted_noise_constrained';q=g['Q0'].clone().requires_grad_(True)
                    with torch.no_grad():
                        before=model(rawbatch);initial=model.block(V.residual_preactivation(rawbatch,white,q,g,model.proj_in.bias));assert torch.equal(before,initial) and torch.equal(V.folded_weight(q,g),w)
                    model.train();torch.cuda.manual_seed(1729);tic=time.monotonic()
                    pred=model.block(V.residual_preactivation(rawbatch,white,q,g,model.proj_in.bias));logits=old['logit_scale'].exp().clamp_max(100).cuda()*F.normalize(pred,dim=-1)@F.normalize(target,dim=-1).T;labels=torch.arange(1024,device='cuda');loss=.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels));loss.backward();assert torch.isfinite(q.grad).all()
                    with torch.no_grad():
                        updated,step=V.fixed_step(q,q.grad,constrained);fold=V.folded_weight(updated,g);error=V.metric_error(fold,g)
                        if constrained:assert error<2e-5
                        model.eval();expanded=model.block(V.residual_preactivation(rawbatch,white,updated,g,model.proj_in.bias));model.proj_in.weight.copy_(fold);folded=model(rawbatch)
                        rel=float(((folded-expanded).norm(dim=1)/expanded.norm(dim=1)).max());assert rel<3e-5;model.proj_in.weight.copy_(w)
                    profiles.append(dict(role=role,arm=arm,profile_queries=1024,temporary_updates=1,loss=float(loss.detach()),native_zero_condition_bitwise=True,zero_fold_weight_bitwise=True,folded_condition_relative_max=rel,used_regularized_noise_metric_relative_error=error,step=step,seconds=time.monotonic()-tic))
                    log(profiles[-1]);del q,updated,pred,folded,expanded,fold
                del g,rawbatch,target,white
            geometries[role]=geos;del model,w,noise,old;torch.cuda.empty_cache()
        U.save_array(a.output/'noise_coordinate_permutation.npy',perm.cpu().numpy());check()
        done=dict(status='D23_ALL3_NATIVE_ROLES_REPEAT_METRICS_AND9_ACTUAL_GRADIENT_PROFILES_PASSED',spec=U.ref(a.spec),source=U.ref(__file__),wrapper=wrapper,geometries=geometries,noise_coordinate_permutation=U.ref(a.output/'noise_coordinate_permutation.npy'),profiles=profiles,
            all13230_physical_mean4_values_bitwise_original=True,noise_uses_only_each_role_fit_concepts=True,normal_inference_individual_trials=False,new_final_trained_models=0,temporary_profile_updates=9,new_UNet_calls=0,new_images=0,formal_D_resumed=False,
            seconds=time.monotonic()-start,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes)
        U.atomic(a.output/'completion.json',done);print(json.dumps({k:v for k,v in done.items() if k not in ('profiles','geometries')}))
    except BaseException as exc:U.atomic(a.output/'failure.json',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise

if __name__=='__main__':main()
