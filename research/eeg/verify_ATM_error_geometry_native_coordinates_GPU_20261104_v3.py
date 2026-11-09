"""Frozen all400/all50 native-coordinate verification; old raw-v failure retained.

No weights, targets, sampling policy or output conditions are changed. The old
raw-v test is reported separately, never rewritten as passed.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,gc,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--study-policy',required=True);ap.add_argument('--verification-policy',required=True);args=ap.parse_args()
q=json.loads(Path(args.study_policy).read_bytes());vpolicy=json.loads(Path(args.verification_policy).read_bytes());sys.path.insert(0,q['code_root'])
import atm_error_geometry_contract_20261104_v1 as K
parent,ctx,e,C=K.setup(q);before=K.resources(q,parent,ctx,e,C,starting=True);out=Path(q['output_root']);dest=Path(vpolicy['output_root']);started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.sha(__file__)==vpolicy['sources']['verify']['sha256'] and vpolicy['native_coordinate_tolerance']==5e-5
assert K.ref(args.study_policy)==vpolicy['study_policy'];K.checked(vpolicy['original_CPU_failure']);K.checked(vpolicy['original_controller_failure'])
repo=Path(ctx['base']['ctx']['relative']['repository_root']);job=K.read(repo/'audit'/(vpolicy['job_name']+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==parent['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
import atm_error_geometry_model_20261104_v1 as S
import atm_error_geometry_math_20261104_v1 as M
from diffusers import DDPMScheduler
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
stats=dict(np.load(K.checked(K.read(out/'statistics_completion.json')['statistics']),allow_pickle=False));st=S.tensors(stats,'cuda');sc=S.tensors(stats,'cpu');alphas=DDPMScheduler().alphas_cumprod
native=S.original_module(K.checked(ctx['base']['parent']['native_prior_source']));prepared=K.read(out/'prepared.json');values=np.load(K.checked(prepared['conditions']),allow_pickle=False);queries=K.read(K.checked(prepared['queries']))
assert values.shape==(400,9,1024);direct=values[:,2];basis64=stats['basis'];basis_cpu=sc['basis'];all_metrics=[];summary=[];exact_trace_keys=('states','predictions','clean')
def guard():
    assert time.monotonic()-started<850 and time.time()<vpolicy['absolute_deadline_epoch']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<parent['maximum_extra_CPU_bytes']
    assert torch.cuda.max_memory_allocated()<parent['maximum_GPU_bytes'];return K.resources(q,parent,ctx,e,C)
try:
    for mode in K.MODELS:
        fit=K.read(out/'fits_recovery_v2'/mode/'completion.json');checkpoint=K.checked(fit['checkpoint']);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert state['updates']==1950 and state['epochs']==150 and torch.all(state['coverage']==150) and state['epoch_log']==K.ref(out/'fits_recovery_v2'/mode/'epoch_log.jsonl')
        cpu=native.DiffusionPriorUNet(cond_dim=1024,dropout=.1).eval().requires_grad_(False);cpu.load_state_dict(state['model'],strict=True)
        assert len(state['optimizer']['state'])==len(list(cpu.parameters())) and state['noise_RNG'].numel()>0 and state['CUDA_RNG'].numel()>0
        net=native.DiffusionPriorUNet(cond_dim=1024,dropout=.1).eval().requires_grad_(False);net.load_state_dict(state['model'],strict=True);net.cuda();del state
        original_clean=S.clean;metrics=np.zeros((400,50,5),np.float64);exact_all400=True;exact_first2=True;rotation_error=0.
        for lo in range(0,400,128):
            raw=torch.from_numpy(direct[lo:lo+128].copy()).cuda();captured=[]
            def observe(*a,**kw):
                dd,vv,pp=original_clean(*a,**kw)
                captured.append(dict(z=a[2].cpu(),t=a[3].cpu(),c=a[4].cpu(),network_input=pp['network_input'].cpu(),prediction=vv.cpu(),clean=dd.cpu(),native_clean=(dd@st['basis'].T).cpu()))
                return dd,vv,pp
            S.clean=observe
            try:result,trace=S.sample(mode,net,raw,st,[1729+x['array_row'] for x in queries[lo:lo+len(raw)]],trace=lo==0)
            finally:S.clean=original_clean
            original=values[lo:lo+len(raw),K.ARMS.index(mode)];assert np.array_equal(result.cpu().numpy(),original),'Frozen original all400 output changed during observation replay'
            assert len(captured)==50
            if lo==0:
                old=np.load(out/(mode+'_first2_all50_states.npz'),allow_pickle=False)
                for key,field in (('states','state'),('predictions','prediction'),('clean','clean')):
                    assert np.array_equal(np.stack([x[field] for x in trace]),old[key]),'Original first2 full50 trajectory changed'
            canonical_c=direct[lo:lo+len(raw)].astype(np.float64)@basis64
            rotation_error=max(rotation_error,float(np.max(abs(captured[0]['c'].numpy().astype(np.float64)-canonical_c))))
            for j,x in enumerate(captured):
                with torch.inference_mode():d_cpu,v_cpu,p_cpu=original_clean(mode,cpu,x['z'],x['t'],x['c'],sc,alphas)
                cpu_native=(d_cpu@basis_cpu.T).numpy();gpu_native=x['native_clean'].numpy();gpu_prediction=x['prediction'].numpy()
                raw_error=np.max(abs(v_cpu.numpy()-gpu_prediction)/(1+abs(gpu_prediction)),axis=1)
                clean_error=np.max(abs(cpu_native-gpu_native)/(1+abs(gpu_native)),axis=1)
                eigen_error=np.max(abs(d_cpu.numpy()-x['clean'].numpy())/(1+abs(x['clean'].numpy())),axis=1)
                input_error=np.max(abs(p_cpu['network_input'].numpy()-x['network_input'].numpy())/(1+abs(x['network_input'].numpy())),axis=1)
                if mode=='native_v':
                    aa=np.sqrt(alphas[x['t']].numpy().astype(np.float64))[:,None];bb=np.sqrt(1-alphas[x['t']].numpy().astype(np.float64))[:,None]
                    independent=(aa*x['z'].numpy()-bb*gpu_prediction)@basis64.T
                else:
                    lam=stats[{'iso_error':'isotropic_eigenvalues','paired_error':'eigenvalues','wrong_error':'wrong_eigenvalues'}[mode]]
                    aa,bb,ss,AA,BB=M.coefficients(alphas[x['t']].numpy(),lam);mm=x['c'].numpy().astype(np.float64)+stats['mean_error_rotated'];uu=(x['z'].numpy()-aa*mm)/ss
                    independent=(mm+np.sqrt(lam)*(AA*uu-BB*gpu_prediction))@basis64.T
                math_error=np.max(abs(independent-gpu_native)/(1+abs(gpu_native)),axis=1)
                metrics[lo:lo+len(raw),j]=np.stack((raw_error,clean_error,eigen_error,input_error,math_error),axis=1)
            assert metrics[lo:lo+len(raw),:,1].max()<5e-5 and metrics[lo:lo+len(raw),:,4].max()<5e-5
            guard();K.write(dest/'progress.json',dict(arm=mode,completed_queries=min(lo+128,400),all50=True,raw_v_original_threshold_not_silently_waived=True))
            del captured,result,trace;gc.collect()
        maxima=metrics.max((0,1));summary.append(dict(arm=mode,all400_GPU_outputs_byte_exact=True,original_first2_all50_states_predictions_clean_byte_exact=True,
            CPU_neural_raw_v_scaled_max=float(maxima[0]),native_clean_CPU_scaled_max=float(maxima[1]),eigen_clean_CPU_scaled_max=float(maxima[2]),
            preconditioned_input_CPU_scaled_max=float(maxima[3]),independent_NumPy64_native_clean_scaled_max=float(maxima[4]),
            GPU_rotation_vs_NumPy64_maxabs=rotation_error,original_raw_v_criterion_passed=bool(maxima[0]<5e-5),native_tolerance=5e-5,
            final_optimizer_RNG_coverage150_verified=True))
        all_metrics.append(metrics);del net,cpu;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(dest/'all400_all50_CPU_native_errors.npz',metrics=np.stack(all_metrics),arms=np.array(K.MODELS),
        fields=np.array(['raw_v_scaled','native_clean_scaled','eigen_clean_scaled','input_scaled','NumPy_native_clean_scaled']))
    proof=dict(status='PASS_ALL400_ALL50_NATIVE_COORDINATE_REPLAY_ORIGINAL_RAW_V_FAILURE_RETAINED',UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        study_policy=K.ref(args.study_policy),verification_policy=K.ref(args.verification_policy),prepared=K.ref(out/'prepared.json'),original_CPU_failure=vpolicy['original_CPU_failure'],
        original_raw_v_gate_NOT_claimed_passed=True,criteria_amended_before_any_research_image_or_effect_read=True,all400_all4_original_GPU_final_conditions_byte_exact=True,
        CPU_teacher_forced_neural_sample_forwards=80000,all4x400x50_NumPy_native_clean_transforms=True,summary=summary,raw_errors=K.ref(dest/'all400_all50_CPU_native_errors.npz'),
        native_clean_tolerance_unchanged=5e-5,all_models_and_conditions_unchanged=True,new_training_updates=0,new_research_images=0,
        CPU_sampler_trajectories_not_claimed=True,seconds=time.monotonic()-started,resources_after=guard())
    K.write(dest/'completion.json',proof);print(json.dumps(proof),flush=True)
except BaseException as exc:
    K.write(dest/'failure.json',dict(status='FROZEN_NATIVE_COORDINATE_VERIFICATION_FAILED_PRESERVED',error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
