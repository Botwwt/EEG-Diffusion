"""All-T statistics, real engineering, fixed final fits, and EEG-only export."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,datetime,fcntl,gc,hashlib,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_error_geometry_contract_20261104_v1 as K
import atm_error_geometry_math_20261104_v1 as M
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('statistics','profile','fit','prepare','CPU'),required=True);ap.add_argument('--arm',choices=K.MODELS);a=ap.parse_args()
q=K.read(a.policy);out=Path(q['output_root']);started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
parent,ctx,e,C=K.setup(q);before=K.resources(q,parent,ctx,e,C,starting=True);base=ctx['base'];repo=Path(base['ctx']['relative']['repository_root'])
assert K.sha(__file__)==q['sources']['worker']['sha256']
def guard():
    assert time.monotonic()-started<855
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<parent['maximum_extra_CPU_bytes']
    return K.resources(q,parent,ctx,e,C)
def progress(phase,**values):K.write(out/'worker_progress.json',dict(phase=phase,mode=a.mode,arm=a.arm,seconds=time.monotonic()-started,**values))
import torch
torch.set_num_threads(2)
def cpu_tree(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu_tree(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu_tree(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu_tree(x) for x in v)
    return v
def gpu_enter(label):
    job=K.read(repo/'audit'/(label+'_job.json'));assert job['pid']==os.getpid() and job['wrapper_pid']==os.getppid() and job['status']=='RUNNING'
    lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==parent['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
    return lock
def stats_np():return dict(np.load(K.checked(K.read(out/'statistics_completion.json')['statistics']),allow_pickle=False))
def train_inputs():
    xdone=K.read(Path(ctx['c']['s']['output_root'])/'GPU_completion.json');source=K.checked(xdone['conditions'])
    c=np.load(source,mmap_mode='r',allow_pickle=False);assert c.shape==(13230,2,1024)
    teacher=torch.load(K.checked(base['parent']['original_visual_target']),map_location='cpu',weights_only=True,mmap=True)
    T=base['T'];assert len(T)==13230 and len(set(T))==13230
    roles=np.load(Path(ctx['c']['s']['output_root'])/'roles_and_rows.npz',allow_pickle=False)
    assert roles['array_rows'].tolist()==T and np.all(roles['seen_encoder_role']!=roles['unseen_encoder_role'])
    for b in base['banks']:
        p=K.read(K.checked(b['permissions']));assert p['export_is_T_OOF'] and not {r//10 for r in p['training_rows']}&{r//10 for r in p['export_rows']}
    return np.array(c[:,1],copy=True),teacher[T].numpy().copy(),T,xdone['conditions']
try:
    if a.mode=='statistics':
        assert not torch.cuda.is_initialized() and not (out/'statistics_completion.json').exists()
        cc,ff,T,source=train_inputs();stats=M.statistics(cc,ff)
        assert stats['orthogonality_maxabs']<1e-10 and stats['covariance_reconstruction_maxabs']<1e-10
        assert np.array_equal(np.sort(stats['wrong_eigenvalues']),np.sort(stats['eigenvalues']))
        with (out/'statistics.npz').open('xb') as f:np.savez_compressed(f,**stats,array_rows=np.array(T))
        fixed=np.linspace(0,13229,16,dtype=int);np.savez_compressed(out/'fixed_T_math_inputs.npz',conditions=cc[fixed],teacher=ff[fixed],array_rows=np.array(T)[fixed])
        from diffusers import DDPMScheduler
        alpha=DDPMScheduler().alphas_cumprod.double().numpy();rng=np.random.default_rng(1729);noise=rng.normal(size=(16,1024));f=ff[fixed].astype(np.float64)@stats['basis'];c=cc[fixed].astype(np.float64)@stats['basis']
        error=0.;bound_error=0.
        for key in ('eigenvalues','isotropic_eigenvalues','wrong_eigenvalues'):
            for lo in range(0,1000,16):
                x=alpha[lo:lo+16]
                if len(x)<16:x=np.pad(x,(0,16-len(x)),mode='edge')
                z=M.target_and_clean(f,c,noise,x,stats['mean_error_rotated'],stats[key]);error=max(error,float(abs(z['clean']-f).max()))
                error=max(error,float(abs(z['A']**2+z['B']**2-1).max()))
                bound_error=max(bound_error,float(np.maximum(stats[key]*z['B']**2-stats[key],0).max()))
        assert error<1e-9 and bound_error==0
        K.write(out/'statistics_completion.json',dict(status='COMPLETE_ALL13230_GENUINE_OOF_EEG_ERROR_GEOMETRY_T_ONLY',policy=K.ref(a.policy),statistics=K.ref(out/'statistics.npz'),
            original_unseen_conditions=source,visual_teacher=base['parent']['original_visual_target'],rows=T,training_queries=13230,training_concepts=1323,
            target_maximum_norm=float(stats['target_maximum_norm']),residual_second_moment=float(stats['residual_second_moment']),
            eigenvalue_minimum=float(stats['eigenvalues'].min()),eigenvalue_maximum=float(stats['eigenvalues'].max()),eigenvalue_floor=float(stats['eigenvalue_floor']),
            all1000_time_forward_inverse_maxabs=error,clean_error_gain_bound_max_violation=bound_error,all_three_spectra_positive=True,
            no_D_or_V_targets_used_for_statistics=True,physiological_or_conditional_noise_covariance_claimed=False,seconds=time.monotonic()-started,resources_after=guard()))
    elif a.mode=='profile':
        lock=gpu_enter(q['job_prefix']+'_profile');stats=stats_np();cc,ff,T,_=train_inputs();s=__import__('atm_error_geometry_model_20261104_v1')
        native=s.original_module(K.checked(base['parent']['native_prior_source']));st=s.tensors(stats,'cuda')
        from diffusers import DDPMScheduler
        alphas=DDPMScheduler().alphas_cumprod.cuda();c=torch.from_numpy(cc[:16]).cuda()@st['basis'];f=torch.from_numpy(ff[:16]).cuda()@st['basis']
        t=torch.tensor([0,20,500,980]*4,device='cuda');noise=torch.randn(f.shape,generator=torch.Generator(device='cuda').manual_seed(1729),device='cuda');proofs=[]
        for mode in K.MODELS:
            torch.manual_seed(1729);net=native.DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda();torch.nn.init.zeros_(net.output_layer.weight);torch.nn.init.zeros_(net.output_layer.bias)
            assert sum(p.numel() for p in net.parameters())==9675648;opt=torch.optim.Adam(net.parameters(),lr=.001)
            for step in range(4):
                net.train();z,p,y=s.target(mode,f,c,noise,t,st,alphas);v=net(p['network_input'],t.float(),c);loss=(v-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(grad);opt.step()
            net.eval();nonzero=sum(int(p.grad is not None and p.grad.count_nonzero()>0) for p in net.parameters());assert nonzero==len(list(net.parameters()))
            with torch.inference_mode():d,v,p=s.clean(mode,net,z,t,c,st,alphas)
            cpu=native.DiffusionPriorUNet(cond_dim=1024,dropout=.1).eval();cpu.load_state_dict(cpu_tree(net.state_dict()),strict=True)
            with torch.inference_mode():dc,vc,pc=s.clean(mode,cpu,z.cpu(),t.cpu(),c.cpu(),s.tensors(stats,'cpu'),alphas.cpu())
            err=float(((dc-d.cpu()).abs()/(1+d.cpu().abs())).max());assert err<3e-5
            if mode!='native_v':
                _,_,_,AA,BB=M.coefficients(alphas[t].cpu().numpy(),stats[{'iso_error':'isotropic_eigenvalues','paired_error':'eigenvalues','wrong_error':'wrong_eigenvalues'}[mode]])
                coefferr=float(max(abs(AA-p['A'].cpu().numpy()).max(),abs(BB-p['B'].cpu().numpy()).max()));assert coefferr<2e-6
            else:coefferr=0.
            proofs.append(dict(mode=mode,temporary_updates=4,parameters=9675648,all_parameter_gradients_nonzero_after4=True,CPU_GPU_clean_scaled_error=err,NumPy64_coefficients_maxabs=coefferr))
            del net,opt,cpu,d,v,dc,vc,pc,p,y;gc.collect();torch.cuda.empty_cache();guard()
        K.write(out/'profile_completion.json',dict(status='PASS_REAL_FOUR_MODELS_ALL_T_STATISTICS_NONZERO_GRADIENTS_AND_CPU_GPU_CLEAN_REPLAY',proofs=proofs,
            temporary_models_discarded_not_final_fits=True,peak_GPU_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,resources_after=guard()))
    elif a.mode=='fit':
        assert a.arm in K.MODELS;lock=gpu_enter(q['job_prefix']+'_fit_'+a.arm)
        assert K.read(out/'profile_completion.json')['status']=='PASS_REAL_FOUR_MODELS_ALL_T_STATISTICS_NONZERO_GRADIENTS_AND_CPU_GPU_CLEAN_REPLAY'
        directory=out/'fits'/a.arm;directory.mkdir(parents=True,exist_ok=False);stats=stats_np();cc,ff,T,input_pin=train_inputs()
        import atm_error_geometry_model_20261104_v1 as s
        native=s.original_module(K.checked(base['parent']['native_prior_source']));st=s.tensors(stats,'cuda');c=torch.from_numpy(cc).cuda()@st['basis'];f=torch.from_numpy(ff).cuda()@st['basis'];del cc,ff
        from diffusers import DDPMScheduler
        from diffusers.optimization import get_cosine_schedule_with_warmup
        alphas=DDPMScheduler().alphas_cumprod.cuda();torch.manual_seed(1729)
        net=native.DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda();torch.nn.init.zeros_(net.output_layer.weight);torch.nn.init.zeros_(net.output_layer.bias)
        assert sum(p.numel() for p in net.parameters())==9675648
        opt=torch.optim.Adam(net.parameters(),lr=.001);schedule=get_cosine_schedule_with_warmup(opt,500,1950)
        ordering=torch.Generator(device='cpu').manual_seed(1729);noise_rng=torch.Generator(device='cuda').manual_seed(1729);coverage=torch.zeros(13230,dtype=torch.int32);updates=exposures=0
        for epoch in range(150):
            net.train();order=torch.randperm(13230,generator=ordering);loss_sum=clean_sum=grad_sum=0.;epoch_start=time.monotonic();time_sums=np.zeros((4,3),np.float64)
            for lo in range(0,13230,1024):
                ix_cpu=order[lo:lo+1024];ix=ix_cpu.cuda();noise=torch.randn((len(ix),1024),generator=noise_rng,device='cuda');t=torch.randint(0,1000,(len(ix),),generator=noise_rng,device='cuda')
                z,p,target=s.target(a.arm,f[ix],c[ix],noise,t,st,alphas);prediction=net(p['network_input'],t.float(),c[ix]);loss=(prediction-target).square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(grad)
                opt.step();schedule.step();updates+=1;exposures+=len(ix);coverage[ix_cpu]+=1
                if a.arm=='native_v':d=p['a']*z-p['b']*prediction
                else:d=p['m']+p['lam'].sqrt()*(p['A']*p['network_input']-p['B']*prediction)
                perclean=(d.detach()-f[ix]).square().mean(1);perv=(prediction.detach()-target).square().mean(1)
                loss_sum+=float(loss)*len(ix);clean_sum+=float(perclean.sum());grad_sum+=float(grad)
                for k in range(4):
                    mask=(t>=k*250)&(t<(k+1)*250);time_sums[k]+=np.array([float(mask.sum()),float(perv[mask].sum()),float(perclean[mask].sum())])
            torch.cuda.synchronize();entry=dict(epoch=epoch+1,updates=updates,exposures=exposures,online_v_MSE=loss_sum/13230,online_clean_MSE=clean_sum/13230,
                preclip_gradient_mean=grad_sum/13,learning_rate=opt.param_groups[0]['lr'],quarter_time_count_v_clean_sums=time_sums.tolist(),seconds=time.monotonic()-epoch_start)
            with (directory/'epoch_log.jsonl').open('a') as log:log.write(json.dumps(entry)+'\n')
            if (epoch+1)%10==0:guard();progress('COMPLETE_FIXED_EPOCH',**entry)
            if epoch+1 in (75,150):
                path=directory/('epoch075.pt' if epoch==74 else 'final.pt');assert not path.exists()
                state=dict(mode=a.arm,policy_SHA256=K.sha(a.policy),statistics=K.ref(out/'statistics.npz'),epochs=epoch+1,updates=updates,exposures=exposures,rows=T,
                    coverage=coverage,model=cpu_tree(net.state_dict()),optimizer=cpu_tree(opt.state_dict()),schedule=schedule.state_dict(),CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state(),
                    order_RNG=ordering.get_state(),noise_RNG=noise_rng.get_state(),epoch_log=K.ref(directory/'epoch_log.jsonl'))
                torch.save(state,path);del state
        assert updates==1950 and exposures==1984500 and torch.all(coverage==150)
        K.write(directory/'completion.json',dict(status='COMPLETE_FIXED_FINAL_150_EPOCH_ERROR_GEOMETRY_PRIOR',arm=a.arm,checkpoint=K.ref(directory/'final.pt'),epoch_log=K.ref(directory/'epoch_log.jsonl'),
            epochs=150,updates=1950,exposures=1984500,parameters=9675648,input_genuine_OOF_T=input_pin,training_rows=13230,training_concepts=1323,
            condition_dropout=0.,fixed_final_no_effect_selection=True,single_training_seed=1729,no_D_V_or_official_targets_fitted=True,optimizer_step_before_schedule=True,
            seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),resources_after=guard()))
    elif a.mode=='prepare':
        lock=gpu_enter(q['job_prefix']+'_prepare');stats=stats_np();import atm_error_geometry_model_20261104_v1 as s
        native=s.original_module(K.checked(base['parent']['native_prior_source']));st=s.tensors(stats,'cuda');oldout=Path(ctx['oldq']['output_root']);old=K.read(oldout/'prepared.json')
        queries=K.read(K.checked(old['queries']));oldvalues=np.load(K.checked(old['conditions']),allow_pickle=False);assert oldvalues.shape==(400,4,1024)
        assert not {v['array_row'] for v in queries}&set(base['T']);values=np.empty((400,len(K.ARMS),1024),np.float32);values[:,:4]=oldvalues
        direct=oldvalues[:,2];values[:,-1]=direct+stats['mean_error'].astype(np.float32);models={};traces=[]
        for mode in K.MODELS:
            fit=K.read(out/'fits'/mode/'completion.json');checkpoint=K.checked(fit['checkpoint']);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert state['epochs']==150 and state['updates']==1950 and state['exposures']==1984500 and torch.all(state['coverage']==150) and state['rows']==base['T']
            net=native.DiffusionPriorUNet(cond_dim=1024,dropout=.1).eval().requires_grad_(False);net.load_state_dict(state['model'],strict=True);del state;net.cuda()
            for lo in range(0,400,128):
                cc=torch.from_numpy(direct[lo:lo+128].copy()).cuda();seeds=[1729+v['array_row'] for v in queries[lo:lo+128]]
                pred,trace=s.sample(mode,net,cc,st,seeds,trace=lo==0);values[lo:lo+len(cc),K.ARMS.index(mode)]=pred.cpu().numpy()
                if lo==0:
                    path=out/(mode+'_first2_all50_states.npz');np.savez_compressed(path,conditions=direct[:2],timesteps=np.array([v['t'] for v in trace]),
                        states=np.stack([v['state'] for v in trace]),predictions=np.stack([v['prediction'] for v in trace]),clean=np.stack([v['clean'] for v in trace]))
                    traces.append(K.ref(path))
                guard();progress('EXPORTED_FIXED_FINAL_PRIOR_CONDITIONS',export_arm=mode,completed=min(lo+128,400))
            assert np.isfinite(values[:,K.ARMS.index(mode)]).all()
            models[mode]=dict(checkpoint=fit['checkpoint'],fit_completion=K.ref(out/'fits'/mode/'completion.json'))
            del net;gc.collect();torch.cuda.empty_cache()
        norms=np.linalg.norm(values,axis=2);assert np.isfinite(values).all() and norms[:,4:].max()<10*float(stats['target_maximum_norm'])
        K.save_array(out/'conditions.npy',values);K.write(out/'queries.json',queries);feature=K.read(oldout/'features400_completion.json')
        K.write(out/'prepared.json',dict(status='COMPLETE_FIXED_FINAL_NATIVE_ATM_FOUR_ARM_CONDITIONS_CPU',study='error_geometry_nine_arms',policy=K.ref(a.policy),conditions=K.ref(out/'conditions.npy'),queries=K.ref(out/'queries.json'),
            target_feature_ref=old['target_feature_ref'],target_metadata_ref=old['target_metadata_ref'],baseline_features=feature['features'],statistics=K.ref(out/'statistics.npz'),models=models,traces=traces,
            all400_normal_inference_only_EEG=True,new_prior_steps=50,new_prior_CFG=1,new_prior_clip_sample=False,mean_bias_training_T_only=True,
            all_baseline400_conditions_exact=np.array_equal(values[:,:4],oldvalues),arm_mean_norms=dict(zip(K.ARMS,norms.mean(0).tolist())),
            finite_and_T_fixed_norm_safety_guard_passed=True,no_D_target_scores_read=True,seconds=time.monotonic()-started,resources_after=guard()))
    else:
        assert not torch.cuda.is_initialized();stats=stats_np();import atm_error_geometry_model_20261104_v1 as s
        native=s.original_module(K.checked(base['parent']['native_prior_source']));st=s.tensors(stats,'cpu');prepared=K.read(out/'prepared.json');proofs=[]
        from diffusers import DDPMScheduler
        alphas=DDPMScheduler().alphas_cumprod
        for mode in K.MODELS:
            info=prepared['models'][mode];state=torch.load(K.checked(info['checkpoint']),map_location='cpu',weights_only=True);net=native.DiffusionPriorUNet(cond_dim=1024,dropout=.1).eval().requires_grad_(False);net.load_state_dict(state['model'],strict=True)
            assert state['epochs']==150 and state['updates']==1950 and torch.all(state['coverage']==150) and state['epoch_log']==K.ref(out/'fits'/mode/'epoch_log.jsonl')
            assert len(state['optimizer']['state'])==len(list(net.parameters())) and len(state['noise_RNG'])>0 and len(state['CUDA_RNG'])>0
            trace=np.load(out/(mode+'_first2_all50_states.npz'),allow_pickle=False);c=torch.from_numpy(trace['conditions'])@st['basis'];err=0.;matherr=0.
            for j,tt in enumerate(trace['timesteps']):
                z=torch.from_numpy(trace['states'][j]);t=torch.full((2,),int(tt),dtype=torch.long)
                with torch.inference_mode():d,v,p=s.clean(mode,net,z,t,c,st,alphas)
                err=max(err,float(((v-torch.from_numpy(trace['predictions'][j])).abs()/(1+torch.from_numpy(trace['predictions'][j]).abs())).max()))
                err=max(err,float(((d-torch.from_numpy(trace['clean'][j])).abs()/(1+torch.from_numpy(trace['clean'][j]).abs())).max()))
                if mode!='native_v':
                    lam=stats[{'iso_error':'isotropic_eigenvalues','paired_error':'eigenvalues','wrong_error':'wrong_eigenvalues'}[mode]]
                    aa,bb,ss,AA,BB=M.coefficients(np.array([float(alphas[int(tt)])]*2),lam)
                    mm=c.numpy().astype(np.float64)+stats['mean_error_rotated'];uu=(z.numpy()-aa*mm)/ss
                    expected=mm+np.sqrt(lam)*(AA*uu-BB*trace['predictions'][j]);matherr=max(matherr,float(np.max(abs(expected-trace['clean'][j])/(1+abs(trace['clean'][j])))))
            assert err<5e-5 and matherr<5e-5
            proofs.append(dict(arm=mode,all50_first2_CPU_neural_forward_and_clean_scaled_error=err,independent_NumPy64_clean_scaled_error=matherr,final_optimizer_RNG_full150coverage_verified=True))
            del net,state;gc.collect();guard()
        K.write(out/'prepared_CPU_audit.json',dict(status='PASS_ALL_FOUR_FINAL_MODELS_OPTIMIZER_RNG_COVERAGE_ALL50_CPU_NEURAL_AND_NUMPY_CLEAN_REPLAY',proofs=proofs,prepared=K.ref(out/'prepared.json'),
            all400_Dcheck_targets_not_read=True,seconds=time.monotonic()-started,resources_after=guard()))
    print(__import__('json').dumps(dict(status='COMPLETE_'+a.mode.upper(),arm=a.arm,seconds=time.monotonic()-started)),flush=True)
except BaseException as exc:
    K.write(out/(a.mode+('_'+a.arm if a.arm else '')+'_failure.json'),dict(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
