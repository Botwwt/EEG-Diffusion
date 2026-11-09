"""D39 native EEG-core one complete-T pass, append-only exact checkpoints.

Only the original four EEG-core tensors learn. Old output affine and ordinary,
native A/P/IP/UNet remain fixed. No development/test values are read.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,datetime,gc,hashlib,json,math,time,traceback
from pathlib import Path
from profile_BrainHIVE_measurement_prior_GPU_20261028_v1 import read,sha,ref,checked,write,ROOT,P

def seed(label):return int(hashlib.sha256(('D39|1729|'+label).encode()).hexdigest()[:16],16)%(2**63-1)

def cpu_tree(x,torch):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v,torch) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v,torch) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v,torch) for v in x)
    return x

def digest(module):
    h=hashlib.sha256()
    for name,t in module.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def atomic_json(path,value):
    temporary=path.with_name(path.name+'.pending');assert not temporary.exists()
    with temporary.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def run(sp,arm,job,segment):
    began=time.monotonic();spec=read(sp);assert spec['status']=='D39_FOUR_ARM_NATIVE_CORE_ONE_COMPLETE_T_PASS_LOCKED'
    assert arm in spec['arms'] and spec['unique_T_queries']==13230 and spec['passes']==1 and spec['expected_updates_per_arm']==827
    assert spec['ordinary_in_training_graph'] is False
    for x in spec['source_pins']:checked(x)
    S=read(segment);assert S['arm']==arm and S['GPU_job_name']==job and S['training_spec_sha256']==sha(sp)
    B=Path(spec['output'])/'arms'/arm;B.mkdir(parents=True,exist_ok=True);assert not (B/'completion.json').exists()
    active='INPUTS';updates=cursor=0;current_commit=None;log=None
    try:
        import brainhive_paired_study_contract_20261025_v1 as C
        engineering=read(checked(spec['engineering']));engineering_spec=read(checked(engineering['spec']))
        C.setup_runtime(dict(runtime_manifest=engineering_spec['runtime_manifest']))
        import numpy as np
        import torch
        from torch.nn import functional as F
        import brainhive_measurement_prior_20261028_v1 as M
        from brainhive_reversible_eeg_20261028_v1 import author_brain_class
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        from brainhive_component_generator_v1 import load_native,denoise
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        from verify_HIVE_second_host_native_GPU_20261028_v1 import wrapper,memory
        wrapper_proof=wrapper(job);baseCPU=memory()['VmRSS'];torch.set_num_threads(2)
        torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        cpu=read(checked(engineering_spec['CPU_completion']));cpuspec=read(checked(cpu['spec']));D30=read(checked(cpuspec['D30_spec']))
        files={Path(x['path']).name:x for x in cpu['files']}
        raw=np.load(checked(files['T13230_mean4.npy']),mmap_mode='r',allow_pickle=False)
        targets=np.load(checked(D30['T_targets']),mmap_mode='r',allow_pickle=False)
        latents=np.load(checked(engineering_spec['native_T_latents']),mmap_mode='r',allow_pickle=False)
        parent=torch.load(checked(D30['roles']['full']['checkpoint']),weights_only=True,map_location='cpu')
        frozen=read(checked(engineering_spec['ordinary_spec']));old=torch.load(checked(frozen['checkpoint']),weights_only=True,map_location='cpu')
        assert old['calibration']['selected_strength']==1
        cls=author_brain_class(checked(D30['author_models_source']),checked(D30['author_layers_source']))
        original=cls(4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False).cuda();original.load_state_dict(parent['model'],strict=True)
        student=copy.deepcopy(original).eval();params=[]
        for name,p in student.named_parameters():
            if name not in ['block.layers.0.norm.weight','block.layers.0.norm.bias']:p.requires_grad_(True);params.append((name,p))
        assert len(params)==4 and sum(p.numel() for _,p in params)==6450176
        gamma=original.block.layers[0].norm.weight.detach();beta=original.block.layers[0].norm.bias.detach()
        ordinary=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False).cuda();ordinary.load_state_dict(old['model'],strict=True)
        precision=None
        if arm in ['measurement','permuted_measurement']:
            name='measurement_precision.npy' if arm=='measurement' else 'permuted_precision.npy'
            precision=torch.from_numpy(np.load(checked(files[name]),allow_pickle=False)).float().cuda()
        active='STRICT_NATIVE_FROZEN_LOAD';observer_dir=B/(job+'_converted');observer_dir.mkdir()
        observer=ConvertedWeightAudit(torch,observer_dir);observer.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:observer.restore()
        loader=observer.complete()
        pipe.register_modules(vae=None,text_encoder=None,text_encoder_2=None);gc.collect();torch.cuda.empty_cache()
        host_before={name:digest(module) for name,module in [('UNet',pipe.unet),('A',adapter),('P',projector),('ordinary',ordinary),('parent',original)]}
        assert host_before['parent']==engineering['original_model_fingerprint']
        pipe.unet.train();pipe.unet.enable_gradient_checkpointing();assert float(pipe.unet.config.dropout)==0
        from diffusers import DDPMScheduler
        scheduler=DDPMScheduler.from_config(pipe.scheduler.config);assert scheduler.config.prediction_type=='epsilon'
        optimizer=torch.optim.AdamW([p for _,p in params],lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
        order=torch.randperm(13230,generator=torch.Generator().manual_seed(seed('one_full_T_order')))
        orderSHA=hashlib.sha256(order.numpy().tobytes()).hexdigest()
        noise_rng=torch.Generator(device='cuda').manual_seed(seed('shared_training_noise'))
        coverage=torch.zeros(13230,dtype=torch.bool);loss_sum=0.;elapsed_prefix=0.
        logfile=B/'training.jsonl'
        if S['start_updates']:
            current_commit=read(checked(S['resume_commit']))
            saved=torch.load(checked(current_commit['checkpoint']),weights_only=True,map_location='cpu')
            assert current_commit['updates']==saved['updates']==S['start_updates'] and saved['arm']==arm and saved['spec_sha256']==sha(sp)
            assert saved['order_sha256']==orderSHA and saved['log_bytes']==logfile.stat().st_size and sha(logfile)==saved['log_sha256']
            student.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            updates=saved['updates'];cursor=saved['cursor'];coverage=saved['coverage'];loss_sum=saved['loss_sum'];elapsed_prefix=saved['seconds']
            noise_rng.set_state(saved['noise_rng']);torch.set_rng_state(saved['CPU_rng']);torch.cuda.set_rng_state_all(saved['CUDA_rng'])
            del saved
            print(json.dumps(dict(stage='D39_EXACT_COMMIT_RESUMED',arm=arm,updates=updates,cursor=cursor,checkpoint=current_commit['checkpoint'])),flush=True)
        else:
            assert not logfile.exists() and not list(B.glob('checkpoint_*.json'))
            torch.manual_seed(1729);assert digest(student)==engineering['original_model_fingerprint']
        assert coverage.sum()==cursor and coverage[order[:cursor]].all() and not coverage[order[cursor:]].any()
        initial_updates=updates;last_committed=updates
        log=logfile.open('a',buffering=1)
        def budget():
            assert memory()['VmHWM']-baseCPU<10<<30 and torch.cuda.max_memory_allocated()<20<<30
            assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())<2<<30
        def commit():
            nonlocal current_commit,last_committed
            if updates==last_committed and current_commit is not None:return
            log.flush();os.fsync(log.fileno());path=B/f'checkpoint_{updates:06d}.pt';assert not path.exists()
            pending=path.with_name(path.name+'.pending');assert not pending.exists()
            payload=dict(schema='D39_EXACT_NATIVE_CORE_ONE_PASS_COMMIT_V1',arm=arm,spec_sha256=sha(sp),
                model=cpu_tree(student.state_dict(),torch),optimizer=cpu_tree(optimizer.state_dict(),torch),
                updates=updates,cursor=cursor,coverage=coverage.clone(),order_sha256=orderSHA,
                noise_rng=noise_rng.get_state().cpu(),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all(),
                log_bytes=logfile.stat().st_size,log_sha256=sha(logfile),loss_sum=loss_sum,seconds=elapsed_prefix+time.monotonic()-began)
            with pending.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.replace(pending,path);del payload
            current_commit=dict(status='D39_NATIVE_CORE_APPEND_ONLY_COMMITTED_CHECKPOINT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,
                updates=updates,cursor=cursor,checkpoint=ref(path),log_bytes=logfile.stat().st_size,log_sha256=sha(logfile),spec_sha256=sha(sp),
                optimizer_RNG_and_order_coverage_present=True)
            write(B/f'checkpoint_{updates:06d}.json',current_commit);last_committed=updates
            atomic_json(B/'latest_commit.json',current_commit)
        active='ONE_COMPLETE_T_PASS'
        for offset in range(cursor,13230,16):
            if updates-initial_updates>=spec['maximum_updates_per_GPU_job'] or time.monotonic()-began>770:break
            budget();indices=order[offset:offset+16];n=len(indices);ids=indices.numpy();step_start=time.monotonic()
            x=torch.from_numpy(np.array(raw[ids],copy=True)).cuda();target=torch.from_numpy(np.array(targets[ids],copy=True)).cuda()
            z=torch.from_numpy(np.array(latents[ids],copy=True)).cuda()
            with torch.no_grad():
                origin=original(x);unit_origin=(origin-beta)/gamma
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):teacher_tokens=projector(adapter(target)).float().detach()
            noise=torch.randn(z.shape,device='cuda',generator=noise_rng);t=torch.randint(0,1000,(n,),device='cuda',generator=noise_rng)
            noisy=scheduler.add_noise(z,noise,t);optimizer.zero_grad(set_to_none=True)
            c=student(x);unit=(c-beta)/gamma
            # Train on the native branch. The immutable ordinary is separately
            # switchable at evaluation and never shapes this training target.
            corrected=c
            with torch.autocast('cuda',dtype=torch.bfloat16):
                tokens=projector(adapter(corrected));prediction=denoise(pipe,prompts,noisy,t,tokens)
                epsilon_loss=F.mse_loss(prediction.float(),noise)
                token_loss=(1-F.cosine_similarity(tokens.float(),teacher_tokens,dim=-1)).mean()
            anchor=M.anchor_penalty(unit,unit_origin,precision,arm)
            loss=epsilon_loss+.1*token_loss+anchor;assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in params)
            assert all(p.grad is None for module in [original,ordinary,adapter,projector,pipe.unet] for p in module.parameters())
            gradient=torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True);optimizer.step()
            assert not coverage[indices].any();coverage[indices]=True;cursor=offset+n;updates+=1;loss_sum+=float(loss.detach())*n
            event=dict(stage='D39_REAL_NATIVE_CORE_GENERATOR_TRAINING',arm=arm,updates=updates,cursor=cursor,indices=ids.tolist(),t=t.tolist(),
              noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),loss=float(loss.detach()),epsilon_loss=float(epsilon_loss.detach()),
              token_loss=float(token_loss.detach()),anchor_loss=float(anchor.detach()),gradient_norm=float(gradient),
              unit_shift_RMS=float((unit-unit_origin).square().mean().sqrt()),step_seconds=time.monotonic()-step_start)
            log.write(json.dumps(event)+'\n')
            if updates%16==0 or cursor==13230:
                atomic_json(B/'progress.json',dict(status='D39_RUNNING_REAL_NATIVE_CORE_TRAINING',**event,expected_updates=827,expected_cursor=13230))
                print(json.dumps(event),flush=True)
            if updates%64==0:commit()
        assert updates>initial_updates,'No progress is never a successful resumable segment'
        commit();log.close();log=None
        for name in ['block.layers.0.norm.weight','block.layers.0.norm.bias']:
            assert torch.equal(student.state_dict()[name].detach().cpu(),parent['model'][name])
        after={name:digest(module) for name,module in [('UNet',pipe.unet),('A',adapter),('P',projector),('ordinary',ordinary),('parent',original)]}
        assert after==host_before and digest(student)!=host_before['parent']
        budget()
        if cursor==13230:
            assert updates==827 and coverage.all()
            path=B/'fixed_final_encoder.pt';assert not path.exists()
            with path.open('xb') as f:torch.save(dict(schema='D39_FIXED_FINAL_NATIVE_ENCODER_V1',arm=arm,model=cpu_tree(student.state_dict(),torch),
                updates=updates,query_exposures=cursor,spec_sha256=sha(sp),original_output_affine_frozen=True,old_ordinary_SHA=engineering['frozen_ordinary_SHA']),f)
            write(B/'completion.json',dict(status='D39_ONE_COMPLETE_T13230_PASS_FIXED_FINAL_NATIVE_CORE_TRAINED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,
                spec=ref(sp),model=ref(path),last_commit=ref(B/f'checkpoint_{updates:06d}.json'),unique_T_queries=13230,concepts=1323,subjects=1,
                updates=827,query_exposures=13230,trainable_native_parameters=6450176,frozen_affine_parameters=2048,additional_inference_parameters=0,
                parent_checkpoint=D30['roles']['full']['checkpoint'],old_ordinary_SHA=engineering['frozen_ordinary_SHA'],old_ordinary_strength=1,
                same_native_generator_weights_unchanged=True,full_ten_subject_or_OOF_training=False,not_whole_pipeline_OOF=True,
                Dfit_Dcheck_V_or_official_values_read=False,fixed_final_not_held_selected=True,new_research_images=0,
                ordinary_in_training_graph=False,ordinary_independently_switchable_at_inference=True,
                normal_inference_UNet_NFE=4,one_inference_encoder_only=True))
        write(B/(job+'_segment_completion.json'),dict(status='D39_NATIVE_CORE_SEGMENT_COMMITTED_AND_FROZEN_HOST_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          arm=arm,start_updates=initial_updates,end_updates=updates,cursor=cursor,latest_commit=ref(B/f'checkpoint_{updates:06d}.json'),wrapper=wrapper_proof,
          loader=loader,frozen_host_fingerprints=after,seconds=time.monotonic()-began,peak_extra_CPU_bytes=memory()['VmHWM']-baseCPU,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
          actual_full_T_done=cursor==13230,additional_inference_parameters=0,training_is_not_image_efficacy=True))
    except BaseException as e:
        if log is not None:log.flush();os.fsync(log.fileno());log.close()
        p=B/(job+'_failure.json')
        if not p.exists():write(p,dict(status='D39_TRAINING_FAILURE_PREFIX_AND_OLDER_COMMITS_RETAINED',arm=arm,stage=active,error=repr(e),traceback=traceback.format_exc(),updates=updates,cursor=cursor,seconds=time.monotonic()-began))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ['spec','segment']:p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--gpu-job-name',required=True);a=p.parse_args();run(a.spec,a.arm,a.gpu_job_name,a.segment)
