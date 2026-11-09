"""Finite real paired-effect engineering or full-T training; fixed final head."""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import brainhive_paired_study_contract_20261025_v1 as C


def save_tensor(torch,path,value):
    temporary=path.with_name(path.name+'.pending');assert not temporary.exists()
    with temporary.open('xb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def restore_committed_log(path,saved):
    content=path.read_bytes();assert 0<=saved['log_bytes']<=len(content)
    committed=content[:saved['log_bytes']]
    assert hashlib.sha256(committed).hexdigest()==saved['log_sha256']
    if len(content)>len(committed):
        suffix=path.with_name('uncommitted_log_tail_'+str(os.getpid())+'.jsonl');assert not suffix.exists()
        suffix.write_bytes(content[len(committed):])
        temporary=path.with_name(path.name+'.rollback.pending');assert not temporary.exists()
        with temporary.open('xb') as stream:stream.write(committed);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)


def main():
    invoked=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--arm',choices=C.ARMS);parser.add_argument('--profile-only',action='store_true')
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config=C.read(args.registration)
    study_core=C.validate(config,all_inputs=True);plan,plan_sha=C.prepare_plan(config)
    assert args.profile_only or args.arm in C.ARMS
    if args.preflight_only:
        if not args.profile_only:C.require_admission(config)
        print(json.dumps(dict(status='REAL_FULL132300_PAIR_PLAN_CPU_PREFLIGHT_PASSED',queries=plan['queries'],
            groups=plan['groups'],plan_sha256=plan_sha,CUDA_initialized=False)));return
    assert C.ROOT==C.REMOTE
    C.setup_runtime(config)
    import torch
    import numpy as np
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_data_v1 import load_training
    from brainhive_component_generator_v1 import load_native
    from brainhive_paired_effect_integration_20261025_v1 import PairedEffectCapture,paired_step
    from train_BrainHIVE_bounded_component_v1 import seed,development
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 import module_digest
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.profile_only:
        assert not args.resume
        output=Path(config['profile_output']);output.mkdir(parents=True,exist_ok=False)
        job_name=config['profile_job_name'];maximum=1800
    else:
        C.require_admission(config);C.resources(config,training_start=True)
        output=C.DATA/'arms'/args.arm/'train'
        assert output.exists() if args.resume else not output.exists()
        output.mkdir(parents=True,exist_ok=True)
        assert not (output/'completion.json').exists()
        job_name=config['training_job_names'][args.arm][int(args.resume)]
        maximum=config['maximum_training_seconds_per_job']
    wrapper,budget=C.verify_wrapper(config,job_name,maximum,output)
    C.resources(config,training_start=True)
    interrupted=[False]
    signal.signal(signal.SIGTERM,lambda signum,frame:interrupted.__setitem__(0,True))
    loader_dir=output/('loader_'+str(os.getpid()));loader_dir.mkdir()
    audit=ConvertedWeightAudit(torch,loader_dir);audit.install()
    try:pipe,adapter,projector,prompts,host=load_native(training=True)
    finally:audit.restore()
    loader_proof=audit.complete();budget.check()
    inputs=load_training();assert inputs['input_contract_sha256']==config['input_bindings']['input_contract']['sha256']
    vectors={k:torch.tensor(plan[k],dtype=torch.long,device='cuda') for k in
             ('partner','shuffled','shuffled_partner','expected_affine_ids')}
    assert torch.equal(inputs['affine_ids'],vectors['expected_affine_ids'])
    assert torch.equal(inputs['affine_ids'],inputs['affine_ids'][vectors['partner']])
    latent=C.bound(config['input_bindings']['latent_receipt'])
    assert latent['array_rows']==inputs['array_rows'] and C.sha(latent['path'])==latent['sha256']
    array=np.load(latent['path'],mmap_mode='r');assert array.shape==(13230,4,64,64) and array.dtype==np.float32
    latents=torch.tensor(array,device='cuda');del array
    from diffusers import DDPMScheduler
    scheduler=DDPMScheduler.from_config(pipe.scheduler.config)
    collector=PairedEffectCapture(pipe.unet);assert len(collector.layers)==70
    orders=[torch.randperm(132300,generator=torch.Generator().manual_seed(seed('pass'+str(i)))) for i in range(3)]
    invariant=dict(study_core_sha256=study_core,pair_plan_sha256=plan_sha,
        input_contract_sha256=inputs['input_contract_sha256'],host=host,head_parameters=525568,
        orders_sha256=[hashlib.sha256(order.numpy().tobytes()).hexdigest() for order in orders],
        explicit_UNet_forwards_per_update=2,primary_query_exposures=396900,paired_query_exposures=396900,
        input_role='OOF_T',source_sha256=config['core_sources'],training_seed=1729,
        normal_inference_uses_teacher=False,optimizer='AdamW lr1e-4 wd.01 default betas/eps')
    try:
        if args.profile_only:
            with gzip.open(output/'pair_plan.json.gz','wt',encoding='utf-8') as stream:json.dump(plan,stream)
            selected=[]
            for group in plan['groups']:
                candidates=[i for i,a in enumerate(plan['expected_affine_ids']) if a==group['affine_id']]
                candidates.sort(key=lambda i:hashlib.sha256(('1729|engineering_query|'+str(i)).encode()).digest())
                selected.append(candidates[:2])
            fixed=[v[0] for v in selected]+[v[1] for v in selected[:12]]
            assert len(set(fixed))==32 and len({plan['expected_affine_ids'][i] for i in fixed})==20
            before={k:module_digest(m) for k,m in [('UNet_IP',pipe.unet),('A',adapter),('P',projector)]}
            records=[]
            for arm in C.ARMS:
                torch.manual_seed(1729);model=BoundedConditionAdapter(predict_risk=False).cuda()
                optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
                rng=torch.Generator(device='cuda').manual_seed(seed('shared_training_noise'))
                for update in range(2):
                    budget.check();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
                    indices=torch.tensor(fixed[update*16:(update+1)*16])
                    result=paired_step(model,pipe,adapter,projector,prompts,scheduler,collector,inputs,latents,vectors,indices,rng,arm)
                    torch.cuda.synchronize();forward_seconds=time.monotonic()-start
                    gradients=torch.autograd.grad(result['terms']['D'],(result['changed_i'],result['changed_j'],model.direction.weight),retain_graph=True)
                    norms=[float(v.norm()) for v in gradients]
                    assert all(math_value>0 and np.isfinite(math_value) for math_value in norms)
                    auxiliary_gradient=torch.autograd.grad(result['auxiliary'],model.direction.weight,retain_graph=True)[0]
                    active_norm=float(auxiliary_gradient.norm());assert np.isfinite(active_norm)
                    assert active_norm==0 if arm=='O' else active_norm>0
                    assert float(result['diagnostics']['valid_fraction'])>=.9
                    assert float(result['diagnostics']['shuffled_valid_fraction'])>=.9
                    torch.cuda.synchronize();backward_started=time.monotonic()
                    result['loss'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    assert all(p.grad is None for module in (pipe.unet,adapter,projector) for p in module.parameters())
                    optimizer.step();torch.cuda.synchronize()
                    seconds=forward_seconds+time.monotonic()-backward_started
                    record=dict(arm=arm,engineering_update=update+1,indices=indices.tolist(),
                        active_auxiliary_gradient_norm=active_norm,difference_gradient_norms_i_j_shared_head=norms,
                        loss=float(result['loss'].detach()),terms={k:float(v.detach()) for k,v in result['terms'].items()},
                        diagnostics={k:float(v) for k,v in result['diagnostics'].items()},update_seconds=seconds,
                        explicit_UNet_forwards=result['explicit_UNet_forwards'],same_parent_Q_verified=True,total_gradient_norm=float(norm))
                    records.append(record);print(json.dumps(record),flush=True);del result,gradients,auxiliary_gradient
                    budget.check()
                del model,optimizer;gc.collect();torch.cuda.empty_cache()
            after={k:module_digest(m) for k,m in [('UNet_IP',pipe.unet),('A',adapter),('P',projector)]};assert before==after
            # Exercise the SAME generation function used by the formal entry.
            # No scores, trained heads or selected images; exact zero/disable
            # equivalence on the first previously registered T32 query only.
            from generate_BrainHIVE_paired_effect_20261025_v1 import sample_native
            from brainhive_functional_training_contract_v2 import query_identity
            query=C.bound(config['input_bindings']['postfit_queries'])['rows'][0]
            position=inputs['array_rows'].index(query['array_row'])
            subject=inputs['subjects'].index(query['subject']);index=subject*13230+position
            raw=inputs['condition'][index:index+1];aid=inputs['affine_ids'][index:index+1]
            pipe.unet.eval();pipe.unet.disable_gradient_checkpointing()
            # load_native(training=True) keeps VAE FP32. Restore its native
            # inference FP16 storage so the pipeline's own force-upcast branch
            # casts BOTH decoder weights and latent input consistently.
            pipe.vae.to(dtype=torch.float16)
            infer_prompts=tuple(v.detach().to(torch.float16) for v in prompts)
            zero=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
            changed,_=zero(raw,inputs['weight'][aid],inputs['bias'][aid],gate_mode='fixed')
            disabled,_=zero(raw,inputs['weight'][aid],inputs['bias'][aid],enabled=False)
            assert torch.equal(raw,changed) and torch.equal(raw,disabled)
            pixel_hashes=[];initial_hashes=[]
            for label,value in [('native',raw),('zero',changed),('disabled',disabled)]:
                with torch.inference_mode():
                    tokens=projector(adapter(value.to(torch.float16)))
                    image,observation=sample_native(pipe,infer_prompts,tokens,query['image_seed'],torch)
                pixels=np.asarray(image);pixel_hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
                initial_hashes.append(observation['initial_UNet_input_sha256'])
                image.save(output/(label+'.webp'),format='WEBP',lossless=True)
            assert len(set(pixel_hashes))==len(set(initial_hashes))==1
            after={k:module_digest(m) for k,m in [('UNet_IP',pipe.unet),('A',adapter),('P',projector)]};assert before==after
            budget.check();forecast=1.25*max(r['update_seconds'] for r in records)*24807
            assert forecast<config['maximum_cumulative_training_seconds_per_arm']
            C.atomic(output/'completion.json',dict(status='COMPLETE_REAL_PAIRED_FOUR_ARM_ENGINEERING_PROFILE',
                study_core_sha256=study_core,registration=C.ref(args.registration),pair_plan_sha256=plan_sha,
                full_input_queries=132300,affine_groups_in_GPU_batches=20,rows=records,loader_proof=loader_proof,
                frozen_hash_before=before,frozen_hash_after=after,all_checks_passed=True,
                forecast_seconds_per_arm=forecast,forecast_is_not_guarantee=True,
                peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,
                elapsed_seconds=time.monotonic()-invoked,temporary_head_engineering_updates=8,
                native_zero_disabled_pixel_SHA_identical=True,pixel_sha256=pixel_hashes[0],
                first_T32_generation_engineering_queries=1,engineering_images=3,
                formal_training=False,trained_checkpoints_saved=0,V_or_test_scores_read=False))
            print('COMPLETE_REAL_PAIRED_FOUR_ARM_ENGINEERING_PROFILE',flush=True);return
        arm=args.arm;locked=output/'input_provenance.json'
        invariant['arm']=arm
        if locked.exists():assert C.read(locked)==invariant
        else:C.atomic(locked,invariant)
        C.atomic(output/('launch_'+str(os.getpid())+'.json'),dict(wrapper=wrapper,registration=C.ref(args.registration),
            provenance_sha256=C.sha(locked),loader_proof=loader_proof,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        torch.manual_seed(1729);model=BoundedConditionAdapter(predict_risk=False).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        rng=torch.Generator(device='cuda').manual_seed(seed('shared_training_noise'))
        cursor=updates=seen=0;cumulative=loss_sum=0.;coverage=torch.zeros(3,132300,dtype=torch.bool)
        log_path=output/'training.jsonl'
        if args.resume:
            saved=torch.load(output/'resume.pt',map_location='cpu',weights_only=False)
            assert saved['provenance_sha256']==C.sha(locked)
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            cursor,updates,seen,cumulative,loss_sum=[saved[k] for k in ('cursor','updates','seen','seconds','loss_sum')]
            coverage=saved['coverage'];rng.set_state(saved['noise_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            restore_committed_log(log_path,saved)
        last_saved=time.monotonic()
        def snapshot(log):
            log.flush();os.fsync(log.fileno());payload=log_path.read_bytes()
            save_tensor(torch,output/'resume.pt',dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),cursor=cursor,updates=updates,seen=seen,coverage=coverage,
                seconds=cumulative+time.monotonic()-invoked,loss_sum=loss_sum,noise_rng=rng.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),provenance_sha256=C.sha(locked),
                log_bytes=len(payload),log_sha256=hashlib.sha256(payload).hexdigest()))
        with log_path.open('a',buffering=1) as log:
            for epoch in range(cursor//132300,3):
                begin=cursor%132300 if epoch==cursor//132300 else 0
                for offset in range(begin,132300,16):
                    elapsed=time.monotonic()-invoked
                    if interrupted[0] or elapsed>maximum-120 or cumulative+elapsed>172800-120:
                        snapshot(log);C.atomic(output/'progress.json',dict(status='PAIRED_TRAIN_FINITE_STOP_EXACT_RESUME',updates=updates,cursor=cursor))
                        raise RuntimeError('Finite stop; exact checkpoint retained')
                    budget.check()
                    indices=orders[epoch][offset:offset+16];optimizer.zero_grad(set_to_none=True)
                    result=paired_step(model,pipe,adapter,projector,prompts,scheduler,collector,inputs,latents,vectors,indices,rng,arm)
                    result['loss'].backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    assert all(p.grad is None for module in (pipe.unet,adapter,projector) for p in module.parameters())
                    optimizer.step();assert not coverage[epoch,indices].any();coverage[epoch,indices]=True
                    n=len(indices);cursor=epoch*132300+offset+n;updates+=1;seen+=n;loss_sum+=float(result['loss'].detach())*n
                    if updates%64==0 or offset+n==132300:
                        event=dict(status='PAIRED_EFFECT_FULL_T_TRAINING',arm=arm,updates=updates,expected_updates=24807,
                            cursor=cursor,seen_primary_queries=seen,seen_paired_queries=seen,expected_primary_exposures=396900,
                            loss_mean=loss_sum/seen,last_loss=float(result['loss'].detach()),epsilon_loss=float(result['epsilon_loss'].detach()),
                            token_loss=float(result['token_loss'].detach()),auxiliary=float(result['auxiliary'].detach()),
                            terms={k:float(v.detach()) for k,v in result['terms'].items()},
                            diagnostics={k:float(v) for k,v in result['diagnostics'].items()},gradient_norm=float(gradient),
                            seconds=cumulative+time.monotonic()-invoked,UNet_forwards=2)
                        log.write(json.dumps(event)+'\n');C.atomic(output/'progress.json',event);print(json.dumps(event),flush=True)
                    del result
                    if updates%1024==0:C.validate(config);C.resources(config)
                    if time.monotonic()-last_saved>120:snapshot(log);last_saved=time.monotonic()
            assert coverage.all() and updates==24807 and seen==396900
            snapshot(log)
        calibration=development(model.eval(),adapter,projector,'ordinary')
        C.atomic(output/'calibration.json',calibration)
        save_tensor(torch,output/'component_final.pt',dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            study_core_sha256=study_core,arm=arm,seed=1729,updates=updates,seen_queries=seen,passes=3,
            calibration=calibration,provenance_sha256=C.sha(locked),pair_plan_sha256=plan_sha))
        C.atomic(output/'completion.json',dict(status='PAIRED_EFFECT_FIXED_FINAL_FULL_T_COMPLETE',arm=arm,
            study_core_sha256=study_core,parameters=525568,subjects=10,updates=updates,primary_exposures=seen,
            paired_exposures=seen,passes=3,checkpoint=C.ref(output/'component_final.pt'),calibration=C.ref(output/'calibration.json'),
            provenance=C.ref(locked),training_seconds=cumulative+time.monotonic()-invoked,
            peak_GPU_bytes=budget.peak_GPU_bytes,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,image_gain_established=False))
        print('PAIRED_EFFECT_FIXED_FINAL_FULL_T_COMPLETE',flush=True)
    except BaseException as error:
        C.atomic(output/('failure_'+str(os.getpid())+'.json'),dict(error=repr(error),formal_training=not args.profile_only,
            automatic_retry=False,profile_only=args.profile_only));raise
    finally:collector.close()


if __name__=='__main__':main()
