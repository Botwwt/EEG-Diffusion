"""D20 fixed one complete T400 pass per matched IP-LoRA arm.

This is a discovery subset, not full ten-subject formal T training. All held
readouts are logs only. The fixed final state is the only image candidate.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_error_pair_lora_20261027_v2 as V
from brainhive_bounded_component_v1 import BoundedConditionAdapter
from profile_BrainHIVE_error_pair_lora_GPU_20261027_v2 import fingerprint
from fit_BrainHIVE_image_domain_GPU_20261027_v2 import cpu

def wrong_positions(records):
    groups={}
    for i,r in enumerate(records):groups.setdefault(r['concept_id'],[]).append(i)
    assert len(groups)==40 and all(len(v)==10 for v in groups.values());perm=np.empty(400,dtype=np.int64)
    for ids in groups.values():perm[ids]=np.roll(ids,-1)
    assert sorted(perm.tolist())==list(range(400)) and (perm!=np.arange(400)).all();return perm

def corrected_Dfit(prep,contract,checkpoint):
    c=torch.from_numpy(np.load(U.checked(prep['conditions']['Dfit200']),allow_pickle=False))
    subject=next(s for s in contract['subjects'] if s['subject']=='sub-01');src=next(s for s in subject['sources'] if s['role']=='full')
    cp=torch.load(U.checked({'path':src['checkpoint_path'],'sha256':src['checkpoint_sha256']}),weights_only=True,map_location='cpu')['model']
    saved=torch.load(U.checked(checkpoint),weights_only=False,map_location='cpu');model=BoundedConditionAdapter().eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True)
    assert saved['calibration']['selected_strength']==1.
    with torch.no_grad():out=model(c,cp['block.layers.0.norm.weight'][None].expand_as(c),cp['block.layers.0.norm.bias'][None].expand_as(c))[0]
    return out

def main():
    p=argparse.ArgumentParser()
    for name in ('spec','native-registration','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=V.ARMS,required=True);p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();started=time.monotonic();assert a.max_seconds==600 and not a.output.exists();a.output.mkdir(parents=True)
    active='INPUTS';stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));updates=0;exposures=0;coverage=torch.zeros(400,dtype=torch.bool)
    layers=None;optim=None;noise_rng=None
    def log(x):
        x=x|{'seconds':time.monotonic()-started}
        with (a.output/'training_log.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(x),flush=True)
    try:
        spec=U.read(a.spec);assert spec['passes']==1 and spec['batch_pairs']==2 and spec['updates_per_arm']==200 and spec['fixed_final_checkpoint']
        for item in spec['source_pins']:U.checked(item)
        profile=U.read(U.checked(spec['profile']));assert profile['frozen_weights_unchanged'] and profile['temporary_states_discarded']
        old=U.read(U.checked(spec['D19_fit']));prep=U.read(U.checked(old['prepared_inputs']));mapping=U.read(U.checked(prep['mapping']))
        contract=U.read(U.checked(spec['native_input_contract']))
        train_lat=np.load(U.checked(old['encoded_latents']['T400']),allow_pickle=False,mmap_mode='r');held_lat=np.load(U.checked(old['encoded_latents']['Dfit200']),allow_pickle=False,mmap_mode='r')
        original_c=np.load(U.checked(profile['ordinary_condition_bank']),allow_pickle=False);assert original_c.shape==(400,1024)
        held_c=corrected_Dfit(prep,contract,spec['ordinary_checkpoint'])
        U.save_array(a.output/'Dfit200_original_ordinary_conditions.npy',held_c.numpy())
        import brainhive_paired_study_contract_20261025_v1 as C
        config=C.read(a.native_registration);C.validate(config);C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=600,maximum_output_bytes=192<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stopped[0] and time.monotonic()-started<560,'D20 finite fit budget with checkpoint reserve';budget.check()
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);C.resources(config,training_start=True)
        from brainhive_component_generator_v1 import load_native,denoise
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        active='STRICT_UNCHANGED_NATIVE_LOAD';folder=a.output/'loader';folder.mkdir();audit=ConvertedWeightAudit(torch,folder);audit.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader=audit.complete();frozen=list(pipe.unet.named_parameters());before=fingerprint(frozen);assert before==profile['original_UNet_parameter_sha256']
        # Parities were established in the engineering job. Fixed new models
        # restart from the same seed, not from any temporary profile state.
        torch.manual_seed(1729);layers=V.install(pipe.unet);params=V.new_parameters(layers);initial=V.new_state(layers)
        initial_sha=fingerprint([(n+'/'+k,v) for n,x in initial.items() for k,v in x.items()])
        optim=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.99),weight_decay=.01)
        pipe.unet.train();pipe.unet.enable_gradient_checkpointing();assert float(pipe.unet.config.dropout)==0
        from diffusers import DDPMScheduler
        schedule=DDPMScheduler.from_config(pipe.scheduler.config);assert schedule.config.prediction_type=='epsilon'
        with torch.no_grad():
            train_tokens=projector(adapter(torch.from_numpy(original_c).cuda().half())).detach()
            held_tokens=projector(adapter(held_c.cuda().half())).detach()
        x=torch.from_numpy(np.array(train_lat,copy=True)).cuda();hx=torch.from_numpy(np.array(held_lat,copy=True)).cuda()
        perm=wrong_positions(mapping['T400']);U.save_array(a.output/'sameconcept_negative_permutation.npy',perm)
        order=torch.randperm(400,generator=torch.Generator().manual_seed(20201729));U.save_array(a.output/'training_order.npy',order.numpy())
        noise_rng=torch.Generator(device='cuda').manual_seed(20201730);initial_rng=noise_rng.get_state().cpu()
        held_rng=torch.Generator(device='cuda').manual_seed(20201731);hn=torch.randn((200,4,64,64),device='cuda',generator=held_rng);ht=torch.randint(0,1000,(200,),device='cuda',generator=held_rng)
        U.save_array(a.output/'Dfit200_t.npy',ht.cpu().numpy());U.save_array(a.output/'Dfit200_noise.npy',hn.cpu().numpy())
        def held(label):
            all_error=[];all_ref=[]
            with torch.no_grad():
                for lo in range(0,200,2):
                    check();sl=slice(lo,lo+2);eps=torch.cat([hn[sl],hn[sl]]);t=torch.cat([ht[sl],ht[sl]])
                    states=schedule.add_noise(torch.cat([hx[sl,0],hx[sl,1]]),eps,t);tokens=torch.cat([held_tokens[sl],held_tokens[sl]])
                    V.enabled(layers,False)
                    with torch.autocast('cuda',dtype=torch.bfloat16):reference=denoise(pipe,prompts,states,t,tokens)
                    V.enabled(layers,True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=denoise(pipe,prompts,states,t,tokens)
                    all_error.append((out.float()-eps).square().flatten(1).mean(1).reshape(2,2).T.cpu())
                    all_ref.append((reference.float()-eps).square().flatten(1).mean(1).reshape(2,2).T.cpu())
            e=torch.cat(all_error);r=torch.cat(all_ref);delta=e-r;margin=V.BETA*(delta[:,0]-delta[:,1])
            U.save_array(a.output/(label+'_Dfit_errors.npy'),e.numpy());U.save_array(a.output/(label+'_Dfit_reference_errors.npy'),r.numpy())
            log(dict(stage='FIXED_ALL200_DFIT_LOG_ONLY',label=label,positive_MSE=float(e[:,0].mean()),negative_MSE=float(e[:,1].mean()),preference_loss=float(torch.nn.functional.softplus(margin).mean()),delta_positive=float(delta[:,0].mean()),delta_negative=float(delta[:,1].mean()),no_model_selection=True))
        active='INITIAL_ALL200_DFIT_LOG';held('initial')
        active='ONE_COMPLETE_T400_PASS';summed={};training_started=time.monotonic()
        for start in range(0,400,2):
            check();indices=order[start:start+2];ids=indices.cuda();n=2;negative=perm[indices.numpy()] if a.arm=='sameconcept_wrong_error' else indices.numpy()
            z=torch.cat([x[ids,0],x[torch.as_tensor(negative,device='cuda'),1]])
            noise=torch.randn((n,4,64,64),device='cuda',generator=noise_rng);t0=torch.randint(0,1000,(n,),device='cuda',generator=noise_rng)
            eps=torch.cat([noise,noise]);t=torch.cat([t0,t0]);noisy=schedule.add_noise(z,eps,t);tokens=torch.cat([train_tokens[ids],train_tokens[ids]])
            V.enabled(layers,False)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):reference=denoise(pipe,prompts,noisy,t,tokens).detach()
            V.enabled(layers,True);optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=denoise(pipe,prompts,noisy,t,tokens);loss,terms=V.objective(out,reference,eps,a.arm)
            assert torch.isfinite(loss);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            assert all(p.grad is None for _,p in frozen) and all(m.up.grad is not None and torch.isfinite(m.up.grad).all() for _,m in layers)
            optim.step();ratio=max(m.project() for _,m in layers);assert not coverage[indices].any();coverage[indices]=True;updates+=1;exposures+=n
            event=dict(update=updates,indices=indices.tolist(),t=t0.tolist(),noise_sha256=hashlib.sha256(noise.cpu().contiguous().numpy().tobytes()).hexdigest(),loss=float(loss.detach()),gradient_norm=float(gradient),maximum_weight_ratio=ratio,**{k:float(v.detach()) for k,v in terms.items()})
            for k,v in terms.items():summed[k]=summed.get(k,0.)+float(v.detach())*n
            log(dict(stage=active,**event))
        training_seconds=time.monotonic()-training_started;assert updates==200 and exposures==400 and coverage.all()
        active='FINAL_ALL200_DFIT_LOG';held('fixed_final')
        active='FROZEN_WEIGHT_AND_CHECKPOINT_CLOSURE';after=fingerprint(frozen);assert after==before
        model_path=a.output/'fixed_final_IP_LoRA.pt'
        torch.save(dict(schema='D20_ERROR_PAIR_IP_LORA_FIXED_LAST_V1',arm=a.arm,model=V.new_state(layers),optimizer=cpu(optim.state_dict()),seed=1729,updates=updates,query_exposures=exposures,coverage=coverage,
            original_UNet_sha256=before,initial_state_sha256=initial_sha,initial_noise_rng=initial_rng,final_noise_rng=noise_rng.get_state().cpu(),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all(),ordinary_checkpoint=spec['ordinary_checkpoint'],spec=U.ref(a.spec)),model_path)
        U.atomic(a.output/'completion.json',dict(status='D20_ONE_COMPLETE_T400_PASS_FIXED_FINAL_IP_LORA_TRAINED',arm=a.arm,spec=U.ref(a.spec),source=U.ref(__file__),model=U.ref(model_path),profile=spec['profile'],wrapper=wrapper,loader=loader,host=host,
            original_UNet_sha256=before,original_weights_unchanged=after==before,initial_state_sha256=initial_sha,new_parameters=sum(p.numel() for p in params),rank=V.RANK,
            passes=1,unique_T_queries=400,T_concepts=40,subjects=1,updates=updates,query_exposures=exposures,coverage_all=True,full_all_subject_T_training=False,
            mean_T_losses={k:v/400 for k,v in summed.items()},ordinary_checkpoint=spec['ordinary_checkpoint'],Dfit_queries=200,Dfit_model_or_strength_selection=False,Dcheck_targets_read=False,
            wrong_negative=U.ref(a.output/'sameconcept_negative_permutation.npy'),training_order=U.ref(a.output/'training_order.npy'),ordinary_condition_bank=profile['ordinary_condition_bank'],
            negative_images_from='original native parent, same-query unless registered sameconcept control; not newly sampled from ordinary',normal_inference_target_access=False,normal_inference_UNet_NFE=4,
            fixed_final_checkpoint=True,exact_Turbo_likelihood_or_posterior=False,formal_D_resumed=False,training_seconds=training_seconds,seconds=time.monotonic()-started,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=torch.cuda.max_memory_allocated(),SOTA_established=False))
        log({'stage':'COMPLETE_FIXED_DISCOVERY_TRAINING'})
    except BaseException as e:
        if layers is not None and optim is not None:
            torch.save(dict(model=V.new_state(layers),optimizer=cpu(optim.state_dict()),updates=updates,query_exposures=exposures,coverage=coverage,noise_rng=noise_rng.get_state().cpu() if noise_rng is not None else None),a.output/'failed_prefix.pt')
        U.atomic(a.output/'failure.json',dict(status='FAILED_D20_FIXED_FIT_PREFIX_PRESERVED',arm=a.arm,stage=active,error=repr(e),updates=updates,query_exposures=exposures,seconds=time.monotonic()-started,formal_D_resumed=False));traceback.print_exc();raise

if __name__=='__main__':main()
