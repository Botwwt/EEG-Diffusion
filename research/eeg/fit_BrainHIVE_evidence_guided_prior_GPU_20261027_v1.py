"""D14: true marginal visual flow, frozen D13 c-only evidence, final10 fixed passes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_evidence_guided_prior_20261027_v1 as V
import brainhive_noisy_EEG_evidence_20261027_v1 as E


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','native-registration','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True); p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args(); start=time.monotonic(); assert a.max_seconds==180
    assert not a.output.exists(); a.output.mkdir(parents=True)
    stopped=[False]; signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    model=None; optimizer=None; updates=0; exposures=0; active='INPUTS'
    def log(value):
        value=value|dict(seconds=time.monotonic()-start)
        with (a.output/'training_log.jsonl').open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(value,allow_nan=False),flush=True)
    try:
        spec=U.read(a.spec); assert spec['schema']==V.SCHEMA and spec['epochs']==10 and spec['batch_size']==1024
        for ref in spec['source_pins']: U.checked(ref)
        assert U.read(U.checked(spec['CPU_checks']))['status']=='D14_ALL_CPU_MATH_AND_PRODUCTION_CHECKS_PASSED'
        old=U.read(U.checked(spec['D12_fit'])); evidence=U.read(U.checked(spec['D13_completion']))
        readout=U.read(U.checked(evidence['readout']))
        assert readout['decision']['action']=='BASE_C_ONLY_EVIDENCE_SUPPORTS_SEPARATE_TEST_NO_RAW_INCREMENT'
        assert readout['raw_EEG_increment_supported'] is False
        d09=U.read(U.checked(spec['D09_completion'])); inputs=U.read(U.checked(d09['inputs'])); panel=U.read(U.checked(inputs['panel']))
        import brainhive_paired_study_contract_20261025_v1 as C
        config=C.read(a.native_registration); C.validate(config); C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch
        doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=a.max_seconds,maximum_output_bytes=512<<20,
            maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
        wrapper=D.verify_wrapper(doc); budget=D.Budget(doc)
        def check(work=False):
            assert not stopped[0] and time.monotonic()-start<(130 if work else 165),'D14 finite work budget'
            budget.check()
        torch.set_num_threads(2); torch.manual_seed(1729); np.random.seed(1729); torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True); torch.cuda.reset_peak_memory_stats(); C.resources(config,training_start=True)
        previous=torch.load(U.checked(old['model']),weights_only=True,map_location='cpu',mmap=True)
        mean=previous['mean'].clone().cuda(); scale=previous['scale'].clone().cuda()
        fw=previous['F_weight'].clone().cuda(); fb=previous['F_bias'].clone().cuda()
        split=U.read(U.checked(inputs['split'])); rows=split['image_rows']['T']; assert len(rows)==13230
        assert set(rows).isdisjoint(split['image_rows']['D']+split['image_rows']['V'])
        contract_path=Path('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json')
        contract=U.read(contract_path)
        targets_ref=dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])
        targets=torch.load(U.checked(targets_ref),weights_only=True,map_location='cpu',mmap=True)
        y=((targets[rows].clone().cuda()-fb)/fw-mean)/scale
        assert y.shape==(13230,1024) and torch.isfinite(y).all()
        # Every subject in D13 has precisely the same uniform T visual marginal.
        # Repetition here matches prior D12 exposures, not additional EEG fitting.
        target_index=torch.arange(13230,device='cuda').repeat(10); n=len(target_index)
        fit_q=panel['Dfit']; fit_t=torch.linspace(.05,.95,200,device='cuda')
        fit_z,_=V.OLD.query_noise(fit_q); fit_z=fit_z.cuda()
        fit_y=((targets[[q['array_row'] for q in fit_q]].clone().cuda()-fb)/fw-mean)/scale
        fit_x=(1-fit_t[:,None])*fit_z+fit_t[:,None]*fit_y; fit_velocity=fit_y-fit_z
        model=V.Marginal().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.99),weight_decay=.01)
        params=sum(v.numel() for v in model.parameters()); assert params==4483072
        U.atomic(a.output/'provenance.json',dict(schema=V.SCHEMA,spec=U.ref(a.spec),source=U.ref(__file__),wrapper=wrapper,
            D12_normalizer=old['model'],D13_frozen_heads=evidence['final_heads'],target_cache=targets_ref,
            input_contract=U.ref(contract_path),T_rows=rows,unique_visual_targets=13230,virtual_visual_indices_per_pass=n,
            new_EEG_queries_fitted=0,new_parameters=params,marginal_has_no_c_or_subject_input=True,
            fixed_final_epoch=10,Dfit_for_fitting_or_selection=False,Dcheck_targets_for_condition_sampling=False,
            original_D_resumed=False,UNet_training_or_loading=False))
        @torch.no_grad()
        def held(epoch):
            model.eval()
            with torch.autocast('cuda',dtype=torch.bfloat16): v=model(fit_x,fit_t).float()
            log(dict(stage='FIXED_DFIT_VELOCITY_LOG_ONLY',epoch=epoch,velocity_MSE=float((v-fit_velocity).square().mean()),fit_or_checkpoint_selection=False))
        held(0); fit_start=time.monotonic(); active='TRAINING_TRUE_VISUAL_MARGINAL'
        for epoch in range(10):
            model.train(); order=torch.randperm(n,device='cuda'); total=0.; seen=0
            order_sha=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
            for pos in range(0,n,1024):
                if updates%8==0: check(work=True)
                ix=order[pos:pos+1024]; target=y[target_index[ix]]; z=torch.randn_like(target); t=torch.rand(len(ix),device='cuda')
                x=(1-t[:,None])*z+t[:,None]*target; optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=(model(x,t).float()-(target-z)).square().mean()
                assert torch.isfinite(loss); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(grad) and grad>0
                if updates==0:
                    U.atomic(a.output/'first_gradient_profile.json',dict(batch=len(ix),finite_nonzero_gradient=True,gradient_norm=float(grad),new_parameters=params,
                        GPU_allocated_bytes=torch.cuda.max_memory_allocated(),UNet_forwards=0,marginal_no_EEG_or_subject_input=True))
                optimizer.step(); updates+=1; exposures+=len(ix); seen+=len(ix); total+=float(loss)*len(ix)
                if updates%25==0: log(dict(stage='TRAIN_UPDATE',epoch=epoch+1,updates=updates,visual_exposures=exposures,velocity_MSE=float(loss)))
            assert seen==n
            log(dict(stage='COMPLETE_VISUAL_T_PASS',epoch=epoch+1,updates=updates,visual_exposures=exposures,
                train_velocity_MSE=total/seen,order_sha256=order_sha,all132300_visual_indices_seen_once=True)); held(epoch+1)
        torch.cuda.synchronize(); training_seconds=time.monotonic()-fit_start
        assert updates==1300 and exposures==1323000; check(); active='FIXED_FINAL_CHECKPOINT'
        checkpoint=a.output/'final_model.pt'
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
            mean=mean.cpu(),scale=scale.cpu(),F_weight=fw.cpu(),F_bias=fb.cpu(),epoch=10,updates=updates,visual_exposures=exposures,
            CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state(),spec_sha256=U.sha(a.spec)),checkpoint)
        saved=torch.load(U.checked(evidence['final_heads']),weights_only=True,map_location='cpu',mmap=True)
        for name,value in [('F_mean',mean),('F_scale',scale),('F_weight',fw),('F_bias',fb)]: assert torch.equal(saved[name],value.cpu())
        head=E.Evidence().cuda(); head.load_state_dict(saved['models']['c_only'],strict=True); head.eval().requires_grad_(False)
        assert sum(v.numel() for v in head.parameters())==2748065
        cache=torch.load(U.checked(inputs['native_cache']),weights_only=True,map_location='cpu',mmap=True)
        lookup={int(row):j for j,row in enumerate(cache['array_rows'])}; qs=panel['Dcheck']
        indices=[lookup[q['array_row']] for q in qs]; assert cache['array_rows'][indices].tolist()==[q['array_row'] for q in qs]
        c=cache['condition'][indices].clone().cuda(); subjects=torch.zeros(400,dtype=torch.long,device='cuda')
        permutation=V.OLD.OLD.control_permutation(qs); initial,seeds=V.OLD.query_noise(qs); z=initial.cuda()
        active='FIXED_ALL400_THREE_ARM_CONDITIONS_BEFORE_IMAGES'; model.eval(); sampling_start=time.monotonic()
        finals=[]; raw=[]; state=[]; guidance=[]; diagnostics=[]
        for arm in V.ARMS:
            check(); condition=c[torch.as_tensor(permutation,device='cuda')] if arm=='guided_shuffled_c' else c
            final,endpoint,trace=V.sample(model,head if arm!='marginal_prior' else None,z,condition,subjects,mean,scale,fw,fb)
            finals.append(final.cpu()); raw.append(endpoint.cpu()); state.append(trace['states']); guidance.append(trace['guidance']); diagnostics.append(trace['diagnostics'])
        torch.cuda.synchronize(); sampling_seconds=time.monotonic()-sampling_start
        bank=torch.stack(finals,1).numpy(); assert bank.shape==(400,3,1024) and np.isfinite(bank).all()
        arrays={'conditions_all400':bank,'conditional_initial_noise_all400':initial.numpy(),'raw_preprojection_u_all400':torch.stack(raw,1).numpy(),
            'trajectory_states':np.stack(state,1),'trajectory_guidance':np.stack(guidance,1),'all32_step_diagnostics':np.stack(diagnostics,1)}
        for name,value in arrays.items(): U.save_array(a.output/(name+'.npy'),value)
        lock=dict(status='D14_ALL400_THREE_ARM_CONDITIONS_LOCKED_BEFORE_IMAGES',query_ids=[q['query_id'] for q in qs],model=U.ref(checkpoint),
            frozen_D13_head=evidence['final_heads'],frozen_head_arm='c_only',arms=V.ARMS,
            conditions=U.ref(a.output/'conditions_all400.npy'),conditional_initial_noise=U.ref(a.output/'conditional_initial_noise_all400.npy'),
            conditional_noise_seeds=seeds,raw_preprojection_u=U.ref(a.output/'raw_preprojection_u_all400.npy'),shuffled_EEG_query_permutation=permutation.tolist(),
            trajectory_states=U.ref(a.output/'trajectory_states.npy'),trajectory_guidance=U.ref(a.output/'trajectory_guidance.npy'),
            all32_step_diagnostics=U.ref(a.output/'all32_step_diagnostics.npy'),small_flow_steps_per_query=32,conditional_samples_per_query=1,
            guidance_formula='(1-t)*grad_x h, original frozen D13 r=t*h, fixed coefficient1',guidance_hyperparameter_search=False,
            correct_and_wrong_EEG_same_initial_condition_noise=True,query_target_F_features_used=False,query_target_images_used=False,
            target_rank_or_DINO_used_for_choices=False,normal_inference_only_EEG_subject_and_frozen_trained_model=True,
            host_F_affine_projection_used=True,no_query_target_norm_or_scaling=True,exact_posterior_established=False,original_D_resumed=False)
        U.atomic(a.output/'all400_condition_lock.json',lock)
        result=dict(status='D14_MARGINAL_FIT_AND_ALL400_FIXED_C_EVIDENCE_CONDITIONS_COMPLETE',schema=V.SCHEMA,spec=U.ref(a.spec),source=U.ref(__file__),
            D09_completion=spec['D09_completion'],D09_inputs=d09['inputs'],panel=inputs['panel'],lock=U.ref(a.output/'all400_condition_lock.json'),
            model=U.ref(checkpoint),frozen_D13_head=evidence['final_heads'],baseline_generated_features=d09['check_generated_features'],
            target_features=d09['check_target_features'],baseline_manifest=d09['check_manifest'],new_parameters=params,frozen_evidence_parameters=2748065,
            updates=updates,visual_exposures=exposures,full_passes=10,new_EEG_queries_fitted=0,training_body_seconds=training_seconds,
            sample_all1200_seconds=sampling_seconds,normal_inference_target_access=False,original_D_resumed=False,EEG_or_UNet_base_weights_updated=False,
            paired_correct_and_shuffled_conditions_identical=bool(np.array_equal(bank[:,0],bank[:,1])),
            marginal_conditions_identical_to_guided=bool(np.array_equal(bank[:,0],bank[:,2])),
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,seconds=time.monotonic()-start,
            no_automatic_hyperparameter_search_or_retry=True,independent_confirmation=False,SOTA_established=False)
        U.checked(evidence['final_heads'])
        for ref in spec['source_pins']: U.checked(ref)
        check(); U.atomic(a.output/'completion.json',result); log(dict(stage='COMPLETE',updates=updates,visual_exposures=exposures)); return 0
    except BaseException as error:
        if model is not None: torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict() if optimizer else None,updates=updates,visual_exposures=exposures),a.output/'incomplete_checkpoint.pt')
        U.atomic(a.output/'failure.json',dict(status='D14_INCOMPLETE_PRESERVED_NO_AUTOMATIC_RETRY',active=active,error=repr(error),traceback=traceback.format_exc(),
            updates=updates,visual_exposures=exposures,seconds=time.monotonic()-start)); raise


if __name__=='__main__': raise SystemExit(main())
