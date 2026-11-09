"""D12: ten complete OOF-T passes, final epoch only; fixed one-sample inference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_conditional_flow_20261027_v1 as V


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','d09-completion','native-registration','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True); p.add_argument('--max-seconds',type=float,required=True)
    args=p.parse_args(); started=time.monotonic(); assert args.max_seconds==240
    assert not args.output.exists(); args.output.mkdir(parents=True)
    stop=[False]; signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    active='INPUTS'; model=None; optimizer=None; updates=0; exposures=0; epoch=-1; position=0; order=None
    def log(value):
        value=value|dict(seconds=time.monotonic()-started)
        with (args.output/'training_log.jsonl').open('a') as s: s.write(json.dumps(value,allow_nan=False)+'\n'); s.flush(); os.fsync(s.fileno())
        print(json.dumps(value,allow_nan=False),flush=True)
    try:
        spec=U.read(args.spec); assert spec['schema']==V.SCHEMA and spec['epochs']==10 and spec['batch_size']==1024
        for source in spec['source_pins']: U.checked(source)
        assert U.ref(args.d09_completion)==spec['D09_completion']
        d09=U.read(args.d09_completion); inputs=U.read(U.checked(d09['inputs'])); panel=U.read(U.checked(inputs['panel']))
        import brainhive_paired_study_contract_20261025_v1 as C
        config=C.read(args.native_registration); C.validate(config); C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch
        document=dict(gpu_job_name=args.gpu_job_name,maximum_seconds=args.max_seconds,
            maximum_output_bytes=512<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,
            minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
        wrapper=D.verify_wrapper(document); budget=D.Budget(document)
        def check(work=False):
            assert not stop[0] and time.monotonic()-started < (180 if work else 230), 'D12 fixed training/export work deadline'
            budget.check()
        torch.set_num_threads(2); torch.manual_seed(1729); np.random.seed(1729); torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True); C.resources(config,training_start=True); check()
        from brainhive_component_data_v1 import load_training
        from brainhive_native_models_v1 import prior_models
        active='ALL10_ACTUAL_OOF_T_INPUTS'
        training=load_training('cuda'); n=len(training['condition']); assert n==132300
        assert training['subjects']==[f'sub-{j:02d}' for j in range(1,11)]
        split=U.read(U.checked(inputs['split'])); assert training['array_rows']==split['image_rows']['T']
        assert set(training['array_rows']).isdisjoint(split['image_rows']['D']+split['image_rows']['V'])
        condition=training['condition']; assert torch.isfinite(condition).all()
        subject=torch.arange(10,device='cuda').repeat_interleave(13230)
        lookup={row:j for j,row in enumerate(training['array_rows'])}
        target_index=torch.arange(13230,device='cuda').repeat(10)
        frozen=prior_models('cpu'); norm=frozen['fusion_encoder'].vision_model.block.layers[0].norm
        weight=norm.weight.detach().clone().cuda(); bias=norm.bias.detach().clone().cuda()
        assert norm.eps==1e-6 and weight.abs().min()>1e-6
        affine_source=dict(weight_sha256=hashlib.sha256(weight.cpu().numpy().tobytes()).hexdigest(),
            bias_sha256=hashlib.sha256(bias.cpu().numpy().tobytes()).hexdigest(),module='fusion_encoder.vision_model.block.layers.0.norm',eps=1e-6)
        del norm,frozen
        u=(training['target']-bias)/weight
        mean=u.mean(0); scale=(u-mean).square().mean().sqrt(); assert .01<float(scale)<10
        y=(u-mean)/scale; assert y.shape==(13230,1024) and torch.isfinite(y).all()
        # Dfit is read solely for logging. It never enters optimizer or epoch selection.
        cache=torch.load(U.checked(inputs['native_cache']),weights_only=True,map_location='cpu',mmap=True)
        cache_lookup={int(row):j for j,row in enumerate(cache['array_rows'])}
        def native(qs):
            ids=[cache_lookup[q['array_row']] for q in qs]
            assert cache['array_rows'][ids].tolist()==[q['array_row'] for q in qs]
            return cache['condition'][ids].clone().cuda()
        fit_queries=panel['Dfit']; assert len(fit_queries)==200
        fit_c=native(fit_queries); fit_subject=torch.zeros(200,dtype=torch.long,device='cuda')
        contract=U.read('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json')
        all_F=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),
            weights_only=True,map_location='cpu',mmap=True)
        fit_F=all_F[[q['array_row'] for q in fit_queries]].clone().cuda()
        fit_y=((fit_F-bias)/weight-mean)/scale
        fit_z,fit_seeds=V.query_noise(fit_queries); fit_z=fit_z.cuda()
        fit_t=torch.linspace(.05,.95,200,device='cuda'); fit_x=(1-fit_t[:,None])*fit_z+fit_t[:,None]*fit_y
        fit_v=fit_y-fit_z
        group_perm=U.derangement(20); fit_permutation=(group_perm[:,None]*10+np.arange(10)[None]).reshape(-1)
        assert all(fit_queries[i]['concept_id']!=fit_queries[j]['concept_id'] for i,j in enumerate(fit_permutation))
        fit_wrong=fit_c[torch.as_tensor(fit_permutation,device='cuda')]
        model=V.Flow().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.99),weight_decay=.01)
        params=sum(x.numel() for x in model.parameters()); assert params<6_000_000
        U.atomic(args.output/'provenance.json',dict(schema=V.SCHEMA,spec=U.ref(args.spec),source=U.ref(__file__),
            D09_completion=U.ref(args.d09_completion),wrapper=wrapper,training_input_contract_sha256=training['input_contract_sha256'],
            actual_training_queries=n,unique_training_visual_targets=13230,subjects=training['subjects'],
            new_parameters=params,epochs=10,batch_size=1024,expected_updates=1300,expected_exposures=1323000,
            fixed_final_epoch=True,Dfit_for_gradient_or_checkpoint_selection=False,Dcheck_F_targets_read=False,
            projection_uses_frozen_F_output_affine=True,affine_source=affine_source,
            target_center_and_scalar_scale_from_T_only=True,target_scalar_scale=float(scale),
            no_PCA_or_rank_truncation=True,training_sphere_input_mean_abs_max=float(u.mean(-1).abs().max()),
            training_sphere_input_second_moment_range=[float(u.square().mean(-1).min()),float(u.square().mean(-1).max())],
            original_D_resumed=False,UNet_training_or_loading=False))
        @torch.no_grad()
        def held(epoch):
            model.eval()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                correct=model(fit_x,fit_t,fit_c,fit_subject).float()
                wrong=model(fit_x,fit_t,fit_wrong,fit_subject).float()
            log(dict(stage='FIXED_DFIT_LOSS_DIAGNOSTIC',epoch=epoch,queries=200,
                correct_velocity_MSE=float((correct-fit_v).square().mean()),wrong_EEG_velocity_MSE=float((wrong-fit_v).square().mean()),
                wrong_minus_correct=float(((wrong-fit_v).square()-(correct-fit_v).square()).mean()),
                same_fixed_states_times_and_noise=True,fit_or_selection_permission=False))
        held(0); fit_started=time.monotonic(); active='TRAINING_FULL_OOF_T_10_PASSES'
        first_gradient_profile=None
        for epoch in range(10):
            model.train(); order=torch.randperm(n,device='cuda'); total=0.; norm_total=0.; seen=0
            order_sha=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
            for position in range(0,n,1024):
                if updates%8==0: check(work=True)
                ix=order[position:position+1024]; target=y[target_index[ix]]; noise=torch.randn_like(target)
                t=torch.rand(len(ix),device='cuda'); x=(1-t[:,None])*noise+t[:,None]*target
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=model(x,t,condition[ix],subject[ix]).float(); loss=(prediction-(target-noise)).square().mean()
                assert torch.isfinite(loss)
                loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(grad) and grad>0
                if first_gradient_profile is None:
                    first_gradient_profile=dict(first_actual_batch=len(ix),finite_nonzero_gradient=True,
                        global_gradient_norm=float(grad),all_base_models_frozen=True,UNet_forwards=0,
                        GPU_allocated_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=budget.peak_extra_CPU_bytes)
                    U.atomic(args.output/'first_gradient_profile.json',first_gradient_profile)
                optimizer.step(); updates+=1; exposures+=len(ix); seen+=len(ix)
                total+=float(loss)*len(ix); norm_total+=float(grad)*len(ix)
                if updates%25==0: log(dict(stage='TRAIN_UPDATE',epoch=epoch+1,updates=updates,exposures=exposures,
                    current_batch_velocity_MSE=float(loss),mean_gradient_norm=norm_total/seen))
            assert seen==n
            log(dict(stage='COMPLETE_T_PASS',epoch=epoch+1,updates=updates,exposures=exposures,
                train_velocity_MSE=total/seen,mean_gradient_norm=norm_total/seen,order_sha256=order_sha,
                all10_all13230_queries_exposed_once_this_pass=True))
            held(epoch+1)
        torch.cuda.synchronize(); training_seconds=time.monotonic()-fit_started
        assert updates==1300 and exposures==1323000; check()
        active='SAVE_FIXED_FINAL_MODEL'
        checkpoint=args.output/'final_model.pt'
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
            mean=mean.cpu(),scale=scale.cpu(),F_weight=weight.cpu(),F_bias=bias.cpu(),
            epoch=10,updates=updates,exposures=exposures,CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state(),
            input_contract_sha256=training['input_contract_sha256'],spec_sha256=U.sha(args.spec)),checkpoint)
        active='ONE_FIXED_SAMPLE_CORRECT_AND_SHUFFLED_EEG_ALL400'
        model.eval(); check_queries=panel['Dcheck']; c=native(check_queries)
        permutation=V.OLD.control_permutation(check_queries); initial,seeds=V.query_noise(check_queries)
        z=initial.cuda(); ids=torch.zeros(400,dtype=torch.long,device='cuda')
        sample_start=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            correct,raw_correct=V.sample(model,z,c,ids,mean,scale,weight,bias)
            wrong,raw_wrong=V.sample(model,z,c[torch.as_tensor(permutation,device='cuda')],ids,mean,scale,weight,bias)
        torch.cuda.synchronize(); sampling_seconds=time.monotonic()-sample_start
        bank=torch.stack([correct,wrong],1).cpu().numpy(); assert bank.shape==(400,2,1024) and np.isfinite(bank).all()
        U.save_array(args.output/'conditions_all400.npy',bank)
        U.save_array(args.output/'conditional_initial_noise_all400.npy',initial.numpy())
        U.save_array(args.output/'raw_preprojection_u_all400.npy',torch.stack([raw_correct,raw_wrong],1).cpu().numpy())
        lock=dict(status='D12_ALL400_FIXED_MODEL_SAMPLES_LOCKED_BEFORE_D12_IMAGES',query_ids=[q['query_id'] for q in check_queries],
            conditions=U.ref(args.output/'conditions_all400.npy'),model=U.ref(checkpoint),
            conditional_initial_noise=U.ref(args.output/'conditional_initial_noise_all400.npy'),conditional_noise_seeds=seeds,
            raw_preprojection_u=U.ref(args.output/'raw_preprojection_u_all400.npy'),shuffled_EEG_query_permutation=permutation.tolist(),
            small_flow_steps_per_query=32,conditional_samples_per_query=1,correct_and_wrong_EEG_same_initial_condition_noise=True,
            query_target_F_features_used=False,query_target_images_used=False,target_rank_or_DINO_used_for_choices=False,
            normal_inference_only_EEG_subject_and_frozen_trained_model=True,new_fitted_weights=params,
            host_F_affine_projection_used=True,no_query_target_norm_or_scaling=True,
            original_D_resumed=False,development_only=True,independent_confirmation=False)
        U.atomic(args.output/'all400_condition_lock.json',lock)
        final=dict(status='D12_FULL_OOF_T_10_EPOCHS_AND_ALL400_ONE_SAMPLE_CONDITIONS_COMPLETE',schema=V.SCHEMA,
            source=U.ref(__file__),spec=U.ref(args.spec),D09_completion=U.ref(args.d09_completion),D09_inputs=d09['inputs'],panel=inputs['panel'],
            lock=U.ref(args.output/'all400_condition_lock.json'),model=U.ref(checkpoint),native_EEG_cache=inputs['native_cache'],
            native_checkpoint=inputs['native_checkpoint'],baseline_generated_features=d09['check_generated_features'],
            target_features=d09['check_target_features'],baseline_manifest=d09['check_manifest'],
            input_contract_sha256=training['input_contract_sha256'],updates=updates,main_query_exposures=exposures,full_passes=10,
            new_parameters=params,training_body_seconds=training_seconds,sample_all800_seconds=sampling_seconds,
            normal_inference_target_access=False,query_targets_used_for_condition_inference=False,original_D_resumed=False,
            EEG_or_UNet_base_weights_updated=False,paired_correct_and_shuffled_conditions_identical=bool(np.array_equal(bank[:,0],bank[:,1])),
            first_gradient_profile=U.ref(args.output/'first_gradient_profile.json'),peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,
            peak_GPU_bytes=budget.peak_GPU_bytes,seconds=time.monotonic()-started,
            no_automatic_hyperparameter_search_or_retry=True,independent_confirmation=False,SOTA_established=False)
        for source in spec['source_pins']: U.checked(source)
        check(); U.atomic(args.output/'completion.json',final); log(dict(stage='COMPLETE',updates=updates,exposures=exposures)); return 0
    except BaseException as error:
        if model is not None:
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict() if optimizer else None,
                epoch=epoch,updates=updates,exposures=exposures,position=position,order=order,
                CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state()),args.output/'incomplete_checkpoint.pt')
        U.atomic(args.output/'failure.json',dict(status='D12_FIT_OR_EXPORT_INCOMPLETE_NO_AUTOMATIC_RETRY',active=active,
            error=repr(error),traceback=traceback.format_exc(),updates=updates,exposures=exposures,seconds=time.monotonic()-started))
        raise


if __name__=='__main__': raise SystemExit(main())
