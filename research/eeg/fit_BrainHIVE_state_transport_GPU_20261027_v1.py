"""D18 matched global c-to-F flows, complete all10 OOF-T, deterministic FP32 ODE."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_state_transport_20261027_v1 as V
import brainhive_state_transport_data_20261027_v1 as DATA

def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return x

def state_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    for key in ('spec','native-registration','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();started=time.monotonic();assert a.max_seconds==300 and not a.output.exists();a.output.mkdir(parents=True)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));active='INPUTS';updates=0;exposures=0;models={};optimizers={};epoch=-1;position=0;order=None
    def log(x):
        x=x|dict(seconds=time.monotonic()-started)
        with (a.output/'training_log.jsonl').open('a') as stream:stream.write(json.dumps(x)+'\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(x),flush=True)
    try:
        spec=U.read(a.spec);assert spec['schema']==V.SCHEMA and spec['steps']==64 and spec['epochs']==10
        for item in spec['source_pins']:U.checked(item)
        import brainhive_paired_study_contract_20261025_v1 as C
        config=C.read(a.native_registration);C.validate(config);C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=300,maximum_output_bytes=768<<20,maximum_extra_CPU_bytes=10<<30,
            maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check(training=False):
            assert not stop[0] and time.monotonic()-started<(220 if training else 280),'D18 finite training/export deadline';budget.check()
        torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(1729);torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);C.resources(config,training_start=True)
        training,metadata=DATA.training_inputs('cuda');raw=training['condition'];target=training['target'];assert len(raw)==132300
        assert training['subjects']==[f'sub-{i:02d}' for i in range(1,11)]
        mean,scale=V.shared_coordinates(raw,target);source=(raw-mean)/scale;y=(target-mean)/scale
        permutation=V.concept_endpoint_permutation(training['array_rows'],metadata);permutation_t=torch.as_tensor(permutation,device='cuda')
        target_index=torch.arange(13230,device='cuda').repeat(10)
        development=DATA.development_inputs(spec,'cuda');dsource=development['raw'];dtarget=development['target'];queries=development['queries']
        d09=U.read(U.checked(spec['D09_completion']));inputs=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(inputs['panel']))
        cache=torch.load(U.checked(inputs['native_cache']),weights_only=True,map_location='cpu',mmap=True);lookup={int(row):i for i,row in enumerate(cache['array_rows'])}
        dc=cache['condition'][[lookup[q['array_row']] for q in panel['Dcheck']]].clone().cuda()
        original=np.load(U.checked(d09['check_conditions']),allow_pickle=False);assert np.array_equal(dc.cpu().numpy(),original[:,0])
        fit_lookup={(q['subject_id'],q['array_row']):i for i,q in enumerate(queries)}
        fit_ix=torch.tensor([fit_lookup[('sub-01',q['array_row'])] for q in panel['Dfit']],device='cuda');assert len(fit_ix)==200
        t_fixed=torch.linspace(.05,.95,200,device='cuda');held_x=(dsource[fit_ix]-mean)/scale;held_y=(dtarget[fit_ix]-mean)/scale
        held_state=(1-t_fixed[:,None])*held_x+t_fixed[:,None]*held_y;held_velocity=held_y-held_x
        torch.manual_seed(1729);base=V.StateFlow();base_state=base.state_dict();initial_sha=state_sha(base_state)
        for arm in V.ARMS:
            model=copy.deepcopy(base).cuda();assert state_sha(model.state_dict())==initial_sha
            models[arm]=model;optimizers[arm]=torch.optim.AdamW(model.parameters(),lr=.0003,betas=(.9,.99),weight_decay=.01)
        del base,base_state
        U.atomic(a.output/'provenance.json',dict(schema=V.SCHEMA,spec=U.ref(a.spec),source=U.ref(__file__),wrapper=wrapper,
            original_training_input_contract_SHA256=training['input_contract_sha256'],T_queries=132300,unique_F_targets=13230,subjects=10,
            parameters_per_field=4483072,initial_state_sha256=initial_sha,all2_initial_states_identical=True,
            source_coordinate='actual raw native EEG output, no per-model affine removal',normalization='one common T-only mean vector and scalar RMS for source and target',
            shared_mean=mean.cpu().tolist(),shared_scalar_RMS=float(scale),Gaussian_source_noise=False,field_inputs=['state','time'],
            training_precision='BF16 field/FP32 velocity MSE',sampling_precision='FP32 field and Heun64 state',new_EEG_or_UNet_updates=0,
            Dfit_optimizer_or_epoch_selection=False,Dcheck_targets_loaded=False,original_D_resumed=False))
        def held(epoch):
            values={}
            with torch.no_grad():
                for arm,model in models.items():
                    model.eval()
                    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(held_state,t_fixed).float()
                    values[arm]=float((pred-held_velocity).square().mean())
            log(dict(stage='FIXED_DFIT200_CORRECT_VELOCITY_LOG_ONLY',epoch=epoch,velocity_MSE=values,queries=200))
        held(0);active='ALL10_FULL_T10_MATCHED_TWO_FIELD_TRAINING';fit_started=time.monotonic()
        for epoch in range(10):
            order=torch.randperm(132300,device='cuda');coverage=np.zeros(132300,np.int8);sums={arm:0. for arm in V.ARMS};seen=0
            for position in range(0,132300,1024):
                if updates%8==0:check(training=True)
                ix=order[position:position+1024];coverage[ix.cpu().numpy()]+=1;time_draw=torch.rand(len(ix),device='cuda');xx=source[ix]
                for arm in V.ARMS:
                    model=models[arm];model.train();optimizer=optimizers[arm];optimizer.zero_grad(set_to_none=True)
                    ti=target_index[ix];yy=y[permutation_t[ti] if arm=='sameconcept_wrong_transport' else ti]
                    state=(1-time_draw[:,None])*xx+time_draw[:,None]*yy
                    with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(state,time_draw).float();loss=(prediction-(yy-xx)).square().mean()
                    assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    if updates==16:
                        branch=model.state.weight.grad;assert branch is not None and torch.isfinite(branch).all() and branch.norm()>0
                        U.atomic(a.output/(arm+'_actual_state_branch_gradient.json'),dict(stage='ACTUAL_UPDATE17',finite_nonzero=True,state_gradient_norm=float(branch.norm()),global_gradient_norm=float(norm),batch=len(ix)))
                    optimizer.step();sums[arm]+=float(loss.detach())*len(ix)
                updates+=1;exposures+=len(ix);seen+=len(ix)
                if updates%100==0:log(dict(stage='TRAIN_UPDATE',epoch=epoch+1,updates_per_field=updates,exposures_per_field=exposures))
            assert seen==132300 and np.all(coverage==1) and updates==130*(epoch+1)
            log(dict(stage='COMPLETE_T_PASS',epoch=epoch+1,updates_per_field=updates,exposures_per_field=exposures,
                full_coverage_min=int(coverage.min()),full_coverage_max=int(coverage.max()),order_sha256=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest(),
                velocity_MSE={arm:value/seen for arm,value in sums.items()}));held(epoch+1)
        training_seconds=time.monotonic()-fit_started;assert updates==1300 and exposures==1323000
        for model in models.values():model.eval().requires_grad_(False)
        active='FIXED_FINAL_MODEL_SAVE';checkpoint=a.output/'fixed_final_fields.pt'
        torch.save(dict(models={arm:cpu(model.state_dict()) for arm,model in models.items()},optimizers={arm:cpu(opt.state_dict()) for arm,opt in optimizers.items()},
            mean=mean.cpu(),scale=scale.cpu(),epoch=10,updates_per_field=1300,exposures_per_field=1323000,steps=64,
            CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),spec=U.ref(a.spec)),checkpoint)
        def integrate(model,value,steps=64,reverse=False):
            pieces=[]
            for batch in value.split(256):check();pieces.append(V.transport(model,batch,mean,scale,steps,reverse))
            return torch.cat(pieces)
        active='FULL8200_DFIT_AND400_SOURCE_ODE_DIAGNOSTICS';fit_output=[];fit_back=[];dc_output=[];dc_back=[];dc_refine=[];numeric={};geometry={};supported=True
        for arm,model in models.items():
            check();out=integrate(model,dsource);back=integrate(model,out,reverse=True);dout=integrate(model,dc);dback=integrate(model,dout,reverse=True);refined=integrate(model,dc,steps=128)
            dferr=(back-dsource).norm(dim=-1)/dsource.norm(dim=-1);dcerr=(dback-dc).norm(dim=-1)/dc.norm(dim=-1);step_error=(refined-dout).norm(dim=-1)/dc.norm(dim=-1)
            numeric[arm]=dict(Dfit_max_relative_roundtrip=float(dferr.max()),Dcheck_max_relative_roundtrip=float(dcerr.max()),Dcheck_max_relative64_vs128_refinement=float(step_error.max()),
                fixed_steps=64,refinement_changes_no_model_or_samples=True,all8200_and400_used=True)
            supported &= max(numeric[arm]['Dfit_max_relative_roundtrip'],numeric[arm]['Dcheck_max_relative_roundtrip'],numeric[arm]['Dcheck_max_relative64_vs128_refinement'])<=.01
            mask=torch.triu(torch.ones(10,10,device='cuda',dtype=torch.bool),1)
            native_distance=torch.cdist(dsource.reshape(-1,10,1024),dsource.reshape(-1,10,1024))[:,mask]
            output_distance=torch.cdist(out.reshape(-1,10,1024),out.reshape(-1,10,1024))[:,mask]
            assert (native_distance>0).all();ratio=(output_distance/native_distance).reshape(-1)
            geometry[arm]=dict(within_concept_pairs=int(len(ratio)),distance_ratio_quantiles=torch.quantile(ratio,torch.tensor([0.,.01,.05,.5,.95,.99,1.],device='cuda')).cpu().tolist(),
                fraction_distance_ratio_below_point1=float((ratio<.1).float().mean()),raw_condition_norm_mean=float(out.norm(dim=-1).mean()),
                no_information_preservation_or_semantic_gain_conclusion=True)
            fit_output.append(out.cpu());fit_back.append(back.cpu());dc_output.append(dout.cpu());dc_back.append(dback.cpu());dc_refine.append(refined.cpu())
            log(dict(stage='COMPLETE_ALL_SOURCE_NUMERICAL_CHECK',arm=arm,**numeric[arm],geometry=geometry[arm]))
        for name,value in [('Dfit_conditions',torch.stack(fit_output,1)),('Dfit_roundtrip',torch.stack(fit_back,1)),('Dfit_source',dsource.cpu()),('Dfit_targets',dtarget.cpu()),
            ('Dcheck_conditions',torch.stack(dc_output,1)),('Dcheck_roundtrip',torch.stack(dc_back,1)),('Dcheck_refined128',torch.stack(dc_refine,1)),('Dcheck_source',dc.cpu())]:U.save_array(a.output/(name+'.npy'),value.numpy())
        U.atomic(a.output/'endpoint_coupling.json',dict(T_array_rows=training['array_rows'],sameconcept_wrong_endpoint_positions=permutation.tolist(),
            same_subject_source_conditions_unchanged=True,target_marginals_identical=True,permutation_inference_usage=False))
        U.atomic(a.output/'Dfit_queries.json',dict(queries=queries,groups=development['groups']))
        diagnostics=dict(status='D18_COMPLETE_FIXED64_ODE_FULL_SOURCE_ROUNDTRIP_AND_REFINEMENT_DIAGNOSTICS',numerical_gate_passed=bool(supported),arms=numeric,geometry=geometry,
            gate='both fields, all8200Dfit/all400Dcheck: max relative reverse64 and forward64-vs128 <=.01; numerical only',images_or_target_F_used_to_choose_steps=False,
            finite_inverse_error_is_not_semantic_or_information_preservation_proof=True)
        U.atomic(a.output/'numerical_diagnostics.json',diagnostics)
        lock=dict(status='D18_ALL400_TWO_FINAL_FIELDS_LOCKED_BEFORE_IMAGES',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            conditions=U.ref(a.output/'Dcheck_conditions.npy'),source_conditions=U.ref(a.output/'Dcheck_source.npy'),model=U.ref(checkpoint),spec=U.ref(a.spec),
            query_ids=[q['query_id'] for q in panel['Dcheck']],arms=V.ARMS,field_inputs=['state','time'],source='original native EEG c',
            numerical_gate_passed=bool(supported),fixed_steps=64,small_field_evaluations_per_query=128,independent_Gaussian_noise=False,
            no_per_query_target_norm_or_F_affine_projection=True,query_target_F_or_images_used=False,target_rank_or_DINO_used_for_choices=False,
            normal_inference_only_current_query_EEG=True,normal_inference_UNet_NFE=4,original_D_resumed=False)
        U.atomic(a.output/'condition_lock.json',lock)
        for item in spec['source_pins']:U.checked(item)
        check();assert sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())<768<<20
        done=dict(status='D18_MATCHED_TWO_GLOBAL_FIELDS_FULL_T10_AND_ALL_SOURCE_DIAGNOSTICS_COMPLETE',schema=V.SCHEMA,source=U.ref(__file__),spec=U.ref(a.spec),wrapper=wrapper,
            model=U.ref(checkpoint),condition_lock=U.ref(a.output/'condition_lock.json'),diagnostics=U.ref(a.output/'numerical_diagnostics.json'),Dfit_queries=U.ref(a.output/'Dfit_queries.json'),
            Dfit_conditions=U.ref(a.output/'Dfit_conditions.npy'),Dfit_source=U.ref(a.output/'Dfit_source.npy'),Dfit_targets=U.ref(a.output/'Dfit_targets.npy'),Dfit_roundtrip=U.ref(a.output/'Dfit_roundtrip.npy'),
            Dcheck_roundtrip=U.ref(a.output/'Dcheck_roundtrip.npy'),Dcheck_refined128=U.ref(a.output/'Dcheck_refined128.npy'),endpoint_coupling=U.ref(a.output/'endpoint_coupling.json'),
            numerical_gate_passed=bool(supported),image_admission='NUMERICAL_ONLY_SUPPORTS_REGISTERED_IMAGE_TEST' if supported else 'STOP_NUMERICALLY_UNRESOLVED_NO_IMAGES',
            fields=2,parameters_per_field=4483072,epochs=10,updates_per_field=1300,query_exposures_per_field=1323000,all10_complete_real_OOF_T=True,
            no_Dfit_optimizer_or_checkpoint_selection=True,no_Dcheck_target_access=True,old_EEG_prior_or_UNet_weights_updated=False,original_D_resumed=False,
            independent_Gaussian_source_noise=False,normal_inference_only_current_query_EEG=True,inference_small_field_NFE=128,inference_UNet_NFE=4,
            training_seconds=training_seconds,seconds=time.monotonic()-started,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,
            independent_confirmation=False,SOTA_established=False)
        U.atomic(a.output/'completion.json',done);log(dict(stage='COMPLETE',numerical_gate_passed=bool(supported)));return 0
    except BaseException as error:
        if models:
            torch.save(dict(models={arm:cpu(model.state_dict()) for arm,model in models.items()},optimizers={arm:cpu(opt.state_dict()) for arm,opt in optimizers.items()},
                updates=updates,exposures=exposures,epoch=epoch,position=position,order=cpu(order),CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all()),a.output/'partial_training_state.pt')
        U.atomic(a.output/'failure.json',dict(status='D18_INCOMPLETE_RETAINED_NO_AUTOMATIC_RETRY',active=active,updates_per_field=updates,exposures_per_field=exposures,
            seconds=time.monotonic()-started,error=repr(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':raise SystemExit(main())
