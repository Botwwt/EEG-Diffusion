"""D16 four matched small heads, full OOF-T10 passes, fixed last; no UNet training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_response_residual_20261027_v1 as V
import brainhive_response_residual_data_20261027_v1 as DATA


def main():
    p=argparse.ArgumentParser()
    for n in ('spec','native-registration','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();start=time.monotonic();assert a.max_seconds==300 and not a.output.exists();a.output.mkdir(parents=True)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));active='INPUTS';updates=0;exposures=0;models=None;optimizers=None
    def log(value):
        value=value|dict(seconds=time.monotonic()-start)
        with (a.output/'training_log.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(value,allow_nan=False),flush=True)
    try:
        spec=U.read(a.spec);assert spec['schema']==V.SCHEMA and spec['epochs']==10 and spec['groups_per_batch']==100
        for item in spec['source_pins']:U.checked(item)
        assert U.read(U.checked(spec['CPU_checks']))['status']=='D16_ALL7_NATIVE_RESPONSE_GEOMETRY_LOSS_AND_GATE_CPU_CHECKS_PASSED'
        import brainhive_paired_study_contract_20261025_v1 as C
        config=C.read(a.native_registration);C.validate(config);C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=a.max_seconds,maximum_output_bytes=512<<20,
            maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check(training=False):assert not stop[0] and time.monotonic()-start<(230 if training else 280),'D16 finite fixed work deadline';budget.check()
        torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(1729);np.random.seed(1729)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
        C.resources(config,training_start=True);active='ALL10_FULL_T_AND_T_ONLY_RESPONSE_NORMALIZERS'
        train=DATA.load_T(spec);state=train['state'];check();held=DATA.load_Dfit(spec,state)
        assert train['u'].shape==(132300,1024) and held['u'].shape==(8200,1024)
        base=V.ResponseAdapter().cuda();models={arm:copy.deepcopy(base) for arm in V.ARMS};del base
        optimizers={arm:torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.99),weight_decay=.01) for arm,model in models.items()}
        params=sum(x.numel() for x in models[V.ARMS[0]].parameters());assert params==527360
        initial_hash={}
        for arm,model in models.items():
            digest=hashlib.sha256()
            for name,value in sorted(model.state_dict().items()):digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            initial_hash[arm]=digest.hexdigest()
        assert len(set(initial_hash.values()))==1
        U.atomic(a.output/'provenance.json',dict(spec=U.ref(a.spec),source=U.ref(__file__),wrapper=wrapper,
            source_training_input_contract_sha256=train['input_contract_sha256'],initial_state_sha256=initial_hash,parameters_per_head=params,
            all_original_encoder_generator_and_linear_readout_weights_frozen=True,actual_training_queries=132300,groups=13230,images_per_group=10,
            response_T_scalar_RMS_by_subject_and_block=state['response_scale'].detach().cpu().squeeze(-1).tolist(),
            checkpoint_rule='fixed_final10th_complete_T_pass',no_Dfit_gradient_or_epoch_selection=True,new_UNet_training=False,original_D_resumed=False))
        # Dfit is an exposed development diagnostic, never a gradient or epoch selector.
        fit_queries=U.read(U.checked(spec['fixed_panel']))['Dfit'];held_lookup={(q['subject_id'],q['array_row']):i for i,q in enumerate(held['queries'])}
        diagnostic_indices=torch.as_tensor([held_lookup[(q['subject_id'],q['array_row'])] for q in fit_queries],device='cuda');assert len(diagnostic_indices)==200
        assert all(len({q['concept_id'] for q in fit_queries[lo:lo+10]})==1 for lo in range(0,200,10))
        first_profile=None
        def forward(arm,data,ix,source_indices=None):
            sid=data['subject'][ix];u=data['u'][ix];cm=state['c_mean'][sid];cs=state['c_scale'][sid]
            features=(u-cm)/cs;a=data['atms'][ix]
            r=data['responses'][ix] if arm!='shuffled_responses' else data['responses'][data['permutation'][ix]]
            if arm=='c_only':context=features[:,None].expand(-1,5,-1)
            elif arm=='final_ATMS':context=a[:,None].expand(-1,5,-1)
            else:context=torch.cat([a[:,None],r],1)
            with torch.autocast('cuda',dtype=torch.bfloat16):correction,stats=models[arm](features,context,u)
            # The fixed proxy readout stays FP32, and all its weights stay frozen.
            prediction=((u+correction-cm)/cs)@state['Bc']
            return prediction,correction,stats
        @torch.no_grad()
        def diagnostic(epoch):
            result={}
            for arm,model in models.items():
                model.eval();prediction,correction,stats=forward(arm,held,diagnostic_indices);target=held['target'][diagnostic_indices]
                result[arm]=dict(MSE=float((prediction-target).square().mean()),within10_CE=float(V.local_loss(prediction,target)),
                    mean_used_radius=float(stats['used_radius'].mean()),native_u_correction_norm_mean=float(correction.norm(dim=-1).mean()))
            log(dict(stage='FIXED_DFIT200_LOG_ONLY',epoch=epoch,values=result,gradient_or_epoch_selection=False))
        diagnostic(0);fit_start=time.monotonic();active='FOUR_MATCHED_FULL_T_PASSES'
        for epoch in range(10):
            order=torch.randperm(13230,device='cuda');order_sha=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
            totals={arm:dict(MSE=0.,CE=0.,loss=0.,radius=0.) for arm in V.ARMS};seen=0
            for position in range(0,len(order),100):
                if updates%8==0:check(training=True)
                ix=train['groups'][order[position:position+100]].reshape(-1);target=train['target_unique'][ix%13230]
                for arm,model in models.items():
                    model.train();optimizer=optimizers[arm];optimizer.zero_grad(set_to_none=True)
                    prediction,correction,stats=forward(arm,train,ix)
                    mse=(prediction-target).square().mean();ce=V.local_loss(prediction,target);loss=mse+.1*ce;assert torch.isfinite(loss)
                    loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gradient) and gradient>0
                    if updates==0 and arm==V.ARMS[0]:
                        first_profile=dict(actual_batch_queries=len(ix),groups=len(ix)//10,global_gradient_norm=float(gradient),finite_nonzero_gradient=True,
                            parameters_per_head=params,all_original_models_frozen=True,UNet_forwards=0,GPU_allocated_bytes=torch.cuda.max_memory_allocated())
                        U.atomic(a.output/'first_gradient_profile.json',first_profile)
                    if updates==16 and arm=='paired_responses':
                        value=model.token.weight.grad.norm();assert value>0 and torch.isfinite(value)
                        U.atomic(a.output/'actual_response_branch_gradient.json',dict(update_before_step=16,token_projection_gradient_norm=float(value),finite_nonzero=True))
                    optimizer.step()
                    for key,value in [('MSE',mse),('CE',ce),('loss',loss),('radius',stats['used_radius'].mean())]:totals[arm][key]+=float(value)*len(ix)
                updates+=1;exposures+=len(ix);seen+=len(ix)
                if updates%25==0:log(dict(stage='TRAIN_UPDATE',epoch=epoch+1,updates_per_head=updates,exposures_per_head=exposures,
                    running={arm:{k:v/seen for k,v in d.items()} for arm,d in totals.items()}))
            assert seen==132300
            log(dict(stage='COMPLETE_T_PASS',epoch=epoch+1,updates_per_head=updates,exposures_per_head=exposures,
                all132300_queries_seen_once_per_head=True,group_order_sha256=order_sha,means={arm:{k:v/seen for k,v in d.items()} for arm,d in totals.items()}));diagnostic(epoch+1)
        torch.cuda.synchronize();fit_seconds=time.monotonic()-fit_start;assert updates==1330 and exposures==1323000;check()
        active='FIXED_FINAL10_CHECKPOINT'
        checkpoint=a.output/'fixed_final_heads.pt';torch.save(dict(models={arm:{k:v.detach().cpu() for k,v in model.state_dict().items()} for arm,model in models.items()},
            optimizers={arm:opt.state_dict() for arm,opt in optimizers.items()},state={k:v.detach().cpu() for k,v in state.items()},
            epoch=10,updates_per_head=updates,query_exposures_per_head=exposures,spec_sha256=U.sha(a.spec),CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state()),checkpoint)
        fixed_sha=U.sha(checkpoint);active='ALL8200_COMPLETE_DFIT_FROZEN_FINAL'
        predictions={};conditions={};radii={}
        with torch.no_grad():
            for arm,model in models.items():
                model.eval();parts=[];cond=[];rs=[]
                for lo in range(0,8200,500):
                    check();ix=torch.arange(lo,min(lo+500,8200),device='cuda');prediction,correction,stats=forward(arm,held,ix)
                    parts.append(prediction.cpu());cond.append((held['condition'][ix]+held['weight'][ix]*correction).cpu());rs.append(stats['used_radius'].cpu())
                predictions[arm]=torch.cat(parts);conditions[arm]=torch.cat(cond);radii[arm]=torch.cat(rs)
        target=held['target'].cpu();values={arm:V.OLD.metrics(prediction,target,held['groups']) for arm,prediction in predictions.items()}
        linear={name:V.OLD.metrics(held['linear_predictions'][:,j].cpu(),target,held['groups']) for name,j in [('linear_c',0),('linear_joint',2)]}
        subjects=[q['subject_id'] for q in held['queries']];concepts=[q['concept_id'] for q in held['queries']]
        result=V.readout(values,linear,subjects,concepts);U.atomic(a.output/'readout.json',result)
        U.save_array(a.output/'Dfit_predictions.npy',torch.stack([predictions[a] for a in V.ARMS],1).numpy())
        U.save_array(a.output/'Dfit_native_conditions.npy',torch.stack([conditions[a] for a in V.ARMS],1).numpy())
        U.save_array(a.output/'Dfit_standardized_targets.npy',target.numpy());U.save_array(a.output/'Dfit_used_radius.npy',torch.stack([radii[a] for a in V.ARMS],1).numpy())
        U.atomic(a.output/'queries_and_controls.json',dict(queries=held['queries'],groups=held['groups'],permutation=held['permutation'].cpu().tolist()))
        fields=['subject_id','array_row','concept_id','stimulus_id','arm']+list(values[V.ARMS[0]])
        with (a.output/'per_query.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i,q in enumerate(held['queries']):
                for arm,v in (values|linear).items():writer.writerow(q|dict(arm=arm)|{k:float(x[i]) for k,x in v.items()})
        assert U.sha(checkpoint)==fixed_sha
        for item in spec['source_pins']:U.checked(item)
        check();complete=dict(status='D16_FOUR_MATCHED_HEADS_FULL_T10_AND_ALL8200_DFIT_COMPLETE',schema=V.SCHEMA,spec=U.ref(a.spec),source=U.ref(__file__),
            response_export=spec['response_export'],D15_probe=spec['D15_probe'],model=U.ref(checkpoint),readout=U.ref(a.output/'readout.json'),
            predictions=U.ref(a.output/'Dfit_predictions.npy'),conditions=U.ref(a.output/'Dfit_native_conditions.npy'),targets=U.ref(a.output/'Dfit_standardized_targets.npy'),
            used_radius=U.ref(a.output/'Dfit_used_radius.npy'),queries_and_controls=U.ref(a.output/'queries_and_controls.json'),per_query=U.ref(a.output/'per_query.csv'),
            heads=4,parameters_per_head=params,epochs_per_head=10,updates_per_head=updates,query_exposures_per_head=exposures,all_real_OOF_T_queries=132300,
            Dfit_queries=8200,Dfit_concepts=82,subjects=10,training_body_seconds=fit_seconds,prediction_gate_passed=result['prediction_gate_passed'],
            no_Dfit_optimizer_or_checkpoint_selection=True,Dcheck_effects_read=False,new_images=0,UNet_forwards=0,original_D_resumed=False,
            no_automatic_retry_or_hyperparameter_search=True,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,seconds=time.monotonic()-start,
            actual_image_gain_established=False,independent_confirmation=False,SOTA_established=False)
        U.atomic(a.output/'completion.json',complete);log(dict(stage='COMPLETE',decision=result['decision']));return 0
    except BaseException as error:
        if models is not None:torch.save(dict(models={arm:model.state_dict() for arm,model in models.items()},optimizers={arm:opt.state_dict() for arm,opt in optimizers.items()},updates_per_head=updates,exposures_per_head=exposures,CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state()),a.output/'incomplete_heads.pt')
        U.atomic(a.output/'failure.json',dict(status='D16_FIT_INCOMPLETE_PRESERVED_NO_AUTOMATIC_RETRY',active=active,updates_per_head=updates,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise


if __name__=='__main__':raise SystemExit(main())
