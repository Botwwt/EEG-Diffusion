"""D13 three matched noisy joint/product heads; fixed10passes, no new images."""
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
import torch.nn.functional as F
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_noisy_EEG_evidence_20261027_v1 as V
import brainhive_conditional_flow_20261027_v1 as FLOW


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','native-registration','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True); p.add_argument('--max-seconds',type=float,required=True)
    args=p.parse_args(); start=time.monotonic(); assert args.max_seconds==360
    assert not args.output.exists(); args.output.mkdir(parents=True)
    stop=[False]; signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    active='INPUTS'; models={}; optimizers={}; updates=0; exposures=0
    def log(x):
        x=x|dict(seconds=time.monotonic()-start)
        with (args.output/'training_log.jsonl').open('a') as f: f.write(json.dumps(x,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(x,allow_nan=False),flush=True)
    try:
        spec=U.read(args.spec); assert spec['schema']==V.SCHEMA and spec['epochs']==10
        for source in spec['source_pins']: U.checked(source)
        math=U.read(U.checked(spec['CPU_checks'])); assert math['status']=='D13_ALL11_CPU_MATH_AND_DECISION_CHECKS_PASSED'
        d12=U.read(U.checked(spec['D12_fit'])); d12lock=U.read(U.checked(d12['lock']))
        inputs=U.read(U.checked(d12['D09_inputs'])); panel=U.read(U.checked(d12['panel']))
        raw_inputs=U.read(U.checked(spec['raw_T_inputs'])); index=[json.loads(s) for s in Path(raw_inputs['prediction_index']).read_text().splitlines()]
        assert U.sha(raw_inputs['prediction_index'])==raw_inputs['prediction_index_sha256']
        import brainhive_paired_study_contract_20261025_v1 as C
        config=C.read(args.native_registration); C.validate(config); C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch
        document=dict(gpu_job_name=args.gpu_job_name,maximum_seconds=360,maximum_output_bytes=512<<20,
            maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,
            resource_authorization=config['resource_authorization'])
        wrapper=D.verify_wrapper(document); budget=D.Budget(document)
        def check(training=False):
            assert not stop[0] and time.monotonic()-start<(230 if training else 340),'D13 finite fixed work budget'
            budget.check()
        torch.set_num_threads(2); torch.manual_seed(1729); np.random.seed(1729); torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True); C.resources(config,training_start=True)
        from brainhive_component_data_v1 import load_training
        active='BOUND_ALL10_OOF_C_AND_RAW_T_R4'
        training=load_training('cuda'); n=len(training['condition']); assert n==132300
        assert training['subjects']==[f'sub-{j:02d}' for j in range(1,11)]
        rows=training['array_rows']; assert rows==raw_inputs['source_rows']==[q['array_row'] for q in index]
        c=training['condition']; subject=torch.arange(10,device='cuda').repeat_interleave(13230)
        raw=torch.empty((n,4250),device='cuda'); raw_mean=[]; raw_std=[]
        for j,name in enumerate(training['subjects']):
            check(); item=raw_inputs['subjects'][name]; path=U.checked(dict(path=item['path'],sha256=item['sha256']))
            array=np.load(path,mmap_mode='r',allow_pickle=False); assert array.shape==(13230,17,250)
            value=torch.from_numpy(np.array(array,copy=True).reshape(13230,4250)).cuda()
            mean=value.mean(0); std=value.std(0,unbiased=False).clamp_min(1e-4)
            raw[j*13230:(j+1)*13230]=(value-mean)/std; raw_mean.append(mean); raw_std.append(std); del value,array
        raw_mean=torch.stack(raw_mean); raw_std=torch.stack(raw_std)
        old=torch.load(U.checked(d12['model']),weights_only=True,map_location='cpu',mmap=True)
        mean=old['mean'].cuda(); scale=old['scale'].cuda(); fw=old['F_weight'].cuda(); fb=old['F_bias'].cuda()
        u=(training['target']-fb)/fw; assert torch.allclose(u.mean(0),mean,atol=2e-6,rtol=2e-6)
        assert torch.allclose((u-mean).square().mean().sqrt(),scale,atol=2e-6,rtol=2e-6)
        y=(u-mean)/scale; target_index=torch.arange(13230,device='cuda').repeat(10)
        bindings_path=Path('/root/brainhive_reproduction_20261020_v1/data/actual_input_bindings.json'); bindings=U.read(bindings_path)
        source=bindings['subjects']['sub-01']; assert source['train_repeats']==4
        raw_path=U.checked(dict(path=source['train_path'],sha256=source['train_sha256']))
        released=torch.load(raw_path,weights_only=True,map_location='cpu',mmap=True); channels=bindings['selected_indices']
        assert len(channels)==17 and released.shape==(16540,63,250) and channels==raw_inputs['selected_indices']
        selected=list(range(10))+list(range(13220,13230)); rrows=[rows[i] for i in selected]
        original=released[rrows][:,channels,:].numpy(); derived=np.load(raw_inputs['subjects']['sub-01']['path'],mmap_mode='r')[selected]
        assert np.array_equal(original,derived)
        cache=torch.load(U.checked(inputs['native_cache']),weights_only=True,map_location='cpu',mmap=True)
        lookup={int(row):i for i,row in enumerate(cache['array_rows'])}
        contract=U.read('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json')
        all_F=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
        def held_data(split):
            qs=panel[split]; ids=[q['array_row'] for q in qs]; positions=[lookup[i] for i in ids]
            assert cache['array_rows'][positions].tolist()==ids
            h_c=cache['condition'][positions].clone().cuda()
            h_b=released[ids][:,channels,:].reshape(len(ids),4250).contiguous().cuda()
            h_b=(h_b-raw_mean[0])/raw_std[0]
            h_y=((all_F[ids].clone().cuda()-fb)/fw-mean)/scale
            z,seeds=FLOW.query_noise(qs); z=z.cuda()
            seed=1729+(0 if split=='Dfit' else 1); gen=np.random.default_rng(seed)
            product=gen.integers(0,len(qs),len(qs)); null=gen.integers(0,len(qs),len(qs))
            return dict(queries=qs,c=h_c,b=h_b,y=h_y,z=z,seeds=seeds,product=product,null=null,
                subjects=torch.zeros(len(qs),device='cuda',dtype=torch.long))
        fit_data=held_data('Dfit')  # no Dcheck F/raw values are taken until the checkpoint is final.
        @torch.no_grad()
        def classify(data,save_prefix=None):
            N=len(data['queries']); pieces={arm:[] for arm in V.ARMS}
            k=torch.as_tensor(data['product'],device='cuda'); m=torch.as_tensor(data['null'],device='cuda')
            cc=torch.cat([data['c'],data['c'][k]]); bb=torch.cat([data['b'],data['b'][k]])
            nullbb=torch.cat([data['b'][m]]*2); ss=torch.cat([data['subjects']]*2)
            for value in V.TIMES:
                x=(1-value)*data['z']+value*data['y']; xx=torch.cat([x,x]); tt=torch.full((2*N,),value,device='cuda')
                for arm in V.ARMS:
                    models[arm].eval(); out=models[arm](xx,tt,cc,nullbb if arm=='shuffled_raw_train' else bb,ss,arm).float()
                    pieces[arm].append(torch.stack([out[:N],out[N:]],-1).cpu().numpy())
            result={arm:np.stack(value,1) for arm,value in pieces.items()}
            if save_prefix:
                for arm,value in result.items(): U.save_array(args.output/(save_prefix+'_'+arm+'_logits.npy'),value)
            return result
        base=V.Evidence().cuda(); models={arm:(base if i==0 else copy.deepcopy(base)) for i,arm in enumerate(V.ARMS)}
        parameters=sum(p.numel() for p in base.parameters()); assert parameters<3_500_000
        assert all(torch.equal(a,b) for arm in V.ARMS[1:] for a,b in zip(models['c_only'].state_dict().values(),models[arm].state_dict().values()))
        init_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in base.state_dict().values())).hexdigest()
        optimizers={arm:torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.99),weight_decay=.01) for arm,model in models.items()}
        U.atomic(args.output/'provenance.json',dict(schema=V.SCHEMA,spec=U.ref(args.spec),source=U.ref(__file__),wrapper=wrapper,
            raw_T_inputs=spec['raw_T_inputs'],old_D12_normalizer_source=d12['model'],input_contract_sha256=training['input_contract_sha256'],
            actual_OOF_queries=132300,training_subjects=10,raw_mean4_channels=17,raw_feature_dimensions=4250,
            raw_standardization_only_T=True,raw_sub01_T_rows_replayed_exactly=20,raw_event_pairing_independently_rebuilt=False,
            original_preprocessing_MVNN_training_only_established=False,parameters_per_arm=parameters,arms=list(V.ARMS),
            identical_architecture_and_initialization=True,initial_state_sha256=init_hash,
            product_brain_pair_c_and_b_always_same_source=True,product_with_replacement_self_pairs_permitted=True,
            Dfit_logging_only=True,Dcheck_optimization_or_checkpoint_selection=False,old_B_VAE_density_ratio_experiment=False,
            new_image_generation=0,UNet_loaded_or_trained=False,formal_D_resumed=False))
        def held_log(epoch):
            data=classify(fit_data)
            log(dict(stage='FIXED_DFIT_NOISY_LOGLOSS',epoch=epoch,
                metrics={arm:V.metrics(v[:,:,0],v[:,:,1]) for arm,v in data.items()},no_fitting_or_checkpoint_choice=True))
        held_log(0); active='THREE_MATCHED_HEADS_FULL_T_10_PASSES'; train_start=time.monotonic(); profile={}
        for epoch in range(10):
            for model in models.values(): model.train()
            order=torch.randperm(n,device='cuda'); totals={a:0. for a in V.ARMS}; seen=0
            for position in range(0,n,1024):
                if updates%8==0: check(training=True)
                ix=order[position:position+1024]; count=len(ix); s=subject[ix]
                negative=V.product_indices(s); null=V.product_indices(s)
                target=y[target_index[ix]]; noise=torch.randn_like(target); t=torch.rand(count,device='cuda')
                x=(1-t[:,None])*noise+t[:,None]*target
                xx=torch.cat([x,x]); tt=torch.cat([t,t]); cc=torch.cat([c[ix],c[negative]])
                bb=torch.cat([raw[ix],raw[negative]]); nullbb=torch.cat([raw[null]]*2); ss=torch.cat([s,s])
                label=torch.cat([torch.ones(count,device='cuda'),torch.zeros(count,device='cuda')])
                current={}
                for arm in V.ARMS:
                    optimizer=optimizers[arm]; optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=models[arm](xx,tt,cc,nullbb if arm=='shuffled_raw_train' else bb,ss,arm).float()
                        loss=F.binary_cross_entropy_with_logits(logits,label)
                    assert torch.isfinite(loss); loss.backward(); gradient=torch.nn.utils.clip_grad_norm_(models[arm].parameters(),1.)
                    assert torch.isfinite(gradient) and gradient>0; optimizer.step()
                    current[arm]=float(loss); totals[arm]+=float(loss)*count
                    if updates==0: profile[arm]=dict(finite_nonzero_gradient=True,gradient_norm=float(gradient),batch_queries=count,labelled_rows=2*count)
                updates+=1; exposures+=count; seen+=count
                if updates%25==0: log(dict(stage='TRAIN_UPDATE',epoch=epoch+1,updates_per_head=updates,main_query_exposures_per_head=exposures,log_loss=current))
            assert seen==n
            log(dict(stage='COMPLETE_T_PASS',epoch=epoch+1,updates_per_head=updates,query_exposures_per_head=exposures,
                train_log_loss={a:totals[a]/seen for a in V.ARMS},all132300_queries_seen_once_per_head=True,
                order_sha256=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()))
            held_log(epoch+1)
        torch.cuda.synchronize(); train_seconds=time.monotonic()-train_start
        assert updates==1300 and exposures==1323000; check()
        U.atomic(args.output/'first_actual_gradients.json',profile)
        active='FREEZE_ALL_THREE_FINAL_HEADS'
        checkpoint=args.output/'final_heads.pt'
        torch.save(dict(models={a:{k:v.detach().cpu() for k,v in m.state_dict().items()} for a,m in models.items()},
            optimizers={a:o.state_dict() for a,o in optimizers.items()},raw_T_mean=raw_mean.cpu(),raw_T_std=raw_std.cpu(),
            F_mean=mean.cpu(),F_scale=scale.cpu(),F_weight=fw.cpu(),F_bias=fb.cpu(),epoch=10,updates_per_head=updates,
            query_exposures_per_head=exposures,CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state(),spec_sha256=U.sha(args.spec)),checkpoint)
        fixed_hash=U.sha(checkpoint)
        for model in models.values(): model.eval().requires_grad_(False)
        active='FROZEN_DFIT_AND_DCHECK_FULL_DIAGNOSTICS'
        held={'Dfit':fit_data,'Dcheck':held_data('Dcheck')}; all_logits={}; retrieval={}; maps={}; interventions={}
        for split,data in held.items():
            check(); N=len(data['queries']); all_logits[split]=classify(data,split)
            qs=data['queries']; maps[split]=dict(query_ids=[q['query_id'] for q in qs],product_brain_indices=data['product'].tolist(),
                independent_raw_indices=data['null'].tolist(),conditional_noise_seeds=data['seeds'],
                product_c_and_raw_share_source=True,product_may_contain_true_or_sameconcept_pair=True)
            candidates=np.asarray([list(range(i//10*10,i//10*10+10)) for i in range(N)])
            j=torch.as_tensor(candidates.reshape(-1),device='cuda'); im=torch.arange(N,device='cuda').repeat_interleave(10)
            k=torch.as_tensor(data['null'],device='cuda'); score={a:np.full((N,4,10),np.nan,np.float32) for a in V.ARMS}
            with torch.no_grad():
                for ti,t in enumerate(V.TIMES):
                    x=(1-t)*data['z']+t*data['y']
                    for lo in range(0,len(j),512):
                        jj=j[lo:lo+512]; ii=im[lo:lo+512]; tt=torch.full((len(jj),),t,device='cuda')
                        for arm in V.ARMS:
                            b=data['b'][k[jj]] if arm=='shuffled_raw_train' else data['b'][jj]
                            value=models[arm](x[ii],tt,data['c'][jj],b,data['subjects'][jj],arm).cpu().numpy()
                            positions=np.arange(lo,lo+len(jj)); score[arm][positions//10,ti,positions%10]=value
            retrieval[split]={}
            for arm,value in score.items():
                assert np.isfinite(value).all()
                U.save_array(args.output/(split+'_'+arm+'_sameconcept_scores.npy'),value)
                average=value.mean(1); positive=average[np.arange(N),np.arange(N)%10]
                retrieval[split][arm]=dict(top1=float((average.argmax(1)==np.arange(N)%10).mean()),
                    correct_minus_other9_logit=float((positive-(average.sum(1)-positive)/9).mean()),queries=N,
                    readout='held visual-state to EEG matching diagnosis; not a reconstruction input or image metric')
            same=np.arange(N)//10*10+(np.arange(N)+1)%10; group=U.derangement(N//10)
            other=(group[:,None]*10+np.arange(10)[None]).reshape(-1)
            counter={}
            with torch.no_grad():
                for name,perm in [('sameconcept_raw',same),('otherconcept_raw',other)]:
                    table=[]
                    for t in V.TIMES:
                        x=(1-t)*data['z']+t*data['y']; tt=torch.full((N,),t,device='cuda')
                        correct=models['raw_EEG'](x,tt,data['c'],data['b'],data['subjects'],'raw_EEG')
                        mixed=models['raw_EEG'](x,tt,data['c'],data['b'][torch.as_tensor(perm,device='cuda')],data['subjects'],'raw_EEG')
                        table.append((correct-mixed).cpu().numpy())
                    table=np.stack(table,1); U.save_array(args.output/(split+'_'+name+'_intervention.npy'),table)
                    counter[name]=V.interval(table.mean(1),qs)
            interventions[split]=dict(correct_c_raw_shuffle_sensitivity=counter,
                mixed_c_b_may_be_off_joint_support=True,probability_calibration_or_causal_increment_claim=False)
        U.atomic(args.output/'held_pairing_maps.json',maps)
        readout=V.readout(all_logits,retrieval,{k:v['queries'] for k,v in held.items()})
        readout['raw_counterfactual_sensitivity']=interventions
        U.atomic(args.output/'readout.json',readout)
        # Capture the existing D12 sampler without changing its numerical operations.
        active='OLD_D12_TRAJECTORY_REPLAY_AND_FROZEN_INPUT_GRADIENTS'; data=held['Dcheck']; prior=FLOW.Flow().cuda().eval().requires_grad_(False)
        prior.load_state_dict(old['model'],strict=True); states={}
        class Observed:
            def condition(self,c,s): return prior.condition(c,s)
            def velocity(self,x,t,context):
                k=round(float(t[0])*32)
                if k in (4,12,20,28): states[k]=x.detach().clone()
                return prior.velocity(x,t,context)
        z,_=FLOW.query_noise(data['queries']); z=z.cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            replay,_=FLOW.sample(Observed(),z,data['c'],data['subjects'],mean,scale,fw,fb)
        bank=np.load(U.checked(d12lock['conditions']),allow_pickle=False)[:,0]
        rel=(replay.cpu()-torch.from_numpy(bank)).norm(dim=1)/torch.from_numpy(bank).norm(dim=1)
        exact=np.array_equal(replay.cpu().numpy(),bank); assert float(rel.max())<1e-5
        gradient_summary={}; target_direction=data['y']-data['y'].reshape(40,10,1024).mean(1).repeat_interleave(10,0)
        for domain in ('real_forward','old_D12_generated_trajectory'):
            check(); gradients=[]; xs=[]
            for ti,t in enumerate(V.TIMES):
                x0=(1-t)*data['z']+t*data['y'] if domain=='real_forward' else states[(4,12,20,28)[ti]]
                pieces=[]
                for lo in range(0,400,64):
                    tt=torch.full((len(x0[lo:lo+64]),),t,device='cuda'); arm_grads=[]
                    for arm in V.ARMS:
                        x=x0[lo:lo+64].detach().clone().requires_grad_(True)
                        k=torch.as_tensor(data['null'][lo:lo+64],device='cuda')
                        b=data['b'][k] if arm=='shuffled_raw_train' else data['b'][lo:lo+64]
                        value=models[arm](x,tt,data['c'][lo:lo+64],b,data['subjects'][lo:lo+64],arm)
                        g=torch.autograd.grad(value.sum(),x)[0]; assert torch.isfinite(g).all(); arm_grads.append(g.detach())
                    pieces.append(torch.stack(arm_grads,1))
                gradients.append(torch.cat(pieces).cpu().numpy()); xs.append(x0.cpu().numpy())
            gradients=np.stack(gradients,1); xs=np.stack(xs,1)
            U.save_array(args.output/(domain+'_gradients.npy'),gradients); U.save_array(args.output/(domain+'_states.npy'),xs)
            inc=torch.from_numpy(gradients[:,:,1]-gradients[:,:,0]).cuda(); direction=target_direction[:,None].expand_as(inc)
            norm=inc.norm(dim=-1); cosine=F.cosine_similarity(inc,direction,dim=-1)
            gradient_summary[domain]=dict(increment_gradient_norm_mean=float(norm.mean()),increment_gradient_norm_max=float(norm.max()),
                increment_to_target_withinconcept_direction_cosine_mean=float(cosine.mean()),all_finite=True,
                zero_gradient_fraction=float((norm==0).float().mean()),current_state_graph_retained=True,
                actual_sampling_distribution_calibration_established=False,gradient_quality_or_image_gain_established=False)
        U.atomic(args.output/'actual_state_gradient_diagnostic.json',dict(status='D13_ALL1600_STATES_BOTH_DOMAINS_ALL3_HEAD_X_GRADIENTS_COMPLETE',
            D12_replay_bitwise_equal=bool(exact),D12_max_relative_L2=float(rel.max()),source_D12_model=d12['model'],
            time_grid=list(V.TIMES),domains=gradient_summary,no_Turbo_score_or_sampling_modified=True,
            target_direction_used_for_postfit_diagnosis_only=True))
        fields=['query_id','partition','concept_id','arm','log_loss','Brier','sameconcept_top1','sameconcept_logit_margin']
        with (args.output/'per_query.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for split,data in held.items():
                for arm in V.ARMS:
                    a=V.arrays(all_logits[split][arm][:,:,0],all_logits[split][arm][:,:,1])
                    scores=np.load(args.output/(split+'_'+arm+'_sameconcept_scores.npy')).mean(1)
                    for i,q in enumerate(data['queries']):
                        positive=scores[i,i%10]
                        writer.writerow(dict(query_id=q['query_id'],partition=split,concept_id=q['concept_id'],arm=arm,
                            log_loss=float(a['log_loss'][i].mean()),Brier=float(a['Brier'][i].mean()),
                            sameconcept_top1=int(scores[i].argmax()==i%10),sameconcept_logit_margin=float(positive-(scores[i].sum()-positive)/9)))
        assert U.sha(checkpoint)==fixed_hash
        for item in spec['source_pins']: U.checked(item)
        check(); assert sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())<512<<20
        result=dict(status='COMPLETE_D13_MATCHED_NOISY_EEG_EVIDENCE_DIAGNOSTIC',schema=V.SCHEMA,spec=U.ref(args.spec),source=U.ref(__file__),
            final_heads=U.ref(checkpoint),readout=U.ref(args.output/'readout.json'),actual_state_gradients=U.ref(args.output/'actual_state_gradient_diagnostic.json'),
            pairing_maps=U.ref(args.output/'held_pairing_maps.json'),parameters_per_arm=parameters,arms=list(V.ARMS),
            full_passes_per_head=10,updates_per_head=updates,main_query_exposures_per_head=exposures,
            labeled_positive_and_product_rows_per_head=2*exposures,training_body_seconds=train_seconds,
            Dfit_queries=200,Dcheck_queries=400,diagnostic_only=True,held_diagnostic_uses_target_F=True,
            held_F_or_raw_EEG_used_in_optimizer=False,final_checkpoint_selected_by_held_metrics=False,
            actual_image_gain_established=False,new_images=0,UNet_evaluations=0,old_B_study_completed=False,
            automatic_image_admission=False,formal_D_resumed=False,no_automatic_retry_or_hyperparameter_sweep=True,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,seconds=time.monotonic()-start)
        U.atomic(args.output/'completion.json',result); log(dict(stage='COMPLETE',decision=readout['decision'])); return 0
    except BaseException as error:
        if models:
            torch.save(dict(models={a:m.state_dict() for a,m in models.items()},optimizers={a:o.state_dict() for a,o in optimizers.items()},
                updates_per_head=updates,query_exposures_per_head=exposures,CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state()),args.output/'incomplete_heads.pt')
        U.atomic(args.output/'failure.json',dict(status='D13_INCOMPLETE_PRESERVED_NO_AUTOMATIC_RETRY',active=active,error=repr(error),
            traceback=traceback.format_exc(),updates_per_head=updates,query_exposures_per_head=exposures,seconds=time.monotonic()-start))
        raise


if __name__=='__main__': raise SystemExit(main())
