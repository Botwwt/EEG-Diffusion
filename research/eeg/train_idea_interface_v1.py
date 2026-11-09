"""Matched complete-data native projector training for all five required arms.

Prepared executable, not permission to bypass a real per-arm lifecycle budget
or the Conditional fidelity prerequisites. No formal images or V/test reads.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import time

from profile_standard_generator import load_pipeline,torch,ROOT
from diffusers import DDPMScheduler
import torch.nn.functional as F
from native_error_channel import NativeErrorChannel,stream_seed

ARMS=['Clean','Iso','Moment','Conditional','RealPaired']


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def load(path):return json.loads(Path(path).read_text())


def write(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def save(path,value):
    temporary=path.with_name(path.name+'.tmp');torch.save(value,temporary);os.replace(temporary,path)


def tensor_sha(value):return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ['recipe','cache-config','validation-output','budget-receipt','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();started=time.monotonic();torch.set_num_threads(2)
    recipe_path=Path(args.recipe);recipe=load(recipe_path);arm=args.arm
    assert recipe['version']=='idea_interface_training_recipe_v1' and recipe['seed']==1729
    assert recipe['arms']==ARMS and recipe['train_passes']==1 and recipe['training_queries_per_pass']==132300
    for source,digest in recipe['source_sha256'].items():assert sha(ROOT/source)==digest,source
    gate_path=Path(args.budget_receipt);gate=load(gate_path)
    assert gate['arm']==arm and gate['arm_lifecycle_funded'] and gate['recipe_sha256']==sha(recipe_path)
    assert gate['formal_image_scope_queries']==16600 and gate['total_GPU_hour_cap']==40
    assert not gate['scientific_goal_achieved'] and gate['required_all_five_arms_preserved']
    parent=Path('/proc')/str(os.getppid())/'cmdline'
    assert parent.exists() and str(ROOT/'run_gpu_job.py').encode() in parent.read_bytes()
    cache_path=Path(args.cache_config);cache=load(cache_path);cache_root=Path(cache['condition_cache_output_root'])
    completion=load(cache_root/'completion.json');assert completion['status']=='COMPLETED_ALL10_CONCEPT_OOF_T_AND_MATCHED_D_CHAINS'
    assert completion['config_sha256']==sha(cache_path) and completion['T_OOF_queries']==132300 and completion['subjects']==10
    assert cache['condition_chain']=='REF_REG' and cache['seed']==1729
    assert gate['cache_config_sha256']==sha(cache_path)
    protocol=load(ROOT/'configs/split_manifest.json');assert sha(ROOT/'configs/split_manifest.json')==cache['split_sha256']
    subjects=protocol['subjects'];t_rows=protocol['image_rows']['T'];assert len(subjects)==10 and len(t_rows)==13230
    bank=torch.load(cache_root/'C_I_T.pt',map_location='cpu',weights_only=True)
    assert torch.equal(bank['array_rows'],torch.tensor(t_rows)) and bank['partition']=='T'
    assert len(set(bank['stimulus_ids']))==13230 and bank['C_I'].shape==(13230,1024)
    receipts={x['path']:x for x in completion['reports']};paired=[];source_proofs=[]
    for subject in subjects:
        path=cache_root/subject/'T_oof.pt';assert sha(path)==receipts[str(path)]['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        assert data['query_ids']==[f'{subject}/{stim}/R4' for stim in bank['stimulus_ids']]
        assert torch.equal(data['array_rows'],bank['array_rows']) and data['C_I_bank_sha256']==sha(cache_root/'C_I_T.pt')
        assert data['concept_ids']==bank['concept_ids'] and set(data['teacher_roles'])=={'oof0','oof1'}
        expected=[f'oof{protocol["oof_fold_by_concept"][concept]}' for concept in bank['concept_ids']]
        assert data['teacher_roles']==expected and data['C_E'].shape==(13230,1024) and torch.isfinite(data['C_E']).all()
        paired.append(data['C_E']);source_proofs.append(dict(subject=subject,path=str(path),sha256=sha(path)))
    paired=torch.stack(paired)
    validation=Path(args.validation_output);channels=[];neighbors=None
    validation_summary=load(validation/'summary.json') if (validation/'summary.json').exists() else None
    if arm=='Conditional':
        assert validation_summary is not None
        assert validation_summary['gate']['native_energy_CI_upper_below_zero_vs_Iso_and_Moment']
        assert validation_summary['gate']['projected_energy_CI_upper_below_zero_vs_Iso_and_Moment']
        response_path=Path(gate['UNet_response_summary_path']);response=load(response_path)
        assert sha(response_path)==gate['UNet_response_summary_sha256'] and response['gate_passed']
        assert response['queries']==8300 and response['subjects']==10
    if arm in ['Iso','Moment','Conditional']:
        assert validation_summary is not None and validation_summary['subjects']==10
        provenance=load(validation/'input_provenance.json')
        assert provenance['cache_config_sha256']==sha(cache_path) and provenance['cache_completion_sha256']==sha(cache_root/'completion.json')
        for index,subject in enumerate(subjects):
            fitted=torch.load(validation/subject/'error_channel.pt',map_location='cpu',weights_only=True)
            assert fitted['actual_scientific_fit'] and fitted['T_source_sha256']==source_proofs[index]['sha256']
            assert fitted['plan_sha256']==validation_summary['plan_sha256']
            channels.append(NativeErrorChannel(fitted['state']))
        if arm=='Conditional':
            neighbors=channels[0].nearest(bank['C_I'],bank['concept_ids'])
            for channel in channels[1:]:
                assert torch.equal(channel.state['pca']['basis'],channels[0].state['pca']['basis'])
                assert torch.equal(channel.keys,channels[0].keys)
    latent_root=Path(recipe['latent_bank_root']);latent_summary=load(latent_root/'completion.json')
    assert sha(latent_root/'completion.json')==gate['latent_bank_completion_sha256']
    assert latent_summary['status']=='COMPLETED_ALL13230_T_IMAGE_LATENTS' and latent_summary['recipe_sha256']==sha(recipe_path)
    assert latent_summary['array_rows']==t_rows and not latent_summary['V_or_D_or_official_test_images_read']
    latents=[]
    for part in latent_summary['parts']:
        assert sha(part['path'])==part['sha256'];x=torch.load(part['path'],map_location='cpu',weights_only=True)
        assert x['array_rows']==t_rows[part['start']:part['stop']] and x['recipe_sha256']==sha(recipe_path)
        assert x['latents'].dtype==torch.float32 and torch.isfinite(x['latents']).all();latents.append(x['latents'])
    target_bank=torch.cat(latents).cuda();del latents
    assert target_bank.shape==(13230,4,64,64)
    output=Path(args.output)
    if args.resume:assert output.exists() and (output/'resume.pt').exists()
    else:assert not output.exists();output.mkdir(parents=True)
    assert not (output/'completion.json').exists()
    lock=dict(arm=arm,recipe_sha256=sha(recipe_path),cache_config_sha256=sha(cache_path),
        latent_bank_completion_sha256=sha(latent_root/'completion.json'),training_source_proofs=source_proofs,
        input_contract='Raw native pooled1024; RealPaired uses RAW genuine concept-OOF prediction, no calibration of paired input',
        trainable_scope='Exactly original native ImageProjection, frozen IP attention and SDXL UNet backbone',
        formal_image_scope_queries=16600,one_seed=1729)
    if (output/'input_provenance.json').exists():assert load(output/'input_provenance.json')==lock
    else:write(output/'input_provenance.json',lock)
    write(output/f'launch_budget_{time.time_ns()}.json',dict(budget_receipt_path=str(gate_path),
        budget_receipt_sha256=sha(gate_path),resume=args.resume,recipe_sha256=sha(recipe_path)))
    pipe,infos,model_receipt,prompts=load_pipeline();prompt,negative,pool,negative_pool=prompts
    selected={}
    for name,parameter in pipe.unet.named_parameters():
        if name.startswith('encoder_hid_proj.'):
            parameter.requires_grad_(True);parameter.data=parameter.data.float();selected[name]=parameter
    assert len(selected)==4 and sum(p.numel() for p in selected.values())==8400896
    initial={name:p.detach().float().cpu().clone() for name,p in selected.items()}
    initialization_sha={name:tensor_sha(value) for name,value in initial.items()}
    pipe.unet.train();pipe.unet.enable_gradient_checkpointing()
    assert float(pipe.unet.config.dropout)==0.,'Frozen generator stochastic dropout would need its own contract'
    optimizer=torch.optim.AdamW(selected.values(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    scheduler=DDPMScheduler.from_config(pipe.scheduler.config);assert scheduler.config.prediction_type=='epsilon'
    time_ids=pipe._get_add_time_ids((512,512),(0,0),(512,512),dtype=prompt.dtype,text_encoder_projection_dim=pool.shape[-1]).cuda()
    order=torch.randperm(132300,generator=torch.Generator().manual_seed(stream_seed(1729,'idea-interface-order','pass0')))
    batch=recipe['batch_size'];updates=0;cursor=0;coverage=torch.zeros(10,13230,dtype=torch.bool)
    generator=torch.Generator(device='cuda').manual_seed(stream_seed(1729,'idea-interface-latent-noise-times'))
    cumulative_seconds=0.;epoch_loss_sum=0.;seen=0
    if args.resume:
        saved=torch.load(output/'resume.pt',map_location='cpu',weights_only=False)
        assert saved['input_provenance_sha256']==sha(output/'input_provenance.json')
        for name,p in selected.items():p.data.copy_(saved['projection'][name].cuda())
        optimizer.load_state_dict(saved['optimizer']);cursor=saved['cursor'];updates=saved['updates'];coverage=saved['coverage']
        generator.set_state(saved['generator_state']);cumulative_seconds=saved['seconds'];epoch_loss_sum=saved['loss_sum'];seen=saved['seen']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def snapshot():
        save(output/'resume.pt',dict(projection={n:p.detach().cpu() for n,p in selected.items()},optimizer=optimizer.state_dict(),
            cursor=cursor,updates=updates,coverage=coverage,generator_state=generator.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            input_provenance_sha256=sha(output/'input_provenance.json'),seconds=cumulative_seconds+time.monotonic()-started,
            loss_sum=epoch_loss_sum,seen=seen))
    last_snapshot=time.monotonic()
    log=(output/'training.jsonl').open('a',buffering=1);torch.cuda.reset_peak_memory_stats()
    for start in range(cursor,132300,batch):
        if ((ROOT/'STOP_IDEA_INTERFACE_TRAINING').exists()
            or cumulative_seconds+time.monotonic()-started>recipe['total_training_seconds_cap_per_arm']-60
            or time.monotonic()-started>gate['this_launch_training_cap_seconds']-60):
            snapshot();write(output/'progress.json',dict(status='RESOURCE_STOPPED_RESUMABLE_NOT_SCIENTIFIC_FAILURE',updates=updates,seen=seen));raise RuntimeError('Preserved at complete minibatch boundary')
        indices=order[start:start+batch];subject_indices=indices//13230;positions=indices%13230;n=len(indices)
        condition=bank['C_I'][positions].clone()
        if arm=='RealPaired':condition=paired[subject_indices,positions].clone()
        elif arm!='Clean':
            for subject_index,subject in enumerate(subjects):
                use=torch.where(subject_indices==subject_index)[0]
                if not len(use):continue
                pos=positions[use];concepts=[bank['concept_ids'][i] for i in pos.tolist()]
                # Iso/Moment share standard normal draws; conditional resampling
                # has a separate source stream. Latent noise never uses these streams.
                namespace='common-normal' if arm in ['Iso','Moment'] else 'conditional-source'
                streams=[stream_seed(1729,'idea-adapter-condition','pass0',subject,bank['stimulus_ids'][i],namespace) for i in pos.tolist()]
                value,_=channels[subject_index].sample(arm,bank['C_I'][pos],concepts,streams,
                    neighbors[pos] if neighbors is not None else None);condition[use]=value
        assert torch.isfinite(condition).all()
        dropped=torch.tensor([stream_seed(1729,'idea-interface-drop',int(i))%1000 < 100 for i in indices.tolist()])
        condition[dropped]=0
        latent=target_bank[positions.cuda()]
        noise=torch.randn(latent.shape,device='cuda',dtype=torch.float32,generator=generator)
        t=torch.randint(0,1000,(n,),device='cuda',generator=generator)
        noisy=scheduler.add_noise(latent,noise,t)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            predicted=pipe.unet(noisy.to(torch.bfloat16),t,encoder_hidden_states=prompt.repeat(n,1,1),
                added_cond_kwargs=dict(text_embeds=pool.repeat(n,1),time_ids=time_ids.repeat(n,1),
                    image_embeds=[condition.cuda()[:,None,:]]),return_dict=False)[0]
            loss=F.mse_loss(predicted.float(),noise)
        assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(selected.values(),1.,error_if_nonfinite=True)
        assert float(norm)>0 and all(p.grad is not None for p in selected.values())
        assert all(p.grad is None for name,p in pipe.unet.named_parameters() if name not in selected)
        optimizer.step();updates+=1;seen+=n;cursor=start+n;epoch_loss_sum+=float(loss)*n
        assert not coverage[subject_indices,positions].any();coverage[subject_indices,positions]=True
        if updates%32==0 or cursor==132300:
            event=dict(stage='actual_image_denoising_training',arm=arm,updates=updates,seen_queries=seen,
                complete_query_pass_fraction=seen/132300,mean_loss_so_far=epoch_loss_sum/seen,
                last_minibatch_loss=float(loss),last_minibatch_timesteps=t.detach().cpu().tolist(),
                gradient_norm_preclip=float(norm),learning_rate=optimizer.param_groups[0]['lr'],
                trainable_parameters=8400896,seconds=cumulative_seconds+time.monotonic()-started)
            log.write(json.dumps(event)+'\n');write(output/'progress.json',dict(status='RUNNING',**event));print(json.dumps(event),flush=True)
        if time.monotonic()-last_snapshot>120:
            snapshot();last_snapshot=time.monotonic()
    assert coverage.all() and seen==132300 and updates==math.ceil(132300/batch)
    final={name:p.detach().cpu().clone() for name,p in selected.items()}
    difference=sum(float((final[n]-initial[n]).double().square().sum()) for n in final)
    denominator=sum(float(x.double().square().sum()) for x in initial.values())
    assert difference>0,'An unchanged branch is not trained'
    save(output/'projection_final.pt',dict(state_dict=final,arm=arm,recipe_sha256=sha(recipe_path),
        initialization_tensor_sha256=initialization_sha,training_queries=132300,updates=updates))
    report=dict(status='COMPLETED_REGISTERED_MATCHED_FULL_QUERY_PASS',arm=arm,seed=1729,subjects=10,
        completed_query_passes=1,unique_training_images=13230,seen_training_queries=132300,
        each_subject_query_coverage=coverage.sum(1).tolist(),equivalent_unique_image_exposures=10,
        updates=updates,batch_size=batch,trainable_parameters=8400896,frozen_native_IP_attention_parameters=340787200,
        trained_scope='Native ImageProjection subset; not complete349188096-parameter IP training',
        initialization='Existing public pretrained projection, never random reset',initialization_tensor_sha256=initialization_sha,
        relative_parameter_L2_change=math.sqrt(difference/denominator),mean_training_noise_MSE=epoch_loss_sum/seen,
        checkpoint=str(output/'projection_final.pt'),checkpoint_sha256=sha(output/'projection_final.pt'),
        source_input_provenance_sha256=sha(output/'input_provenance.json'),recipe_sha256=sha(recipe_path),
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),seconds=cumulative_seconds+time.monotonic()-started,
        prediction_type='epsilon',scheduler=dict(scheduler.config),generator_revision=model_receipt['revision'],
        V_or_official_test_used=False,checkpoint_selection='Last registered update, no V/test-based choice',
        formal_images_generated=0,image_gain_established=False,convergence_not_inferred_from_epoch_completion=True)
    write(output/'completion.json',report)
    if (output/'resume.pt').exists():(output/'resume.pt').unlink()
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
