"""One fixed-final scaled native ATM denoiser arm and EEG-only latent generation.

Common v-prediction is used in every arm for bounded low-SNR reconstruction.
This is a constructed semantic-latent experiment, not full ATM reproduction.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,'/root/avde_rime_20261015')
from run_fast_learning import digest,write


def parameter_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--arm',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());arm=a.arm;assert arm in cfg['arms'];out=Path(a.out);assert out==Path(cfg['out'])/arm
    out.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    report=dict(status='running',stage='verify_contract',arm=arm,optimizer_updates=0,official_test_read=False,
        development_target_features_read=False,gallery_read=False,source_sha256=digest(__file__),configuration_sha256=digest(a.config))
    write(out/'config.json',cfg);write(out/'progress.json',report)
    try:
        import numpy as np
        import torch
        from diffusers import DDIMScheduler
        from precision_score_math import precision_residual
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
        assert torch.__version__.split('+')[0]=='2.5.0' and torch.version.cuda=='12.4'
        assert shutil.disk_usage(out).free>2*1024**3
        torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        prepared=Path(cfg['prepared']);manifest=json.loads((prepared/'data_manifest.json').read_text());assert manifest['status']=='completed'
        assert digest(manifest['training_packet']['path'])==manifest['training_packet']['sha256'] and digest(manifest['query_packet']['path'])==manifest['query_packet']['sha256']
        data=torch.load(manifest['training_packet']['path'],weights_only=True,map_location='cpu')
        query=torch.load(manifest['query_packet']['path'],weights_only=True,map_location='cpu');d=cfg['latent_dimension']
        assert data['y'].shape==data['z'].shape==(14540,d) and query['z'].shape==(200,d) and query['R']==4
        assert set(query)=={'schema','z','query_ids','trial_ids','session_ids','R','source_query_sha256','split_sha256'}
        assert data['split_sha256']==query['split_sha256']==manifest['split_manifest_sha256']
        assert len(torch.unique(query['trial_ids']))==800 and torch.equal(query['query_ids'],torch.arange(200))
        source=Path(cfg['source']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==cfg['source_commit']
        sys.path.insert(0,str(source/'Generation'));from diffusion_prior import DiffusionPriorUNet
        class_hash=hashlib.sha256(inspect.getsource(DiffusionPriorUNet).encode()).hexdigest();assert class_hash==manifest['source_class_sha256']
        model=DiffusionPriorUNet(embed_dim=2*d,cond_dim=1,hidden_dim=cfg['hidden_dimensions'],time_embed_dim=cfg['time_embed_dimension'])
        canary_rng=torch.Generator().manual_seed(2026101651)
        canary=torch.randn((4,2*d),generator=canary_rng);target=torch.randn((4,2*d),generator=canary_rng)
        canary_t=torch.tensor([0,100,500,999]);canary_v=torch.tensor([[0.],[.1],[1.],[3.]])
        cpu_prediction=model(canary,canary_t,canary_v);cpu_prediction.sub(target).square().mean().backward()
        cpu_gradients=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
        model=model.cuda();model.zero_grad(set_to_none=True)
        gpu_prediction=model(canary.cuda(),canary_t.cuda(),canary_v.cuda());gpu_prediction.sub(target.cuda()).square().mean().backward()
        gpu_gradients=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).cpu()
        output_error=float((cpu_prediction.detach()-gpu_prediction.detach().cpu()).abs().max())
        gradient_error=float(torch.linalg.vector_norm(cpu_gradients-gpu_gradients)/torch.linalg.vector_norm(cpu_gradients))
        assert output_error<5e-5 and gradient_error<1e-4
        write(out/'FP32_CPU_GPU_semantic_check.json',dict(status='completed',output_max_abs_error=output_error,
            gradient_relative_L2_error=gradient_error,AMP=False,TF32=False,compile=False,nonzero_native_initialization=True))
        model.zero_grad(set_to_none=True)
        # Only the final projection, commonly in every arm; internal branches retain gradients.
        torch.nn.init.zeros_(model.output_layer.weight);torch.nn.init.zeros_(model.output_layer.bias)
        initial=parameter_digest(model);parameters=sum(p.numel() for p in model.parameters());assert parameters==107800
        y=data['y'].cuda();zbase=data['z'].cuda();chol=data['isotropic_chol' if 'isotropic' in arm else 'repeat_chol'].cuda()
        scheduler=DDIMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',clip_sample=False,prediction_type='v_prediction')
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],foreach=False)
        base_rng=torch.Generator(device='cuda').manual_seed(0);aux_rng=torch.Generator(device='cuda').manual_seed(2026101650)
        use_joint=arm not in ['unconditional','conditional_repeat'];use_pde=arm.startswith('precision_pde');use_extra=arm=='joint_repeat_extraDSM'
        report.update(stage='training',initial_parameter_sha256=initial,parameters=parameters,
            data_manifest_sha256=digest(prepared/'data_manifest.json'),source_class_sha256=class_hash,
            prediction_type='v_prediction in all arms; common implementation choice, not claimed innovation',
            generator='DDIM50 eta0, no CFG/temperature/candidate selection',pde_operator='isotropic trace matched' if 'isotropic' in arm else 'training physical-repeat estimate',
            training_only_images=14540,training_only_concepts=1454,physical_EEG_trials=58160,training_seed=0)
        write(out/'preregistration.json',report);write(out/'progress.json',report);torch.cuda.reset_peak_memory_stats()
        logs=[];pde_points=0;extra_points=0;training_start=time.monotonic()
        def draw(count,rng,interior=False):
            index=torch.randint(len(y),(count,),device='cuda',generator=rng)
            t=torch.randint(50 if interior else 0,950 if interior else 1000,(count,),device='cuda',generator=rng)
            e=torch.randn((count,d),device='cuda',generator=rng);ez=torch.randn((count,d),device='cuda',generator=rng)
            v=.5+2*torch.rand(count,device='cuda',generator=rng) if interior else .1+2.9*torch.rand(count,device='cuda',generator=rng)
            if not interior:v=torch.where(torch.rand(count,device='cuda',generator=rng)<.5,torch.zeros_like(v),v)
            xx=scheduler.add_noise(y[index],e,t);target=scheduler.get_velocity(y[index],e,t)
            zz=zbase[index]+v.sqrt()[:,None]*(ez@chol.T)
            if arm=='unconditional':zz=torch.zeros_like(zz);v=torch.zeros_like(v)
            return xx,zz,t,v,target,ez
        def dsm(pack):
            xx,zz,t,v,target,ez=pack;pred=model(torch.cat([xx,zz],-1),t,v[:,None])
            lx=(pred[:,:d]-target).square().mean();mask=(v>0).float()
            lz=((pred[:,d:]-ez).square().mean(1)*mask).sum()/mask.sum().clamp_min(1)
            return lx,lz
        model.train()
        with (out/'updates.jsonl').open('x') as journal:
            for step in range(cfg['updates']):
                rate=cfg['learning_rate']*min(1,(step+1)/cfg['warmup_updates'])*.5*(1+math.cos(math.pi*max(0,step-cfg['warmup_updates'])/(cfg['updates']-cfg['warmup_updates'])))
                for group in optimizer.param_groups:group['lr']=rate
                optimizer.zero_grad(set_to_none=True);lx,lz=dsm(draw(cfg['batch_size'],base_rng));loss=lx+(lz if use_joint else 0)
                regularizer=torch.zeros((),device='cuda');extra=torch.zeros((),device='cuda')
                active=step>=cfg['warmup_updates'] and (step+1)%cfg['pde_every']==0
                if active and use_pde:
                    xx,zz,t,v,_,_=draw(cfg['pde_batch'],aux_rng,interior=True)
                    residual=precision_residual(model,xx,zz,t,v,chol,cfg['spatial_h'],cfg['variance_h'],cfg['drift_h'])
                    regularizer=residual.square().mean();loss=loss+cfg['pde_weight']*regularizer;pde_points+=cfg['pde_batch']*(2*d+5)
                if active and use_extra:
                    count=cfg['pde_batch']*(2*d+5);ex,ez=dsm(draw(count,aux_rng));extra=ex+ez
                    loss=(cfg['batch_size']*loss+count*extra)/(cfg['batch_size']+count);extra_points+=count
                assert torch.isfinite(loss),f'Nonfinite objective step{step}'
                loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),2.);assert torch.isfinite(gradient)
                before=[p.detach().clone() for p in model.parameters()] if step<5 else None
                optimizer.step();report['optimizer_updates']=step+1
                if step<5 or (step+1)%25==0 or step+1==cfg['updates']:
                    torch.cuda.synchronize();row=dict(update=step+1,velocity_MSE=float(lx.detach()),EEG_noise_MSE=float(lz.detach()),
                        precision_PDE_MSE=float(regularizer.detach()),extra_DSM=float(extra.detach()),objective=float(loss.detach()),
                        gradient_L2_before_clip=float(gradient),learning_rate=rate,elapsed_training_seconds=time.monotonic()-training_start)
                    if before is not None:row['actual_parameter_update_L2']=float(sum((p.detach()-old).double().square().sum() for p,old in zip(model.parameters(),before)).sqrt())
                    journal.write(json.dumps(row)+'\n');journal.flush();logs.append(row)
                    report.update(last_update=row,training_seconds=time.monotonic()-training_start);write(out/'progress.json',report)
        torch.cuda.synchronize();training_seconds=time.monotonic()-training_start
        weights=out/'final_model.pt';torch.save(dict(model=model.cpu().state_dict(),arm=arm,config=cfg,prediction_type='v_prediction',
            initial_parameter_sha256=initial,updates=cfg['updates'],data_manifest_sha256=digest(prepared/'data_manifest.json')),weights)
        weight_manifest=dict(status='completed',schema=cfg['schema'],selection='fixed_final_update_no_validation_selection',arm=arm,
            path=str(weights),bytes=weights.stat().st_size,sha256=digest(weights),parameters=parameters,updates=cfg['updates'],seed=0,
            data_manifest_sha256=digest(prepared/'data_manifest.json'),split_manifest_sha256=manifest['split_manifest_sha256'],encoder=manifest['encoder'],
            source_class_sha256=class_hash,initial_parameter_sha256=initial)
        write(out/'weight_manifest.json',weight_manifest);model=model.cuda().eval().requires_grad_(False)
        report.update(stage='EEG_only_semantic_latent_generation',training_seconds=training_seconds);write(out/'progress.json',report)
        conditions=['prior'] if arm=='unconditional' else ['paired','permuted','missing']
        zquery=query['z'].cuda();outputs={};sample_start=time.monotonic()
        with torch.no_grad():
            for condition in conditions:
                rng=torch.Generator(device='cuda').manual_seed(cfg['generation_seed']);state=torch.randn((200,d),device='cuda',generator=rng)
                cond=zquery if condition=='paired' else zquery.roll(1,0) if condition=='permuted' else torch.zeros_like(zquery)
                scheduler.set_timesteps(cfg['generation_steps'],device='cuda')
                for t in scheduler.timesteps:
                    tt=t.expand(200);vv=torch.zeros(200,1,device='cuda')
                    velocity=model(torch.cat([state,cond],-1),tt,vv)[:,:d]
                    state=scheduler.step(velocity,t,state,eta=0.,generator=rng).prev_sample
                assert torch.isfinite(state).all();outputs[condition]=state.cpu().numpy()
        torch.cuda.synchronize();sample_seconds=time.monotonic()-sample_start
        sample_path=out/'EEG_only_generated_semantic_latents.npz';np.savez_compressed(sample_path,**outputs)
        write(out/'generation_manifest.json',dict(status='completed',schema=cfg['schema'],arm=arm,conditions=conditions,query_count=200,R=4,
            generated_latent_count=200*len(conditions),one_sample_per_query_condition_seed=True,best_of_N=False,
            actual_images_generated=0,visual_feature_dimension=d,sampling_seed=cfg['generation_seed'],steps=cfg['generation_steps'],
            sampler='common DDIM eta0 v_prediction, no clipping/CFG/guidance/gallery',query_packet_sha256=manifest['query_packet']['sha256'],
            weights_sha256=digest(weights),sample=dict(path=str(sample_path),bytes=sample_path.stat().st_size,sha256=digest(sample_path)),
            arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in outputs.items()},
            preprocessing=manifest['preprocessing'],split_manifest_sha256=manifest['split_manifest_sha256'],encoder=manifest['encoder'],
            official_test_read=False,query_target_image_class_caption_token_gallery_absent=True,
            original_VAR_probability_risk_guarantee_claimed=False))
        report.update(status='completed',stage='fixed_arm_training_and_latent_generation_completed',training_seconds=training_seconds,
            sampling_seconds=sample_seconds,pde_forward_points=pde_points,extra_DSM_forward_points=extra_points,
            base_DSM_examples=cfg['updates']*cfg['batch_size'],unique_training_images=14540,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            actual_images_generated=0,semantic_latent_samples_generated=200*len(conditions),weight_manifest=weight_manifest,
            training_budget_equal_updates_but_not_equal_compute=True,full_ATM_reproduction_or_image_SOTA_established=False)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,message=str(error));raise
    finally:
        report.update(seconds=time.monotonic()-begin,utc=datetime.now(timezone.utc).isoformat());write(out/'progress.json',report);write(out/'results.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
