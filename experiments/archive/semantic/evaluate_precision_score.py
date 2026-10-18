"""Evaluate ALL fixed semantic-latent samples only after every GPU producer exits."""
import argparse
from datetime import datetime,timezone
import inspect
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,'/root/avde_rime_20261015')
from run_fast_learning import digest,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(exist_ok=False);begin=time.monotonic()
    report=dict(status='running',stage='verify_all_actual_producer_exits',official_test_read=False,actual_images_evaluated=0,
        Inception_used=False,source_sha256=digest(__file__),GPU_hours=0)
    write(out/'progress.json',report)
    try:
        import numpy as np
        import torch
        from diffusers import DDIMScheduler
        from precision_score_math import precision_residual
        sys.path.insert(0,'/root/avde_reproduction_20261016')
        from native_statistics import concept_bootstrap_counts,paired_identification_interval,paired_query_interval
        torch.set_num_threads(2);assert not torch.cuda.is_available()
        ledger=[json.loads(x) for x in Path('/root/rime_var_20261015/artifacts/execution_ledger.jsonl').read_text().splitlines()]
        selected_producers={}
        for arm in cfg['arms']:
            producer_name=cfg.get('evaluation_producers',{}).get(arm,f'precision_score_{arm}_seed0_GPU_v1')
            producer=[r for r in ledger if r['experiment']==producer_name][-1]
            assert producer['status']=='completed' and producer['exit_code']==0
            selected_producers[arm]=producer
        prepared=Path(cfg['prepared']);manifest=json.loads((prepared/'data_manifest.json').read_text());transforms=np.load(manifest['transforms']['path'])
        assert digest(manifest['transforms']['path'])==manifest['transforms']['sha256']
        train=torch.load(manifest['training_packet']['path'],map_location='cpu',weights_only=True)
        queries=torch.load(manifest['query_packet']['path'],map_location='cpu',weights_only=True)
        s1=Path(cfg['stage1']);s1cfg=json.loads((s1/'config.json').read_text());split=json.loads((s1/'split_manifest.json').read_text())
        assert digest(s1/'split_manifest.json')==manifest['split_manifest_sha256'];ids=split['development_query_image_ids']
        assert len(ids)==len(set(i//10 for i in ids))==200
        features=torch.load(s1cfg['features'],map_location='cpu',weights_only=True)['img_features'][ids].float()
        features=torch.nn.functional.normalize(features,dim=-1).numpy().astype(np.float64)
        target_y=(features-transforms['y_mean'])@transforms['y_projection'];yt=torch.tensor(target_y,dtype=torch.float32)
        sys.path.insert(0,str(Path(cfg['source'])/'Generation'));from diffusion_prior import DiffusionPriorUNet
        assert hashlib.sha256(inspect.getsource(DiffusionPriorUNet).encode()).hexdigest()==manifest['source_class_sha256']
        scheduler=DDIMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',clip_sample=False,prediction_type='v_prediction')
        rng=torch.Generator().manual_seed(cfg['denoising_evaluation_seed']);noise=torch.randn((5,200,64),generator=rng);times=[50,200,500,800,950]
        counts=concept_bootstrap_counts(200,repetitions=20000,seed=cfg['bootstrap_seed'])
        summary={};vectors={};matrices={};denoising={};costs={};initial=set();weights={};pde_measurements={}
        def score(name,generated):
            reconstructed=transforms['y_mean']+(np.asarray(generated,dtype=np.float64)*transforms['y_scale'])@transforms['y_basis'].T
            reconstructed/=np.linalg.norm(reconstructed,axis=1,keepdims=True)
            similarity=features@reconstructed.T;assert np.isfinite(similarity).all()
            comparison=(np.diag(similarity)[None,:]>similarity).astype(np.float64);np.fill_diagonal(comparison,0)
            ranks=(similarity>=np.diag(similarity)[None,:]).sum(0)
            values=dict(GeneratedSemanticLatentCosineTwoWay=(comparison.sum(0)/199).tolist(),
                matched_CLIP_cosine=np.diag(similarity).tolist(),PCA64_MSE=np.mean((generated-target_y)**2,axis=1).tolist(),
                generated_latent_gallery_rank_pessimistic=ranks.tolist())
            matrices[name]=comparison;vectors[name]=values
            summary[name]={k:float(np.mean(v)) for k,v in values.items()}
            summary[name].update(generated_latent_gallery_Recall1=float((ranks==1).mean()),generated_latent_gallery_Recall5=float((ranks<=5).mean()))
        # Fixed training-only linear readout: a useful context baseline, not a sampler.
        ztrain=train['z'].double().numpy();ytrain=train['y'].double().numpy();ridge=np.linalg.solve(ztrain.T@ztrain+.1*len(ztrain)*np.eye(64),ztrain.T@ytrain)
        score('ridge_training_only_context_baseline',queries['z'].numpy()@ridge);score('PCA64_projection_oracle_interface_only',target_y)
        for arm in cfg['arms']:
            root=Path(cfg.get('evaluation_arm_roots',{}).get(arm,str(Path(cfg['out'])/arm)))
            command=selected_producers[arm]['command'];assert isinstance(command,list)
            assert Path(command[command.index('--out')+1]).resolve()==root.resolve()
            wm=json.loads((root/'weight_manifest.json').read_text());gm=json.loads((root/'generation_manifest.json').read_text())
            assert wm['arm']==gm['arm']==arm
            assert digest(wm['path'])==wm['sha256'] and digest(gm['sample']['path'])==gm['sample']['sha256']
            assert gm['query_packet_sha256']==manifest['query_packet']['sha256'] and wm['updates']==cfg['updates']
            initial.add(wm['initial_parameter_sha256']);weights[arm]=wm
            generated=np.load(gm['sample']['path'])
            for condition in gm['conditions']:score(arm+'/'+condition,generated[condition])
            model=DiffusionPriorUNet(embed_dim=128,cond_dim=1,hidden_dim=cfg['hidden_dimensions'],time_embed_dim=cfg['time_embed_dimension'])
            state=torch.load(wm['path'],map_location='cpu',weights_only=True);model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
            conds=['prior'] if arm=='unconditional' else ['paired','permuted','missing']
            with torch.no_grad():
                for condition in conds:
                    z=queries['z'] if condition=='paired' else queries['z'].roll(1,0) if condition=='permuted' else torch.zeros_like(queries['z'])
                    values=[]
                    for k,t in enumerate(times):
                        tt=torch.full((200,),t,dtype=torch.long);x=scheduler.add_noise(yt,noise[k],tt);target=scheduler.get_velocity(yt,noise[k],tt)
                        pred=model(torch.cat([x,z],-1),tt,torch.zeros(200,1))[:,:64]
                        values.append((pred-target).square().mean(-1).numpy())
                    denoising[arm+'/'+condition]=np.stack(values)
                    vectors[arm+'/'+condition]['velocity_MSE_5time_average']=np.stack(values).mean(0).tolist()
                    summary[arm+'/'+condition]['velocity_MSE_5time_average']=float(np.stack(values).mean())
                # Fixed16-query auxiliary residual, no selection or extra training.
                if arm!='unconditional':
                    chol=train['isotropic_chol' if 'isotropic' in arm else 'repeat_chol'];z=queries['z'][:16]
                    erng=torch.Generator().manual_seed(2026101652);e=torch.randn((16,64),generator=erng);ez=torch.randn((16,64),generator=erng)
                    tt=torch.full((16,),500,dtype=torch.long);v=torch.ones(16);x=scheduler.add_noise(yt[:16],e,tt);zz=z+ez@chol.T
                    residual=precision_residual(model,x,zz,tt,v,chol,cfg['spatial_h'],cfg['variance_h'],cfg['drift_h'])
                    pde_measurements[arm]=float(residual.square().mean())
            r=json.loads((root/'results.json').read_text());costs[arm]={k:r[k] for k in ['training_seconds','sampling_seconds','seconds','peak_allocated_bytes','pde_forward_points','extra_DSM_forward_points','base_DSM_examples']}
            report.update(stage='evaluate_all_fixed_semantic_latents',completed_arms=len(weights));write(out/'progress.json',report)
        assert len(initial)==1,'Matched arms must start from bitwise equal native tensors'
        pairs=[('PDE_repeat_minus_extra_DSM','precision_pde_repeat/paired','joint_repeat_extraDSM/paired'),
            ('PDE_repeat_minus_joint_repeat','precision_pde_repeat/paired','joint_repeat/paired'),
            ('PDE_repeat_minus_PDE_isotropic','precision_pde_repeat/paired','precision_pde_isotropic/paired'),
            ('PDE_repeat_minus_conditional','precision_pde_repeat/paired','conditional_repeat/paired'),
            ('PDE_repeat_pairing','precision_pde_repeat/paired','precision_pde_repeat/permuted'),
            ('PDE_repeat_minus_prior','precision_pde_repeat/paired','unconditional/prior'),
            ('joint_repeat_pairing','joint_repeat/paired','joint_repeat/permuted')]
        comparisons={}
        for name,left,right in pairs:
            comparisons[name]={'GeneratedSemanticLatentCosineTwoWay':paired_identification_interval(matrices[left],matrices[right],counts)}
            for metric in ['matched_CLIP_cosine','PCA64_MSE','velocity_MSE_5time_average']:
                comparisons[name][metric]=paired_query_interval(vectors[left][metric],vectors[right][metric],counts)
        interaction=paired_identification_interval(matrices['precision_pde_repeat/paired']-matrices['joint_repeat/paired'],matrices['precision_pde_isotropic/paired']-matrices['joint_isotropic/paired'],counts)
        write(out/'per_query_metrics.json',dict(query_ids=list(range(200)),target_image_ids=ids,values=vectors))
        np.savez_compressed(out/'comparisons_and_denoising.npz',**{'comparison_'+k.replace('/','__'):v for k,v in matrices.items()},
            **{'denoising_'+k.replace('/','__'):v for k,v in denoising.items()})
        report.update(status='completed',stage='all7_arms_semantic_latent_evaluation_completed',summary=summary,comparisons=comparisons,
            repeat_operator_PDE_minus_isotropic_PDE_interaction=interaction,heldout_first16_precision_PDE_residual=pde_measurements,
            costs=costs,weights=weights,initial_parameters_bitwise_equal_all7=True,
            statistics=dict(bootstrap_samples=20000,bootstrap_seed=cfg['bootstrap_seed'],unit='200 internal concepts, identification query/distractor axes jointly',
                training_seeds=1,generation_seeds=1,confirmatory=False,research_wide_multiplicity_adjusted=False,continuous_EEG_overlap_not_modeled=True),
            definitions=dict(GeneratedSemanticLatentCosineTwoWay='Generated semantic vector inverse-PCA to CLIP-H14; cosine vs matched real image embedding and each199 distractors. NOT a generated image, Inception identification, CLIPScore or independent evaluator.',
                velocity_MSE_5time_average='Fixed5 VP image noise times, shared noises; aggregate per query before concept bootstrap. A denoising objective, not NLL or mutual information.',
                projection_oracle='Ground-truth visual coordinate upper reference only; not EEG performance or a learned diffusion oracle'),
            primary_component_contrast='PDE_repeat_minus_extra_DSM',all200_queries_included=True,training_only_coordinate_fits=True,
            evaluated_arm_roots=cfg.get('evaluation_arm_roots',{}),evaluation_producer_overrides=cfg.get('evaluation_producers',{}),
            full_ATM_or_AVDE_reproduction=False,image_generation_improvement_or_SOTA_established=False)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,message=str(error));raise
    finally:
        report.update(seconds=time.monotonic()-begin,utc=datetime.now(timezone.utc).isoformat());write(out/'progress.json',report);write(out/'results.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['weights']}),flush=True)


if __name__=='__main__':main()
