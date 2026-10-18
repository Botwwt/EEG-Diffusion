"""Training-only loss-unit diagnosis and one declared gradient-scale amendment.

No development queries, targets, or evaluation results are read. No optimizer
updates. Fixed-final ordinary joint DSM weights are reference functions only.
"""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,'/root/avde_rime_20261015')
from run_fast_learning import digest,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(exist_ok=False);begin=time.monotonic()
    report=dict(status='running',stage='training_only_gradient_unit_diagnosis',optimizer_updates=0,
        official_test_read=False,development_queries_read=False,development_targets_or_results_read=False,source_sha256=digest(__file__),
        preregistration=dict(target_PDE_to_visual_gradient_ratio=.3,training_batches=8,weight_clip=[.01,100.],
            rule='median over8 fixed training draws of .3*norm(g_visual_at_v0)/norm(g_PDE), then clip; no search or development selection',
            FP32_FP64_gradient_relative_error_limit=.02,FP32_FP64_residual_relative_error_limit=.01))
    write(out/'preregistration.json',report);write(out/'progress.json',report)
    try:
        import numpy as np
        import torch
        from diffusers import DDIMScheduler
        from precision_score_math import precision_residual
        assert not torch.cuda.is_available();torch.set_num_threads(2)
        data_path=Path(cfg['prepared'])/'training_only.pt';data=torch.load(data_path,map_location='cpu',weights_only=True)
        assert data['z'].shape==data['y'].shape==(14540,64)
        sys.path.insert(0,str(Path(cfg['source'])/'Generation'));from diffusion_prior import DiffusionPriorUNet
        scheduler=DDIMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',clip_sample=False,prediction_type='v_prediction')
        def flat_grad(loss,model):
            params=list(model.parameters());grads=torch.autograd.grad(loss,params,allow_unused=True)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(params,grads)])
        measurements={};configs={}
        for kind,reference,arm in [('repeat','joint_repeat','precision_pde_repeat'),('isotropic','joint_isotropic','precision_pde_isotropic')]:
            root=Path(cfg['out'])/reference;wm=json.loads((root/'weight_manifest.json').read_text());assert digest(wm['path'])==wm['sha256']
            model=DiffusionPriorUNet(embed_dim=128,cond_dim=1,hidden_dim=cfg['hidden_dimensions'],time_embed_dim=cfg['time_embed_dimension'])
            model.load_state_dict(torch.load(wm['path'],map_location='cpu',weights_only=True)['model']);model.eval()
            chol=data['repeat_chol' if kind=='repeat' else 'isotropic_chol'];rng=torch.Generator().manual_seed(2026101653)
            rows=[];numeric=None
            for batch in range(8):
                idx=torch.randint(len(data['y']),(256,),generator=rng);t=torch.randint(0,1000,(256,),generator=rng)
                eps=torch.randn((256,64),generator=rng);y=data['y'][idx];z=data['z'][idx]
                x=scheduler.add_noise(y,eps,t);target=scheduler.get_velocity(y,eps,t)
                prediction=model(torch.cat([x,z],-1),t,torch.zeros(256,1))[:,:64]
                gx=flat_grad((prediction-target).square().mean(),model)
                pi=idx[:8];pt=torch.randint(50,950,(8,),generator=rng);v=.5+2*torch.rand(8,generator=rng)
                pe=torch.randn((8,64),generator=rng);ez=torch.randn((8,64),generator=rng)
                px=scheduler.add_noise(data['y'][pi],pe,pt);pz=data['z'][pi]+v.sqrt()[:,None]*(ez@chol.T)
                residual=precision_residual(model,px,pz,pt,v,chol,cfg['spatial_h'],cfg['variance_h'],cfg['drift_h'])
                gp=flat_grad(residual.square().mean(),model)
                nx=float(gx.norm());npde=float(gp.norm());assert nx>0 and npde>1e-10
                rows.append(dict(visual_gradient_L2=nx,PDE_gradient_L2=npde,PDE_MSE=float(residual.detach().square().mean()),
                    original_weighted_gradient_ratio=cfg['pde_weight']*npde/nx,
                    visual_PDE_gradient_cosine=float(gx@gp/(gx.norm()*gp.norm())),proposed_weight=.3*nx/npde))
                if batch==0:
                    double=copy.deepcopy(model).double();double.time_proj.register_forward_hook(lambda m,inputs,value:value.double())
                    r64=precision_residual(double,px.double(),pz.double(),pt,v.double(),chol.double(),cfg['spatial_h'],cfg['variance_h'],cfg['drift_h'])
                    g64=flat_grad(r64.square().mean(),double)
                    numeric=dict(PDE_gradient_relative_L2_error=float((gp.double()-g64).norm()/g64.norm()),
                        PDE_residual_relative_L2_error=float((residual.detach().double()-r64.detach()).norm()/r64.detach().norm()),
                        FP64_time_projection='same native float32 sinusoidal values, cast to double after projection; no author source edit')
                    assert numeric['PDE_gradient_relative_L2_error']<.02 and numeric['PDE_residual_relative_L2_error']<.01,numeric
            raw=float(np.median([r['proposed_weight'] for r in rows]));weight=float(np.clip(raw,.01,100.))
            new=copy.deepcopy(cfg);new['pde_weight']=weight;new['out']='/root/autodl-tmp/precision_score_gradient_training_20261017_v2'
            new['scope']=cfg['scope']+' Training-only gradient-unit amendment v2; fixed median30percent visual-gradient rule, no development metric tuning. Original seven arms retained.'
            new['gradient_calibration']=dict(path=str(out/'results.json'),kind=kind,source_sha256=digest(__file__),training_only=True,
                reference_weight_sha256=wm['sha256'],target_ratio=.3,weight_unclipped=raw,weight_clipped=weight)
            config_path=out/(kind+'_config.json');write(config_path,new);configs[kind]=dict(path=str(config_path),sha256=digest(config_path),arm=arm,pde_weight=weight)
            measurements[kind]=dict(batches=rows,numerical_check=numeric,weight_unclipped=raw,weight_clipped=weight,
                original_gradient_ratio_median=float(np.median([r['original_weighted_gradient_ratio'] for r in rows])),
                calibrated_gradient_ratio_median=float(np.median([weight*r['PDE_gradient_L2']/r['visual_gradient_L2'] for r in rows])))
        # The evaluation adapter only changes producer/root resolution; metric code is common.
        composite=copy.deepcopy(cfg)
        composite['evaluation_arm_roots']={v['arm']:str(Path('/root/autodl-tmp/precision_score_gradient_training_20261017_v2')/v['arm']) for v in configs.values()}
        composite['evaluation_producers']={v['arm']:f'precision_score_gradient_{v["arm"]}_seed0_GPU_v2' for v in configs.values()}
        write(out/'evaluation_config.json',composite)
        report.update(status='completed',stage='training_only_unit_calibration_completed',measurements=measurements,configs=configs,
            data_sha256=digest(data_path),evaluation_config_sha256=digest(out/'evaluation_config.json'),
            method_innovation_claimed_for_gradient_calibration=False,development_performance_not_measured_here=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,message=str(error));raise
    finally:
        report.update(seconds=time.monotonic()-begin,utc=datetime.now(timezone.utc).isoformat());write(out/'progress.json',report);write(out/'results.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
