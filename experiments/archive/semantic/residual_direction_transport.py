"""TRAIN-calibrated mean correction, then Gaussian and frozen ATM transport.

An optimization/representation diagnosis, not a new residual-diffusion claim.
Generation has no target/gallery input. No image-quality or SOTA claim.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, '/root/avde_rime_20261015')
from run_fast_learning import digest, write

PREP=Path('/root/autodl-tmp/precision_score_data_20261017_v1')
FIT=Path('/root/residual_direction_fit_20261017_v1')
GEN=Path('/root/residual_direction_generation_20261017_v1')
SEEDS=list(range(2026101680,2026101685))
ARMS=['proper','contrastive','joint','shuffled']
UPDATES=5000


def head_class():
    import torch
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net=torch.nn.Sequential(torch.nn.Linear(64,256),torch.nn.GELU(),
                torch.nn.Linear(256,256),torch.nn.GELU(),torch.nn.Linear(256,64))
        def forward(self,z): return self.net(z)
    return Head


def tensor_digest(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def base_context():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    manifest=json.loads((PREP/'data_manifest.json').read_text())
    cfg=json.loads((PREP/'config.json').read_text())
    for key in ['training_packet','query_packet','transforms']:
        assert digest(manifest[key]['path'])==manifest[key]['sha256']
    return manifest,cfg,np.load(manifest['transforms']['path'])


def alpha_fit(y,mu,delta,precision):
    import numpy as np
    cross=float(np.einsum('ni,ij,nj->',delta,precision,y-mu)/len(y))
    energy=float(np.einsum('ni,ij,nj->',delta,precision,delta)/len(y))
    alpha=float(np.clip(cross/max(energy,1e-30),0,1))
    change=(alpha*alpha*energy-2*alpha*cross)/(2*y.shape[1])
    return dict(alpha=alpha,residual_correlation=cross,correction_energy=energy,
                predicted_calibration_NLL_delta_nats_per_dim=change)


def synthetic_checks():
    import numpy as np
    rng=np.random.default_rng(2026101676);y=rng.normal(size=(17,4));m=rng.normal(size=y.shape)
    delta=rng.normal(size=y.shape);a=rng.normal(size=(4,4));c=a@a.T+np.eye(4);p=np.linalg.inv(c)
    fit=alpha_fit(y,m,delta,p);alpha=fit['alpha']
    measured=np.einsum('ni,ij,nj->',y-m-alpha*delta,p,y-m-alpha*delta)/(2*17*4)-np.einsum('ni,ij,nj->',y-m,p,y-m)/(2*17*4)
    assert abs(measured-fit['predicted_calibration_NLL_delta_nats_per_dim'])<1e-12
    curve=[np.einsum('ni,ij,nj->',y-m-t*delta,p,y-m-t*delta) for t in np.linspace(0,1,1001)]
    assert measured<=1e-12 and min(curve)>=np.einsum('ni,ij,nj->',y-m-alpha*delta,p,y-m-alpha*delta)-1e-10
    aa=.6;sig=.8;noise=rng.normal(size=y.shape);xt=aa*y+sig*noise
    shifted=aa*(y+delta)+sig*noise
    v=aa*noise-sig*y;vp=aa*noise-sig*(y+delta)
    assert np.allclose(shifted,xt+aa*delta) and np.allclose(vp,v-sig*delta)
    return dict(status='completed',scope='synthetic-pass only',quadratic_NLL_identity=True,
                constrained_alpha_optimum=True,VP_translation_identity=True)


def fit(out,report):
    import numpy as np
    import torch
    from run_semantic_fiber import fit_joint,mean4_head
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    manifest,cfg,tf=base_context();write(out/'synthetic_checks.json',synthetic_checks())
    data=torch.load(manifest['training_packet']['path'],map_location='cpu',weights_only=True)
    y=data['y'].numpy().astype(np.float64);z=data['z'].numpy().astype(np.float64)
    partition=json.loads(Path('/root/semantic_fiber_generation_20261017_v1/direction_calibration.json').read_text())
    fitmask=np.isin(data['concept_ids'].numpy(),partition['direction_fit_concepts']);calmask=~fitmask
    assert fitmask.sum()==11630 and calmask.sum()==2910
    old=Path(cfg['repeat_run']);tm=json.loads((old/'training_cache_manifest.json').read_text())
    assert digest(tm['cache']['path'])==tm['cache']['sha256']
    raw=np.load(tm['cache']['path']).astype(np.float64);session=np.asarray(tm['session_ids']);cal=np.load(old/'calibration.npz')
    for level,bias in zip(cal['levels'],cal['bias']):raw[session==level]-=bias
    trials=(raw-tf['z_mean'])@tf['z_projection'];del raw
    assert np.allclose(trials.mean(1),z,atol=5e-6)
    baselines={}
    for stage,mask in [('calibration_fit',fitmask),('full_refit',np.ones(len(y),dtype=bool))]:
        mean,cov=fit_joint(y[mask],trials[mask]);my,mz,a,c=mean4_head(mean,cov)
        baselines[stage]=(my,mz,a,c)
    frozen=np.load('/root/semantic_fiber_generation_20261017_v1/training_only_models.npz')
    for actual,expected in zip(baselines['full_refit'],[frozen['prior_mean'],frozen['EEG_mean'],frozen['mean4_regression'],frozen['cov_mean4_exact']]):
        assert np.allclose(actual,expected,atol=1e-8)
    native=json.loads((Path(cfg['stage1'])/'config.json').read_text())
    features=torch.load(native['features'],map_location='cpu',weights_only=True)['img_features'][data['image_ids']].float()
    features=torch.nn.functional.normalize(features,dim=-1).numpy().astype(np.float64)
    fmap=tf['y_basis']*tf['y_scale'][None,:];gram=fmap.T@fmap;offset=tf['y_mean']@fmap
    projected=features@fmap;constant=features@tf['y_mean']
    ids=data['image_ids'].cuda();concepts=data['concept_ids'].cuda();Y=torch.tensor(y,dtype=torch.float32,device='cuda');Z=torch.tensor(z,dtype=torch.float32,device='cuda')
    proj=torch.tensor(projected,dtype=torch.float32,device='cuda');const=torch.tensor(constant,dtype=torch.float32,device='cuda')
    gram=torch.tensor(gram,dtype=torch.float32,device='cuda');offset=torch.tensor(offset,dtype=torch.float32,device='cuda')
    mean_norm=float(tf['y_mean']@tf['y_mean']);Head=head_class();alphas={};weights={};metadata={}
    # Meaningful FP32 forward/backward check before zeroing the last projection.
    torch.manual_seed(0);canary=Head();xx=torch.randn(7,64);target=torch.randn(7,64)
    cp=canary(xx);(cp-target).square().mean().backward();cg=torch.cat([p.grad.ravel() for p in canary.parameters()]).clone()
    canary=canary.cuda();canary.zero_grad();gp=canary(xx.cuda());(gp-target.cuda()).square().mean().backward();gg=torch.cat([p.grad.ravel() for p in canary.parameters()]).cpu()
    oe=float((cp.detach()-gp.detach().cpu()).abs().max());ge=float((cg-gg).norm()/cg.norm());assert oe<5e-5 and ge<1e-4
    write(out/'FP32_CPU_GPU_check.json',dict(status='completed',max_output_error=oe,relative_gradient_error=ge,AMP=False,TF32=False));del canary
    torch.cuda.reset_peak_memory_stats();train_start=time.monotonic()
    for stage,mask in [('calibration_fit',fitmask),('full_refit',np.ones(len(y),dtype=bool))]:
        my,mz,a,c=baselines[stage];mu=my+(z-mz)@a.T
        M=torch.tensor(mu,dtype=torch.float32,device='cuda');precision=torch.tensor(np.linalg.inv(c),dtype=torch.float32,device='cuda')
        allowed=torch.tensor(np.flatnonzero(mask),device='cuda');calidx=torch.tensor(np.flatnonzero(calmask),device='cuda')
        for arm in ARMS:
            torch.manual_seed(0);torch.cuda.manual_seed_all(0);model=Head().cuda()
            torch.nn.init.zeros_(model.net[-1].weight);torch.nn.init.zeros_(model.net[-1].bias)
            initial=tensor_digest(model);rng=torch.Generator(device='cuda').manual_seed(0)
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01,foreach=False)
            # Concept-disjoint fixed cyclic shuffle; same training images/budget.
            order=np.argsort(data['image_ids'].numpy()[mask]);perm=allowed[torch.tensor(np.roll(order,10),device='cuda')]
            assert torch.all(concepts[allowed]!=concepts[perm])
            journal=out/(stage+'_'+arm+'_updates.jsonl');model.train()
            with journal.open('x') as stream:
                for step in range(UPDATES):
                    pos=torch.randint(len(allowed),(256,),device='cuda',generator=rng);index=allowed[pos]
                    ti=perm[pos] if arm=='shuffled' else index
                    rate=3e-4*min(1.,(step+1)/200)*.5*(1+math.cos(math.pi*max(0,step-200)/(UPDATES-200)))
                    for group in optimizer.param_groups:group['lr']=rate
                    optimizer.zero_grad(set_to_none=True);delta=model(Z[index]);pred=M[index]+delta
                    err=pred-Y[ti];proper=.5*torch.einsum('ni,ij,nj->',err,precision,err)/(len(index)*64)
                    norm=(torch.einsum('ni,ij,nj->n',pred,gram,pred)+2*(pred*offset).sum(1)+mean_norm).clamp_min(1e-12).sqrt()
                    logits=(pred@proj[ti].T+const[ti][None,:])/norm[:,None]/.07
                    positive=ids[ti][:,None]==ids[ti][None,:];sameconcept=concepts[ti][:,None]==concepts[ti][None,:]
                    contrast=(torch.logsumexp(logits.masked_fill(sameconcept&~positive,-torch.inf),1)-torch.logsumexp(logits.masked_fill(~positive,-torch.inf),1)).mean()
                    loss=proper if arm=='proper' else contrast if arm=='contrastive' else proper+.05*contrast
                    assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),2.);assert torch.isfinite(grad)
                    before=[p.detach().clone() for p in model.parameters()] if step<5 else None
                    optimizer.step();report['actual_optimizer_updates']=report.get('actual_optimizer_updates',0)+1
                    if step<5 or (step+1)%100==0:
                        row=dict(update=step+1,loss=float(loss.detach()),proper=float(proper.detach()),contrastive=float(contrast.detach()),gradient_norm_before_clip=float(grad))
                        if before is not None:row['actual_parameter_update_L2']=float(sum((p.detach()-old).double().square().sum() for p,old in zip(model.parameters(),before)).sqrt())
                        stream.write(json.dumps(row)+'\n');stream.flush()
                        write(out/'progress.json',dict(**report,stage=stage,arm=arm,update=step+1,elapsed_training_seconds=time.monotonic()-train_start))
            model.eval()
            with torch.no_grad():delta_cal=model(Z[calidx]).cpu().numpy().astype(np.float64)
            if stage=='calibration_fit':
                alphas[arm]=alpha_fit(y[calmask],mu[calmask],delta_cal,np.linalg.inv(c))
                write(out/'frozen_training_calibration.json',dict(status='running',values=alphas,official_test_read=False,main_development_targets_read=False))
            weights[stage+'_'+arm]={k:v.cpu() for k,v in model.state_dict().items()}
            metadata[stage+'_'+arm]=dict(parameters=sum(p.numel() for p in model.parameters()),initial_sha256=initial,updates=UPDATES)
            del model,optimizer
    # No calibration modification after the full-data refits.
    torch.cuda.synchronize();my,mz,a,c=baselines['full_refit']
    torch.save(dict(weights=weights,alphas=alphas,metadata=metadata,seed=0),out/'heads.pt')
    np.savez_compressed(out/'frozen_reference.npz',prior_mean=my,EEG_mean=mz,mean4_regression=a,covariance=c)
    write(out/'frozen_training_calibration.json',dict(status='completed',values=alphas,all_arms_always_reported=True,main_arm='joint',primary_control='proper',
        alpha_rule='clip(delta^T C^-1 residual / delta^T C^-1 delta,0,1)',fit_concepts=1163,calibration_concepts=291,
        encoder_and_coordinate_transforms_previously_saw_all_TRAIN_concepts=True,main_development_targets_read=False,official_test_read=False))
    report.update(status='completed',training_seconds=time.monotonic()-train_start,actual_optimizer_updates=8*UPDATES,
        metadata=metadata,alphas=alphas,training_images=14540,training_concepts=1454,training_seed=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        heads_sha256=digest(out/'heads.pt'),reference_sha256=digest(out/'frozen_reference.npz'),data_manifest_sha256=digest(PREP/'data_manifest.json'),actual_images=0)


def generate(out,report):
    import numpy as np
    import torch
    from diffusers import DDIMScheduler
    from semantic_fiber_math import sqrt_spd
    manifest,cfg,tf=base_context();receipt=json.loads((FIT/'results.json').read_text());assert receipt['status']=='completed'
    assert digest(FIT/'heads.pt')==receipt['heads_sha256'];heads=torch.load(FIT/'heads.pt',map_location='cpu',weights_only=True)
    q=torch.load(manifest['query_packet']['path'],map_location='cpu',weights_only=True)
    assert set(q)=={'schema','z','query_ids','trial_ids','session_ids','R','source_query_sha256','split_sha256'} and q['R']==4
    assert len(torch.unique(q['trial_ids']))==800
    base=np.load(FIT/'frozen_reference.npz');my=base['prior_mean'];a=base['mean4_regression'];c=base['covariance'];root=sqrt_spd(c)
    Head=head_class();models={}
    for arm in ARMS:
        model=Head().eval();model.load_state_dict(heads['weights']['full_refit_'+arm]);models[arm]=model
    source=Path(cfg['source']);sys.path.insert(0,str(source/'Generation'));from diffusion_prior import DiffusionPriorUNet
    native_dir=Path(cfg['out'])/'conditional_repeat';state=torch.load(native_dir/'final_model.pt',map_location='cpu',weights_only=True)
    assert state['config']==cfg and state['updates']==2000
    native=DiffusionPriorUNet(embed_dim=128,cond_dim=1,hidden_dim=cfg['hidden_dimensions'],time_embed_dim=64)
    native.load_state_dict(state['model']);native=native.cuda().eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',clip_sample=False,prediction_type='v_prediction')
    samples={};frozen={};checks=[]
    for condition in ['paired','permuted','missing']:
        z=q['z'] if condition=='paired' else q['z'].roll(1,0) if condition=='permuted' else torch.zeros_like(q['z'])
        mu=my+(z.numpy().astype(np.float64)-base['EEG_mean'])@a.T if condition!='missing' else np.tile(my,(200,1))
        corrections={'reference':np.zeros_like(mu),'gain4':3*(mu-my)}
        with torch.no_grad():
            for arm,model in models.items():
                corrections[arm]=heads['alphas'][arm]['alpha']*model(z).numpy().astype(np.float64) if condition!='missing' else np.zeros_like(mu)
        for name,delta in corrections.items():
            frozen['mean_'+name+'__'+condition]=mu+delta;frozen['delta_'+name+'__'+condition]=delta
        with torch.no_grad():
            for seed in SEEDS:
                rng=torch.Generator(device='cuda').manual_seed(seed);xx=torch.randn((200,64),device='cuda',generator=rng);scheduler.set_timesteps(50,device='cuda')
                for t in scheduler.timesteps:
                    velocity=native(torch.cat([xx,z.cuda()],-1),t.expand(200),torch.zeros(200,1,device='cuda'))[:,:64]
                    xx=scheduler.step(velocity,t,xx,eta=0.).prev_sample
                nn=xx.cpu().numpy().astype(np.float64);gg=mu+np.random.default_rng(seed).standard_normal((200,64))@root.T
                for name,delta in corrections.items():
                    samples['gaussian/'+name+'/'+condition+'/'+str(seed)]=gg+delta
                    samples['native/'+name+'/'+condition+'/'+str(seed)]=nn+delta
                if condition=='missing':assert all(np.array_equal(nn,nn+d) for d in corrections.values())
        checks.append(dict(condition=condition,missing_identity=condition!='missing' or all(np.count_nonzero(d)==0 for d in corrections.values())))
    np.savez_compressed(out/'EEG_only_samples.npz',**samples);np.savez_compressed(out/'frozen_EEG_only_means.npz',covariance=c,**frozen)
    report.update(status='completed',query_count=200,R=4,generation_seeds=SEEDS,actual_images=0,
        vectors=len(samples)*200,methods=['reference','gain4']+ARMS,backends=['gaussian','native'],checks=checks,
        targets_gallery_forecast_read=False,sample_sha256=digest(out/'EEG_only_samples.npz'),
        source_native_checkpoint_sha256=digest(native_dir/'final_model.pt'),native_training_updates=2000,
        transport='Y_new=Y_base+delta(E); unit-Jacobian endpoint pushforward, unchanged conditional sample differences; not newly trained diffusion or image generation')


def evaluate(out,report):
    import numpy as np
    import torch
    from mean_margin_prediction import probabilities
    manifest,cfg,tf=base_context();assert json.loads((GEN/'results.json').read_text())['status']=='completed'
    split=json.loads((Path(cfg['stage1'])/'split_manifest.json').read_text());ids=split['development_query_image_ids']
    native=json.loads((Path(cfg['stage1'])/'config.json').read_text())
    features=torch.load(native['features'],map_location='cpu',weights_only=True)['img_features'][ids].float()
    features=torch.nn.functional.normalize(features,dim=-1).numpy().astype(np.float64);target=(features-tf['y_mean'])@tf['y_projection']
    model=np.load(GEN/'frozen_EEG_only_means.npz');samples=np.load(GEN/'EEG_only_samples.npz');c=model['covariance'];precision=np.linalg.inv(c)
    summary={};matrices={};per={};raws={}
    names=sorted({k.rsplit('/',1)[0] for k in samples.files})
    for name in names:
        backend,method,condition=name.split('/');values=[samples[name+'/'+str(seed)] for seed in SEEDS];wins=[];cos=[];rr=[];mse=[]
        for y in values:
            image=tf['y_mean']+(y*tf['y_scale'])@tf['y_basis'].T;image/=np.linalg.norm(image,axis=1,keepdims=True)
            sim=features@image.T;win=(np.diag(sim)[None,:]>sim).astype(float);np.fill_diagonal(win,0);wins.append(win)
            cos.append(np.diag(sim));rr.append((sim>=np.diag(sim)[None,:]).sum(0));mse.append(((target-y)**2).mean(1))
        matrices[name]=np.mean(wins,0);per[name]=dict(cosine=np.mean(cos,0).tolist(),MSE=np.mean(mse,0).tolist(),Recall1=np.mean(np.array(rr)==1,0).tolist())
        diversity=np.mean([np.mean((values[i]-values[j])**2,1) for i in range(5) for j in range(i+1,5)],0)
        summary[name]=dict(identification=float(matrices[name].sum()/(200*199)),Recall1=float((np.array(rr)==1).mean()),Recall5=float((np.array(rr)<=5).mean()),
            cosine=float(np.mean(cos)),MSE=float(np.mean(mse)),diversity=float(diversity.mean()),latent_NLL=None,expected_identification=None)
        if backend=='gaussian':
            mean=model['mean_'+method+'__'+condition];delta=target-mean
            nll=.5*(64*np.log(2*np.pi)+np.linalg.slogdet(c)[1]+np.einsum('ni,ij,nj->n',delta,precision,delta))/64
            summary[name]['latent_NLL']=float(nll.mean());per[name]['latent_NLL']=nll.tolist()
            p=probabilities(features,mean,c,tf);summary[name]['expected_identification']=float(p.sum()/(200*199))
            d=model['delta_'+method+'__'+condition];baseline=target-model['mean_reference__'+condition]
            actual=np.einsum('ni,ij,nj->n',delta,precision,delta)-np.einsum('ni,ij,nj->n',baseline,precision,baseline)
            predicted=np.einsum('ni,ij,nj->n',d,precision,d)-2*np.einsum('ni,ij,nj->n',d,precision,baseline)
            assert abs(actual-predicted).max()<1e-8
        if method=='reference':raws[(backend,condition)]=values
    for name in names:
        backend,method,condition=name.split('/');delta=model['delta_'+method+'__'+condition]
        for i,seed in enumerate(SEEDS):assert np.max(abs(samples[name+'/'+str(seed)]-raws[(backend,condition)][i]-delta))<1e-12
        assert abs(summary[name]['diversity']-summary[backend+'/reference/'+condition]['diversity'])<1e-12
    sys.path.insert(0,'/root/avde_reproduction_20261016');from native_statistics import concept_bootstrap_counts,paired_identification_interval,paired_query_interval
    counts=concept_bootstrap_counts(200,repetitions=20000,seed=2026101685);comparisons={}
    for backend in ['gaussian','native']:
        for right in ['reference','proper','contrastive','shuffled','gain4','permuted']:
            left=backend+'/joint/paired';control=backend+'/joint/permuted' if right=='permuted' else backend+'/'+right+'/paired'
            result=dict(identification=paired_identification_interval(matrices[left],matrices[control],counts))
            for metric in per[left]:result[metric]=paired_query_interval(per[left][metric],per[control][metric],counts)
            comparisons[backend+'/joint_minus_'+right]=result
    np.savez_compressed(out/'identification_matrices.npz',**matrices);write(out/'per_query_metrics.json',dict(query_image_ids=ids,values=per))
    gate={}
    for backend in ['gaussian','native']:
        x=comparisons[backend+'/joint_minus_proper'];b=comparisons[backend+'/joint_minus_reference']
        # Report interval dictionaries; no automatic posthoc threshold fitting.
        gate[backend]=dict(primary_comparison='joint_minus_proper',joint_minus_proper=x,joint_minus_reference=b)
    report.update(status='completed',summary=summary,comparisons=comparisons,gate=gate,
        generation_seeds=SEEDS,training_seed=0,training_updates=8*UPDATES,actual_images=0,
        Gaussian_NLL_identity_verified_on_real_targets=True,unchanged_sample_differences_and_diversity_verified=True,
        forecast_status='Post-generation exact Gaussian expectation diagnostic, not a prospective prediction',
        native_likelihood_estimated=False,official_test_read=False,new_method_or_SOTA_proven=False,
        caveats=['Same repeatedly used200 development concepts; descriptive bootstrap only','CLIP is supervision, not independent image evaluation','Residual correction/translation and loss decomposition are existing tools, not established novelty'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fit','generate','evaluate'],required=True);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report=dict(status='running',mode=args.mode,source_sha256=digest(__file__),official_test_read=False)
    write(out/'preregistration.json',dict(**report,UTC=datetime.now(timezone.utc).isoformat(),arms=ARMS,updates=UPDATES,training_seed=0,generation_seeds=SEEDS,
        primary='joint versus proper at identical architecture and training budget; require identification gain without worsened Gaussian NLL/cosine and verify transfer to frozen native sampler'))
    try:globals()[args.mode](out,report)
    except BaseException as error:report.update(status='failed',error_type=type(error).__name__,message=str(error));raise
    finally:
        report.update(seconds=time.monotonic()-start,UTC=datetime.now(timezone.utc).isoformat());write(out/'results.json',report);write(out/'progress.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
