"""Mechanism audit, NOT a new-method/SOTA implementation.

Synthetic exact binary latent counterexample; separately a TRAIN-fitted Gaussian
working model on real trials. No assumption that real EEG is Gaussian or that
current mean-conditioning models use product composition.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, '/root/avde_rime_20261015')
from run_fast_learning import digest, write


def synthetic():
    import numpy as np
    means=np.array([-1.,1.]); error=.25
    rows=[];max_fd_error=0.
    for sigma in [.2,.5,1.,2.,4.]:
        grid=np.linspace(-1-6*sigma,1+6*sigma,2401)
        def density_score(e,x):
            like=np.prod(np.where(np.asarray(e)[:,None]==means[None,:],1-error,error),axis=0) if len(e) else np.ones(2)
            weight=like/like.sum();pdf=np.exp(-.5*((x[:,None]-means)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
            p=pdf@weight;s=(pdf*(-(x[:,None]-means)/sigma**2))@weight/p
            return p,s
        max_error=0.; rms=0.;mass=0.
        for e in itertools.product([-1.,1.],repeat=4):
            p,s=density_score(e,grid);p0,s0=density_score([],grid)
            singles=[density_score([v],grid)[1] for v in e]
            naive=sum(singles)-3*s0; correction=s-naive
            # grad log p(E_joint|x)/product p(E_r|x), all x-constant factors cancel.
            def logratio(x):
                pj=density_score(e,x)[0];pb=density_score([],x)[0]
                return np.log(pj)+3*np.log(pb)-sum(np.log(density_score([v],x)[0]) for v in e)
            h=1e-4; fd=(logratio(grid+h)-logratio(grid-h))/(2*h)
            assert np.isfinite(fd).all();max_fd_error=max(max_fd_error,float(np.max(abs(fd-correction))))
            assert np.allclose(naive+correction,s,atol=1e-12)
            # For this IID binary channel, the count / deterministic mean is sufficient.
            assert np.allclose(s,density_score(sorted(e),grid)[1],atol=1e-12)
            weight=.5*sum(np.prod([1-error if v==y else error for v in e]) for y in means)
            rms+=weight*np.trapz(p*(naive-s)**2,grid);mass+=weight*np.trapz(p,grid)
            max_error=max(max_error,float(abs(naive-s).max()))
        rows.append(dict(noise_sigma=sigma,naive_score_weighted_RMSE=float(np.sqrt(rms/mass)),max_abs_error=max_error))
    assert max_fd_error<1e-4
    assert any(r['naive_score_weighted_RMSE']>.1 for r in rows)
    return dict(status='completed',validation='synthetic-pass only',conditional_trials_independent_given_clean_Y=True,
        conditionally_independent_given_noisy_X=False,naive_product_is_not_exact=True,
        conditional_copula_correction_identity_passed=True,finite_difference_max_error=max_fd_error,
        deterministic_mean_sufficient_in_this_counterexample=True,rows=rows,
        originality_or_real_performance_proven=False)


def fit_and_generate(out,report):
    import numpy as np
    import torch
    from diffusers import DDIMScheduler
    assert not torch.cuda.is_available();torch.set_num_threads(2)
    prepared=Path('/root/autodl-tmp/precision_score_data_20261017_v1');cfg=json.loads((prepared/'config.json').read_text())
    dm=json.loads((prepared/'data_manifest.json').read_text());tf=np.load(dm['transforms']['path'])
    data=torch.load(dm['training_packet']['path'],map_location='cpu',weights_only=True)
    old=Path(cfg['repeat_run']);manifest=json.loads((old/'training_cache_manifest.json').read_text())
    assert digest(manifest['cache']['path'])==manifest['cache']['sha256']
    raw=np.load(manifest['cache']['path']).astype(np.float64);sessions=np.asarray(manifest['session_ids'])
    cal=np.load(old/'calibration.npz')
    def project(values,ss):
        z=values.astype(np.float64,copy=True)
        for session,bias in zip(cal['levels'],cal['bias']):z[ss==session]-=bias
        return (z-tf['z_mean'])@tf['z_projection']
    z=project(raw,sessions);y=data['y'].numpy().astype(np.float64);assert z.shape==(14540,4,64)
    assert np.allclose(z.mean(1),data['z'].numpy(),atol=5e-6)
    alltrain=np.concatenate([y,z.reshape(len(z),-1)],1);mean=alltrain.mean(0)
    centered=alltrain-mean;cov=centered.T@centered/(len(y)-1)
    cov=.95*cov+.05*np.diag(np.diag(cov));assert np.linalg.eigvalsh(cov).min()>0
    np.savez_compressed(out/'training_joint_gaussian.npz',mean=mean,covariance=cov)
    base_cfg=json.loads(Path('/root/avde_rime_20261015/native680_stage2_v1.json').read_text())
    packet=torch.load(base_cfg['generation_query_packet'],map_location='cpu',weights_only=True)
    qraw=np.load(old/'EEG_only_query_fusions.npz')['single_trials'];q=project(qraw,packet['session_ids'].numpy())
    assert q.shape==(200,4,64) and len(torch.unique(packet['trial_ids']))==800
    subsets=[()] + [(i,) for i in range(4)] + list(itertools.combinations(range(4),2)) + [(0,1,2,3)]
    # Every subset derives from ONE fitted joint Gaussian; there is no capacity mismatch.
    heads={}
    for sub in subsets:
        indices=np.concatenate([np.arange(64+64*i,128+64*i) for i in sub]) if sub else np.array([],int)
        cy=cov[:64,:64];my=mean[:64]
        if sub:
            a=np.linalg.solve(cov[np.ix_(indices,indices)],cov[indices,:64]).T
            c=cy-a@cov[indices,:64]
        else:a=np.zeros((64,0));c=cy
        heads[sub]=(indices,a,c,my)
    # Mean4 Gaussian is a strong deterministic sufficient-statistic control.
    p=np.tile(np.eye(64)/4,(1,4));ce=p@cov[64:,64:]@p.T;cross=cov[:64,64:]@p.T
    ma=np.linalg.solve(ce,cross.T).T;mc=cov[:64,:64]-ma@cross.T
    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',clip_sample=False,prediction_type='v_prediction')
    scheduler.set_timesteps(50);alphas=scheduler.alphas_cumprod.numpy().astype(np.float64)
    # Precompute exact noisy Gaussian inverses. Sampling itself reads EEG only.
    inverses={}
    for t in scheduler.timesteps.tolist():
        aa=alphas[t]
        for sub,(_,_,c,_) in heads.items():inverses[t,sub]=np.linalg.inv(aa*c+(1-aa)*np.eye(64))
        inverses[t,'mean']=np.linalg.inv(aa*mc+(1-aa)*np.eye(64))
    generated={};seed=2026101656;start=time.monotonic()
    for condition in ['paired','permuted','missing']:
        qq=q if condition=='paired' else np.roll(q,1,axis=0) if condition=='permuted' else np.zeros_like(q)
        flat=qq.reshape(200,-1);mus={}
        for sub,(idx,a,c,my) in heads.items():mus[sub]=my+(flat[:,idx-64]-mean[idx])@a.T if sub else np.broadcast_to(my,(200,64))
        mus['mean']=mean[:64]+(qq.mean(1)-mean[64:].reshape(4,64).mean(0))@ma.T
        for method in ['prior','mean4','full_joint','naive_product','pair_correction']:
            if method=='prior' and condition!='paired':continue
            gen=torch.Generator().manual_seed(seed);state=torch.randn((200,64),generator=gen,dtype=torch.float64)
            for tt in scheduler.timesteps:
                t=int(tt);aa=alphas[t];a=np.sqrt(aa);sigma=np.sqrt(1-aa);x=state.numpy()
                def score(sub):return -(x-a*mus[sub])@inverses[t,sub].T
                s0=score(())
                if method=='prior':s=s0
                elif method=='mean4':s=score('mean')
                elif method=='full_joint':s=score((0,1,2,3))
                else:
                    singles=sum(score((i,)) for i in range(4))
                    s=singles-3*s0 if method=='naive_product' else sum(score(sub) for sub in itertools.combinations(range(4),2))-2*singles+3*s0
                velocity=torch.from_numpy((-sigma*s-sigma*x)/a)
                state=scheduler.step(velocity,tt,state,eta=0.,generator=gen).prev_sample
            assert torch.isfinite(state).all();generated[method+'/'+('prior' if method=='prior' else condition)]=state.numpy()
    sample=out/'EEG_only_generated_latents.npz';np.savez_compressed(sample,**generated)
    report.update(status='completed',stage='working_gaussian_EEG_only_generation_completed',training_seeds=[],
        parameter_fit='Closed-form Gaussian covariance, fixed diagonal shrinkage .05; TRAIN only; not a trained neural diffusion',
        actual_optimizer_updates=0,training_concepts=1454,training_images=14540,physical_training_trials=58160,query_concepts=200,R=4,
        query_trial_count=800,conditions=list(generated),generation_seed=seed,DDIM_steps=50,
        development_targets_gallery_read=False,sampling_seconds=time.monotonic()-start,
        data_manifest_sha256=digest(prepared/'data_manifest.json'),training_trial_cache_sha256=manifest['cache']['sha256'],
        source_query_sha256=digest(base_cfg['generation_query_packet']),sample_sha256=digest(sample),
        models_from_identical_joint_covariance=True,novel_method_or_current_failure_explanation_proven=False)


def evaluate(out,report):
    import numpy as np
    import torch
    assert not torch.cuda.is_available();torch.set_num_threads(2)
    ledger=[json.loads(x) for x in Path('/root/rime_var_20261015/artifacts/execution_ledger.jsonl').read_text().splitlines()]
    producer=[x for x in ledger if x['experiment']=='repeat_score_composition_generation_CPU_v1'][-1]
    assert producer['status']=='completed' and producer['exit_code']==0
    prepared=Path('/root/autodl-tmp/precision_score_data_20261017_v1');cfg=json.loads((prepared/'config.json').read_text());tf=np.load(prepared/'training_only_transforms.npz')
    s1=Path(cfg['stage1']);split=json.loads((s1/'split_manifest.json').read_text());s1cfg=json.loads((s1/'config.json').read_text())
    target=torch.load(s1cfg['features'],map_location='cpu',weights_only=True)['img_features'][split['development_query_image_ids']].float()
    target=torch.nn.functional.normalize(target,dim=-1).numpy().astype(np.float64)
    samples=np.load('/root/repeat_score_composition_generation_20261017_v1/EEG_only_generated_latents.npz');summary={};matrices={};vectors={}
    for name in samples.files:
        values=samples[name];image=tf['y_mean']+(values*tf['y_scale'])@tf['y_basis'].T;image/=np.linalg.norm(image,axis=1,keepdims=True)
        sim=target@image.T;wins=(np.diag(sim)[None,:]>sim).astype(float);np.fill_diagonal(wins,0)
        rank=(sim>=np.diag(sim)[None,:]).sum(0);matrices[name]=wins
        summary[name]=dict(GeneratedSemanticLatentCosineTwoWay=float(wins.sum()/(200*199)),Recall1=float((rank==1).mean()),Recall5=float((rank<=5).mean()),matched_cosine=float(np.diag(sim).mean()))
        vectors[name]=dict(rank=rank.tolist(),matched_cosine=np.diag(sim).tolist())
    sys.path.insert(0,'/root/avde_reproduction_20261016');from native_statistics import concept_bootstrap_counts,paired_identification_interval
    counts=concept_bootstrap_counts(200,repetitions=20000,seed=2026101657)
    pairs=[('joint_minus_naive','full_joint/paired','naive_product/paired'),('pair_minus_naive','pair_correction/paired','naive_product/paired'),
        ('joint_minus_mean','full_joint/paired','mean4/paired'),('joint_pairing','full_joint/paired','full_joint/permuted')]
    comparisons={name:paired_identification_interval(matrices[l],matrices[r],counts) for name,l,r in pairs}
    np.savez_compressed(out/'comparison_matrices.npz',**matrices);write(out/'per_query_metrics.json',vectors)
    report.update(status='completed',stage='working_gaussian_latent_audit_evaluated',summary=summary,comparisons=comparisons,
        bootstrap_seed=2026101657,bootstrap_repetitions=20000,training_seeds=0,generation_seeds=1,
        no_comparison_to_neural_model_due_to_different_prior=True,
        caveat='Working Gaussian mechanism control; not novel architecture, current-model causal diagnosis, independent image evaluator, or SOTA',
        actual_images_evaluated=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['synthetic','generate','evaluate'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=False);start=time.monotonic()
    report=dict(status='running',mode=a.mode,official_test_read=False,actual_images_generated=0,source_sha256=digest(__file__))
    write(out/'preregistration.json',report)
    try:
        if a.mode=='synthetic':report.update(status='completed',synthetic=synthetic())
        elif a.mode=='generate':fit_and_generate(out,report)
        else:evaluate(out,report)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,message=str(error));raise
    finally:
        report.update(seconds=time.monotonic()-start);write(out/'progress.json',report);write(out/'results.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
