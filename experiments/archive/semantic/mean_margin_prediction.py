"""TRAIN-only gain selection, frozen analytic forecast, then new-seed measurement.

The analytic formula is exact for the specified Gaussian semantic generator and
cosine pair comparison. It is not an Inception/image-quality theorem.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,'/root/avde_rime_20261015')
from run_fast_learning import digest,write

GAIN_GRID=[.5,1.,1.5,2.,3.,4.]
SEEDS=[2026101670,2026101671,2026101672,2026101673,2026101674]
MODEL_DIR=Path('/root/mean_margin_calibration_20261017_v1')
FORECAST_DIR=Path('/root/mean_margin_forecast_20261017_v1')
GEN_DIR=Path('/root/mean_margin_generation_20261017_v1')


def probabilities(features,mean,c,tf):
    import numpy as np
    from scipy.special import ndtr
    projection=features@(tf['y_basis']*tf['y_scale'][None,:])
    diff=projection[:,None,:]-projection[None,:,:]
    visualmean=tf['y_mean']+mean@(tf['y_basis']*tf['y_scale'][None,:]).T
    margin=np.einsum('ij,ij->i',features,visualmean)[:,None]-visualmean@features.T
    variance=np.einsum('ijk,kl,ijl->ij',diff,c,diff)
    assert np.min(variance)>-1e-10
    degenerate=variance<1e-20
    p=ndtr(margin/np.sqrt(np.maximum(variance,1e-20)))
    p=np.where(degenerate,(margin>0).astype(float),p);np.fill_diagonal(p,0)
    return p.T


def validate_formula():
    import numpy as np
    rng=np.random.default_rng(2026101669);features=rng.normal(size=(6,8));features/=np.linalg.norm(features,axis=1,keepdims=True)
    features[1]=features[0]
    tf=dict(y_basis=rng.normal(size=(8,3)),y_scale=np.array([.5,1.,2.]),y_mean=rng.normal(size=8))
    mean=rng.normal(size=(6,3));c=np.array([[1.,.2,0],[.2,.8,.1],[0,.1,.6]])
    p=probabilities(features,mean,c,tf);measured=np.zeros_like(p)
    for i in range(6):
        samples=mean[i]+rng.standard_normal((100000,3))@np.linalg.cholesky(c).T
        image=tf['y_mean']+(samples*tf['y_scale'])@tf['y_basis'].T
        image/=np.linalg.norm(image,axis=1,keepdims=True)
        scores=image@features.T;measured[:,i]=(scores[:,i,None]>scores).mean(0)
    np.fill_diagonal(measured,0);err=float(abs(measured-p).max());assert err<.008
    assert p[0,1]==p[1,0]==0
    return dict(status='completed',validation='synthetic-pass only',Monte_Carlo_draws_per_query=100000,
        maximum_forecast_probability_error=err,degenerate_duplicate_feature_tie_passed=True)


def context():
    import torch
    import numpy as np
    assert not torch.cuda.is_available();torch.set_num_threads(2)
    prepared=Path('/root/autodl-tmp/precision_score_data_20261017_v1')
    cfg=json.loads((prepared/'config.json').read_text());s1=Path(cfg['stage1']);scfg=json.loads((s1/'config.json').read_text())
    return prepared,cfg,scfg,np.load(prepared/'training_only_transforms.npz')


def affine_cfg(mu,c,c0,my,w):
    import numpy as np
    import torch
    from diffusers import DDIMScheduler
    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',clip_sample=False,prediction_type='v_prediction');scheduler.set_timesteps(50)
    j=np.eye(64);b=np.zeros_like(mu)
    for tt in scheduler.timesteps:
        aa=float(scheduler.alphas_cumprod[int(tt)]);a=np.sqrt(aa);sig=np.sqrt(1-aa)
        pc=np.linalg.inv(aa*c+(1-aa)*np.eye(64));p0=np.linalg.inv(aa*c0+(1-aa)*np.eye(64))
        precision=w*pc+(1-w)*p0;term=w*mu@pc.T+(1-w)*my@p0.T
        jac=sig/a*(precision-np.eye(64))
        m=scheduler.step(torch.from_numpy(jac.T.copy()),tt,torch.eye(64,dtype=torch.float64),eta=0.).prev_sample.numpy().T
        n=scheduler.step(torch.from_numpy(-sig*term),tt,torch.zeros_like(torch.from_numpy(mu)),eta=0.).prev_sample.numpy()
        j=m@j;b=b@m.T+n
    return b,j


def calibrate(out,report):
    import numpy as np
    import torch
    from run_semantic_fiber import fit_joint,mean4_head
    from semantic_fiber_math import sqrt_spd
    prepared,cfg,scfg,tf=context();old=Path(cfg['repeat_run'])
    write(out/'synthetic_margin_formula_check.json',validate_formula())
    data=torch.load(prepared/'training_only.pt',map_location='cpu',weights_only=True)
    partition=json.loads(Path('/root/semantic_fiber_generation_20261017_v1/direction_calibration.json').read_text())
    concepts=data['concept_ids'].numpy();fit=np.isin(concepts,partition['direction_fit_concepts']);calmask=~fit
    tm=json.loads((old/'training_cache_manifest.json').read_text());assert digest(tm['cache']['path'])==tm['cache']['sha256']
    raw=np.load(tm['cache']['path']).astype(np.float64);cal=np.load(old/'calibration.npz');sessions=np.asarray(tm['session_ids'])
    for level,bias in zip(cal['levels'],cal['bias']):raw[sessions==level]-=bias
    z=(raw-tf['z_mean'])@tf['z_projection'];del raw
    y=data['y'].numpy().astype(np.float64);mean,cov=fit_joint(y[fit],z[fit]);my,mz,a,c=mean4_head(mean,cov)
    # Exactly one predeclared offset0 image per TRAIN calibration concept.
    selected=calmask&(data['image_ids'].numpy()%10==0);ids=data['image_ids'].numpy()[selected]
    assert len(ids)==291 and set(ids//10)==set(partition['diagnostic_calibration_concepts'])
    split=json.loads((Path(cfg['stage1'])/'split_manifest.json').read_text());assert not set(ids)&set(split['development_query_image_ids'])
    features=torch.load(scfg['features'],map_location='cpu',weights_only=True)['img_features'][ids].float()
    features=torch.nn.functional.normalize(features,dim=-1).numpy().astype(np.float64)
    mu=my+(z[selected].mean(1)-mz)@a.T;curve=[]
    for gain in GAIN_GRID:
        p=probabilities(features,my+gain*(mu-my),c,tf)
        curve.append(dict(gain=gain,predicted_identification=float(p.sum()/(291*290))))
    chosen=max(curve,key=lambda row:(row['predicted_identification'],-abs(row['gain']-1)))['gain']
    # Freeze six candidate/baseline laws BEFORE any development forecast is read.
    frozen=np.load('/root/semantic_fiber_generation_20261017_v1/training_only_models.npz')
    means=np.load('/root/semantic_fiber_generation_20261017_v1/conditional_means.npz')
    previous=np.load('/root/semantic_mean_cov_generation_20261017_v1/affine_models.npz')
    reference=frozen['cov_mean4_exact'];root=sqrt_spd(reference);f=np.load('/root/repeat_score_composition_generation_20261017_v1/training_joint_gaussian.npz')
    my=f['mean'][:64];c0=f['covariance'][:64,:64];models={}
    for condition in ['paired','permuted','missing']:
        mu=means['mean4_exact/'+condition]
        for name,gain in [('reference',1.),('fixed_gain2',2.),('calibrated_gain',chosen)]:
            models['mean_'+name+'__'+condition]=my+gain*(mu-my);models['root_'+name]=root
        models['mean_CFG2__'+condition]=previous['cfg_mean_'+condition];models['root_CFG2']=previous['cfg_noise_jacobian']
        b,j=affine_cfg(mu,reference,c0,my,chosen);models['mean_CFG_calibrated__'+condition]=b;models['root_CFG_calibrated']=j
        models['mean_product__'+condition]=previous['product_mean_'+condition];models['root_product']=previous['product_noise_jacobian']
    np.savez_compressed(out/'frozen_EEG_only_models.npz',**models)
    report.update(status='completed',stage='training_only_gain_frozen_before_development_forecast',gain_grid=GAIN_GRID,
        selected_gain=chosen,training_calibration_curve=curve,fit_concepts=len(partition['direction_fit_concepts']),
        calibration_concepts=291,calibration_image_offset=0,calibration_image_ids=ids.tolist(),generation_seeds=SEEDS,
        development_target_rows_or_results_used=False,official_test_read=False,
        feature_file_contains_reserved_development_rows_but_only_declared_training_rows_indexed=True,
        encoder_previously_saw_all_training_concepts=True,actual_new_optimizer_updates=0,
        model_sha256=digest(out/'frozen_EEG_only_models.npz'),
        primary_comparison='calibrated_gain_minus_fixed_gain2',
        planned_comparisons=['calibrated_gain_minus_reference','calibrated_gain_minus_fixed_gain2','calibrated_gain_minus_CFG_calibrated','calibrated_gain_minus_CFG2','calibrated_gain_minus_product','calibrated_pairing'])


def forecast(out,report):
    import numpy as np
    import torch
    prepared,cfg,scfg,tf=context();cal=json.loads((MODEL_DIR/'results.json').read_text());assert cal['status']=='completed'
    model=np.load(MODEL_DIR/'frozen_EEG_only_models.npz');assert digest(MODEL_DIR/'frozen_EEG_only_models.npz')==cal['model_sha256']
    split=json.loads((Path(cfg['stage1'])/'split_manifest.json').read_text())
    feats=torch.load(scfg['features'],map_location='cpu',weights_only=True)['img_features'][split['development_query_image_ids']].float()
    feats=torch.nn.functional.normalize(feats,dim=-1).numpy().astype(np.float64);mats={};summary={}
    methods=['reference','fixed_gain2','calibrated_gain','CFG2','CFG_calibrated','product']
    for method in methods:
        for condition in ['paired','permuted','missing']:
            key=method+'/'+condition;r=model['root_'+method]
            p=probabilities(feats,model['mean_'+method+'__'+condition],r@r.T,tf);mats[key]=p
            summary[key]=float(p.sum()/(200*199))
    np.savez_compressed(out/'predicted_identification_probabilities.npz',**mats)
    report.update(status='completed',stage='frozen_laws_analytic_forecast_no_new_samples',summary=summary,
        selected_gain=cal['selected_gain'],generation_seeds=SEEDS,
        models_sha256=cal['model_sha256'],predicted_probabilities_sha256=digest(out/'predicted_identification_probabilities.npz'),
        can_change_gain_or_cancel_generation_based_on_forecast=False,
        scope='Exact expected cosine pair identification for the DEFINED Gaussian semantic law; not actual EEG posterior accuracy or Inception quality')


def generate(out,report):
    import numpy as np
    cal=json.loads((MODEL_DIR/'results.json').read_text());assert cal['status']=='completed'
    assert digest(MODEL_DIR/'frozen_EEG_only_models.npz')==cal['model_sha256'];model=np.load(MODEL_DIR/'frozen_EEG_only_models.npz')
    samples={}
    for key in model.files:
        if not key.startswith('mean_'):continue
        method,condition=key.removeprefix('mean_').split('__')
        for seed in SEEDS:
            noise=np.random.default_rng(seed).standard_normal((200,64));value=model[key]+noise@model['root_'+method].T
            assert np.isfinite(value).all();samples[method+'/'+condition+'/'+str(seed)]=value
    np.savez_compressed(out/'EEG_only_generated_latents.npz',**samples)
    report.update(status='completed',stage='fixed_new_seed_samples_generated_without_forecast_or_targets',selected_gain=cal['selected_gain'],
        generation_seeds=SEEDS,query_count=200,R=4,actual_new_optimizer_updates=0,actual_new_images=0,
        development_targets_gallery_forecast_read=False,sample_sha256=digest(out/'EEG_only_generated_latents.npz'))


def evaluate(out,report):
    import numpy as np
    import torch
    prepared,cfg,scfg,tf=context();assert json.loads((GEN_DIR/'results.json').read_text())['status']=='completed'
    ledger=[json.loads(x) for x in Path('/root/rime_var_20261015/artifacts/execution_ledger.jsonl').read_text().splitlines()]
    receipt=[x for x in ledger if x['experiment']=='mean_margin_generation_CPU_v1'][-1];assert receipt['status']=='completed' and receipt['exit_code']==0
    split=json.loads((Path(cfg['stage1'])/'split_manifest.json').read_text());ids=split['development_query_image_ids']
    feats=torch.load(scfg['features'],map_location='cpu',weights_only=True)['img_features'][ids].float();feats=torch.nn.functional.normalize(feats,dim=-1).numpy().astype(np.float64)
    target=(feats-tf['y_mean'])@tf['y_projection'];model=np.load(MODEL_DIR/'frozen_EEG_only_models.npz');samples=np.load(GEN_DIR/'EEG_only_generated_latents.npz')
    forecast=json.loads((FORECAST_DIR/'results.json').read_text());predicted=np.load(FORECAST_DIR/'predicted_identification_probabilities.npz')
    matrices={};summary={};vectors={};diversity={};ranks={}
    for name in sorted({k.rsplit('/',1)[0] for k in samples.files}):
        method,condition=name.split('/');wins=[];cos=[];rr=[];vals=[samples[name+'/'+str(seed)] for seed in SEEDS]
        for value in vals:
            image=tf['y_mean']+(value*tf['y_scale'])@tf['y_basis'].T;image/=np.linalg.norm(image,axis=1,keepdims=True)
            sim=feats@image.T;win=(np.diag(sim)[None,:]>sim).astype(float);np.fill_diagonal(win,0);wins.append(win);cos.append(np.diag(sim));rr.append((sim>=np.diag(sim)[None,:]).sum(0))
        matrices[name]=np.mean(wins,0);ranks[name]=np.stack(rr);r=model['root_'+method];c=r@r.T;delta=target-model['mean_'+method+'__'+condition]
        nll=.5*(64*np.log(2*np.pi)+np.linalg.slogdet(c)[1]+np.einsum('ni,ij,nj->n',delta,np.linalg.inv(c),delta))/64
        distances=np.mean([np.mean((vals[i]-vals[j])**2,1) for i in range(5) for j in range(i+1,5)],0);diversity[name]=distances
        vectors[name]=dict(cosine=np.mean(cos,0).tolist(),per_seed_ranks=np.stack(rr).tolist(),latent_NLL=nll.tolist())
        measured=float(matrices[name].sum()/(200*199));prediction=forecast['summary'][name]
        summary[name]=dict(measured_identification=measured,forecast_identification=prediction,forecast_error_pp=(measured-prediction)*100,
            Recall1=float((ranks[name]==1).mean()),Recall5=float((ranks[name]<=5).mean()),matched_cosine=float(np.mean(cos)),
            latent_NLL_nats_per_dimension=float(nll.mean()),five_seed_latent_diversity=float(distances.mean()),conditional_variance_trace=float(np.trace(c)))
    sys.path.insert(0,'/root/avde_reproduction_20261016');from native_statistics import concept_bootstrap_counts,paired_identification_interval,paired_query_interval
    counts=concept_bootstrap_counts(200,repetitions=20000,seed=2026101675);cal=json.loads((MODEL_DIR/'results.json').read_text());comparisons={}
    for name in cal['planned_comparisons']:
        left='calibrated_gain/paired';right='calibrated_gain/permuted' if name=='calibrated_pairing' else name.removeprefix('calibrated_gain_minus_')+'/paired'
        ri=paired_identification_interval(matrices[left],matrices[right],counts);ri.update(independent_training_seeds=0,model_fit='closed_form')
        forecast_delta=forecast['summary'][left]-forecast['summary'][right]
        entry=dict(measured=ri,forecast_delta=forecast_delta,forecast_delta_error_pp=(ri['paired_minus_control']-forecast_delta)*100)
        for metric in ['cosine','latent_NLL']:entry[metric]=paired_query_interval(vectors[left][metric],vectors[right][metric],counts)
        entry['Recall1']=paired_query_interval((ranks[left]==1).mean(0),(ranks[right]==1).mean(0),counts)
        comparisons[name]=entry
    np.savez_compressed(out/'measured_matrices.npz',**matrices);write(out/'per_query_metrics.json',dict(target_image_ids=ids,values=vectors))
    report.update(status='completed',stage='forecast_checked_against_new_seeds_all200_queries',selected_gain=cal['selected_gain'],summary=summary,comparisons=comparisons,
        generation_seeds=SEEDS,training_seeds=0,actual_new_images=0,actual_new_optimizer_updates=0,
        mathematical_formula='Phi((mean pair margin)/(standard deviation)), exact for the specified Gaussian output law; cosine normalization cancels in pairwise comparison',
        caveat='Repeatedly used development concepts; new seeds test stochastic forecast, not new-concept confirmatory validity. CLIP is supervision, not independent image evaluator.',
        primary_comparison=cal['primary_comparison'],novel_method_or_SOTA_proven=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['calibrate','forecast','generate','evaluate'],required=True);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=False);start=time.monotonic();report=dict(status='running',mode=args.mode,source_sha256=digest(__file__),official_test_read=False)
    write(out/'preregistration.json',dict(**report,gain_grid=GAIN_GRID,generation_seeds=SEEDS,rule='TRAIN calibration expected Gaussian pair identification maximum; all six fixed laws always generated; no development strength selection'))
    try:globals()[args.mode](out,report)
    except BaseException as error:report.update(status='failed',error_type=type(error).__name__,message=str(error));raise
    finally:report.update(seconds=time.monotonic()-start);write(out/'results.json',report);write(out/'progress.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
