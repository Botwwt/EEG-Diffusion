"""Run the preregistered semantic tilt mechanism test. No image/SOTA substitution."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,'/root/avde_rime_20261015')
from run_fast_learning import digest,write


def mean4_head(mean,cov):
    import numpy as np
    p=np.tile(np.eye(64)/4,(1,4));cross=cov[:64,64:]@p.T
    a=np.linalg.solve(p@cov[64:,64:]@p.T,cross.T).T
    return mean[:64],mean[64:].reshape(4,64).mean(0),a,cov[:64,:64]-a@cross.T


def fit_joint(y,z):
    import numpy as np
    values=np.concatenate([y,z.reshape(len(y),-1)],1);mean=values.mean(0);values-=mean
    cov=values.T@values/(len(y)-1);return mean,.95*cov+.05*np.diag(np.diag(cov))


def generate(cfg,out,report):
    import numpy as np
    import torch
    from diffusers import DDIMScheduler
    from semantic_fiber_math import budget_tilt,uniform_kl_scale,kl_cov,fiber_check,exact_gaussian_vp_flow,validate_math
    assert not torch.cuda.is_available();torch.set_num_threads(2)
    write(out/'synthetic_tests.json',validate_math())
    prepared=Path(cfg['prepared']);basecfg=json.loads((prepared/'config.json').read_text());dm=json.loads((prepared/'data_manifest.json').read_text())
    data=torch.load(dm['training_packet']['path'],map_location='cpu',weights_only=True)
    query=torch.load(dm['query_packet']['path'],map_location='cpu',weights_only=True)
    assert digest(dm['training_packet']['path'])==dm['training_packet']['sha256']
    assert digest(dm['query_packet']['path'])==dm['query_packet']['sha256']
    assert query['R']==4 and len(torch.unique(query['trial_ids']))==800
    tf=np.load(dm['transforms']['path']);old=Path(basecfg['repeat_run']);manifest=json.loads((old/'training_cache_manifest.json').read_text())
    assert digest(manifest['cache']['path'])==manifest['cache']['sha256']
    raw=np.load(manifest['cache']['path']);cal=np.load(old/'calibration.npz');sessions=np.asarray(manifest['session_ids'])
    def project(values,ss):
        z=values.astype(np.float64,copy=True)
        for session,bias in zip(cal['levels'],cal['bias']):z[ss==session]-=bias
        return (z-tf['z_mean'])@tf['z_projection']
    z=project(raw,sessions);del raw
    y=data['y'].numpy().astype(np.float64);concepts=data['concept_ids'].numpy()
    assert z.shape==(14540,4,64) and np.allclose(z.mean(1),data['z'].numpy(),atol=5e-6)
    unique=np.unique(concepts);rng=np.random.default_rng(cfg['training_partition_seed']);order=rng.permutation(unique)
    nc=round(len(unique)*cfg['calibration_concept_fraction']);cal_concepts=order[:nc];fit_concepts=order[nc:]
    fitmask=np.isin(concepts,fit_concepts);calmask=~fitmask;prediction=np.zeros_like(y)
    for fold in range(2):
        held=np.isin(concepts,fit_concepts[fold::2]);fit=fitmask&~held
        mean,cov=fit_joint(y[fit],z[fit]);my,mz,a,_=mean4_head(mean,cov)
        prediction[held]=my+(z[held].mean(1)-mz)@a.T
    yc=y[fitmask];error=yc-prediction[fitmask]
    gain=(yc.T@yc-error.T@error)/len(yc);gain=(gain+gain.T)/2
    eigen,vectors=np.linalg.eigh(gain);k=cfg['rank'];p=vectors[:,-k:].T;low=vectors[:,:k].T
    mean,cov=fit_joint(y[fitmask],z[fitmask]);my,mz,a,_=mean4_head(mean,cov)
    ym=y[calmask];pred=my+(z[calmask].mean(1)-mz)@a.T
    gain_per=(ym@p.T)**2-((ym-pred)@p.T)**2
    diagnostic=dict(status='completed',direction_fit_concepts=fit_concepts.tolist(),diagnostic_calibration_concepts=cal_concepts.tolist(),
        all_gain_eigenvalues=eigen.tolist(),selected_rank=k,calibration_projected_gain_per_image=gain_per.sum(1).tolist(),
        calibration_mean_projected_gain=float(gain_per.sum(1).mean()),positive_gain_directions=int((eigen>0).sum()),
        encoder_and_PCA_previously_fit_all_training_concepts=True,not_independent_encoder_validation=True)
    write(out/'direction_calibration.json',diagnostic)
    assert digest(cfg['old_joint_model'])==cfg['old_joint_model_sha256'];f=np.load(cfg['old_joint_model']);mean=f['mean'];cov=f['covariance']
    my,mz,a,c=mean4_head(mean,cov);c=(c+c.T)/2;prior=cov[:64,:64]
    qraw=np.load(old/'EEG_only_query_fusions.npz')['single_trials']
    base=json.loads(Path('/root/avde_rime_20261015/native680_stage2_v1.json').read_text())
    meta=torch.load(base['generation_query_packet'],map_location='cpu',weights_only=True)
    qtrials=project(qraw,meta['session_ids'].numpy());qmean=query['z'].numpy().astype(np.float64)
    assert np.allclose(qtrials.mean(1),qmean,atol=5e-6)
    q,beta=budget_tilt(c,p,cfg['KL_to_conditional_reference_nats_per_vector'])
    randomp=np.linalg.qr(np.random.default_rng(cfg['random_subspace_seed']).normal(size=(64,k)))[0].T
    randomq,randomb=budget_tilt(c,randomp,4.);lowq,lowb=budget_tilt(c,low,4.)
    ru=uniform_kl_scale(64,4.);rt=np.trace(q)/np.trace(c)
    re=np.exp((np.linalg.slogdet(q)[1]-np.linalg.slogdet(c)[1])/64)
    covariances=dict(mean4_exact=c,selective_K4=q,uniform_K4=ru*c,uniform_trace=rt*c,
        uniform_entropy=re*c,random_K4=randomq,lowgain_K4=lowq,prior_exact=prior)
    fibers={};model={};stats={}
    for name,value in covariances.items():
        assert np.linalg.eigvalsh(value).min()>0
        model['cov_'+name]=value
        stats[name]=dict(trace=float(np.trace(value)),logdet=float(np.linalg.slogdet(value)[1]),
            KL_to_mean4_reference_covariance_component=kl_cov(value,c),projected_conditional_variance=float(np.trace(p@value@p.T)))
        if name not in ['prior_exact']:
            fibers[name]=fiber_check(c,value,p);stats[name]['KL_to_mean4_reference']=kl_cov(value,c)
    assert max(fibers['selective_K4'].values())<1e-9
    assert abs(stats['selective_K4']['trace']-stats['uniform_trace']['trace'])<1e-9
    assert abs(stats['selective_K4']['logdet']-stats['uniform_entropy']['logdet'])<1e-9
    assert abs(stats['selective_K4']['KL_to_mean4_reference']-stats['uniform_K4']['KL_to_mean4_reference'])<1e-9
    subsetheads={}
    for sub in [(i,) for i in range(4)]+[(0,1,2,3)]:
        idx=np.concatenate([np.arange(64+64*i,128+64*i) for i in sub])
        aa=np.linalg.solve(cov[np.ix_(idx,idx)],cov[idx,:64]).T;cc=cov[:64,:64]-aa@cov[idx,:64]
        subsetheads[sub]=(idx,aa,(cc+cc.T)/2)
    covariances['fulljoint_exact']=subsetheads[(0,1,2,3)][2];model['cov_fulljoint_exact']=covariances['fulljoint_exact']
    model.update(prior_mean=my,EEG_mean=mz,mean4_regression=a,projection=p,lowgain_projection=low,random_projection=randomp)
    np.savez_compressed(out/'training_only_models.npz',**model)
    np.savez_compressed(out/'EEG_only_query_coordinates.npz',z=qmean,trial_ids=query['trial_ids'].numpy(),query_ids=query['query_ids'].numpy(),R=4)
    write(out/'distribution_contract.json',dict(status='completed',conditional_reference_covariances=stats,
        selective_beta=beta,random_beta=randomb,lowgain_beta=lowb,uniform_scales=dict(KL=ru,trace=rt,entropy=re),
        fiber_preservation_checks=fibers,KL_units='nats per64-dimensional latent vector, not VAR tokens',
        data_manifest_sha256=digest(prepared/'data_manifest.json'),old_joint_model_sha256=cfg['old_joint_model_sha256']))
    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',clip_sample=False,prediction_type='v_prediction');scheduler.set_timesteps(50)
    alphas=scheduler.alphas_cumprod.numpy().astype(np.float64);inverses={}
    for tt in scheduler.timesteps.tolist():
        for name,cc in [('mean',c),('prior',prior)]+[(str(i),subsetheads[(i,)][2]) for i in range(4)]:
            inverses[tt,name]=np.linalg.inv(alphas[tt]*cc+(1-alphas[tt])*np.eye(64))
    outputs={};means={};begin=time.monotonic();max_endpoint_error=0.
    for condition in ['paired','permuted','missing']:
        qq=qmean if condition=='paired' else np.roll(qmean,1,axis=0) if condition=='permuted' else np.zeros_like(qmean)
        qt=qtrials if condition=='paired' else np.roll(qtrials,1,axis=0) if condition=='permuted' else np.zeros_like(qtrials)
        mu=my+(qq-mz)@a.T;fullflat=qt.reshape(200,-1);submus={}
        for sub,(idx,aa,cc) in subsetheads.items():submus[sub]=my+(fullflat[:,idx-64]-mean[idx])@aa.T
        for method in cfg['methods']:
            if method=='prior_exact' and condition!='paired':continue
            for seed in cfg['generation_seeds']:
                noise=np.random.default_rng(seed).standard_normal((200,64))
                if method=='mean_decision':value=mu.copy()
                elif method in covariances:
                    mm=np.broadcast_to(my,(200,64)) if method=='prior_exact' else submus[(0,1,2,3)] if method=='fulljoint_exact' else mu
                    value=exact_gaussian_vp_flow(mm,covariances[method],noise,50)
                    coarse=exact_gaussian_vp_flow(mm,covariances[method],noise,1)
                    max_endpoint_error=max(max_endpoint_error,float(abs(value-coarse).max()))
                    means[method+'/'+('prior' if method=='prior_exact' else condition)]=mm
                else:
                    state=torch.from_numpy(noise.copy())
                    for tt in scheduler.timesteps:
                        t=int(tt);aa=alphas[t];ax=np.sqrt(aa);sig=np.sqrt(1-aa);xx=state.numpy()
                        s0=-(xx-ax*my)@inverses[t,'prior'].T
                        sc=-(xx-ax*mu)@inverses[t,'mean'].T
                        if method=='mean4_DDIM50':score=sc
                        elif method=='cfg2_DDIM50':score=2*sc-s0
                        else:
                            assert method=='naiveproduct_DDIM50'
                            score=sum(-(xx-ax*submus[(i,)])@inverses[t,str(i)].T for i in range(4))-3*s0
                        velocity=torch.from_numpy((-sig*score-sig*xx)/ax)
                        state=scheduler.step(velocity,tt,state,eta=0.).prev_sample
                    value=state.numpy()
                assert np.isfinite(value).all()
                key=method+'/'+('prior' if method=='prior_exact' else condition)+'/'+str(seed);outputs[key]=value
                if method=='mean_decision':means[method+'/'+condition]=mu
    assert max_endpoint_error<1e-10
    np.savez_compressed(out/'EEG_only_generated_latents.npz',**outputs);np.savez_compressed(out/'conditional_means.npz',**means)
    report.update(status='completed',stage='all_frozen_methods_EEG_only_generated',methods=cfg['methods'],query_count=200,R=4,
        generation_seeds=cfg['generation_seeds'],actual_generated_semantic_vectors=len(outputs)*200,actual_images_generated=0,
        actual_optimizer_updates=0,development_target_gallery_read=False,official_test_read=False,
        direction_calibration_gain=diagnostic['calibration_mean_projected_gain'],generation_seconds=time.monotonic()-begin,
        exact_flow_1_vs50_max_error=max_endpoint_error,sample_sha256=digest(out/'EEG_only_generated_latents.npz'),
        reference_model_sha256=digest(out/'training_only_models.npz'),working_Gaussian_only=True,method_innovation_verified=False)


def evaluate(cfg,out,report):
    import numpy as np
    import torch
    assert not torch.cuda.is_available();torch.set_num_threads(2)
    producer=Path(cfg['out']);g=json.loads((producer/'results.json').read_text());assert g['status']=='completed'
    ledger=[json.loads(x) for x in Path('/root/rime_var_20261015/artifacts/execution_ledger.jsonl').read_text().splitlines()]
    last=[x for x in ledger if x['experiment']=='semantic_fiber_generation_CPU_v1'][-1];assert last['status']=='completed' and last['exit_code']==0
    assert digest(producer/'EEG_only_generated_latents.npz')==g['sample_sha256']
    prepared=Path(cfg['prepared']);base=json.loads((prepared/'config.json').read_text());s1=Path(base['stage1'])
    split=json.loads((s1/'split_manifest.json').read_text());s1cfg=json.loads((s1/'config.json').read_text())
    features=torch.load(s1cfg['features'],map_location='cpu',weights_only=True)['img_features'][split['development_query_image_ids']].float()
    features=torch.nn.functional.normalize(features,dim=-1).numpy().astype(np.float64);tf=np.load(prepared/'training_only_transforms.npz')
    target=(features-tf['y_mean'])@tf['y_projection'];model=np.load(producer/'training_only_models.npz')
    samples=np.load(producer/'EEG_only_generated_latents.npz');means=np.load(producer/'conditional_means.npz')
    summary={};vectors={};matrices={};all_ranks={};diversity={};all_cosines={}
    conditions={key.rsplit('/',1)[0] for key in samples.files}
    for name in sorted(conditions):
        wins=[];ranklist=[];cosines=[];mse=[];generated=[]
        for seed in cfg['generation_seeds']:
            y=samples[name+'/'+str(seed)];generated.append(y)
            reconstructed=tf['y_mean']+(y*tf['y_scale'])@tf['y_basis'].T;reconstructed/=np.linalg.norm(reconstructed,axis=1,keepdims=True)
            sim=features@reconstructed.T;comparison=(np.diag(sim)[None,:]>sim).astype(float);np.fill_diagonal(comparison,0)
            wins.append(comparison);ranklist.append((sim>=np.diag(sim)[None,:]).sum(0));cosines.append(np.diag(sim));mse.append(np.mean((y-target)**2,axis=1))
        matrices[name]=np.mean(wins,axis=0);ranks=np.stack(ranklist);cosine=np.stack(cosines).mean(0);mse=np.stack(mse).mean(0)
        vectors[name]=dict(cosine=cosine.tolist(),PCA64_MSE=mse.tolist(),per_seed_ranks=ranks.tolist())
        all_ranks[name]=ranks;all_cosines[name]=cosine
        distances=np.stack([np.mean((generated[i]-generated[j])**2,axis=1) for i,j in itertools.combinations(range(3),2)]).mean(0)
        diversity[name]=distances.tolist()
        summary[name]=dict(GeneratedSemanticLatentCosineTwoWay=float(matrices[name].sum()/(200*199)),
            Recall1=float((ranks==1).mean()),Recall5=float((ranks<=5).mean()),matched_cosine=float(cosine.mean()),PCA64_MSE=float(mse.mean()),
            measured_three_seed_latent_pairwise_squared_distance_per_dimension=float(distances.mean()))
        method=name.split('/')[0]
        if 'cov_'+method in model:
            c=model['cov_'+method];delta=target-means[name];nll=.5*(64*np.log(2*np.pi)+np.linalg.slogdet(c)[1]+np.einsum('ni,ij,nj->n',delta,np.linalg.inv(c),delta))/64
            vectors[name]['working_latent_NLL_nats_per_dimension']=nll.tolist();summary[name]['working_latent_NLL_nats_per_dimension']=float(nll.mean())
        elif method=='mean_decision':summary[name]['working_latent_NLL_nats_per_dimension']=None
    sys.path.insert(0,'/root/avde_reproduction_20261016');from native_statistics import concept_bootstrap_counts,paired_identification_interval,paired_query_interval
    counts=concept_bootstrap_counts(200,repetitions=cfg['bootstrap_repetitions'],seed=cfg['bootstrap_seed']);comparisons={}
    for name in cfg['planned_contrasts']:
        right='selective_K4/permuted' if name=='selective_K4_pairing' else name.replace('selective_K4_minus_','')+'/paired'
        left='selective_K4/paired';entry=dict(GeneratedSemanticLatentCosineTwoWay=paired_identification_interval(matrices[left],matrices[right],counts))
        # No neural training seeds in this closed-form model. Fix generic helper metadata.
        entry['GeneratedSemanticLatentCosineTwoWay'].update(independent_training_seeds=0,model_fit='closed_form')
        for metric in ['cosine','PCA64_MSE']:
            entry[metric]=paired_query_interval(vectors[left][metric],vectors[right][metric],counts)
        entry['Recall1']=paired_query_interval((all_ranks[left]==1).mean(0),(all_ranks[right]==1).mean(0),counts)
        entry['latent_diversity']=paired_query_interval(diversity[left],diversity[right],counts)
        comparisons[name]=entry
    write(out/'per_query_metrics.json',dict(target_image_ids=split['development_query_image_ids'],values=vectors,diversity=diversity))
    np.savez_compressed(out/'comparison_matrices.npz',**matrices)
    report.update(status='completed',stage='all_methods_all200_queries_evaluated',summary=summary,comparisons=comparisons,
        primary_contrast=cfg['primary_contrast'],training_seeds=0,generation_seeds=cfg['generation_seeds'],
        statistics='Exploratory: average metrics, never predictions, across3 seeds; 200 concepts query/distractor axes jointly bootstrapped; reused development set; no multiplicity correction',
        definitions='Semantic-latent cosine identification; CLIP is training target; not Inception, generated image identification, independent evaluator or token NLL',
        actual_images_evaluated=0,official_test_read=False,full_ATM_reproduction=False,image_SOTA_established=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['generate','evaluate'],required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(exist_ok=False);begin=time.monotonic()
    report=dict(status='running',mode=args.mode,source_sha256=digest(__file__),configuration_sha256=digest(args.config),official_test_read=False)
    write(out/'preregistration.json',dict(**report,config=cfg));write(out/'progress.json',report)
    try:
        if args.mode=='generate':assert str(out)==cfg['out'];generate(cfg,out,report)
        else:evaluate(cfg,out,report)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,message=str(error));raise
    finally:
        report.update(seconds=time.monotonic()-begin);write(out/'results.json',report);write(out/'progress.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
