"""Posthoc factorial mechanism audit; no new learned parameters or selected seeds."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,'/root/avde_rime_20261015')
from run_fast_learning import digest,write


def generate(cfg,out,report):
    import numpy as np
    import torch
    from diffusers import DDIMScheduler
    from semantic_fiber_math import sqrt_spd
    from run_semantic_fiber import mean4_head
    assert not torch.cuda.is_available();torch.set_num_threads(2)
    parent=Path(cfg['out']);model=np.load(parent/'training_only_models.npz');original=np.load(parent/'EEG_only_generated_latents.npz')
    query=np.load(parent/'EEG_only_query_coordinates.npz')['z'];c=model['cov_mean4_exact'];base_root=sqrt_spd(c)
    f=np.load(cfg['old_joint_model']);mean=f['mean'];cov=f['covariance'];my,mz,a,_=mean4_head(mean,cov)
    prepared=Path(cfg['prepared']);basecfg=json.loads((prepared/'config.json').read_text());old=Path(basecfg['repeat_run'])
    raw=np.load(old/'EEG_only_query_fusions.npz')['single_trials'].astype(np.float64);cal=np.load(old/'calibration.npz');tf=np.load(prepared/'training_only_transforms.npz')
    gc=json.loads(Path('/root/avde_rime_20261015/native680_stage2_v1.json').read_text());meta=torch.load(gc['generation_query_packet'],map_location='cpu',weights_only=True)
    for session,bias in zip(cal['levels'],cal['bias']):raw[meta['session_ids'].numpy()==session]-=bias
    trials=(raw-tf['z_mean'])@tf['z_projection'];heads={}
    for i in range(4):
        idx=np.arange(64+i*64,128+i*64);aa=np.linalg.solve(cov[np.ix_(idx,idx)],cov[idx,:64]).T
        heads[i]=(idx,aa,cov[:64,:64]-aa@cov[idx,:64])
    scheduler=DDIMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',clip_sample=False,prediction_type='v_prediction');scheduler.set_timesteps(50)
    outputs={};models={};replay_errors={}
    for condition in ['paired','permuted','missing']:
        q=query if condition=='paired' else np.roll(query,1,0) if condition=='permuted' else np.zeros_like(query)
        qt=trials if condition=='paired' else np.roll(trials,1,0) if condition=='permuted' else np.zeros_like(trials)
        mu=my+(q-mz)@a.T;submu={i:my+(qt[:,i]-mean[idx])@aa.T for i,(idx,aa,cc) in heads.items()}
        for kind in ['cfg','product']:
            j=np.eye(64);b=np.zeros((200,64))
            for tt in scheduler.timesteps:
                t=int(tt);alpha=float(scheduler.alphas_cumprod[t]);ax=np.sqrt(alpha);sig=np.sqrt(1-alpha)
                p0=np.linalg.inv(alpha*cov[:64,:64]+(1-alpha)*np.eye(64));pc=np.linalg.inv(alpha*c+(1-alpha)*np.eye(64))
                if kind=='cfg':precision=2*pc-p0;meanterm=2*mu@pc.T-my@p0.T
                else:
                    invs={i:np.linalg.inv(alpha*cc+(1-alpha)*np.eye(64)) for i,(_,_,cc) in heads.items()}
                    precision=sum(invs.values())-3*p0;meanterm=sum(submu[i]@invs[i].T for i in range(4))-3*my@p0.T
                vjac=sig/ax*(precision-np.eye(64))
                # Use the actual scheduler to preserve its FP32 coefficient rounding.
                step_j=scheduler.step(torch.from_numpy(vjac.T.copy()),tt,torch.eye(64,dtype=torch.float64),eta=0.).prev_sample.numpy().T
                step_b=scheduler.step(torch.from_numpy(-sig*meanterm),tt,torch.zeros((200,64),dtype=torch.float64),eta=0.).prev_sample.numpy()
                j=step_j@j;b=b@step_j.T+step_b
            models[kind+'_noise_jacobian']=j;models[kind+'_mean_'+condition]=b
            for seed in cfg['generation_seeds']:
                noise=np.random.default_rng(seed).standard_normal((200,64))
                outputs[kind+'_mean_only/'+condition+'/'+str(seed)]=b+noise@base_root
                outputs[kind+'_cov_only/'+condition+'/'+str(seed)]=mu+noise@j.T
                both=b+noise@j.T;outputs[kind+'_both_replay/'+condition+'/'+str(seed)]=both
                source=('cfg2_DDIM50' if kind=='cfg' else 'naiveproduct_DDIM50')+'/'+condition+'/'+str(seed)
                err=float(abs(both-original[source]).max());assert err<1e-10;replay_errors[source]=err
        for seed in cfg['generation_seeds']:
            noise=np.random.default_rng(seed).standard_normal((200,64))
            outputs['scalar_mean_gain2/'+condition+'/'+str(seed)]=my+2*(mu-my)+noise@base_root
        models['reference_mean_'+condition]=mu
    models['reference_covariance']=c
    np.savez_compressed(out/'generated_latents.npz',**outputs);np.savez_compressed(out/'affine_models.npz',**models)
    mean_shift={kind:float(np.mean((models[kind+'_mean_paired']-models['reference_mean_paired'])**2)) for kind in ['cfg','product']}
    report.update(status='completed',stage='all_factorial_controls_generated',actual_new_optimizer_updates=0,
        actual_new_images=0,query_count=200,R=4,generation_seeds=cfg['generation_seeds'],
        mean_field_shift_MSE=mean_shift,replay_max_error=max(replay_errors.values()),
        reference_variance_trace=float(np.trace(c)),variance_traces={kind:float(np.sum(models[kind+'_noise_jacobian']**2)) for kind in ['cfg','product']},
        development_targets_gallery_read=False,reference_samples_sha256=digest(parent/'EEG_only_generated_latents.npz'),
        sample_sha256=digest(out/'generated_latents.npz'),working_gaussian_mechanism_only=True)


def evaluate(cfg,out,report):
    import numpy as np
    import torch
    assert not torch.cuda.is_available();torch.set_num_threads(2)
    prod=Path('/root/semantic_mean_cov_generation_20261017_v1')
    r=json.loads((prod/'results.json').read_text());assert r['status']=='completed' and r['replay_max_error']<1e-10
    ledger=[json.loads(x) for x in Path('/root/rime_var_20261015/artifacts/execution_ledger.jsonl').read_text().splitlines()]
    receipt=[x for x in ledger if x['experiment']=='semantic_mean_cov_generation_CPU_v1'][-1];assert receipt['status']=='completed' and receipt['exit_code']==0
    prepared=Path(cfg['prepared']);base=json.loads((prepared/'config.json').read_text());s1=Path(base['stage1']);split=json.loads((s1/'split_manifest.json').read_text());s1cfg=json.loads((s1/'config.json').read_text())
    target=torch.load(s1cfg['features'],map_location='cpu',weights_only=True)['img_features'][split['development_query_image_ids']].float()
    target=torch.nn.functional.normalize(target,dim=-1).numpy().astype(np.float64);tf=np.load(prepared/'training_only_transforms.npz')
    ytrue=(target-tf['y_mean'])@tf['y_projection'];samples=np.load(prod/'generated_latents.npz');models=np.load(prod/'affine_models.npz')
    old=Path('/root/semantic_fiber_evaluation_20261017_v1');oldmat=np.load(old/'comparison_matrices.npz');oldr=json.loads((old/'results.json').read_text())
    oldv=json.loads((old/'per_query_metrics.json').read_text())['values'];summary={};matrices={};vectors={};ranks={};diversity={}
    for name in sorted({k.rsplit('/',1)[0] for k in samples.files}):
        values=[samples[name+'/'+str(seed)] for seed in cfg['generation_seeds']];wins=[];cosine=[];rank=[]
        for value in values:
            image=tf['y_mean']+(value*tf['y_scale'])@tf['y_basis'].T;image/=np.linalg.norm(image,axis=1,keepdims=True)
            sim=target@image.T;win=(np.diag(sim)[None,:]>sim).astype(float);np.fill_diagonal(win,0);wins.append(win)
            cosine.append(np.diag(sim));rank.append((sim>=np.diag(sim)[None,:]).sum(0))
        matrices[name]=np.mean(wins,0);ranks[name]=np.stack(rank)
        vectors[name]=dict(cosine=np.mean(cosine,0).tolist(),PCA64_MSE=np.mean([np.mean((value-ytrue)**2,1) for value in values],0).tolist(),per_seed_ranks=np.stack(rank).tolist())
        distances=np.mean([np.mean((values[i]-values[j])**2,1) for i,j in itertools.combinations(range(3),2)],0);diversity[name]=distances
        method,condition=name.split('/');kind=method.split('_')[0]
        mean=models[kind+'_mean_'+condition] if method.endswith('mean_only') or method.endswith('both_replay') else models['reference_mean_'+condition]
        if method=='scalar_mean_gain2':
            f=np.load(cfg['old_joint_model']);my=f['mean'][:64];mean=my+2*(models['reference_mean_'+condition]-my)
        c=models['reference_covariance'] if method.endswith('mean_only') or method=='scalar_mean_gain2' else models[kind+'_noise_jacobian']@models[kind+'_noise_jacobian'].T
        delta=ytrue-mean;nll=.5*(64*np.log(2*np.pi)+np.linalg.slogdet(c)[1]+np.einsum('ni,ij,nj->n',delta,np.linalg.inv(c),delta))/64
        vectors[name]['working_latent_NLL_nats_per_dimension']=nll.tolist()
        summary[name]=dict(GeneratedSemanticLatentCosineTwoWay=float(matrices[name].sum()/(200*199)),Recall1=float((ranks[name]==1).mean()),Recall5=float((ranks[name]<=5).mean()),matched_cosine=float(np.mean(cosine)),
            latent_diversity=float(distances.mean()),working_latent_NLL_nats_per_dimension=float(nll.mean()),analytic_conditional_variance_trace=float(np.trace(c)))
        if method.endswith('both_replay'):
            ref=('cfg2_DDIM50' if kind=='cfg' else 'naiveproduct_DDIM50')+'/'+condition
            assert np.array_equal(matrices[name],oldmat[ref]);assert np.array_equal(ranks[name],np.array(oldv[ref]['per_seed_ranks']))
    sys.path.insert(0,'/root/avde_reproduction_20261016');from native_statistics import concept_bootstrap_counts,paired_identification_interval,paired_query_interval
    counts=concept_bootstrap_counts(200,repetitions=20000,seed=2026101666);comparisons={}
    pairs=[]
    for kind in ['cfg','product']:
        pairs += [(kind+'_mean_only_minus_reference',kind+'_mean_only/paired','mean4_exact/paired'),
            (kind+'_cov_only_minus_reference',kind+'_cov_only/paired','mean4_exact/paired'),
            (kind+'_mean_only_minus_full',kind+'_mean_only/paired',kind+'_both_replay/paired'),
            (kind+'_mean_only_pairing',kind+'_mean_only/paired',kind+'_mean_only/permuted')]
    pairs += [('scalar_gain2_minus_reference','scalar_mean_gain2/paired','mean4_exact/paired')]
    for name,l,rr in pairs:
        rm=matrices[rr] if rr in matrices else oldmat[rr]
        ri=paired_identification_interval(matrices[l],rm,counts);ri.update(independent_training_seeds=0,model_fit='closed_form')
        entry=dict(GeneratedSemanticLatentCosineTwoWay=ri)
        rv=vectors[rr] if rr in vectors else oldv[rr]
        for metric in ['cosine','PCA64_MSE']:entry[metric]=paired_query_interval(vectors[l][metric],rv[metric],counts)
        comparisons[name]=entry
    write(out/'per_query_metrics.json',dict(values=vectors,diversity={k:v.tolist() for k,v in diversity.items()}));np.savez_compressed(out/'comparison_matrices.npz',**matrices)
    report.update(status='completed',stage='factorial_mechanisms_all200_evaluated',summary=summary,comparisons=comparisons,
        original_guidance_matrices_and_ranks_bitwise_reproduced=True,actual_new_images=0,actual_new_optimizer_updates=0,
        bootstrap_seed=2026101666,bootstrap_unit='200 concepts, query and distractor axes jointly; metrics averaged across3 seeds',
        posthoc=True,independent_evaluator=False,novel_method_or_SOTA_proven=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['generate','evaluate'],required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(exist_ok=False);start=time.monotonic()
    report=dict(status='running',posthoc=True,mode=args.mode,source_sha256=digest(__file__),official_test_read=False)
    write(out/'preregistration.json',dict(**report,config=cfg));write(out/'progress.json',report)
    try:
        if args.mode=='generate':generate(cfg,out,report)
        else:evaluate(cfg,out,report)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,message=str(error));raise
    finally:
        report.update(seconds=time.monotonic()-start);write(out/'results.json',report);write(out/'progress.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
