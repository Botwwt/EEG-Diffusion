"""Measured full-query SORT endpoint pilot; every stage has explicit IO.

No SDXL, candidate image selection, official test, or development supervision in
generation/transport. Generation retains the native raw-scale prior condition.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch
from scipy.special import softmax
from sort_math import unit,anchor_directions,repeat_scale,conformal_scores,conformal_quantile,boundaries,project_intersection,random_rank_directions


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(path)


def now():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def condition(x,name):return np.roll(x,1,axis=0) if name=='permuted' else x


def verified_asset(directory,name):
    directory=Path(directory);manifest=load(directory/'asset_manifest.json')
    assert digest(directory/name)==manifest[name]['sha256'],name
    return directory/name


def prepare(cfg,out,report):
    assert not torch.cuda.is_available(),'CPU-only IO stage'
    eroot=Path(cfg['encoder']);em=load(eroot/'EEG_only_cache_manifest.json');sp=load(eroot/'split_manifest.json')
    source=load(cfg['data_config']);assets=load(Path(source['cache_validation'])/'asset_manifest.json')
    assert digest(assets['features']['path'])==assets['features']['sha256']
    raw=torch.load(assets['features']['path'],map_location='cpu',weights_only=True)['img_features'].float().numpy()
    bank_ids=np.asarray(em['caches']['fit']['image_ids']);assert len(bank_ids)==11630
    groups={}
    rng=np.random.default_rng(cfg['partition_seed']);cal=np.sort(np.asarray(sp['calibration_concepts']))
    order=rng.permutation(cal);tune=np.sort(order[:cfg['tuning_concepts']]);remain=np.sort(order[cfg['tuning_concepts']:])
    assert (len(tune),len(remain))==(145,146)
    assert not set(tune)&set(remain) and not set(cal)&set(sp['development_concepts'])
    arrays={};manifest={}
    for group,cache_name,ids in [('tuning','calibration',tune*10),('calibration','calibration',remain*10),
                               ('development','development',np.asarray(em['caches']['development']['image_ids']))]:
        cache=em['caches'][cache_name];lookup={int(i):j for j,i in enumerate(cache['image_ids'])};idx=np.array([lookup[int(i)] for i in ids])
        for kind in ['waveform_mean','single_trials']:
            f=cache[kind];assert digest(f['path'])==f['sha256']
            value=np.array(np.load(f['path'],mmap_mode='r')[idx]);assert np.isfinite(value).all()
            arrays[group+'/'+('e0' if kind=='waveform_mean' else 'trials')]=value
        arrays[group+'/image_ids']=ids
        manifest[group]=dict(image_ids=ids.tolist(),concepts=(ids//10).tolist(),queries=len(ids),R=4)
        np.savez_compressed(out/(group+'_targets_EVALUATION_OR_CALIBRATION_ONLY.npz'),raw=raw[ids],unit=unit(raw[ids]).astype('float32'),image_ids=ids)
    fit_cache=em['caches']['fit']['waveform_mean'];assert digest(fit_cache['path'])==fit_cache['sha256']
    a_fit=unit(np.load(fit_cache['path'],mmap_mode='r'));bank=unit(raw[bank_ids])
    floor=max(1e-4,float(np.sqrt(np.mean(np.sum((a_fit-bank)**2,axis=1))/1024)))
    np.save(out/'fit_bank_unit.npy',bank.astype('float32'));np.savez_compressed(out/'EEG_ONLY.npz',**arrays)
    assert not set(bank_ids//10)&set(cal) and not set(bank_ids//10)&set(sp['development_concepts'])
    contract=dict(status='completed',created_UTC=now(),fit_bank_images=11630,fit_concepts=1163,
        fit_bank_original_ids=bank_ids.tolist(),groups=manifest,encoder_sha256=em['encoder_sha256'],source_cache_manifest_sha256=digest(eroot/'EEG_only_cache_manifest.json'),
        fit_scale_floor=floor,scale='sqrt(population trial directional dispersion + fit-only residual_floor²); no1/R',
        bank_norm_median=float(np.median(np.linalg.norm(raw[bank_ids],axis=1))),
        preprocessing='Published250 samples; upstream train-only whitening and event independence unresolved',
        historical_calibration_and_dev_reuse=True,formal_coverage_claimed=False,official_test_read=False,
        target_cache_only_read_in_prepare_calibrate_evaluate=True)
    write(out/'data_contract.json',contract)
    write(out/'asset_manifest.json',{p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir()
        if p.is_file() and (p.suffix in ['.npy','.npz'] or p.name=='data_contract.json')})
    report.update(status='completed',groups={k:v['queries'] for k,v in manifest.items()},fit_scale_floor=floor,new_optimizer_updates=0)


def generate(cfg,out,report):
    assert torch.cuda.is_available();base=Path(cfg['out'])/'prepare';packet=np.load(verified_asset(base,'EEG_ONLY.npz'));contract=load(verified_asset(base,'data_contract.json'))
    from paper_main_runtime import load_prior
    from native_prior_witness import enable_native_batch_sampler
    prior_root=Path(cfg['prior']);trained=load(prior_root/'results.json')
    weights=prior_root/'prior_final_150.pth';assert digest(weights)==trained['weight_sha256'] and trained['optimizer_updates']==6900
    module=load_prior(Path(cfg['source']));model=module.DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda().eval()
    model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True),strict=True)
    pipe=module.Pipe(model,device='cuda');enable_native_batch_sampler(pipe)
    assert pipe.scheduler.config.clip_sample
    stored={};torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
    with torch.no_grad():
        for group in ['tuning','calibration','development']:
            eeg=np.array(packet[group+'/e0']);assert eeg.shape==(contract['groups'][group]['queries'],1024)
            for name in ['paired','permuted']:
                c=torch.from_numpy(condition(eeg,name)).cuda()
                for seed in cfg['generation_seeds']:
                    value=pipe.generate(c_embeds=c,num_inference_steps=50,guidance_scale=5.,generator=torch.Generator(device='cuda').manual_seed(seed))
                    assert value.shape==c.shape and torch.isfinite(value).all()
                    stored[group+'/'+name+'/'+str(seed)]=value.cpu().numpy()
            report.update(group_completed=group,seconds_sampling=time.monotonic()-tick);write(out/'progress.json',report)
    np.savez_compressed(out/'PRIOR_EEG_ONLY.npz',**stored)
    report.update(status='completed',prior_weight_sha256=digest(weights),samples_sha256=digest(out/'PRIOR_EEG_ONLY.npz'),
        actual_stochastic_vectors=sum(len(x) for x in stored.values()),new_images=0,prior_conditions='Unchanged native raw e0, DDPM50, clipTrue, CFG5',
        generation_seeds=cfg['generation_seeds'],training_seed=cfg['training_seed'],
        targets_or_evaluation_gallery_read=False,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())


def geometry(cfg,packet,group,name,bank,floor,random=False):
    a=unit(condition(packet[group+'/e0'],name));trials=condition(packet[group+'/trials'],name)
    d,mu,valid,pairs=anchor_directions(a,bank)
    if random:d,mu,valid,_=random_rank_directions(a,d,seed=cfg['random_direction_seed'])
    sigma=repeat_scale(d,trials,floor)
    return a,d,mu,valid,sigma,pairs


def gating_features(a,v,raw,trial):
    disp=np.sqrt(np.mean((unit(trial)-unit(trial).mean(1,keepdims=True))**2,axis=(1,2)))
    return np.column_stack([np.ones(len(a)),np.sum(a*v,1),np.log(np.maximum(np.linalg.norm(raw,axis=1),1e-12)),disp])


def rank_metrics(feature,target):
    sim=unit(target)@unit(feature).T
    diag=np.diag(sim);rank=(sim>=diag[None,:]).sum(0)
    win=(diag[None,:]>sim).astype(float);np.fill_diagonal(win,0)
    assert np.array_equal(win.sum(0),len(rank)-rank)
    return rank,win,diag


def fit_controls(cfg,packet,samples,target,bank):
    a=unit(packet['tuning/e0']);trials=packet['tuning/trials'];raws=[samples['tuning/paired/'+str(s)] for s in cfg['generation_seeds']]
    vs=[unit(z) for z in raws];tuning=[]
    for lam in cfg['mixture_grid']:
        values=[float((rank_metrics(unit((1-lam)*v+lam*a),target)[0]==1).mean()) for v in vs]
        tuning.append(dict(lam=lam,Recall1=float(np.mean(values))))
    # Smallest lambda among tied best values; no development metric is involved.
    selected=max(tuning,key=lambda x:(x['Recall1'],-x['lam']))['lam']
    features=np.concatenate([gating_features(a,v,z,trials) for v,z in zip(vs,raws)])
    center=features.mean(0);scale=features.std(0);center[0]=0.;scale[0]=1.;scale=np.maximum(scale,1e-8)
    ft=torch.tensor((features-center)/scale,dtype=torch.float64)
    sv=torch.tensor(np.concatenate([v@unit(target).T for v in vs]),dtype=torch.float64)
    sa=torch.tensor(np.tile(a@unit(target).T,(3,1)),dtype=torch.float64)
    dot=torch.tensor(np.concatenate([np.sum(a*v,1) for v in vs]),dtype=torch.float64)
    labels=torch.arange(len(a)).repeat(3);torch.manual_seed(cfg['training_seed'])
    weight=torch.zeros(4,dtype=torch.float64,requires_grad=True);opt=torch.optim.Adam([weight],lr=cfg['gate_learning_rate'])
    for step in range(cfg['gate_steps']):
        lam=torch.sigmoid(ft@weight)
        norm=torch.sqrt(torch.clamp((1-lam)**2+lam**2+2*lam*(1-lam)*dot,min=1e-12))
        logits=((1-lam[:,None])*sv+lam[:,None]*sa)/norm[:,None]/.07
        loss=torch.nn.functional.cross_entropy(logits,labels);opt.zero_grad();loss.backward();opt.step()
    assert all(int(s['step'])==cfg['gate_steps'] for s in opt.state.values())
    bank_s=a@bank.T;temperature_results=[]
    for temp in cfg['bank_temperature_grid']:
        u=unit(softmax(bank_s/temp,axis=1)@bank)
        temperature_results.append(dict(temperature=temp,Recall1=float((rank_metrics(u,target)[0]==1).mean())))
    temp=max(temperature_results,key=lambda x:(x['Recall1'],-x['temperature']))['temperature']
    return dict(status='completed',frozen_UTC=now(),mixture_lambda=selected,mixture_tuning=tuning,
        gate_weights=weight.detach().numpy().tolist(),gate_center=center.tolist(),gate_scale=scale.tolist(),
        gate_actual_Adam_updates=cfg['gate_steps'],gate_final_tuning_CE=float(loss.detach()),
        gate_features=['intercept','cos(a,v)','log raw prior norm','trial unit RMS dispersion'],
        bank_softmean_temperature=temp,bank_temperature_tuning=temperature_results,
        training_concepts=145,training_query_seed_states=435,calibration_or_development_targets_read=False)


def calibrate(cfg,out,report):
    assert not torch.cuda.is_available();base=Path(cfg['out']);prep=base/'prepare'
    packet=np.load(verified_asset(prep,'EEG_ONLY.npz'));assert digest(base/'generate/PRIOR_EEG_ONLY.npz')==load(base/'generate/results.json')['samples_sha256']
    samples=np.load(base/'generate/PRIOR_EEG_ONLY.npz')
    bank=unit(np.load(verified_asset(prep,'fit_bank_unit.npy')));contract=load(verified_asset(prep,'data_contract.json'));floor=contract['fit_scale_floor']
    tuning_targets=np.load(verified_asset(prep,'tuning_targets_EVALUATION_OR_CALIBRATION_ONLY.npz'))['unit']
    controls=fit_controls(cfg,packet,samples,tuning_targets,bank);write(out/'controls_frozen_before_calibration.json',controls)
    cal_targets=np.load(verified_asset(prep,'calibration_targets_EVALUATION_OR_CALIBRATION_ONLY.npz'))['unit']
    result={};scores={}
    for kind in ['repeat','constant','random']:
        a,d,mu,valid,sigma,pairs=geometry(cfg,packet,'calibration','paired',bank,floor,random=kind=='random')
        if kind=='constant':sigma=np.ones_like(sigma)*floor
        values=conformal_scores(d,mu,sigma,valid,cal_targets,rho=cfg['rho']);scores[kind]=values
        result[kind]={str(alpha):conformal_quantile(values,alpha) for alpha in [cfg['alpha_primary']]+cfg['alpha_auxiliary']}
    np.savez_compressed(out/'calibration_scores.npz',**scores)
    write(out/'calibration.json',dict(status='completed',frozen_UTC=now(),q=result,calibration_n=146,
        rho=cfg['rho'],formal_exchangeability_or_coverage_claimed=False,historical_reuse=True,
        controls_frozen_sha256=digest(out/'controls_frozen_before_calibration.json'),development_targets_read=False))
    report.update(status='completed',q=result,gate_Adam_updates=cfg['gate_steps'],formal_coverage=False,new_images=0)


def transport(cfg,out,report):
    assert not torch.cuda.is_available();base=Path(cfg['out']);prep=base/'prepare'
    packet=np.load(verified_asset(prep,'EEG_ONLY.npz'));assert digest(base/'generate/PRIOR_EEG_ONLY.npz')==load(base/'generate/results.json')['samples_sha256']
    samples=np.load(base/'generate/PRIOR_EEG_ONLY.npz')
    bank=unit(np.load(verified_asset(prep,'fit_bank_unit.npy')));contract=load(verified_asset(prep,'data_contract.json'));floor=contract['fit_scale_floor']
    calibration=load(base/'calibrate/calibration.json');controls=load(base/'calibrate/controls_frozen_before_calibration.json')
    assert digest(base/'calibrate/controls_frozen_before_calibration.json')==calibration['controls_frozen_sha256']
    outputs={};raw_norms={};mechanisms={};pair_ids={};total_fallback=0
    for group in ['tuning','calibration','development']:
        for name in ['paired','permuted']:
            a,d,mu,valid,sigma,pairs=geometry(cfg,packet,group,name,bank,floor)
            rd,rmu,rvalid,_=random_rank_directions(a,d,seed=cfg['random_direction_seed']);rsigma=repeat_scale(rd,condition(packet[group+'/trials'],name),floor)
            b0,m0=boundaries(mu,sigma,valid,calibration['q']['repeat'][str(cfg['alpha_primary'])],cfg['rho'])
            pair_ids[group+'/'+name]=pairs
            bank_mean=unit(softmax((a@bank.T)/controls['bank_softmean_temperature'],axis=1)@bank)
            rank=np.array([np.linalg.matrix_rank(x[m],tol=1e-8) if m.any() else 0 for x,m in zip(d,m0)])
            for seed in cfg['generation_seeds']:
                z=samples[group+'/'+name+'/'+str(seed)];v=unit(z)
                f=gating_features(a,v,z,condition(packet[group+'/trials'],name))
                f=(f-np.array(controls['gate_center']))/np.array(controls['gate_scale'])
                gate=1/(1+np.exp(-np.clip(f@np.array(controls['gate_weights']),-50,50)))
                lam=controls['mixture_lambda']
                values={'prior':v,'a0':a,'mixture':unit((1-lam)*v+lam*a),'gate':unit((1-gate[:,None])*v+gate[:,None]*a),'bank_softmean':bank_mean}
                solver={}
                arms=[('sort_repeat_a10',d,mu,valid,sigma,calibration['q']['repeat']['0.1']),
                      ('sort_repeat_a05',d,mu,valid,sigma,calibration['q']['repeat']['0.05']),
                      ('sort_repeat_a20',d,mu,valid,sigma,calibration['q']['repeat']['0.2']),
                      ('sort_constant_a10',d,mu,valid,np.ones_like(sigma)*floor,calibration['q']['constant']['0.1']),
                      ('sort_uncalibrated',d,mu,valid,sigma,0.),
                      ('sort_random_rank_a10',rd,rmu,rvalid,rsigma,calibration['q']['random']['0.1'])]
                for arm,D,M,V,S,Q in arms:
                    bounds,mask=boundaries(M,S,V,Q,cfg['rho'])
                    u,meta,pre=project_intersection(v,D,bounds,mask,a,max_cycles=cfg['projection_max_cycles'],tol=cfg['projection_tolerance'])
                    values[arm]=u;solver[arm]={k:x.tolist() for k,x in meta.items()};total_fallback+=int(meta['fallback'].sum())
                for method,u in values.items():
                    key=group+'/'+name+'/'+method+'/'+str(seed);u=np.asarray(u,dtype='float32');outputs[key]=u
                    # Keep the native prior sample norm for transport/control MSE;
                    # a0 retains its encoder norm. This is not an image interface.
                    raw_norms[key]=np.linalg.norm(condition(packet[group+'/e0'],name),axis=1) if method=='a0' else np.linalg.norm(z,axis=1)
                    margin=np.einsum('bld,bd->bl',d,unit(u))-b0
                    violation=np.maximum(np.where(m0,-margin,0.),0.).max(1)
                    mechanisms[key]=dict(active_edges=m0.sum(1).tolist(),active_rank=rank.tolist(),
                        primary_set_violation=violation.tolist(),movement_from_prior=np.linalg.norm(unit(u)-v,axis=1).tolist(),
                        active_edges_near_boundary=np.sum(m0 & (np.abs(margin)<1e-5),axis=1).tolist(),
                        gate_lambda=gate.tolist() if method=='gate' else None,solver=solver.get(method))
            report.update(last_completed_group=group,last_condition=name,solver_fallbacks=total_fallback)
            write(out/'progress.json',report)
    np.savez_compressed(out/'SORT_EEG_FITBANK_ONLY_ENDPOINTS.npz',**outputs)
    np.savez_compressed(out/'endpoint_norms.npz',**raw_norms);np.savez_compressed(out/'fit_anchor_pairs.npz',**pair_ids)
    write(out/'mechanisms.json',mechanisms)
    report.update(status='completed',methods=11,endpoint_vectors=sum(len(x) for x in outputs.values()),new_stochastic_prior_draws=0,
        endpoint_sha256=digest(out/'SORT_EEG_FITBANK_ONLY_ENDPOINTS.npz'),targets_or_evaluation_gallery_read=False,new_images=0,
        solver_fallbacks=total_fallback)


def evaluate(cfg,out,report):
    assert not torch.cuda.is_available();base=Path(cfg['out']);prep=base/'prepare';trans=base/'transport'
    assert digest(trans/'SORT_EEG_FITBANK_ONLY_ENDPOINTS.npz')==load(trans/'results.json')['endpoint_sha256']
    endpoints=np.load(trans/'SORT_EEG_FITBANK_ONLY_ENDPOINTS.npz');norms=np.load(trans/'endpoint_norms.npz');mechanisms=load(trans/'mechanisms.json')
    packet=np.load(prep/'EEG_ONLY.npz');contract=load(prep/'data_contract.json');bank=unit(np.load(prep/'fit_bank_unit.npy'));cal=load(base/'calibrate/calibration.json')
    sys.path.insert(0,'/root/avde_reproduction_20261016')
    from native_statistics import concept_bootstrap_counts,paired_identification_interval,paired_query_interval
    methods=sorted({k.split('/')[2] for k in endpoints.files});summaries={};per={};matrices={};containment={}
    for group in ['calibration','development']:
        target=np.load(verified_asset(prep,group+'_targets_EVALUATION_OR_CALIBRATION_ONLY.npz'));y=target['unit'];raw=target['raw']
        for name in ['paired','permuted']:
            a,d,mu,valid,sigma,pairs=geometry(cfg,packet,group,name,bank,contract['fit_scale_floor'])
            b,active=boundaries(mu,sigma,valid,cal['q']['repeat']['0.1'],cfg['rho'])
            excluded=np.maximum(np.where(active,b-np.einsum('bld,bd->bl',d,unit(y)),0),0).max(1)>1e-7
            containment[group+'/'+name]=dict(target_exclusion_rate=float(excluded.mean()),per_query_excluded=excluded.tolist(),
                active_query_rate=float(active.any(1).mean()),mean_active_edges=float(active.sum(1).mean()),
                formal_conformal_coverage_claimed=False)
            ranks={}
            for method in methods:
                values=[endpoints[group+'/'+name+'/'+method+'/'+str(s)] for s in cfg['generation_seeds']]
                result=[rank_metrics(x,y) for x in values];rank=np.array([x[0] for x in result]);win=np.mean([x[1] for x in result],0);cos=np.mean([x[2] for x in result],0)
                ranks[method]=rank;key=group+'/'+name+'/'+method;matrices[key]=win
                mse=np.mean([np.mean((x*norms[group+'/'+name+'/'+method+'/'+str(s)][:,None]-raw)**2,1) for s,x in zip(cfg['generation_seeds'],values)],0)
                mechanism=[mechanisms[group+'/'+name+'/'+method+'/'+str(s)] for s in cfg['generation_seeds']]
                violation=np.mean([np.asarray(x['primary_set_violation'])>1e-6 for x in mechanism],0)
                m=dict(Recall1=float((rank==1).mean()),Recall5=float((rank<=5).mean()),semantic_cosine_two_way=float(win.sum()/(len(y)*(len(y)-1))),
                    mean_conservative_rank=float(rank.mean()),matched_cosine=float(cos.mean()),raw_norm_preserving_MSE=float(mse.mean()),
                    unit_MSE=float(2-2*cos.mean()),latent_diversity=float(np.mean([np.mean((values[i]-values[j])**2) for i in range(3) for j in range(i+1,3)])),
                    primary_constraint_violation_rate=float(violation.mean()),mean_movement_from_prior=float(np.mean([x['movement_from_prior'] for x in mechanism])))
                if method.startswith('sort_'):
                    sol=[x['solver'] for x in mechanism];ac=np.asarray([x['active_count'] for x in sol]);fb=np.asarray([x['fallback'] for x in sol])
                    m.update(solver_fallback_rate=float(fb.mean()),solver_fallback_among_active=float(fb.sum()/max(1,(ac>0).sum())),
                             solver_ball_active_rate=float(np.mean([x['ball_active'] for x in sol])),solver_max_cycles=int(np.max([x['cycles'] for x in sol])),
                             solver_max_gap=float(np.max([x['projection_gap'] for x in sol])))
                summaries[key]=m;per[key]=dict(Recall1=(rank==1).mean(0).tolist(),Recall5=(rank<=5).mean(0).tolist(),
                    rank=rank.mean(0).tolist(),cosine=cos.tolist(),MSE=mse.tolist(),primary_constraint_violation=violation.tolist())
            for method in methods:
                for control in ['a0','prior']:
                    a=ranks[method]==1;b=ranks[control]==1
                    summaries[group+'/'+name+'/'+method][control+'_transitions']={
                        'correct_to_wrong':float((b&~a).mean()),'wrong_to_correct':float((~b&a).mean()),
                        'both_correct':float((a&b).mean()),'both_wrong':float((~a&~b).mean())}
    counts=concept_bootstrap_counts(200,repetitions=cfg['bootstrap_repetitions'],seed=cfg['statistics_seed'])
    contrasts=[('sort_repeat_a10',x) for x in ['prior','a0','mixture','gate','bank_softmean','sort_constant_a10','sort_uncalibrated','sort_random_rank_a10']]
    comparisons={};permutation_p={};rng=np.random.default_rng(cfg['statistics_seed']);signs=rng.choice([-1.,1.],size=(20000,200))
    for a,b in contrasts:
        ak='development/paired/'+a;bk='development/paired/'+b
        comparison={k:paired_query_interval(per[ak][k],per[bk][k],counts) for k in ['Recall1','Recall5','rank','cosine','MSE','primary_constraint_violation']}
        comparison['two_way']=paired_identification_interval(matrices[ak],matrices[bk],counts);comparisons[a+'__minus__'+b]=comparison
        if b in ['prior','a0','mixture','gate']:
            delta=np.array(per[ak]['Recall1'])-np.array(per[bk]['Recall1'])
            permutation_p[b]=float((1+np.sum(np.abs(signs@delta/200)>=abs(delta.mean())-1e-12))/20001)
    ordering=sorted(permutation_p,key=permutation_p.get);adjusted={};last=0.
    for i,key in enumerate(ordering):last=max(last,min(1.,(len(ordering)-i)*permutation_p[key]));adjusted[key]=last
    pairing={}
    for method in methods:
        pairing[method]=paired_query_interval(per['development/paired/'+method]['Recall1'],per['development/permuted/'+method]['Recall1'],counts)
    primary=summaries['development/paired/sort_repeat_a10'];competitors=['a0','mixture','gate','bank_softmean']
    best=max(competitors,key=lambda x:summaries['development/paired/'+x]['Recall1'])
    # This label is a reported panel comparison, never deployment selection.
    best_ci=comparisons['sort_repeat_a10__minus__'+best]['Recall1']
    gate=dict(status='completed',descriptive_best_simple_control=best,Recall1_gain=best_ci['paired_minus_control'],interval95=best_ci['interval95'],
        at_least3pp=best_ci['paired_minus_control']>=.03,interval_supports_gain=best_ci['interval95'][0]>0,
        active_query_rate=containment['development/paired']['active_query_rate'],
        engineering_fallback_gate=primary['solver_fallback_among_active']<=.01,
        true_minus_permuted_positive=pairing['sort_repeat_a10']['interval95'][0]>0,
        full_image_validation=False,SOTA_established=False)
    gate['expand_SORT_now']=bool(gate['at_least3pp'] and gate['interval_supports_gain'] and gate['engineering_fallback_gate'] and gate['true_minus_permuted_positive'] and gate['active_query_rate']>.05)
    write(out/'summary.json',summaries);write(out/'per_query_metrics.json',per);write(out/'target_containment.json',containment)
    write(out/'comparisons.json',comparisons);write(out/'stage_gate.json',gate);write(out/'pairing.json',pairing)
    np.savez_compressed(out/'identification_matrices.npz',**matrices)
    report.update(status='completed',stage_gate=gate,primary_result=primary,Recall1_permutation_p=permutation_p,Holm_adjusted_p=adjusted,
        comparisons=comparisons,containment=containment,training_seeds=1,generation_seeds=cfg['generation_seeds'],new_images=0,
        caveats=['Single subject, reused calibration/dev, unresolved upstream whitening/event dependence','Vectors/CLIP-supervised diagnostics, no independent image evaluator or full ATM reproduction','No formal conformal guarantee claimed for historical adaptive reuse'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['prepare','generate','calibrate','transport','evaluate'],required=True)
    args=p.parse_args();cfg=load(args.config);assert cfg['official_test_locked'] and not cfg['formal_conformal_guarantee_claimed']
    out=Path(cfg['out'])/args.mode;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=dict(status='running',mode=args.mode,started_UTC=now(),source_sha256=digest(__file__),math_sha256=digest(Path(__file__).with_name('sort_math.py')),
        config_sha256=digest(args.config),official_test_read=False)
    write(out/'preregistration.json',report);write(out/'progress.json',report)
    try:globals()[args.mode](cfg,out,report)
    except BaseException as e:report.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:report.update(seconds=time.monotonic()-start,finished_UTC=now());write(out/'results.json',report);write(out/'progress.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['comparisons','containment']},ensure_ascii=False))


if __name__=='__main__':main()
