"""All closed Gaussian VAR E/M logs; no checkpoint selection or new fit."""
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np

L=Path(__file__).resolve().parent
SOURCE=L/'audit/EEG_native_marginal_images_20261029_v2/received_stage400_v1'
OUT=L/'audit/VAR_closed_E_M_log_timing_20261101_v1'
ARMS={'coupled':'old','visual':'new'}
FIELDS=('EEG_KL_nats_per_token','preclip_EEG_gradient','backbone_gradient','ELBO_nats_per_token','NLL_nats_per_query','KL_nats_per_query')
def ref(path):
    p=Path(path);raw=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(path):return json.loads(Path(path).read_bytes())
def put(name,value):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def verify_sources():
    received=read(SOURCE/'receiving_completion.json');pins={}
    for arm,prefix in ARMS.items():
        for name in ('fit_events.json','fit_completion.json','fit_CPU_audit.json'):
            path=SOURCE/prefix/name;pin=ref(path);expected=received['files'][prefix+'/'+name]['remote']
            assert all(pin[k]==expected[k] for k in ('bytes','sha256'));pins[arm+'/'+name]=pin
    launch=read(L/'audit/crossdisciplinary_20261028_v1/native_marginal_old_actual_launch_v2_20261029T124202058718Z.json')
    for name in ('train_EEG_native_marginal_20261029_v2.py','EEG_native_marginal_core_20261029_v2.py'):
        pin=next(p for p in launch['source_pins'] if Path(p['path']).name==name);actual=ref(L/name)
        assert all(actual[k]==pin[k] for k in ('bytes','sha256'));pins[name]=actual
    source=(L/'train_EEG_native_marginal_20261029_v2.py').read_text(encoding='utf-8')
    assert 'p=EEGGaussianPrior(encoder,center,scale).cuda().eval()' in source and 'p.train(' not in source
    before=source.index('for inner in range(4):');lastE=source.index('q.step(ids,leaf);last=dict(',before)
    M=source.index('pm,ps=p(eeg,input_chans=channels);loss=diagonal_KL(qm,qs,pm,ps).mean()/5;',lastE)
    update=source.index('op.step();',M);assert before<lastE<M<update
    assert "teacher_pm,teacher_ps=(pm,ps) if s['arm']=='EEG_coupled_teacher' else" in source
    ast.parse(source)
    return pins
def describe(events,coupled):
    count=np.array([e['queries'] for e in events],np.float64);total=float(count.sum());weight=count/total
    arrays={key:np.array([e[key] for e in events],np.float64) for key in FIELDS}
    assert all(np.isfinite(value).all() for value in arrays.values())
    result={key:float(weight@value) for key,value in arrays.items()}
    result.update(batches=len(events),queries=int(total),
        clip_active_batch_fraction=float(np.mean(arrays['preclip_EEG_gradient']>1)),
        clip_active_query_fraction=float(weight@(arrays['preclip_EEG_gradient']>1)),
        nominal_gradient_clip_multiplier_query_mean=float(weight@np.minimum(1,1/np.maximum(arrays['preclip_EEG_gradient'],1e-30))),
        preclip_gradient_median=float(np.median(arrays['preclip_EEG_gradient'])))
    mask=arrays['preclip_EEG_gradient']>0
    energy=np.divide(arrays['backbone_gradient']**2,arrays['preclip_EEG_gradient']**2,out=np.zeros_like(count),where=mask)
    assert np.all(energy<=1+1e-5) and np.all(energy>=0)
    result['backbone_preclip_gradient_energy_fraction_query_mean']=float(weight@energy)
    error=arrays['ELBO_nats_per_token']-(arrays['NLL_nats_per_query']+arrays['KL_nats_per_query'])/5
    result['maximum_logged_ELBO_mean_identity_rounding_error']=float(np.max(np.abs(error)));assert np.max(np.abs(error))<5e-6
    if coupled:
        difference=arrays['EEG_KL_nats_per_token']-arrays['KL_nats_per_query']/5
        result['last_E_step_KL_change_per_token_query_mean']=float(weight@difference)
        result['last_E_step_KL_change_per_token_median']=float(np.median(difference))
        result['last_E_step_KL_increased_batch_fraction']=float(np.mean(difference>0))
        result['last_E_step_KL_increased_query_fraction']=float(weight@(difference>0))
    return result
def run():
    started=time.perf_counter();assert not OUT.exists();OUT.mkdir();pins=verify_sources()
    plan=dict(status='POSTHOC_ALL_CLOSED_TRAINING_LOG_TIMING_ANALYSIS_NOT_CONFIRMATORY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_pins=pins,all_epochs=list(range(1,11)),all_batches_per_epoch=207,expected_events_per_arm=2070,
        expected_query_exposures_per_arm=132300,training_seed=1729,
        timing='E statistics are before the fourth q update; M KL is after the fourth q update and before any p update, same eval p only in coupled arm',
        derived_quantity='delta_last_E = EEG_KL_nats_per_token - KL_nats_per_query/5 for coupled arm only',
        no_epoch_selection_or_recipe_change=True,no_CIs_from_training_batches=True,no_new_effect_scores_read=True,
        full_CUDA_or_models_loaded=False,previous400_scores_previously_known=True)
    put('analysis_scope.json',plan)
    result={};all_events={}
    for arm,prefix in ARMS.items():
        events=read(SOURCE/prefix/'fit_events.json');assert len(events)==2070
        done=read(SOURCE/prefix/'fit_completion.json');audit=read(SOURCE/prefix/'fit_CPU_audit.json')
        assert done['epoch']==10 and done['EEG_updates']==audit['EEG_updates']==2070 and audit['query_exposures']==132300
        assert done['posterior_updates_per_query']==audit['posterior_updates_per_query']==40
        assert [(e['epoch'],e['batch'],e['queries']) for e in events]==[(epoch,batch,64 if batch<207 else 46) for epoch in range(1,11) for batch in range(1,208)]
        assert [e['order_sha256'] for e in done['coverage']]==[e['order_sha256'] for e in read(SOURCE/'old/fit_completion.json')['coverage']]
        all_events[arm]=events
        result[arm]=dict(all=describe(events,arm=='coupled'),epochs={str(epoch):describe([e for e in events if e['epoch']==epoch],arm=='coupled') for epoch in range(1,11)})
    for a,b in zip(all_events['coupled'],all_events['visual']):assert (a['epoch'],a['batch'],a['queries'])==(b['epoch'],b['batch'],b['queries'])
    put('all_epoch_analysis.json',dict(status='COMPLETE_ALL4140_REAL_CLOSED_LOG_EVENTS_TIMING_ANALYSIS',results=result,
        source_pins=pins,total_log_records=4140,total_query_exposures=264600,seconds=time.perf_counter()-started,
        last_E_difference_is_not_full_E_objective_change=True,no_same_target_accuracy_comparison_between_arms=True,
        clipping_multiplier_is_not_Adam_parameter_step_size=True,no_new_model_effect_or_causal_bottleneck_claimed=True,
        no_new_GPU_jobs=True,no_new_images=True,no_checkpoint_selection=True))
    fields=['arm','epoch']+list(result['coupled']['epochs']['1'])
    with (OUT/'all_epoch_summary.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for arm in ARMS:
            for epoch,row in result[arm]['epochs'].items():writer.writerow(dict(arm=arm,epoch=epoch,**row))
    print(json.dumps(dict(status='COMPLETE_ALL4140_REAL_CLOSED_LOG_EVENTS_TIMING_ANALYSIS',
        coupled_epoch1=result['coupled']['epochs']['1'],coupled_epoch10=result['coupled']['epochs']['10'],
        visual_epoch10=result['visual']['epochs']['10'],new_GPU_jobs=0)))
if __name__=='__main__':run()
