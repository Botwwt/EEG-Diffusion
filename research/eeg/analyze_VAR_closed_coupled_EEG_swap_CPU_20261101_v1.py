"""Post-hoc complete-panel matched-EEG intervention analysis; no new fits or images.

The crossed contrast is .5*(g_correct-g_donor).(t_query-t_donor).
It requires same-query latent/token seeds and the same frozen final model.
All 400 exposed development queries and all 40 concepts remain included.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import base64
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np

L=Path(__file__).resolve().parent
P=L/'audit/EEG_native_marginal_images_20261029_v2'
RECEIVED=P/'received_stage400_v1'
OUT=L/'audit/VAR_closed_coupled_EEG_swap_20261101_v1'
ARMS=('coupled','visual','marginal','donor')


def read(path):return json.loads(Path(path).read_bytes())


def ref(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def normal(values):
    values=np.asarray(values,np.float64);norm=np.linalg.norm(values,axis=1,keepdims=True)
    assert np.isfinite(values).all() and np.all(norm>0)
    return values/norm


def scalar_dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))


def run(transport_path):
    started=time.monotonic();assert not OUT.exists()
    transported=read(transport_path);assert transported['exit_code']==0
    assert transported['host_key_SHA256']=='y2DhrdHAWfTCoY9uLz0ve3EzTKYtWNBBWm80QUTOkXg'
    spec_record=json.loads(transported['stdout']);assert spec_record['status']=='READ_PINNED_CLOSED_IMAGE_SPEC_ONLY'
    spec_raw=base64.b64decode(spec_record['base64'],validate=True)
    assert hashlib.sha256(spec_raw).hexdigest()==spec_record['file']['sha256']=='fa93d3128ea55309443a367bfba9d15a46ab6ad0dc43d2f7711f81b685529211'
    assert len(spec_raw)==7760;spec=json.loads(spec_raw)
    source_proof={}
    for name in ('generate_EEG_native_marginal_images_GPU_20261029_v2.py','EEG_native_conditional_variational_core_20261029_v1.py'):
        pins=[pin for pin in spec['source_pins'] if Path(pin['path']).name==name];assert len(pins)==1,(name,pins)
        current=ref(L/name);assert all(current[k]==pins[0][k] for k in ('bytes','sha256'));source_proof[name]=pins[0]
    receiving=read(RECEIVED/'receiving_completion.json');assert receiving['stage']==400
    for key,pin in receiving['files'].items():
        assert ref(RECEIVED/key)['sha256']==pin['remote']['sha256'] and (RECEIVED/key).stat().st_size==pin['remote']['bytes']
    original=read(P/'stage400_results_v1/summary.json')
    assert original['receiving_completion_sha256']==ref(RECEIVED/'receiving_completion.json')['sha256']
    panel=read(RECEIVED/'old/panel400.json');queries=panel['queries'];assert panel['target_gallery']==queries and len(queries)==400
    groups={}
    for i,row in enumerate(queries):
        assert row['target_index']==i;groups.setdefault(row['concept_id'],[]).append(i)
    groups=[groups[key] for key in sorted(groups)]
    assert len(groups)==40 and all(len(group)==10 for group in groups)
    group_array=np.array(groups);concept_index=np.empty(400,np.int64);donor=np.empty(400,np.int64)
    for c,group in enumerate(groups):
        concept_index[group]=c
        for j,index in enumerate(group):donor[index]=group[(j+1)%10]
    records={arm:read(RECEIVED/('old' if arm in ('coupled','donor') else 'new')/(arm+'_records.json')) for arm in ARMS}
    for arm in ARMS:
        assert len(records[arm])==400
        for i,row in enumerate(records[arm]):
            assert row['index']==i and row['query_id']==queries[i]['query_id'] and row['stimulus_id']==queries[i]['stimulus_id']
            assert row['array_row']==queries[i]['array_row'] and row['generation']['one_image'] is True
            assert row['generation']['best_of_N']==1 and row['normal_inference_reads_query_image_or_label_or_gallery'] is False
    for i in range(400):
        correct=records['coupled'][i];wrong=records['donor'][i]
        assert correct['EEG_row']==i and wrong['EEG_row']==int(donor[i])
        assert correct['fit_model_sha256']==wrong['fit_model_sha256']==original['fits']['old']['sha256']
        assert correct['latent_seed']==wrong['latent_seed'] and correct['token_seed']==wrong['token_seed']
        assert correct['generation']==wrong['generation']
        assert wrong['EEG_parameters_sha256']==records['coupled'][donor[i]]['EEG_parameters_sha256']
    OUT.mkdir()
    write(OUT/'analysis_plan.json',dict(status='POST_HOC_ALL400_MATCHED_EEG_SWAP_DIAGNOSTIC_NOT_PREREGISTERED_EFFICACY',
        selected_after_original_results_were_known=True,all_queries=400,concepts=40,
        crossed_contrast='.5*(g_correct-g_donor) dot (t_query-t_donor)',
        global_2AFC_identity='(9*within_concept_2AFC+390*between_concept_2AFC)/399',
        descriptive_bootstrap=dict(draws=1000,seed=1729,unit='concept',same_draws_for_all_quantities=True),
        all_four_arms_reported=list(ARMS),no_new_model_or_threshold_selection=True,
        no_change_to_original_gate=True,independent_confirmation=False,SOTA=False,
        source=ref(__file__),original_result=ref(P/'stage400_results_v1/summary.json')))
    try:
        (OUT/'registered_historical_image_spec.json').write_bytes(spec_raw)
        feature_archive=np.load(P/'stage400_results_v1/all4_frozen_features.npz',allow_pickle=False)
        targets=np.load(RECEIVED/'old/targets400.npy',allow_pickle=False)
        assert targets.shape==(400,768) and np.array_equal(targets,feature_archive['targets'])
        features={}
        for arm in ARMS:
            features[arm]=np.load(RECEIVED/('old' if arm in ('coupled','donor') else 'new')/(arm+'_features.npy'),allow_pickle=False)
            assert features[arm].shape==(400,768) and np.array_equal(features[arm],feature_archive[arm])
        T=normal(targets);G={arm:normal(features[arm]) for arm in ARMS}
        within=concept_index[:,None]==concept_index[None,:];np.fill_diagonal(within,False)
        between=concept_index[:,None]!=concept_index[None,:]
        assert np.all(within.sum(1)==9) and np.all(between.sum(1)==390)
        old_arrays=np.load(P/'stage400_results_v1/per_query_all_metrics.npz',allow_pickle=False)
        values={};matrices={};rank_maxerror=0.;old_reproduction_maxerror=0.;identity_maxerror=0.
        centroids=np.stack([T[group].mean(0) for group in groups]);centroid_normal=normal(centroids)
        for arm in ARMS:
            # FP32 reproduces the original frozen metric implementation/tie rule.
            raw=features[arm];original_q=raw/np.linalg.norm(raw,axis=1,keepdims=True)
            original_t=targets/np.linalg.norm(targets,axis=1,keepdims=True);score32=original_q@original_t.T
            positive=score32[np.arange(400),np.arange(400)]
            credit=(score32<positive[:,None]).astype(np.float64)+.5*(score32==positive[:,None])
            global_afc=(credit.sum(1)-.5)/399
            within_afc=(credit*within).sum(1)/9;between_afc=(credit*between).sum(1)/390
            residual=(9*within_afc+390*between_afc)/399-global_afc
            identity_maxerror=max(identity_maxerror,float(np.max(np.abs(residual))))
            old_reproduction_maxerror=max(old_reproduction_maxerror,float(np.max(np.abs(global_afc-old_arrays[arm+'__DINO2AFC']))))
            assert np.allclose(global_afc,old_arrays[arm+'__DINO2AFC'],atol=1e-14,rtol=0)
            assert np.allclose(within_afc,old_arrays[arm+'__within_concept10_two_afc'],atol=1e-14,rtol=0)
            score=G[arm]@T.T;matrices[arm]=score
            p64=score.diagonal();credit64=(score<p64[:,None])+.5*(score==p64[:,None])
            rank_maxerror=max(rank_maxerror,float(np.max(np.abs((credit64.sum(1)-.5)/399-global_afc))))
            residual_margin=np.einsum('nd,nd->n',G[arm],T-centroids[concept_index])*10/9
            independent=np.array([score[i,i]-math.fsum(float(score[i,j]) for j in groups[concept_index[i]] if j!=i)/9 for i in range(400)])
            assert np.max(np.abs(residual_margin-independent))<1e-12
            values[arm+'__global2afc']=global_afc;values[arm+'__within2afc']=within_afc
            values[arm+'__between2afc']=between_afc;values[arm+'__within_top1']=old_arrays[arm+'__within_concept10_top1']
            values[arm+'__own_stimulus_residual_cosine_margin']=residual_margin
            values[arm+'__concept_centroid40_top1']=((G[arm]@centroid_normal.T).argmax(1)==concept_index).astype(np.float64)
        deltaG=G['coupled']-G['donor'];deltaT=T-T[donor]
        crossed=np.einsum('nd,nd->n',deltaG,deltaT)/2
        correct=np.einsum('nd,nd->n',G['coupled'],deltaT)
        donor_oriented=-np.einsum('nd,nd->n',G['donor'],deltaT)
        direct=np.array([.5*(scalar_dot(G['coupled'][i],T[i])-scalar_dot(G['coupled'][i],T[donor[i]])
            +scalar_dot(G['donor'][i],T[donor[i]])-scalar_dot(G['donor'][i],T[i])) for i in range(400)])
        scalar_maxerror=float(np.max(np.abs(crossed-direct)));assert scalar_maxerror<1e-12
        assert np.max(np.abs(crossed-(correct+donor_oriented)/2))<1e-12
        values.update(crossed_EEG_target_feature_contrast=crossed,correct_EEG_own_minus_donor_target=correct,
            donor_EEG_donor_minus_own_target=donor_oriented)
        attribution={}
        for baseline in ('visual','marginal','donor'):
            name='coupled_minus_'+baseline
            full=values['coupled__global2afc']-values[baseline+'__global2afc']
            w=values['coupled__within2afc']-values[baseline+'__within2afc']
            b=values['coupled__between2afc']-values[baseline+'__between2afc']
            assert np.max(np.abs(full-(9*w+390*b)/399))<1e-12
            values[name+'__global']=full;values[name+'__within']=w;values[name+'__between']=b
            attribution[name]=dict(global_gain=float(full.mean()),within_weighted_contribution=float(9*w.mean()/399),
                between_weighted_contribution=float(390*b.mean()/399),
                meaning='Exact fixed-gallery rank-count decomposition, not a causal attribution')
        names=list(values);matrix=np.column_stack([values[name] for name in names]);assert matrix.shape==(400,len(names))
        per_concept=np.stack([matrix[group].mean(0) for group in groups])
        draws=np.random.default_rng(1729).integers(0,40,size=(1000,40))
        samples=per_concept[draws].mean(1);intervals=np.quantile(samples,[.025,.975],axis=0)
        independent_means=np.array([[math.fsum(float(matrix[i,j]) for i in group)/10 for j in range(len(names))] for group in groups])
        counts=np.array([np.bincount(row,minlength=40) for row in draws])
        independent_samples=counts@independent_means/40
        bootstrap_maxerror=float(np.max(np.abs(samples-independent_samples)));assert bootstrap_maxerror<1e-12
        summary={name:dict(mean=float(matrix[:,j].mean()),descriptive_concept_bootstrap95=intervals[:,j].tolist(),
            positive_concepts=int((per_concept[:,j]>0).sum()),concepts=40,queries=400) for j,name in enumerate(names)}
        np.savez_compressed(OUT/'all400_quantities_and_bootstrap.npz',query_values=matrix,concept_values=per_concept,
            bootstrap_draws=draws,bootstrap_values=samples,donor_indices=donor,quantity_names=np.array(names))
        result=dict(status='COMPLETE_POST_HOC_ALL400_MATCHED_EEG_SWAP_CPU_ANALYSIS_NOT_NEW_EFFICACY',
            UTC=dt.datetime.now(dt.timezone.utc).isoformat(),queries=400,concepts=40,subject='sub-01',EEG_repeat_protocol='R4',
            all400_same_latent_seed=True,all400_same_token_seed=True,all400_same_final_model=True,
            all400_donor_EEG_parameter_hashes_match_correct_source_row=True,donor_cycle_no_fixed_points=True,
            pinned_generation_sources=source_proof,summary=summary,exact_rank_decomposition=attribution,
            independent_checks=dict(original_fp32_metrics_maxerror=old_reproduction_maxerror,
                fp64_vs_original_fp32_rank_maxerror=rank_maxerror,all400_crossed_contrast_scalar_fsum_maxerror=scalar_maxerror,
                all1000_bootstrap_rows_counting_reference_maxerror=bootstrap_maxerror,rank_decomposition_maxerror=identity_maxerror),
            source_refs=[ref(RECEIVED/'receiving_completion.json'),ref(P/'stage400_results_v1/all4_frozen_features.npz'),
                ref(P/'stage400_results_v1/per_query_all_metrics.npz'),ref(transport_path),ref(OUT/'analysis_plan.json')],
            post_hoc_analysis_not_preregistered=True,original_efficacy_gate_changed=False,
            all_generator_RNG_states_or_image_features_replayed_locally=False,
            uncertainty_scope='Descriptive resampling of 40 exposed concepts, one subject and one training seed only',
            new_GPU_jobs=0,new_optimizer_updates=0,new_reconstruction_images=0,new_fullT_fit_or_bank_admitted=False,
            second_method_or_novelty_or_SOTA_proved=False,seconds=time.monotonic()-started)
        assert time.monotonic()-started<120
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<1<<20
        write(OUT/'analysis.json',result)
        print(json.dumps(dict(status=result['status'],crossed=summary['crossed_EEG_target_feature_contrast'],
            correct_margin=summary['correct_EEG_own_minus_donor_target'],donor_margin=summary['donor_EEG_donor_minus_own_target'],
            attribution=attribution,independent_checks=result['independent_checks'],seconds=result['seconds'])))
    except BaseException as error:
        write(OUT/'failure.json',dict(status='POST_HOC_DIAGNOSTIC_FAILURE_PRESERVED',error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec-transport',type=Path,required=True);run(p.parse_args().spec_transport)
