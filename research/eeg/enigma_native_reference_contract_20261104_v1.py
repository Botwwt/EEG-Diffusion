"""Native63 EEG reference; fixed full150 and two independent concept-OOF fits."""
import json,sys
from pathlib import Path
import numpy as np
from enigma_native_reference_math_20261104_v1 import role_rows,canonical_concept_join

ROLES=('full','oof0','oof1')
BASES=('direct','nce_continued','conditional','nce_literal','mse9_literal','mse9_exp')
ARMS=BASES+('enigma_native63',)
MODELS=('enigma_native63',)
METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1',
         'within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
IDENTITY=('query_id','subject_id','concept_id','stimulus_id','array_row','partition','image_seed')
CONTRASTS=tuple(('enigma_native63',a) for a in ('direct','nce_continued','conditional','mse9_literal','mse9_exp'))
_c=None

def setup(path):
    global _c
    q=json.loads(Path(path).read_bytes())
    assert q['status']=='LOCKED_ENIGMA_NATIVE63_FULL150_CONCEPT_OOF_REFERENCE'
    assert q['host']=='old' and q['seed']==1729 and q['epochs']==150 and q['batch']==512
    assert q['maximum_output_bytes']==512<<20 and q['new_updates']==7800 and q['new_exposures']==3969000
    assert q['arms']==list(ARMS) and q['new_images']==400 and q['reused_images']==2400
    if _c is None:
        sys.path.insert(0,str(Path(q['baseline_policy']['path']).parent))
        import atm_objective_scale_contract_20261104_v1 as P
        up=P.setup(q['baseline_policy']['path']);K=up['K']
        for pin in q['sources'].values():K.checked(pin)
        for key in ('baseline_policy','baseline_completion','baseline_prepared','baseline_features',
                    'baseline_metrics','baseline_ordered_records','baseline_conditions','EEG','visual_target',
                    'text_target','text_cache_commit','text_cache_ordering','text_cache_source','visual_ordering',
                    'legacy_inputs','original_models','original_training','source_metrics','source_low_level'):
            K.checked(q[key])
        for key in ('EEG','visual_target','legacy_inputs','source_metrics','source_low_level'):
            assert q[key]==up['q'][key]
        if 'resume_origin_policy' in q:
            origin=K.read(K.checked(q['resume_origin_policy']))
            for key in ('EEG','visual_target','text_target','legacy_inputs','epochs','batch','training','optimizer',
                        'maximum_output_bytes','new_updates','new_exposures','new_images','reused_images'):
                assert q[key]==origin[key]
            for key in ('reused_profile','reused_first_segment','reused_checkpoint'):K.checked(q[key])
            assert q['reused_successful_events'][0]['label']==origin['job_prefix']+'_profile'
            assert q['reused_successful_events'][1]['label']==origin['job_prefix']+'_fit_full_50'
            original_launch=K.read(Path(origin['code_root'])/'actual_launch.json')
            assert not K.process(original_launch['PID'],original_launch['start_ticks'])['live']
        completion=K.read(q['baseline_completion']['path'])
        assert completion['status']=='COMPLETE_NATIVE_EEG_OBJECTIVE_SCALE_FACTORIAL_FULL400'
        launch=K.read(Path(q['baseline_policy']['path']).parent/'actual_launch.json')
        assert not K.process(launch['PID'],launch['start_ticks'])['live']
        inputs=K.read(q['legacy_inputs']['path']);T=np.asarray(inputs['T_array_rows'],np.int64)
        folds=np.asarray(inputs['T_heldout_folds'],np.int64)
        text_order=next(v for v in K.read(q['text_cache_ordering']['path']) if v['split']=='train')
        assert len(text_order['filenames'])==16540 and len(text_order['class_folders'])==1654
        assert text_order['all_filenames_match'] and text_order['all_concept_names_match']
        visual_rows=sorted((v for v in K.read(q['visual_ordering']['path']) if v['split']=='train'),key=lambda v:v['index'])
        assert [v['index'] for v in visual_rows]==list(range(16540))
        assert [Path(v['member']).name for v in visual_rows]==text_order['filenames']
        assert all(Path(v['member']).parent.name==text_order['class_folders'][i//10] for i,v in enumerate(visual_rows))
        split=K.read(K.checked(inputs['split']))
        class_join=canonical_concept_join(text_order['class_folders'],split['train_concepts'])
        canonical=[v['canonical_split_concept'] for v in class_join]
        assert np.array_equal(T,np.asarray(split['image_rows']['T'],np.int64))
        assert set(canonical[int(row)//10] for row in T)==set(split['train_concepts']['T'])
        assert all(fold==split['oof_fold_by_concept'][canonical[int(row)//10]] for row,fold in zip(T,folds))
        committed=K.read(q['text_cache_commit']['path']);assert committed['complete']
        for key,name in (('text_target','train_class_text.pt'),('text_cache_ordering','row_order_contract.json')):
            assert {k:q[key][k] for k in ('bytes','sha256')}==committed['files'][name]
        roles={a:role_rows(T,folds,a) for a in ROLES}
        assert [len(roles[a][0]) for a in ROLES]==[13230,6610,6620]
        records=K.read(q['baseline_ordered_records']['path'])
        baseline={a:[r for r in records if r['arm']==a] for a in BASES}
        assert all(len(x)==400 for x in baseline.values())
        _c=dict(up,q=q,K=K,up=up,P=P,T=T,folds=folds,roles=roles,class_identity_join=class_join,
                D=np.asarray(inputs['D_array_rows'],np.int64),baseline=baseline,
                out=Path(q['output_root']))
    assert _c['q']==q
    return _c

def resources(c,starting=False):
    r=c['P'].resources(c['up'],starting=starting);q=c['q']
    used=sum(p.stat().st_blocks*512 for p in Path(q['project_root']).rglob('*') if p.is_file()) if Path(q['project_root']).exists() else 0
    assert used<=q['maximum_output_bytes']
    required=r['required_including_this_project']+q['maximum_output_bytes']-used
    assert r['free_bytes']>=required,dict(free=r['free_bytes'],required=required)
    return dict(required_remaining_bytes=r['required_including_this_project'],required_including_this_project=required,
        free_bytes=r['free_bytes'],own_used_bytes=used,own_remaining_bytes=q['maximum_output_bytes']-used,
        all_previous_forecasts_including_unused_factorial768MiB_retained=True,
        legacy50GiB_and8GiB_buffer_unchanged=True,actual_CPU_headroom_bytes=r['actual_CPU_headroom_bytes'])

def records(c,n):
    assert n==400
    result=dict(c['baseline'])
    p=c['out']/'images/enigma_native63/images.jsonl'
    result['enigma_native63']=[json.loads(s) for s in p.read_text().splitlines()]
    assert len(result['enigma_native63'])==400
    return result

def paired(arrays,queries):
    groups=[[i for i,r in enumerate(queries) if r['concept_id']==g]
            for g in sorted({r['concept_id'] for r in queries})]
    assert len(groups)==40 and all(len(x)==10 for x in groups)
    draw=np.random.default_rng(1729).integers(0,40,(1000,40));result={}
    for a,b in CONTRASTS:
        result[a+'-'+b]={}
        for key in METRICS:
            difference=arrays[a][key]-arrays[b][key]
            means=np.asarray([difference[g].mean() for g in groups])
            result[a+'-'+b][key]=dict(difference=float(difference.mean()),
                descriptive_concept_bootstrap95=np.quantile(means[draw].mean(1),[.025,.975]).tolist(),
                positive_concepts=int((means>0).sum()),concepts=40)
    return result

def interpret(pairs,n):
    assert n==400
    return dict(action='COMPLETE_FULL400_NATIVE_EEG_REFERENCE',all_metrics_and_strong_controls_retained=True,
        no_single_metric_stop=True,no_best_epoch_or_seed_selection=True,
        author_architecture_reference_NOT_new_innovation=True,full_author_paper_reproduction=False,
        second_independent_method_or_SOTA=False,no_semantic_noninferiority_from_crossing_CI=True)
