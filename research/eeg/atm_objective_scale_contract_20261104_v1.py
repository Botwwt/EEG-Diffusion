"""Matched full-T objective/scale factorial, all400 images before effect reading."""
import json, sys
from pathlib import Path
import numpy as np

BASES = ('direct', 'nce_continued', 'conditional')
MODELS = ('nce_literal', 'mse9_literal', 'mse9_exp')
ARMS = BASES+MODELS
METRICS = ('target_cosine', 'DINO2AFC', 'gallery_top1', 'within_concept10_top1',
           'within_concept10_two_afc', 'PixCorr_RGB', 'SSIM_Y', 'mean_RGB_absolute_error')
IDENTITY = ('query_id', 'subject_id', 'concept_id', 'stimulus_id', 'array_row', 'partition', 'image_seed')
CONTRASTS = (('mse9_literal','nce_literal'), ('mse9_exp','nce_continued'),
             ('nce_continued','nce_literal'), ('mse9_exp','mse9_literal'),
             ('mse9_literal','nce_continued'), ('nce_literal','direct'),
             ('mse9_literal','direct'), ('mse9_exp','direct'),
             ('conditional','mse9_literal'))
INTERACTION = 'MSE_x_scale_interaction'
_c = None


def setup(path):
    global _c
    q=json.loads(Path(path).read_bytes())
    assert q['status']=='LOCKED_FULL400_NATIVE_EEG_OBJECTIVE_SCALE_FACTORIAL'
    assert q['host']=='old' and q['seed']==1729 and q['arms']==list(ARMS)
    assert q['maximum_output_bytes']==768<<20 and q['epochs']==10 and q['batch']==160
    assert q['new_models']==3 and q['new_updates']==2490 and q['new_exposures']==396900
    assert q['no_effect_readout_before_all400'] and q['new_images']==q['reused_images']==1200
    if _c is None:
        sys.path.insert(0,str(Path(q['baseline_policy']['path']).parent))
        import atm_conditional_stimulus_contract_20261104_v1 as P
        up=P.setup(q['baseline_policy']['path']);K=up['K']
        for key in ('EEG','encoder','visual_target','legacy_inputs','model_source_pins'):
            assert q[key]==up['q'][key]
        for pin in q['sources'].values():K.checked(pin)
        for key in ('baseline_policy','baseline_completion','baseline_prepared','baseline_features',
                    'baseline_metrics','baseline_ordered_records','baseline_conditions','reserve_ledger',
                    'source_metrics','source_low_level','original_compute_loss_source'):
            K.checked(q[key])
        for pin in q['model_source_pins']:K.checked(pin)
        for key in ('EEG','encoder','visual_target','legacy_inputs'):K.checked(q[key])
        done=K.read(q['baseline_completion']['path'])
        assert done['stage400_completed'] and done['new_updates']==2490
        launch=K.read(Path(q['baseline_policy']['path']).parent/'actual_launch.json')
        assert not K.process(launch['PID'],launch['start_ticks'])['live']
        records=K.read(q['baseline_ordered_records']['path'])
        baseline={a:[r for r in records if r['arm']==a] for a in BASES}
        assert all(len(v)==400 for v in baseline.values())
        assert K.read(q['baseline_policy']['path'])['sources']==q['reused_NCE_training_sources']
        _c=dict(up,q=q,K=K,out=Path(q['output_root']),up=up,baseline=baseline,
                reserve=K.read(q['reserve_ledger']['path']))
    assert _c['q']==q
    return _c


def resources(c, starting=False):
    from atm_closed_reserve_quote_20261104_v1 import verify_and_quote
    r=verify_and_quote(c['K'],c['reserve'],c['q']['project_root'],c['q']['maximum_output_bytes'])
    info={s.split(':',1)[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if ':' in s and len(s.split())>=2 and s.split()[1].isdigit()}
    headroom=info['MemAvailable']
    maximum=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
    if maximum.exists() and current.exists() and maximum.read_text().strip()!='max':
        headroom=min(headroom,int(maximum.read_text())-int(current.read_text()))
    assert headroom>=(24<<30 if starting else 2<<30)
    r.update(actual_CPU_headroom_bytes=headroom,required_CPU_headroom_bytes=24<<30 if starting else 2<<30)
    return r


def records(c,n):
    assert n==400
    result=dict(c['baseline'])
    for a in MODELS:
        result[a]=[json.loads(s) for s in (c['out']/'images'/a/'images.jsonl').read_text().splitlines()]
        assert len(result[a])==400
    return result


def paired(arrays,queries):
    groups=[[i for i,r in enumerate(queries) if r['concept_id']==g]
            for g in sorted({r['concept_id'] for r in queries})]
    assert len(groups)==40 and all(len(v)==10 for v in groups)
    draw=np.random.default_rng(1729).integers(0,40,(1000,40))
    result={}
    for name,coefficients in [(a+'-'+b,{a:1,b:-1}) for a,b in CONTRASTS]+[
        (INTERACTION,{'mse9_exp':1,'nce_continued':-1,'mse9_literal':-1,'nce_literal':1})]:
        result[name]={}
        for key in METRICS:
            d=sum(v*arrays[a][key] for a,v in coefficients.items())
            means=np.array([d[g].mean() for g in groups])
            result[name][key]=dict(difference=float(d.mean()),
                descriptive_concept_bootstrap95=np.quantile(means[draw].mean(1),[.025,.975]).tolist(),
                positive_concepts=int((means>0).sum()),concepts=40,coefficients=coefficients)
    return result


def interpret(pairs,n):
    assert n==400
    return dict(action='COMPLETE_FIXED_FULL400_OBJECTIVE_SCALE_FACTORIAL',
        all9_contrasts_and_factorial_interaction_all8_metrics=True,
        no_best_cell_or_single_metric_selection=True, no_automatic_seed_or_fit=True,
        no_semantic_noninferiority_from_crossing_CI=True,
        known_MSE_and_temperature_controls_NOT_novel_method=True,
        second_independent_method_or_SOTA=False)
