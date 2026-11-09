"""D40 fixed eight-arm image contracts and pre-registered decisions."""
import hashlib,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
B=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/rapid_iteration_20261027_v1/candidates/D40_spatial_innovation')
EVAL=B/'factorial_images_v1'
ENCODERS=('original','marginal','measurement','permuted_measurement')
BACKGROUNDS=('native','ordinary')
ARMS=tuple(e+'__'+b for e in ENCODERS for b in BACKGROUNDS)
IDENTITY=('query_id','subject_id','concept_id','stimulus_id','array_row','partition','image_seed')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_bytes())
def checked(x):
    p=Path(x['path']);assert p.is_file() and sha(p)==x['sha256'],str(p)
    if 'bytes' in x:assert p.stat().st_size==x['bytes']
    return p
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def array(p,v):
    with Path(p).open('xb') as f:np.save(f,v,allow_pickle=False);f.flush();os.fsync(f.fileno())
def byte_sha(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def queries(panel):
    q=panel['Dcheck'];assert len(q)==400 and len({x['query_id'] for x in q})==400
    assert all(x['subject_id']=='sub-01' and x['partition']=='Dcheck' for x in q)
    for n in [80,400]:
        counts={}
        for x in q[:n]:counts[x['concept_id']]=counts.get(x['concept_id'],0)+1
        assert len(counts)==n//10 and set(counts.values())=={10}
    return q
def load_execution(path):
    x=read(path);assert x['status']=='D40_EIGHT_ARM_IMAGE_EXECUTORS_BOUND_TO_UNCHANGED_PRE_FIT_POLICY'
    for r in x['source_pins']:checked(r)
    p=read(checked(x['policy']));t=read(checked(x['training_spec']))
    assert t['locked_future_image_policy']['sha256']==x['policy']['sha256'] and p['encoders']==list(ENCODERS)
    assert p['ordinary_switch']==[False,True] and p['old_ordinary_strength']==1 and t['ordinary_in_training_graph'] is False
    base=read(checked(p['fixed_panel_from']));q=queries(read(checked(base['panel'])))
    return x,p,t,base,q
def complete_fits(execution):
    t=read(checked(execution['training_spec']));done=read(B/'training_queue_v1/completion.json')
    assert done['status']=='D40_ALL3_SPATIAL_FIRST_LAYER_COMPLETE_T_ONE_PASS_AND_CPU_COMMITS_TRAINING_CLOSED'
    assert done['spec']['sha256']==execution['training_spec']['sha256'] and done['total_updates']==2481 and done['total_T_query_exposures']==39690
    fits={}
    for e in ENCODERS[1:]:
        p=B/'arms'/e/'completion.json';v=read(p)
        assert v['updates']==827 and v['query_exposures']==13230 and v['ordinary_in_training_graph'] is False
        assert v['spec']['sha256']==execution['training_spec']['sha256'] and v['fixed_final_not_held_selected']
        checked(v['model']);fits[e]=v['model']
    return done,fits
def prepared(path):
    x,p,t,base,q=load_execution(path);complete_fits(x)
    done=read(EVAL/'prepared/completion.json')
    assert done['status']=='D40_ALL400x8_FRESH_NATIVE_CORE_CONDITIONS_CPU_EXPORTED' and done['execution']['sha256']==sha(path)
    assert done['arms']==list(ARMS) and done['query_ids']==[v['query_id'] for v in q]
    checked(done['conditions']);return x,base,q,done
def stage_gate(path,stage):
    assert stage in [80,400]
    if stage==400:
        p=EVAL/'screen80/score/completion.json';v=read(p);a=read(EVAL/'screen80/score/independent_CPU_audit.json')
        assert v['decision']['action']=='EXPAND400' and v['execution']['sha256']==sha(path)
        assert a['score_completion']['sha256']==sha(p) and a['decision']==v['decision']
def contrast_pairs():
    pairs={}
    for b in BACKGROUNDS:
        for e in ENCODERS[1:]:pairs[e+'_vs_original__'+b]=(e+'__'+b,'original__'+b)
        for e in ['marginal','permuted_measurement']:pairs['measurement_vs_'+e+'__'+b]=('measurement__'+b,e+'__'+b)
    return pairs
def supports(c):
    return c['DINO2AFC']['difference']>=.01 and c['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and c['target_cosine']['descriptive_concept_bootstrap95'][0]>0 and c['within_concept10_top1']['difference']>=0
def decision(contrasts,stage):
    assert stage in [80,400]
    primary={b:contrasts['measurement_vs_original__'+b] for b in BACKGROUNDS}
    harms={b:c['DINO2AFC']['difference']<=-.02 and c['DINO2AFC']['descriptive_concept_bootstrap95'][1]<0 for b,c in primary.items()}
    action=('STOP_INSTANCE' if all(harms.values()) else 'EXPAND400') if stage==80 else ('PROMISING_FOR_FULL_EVALUATION' if supports(primary['ordinary']) else 'STOP_INSTANCE')
    return dict(action=action,both_backgrounds_large_harm=all(harms.values()),background_harm=harms,
        ordinary_background_support=supports(primary['ordinary']) if stage==400 else None,
        native_background_support=supports(primary['native']) if stage==400 else None,
        controls_not_substituted_for_primary=True)
def evaluation_budget():
    assert sum(p.stat().st_size for p in EVAL.rglob('*') if p.is_file())<2<<30
