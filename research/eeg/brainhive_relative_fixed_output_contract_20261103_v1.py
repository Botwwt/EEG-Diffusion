"""Frozen raw-strength diagnostic, separate from the original calibrated method."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ARMS=('conditional_same','conditional_different','conditional_same_mismatched_teacher','continued_F_cosine')
BASELINES=('parent','ordinary','risk','starting_F')
STATUS='LOCKED_FROZEN_RELATIVE_RAW1_FULL400_DIAGNOSTIC_NOT_NEW_METHOD'
METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1',
         'within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')


def read(p):return json.loads(Path(p).read_bytes())


def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)


def checked(pin):
    actual=ref(pin['path'])
    assert all(actual[k]==v for k,v in pin.items()),dict(expected=pin,actual=actual)
    return Path(pin['path'])


def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.pending');assert not temp.exists()
    with temp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(p)


def process(pid,ticks=None):
    try:st=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return dict(PID=pid,live=False,state='MISSING')
    exact=ticks is None or int(st[19])==ticks
    return dict(PID=pid,start_ticks=int(st[19]),state=st[0],live=exact and st[0] not in ('Z','X'))


def setup(path):
    s=read(path)
    assert s['status']==STATUS and s['host']=='old' and s['seed']==1729
    assert tuple(s['arms'])==ARMS and s['fixed_strength']==1 and s['radius']==.25
    assert s['queries']==400 and s['gallery']==400 and s['new_training_updates']==0
    assert s['maximum_output_bytes']==768<<20 and s['maximum_seconds']==7200
    for p in s['sources'].values():checked(p)
    tp=checked(s['native_training_protocol']);sys.path.insert(0,str(tp.parent))
    import brainhive_native_direction_full_contract_20261103_v1 as C
    train,e,*_=C.setup(tp) # The isolated runtime is initialized exactly once.
    rel=read(checked(s['relative_study']));assert tuple(rel['arms'])==ARMS
    for p in rel['source_pins']:checked(p)
    checked(rel['ordinary']);checked(rel['starting_F'])
    done=read(checked(s['relative_original_completion']))
    assert done['decision']['action']=='STOP_INSTANCE' and done['final_stage']==80
    release=read(checked(s['native_closed_future_allowance']))
    assert release['status']=='RETIRED_UNUSED_NATIVE_27_FITS_FUTURE_ALLOWANCE_ONLY'
    assert release['remaining27fits']=='WITHDRAWN_UNRUN_NOT_SCIENTIFIC_FAILURES' and not release['files_removed']
    assert not release['future_automatic_restart_authorized'] and release['legacy50GiB_buffer8GiB_unchanged']
    assert release['training_protocol']==s['native_training_protocol']
    for owner in release['closed_owner_PIDs']:
        assert not process(owner['PID'],owner['start_ticks'])['live']
    for fp in release['retained_roots_stat_fingerprints']:
        assert C.directory_fingerprint(fp['path'])==fp
    return dict(s=s,C=C,train=train,e=e,relative=rel,release=release)


def resources(context,starting=False):
    s,C,train,e,release=(context[k] for k in ('s','C','train','e','release'))
    prior=C.prior_commitments(e);oldF=C.validate_release(train['closed_F_budget_review'])
    required=prior['required_before_F_release']-oldF['released_future_bytes']
    # The withdrawn 10GiB native project contributes no future output. Its actual
    # 894MB remains on disk and is already excluded from physical free space.
    assert sum(v['allocated_bytes'] for v in release['retained_roots_stat_fingerprints'])+release['released_future_bytes']==10<<30
    groups=[]
    for v in s['other_future_groups']:
        used=sum(C.allocated(p) for p in v['roots']);assert used<=v['cap']
        groups.append(dict(name=v['name'],used=used,cap=v['cap'],remaining=v['cap']-used))
        required+=v['cap']-used
    used=C.allocated(s['code_root'])+C.allocated(s['output_root'])
    assert used<=s['maximum_output_bytes'];required+=s['maximum_output_bytes']-used
    free=shutil.disk_usage(s['output_root']).free
    assert required>=(50<<30)+(8<<30) and free>=required
    assert prior['headroom_bytes']>=(24 if starting else 10)<<30
    return dict(required_remaining_bytes=required,free_bytes=free,own_used_bytes=used,
        own_remaining_bytes=s['maximum_output_bytes']-used,groups=groups,
        inherited_required_before_closed_F_release=prior['required_before_F_release'],
        closed_F_future_release=oldF['released_future_bytes'],
        stopped_native_27_unused_future_released=release['released_future_bytes'],
        legacy_floor_bytes=50<<30,disk_buffer_bytes=8<<30,headroom_bytes=prior['headroom_bytes'],
        files_removed=[],physical_reservation=False)


def assets(context):
    s=context['s'];rel=context['relative']
    d09=read(checked(rel['D09_completion']));i=read(checked(d09['inputs']))
    panel=read(checked(i['panel']))['Dcheck'];metadata=read(checked(i['target_metadata']))['Dcheck']
    assert len(panel)==len(metadata)==400 and len({v['query_id'] for v in panel})==400
    for j,v in enumerate(panel):
        assert v['subject_id']=='sub-01' and v['partition']=='Dcheck'
        assert all(metadata[j][k]==x for k,x in v.items())
    groups={}
    for j,v in enumerate(panel):groups.setdefault(v['concept_id'],[]).append(j)
    assert len(groups)==40 and all(len(v)==10 for v in groups.values())
    old=[json.loads(v) for v in checked(d09['check_manifest']).read_text().splitlines()]
    assert len(old)==1200
    for j,v in enumerate(panel):
        for k,arm in enumerate(('parent','ordinary','risk')):
            row=old[3*j+k];assert row['arm']==arm and all(row[key]==value for key,value in v.items())
    return dict(d09=d09,inputs=i,queries=panel,metadata=metadata,old_records=old)


def fixed_pairs(metrics,queries):
    import numpy as np
    groups=[[i for i,v in enumerate(queries) if v['concept_id']==c] for c in sorted({v['concept_id'] for v in queries})]
    assert len(groups)==40 and all(len(v)==10 for v in groups)
    draws=np.random.default_rng(1729).integers(0,40,size=(1000,40));result={}
    for other in (*BASELINES,*ARMS[1:]):
        values={}
        for key in METRICS:
            delta=metrics[ARMS[0]][key]-metrics[other][key]
            cm=np.array([delta[g].mean() for g in groups]);ci=np.quantile(cm[draws].mean(1),[.025,.975])
            values[key]=dict(difference=float(delta.mean()),descriptive_concept_bootstrap95=ci.tolist(),positive_concepts=int((cm>0).sum()),concepts=40)
        result[ARMS[0]+'-'+other]=values
    return result
