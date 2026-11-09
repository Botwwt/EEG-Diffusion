"""Fixed four-arm, single-subject R80 measurement-count image experiment."""
import json,os,sys
from pathlib import Path
import numpy as np

ARMS=('EMA_original','EMA_count','T_population','T_count')
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
METRICS=('target_cosine','DINO2AFC','gallery_top1','gallery_top5','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')+PAPER
CONTRASTS={'main_count':{'EMA_count':1,'EMA_original':-1},
 'population_count':{'T_count':1,'T_population':-1},
 'population_only':{'T_population':1,'EMA_original':-1},
 'count_x_population':{'T_count':1,'T_population':-1,'EMA_count':-1,'EMA_original':1}}
IDENTITY=('query_id','subject_id','stimulus_id','concept_id','array_row','image_seed','repeat_protocol')
_c=None

def allocated(root):
    return sum(p.stat().st_blocks*512 for p in Path(root).rglob('*') if p.is_file()) if Path(root).exists() else 0

def setup(path):
    global _c
    q=json.loads(Path(path).read_bytes())
    assert q['status']=='LOCKED_COMPLETE_R80_200_FOUR_ARM_COUNT_IMAGE_STUDY'
    assert q['host']=='old' and q['seed']==1729 and q['subject']=='sub-01'
    assert q['arms']==list(ARMS) and q['metrics']==list(METRICS) and q['contrasts']==CONTRASTS
    assert q['new_images']==800 and q['new_models']==q['new_updates']==0
    assert q['maximum_output_bytes']==384<<20 and q['all800_before_effect_read']
    if _c is None:
        sys.path.insert(0,str(Path(q['reference_policy']['path']).parent))
        import enigma_native_reference_contract_20261104_v1 as P
        up=P.setup(q['reference_policy']['path']);K=up['K']
        assert q['native_pixel_reference']==up['baseline']['direct'][0]
        assert q['native_pixel_conditions']==up['q']['baseline_conditions']
        for pin in q['sources'].values():K.checked(pin)
        for pin in q['runtime_sources'].values():K.checked(pin)
        for key in ('reference_policy','reference_completion','measurement_policy','measurement_completion',
                    'measurement_conditions','measurement_CPU_replay','reserve_ledger','identity_table',
                    'EEG_R80','test_layout','test_image_ids','visual_ordering','gallery_manifest','original_metric_assets'):
            K.checked(q[key])
        assert K.read(q['reference_completion']['path'])['status']=='COMPLETE_NATIVE63_FULL150_OOF_AND_FULL400_REFERENCE'
        done=K.read(q['measurement_completion']['path'])
        assert done['status']=='COMPLETE_ALL_T_MEASUREMENT_CALIBRATION_AND_FROZEN_R80_4ARMS'
        assert done['conditions']==q['measurement_conditions'] and done['new_images']==0
        proof=K.read(q['measurement_CPU_replay']['path'])
        assert proof['status']=='PASS_NATIVE_CPU64_PREFIX_AND_ALL800_CPU32_R80_CONDITION_REPLAY'
        rows=K.read(q['identity_table']['path']);assert len(rows)==200
        assert [r['array_row'] for r in rows]==list(range(200))
        assert len({r['query_id'] for r in rows})==len({r['concept_id'] for r in rows})==len({r['stimulus_id'] for r in rows})==200
        ids=K.read(q['test_image_ids']['path']);assert ids==[Path(r['stimulus_id']).stem for r in rows]
        visual=sorted((v for v in K.read(q['visual_ordering']['path']) if v['split']=='test'),key=lambda v:v['index'])
        assert [v['index'] for v in visual]==list(range(200))
        assert [Path(v['member']).stem for v in visual]==ids
        layout=K.read(q['test_layout']['path']);assert layout['selected_image_indices']==list(range(200))
        assert layout['shape']==[200,63,250] and layout['raw_shape']==[200,80,63,250]
        gallery=[json.loads(s) for s in K.checked(q['gallery_manifest']).read_text().splitlines()]
        assert len(gallery)==200
        by_id={r['stimulus_id']:r for r in gallery}
        for r in rows:
            g=by_id[r['stimulus_id']]
            assert r['target']['path']==g['image_path'] and r['target']['sha256']==g['image_sha256'] and r['concept_id']==g['concept_id']
        values=np.load(K.checked(q['measurement_conditions']),allow_pickle=False)
        assert values.shape==(200,4,1024) and values.dtype==np.float32 and np.isfinite(values).all()
        _c=dict(up,q=q,K=K,P=P,up=up,rows=rows,values=values,out=Path(q['output_root']),ledger=K.read(q['reserve_ledger']['path']))
    assert _c['q']==q
    return _c

def resources(c,starting=False):
    K=c['K'];q=c['q'];l=c['ledger']
    assert l['status']=='REVIEWED_FACTORIAL_AND_NATIVE_REFERENCE_TERMINAL_UNUSED_ALLOWANCES_ONLY'
    assert l['legacy_floor_bytes']==50<<30 and l['disk_buffer_bytes']==8<<30 and l['file_deletion']==0
    assert len(l['settlements'])==2
    credit=0
    for r in l['settlements']:
        p=K.read(K.checked(r['policy']));d=K.read(K.checked(r['completion']));la=K.read(K.checked(r['launch']))
        assert d['status']==r['completion_status'] and not K.process(la['PID'],la['start_ticks'])['live']
        assert allocated(r['project'])==r['actual_allocated_bytes'] and p['maximum_output_bytes']==r['maximum_output_bytes']
        assert r['unused_future_allowance']==r['maximum_output_bytes']-r['actual_allocated_bytes']
        for e in d['events']:
            if e['GPU']:
                j=K.read(K.checked(e['receipt']));assert j['status']=='COMPLETED' and j['exit_code']==0
                assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
        credit+=r['unused_future_allowance']
    rr=K.read(K.checked(l['reference_recovery_launch']))
    assert not K.process(rr['PID'],rr['start_ticks'])['live']
    assert credit==l['settled_unused_future_bytes']
    prior=l['unsettled_all_previous_forecasts_bytes']-credit
    assert prior>=l['legacy_floor_bytes']+l['disk_buffer_bytes']
    used=allocated(q['project_root']);assert used<=q['maximum_output_bytes']
    stat=os.statvfs(Path(q['project_root']).parent);free=stat.f_bavail*stat.f_frsize
    required=prior+q['maximum_output_bytes']-used
    assert free>=required,dict(free=free,required=required)
    info={s.split(':',1)[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if ':' in s and len(s.split())>=2 and s.split()[1].isdigit()}
    headroom=info['MemAvailable'];mx=Path('/sys/fs/cgroup/memory.max');cur=Path('/sys/fs/cgroup/memory.current')
    if mx.exists() and cur.exists() and mx.read_text().strip()!='max':headroom=min(headroom,int(mx.read_text())-int(cur.read_text()))
    assert headroom>=(24<<30 if starting else 2<<30)
    return dict(required_prior_bytes=prior,required_including_this_project=required,free_bytes=free,
        own_used_bytes=used,own_remaining_bytes=q['maximum_output_bytes']-used,settled_unused_bytes=credit,
        all_other_forecasts_held_at_snapshot=True,legacy50GiB_buffer8GiB_unchanged=True,actual_CPU_headroom_bytes=headroom)

def metrics(features,target):
    a=np.asarray(features,np.float32);b=np.asarray(target,np.float32)
    assert a.shape==b.shape==(200,768)
    a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True)
    matrix=a@b.T;correct=np.diag(matrix)
    result=dict(target_cosine=correct.astype(np.float64),
        DINO2AFC=((matrix<correct[:,None]).sum(1)+.5*((matrix==correct[:,None]).sum(1)-1))/199,
        gallery_top1=(matrix.argmax(1)==np.arange(200)).astype(np.float64),
        gallery_top5=(np.argsort(-matrix,axis=1,kind='stable')[:,:5]==np.arange(200)[:,None]).any(1).astype(np.float64))
    assert all(np.isfinite(v).all() for v in result.values())
    return result,matrix

def paired(arrays):
    draw=np.random.default_rng(1729).integers(0,200,(1000,200));result={};diffs={}
    for name,coefs in CONTRASTS.items():
        result[name]={}
        for key in METRICS:
            d=sum(v*arrays[a][key] for a,v in coefs.items());diffs[name+'__'+key]=d
            assert d.shape==(200,)
            valid=np.isfinite(d)
            result[name][key]=dict(difference=float(d.mean()) if valid.all() else None,
                descriptive_concept_bootstrap95=np.quantile(d[draw].mean(1),[.025,.975]).tolist() if valid.all() else None,
                defined_queries=int(valid.sum()),total_queries=200,positive_concepts=int((d>0).sum()),coefficients=coefs)
    return result,diffs
