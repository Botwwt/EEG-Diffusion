"""Posthoc stimulus matching in native F coordinates, using closed caches only.

This is an interface diagnostic, not a test of all information recoverable by an
arbitrarily retrained decoder. Query F is a diagnostic target, never an input to G.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,datetime,hashlib,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_bytes())
def pin(p):
    p=Path(p)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def metrics(scores,queries,gallery):
    assert scores.shape==(80,400) and np.isfinite(scores).all()
    by_concept={c:[j for j,q in enumerate(gallery) if q['concept_id']==c] for c in {q['concept_id'] for q in gallery}}
    assert len(by_concept)==40 and {len(v) for v in by_concept.values()}=={10}
    rows=[]
    for i,q in enumerate(queries):
        assert q==gallery[i]; s=scores[i]; group=by_concept[q['concept_id']]
        positive=float(s[i]); other=[j for j in range(400) if j!=i]; within=[j for j in group if j!=i]
        def afc(indices):
            a=s[indices];return (np.count_nonzero(a<positive)+.5*np.count_nonzero(a==positive))/len(indices)
        def top(indices):
            a=s[indices]; maximum=a.max();return float(positive==maximum)/np.count_nonzero(a==maximum)
        rows.append([positive,afc(other),top(range(400)),afc(within),top(group),positive-float(s[within].max())])
    return np.asarray(rows)

def main(transmission,terminal,output):
    begin=time.monotonic();transmission=Path(transmission);terminal=Path(terminal)
    source=read(transmission/'source_and_scope.json')
    assert source['status'] in ('COMPLETE_POSTHOC_CLOSED_FIVE_ARM_TRANSMISSION_AND_LOG_EXTRACTION','COMPLETE_CLOSED_NOO_THREE_FIT_EIGHT_PATH_TRANSMISSION_EXTRACTION')
    done=read(terminal/'actual/controller_completion.json')
    assert done['status']=='COMPLETE_FIXED_FULL_T_AND_NATIVE_IMAGE_STUDY' and done['registration']==source['protocol']
    assert done['final_audit']['all_images_verified']
    local_audit=read(terminal/'independent_local_statistics_and_provenance.json')
    assert local_audit['status'].startswith('PASS_LOCAL_CLOSED_CORE_')
    observed=transmission/'observed_arrays.npz';assert pin(observed)['sha256']==source['NPZ_sha256']
    arrays=np.load(observed,allow_pickle=False)
    gallery=read(terminal/'reference/panel.json')['Dcheck'];assert len(gallery)==400
    teacher_root=ROOT/'audit/teacher_reference_20261027_v1/actual/candidate/inputs'
    completion=read(teacher_root/'completion.json');lock_path=teacher_root/'all400_teacher_lock.json';lock=read(lock_path)
    assert pin(lock_path)['sha256']==completion['lock']['sha256']
    assert completion['panel']['sha256']==pin(terminal/'reference/panel.json')['sha256']
    assert lock['arms'][0]=='oracle_F' and lock['diagnostic_only'] and not lock['normal_inference_supported']
    assert [q['query_id'] for q in gallery]==lock['query_ids']
    assert [q['array_row'] for q in gallery]==[row[0] for row in lock['source_array_rows']]
    assert [row[0] for row in lock['source_query_indices']]==list(range(400))
    teachers=teacher_root/'teacher_conditions_all400.npy';assert pin(teachers)['sha256']==lock['conditions']['sha256']
    visual=np.load(teachers,allow_pickle=False);assert visual.shape==(400,3,1024) and np.isfinite(visual).all()
    visual=visual[:,0].astype(float);visual/=np.linalg.norm(visual,axis=1,keepdims=True)
    methods=sorted(k[len('heldout80__'):-len('__raw')] for k in arrays.files if k.startswith('heldout80__') and k.endswith('__raw'))
    assert len(methods) in (7,8) and 'parent' in methods and 'ordinary' in methods
    query=gallery[:80]; concepts=sorted({q['concept_id'] for q in query});assert len(concepts)==8
    groups=[np.array([i for i,q in enumerate(query) if q['concept_id']==c]) for c in concepts]
    assert {len(g) for g in groups}=={10}
    columns=['target_F_cosine','F_gallery_AFC','F_gallery_top1_fractional_ties','F_within10_AFC','F_within10_top1_fractional_ties','F_within10_margin_to_hardest_wrong']
    values={}; maximum=0.; means={}
    for arm in methods:
        means[arm]={}
        for kind in ('raw','conditions'):
            c=arrays[f'heldout80__{arm}__{kind}'].astype(float);assert c.shape==(80,1024)
            norms=np.linalg.norm(c,axis=1);assert norms.min()>0;c/=norms[:,None]
            scores=c@visual.T
            scalar=np.array([[np.dot(row,target) for target in visual] for row in c])
            error=float(np.max(np.abs(scores-scalar)));assert error<2e-12;maximum=max(maximum,error)
            result=metrics(scores,query,gallery);replayed=metrics(scalar,query,gallery)
            assert np.max(np.abs(result-replayed))<2e-12
            key=arm+'__'+kind;values[key]=result;means[arm][kind]={k:float(result[:,j].mean()) for j,k in enumerate(columns)}
    # A single fixed F vector contains no query identity; fractional ties provide
    # exactly 1/10 within-concept Top1 averaged over all ten targets in each group.
    fixed=metrics(np.broadcast_to(visual[0]@visual.T,(80,400)),query,gallery)
    assert abs(fixed[:,4].mean()-.1)<1e-12
    rng=np.random.default_rng(1729);draws=rng.integers(0,len(groups),(1000,len(groups)))
    contrasts={}
    for arm in methods:
        for kind,base in [('raw','parent'),('conditions','ordinary')]:
            a=values[arm+'__'+kind]-values[base+'__'+kind]
            per=np.stack([a[g].mean(0) for g in groups]);bootstrap=per[draws].mean(1)
            contrasts[arm+'__'+kind+'_minus_'+base]={k:dict(difference=float(a[:,j].mean()),descriptive_concept_bootstrap95=np.quantile(bootstrap[:,j],[.025,.975]).tolist()) for j,k in enumerate(columns)}
    out=Path(output);out.mkdir(exist_ok=False,parents=True)
    np.savez_compressed(out/'all_native_coordinate_metrics.npz',**values)
    report=dict(status='COMPLETE_POSTHOC_CLOSED_NATIVE_F_COORDINATE_STIMULUS_DIAGNOSIS',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_completion=source['completion'],source_protocol=source['protocol'],source_observed_arrays=pin(observed),source_teacher_lock=pin(lock_path),source_teacher_conditions=pin(teachers),source_panel=pin(terminal/'reference/panel.json'),
        subjects=1,subject='sub-01',EEG_repeats=4,queries=80,concepts=8,candidate_gallery=400,columns=columns,means=means,contrasts=contrasts,
        independent_scalar_dot_max_error=maximum,constant_condition_within10_Top1=float(fixed[:,4].mean()),
        scope='Fixed first80 final cached c to original F cosine matching. Changed P can coadapt with c; poor native-F matching is not proof that no other decoder could recover stimulus information.',
        query_visual_features_used_only_as_diagnostic_targets=True,not_normal_inference=True,not_independent_H=True,
        not_an_information_bits_measure=True,no_unique_cause_proof=True,no_gate_or_model_selection_change=True,new_updates=0,new_images=0,new_GPU_jobs=0,seconds=time.monotonic()-begin)
    (out/'analysis.json').write_bytes((json.dumps(report,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(status=report['status'],means={a:{kind:{k:v for k,v in result.items() if k in ('target_F_cosine','F_within10_top1_fractional_ties','F_within10_margin_to_hardest_wrong')} for kind,result in kinds.items()} for a,kinds in means.items()},maximum_scalar_error=maximum,seconds=report['seconds'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--transmission',required=True);p.add_argument('--terminal',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.transmission,a.terminal,a.output)
