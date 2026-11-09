"""Independent full400 head equations plus all actual ranks/contrasts/bootstrap."""
import argparse,datetime,hashlib,json,os
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
from scipy.special import erf
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();E=a.directory
read=lambda p:json.loads(Path(p).read_bytes())
for name,item in read(E/'received_manifest.json')['manifest'].items():
    b=(E/name).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
s=read(E/'protocol/study.json');done=read(E/'actual/completion.json');arms=s['arms'];assert len(arms)==4
bank=np.load(E/'actual/terminal_CPU_export/native400_input_bank.npz',allow_pickle=False);c=bank['condition'].astype(np.float64);w=bank['gamma'].astype(np.float64)[None,:];b=bank['beta'].astype(np.float64)[None,:];head_errors={}
for arm in arms:
    weights=np.load(E/'actual/terminal_CPU_export'/(arm+'_weights.npz'),allow_pickle=False);cal=read(E/'actual'/arm/'calibration.json');alpha=cal['selected_strength']
    assert alpha==cal['strengths'][min(range(5),key=lambda i:(cal['Dfit_token_errors'][i],i))]
    u=(c-b)/w;r2=np.sum(u*u,axis=1,keepdims=True);v=u@weights['trunk.0.weight'].astype(np.float64).T+weights['trunk.0.bias'];h=.5*v*(1+erf(v/np.sqrt(2)))
    d=h@weights['direction.weight'].astype(np.float64).T+weights['direction.bias'];d-=d.mean(axis=1,keepdims=True);d-=u*np.sum(d*u,axis=1,keepdims=True)/r2
    delta=.25*alpha*np.sqrt(r2)*d/np.sqrt(r2+np.sum(d*d,axis=1,keepdims=True));scale=np.sqrt(r2)/np.linalg.norm(u+delta,axis=1,keepdims=True);ideal=c+w*((scale-1)*u+scale*delta)
    cpu=np.load(E/'actual/CPU_audit'/f'expected400_{arm}.npy',allow_pickle=False);gpu=np.load(E/'actual/images'/f'conditions400_{arm}.npy',allow_pickle=False).astype(np.float64)
    cpuerr=float(np.abs(cpu-ideal).max());scaled=float((np.abs(gpu-ideal)/(1+np.abs(ideal))).max());rel=float(np.linalg.norm(gpu-ideal)/np.linalg.norm(ideal));assert cpuerr<1e-10 and scaled<2e-6 and rel<1e-5
    head_errors[arm]=dict(queries=400,CPU64_replay_error=cpuerr,GPU_scaled_error=scaled,GPU_relative_error=rel,selected_strength=alpha)
target=np.load(E/'reference/D09_targets.npy',allow_pickle=False).astype(np.float64);target/=np.linalg.norm(target,axis=1,keepdims=True);old=np.load(E/'reference/D09_old3.npy',allow_pickle=False);starting=np.load(E/'reference/starting_F.npy',allow_pickle=False);queries=read(E/'reference/panel.json')['Dcheck']
errors=[];cells=0;nearzero=[]
def measure(f,n):
    f=f.astype(np.float64);f/=np.linalg.norm(f,axis=1,keepdims=True);sim=f@target.T;out={k:np.empty(n) for k in ('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc')}
    for i in range(n):
        row=sim[i];v=row[i];ids=[j for j,q in enumerate(queries) if q['concept_id']==queries[i]['concept_id']];local=row[ids]
        out['target_cosine'][i]=v;out['DINO2AFC'][i]=(np.count_nonzero(row<v)+.5*(np.count_nonzero(row==v)-1))/399;out['gallery_top1'][i]=int(np.argmax(row)==i)
        out['within_concept10_top1'][i]=int(ids[np.argmax(local)]==i);out['within_concept10_two_afc'][i]=(np.count_nonzero(local<v)+.5*(np.count_nonzero(local==v)-1))/9
    return out
stages=[n for n in (80,400) if (E/'actual/images'/f'score{n}.json').is_file()]
for n in stages:
    score=read(E/'actual/images'/f'score{n}.json');raw=np.load(E/'actual/images'/f'all_query_values{n}.npz',allow_pickle=False)
    metrics={name:measure(old[:n,j],n) for j,name in enumerate(('parent','ordinary','risk'))};metrics['starting_F']=measure(starting[:n],n)
    for arm in arms:metrics[arm]=measure(np.load(E/'actual/images'/f'features{n}_{arm}.npy',allow_pickle=False),n)
    groups=[[i for i in range(n) if queries[i]['concept_id']==c] for c in sorted({q['concept_id'] for q in queries[:n]})];ng=len(groups);draw=np.random.default_rng(1729).integers(0,ng,size=(1000,ng));counts=np.asarray([np.bincount(x,minlength=ng) for x in draw])/ng
    for name,fields in metrics.items():
        for key,x in fields.items():errors.extend([abs(x.mean()-score['means'][name][key]),float(np.abs(x-raw[name+'__'+key]).max())]);cells+=n
    for arm in arms:
        for base in metrics:
            if arm==base:continue
            for key in metrics[arm]:
                delta=metrics[arm][key]-metrics[base][key];cm=np.asarray([delta[g].sum()/len(g) for g in groups]);ci=np.quantile(counts@cm,[.025,.975]);v=score['contrasts'][arm][base][key]
                errors.append(abs(delta.mean()-v['difference']));errors.extend(np.abs(ci-v['descriptive_concept_bootstrap95']).tolist());errors.append(float(np.abs(delta-raw[arm+'_minus_'+base+'__'+key]).max()));cells+=n
                positive=int((cm>0).sum())
                if positive!=v['positive_concepts']:
                    stable=int((cm>1e-12).sum());ambiguous=int((np.abs(cm)<=1e-12).sum());assert stable<=positive<=stable+ambiguous and stable<=v['positive_concepts']<=stable+ambiguous
                    nearzero.append(dict(stage=n,arm=arm,reference=base,metric=key,local=positive,production=v['positive_concepts'],ambiguous=ambiguous))
assert max(errors)<1e-12 and max(stages)==done['final_stage']
r=dict(status='PASS_LOCAL_ALL400_HEADS_AND_ALL_ACTUAL_IMAGE_FEATURE_STATISTICS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),head_queries=1600,head_equations=head_errors,
    stages=stages,actual_metric_and_contrast_cells=cells,statistic_checks=len(errors),maximum_error=max(errors),near_zero_positive_count_ambiguities=nearzero,
    pixels_audited_on_server_only=True,local_vision_reencoding=False,not_independent_H=True,not_SOTA=True)
(E/'independent_local_statistics_and_heads.json').write_bytes((json.dumps(r,indent=2)+'\n').encode());print(json.dumps(r))
