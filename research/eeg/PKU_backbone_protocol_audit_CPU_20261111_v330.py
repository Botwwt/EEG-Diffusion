"""Full-image table audit and the predeclared fresh-MLP versus old-parent protocol check."""
import csv,hashlib,json,math,os,socket,time
from collections import defaultdict
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower();start=time.monotonic()
import numpy as np
R=Path('/dfs/share/maleiLab/wwt/eeg');source=R/'runs/EEG_backbone_scores_v322';out=R/'runs/EEG_backbone_protocol_v330';out.mkdir()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
done=json.loads((source/'analysis_completion.json').read_text());assert done['status']=='COMPLETE_FULL830_ALL5_BACKBONES_OWN_ORDINARY_THIRTEEN_ARMS_ALL14METRICS';assert sha(source/'all_paired_summaries.json')==done['summary_sha256'] and sha(source/'per_query.csv')==done['per_query_sha256'];s=json.loads((source/'all_paired_summaries.json').read_text());tables=defaultdict(dict);fields=list(s['means']['parent']);assert len(fields)==14
with (source/'per_query.csv').open(newline='') as f:
    for r in csv.DictReader(f):
        qi=int(r['query_index']);arm=r['arm'];assert qi not in tables[arm];v={k:float(r[k]) for k in fields};assert all(math.isfinite(x) for x in v.values());tables[arm][qi]=dict(row=int(r['array_row']),concept=r['concept_id'],metrics=v)
assert set(tables)==set(s['arms']) and len(tables)==13 and all(set(v)==set(range(830)) for v in tables.values())
for qi in range(830):assert len({(v[qi]['row'],v[qi]['concept']) for v in tables.values()})==1
concepts=[tables['parent'][qi]['concept'] for qi in range(830)];keys=sorted(set(concepts));assert len(keys)==83;groups={k:[i for i,c in enumerate(concepts) if c==k] for k in keys};assert all(len(v)==10 for v in groups.values());errors=[];near_zero=[]
for arm in s['arms']:
    for m in fields:errors.append(abs(math.fsum(tables[arm][i]['metrics'][m] for i in range(830))/830-s['means'][arm][m]))
for c in s['comparisons']:
    label=c['comparison'];m=c['metric']
    if label.endswith('_x_ordinary_interaction'):
        case=label[:-len('_x_ordinary_interaction')];terms=[(case+'_ordinary',1),(case+'_base',-1),('MLP17_ordinary',-1),('MLP17_base',1)]
    elif label.endswith('_ordinary_minus_base'):
        case=label[:-len('_ordinary_minus_base')];terms=[(case+'_ordinary',1),(case+'_base',-1)]
    else:left,right=label.split('_minus_');terms=[(left,1),(right,-1)]
    delta=[math.fsum(co*tables[arm][i]['metrics'][m] for arm,co in terms) for i in range(830)];errors.append(abs(math.fsum(delta)/830-c['mean']));per=[math.fsum(delta[i] for i in groups[k])/10 for k in keys];errors.extend(abs(a-b) for a,b in zip(per,c['per_concept']))
    for key,a,b in zip(keys,per,c['per_concept']):
        if (a>0,a<0)!=(b>0,b<0):assert max(abs(a),abs(b))<1e-7;near_zero.append(dict(comparison=label,metric=m,concept=key,independent_float64=a,server_reduction=b))
assert max(errors)<1e-7
draw=np.random.default_rng(1729).integers(0,83,size=(1000,83));checks=[]
for left,right in [('MLP17_base','parent'),('MLP17_ordinary','matched_ordinary')]:
    for m in fields:
        delta=[tables[left][i]['metrics'][m]-tables[right][i]['metrics'][m] for i in range(830)];per=np.array([math.fsum(delta[i] for i in groups[k])/10 for k in keys]);ci=np.quantile(per[draw].mean(1),[.025,.975]);checks.append(dict(comparison=left+'_minus_'+right,metric=m,mean=math.fsum(delta)/830,concept95=ci.tolist(),positive_concepts=int((per>0).sum()),negative_concepts=int((per<0).sum()),per_concept=per.tolist()))
result=dict(status='PASS_COMPLETE13_ARM_IMAGE_TABLE_AND_PREDECLARED_MLP_PROTOCOL_CHECKS',CPU_job=os.environ['SLURM_JOB_ID'],image_query_records=13*830,metric_cells=13*830*14,existing_contrast_cells=len(s['comparisons']),max_reconciliation_error=max(errors),near_zero_sign_disagreements=near_zero,protocol_comparisons=checks,source_table_sha256=done['per_query_sha256'],source_summary_sha256=done['summary_sha256'],source_sha256=sha(Path(__file__)),new_training=0,new_images=0,interpretation='Checks the fresh MLP training-protocol baseline against the original parent and matched ordinary. Does not refit or select models. All14 readouts and both fixed pairs reported; concept intervals descriptive on one exposed R4 subject, not independentH, training-seed uncertainty, noninferiority or SOTA.',seconds=time.monotonic()-start)
(out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');assert (out/'completion.json').stat().st_size<1<<20;print('PASS_COMPLETE_IMAGE_TABLE_AND_FRESH_MLP_PROTOCOL',flush=True)
