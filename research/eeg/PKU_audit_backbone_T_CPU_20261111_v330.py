"""Independent stdlib full-record audit of the frozen T-only measurement table."""
import csv,datetime,hashlib,json,math,os,socket,time
from collections import defaultdict
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower();start=time.monotonic();R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_backbone_T_representation_v330';summary=json.loads((out/'completion.json').read_text());data=out/'per_query.csv'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
assert sha(data)==summary['per_query_sha256'];split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());T=set(split['image_rows']['T']);assert len(T)==13230
fields=['F_MSE','F_cosine','F_same_concept_Top1','F_same_concept_2AFC','condition_norm','F_global_Top1','F_global_2AFC'];tables=defaultdict(dict);concept_of={};rows=0
with data.open(newline='') as stream:
    for r in csv.DictReader(stream):
        row=int(r['array_row']);key=(r['case'],r['exposure']);assert row in T and row not in tables[key];concept=r['concept_id'];assert int(r['fold'])==split['oof_fold_by_concept'][concept];concept_of[row]=concept;values={k:float(r[k]) for k in fields};assert all(math.isfinite(v) for v in values.values());tables[key][row]=values;rows+=1
assert rows==132300 and len(tables)==10 and all(set(t)==T for t in tables.values());keys=sorted(set(concept_of.values()));assert len(keys)==1323;groups={k:[r for r in T if concept_of[r]==k] for k in keys};assert all(len(v)==10 for v in groups.values());errors=[];near_zero_signs=[]
for item in summary['summary']:
    t=tables[(item['case'],item['exposure'])]
    for field in fields:errors.append(abs(math.fsum(v[field] for v in t.values())/13230-item['means'][field]))
for c in summary['comparisons']:
    left,right=c['comparison'].split('_minus_');field=c['metric'];tl=tables[(left,c['exposure'])];tr=tables[(right,c['exposure'])];delta={r:tl[r][field]-tr[r][field] for r in T};errors.append(abs(math.fsum(delta.values())/13230-c['mean']));per=[math.fsum(delta[r] for r in groups[k])/10 for k in keys];pos=sum(v>0 for v in per);neg=sum(v<0 for v in per)
    if (pos,neg)!=(c['positive_concepts'],c['negative_concepts']):
        ambiguous=sum(abs(v)<1e-12 for v in per);assert abs(pos-c['positive_concepts'])<=ambiguous and abs(neg-c['negative_concepts'])<=ambiguous;near_zero_signs.append(dict(comparison=c['comparison'],exposure=c['exposure'],metric=field,independent_positive=pos,server_positive=c['positive_concepts'],independent_negative=neg,server_negative=c['negative_concepts'],near_zero_concepts=ambiguous))
assert max(errors)<1e-10
proof=dict(status='PASS_INDEPENDENT_ALL132300_T_RECORDS_PERMISSIONS_MEANS_AND84_CONTRASTS',CPU_job=os.environ['SLURM_JOB_ID'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,views=10,concepts=1323,comparisons=len(summary['comparisons']),metric_cells=rows*7,max_mean_error=max(errors),near_zero_sign_disagreements=near_zero_signs,per_query_sha256=sha(data),summary_sha256=sha(out/'completion.json'),verifier_sha256=sha(Path(__file__)),neural_feature_generation_not_recomputed=True,bootstrap_not_independently_recomputed=True,new_training=0,new_images=0,seconds=time.monotonic()-start);p=out/'independent_CPU_audit_v330.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
