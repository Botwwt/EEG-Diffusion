"""Independent local CSV identity/SEED/paired-bootstrap audit of complete R80."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=h)
def load(p):
 with Path(p).open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
 assert len(rows)==2000 and len({r['query_id'] for r in rows})==2000
 return {r['query_id']:r for r in rows}
def run(folder):
 folder=Path(folder);manifest=json.loads((folder/'receiving_completion.json').read_bytes());assert manifest['status']=='ALL_COMPLETED_R80_EVIDENCE_FILES_EXACTLY_RECEIVED'
 for entry in manifest['entries']:
  p=folder/entry['name'];assert p.resolve().is_relative_to(folder.resolve()) and ref(p)['sha256']==entry['sha256'] and p.stat().st_size==entry['bytes']
 summary=json.loads((folder/'metadata/paired_summary.json').read_bytes());assert summary['status']=='COMPLETE_FIXED_ORDINARY_R80_FULL2000_PAIRED_COMPARISON'
 bc=load(folder/'baseline/common.csv');cc=load(folder/'candidate/common.csv');bp=load(folder/'baseline/paper.csv');cp=load(folder/'candidate/paper.csv');keys=sorted(cc);assert set(keys)==set(bc)==set(bp)==set(cp)
 for k in keys:
  for field in ('subject','stimulus_id','concept_id','image_seed','score_target_path'):assert len({x[k][field] for x in (bc,cc,bp,cp)})==1
  assert cc[k]['generated_path']!=bc[k]['generated_path']
 max_seed_error=0.
 for table in (bc,cc):
  for row in table.values():
   error=abs(float(row['SEED'])-(float(row['EffNet'])+float(row['CapSim'])+float(row['ObjectF1']))/3);max_seed_error=max(max_seed_error,error);assert error<1e-12
   assert 0<=float(row['DINO2AFC'])<=1 and float(row['DINO_Top1'])<=float(row['DINO_Top5'])
 subjects=sorted({r['subject'] for r in cc.values()});concepts=sorted({r['concept_id'] for r in cc.values()});assert len(subjects)==10 and len(concepts)==200
 cell={(cc[k]['subject'],cc[k]['concept_id']):k for k in keys};assert len(cell)==2000
 rng=np.random.default_rng(1729);ws=[];wc=[]
 for _ in range(1000):ws.append(np.bincount(rng.integers(10,size=10),minlength=10));wc.append(np.bincount(rng.integers(200,size=200),minlength=200))
 ws=np.array(ws);wc=np.array(wc);results={};max_error=0.
 groups=[(['DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED'],cc,bc),(['eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav'],cp,bp)]
 for names,candidate,baseline in groups:
  for name in names:
   good=[k for k in keys if candidate[k][name] not in ('','None') and baseline[k][name] not in ('','None')]
   if len(good)!=2000:
    assert summary['results'][name]['status']=='UNKNOWN_UNDEFINED_FULL_DENOMINATOR_RETAINED';results[name]=dict(status='UNKNOWN',defined=len(good),full_denominator=2000);continue
   a=np.array([[float(candidate[cell[(s,c)]][name]) for c in concepts] for s in subjects]);b=np.array([[float(baseline[cell[(s,c)]][name]) for c in concepts] for s in subjects]);d=a-b;assert np.isfinite(a).all() and np.isfinite(b).all()
   boot=np.einsum('bs,sc,bc->b',ws,d,wc,optimize=True)/2000
   computed=[float(a.mean()),float(b.mean()),float(d.mean()),*np.quantile(boot,[.025,.975]).tolist()];old=summary['results'][name];expected=[old['ordinary_mean'],old['parent_mean'],old['paired_ordinary_minus_parent'],*old['descriptive_subject_concept_bootstrap95']];error=max(abs(x-y) for x,y in zip(computed,expected));assert error<5e-10;max_error=max(max_error,error)
   results[name]=dict(status='PASS',ordinary_mean=computed[0],parent_mean=computed[1],paired_difference=computed[2],descriptive_subject_concept_bootstrap95=computed[3:],absolute_recompute_error=error)
 def positive(name,minimum=0.):
  x=results[name];return x['status']=='PASS' and x['paired_difference']>=minimum and x['descriptive_subject_concept_bootstrap95'][0]>0
 gate=positive('DINO2AFC',.01) and positive('SEED') and results['ObjectF1'].get('paired_difference',-1)>=0
 assert bool(gate)==summary['practical_development_gate_passed']
 result=dict(status='PASS_ALL2000_IDENTITIES_SEED_AND_16_PAIRED_METRICS_LOCAL_INDEPENDENT_RECOMPUTATION',queries=2000,subjects=10,concepts=200,scalar_pairs=2000*len(results),metric_count=len(results),results=results,maximum_statistic_error=max_error,maximum_SEED_identity_error=max_seed_error,practical_development_gate_passed=bool(gate),source=ref(__file__),receiving_completion=ref(folder/'receiving_completion.json'),numpy=np.__version__,bootstrap_rounds=1000,seed=1729,scope='Independent local CSV/statistic arithmetic only. Does not re-encode images, rerun detectors, fit models, cover training seeds, or establish SOTA.',SOTA_established=False)
 target=folder/'independent_local_statistics.json'
 with target.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ('status','queries','maximum_statistic_error','practical_development_gate_passed')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();run(a.folder)
