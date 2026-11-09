"""Full fixed2000 paired readout; raw author distances retain their directions."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def readcsv(p):
 with Path(p).open(newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==2000 and len({r['query_id'] for r in rows})==2000;return {r['query_id']:r for r in rows}
def run(path):
 e=json.loads(Path(path).read_bytes());s=json.loads(Path(e['protocol']['path']).read_bytes());assert sha(e['protocol']['path'])==e['protocol']['sha256']
 root=Path(s['output_root']);out=root/'evaluation';audit=json.loads((root/'ordinary/artifact_audit.json').read_bytes());assert audit['queries']==2000
 common=json.loads((out/'scores/summary.json').read_bytes());paper=json.loads((out/'original_paper_metrics/completion.json').read_bytes());assert common['status']=='COMPLETED' and paper['queries']==2000
 for item in (e['baseline_common_csv'],e['baseline_paper_csv']):assert sha(item['path'])==item['sha256']
 baseline=readcsv(e['baseline_common_csv']['path']);candidate=readcsv(out/'scores/per_query.csv');bp=readcsv(e['baseline_paper_csv']['path']);cp=readcsv(out/'original_paper_metrics/per_query.csv');assert set(baseline)==set(candidate)==set(bp)==set(cp)
 subjects=sorted({r['subject'] for r in candidate.values()});concepts=sorted({r['concept_id'] for r in candidate.values()});assert len(subjects)==10 and len(concepts)==200
 keys=sorted(candidate);cells={(candidate[k]['subject'],candidate[k]['concept_id']):k for k in keys};assert len(cells)==2000
 for key in keys:
  for field in ('subject','stimulus_id','concept_id','image_seed','score_target_path'):
   assert candidate[key][field]==baseline[key][field]==bp[key][field]==cp[key][field]
  assert candidate[key]['generated_path']!=baseline[key]['generated_path']
 rng=np.random.default_rng(1729);draws=[(rng.integers(10,size=10),rng.integers(200,size=200)) for _ in range(1000)]
 groups=[('common',['DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED'],candidate,baseline),('paper',['eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav'],cp,bp)]
 results={};table=[];max_error=0.
 for kind,names,left,right in groups:
  for metric in names:
   defined=[k for k in keys if left[k][metric] not in ('','None') and right[k][metric] not in ('','None')]
   if len(defined)!=2000:
    results[metric]=dict(status='UNKNOWN_UNDEFINED_FULL_DENOMINATOR_RETAINED',defined=len(defined),expected=2000);continue
   a=np.array([[float(left[cells[(sub,con)]][metric]) for con in concepts] for sub in subjects]);b=np.array([[float(right[cells[(sub,con)]][metric]) for con in concepts] for sub in subjects]);d=a-b;assert np.isfinite(d).all()
   direct=np.array([d[np.ix_(si,ci)].mean() for si,ci in draws])
   separate=np.array([np.einsum('s,sc,c->',np.bincount(si,minlength=10),d,np.bincount(ci,minlength=200))/2000 for si,ci in draws])
   error=float(np.max(np.abs(direct-separate)));assert error<1e-12;max_error=max(max_error,error)
   results[metric]=dict(status='COMPLETE',ordinary_mean=float(a.mean()),parent_mean=float(b.mean()),paired_ordinary_minus_parent=float(d.mean()),descriptive_subject_concept_bootstrap95=np.quantile(direct,[.025,.975]).tolist(),direction='lower_is_better' if metric in ('eval_effnet','eval_swav') else 'higher_is_better',defined=2000)
   for subidx,sub in enumerate(subjects):
    for conidx,con in enumerate(concepts):table.append(dict(metric=metric,subject=sub,concept_id=con,query_id=cells[(sub,con)],difference=float(d[subidx,conidx])))
 outpath=out/'paired_comparison';outpath.mkdir()
 with (outpath/'all_paired_differences.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 def positive(m,minimum=0.):
  x=results[m];return x.get('status')=='COMPLETE' and x['paired_ordinary_minus_parent']>=minimum and x['descriptive_subject_concept_bootstrap95'][0]>0
 gate=positive('DINO2AFC',.01) and positive('SEED') and results['ObjectF1'].get('paired_ordinary_minus_parent',-1)>=0
 result=dict(status='COMPLETE_FIXED_ORDINARY_R80_FULL2000_PAIRED_COMPARISON',subjects=10,queries=2000,gallery_targets=200,results=results,practical_development_gate_passed=bool(gate),bootstrap_rounds=1000,seed=1729,independent_weighted_bootstrap_max_error=max_error,training_seed_variability_assessed=False,
  ordinary_checkpoint=s['ordinary'],training_seed=s['training_seed'],ordinary_strength=s['ordinary_strength'],protocol=e['protocol'],execution=ref(path),candidate_manifest=audit['manifest'],source=ref(__file__),full_queries_retained=True,
  limits=['ExposedR80 benchmark, not independentH.','Ordinary is a prespecified replicateT1323 fit, not officialfull1654 componentfit.','Oneimage vs author ten-reconstruction metric mean; singletrainingseed.','No new mechanism or image-quality noninferiority established by this comparison.'],SOTA_established=False)
 with (outpath/'summary.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=result['status'],practical_gate=bool(gate),DINO=results['DINO2AFC'],SEED=results['SEED'])))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execution',required=True);q=p.parse_args();run(q.execution)
