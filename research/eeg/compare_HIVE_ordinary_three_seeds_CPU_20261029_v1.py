"""All fixed seeds1729/2027/3407, full16 metrics; never choose best seed."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from brainhive_seed_joint_statistics_20261029_v1 import joint_statistics
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):assert ref(x['path'])==x;return Path(x['path'])
def csvrows(path):
 with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==2000 and len({r['query_id'] for r in rows})==2000;return {r['query_id']:r for r in rows}
def run(path):
 s=read(path);assert s['training_seeds']==[1729,2027,3407] and s['host']=='old';runs=s['runs'];assert [r['seed'] for r in runs]==s['training_seeds']
 output=Path(s['joint_output']);output.mkdir();base=csvrows(checked(s['baseline_common_csv']));base_paper=csvrows(checked(s['baseline_paper_csv']));keys=sorted(base);subjects=sorted({r['subject'] for r in base.values()});concepts=sorted({r['concept_id'] for r in base.values()});assert len(subjects)==10 and len(concepts)==200
 cells={(base[k]['subject'],base[k]['concept_id']):k for k in keys};assert len(cells)==2000;tables=[];all_paths=set();all_inodes=set();sources=[]
 for run in runs:
  lifecycle=read(checked(run['lifecycle_completion']));assert lifecycle['queries']==2000 and lifecycle['status']=='COMPLETE_FIXED_ORDINARY_R80_GENERATION_AND_FULL_PAIRED_EVALUATION'
  common=csvrows(checked(run['common_csv']));paper=csvrows(checked(run['paper_csv']));summary=read(checked(run['paired_summary']));assert summary['ordinary_checkpoint']==run['checkpoint']
  if run['seed']!=1729:assert summary['training_seed']==run['seed']
  assert set(common)==set(paper)==set(base)==set(base_paper)
  for k in keys:
   for field in ('subject','concept_id','stimulus_id','image_seed','score_target_path'):assert common[k][field]==paper[k][field]==base[k][field]==base_paper[k][field]
   assert common[k]['generated_path']==paper[k]['generated_path'] and common[k]['generated_path'] not in all_paths;all_paths.add(common[k]['generated_path']);stat=Path(common[k]['generated_path']).stat();identity=(stat.st_dev,stat.st_ino);assert identity not in all_inodes;all_inodes.add(identity)
  tables.append((common,paper));sources.append(run)
 groups=[('common',['DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED'],0,base),('paper',['eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav'],1,base_paper)];results={};delta_arrays={}
 for kind,names,position,baseline in groups:
  for metric in names:
   defined=sum(t[position][k][metric] not in ('','None') and baseline[k][metric] not in ('','None') for t in tables for k in keys)
   if defined!=6000:results[metric]=dict(status='UNKNOWN_UNDEFINED_FULL6000_DENOMINATOR_RETAINED',defined=defined,expected=6000);continue
   d=np.array([[[float(t[position][cells[(sub,con)]][metric])-float(baseline[cells[(sub,con)]][metric]) for con in concepts] for sub in subjects] for t in tables]);assert np.isfinite(d).all();result=joint_statistics(d);result.update(status='COMPLETE',direction='lower_is_better' if metric in ('eval_effnet','eval_swav') else 'higher_is_better');results[metric]=result;delta_arrays[metric]=d
 def supported(metric,minimum=0.):
  x=results[metric];return x.get('status')=='COMPLETE' and x['mean_difference']>=minimum and x['descriptive_seed_subject_concept_bootstrap95'][0]>0
 gate=supported('DINO2AFC',.01) and supported('SEED') and results['ObjectF1'].get('mean_difference',-1)>=0 and results['DINO2AFC'].get('positive_training_runs')==3
 np.savez_compressed(output/'all16_paired_3x10x200_differences.npz',**delta_arrays)
 result=dict(status='COMPLETE_ALL3_PREDECLARED_TRAINING_RUNS_FULL16_R80_REPORT',training_seeds=s['training_seeds'],queries_per_run=2000,training_run_mean_is_primary=True,all_runs_reported_no_best_seed=True,results=results,descriptive_robustness_gate_passed=bool(gate),source_study=ref(path),sources=sources,all6000_query_cells_retained=True,undefined_metrics_not_dropped=True,kernel_backward_nondeterminism_part_of_training_variation=True,independent_H=False,SOTA_established=False,limits=['Only3 training runs; descriptive bootstrap is not a population seed guarantee.','Official200 already exposed; oldT1323 component fitting,not final1654 fit.','Singleimage perquery; not author ten-reconstruction metric average.'])
 (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],DINO=results['DINO2AFC'],SEED=results['SEED'],robustness_gate=bool(gate),SOTA=False)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--study',required=True);run(p.parse_args().study)
