"""Same locked author metric evaluator, explicit new manifest/output arguments."""
import argparse,csv,json,math,sys,time
from pathlib import Path
from profile_HIVE_Euclidean_F_R80_GPU_20261030_v1 import sha,ref,check,write

def run(path):
 e=json.loads(Path(path).read_bytes());s=json.loads(check(e['protocol']).read_bytes());sys.path.insert(0,s['repository_root'])
 from brainhive_original_metrics_runtime_v1 import OriginalEvaluator,NAMES,torch,transforms
 from PIL import Image
 root=Path(s['output_root']);artifact=json.loads((root/'loss_control/artifact_audit.json').read_bytes());assert artifact['queries']==2000
 manifest=check(artifact['manifest']);rows=[json.loads(x) for x in manifest.read_text().splitlines()];assert len(rows)==2000
 gallery=[json.loads(x) for x in check(s['gallery_manifest']).read_text().splitlines()];real={x['stimulus_id']:x for x in gallery};assert len(real)==200
 torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=root/'evaluation/original_paper_metrics';out.mkdir(parents=True,exist_ok=True);assert not (out/'completion.json').exists();started=time.monotonic()
 evaluator=OriginalEvaluator();config=dict(execution=ref(path),manifest=ref(manifest),author_source_checks=evaluator.checks,one_image_per_query=True,queries=2000,subjects=10)
 cp=out/'configuration.json'
 if cp.exists():assert json.loads(cp.read_bytes())==config
 else:write(cp,config)
 preprocess=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512),transforms.ToTensor()]);cache={}
 for key,r in real.items():
  assert sha(r['image_path'])==r['image_sha256']
  with Image.open(r['image_path']) as im:cache[key]=preprocess(im.convert('RGB'))
 records=[];aggregates={}
 def finite(x):x=float(x);return x if math.isfinite(x) else None
 try:
  for subject in sorted({r['subject'] for r in rows}):
   group=[r for r in rows if r['subject']==subject];assert len(group)==200 and {r['stimulus_id'] for r in group}==set(real)
   save=out/(subject+'.json')
   if save.exists():
    old=json.loads(save.read_bytes());assert old['configuration_SHA256']==sha(cp) and [r['query_id'] for r in old['per_query']]==[r['query_id'] for r in group];records+=old['per_query'];aggregates[subject]=old['aggregate'];continue
   generated=[]
   for r in group:
    assert sha(r['generated_path'])==r['generated_sha256']
    with Image.open(r['generated_path']) as im:generated.append(preprocess(im.convert('RGB')))
   originals=torch.stack([cache[r['stimulus_id']] for r in group]);recons=torch.stack(generated);aggregate,per_query=evaluator.evaluate(originals,recons);part=[]
   for i,r in enumerate(group):
    part.append({k:r[k] for k in ('query_id','subject','stimulus_id','concept_id','generated_path','image_seed','score_target_path')}|{key:finite(per_query[key][i]) for key in NAMES})
   aggregate={key:finite(aggregate[key]) for key in NAMES};write(save,dict(configuration_SHA256=sha(cp),per_query=part,aggregate=aggregate));records+=part;aggregates[subject]=aggregate
   print(json.dumps(dict(stage='FIXED_EUCLIDEAN_F_R80_AUTHOR_METRICS',subject=subject,completed=len(records),expected=2000,seconds=time.monotonic()-started)),flush=True);del generated,originals,recons;torch.cuda.empty_cache()
 finally:evaluator.close()
 assert len(records)==2000
 with (out/'per_query.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 counts={key:sum(r[key] is not None for r in records) for key in NAMES};means={key:sum(aggregates[subject][key] for subject in aggregates)/10 if counts[key]==2000 else None for key in NAMES}
 write(out/'completion.json',dict(status='ALL2000_AUTHOR_CODE_QUERY_METRICS_COMPLETE' if all(x==2000 for x in counts.values()) else 'COMPLETE_WITH_UNDEFINED_METRICS_NO_SILENT_EXCLUSION',queries=2000,subjects=10,means=means,defined_query_counts=counts,per_query=ref(out/'per_query.csv'),configuration=ref(cp),seconds=time.monotonic()-started,independent_H=False,SOTA_established=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execution',required=True);q=p.parse_args();run(q.execution)
