"""Read complete fixed fits and all400 cached probabilities; no image tuning."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'audit/EEG_native_categorical_20261029_v1/received_existing_stage80_v1'
OUT=ROOT/'audit/EEG_native_categorical_20261029_v1/complete_log_probability_diagnostic_v1.json'


def entropy(p):
 return -(p*np.log(np.maximum(p,np.finfo(np.float64).tiny))).sum(-1)


def run():
 receipt=json.loads((SOURCE/'receiving_completion.json').read_bytes())
 for key,item in receipt['files'].items():
  p=Path(item['local']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['remote']['sha256']
 panel=json.loads((SOURCE/'old/panel400.json').read_bytes());groups={}
 for i,q in enumerate(panel['queries']):groups.setdefault(q['concept_id'],[]).append(i)
 keys=sorted(groups);same=np.array([next(g[(g.index(i)+1)%10] for g in groups.values() if i in g) for i in range(400)])
 different=np.empty(400,int)
 for n,k in enumerate(keys):
  for j,i in enumerate(groups[k]):different[i]=groups[keys[(n+1)%40]][j]
 result={}
 for host in ('old','new'):
  events=json.loads((SOURCE/host/'fit_events.json').read_bytes());assert len(events)==2070
  epochs=[]
  for epoch in range(1,11):
   rows=[r for r in events if r['epoch']==epoch];assert len(rows)==207 and sum(r['queries'] for r in rows)==13230
   means={name:float(np.average([r[name] for r in rows],weights=[r['queries'] for r in rows])) for name in ('optimized_loss','native_coarse_NLL_nats_per_query','posterior_entropy','EEG_class_entropy','preclip_gradient','backbone_gradient')}
   epochs.append(dict(epoch=epoch,**means,gradient_clip_fraction=float(np.mean([r['preclip_gradient']>1 for r in rows]))))
  logits=np.load(SOURCE/host/'EEG_logits400.npy',allow_pickle=False).astype(np.float64);p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
  assert p.shape==(400,1000) and np.allclose(p.sum(1),1)
  h=entropy(p);average=p.mean(0);marginal_h=float(entropy(average));conditional_h=float(h.mean())
  # H(mean pi)-mean H(pi) is the finite uniform-query-index/class-draw MI.
  # It is not a population estimate of EEG/visual mutual information.
  js=marginal_h-conditional_h
  hellinger_same=.5*np.square(np.sqrt(p)-np.sqrt(p[same])).sum(1)
  hellinger_diff=.5*np.square(np.sqrt(p)-np.sqrt(p[different])).sum(1)
  by_concept=np.array([(hellinger_diff[g]-hellinger_same[g]).mean() for g in (groups[k] for k in keys)])
  draws=np.random.default_rng(1729).integers(0,40,(1000,40));ci=np.quantile(by_concept[draws].mean(1),[.025,.975]).tolist()
  result[host]=dict(epochs=epochs,complete_training_events=len(events),fixed_final_epoch=10,
   Dcheck400_probability_geometry=dict(mean_conditional_entropy_nats=conditional_h,marginal_entropy_nats=marginal_h,
    exp_mean_conditional_entropy=float(np.exp(conditional_h)),exp_marginal_entropy=float(np.exp(marginal_h)),
    finite_query_index_class_draw_MI_nats=float(js),mean_max_probability=float(p.max(1).mean()),
    mean_largest20_probability_mass=float(np.sort(p,axis=1)[:,-20:].sum(1).mean()),marginal_largest20_mass=float(np.sort(average)[-20:].sum()),
    distinct_argmax_classes=int(len(np.unique(p.argmax(1)))),same_concept_Hellinger_squared=float(hellinger_same.mean()),different_concept_Hellinger_squared=float(hellinger_diff.mean()),
    different_minus_same_Hellinger_squared=float(by_concept.mean()),descriptive_concept_bootstrap95=ci))
 output=dict(status='COMPLETE_FIXED10_ALL2070_LOGS_AND_DCHECK400_PROBABILITIES_CPU_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=result,
  model_changes=0,new_GPU_jobs=0,image_effect_scores_read=0,hyperparameter_or_checkpoint_selection=False,
  limits='Probability concentration and training evidence do not establish image quality, overfitting, physiological uncertainty, original EEG information limits, or a unique failure cause. Finite query-index/class MI is defined only for uniform sampling of these400 cached rows.',
  source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 with OUT.open('x',encoding='utf8') as f:json.dump(output,f,indent=2);f.write('\n')
 print(json.dumps({host:dict(first_epoch_nll=v['epochs'][0]['native_coarse_NLL_nats_per_query'],last_epoch_nll=v['epochs'][-1]['native_coarse_NLL_nats_per_query'],last_epoch_entropy=v['epochs'][-1]['EEG_class_entropy'],Dcheck400=v['Dcheck400_probability_geometry']) for host,v in result.items()}))


if __name__=='__main__':run()
