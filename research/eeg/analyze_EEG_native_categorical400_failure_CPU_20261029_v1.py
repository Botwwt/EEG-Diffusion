"""Posthoc whole-panel attribution; never selects a model or changes gates."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
L=Path(__file__).resolve().parent;D=L/'audit/EEG_native_categorical_20261029_v1';S=D/'stage400_results_v2'


def run():
 summary=json.loads((S/'summary.json').read_bytes());assert summary['status']=='COMPLETE_ALL5_CATEGORICAL_FULL400_OLD_ONLY_SCREEN' and summary['decision']=='STOP_INSTANCE'
 inputs=D/'received_existing_stage80_v1';panel=json.loads((inputs/'old/panel400.json').read_bytes());queries=panel['queries'];gallery=panel['target_gallery'];targets=np.load(inputs/'old/targets400.npy',allow_pickle=False)
 targets=targets/np.linalg.norm(targets,axis=1,keepdims=True);features=np.load(S/'all6_features.npz',allow_pickle=False);metrics=np.load(S/'all_metrics.npz',allow_pickle=False)
 groups={}
 for i,q in enumerate(queries):groups.setdefault(q['concept_id'],[]).append(i)
 keys=sorted(groups);assert len(groups)==40 and {len(g) for g in groups.values()}=={10}
 outside={};errors={}
 for arm in features.files:
  x=features[arm];x=x/np.linalg.norm(x,axis=1,keepdims=True);scores=x@targets.T;out=[]
  for i,q in enumerate(queries):
   others=[j for j,t in enumerate(gallery) if t['concept_id']!=q['concept_id']];assert len(others)==390
   positive=scores[i,i];out.append((np.count_nonzero(scores[i,others]<positive)+.5*np.count_nonzero(scores[i,others]==positive))/390)
  outside[arm]=np.array(out);reconstructed=(390*outside[arm]+9*metrics[arm+'__within_concept10_two_afc'])/399
  errors[arm]=float(np.max(np.abs(reconstructed-metrics[arm+'__DINO2AFC'])));assert errors[arm]<1e-12
 contrasts={}
 for other in ('visual','uniform','same_donor','different_donor','frozen_Gaussian'):
  between=(390/399)*(outside['coupled']-outside[other]);within=(9/399)*(metrics['coupled__within_concept10_two_afc']-metrics[other+'__within_concept10_two_afc'])
  total=metrics['coupled__DINO2AFC']-metrics[other+'__DINO2AFC'];assert np.allclose(between+within,total,atol=1e-12)
  contrasts['coupled_minus_'+other]=dict(total_DINO_gain=float(total.mean()),cross_concept_gallery_contribution=float(between.mean()),within_concept_gallery_contribution=float(within.mean()),
   within10_Top1_gain=float((metrics['coupled__within_concept10_top1']-metrics[other+'__within_concept10_top1']).mean()),within10_two_afc_gain=float((metrics['coupled__within_concept10_two_afc']-metrics[other+'__within_concept10_two_afc']).mean()))
 concentration={}
 gain=metrics['coupled__DINO2AFC']-metrics['visual__DINO2AFC'];concept_gain=np.array([gain[groups[k]].mean() for k in keys])
 for host in ('old','new'):
  logits=np.load(inputs/host/'EEG_logits400.npy',allow_pickle=False).astype(np.float64);p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True);h=-(p*np.log(np.maximum(p,1e-300))).sum(1)
  concept_h=np.array([h[groups[k]].mean() for k in keys]);corr=float(np.corrcoef(concept_h,concept_gain)[0,1])
  concentration[host]=dict(whole40_concept_entropy_vs_main_visual_gain_Pearson=corr,interpretation='Posthoc descriptive association; no causal attribution and no temperature/epoch selection.')
 log=json.loads((D/'complete_log_probability_diagnostic_v1.json').read_bytes())['results']
 result=dict(status='COMPLETE_POSTHOC_ALL400_AFC_EXACT_DECOMPOSITION_NO_GATE_CHANGE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  identity='DINO2AFC=(390*cross_concept_AFC+9*within_concept10_AFC)/399, using this fixed40x10 gallery.',
  gallery_weighting_caveat='The390-to9 weighting is fixed by gallery composition. A large cross-concept contribution partly follows from those weights and does not identify a unique model failure cause; unweighted within10 differences and paired intervals must also be read.',
  means_cross_concept_AFC={a:float(x.mean()) for a,x in outside.items()},contrasts=contrasts,identity_max_error=errors,probability_associations=concentration,
  final_training_NLL_main=log['old']['epochs'][-1]['native_coarse_NLL_nats_per_query'],final_training_NLL_visual=log['new']['epochs'][-1]['native_coarse_NLL_nats_per_query'],
  source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summary_SHA256=hashlib.sha256((S/'summary.json').read_bytes()).hexdigest(),
  new_models=0,new_GPU_jobs=0,decision_unchanged='STOP_INSTANCE',
  conclusions=['Correct EEG pairing has a measured whole-gallery benefit over donor controls, but superiority of coupled learning over the visual-only objective was not established.',
   'The paired DINO gain mostly concerns cross-concept competitors in this gallery. Within10Top1 and its paired intervals remain necessary; whole-gallery DINO gain is not stimulus-detail recovery.',
   'Lower T training likelihood loss and greater class concentration did not establish better images. No held likelihood experiment was run, so neither overfitting nor a unique causal failure mechanism is identified.'])
 path=D/'full400_posthoc_failure_diagnostic_v1.json'
 with path.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(contrasts=contrasts,associations=concentration,identity_max_error=max(errors.values()))))


if __name__=='__main__':run()
