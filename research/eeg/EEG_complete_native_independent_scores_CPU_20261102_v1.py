"""Separate direct frozen-feature ranks and concept bootstrap; no old AR job.

Classical metric arithmetic copied into this study's private source scope.
No historical admission, controller, model, image loader or CUDA is imported.
"""
import numpy as np
METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc')
def independent_arrays(features,targets,queries,gallery):
 normalized=features/np.linalg.norm(features,axis=1,keepdims=True);gallery_unit=targets/np.linalg.norm(targets,axis=1,keepdims=True)
 similarity=normalized@gallery_unit.T;positions={q['stimulus_id']:i for i,q in enumerate(gallery)};result={name:[] for name in METRICS}
 assert len(gallery)==400 and len(positions)==400
 for index,q in enumerate(queries):
  target=positions[q['stimulus_id']];scores=similarity[index];positive=scores[target];others=np.delete(scores,target)
  within=[i for i,row in enumerate(gallery) if row['concept_id']==q['concept_id']];assert len(within)==10;competitors=[i for i in within if i!=target]
  row=dict(target_cosine=float(positive),DINO2AFC=float((np.count_nonzero(others<positive)+.5*np.count_nonzero(others==positive))/399),gallery_top1=float(np.argmax(scores)==target),within_concept10_top1=float(within[int(np.argmax(scores[within]))]==target),within_concept10_two_afc=float((sum(scores[i]<positive for i in competitors)+.5*sum(scores[i]==positive for i in competitors))/9))
  for name in METRICS:result[name].append(row[name])
 return {name:np.array(values,np.float64) for name,values in result.items()}
def independent_contrast(candidate,baseline,queries):
 concepts=sorted({q['concept_id'] for q in queries});indices=[[i for i,q in enumerate(queries) if q['concept_id']==concept] for concept in concepts]
 difference={name:candidate[name]-baseline[name] for name in METRICS};by_concept=np.array([[difference[name][group].mean() for name in METRICS] for group in indices])
 draws=np.random.default_rng(1729).integers(0,len(concepts),(1000,len(concepts)));boot=np.array([by_concept[chosen].mean(0) for chosen in draws]);interval=np.percentile(boot,[2.5,97.5],axis=0)
 return {name:dict(difference=float(difference[name].mean()),descriptive_concept_bootstrap95=interval[:,i].tolist()) for i,name in enumerate(METRICS)}
