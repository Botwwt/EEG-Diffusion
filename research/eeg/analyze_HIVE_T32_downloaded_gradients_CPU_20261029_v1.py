"""Independent QR projection/statistics and explicitly posthoc loss decomposition."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=h)
def run(folder):
 folder=Path(folder);receipt=json.loads((folder/'receiving_completion.json').read_bytes())
 for e in receipt['entries']:assert ref(folder/e['name'])['sha256']==e['sha256']
 source=json.loads((folder/'metadata/completion.json').read_bytes());spec=json.loads((folder/'metadata/protocol.json').read_bytes());z=np.load(folder/'gradients.npz',allow_pickle=False)
 assert source['queries']==32 and source['all32_native_latents_bit_exact'] and z['u'].shape==(32,1024) and z['grad_epsilon'].shape==(32,4,1024)
 records=[];max_error=0.;token_norm_ratios=[];alignment_eps=[];alignment_token=[]
 for i in range(32):
  u=z['u'][i].astype(np.float64);q,_=np.linalg.qr(np.column_stack([np.ones(1024),u]),mode='reduced')
  def project(g):g=np.asarray(g,dtype=np.float64);return g-q@(q.T@g)
  end=project(z['grad_end'][i]);ep=project(z['grad_epsilon'][i].astype(np.float64).mean(axis=0));tok=project(z['grad_token'][i]);total=ep+.1*tok
  def cosine(g):return float(end@g/(np.linalg.norm(end)*np.linalg.norm(g)))
  values=dict(combined=cosine(total),epsilon_only=cosine(ep),token_only=cosine(tok));error=abs(values['combined']-source['rows'][i]['tangent_cosine_point_vs_endpoint']);assert error<1e-12;max_error=max(max_error,error)
  ratio=float(.1*np.linalg.norm(tok)/np.linalg.norm(ep));token_norm_ratios.append(ratio);alignment_eps.append(float(end@ep));alignment_token.append(float(.1*(end@tok)))
  records.append(dict(index=i,query=spec['queries'][i],cosines=values,scaled_token_to_epsilon_gradient_norm=ratio,endpoint_alignment_epsilon=alignment_eps[-1],endpoint_alignment_scaled_token=alignment_token[-1]))
 ordered=sorted(range(32),key=lambda i:spec['queries'][i]['concept_id']);assert len({q['concept_id'] for q in spec['queries']})==32
 rng=np.random.default_rng(1729);draws=np.array([rng.integers(32,size=32) for _ in range(1000)]);stats={}
 for name in ('combined','epsilon_only','token_only'):
  a=np.array([records[i]['cosines'][name] for i in ordered]);stats[name]=dict(mean=float(a.mean()),negative_queries=int((a<0).sum()),descriptive_concept_bootstrap95=np.quantile(a[draws].mean(axis=1),[.025,.975]).tolist(),preregistered=(name=='combined'))
 assert abs(stats['combined']['mean']-source['mean_tangent_cosine'])<1e-12 and np.max(np.abs(np.array(stats['combined']['descriptive_concept_bootstrap95'])-source['descriptive_concept_bootstrap95']))<1e-12
 result=dict(status='PASS_ALL32_QR_GRADIENT_STATISTICS_RECOMPUTATION',source=ref(__file__),receipt=ref(folder/'receiving_completion.json'),queries=32,stats=stats,maximum_primary_cosine_error=max_error,posthoc_decomposition=dict(median_scaled_token_to_epsilon_norm=float(np.median(token_norm_ratios)),total_endpoint_inner_product_epsilon=float(sum(alignment_eps)),total_endpoint_inner_product_scaled_token=float(sum(alignment_token)),note='Component analysis is posthoc, not a registered method-success gate or causal attribution of image gains.'),rows=records,scope='Reprojects exact stored GPU gradients and recomputes statistics only; does not independently rerun UNet/autograd or demonstrate semantic image improvement.',automatic_followup_training=False,SOTA_established=False)
 with (folder/'independent_local_gradient_analysis.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ('status','stats','maximum_primary_cosine_error','posthoc_decomposition')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();run(a.folder)
