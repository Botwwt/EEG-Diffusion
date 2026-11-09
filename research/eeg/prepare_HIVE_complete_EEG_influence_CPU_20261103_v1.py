"""Select exactly registered real OOF conditions, own affines and native T latents."""
import argparse,sys,time
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();started=time.monotonic()
 import json
 spec=json.loads(Path(a.protocol).read_bytes());sys.path.insert(0,str(Path(spec['base_protocol']['path']).parent))
 import brainhive_functional_mean_contract_20261103_v1 as G
 base=G.setup(G.checked(spec['base_protocol']))
 for pin in spec['source_pins']:G.checked(pin)
 import torch
 from brainhive_bounded_component_v1 import BoundedConditionAdapter
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 root=Path(spec['output_root']);assert not (root/'inputs').exists();out=root/'inputs';out.mkdir(parents=True)
 contract=G.read(G.checked(base['input_pins']['component_contract']));split=G.read(G.checked(base['input_pins']['split']))
 rows=spec['queries'];assert len(rows)==80 and len({r['subject'] for r in rows})==10 and len({r['concept_id'] for r in rows})==8
 conditions=[];weights=[];biases=[];sources=[]
 for subject in contract['subjects']:
  local=[r for r in rows if r['subject']==subject['subject']];routed={}
  for source in subject['sources']:
   if source['role'] not in ('oof0','oof1'):continue
   cache_pin=dict(path=source['condition_cache_path'],sha256=source['condition_cache_sha256']);model_pin=dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256'])
   cache=torch.load(G.checked(cache_pin),weights_only=True,map_location='cpu',mmap=True)
   state=torch.load(G.checked(model_pin),weights_only=True,map_location='cpu')['model'];g=state['block.layers.0.norm.weight'];b=state['block.layers.0.norm.bias']
   reg=G.read(G.checked(dict(path=source['parent_registration_path'],sha256=source['parent_registration_sha256'])))
   held={r['array_row']:r['cache_row'] for r in source['groups']['component_train']};trained=set(reg['training_array_rows'])
   assert set(held).isdisjoint(trained) and set(held)|trained==set(split['image_rows']['T'])
   assert set(reg['excluded_oof_rows'])==set(held) and cache['model_sha256']==source['checkpoint_sha256']
   for row in local:
    if row['source_role']!=source['role']:continue
    pair=[]
    for index in (row['array_row'],row['donor_array_row']):
     assert index in held and int(cache['array_rows'][held[index]])==index
     pair.append(cache['condition'][held[index]].clone())
    routed[row['array_row']]=(torch.stack(pair),g.expand(2,-1).clone(),b.expand(2,-1).clone())
   sources.append(dict(subject=subject['subject'],role=source['role'],model=G.ref(model_pin['path']),cache=G.ref(cache_pin['path'])))
   del cache,state
  for row in local:
   c,g,b=routed[row['array_row']];conditions.append(c);weights.append(g);biases.append(b)
 c=torch.stack(conditions);g=torch.stack(weights);b=torch.stack(biases)
 assert c.shape==g.shape==b.shape==(80,2,1024)
 saved=torch.load(G.checked(base['ordinary']),weights_only=True,map_location='cpu');head=BoundedConditionAdapter().eval().requires_grad_(False);head.load_state_dict(saved['model'],strict=True)
 with torch.inference_mode():changed=head(c.reshape(160,1024),g.reshape(160,1024),b.reshape(160,1024))[0].reshape(80,2,1024)
 targets=torch.load(G.checked(base['input_pins']['T_F_targets']),weights_only=True,map_location='cpu',mmap=True)
 truth=torch.stack([targets[[r['array_row'],r['donor_array_row']]].clone() for r in rows])
 latent=np.load(G.checked(spec['T_latent']),mmap_mode='r',allow_pickle=False);positions={r:i for i,r in enumerate(split['image_rows']['T'])}
 z=np.array(latent[[positions[r['array_row']] for r in rows]],copy=True)
 assert z.shape==(80,4,64,64) and z.dtype==np.float32
 np.savez_compressed(out/'prepared_T80.npz',condition=c.numpy(),weight=g.numpy(),bias=b.numpy(),corrected_CPU=changed.numpy(),target=truth.numpy(),latent=z)
 assert not torch.cuda.is_initialized()
 result=dict(status='COMPLETE_REAL10SUBJECT80T_CORRECT_DONOR_INPUTS_NO_CUDA',protocol=G.ref(a.protocol),source=G.ref(__file__),
  arrays=G.ref(out/'prepared_T80.npz'),rows=rows,sources=sources,ordinary=base['ordinary'],upstream_only_is_OOF=True,
  old_ordinary_fitted_full_T=True,selection_labels_only_define_controls=True,new_fits=0,new_GPU_jobs=0,new_images=0,seconds=time.monotonic()-started)
 G.write(out/'completion.json',result);print(G.json.dumps(result))
if __name__=='__main__':main()
