"""Resolve native-call versus helper replay exactly before another scientific probe."""
import argparse,copy,hashlib,inspect,json,sys,time
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--job-name',required=True);a=p.parse_args();start=time.monotonic();s=json.loads(Path(a.spec).read_bytes())
 sys.path.insert(0,str(Path(s['base_protocol']['path']).parent));import brainhive_functional_mean_contract_20261103_v1 as G
 base=G.setup(G.checked(s['base_protocol']));G.wrapper(base,a.job_name);G.resources(base,start=True);rss=G.rss()
 import torch
 from brainhive_component_generator_v1 import load_native,denoise
 from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
 torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory)
 out=Path(s['output_root']);assert not out.exists();out.mkdir();loaderdir=out/'loader';loaderdir.mkdir();loader=ConvertedWeightAudit(torch,loaderdir);loader.install()
 try:pipe,A,P,prompts,host=load_native(training=False)
 finally:loader.restore()
 proof=loader.complete();raw=np.load(G.checked(s['tokens']),mmap_mode='r',allow_pickle=False);tokens=torch.from_numpy(np.array(raw[:2,0],copy=True)).cuda()
 captured=[];pending={}
 def before(m,args,kwargs):
  assert not pending
  pending.update(sample=(args[0] if args else kwargs['sample']).detach().clone(),timestep=(args[1] if len(args)>1 else kwargs['timestep']).detach().clone(),
                 kwargs={k:({kk:vv for kk,vv in v.items()} if isinstance(v,dict) else v) for k,v in kwargs.items()})
 def after(m,args,kwargs,value):
  captured.append(dict(**pending,prediction=(value[0] if isinstance(value,tuple) else value.sample).detach().clone()));pending.clear()
 handles=[pipe.unet.register_forward_pre_hook(before,with_kwargs=True),pipe.unet.register_forward_hook(after,with_kwargs=True)]
 seeds=[row['trajectory_seed'] for row in s['queries'][:2]]
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
   pipe(height=512,width=512,generator=[torch.Generator(device='cuda').manual_seed(seed) for seed in seeds],prompt_embeds=prompts[0].expand(2,-1,-1),
    pooled_prompt_embeds=prompts[1].expand(2,-1),cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,guidance_scale=0.,output_type='latent')
 finally:
  for h in handles:h.remove()
 assert len(captured)==4
 def info(x,y):
  return dict(same_values=bool(torch.equal(x,y)),shape=list(x.shape),reference_shape=list(y.shape),dtype=str(x.dtype),reference_dtype=str(y.dtype),
    stride=list(x.stride()),reference_stride=list(y.stride()),maximum_absolute_error=float((x.float()-y.float()).abs().max()))
 results=[]
 for ti,state in enumerate(captured):
  kw=state['kwargs'];given={'encoder_hidden_states':prompts[0].expand(2,-1,-1),'added_cond_kwargs':dict(text_embeds=prompts[1].expand(2,-1),time_ids=prompts[2].expand(2,-1)),
                            'cross_attention_kwargs':{'ip_hidden_states':tokens},'return_dict':False}
  comparison=dict(keys=list(kw),helper_keys=list(given),encoder=info(given['encoder_hidden_states'],kw['encoder_hidden_states']),
                  pooled=info(given['added_cond_kwargs']['text_embeds'],kw['added_cond_kwargs']['text_embeds']),time_ids=info(given['added_cond_kwargs']['time_ids'],kw['added_cond_kwargs']['time_ids']),
                  tokens=info(tokens,kw['cross_attention_kwargs']['ip_hidden_states']))
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
   helper=denoise(pipe,prompts,state['sample'],state['timestep'],tokens)
   actual=pipe.unet(state['sample'],state['timestep'],**kw)[0]
   contiguous={**given,'encoder_hidden_states':given['encoder_hidden_states'].contiguous(),
               'added_cond_kwargs':{key:value.contiguous() for key,value in given['added_cond_kwargs'].items()}}
   packed=pipe.unet(state['sample'],state['timestep'],**contiguous)[0]
  def error(value):
   diff=value.float()-state['prediction'].float()
   return dict(exact=bool(torch.equal(value,state['prediction'])),maximum_absolute_error=float(diff.abs().max()),relative_L2=float(diff.norm()/state['prediction'].float().norm()))
  results.append(dict(time=ti,conditioning=comparison,helper=error(helper),actual_kwargs=error(actual),contiguous_helper=error(packed)))
 assert all(row['actual_kwargs']['exact'] for row in results),'Exact actual-conditioning replay itself failed; no threshold relaxation'
 done=dict(status='PASS_ACTUAL_NATIVE_KWARGS_REPLAY_EXACT_ALL4_TIMES',results=results,source=G.ref(__file__),spec=G.ref(a.spec),loader=proof,host=host,
   diagnostic_only=True,new_training=0,new_images=0,NEW_contacted=False,seconds=time.monotonic()-start,extra_CPU_bytes=G.rss()-rss,peak_GPU_bytes=torch.cuda.max_memory_allocated())
 assert time.monotonic()-start<270 and G.rss()-rss<10<<30 and torch.cuda.max_memory_allocated()<20<<30
 G.write(out/'completion.json',done);print(json.dumps(done))
if __name__=='__main__':main()
