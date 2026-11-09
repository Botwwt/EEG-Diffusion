"""Actual native BrainMLP derivatives plus matched batch16 temporary Adam engineering."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
from brainhive_native_direction_contract_20261103_v1 import setup,resources,freeze_affine,tensor_digest,projected_losses
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--job-name',required=True);cli=parser.parse_args();started=time.monotonic()
 spec,old,base,G,H=setup(cli.protocol);G.wrapper(base,cli.job_name);root=Path(spec['output_root']);out=root/'GPU';assert not out.exists();out.mkdir()
 before=resources(spec,old,base,G,H,True);base_rss=G.rss();torch=None;stage='LOAD'
 try:
  import torch
  import torch.nn.functional as F
  from brainhive_native_models_v1 import classes
  from brainhive_bounded_component_v1 import BoundedConditionAdapter
  from brainhive_component_generator_v1 import load_native
  from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
  def budget(refresh=False):
   assert time.monotonic()-started<870 and G.rss()-base_rss<10<<30 and torch.cuda.max_memory_allocated()<20<<30
   if refresh:resources(spec,old,base,G,H)
  prep=G.read(root/'inputs/completion.json');assert prep['status']=='COMPLETE_REAL_T80_RAW_R4_TEN_FULL_PARENT_CPU_INPUTS_NO_CUDA'
  data=dict(np.load(G.checked(prep['arrays']),allow_pickle=False));past=G.read(G.checked(old['prepared_completion']))
  states=np.load(G.checked(past['states']),mmap_mode='r',allow_pickle=False);geometry=dict(np.load(G.checked(past['geometry']),allow_pickle=False));times=geometry['timesteps']
  old_inputs=G.read(G.checked(old['prior_prepared']));cache=dict(np.load(G.checked(old_inputs['arrays']),allow_pickle=False))
  head=BoundedConditionAdapter().eval().requires_grad_(False);head.load_state_dict(torch.load(G.checked(base['ordinary']),weights_only=True,map_location='cpu')['model'],strict=True);head=head.cuda();head_sha=tensor_digest(head)
  with torch.no_grad():
   cc=torch.from_numpy(cache['condition']).cuda();ww=torch.from_numpy(cache['weight']).cuda();bb=torch.from_numpy(cache['bias']).cuda()
   changed=head(cc.reshape(160,1024),ww.reshape(160,1024),bb.reshape(160,1024))[0]
  loader=ConvertedWeightAudit(torch,out);loader.install()
  try:pipe,A,P,prompts,host=load_native(training=False)
  finally:loader.restore()
  loading=loader.complete();host_sha={name:tensor_digest(module,True) for name,module in dict(UNet=pipe.unet,A=A,P=P).items()}
  def unet(sample,tt,tokens):
   n=len(sample)
   return pipe.unet(sample,tt,encoder_hidden_states=prompts[0].expand(n,-1,-1).contiguous(),
    added_cond_kwargs=dict(text_embeds=prompts[1].expand(n,-1).contiguous(),time_ids=prompts[2].expand(n,-1).contiguous()),cross_attention_kwargs=dict(ip_hidden_states=tokens),return_dict=False)[0]
  prior_state=np.load(G.checked(old['prior_states']),mmap_mode='r',allow_pickle=False);prior_prediction=np.load(G.checked(old['prior_predictions']),mmap_mode='r',allow_pickle=False)
  parity=[]
  with torch.inference_mode():
   tokens=P(A(changed.half())).reshape(80,2,4,2048)[:2,0]
   for ti,tt in enumerate(times):
    sample=torch.from_numpy(np.array(prior_state[0,:2,ti],copy=True)).cuda()
    with torch.autocast('cuda',dtype=torch.float16):value=unet(sample,torch.tensor(float(tt),device='cuda'),tokens)
    assert np.array_equal(value.cpu().numpy(),prior_prediction[0,:2,ti,0]);parity.append(dict(t=int(tt),exact=True))
  del changed,cc,ww,bb,tokens,value,sample,prior_state,prior_prediction,cache
  pipe.register_modules(vae=None,text_encoder=None,text_encoder_2=None);torch.cuda.empty_cache()
  pipe.unet.float().eval();A.float().eval();P.float().eval();prompts=tuple(v.float() for v in prompts)
  scope=classes();split=G.read(G.checked(base['input_pins']['split']));positions={row:i for i,row in enumerate(split['image_rows']['T'])};latents=np.load(G.checked(old['T_latent']),mmap_mode='r',allow_pickle=False)
  EEG=torch.from_numpy(data['EEG']).cuda();gamma=torch.from_numpy(data['gammas']).cuda();beta=torch.from_numpy(data['betas']).cuda();visual=torch.from_numpy(data['F_targets']).cuda()
  def new_encoder(si):
   pin=spec['full_models'][si]['model'];value=torch.load(G.checked(pin),weights_only=True,map_location='cpu');model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval();model.load_state_dict(value['model'],strict=True);freeze_affine(model);return model,pin
  def tokens_of(model,x,si):
   # Native encoder and old ordinary use FP32 in both training and inference.
   # Only the original frozen A/P/UNet run under the surrounding autocast.
   with torch.autocast('cuda',enabled=False):
    c=model(x.float());corrected=head(c,gamma[si].expand(len(c),-1),beta[si].expand(len(c),-1))[0]
   return P(A(corrected))
  def model_parameters(model):return tuple(p for p in model.parameters() if p.requires_grad)
  def grad(loss,params,retain=False):
   values=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=False);assert all(torch.isfinite(v).all() for v in values);return torch.cat([v.reshape(-1) for v in values])
  def cell_state(domain,index,ti):return torch.from_numpy(np.array(states[domain,index,ti],copy=True)).cuda().unsqueeze(0).repeat(2,1,1,1).contiguous()
  def truth_pair(index,ti,domain):
   row=old['queries'][index];zi=np.array(latents[positions[row['array_row']]],np.float64);zj=np.array(latents[positions[row['donor_array_row']]],np.float64)
   a=np.sqrt(geometry['alpha_cumprod'][int(times[ti])]);s=np.sqrt(1-geometry['alpha_cumprod'][int(times[ti])]);x=np.array(states[domain,index,ti],dtype=np.float64)
   return torch.from_numpy(np.stack(((x-a*zi)/s,(x-a*zj)/s))).cuda()
  values=np.empty((10,2,4,16),np.float64);prediction=np.empty((10,2,4,2,2,4,64,64),np.float32);identities=[];FD=[];CPU_parity=[];snapshot=None;snapshot_names=None
  stage='ALL_TEN_NATIVE_ENCODER_FP32_DERIVATIVES'
  for si,index in enumerate(spec['profile_indices']):
   model,pin=new_encoder(si);params=model_parameters(model);original=tensor_digest(model)
   with torch.no_grad():c=model(EEG[index]);error=float((c-torch.from_numpy(data['conditions_CPU'][index]).cuda()).abs().max());assert torch.allclose(c.cpu(),torch.from_numpy(data['conditions_CPU'][index]),atol=1e-5,rtol=5e-6)
   CPU_parity.append(dict(query=index,subject=old['queries'][index]['subject'],model=pin,condition_CPU_GPU_max_abs=error))
   for ti,tt in enumerate(times):
    budget(ti==0);time_tensor=torch.tensor(float(tt),device='cuda');t0=truth_pair(index,ti,0)
    with torch.no_grad():positive_teacher_tokens=P(A(visual[index]))
    def positive_loss():
     tk=tokens_of(model,EEG[index,:2],si);yp=unet(cell_state(0,index,ti),time_tensor,tk)
     return (yp[0].double()-t0[0]).square().mean()+.1*(1-F.cosine_similarity(tk[0].float(),positive_teacher_tokens[0].float(),dim=-1)).mean()
    lp=positive_loss();gp=grad(lp,params);lp_value=float(lp);del lp
    for di,domain in enumerate(spec['profile_domains']):
     target=truth_pair(index,ti,domain);direction=target[0]-target[1];sample=cell_state(domain,index,ti)
     with torch.no_grad():teacher=unet(sample,time_tensor,P(A(visual[index])))
     yp=unet(sample,time_tensor,tokens_of(model,EEG[index,:2],si));lf,lq,lo,ae,af,energy=projected_losses(yp,teacher,direction)
     gf=grad(lf,params,True);gq=grad(lq,params,ti==0)
     if ti==0:
      go=grad(lo,params);error=float((gf.double()-gq.double()-go.double()).norm()/gf.double().norm().clamp_min(1e-24));assert error<=5e-4
      identities.append(dict(subject=si,domain=domain,time_index=ti,relative_gradient_error=error,scalar_identity=float(lf-lq-lo)))
     dot=lambda a,b:float((a.double()*b.double()).sum())
     pp=dot(gp,gp);ff=dot(gf,gf);qq=dot(gq,gq);pq=dot(gp,gq);fq=dot(gf,gq)
     values[si,di,ti]=[lp_value,float(lf),float(lq),float(lo),float(ae),float(af),float(energy),pp,ff,qq,pq,fq,pq/max(np.sqrt(pp*qq),1e-24),fq/max(np.sqrt(ff*qq),1e-24),float((lf-lq-lo).abs()),error if ti==0 else 0.]
     prediction[si,di,ti]=np.stack((yp.detach().cpu().numpy(),teacher.cpu().numpy()))
     if si==0 and ti==0 and domain==2:
      snapshot=torch.stack((gp,gf,gq)).detach().cpu().numpy();snapshot_names=[name for name,p in model.named_parameters() if p.requires_grad]
      originals=[p.detach().clone() for p in params]
      for label,gg in (('positive',gp),('projected',gq)):
       norm=float(gg.double().norm());assert norm>1e-12;unit=gg/norm;measured=[]
       for sign in (-1.,1.):
        cursor=0
        with torch.no_grad():
         for p,v in zip(params,originals):
          n=p.numel();p.copy_(v+sign*.001*unit[cursor:cursor+n].reshape_as(p));cursor+=n
         if label=='positive':test=positive_loss()
         else:test=projected_losses(unet(sample,time_tensor,tokens_of(model,EEG[index,:2],si)),teacher,direction)[1]
         measured.append(float(test))
       with torch.no_grad():
        for p,v in zip(params,originals):p.copy_(v)
       fd=(measured[1]-measured[0])/.002;err=abs(fd-norm);assert err<=1e-4+.02*norm
       FD.append(dict(loss=label,step=.001,autograd=norm,finite_difference=fd,error=err))
      del originals
     del yp,teacher,lf,lq,lo,ae,af,energy,gf,gq,target,direction,sample
    del gp,t0,positive_teacher_tokens
   assert tensor_digest(model)==original;del model,params
   print(json.dumps(dict(stage=stage,subject=si+1,completed=si+1,expected=10,seconds=time.monotonic()-started)),flush=True)
  np.savez_compressed(out/'FP32_native_gradients.npz',statistics=values,predictions=prediction,full_parameter_snapshot=snapshot)
  # Measured training path: original FP16 values under BF16 autocast and checkpointing.
  pipe.unet.half().train();pipe.unet.enable_gradient_checkpointing();A.half();P.half();prompts=tuple(v.bfloat16() for v in prompts)
  from diffusers import DDPMScheduler
  scheduler=DDPMScheduler.from_config(pipe.scheduler.config);assert scheduler.config.prediction_type=='epsilon'
  assert np.array_equal(scheduler.alphas_cumprod.double().numpy(),geometry['alpha_cumprod'])
  noise_bank=np.load(G.checked(old['prior_noise']),mmap_mode='r',allow_pickle=False)
  assert float(pipe.unet.config.dropout)==0
  model,pin=new_encoder(0);original_model={name:v.detach().cpu().clone() for name,v in model.state_dict().items()};original_model_sha=tensor_digest(model)
  parameters=model_parameters(model);temporary=[];Adam=None;stage='MATCHED_BATCH16_BF16_FOUR_TEMPORARY_UPDATES_PER_ARM'
  for arm in spec['temporary_arms']:
   model.load_state_dict(original_model,strict=True);torch.manual_seed(1729);optimizer=torch.optim.AdamW(parameters,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
   losses=[];steps=[]
   for ti,tt in enumerate(times):
    budget(True);idx=np.array(spec['batch_engineering_queries']);n=len(idx)*2;assert n==16
    zi=[];zj=[]
    for index in idx:
     row=old['queries'][int(index)];zi.append(np.array(latents[positions[row['array_row']]],np.float32));zj.append(np.array(latents[positions[row['donor_array_row']]],np.float32))
    zs=torch.from_numpy(np.stack((zi,zj),axis=1).reshape(n,4,64,64)).cuda();a=float(np.sqrt(geometry['alpha_cumprod'][int(tt)]));sig=float(np.sqrt(1-geometry['alpha_cumprod'][int(tt)]))
    # Same physical Gaussian draw for both members; original sample is reused, no seed search.
    eps=np.array(noise_bank[idx,ti],copy=True);noises=torch.from_numpy(np.repeat(eps[:,None],2,axis=1).reshape(n,4,64,64)).cuda()
    positive_states=scheduler.add_noise(zs,noises,torch.full((n,),int(tt),device='cuda',dtype=torch.long));common=torch.from_numpy(np.repeat(np.array(states[2,idx,ti],copy=True)[:,None],2,axis=1).reshape(n,4,64,64)).cuda()
    ft=visual[idx].reshape(n,1024);ttensor=torch.tensor(float(tt),device='cuda')
    def pair_loss(p,t):
     pv=p.reshape(8,2,4,64,64).double();tv=t.reshape(8,2,4,64,64).double();delta=a/sig*(zs.reshape(8,2,4,64,64)[:,1].double()-zs.reshape(8,2,4,64,64)[:,0].double())
     e=delta.square().mean((1,2,3));assert (e>0).all();ae=((pv[:,0]-pv[:,1])*delta).mean((1,2,3))/e;af=((tv[:,0]-tv[:,1])*delta).mean((1,2,3))/e
     return (ae-af).square().mean()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):teacher_tokens=P(A(ft));teacher_prediction=unet(common,ttensor,teacher_tokens)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     tk=tokens_of(model,EEG[idx,:2].reshape(n,4250),0);yp=unet(positive_states.half(),ttensor,tk)
     base_loss=(yp.float()-noises).square().mean()+.1*(1-F.cosine_similarity(tk.float(),teacher_tokens.float(),dim=-1)).mean()
    base_value=float(base_loss);base_loss.backward();del base_loss,yp,tk
    with torch.autocast('cuda',dtype=torch.bfloat16):
     x=EEG[idx,2:] if arm=='sameconcept_wrong_EEG' else EEG[idx,:2];yp=unet(common.half(),ttensor,tokens_of(model,x.reshape(n,4250),0));qloss=pair_loss(yp,teacher_prediction)
     extra=(0. if arm=='positive_only' else .1)*qloss
    qvalue=float(qloss);extra.backward();del extra,qloss,yp
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));assert norm>0
    if arm=='directional' and ti==0:
     Adam=dict(before=torch.cat([p.detach().reshape(-1) for p in parameters]).cpu().numpy(),gradient=torch.cat([p.grad.detach().reshape(-1) for p in parameters]).cpu().numpy())
    optimizer.step()
    if arm=='directional' and ti==0:
     Adam.update(after=torch.cat([p.detach().reshape(-1) for p in parameters]).cpu().numpy(),exp_avg=torch.cat([optimizer.state[p]['exp_avg'].reshape(-1) for p in parameters]).cpu().numpy(),exp_avg_sq=torch.cat([optimizer.state[p]['exp_avg_sq'].reshape(-1) for p in parameters]).cpu().numpy())
    assert all(p.grad is None for module in (head,pipe.unet,A,P) for p in module.parameters())
    for key in ('block.layers.0.norm.weight','block.layers.0.norm.bias'):assert torch.equal(model.state_dict()[key].cpu(),original_model[key])
    steps.append(dict(t=int(tt),base_loss=base_value,projected_loss=qvalue,gradient_norm=norm,batch=16));losses.append(base_value+(0 if arm=='positive_only' else .1)*qvalue)
    del positive_states,common,teacher_tokens,teacher_prediction,ft,noises,zs;torch.cuda.empty_cache()
   temporary.append(dict(arm=arm,steps=steps,temporary_updates=4,original_model=pin));del optimizer
  model.load_state_dict(original_model,strict=True);assert tensor_digest(model)==original_model_sha
  assert tensor_digest(head)==head_sha and {name:tensor_digest(module,True) for name,module in dict(UNet=pipe.unet,A=A,P=P).items()}==host_sha
  np.savez_compressed(out/'first_directional_Adam_step.npz',**Adam)
  G.checked(base['ordinary']);budget(True)
  complete=dict(status='COMPLETE_NATIVE_EEG_DIRECTIONAL_ENGINEERING_ONLY_NOT_FORMAL_TRAINING',UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),
   protocol=G.ref(cli.protocol),source=G.ref(__file__),CPU_inputs=G.ref(root/'inputs/completion.json'),native_FP16_allfourtime_replay=parity,
   ten_full_parent_live_encoder_CPU_GPU_parity=CPU_parity,profile_indices=spec['profile_indices'],domains=spec['profile_domains'],timesteps=times.tolist(),
   derivatives=G.ref(out/'FP32_native_gradients.npz'),first_Adam_step=G.ref(out/'first_directional_Adam_step.npz'),parameter_names=snapshot_names,trainable_parameters=6450176,
   FP32_gradient_identities=identities,finite_differences=FD,temporary_batch16_BF16_updates=temporary,
   ordinary_frozen=True,native_output_LN_affine_frozen=True,all_encoder_values_restored=True,all_host_FP16_values_restored=True,
   projected_loss_is_new_posthoc_hypothesis=True,full_T_training_admitted=False,new_formal_updates=0,new_images=0,new_models=0,
   loader=loading,host=host,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=G.rss()-base_rss,resources_before=before,resources_after=resources(spec,old,base,G,H))
  G.write(out/'completion.json',complete);print(json.dumps(complete),flush=True)
 except BaseException as error:
  G.write(out/'failure.json',dict(error=repr(error),stage=stage,traceback=traceback.format_exc(),seconds=time.monotonic()-started,
   peak_GPU_bytes=torch.cuda.max_memory_allocated() if torch is not None and torch.cuda.is_initialized() else None,automatic_retry=False));raise
if __name__=='__main__':main()
