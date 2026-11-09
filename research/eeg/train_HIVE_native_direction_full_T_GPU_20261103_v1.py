"""One bounded, exactly resumable segment of the fixed native EEG three-arm T fit."""
import argparse,copy,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import brainhive_native_direction_full_contract_20261103_v1 as C
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--subject',required=True);p.add_argument('--arm',required=True);p.add_argument('--segment',type=int,required=True);p.add_argument('--job-name',required=True);a=p.parse_args();started=time.monotonic()
 s,e,old,base,G,H=C.setup(a.protocol);assert a.subject in s['subjects'] and a.arm in C.ARMS and 0<=a.segment<s['maximum_segments_per_model'];G.wrapper(base,a.job_name)
 root=Path(s['output_root'])/'models'/a.subject/a.arm;root.mkdir(parents=True,exist_ok=True);segment=root/f'segment_{a.segment:03d}';assert not segment.exists();segment.mkdir()
 before=C.resources(s,e,True);baseline_rss=G.rss();torch=None;stage='LOAD';updates=0
 try:
  import torch
  import torch.nn.functional as F
  from diffusers import DDPMScheduler
  from brainhive_native_models_v1 import classes
  from brainhive_bounded_component_v1 import BoundedConditionAdapter
  from brainhive_native_direction_contract_20261103_v1 import freeze_affine,tensor_digest
  from brainhive_component_generator_v1 import load_native
  from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
  def budget(refresh=False):
   assert time.monotonic()-started<870 and G.rss()-baseline_rss<10<<30 and torch.cuda.max_memory_allocated()<20<<30
   if refresh:C.resources(s,e)
  prep=C.read(Path(s['output_root'])/'inputs/completion.json');assert prep['status']=='COMPLETE_T10_THREE_PASS_ALL_CONCEPT_PAIR_ORDERS_AND_NATIVE_INPUTS_CPU'
  plans=dict(np.load(C.checked(prep['plans']),allow_pickle=False));si=s['subjects'].index(a.subject);Trows=plans['training_rows'];pairs=plans['pairs'][si];positions={int(row):i for i,row in enumerate(Trows)}
  bind=C.read(C.checked(e['actual_bindings']));mp=next(v for v in e['full_models'] if v['subject']==a.subject);rp=next(v for v in e['R4_payloads'] if v['subject']==a.subject)
  raw=torch.load(C.checked(rp['payload']),weights_only=True,map_location='cpu',mmap=True);x=raw[Trows][:,bind['selected_indices'],:].reshape(13230,4250).contiguous();assert torch.isfinite(x).all();del raw
  target_all=torch.load(C.checked(base['input_pins']['T_F_targets']),weights_only=True,map_location='cpu');visual=target_all[Trows].contiguous();del target_all
  latents=np.load(C.checked(old['T_latent']),mmap_mode='r',allow_pickle=False)
  initial=torch.load(C.checked(mp['model']),weights_only=True,map_location='cpu')['model'];scope=classes();model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval();model.load_state_dict(initial,strict=True);params=freeze_affine(model)
  gamma=initial['block.layers.0.norm.weight'].cuda();beta=initial['block.layers.0.norm.bias'].cuda();initial_digest=tensor_digest(model)
  head=BoundedConditionAdapter().eval().requires_grad_(False);head.load_state_dict(torch.load(C.checked(base['ordinary']),weights_only=True,map_location='cpu')['model'],strict=True);head=head.cuda();head_digest=tensor_digest(head)
  loader=ConvertedWeightAudit(torch,segment);loader.install()
  try:pipe,A,P,prompts,host=load_native(training=False)
  finally:loader.restore()
  loading=loader.complete();host_digest={name:tensor_digest(module,True) for name,module in dict(UNet=pipe.unet,A=A,P=P).items()}
  pipe.register_modules(vae=None,text_encoder=None,text_encoder_2=None);torch.cuda.empty_cache();pipe.unet.train();pipe.unet.enable_gradient_checkpointing();prompts=tuple(v.bfloat16() for v in prompts)
  assert float(pipe.unet.config.dropout)==0 and all(not p.requires_grad for m in (head,A,P,pipe.unet) for p in m.parameters())
  scheduler=DDPMScheduler.from_config(pipe.scheduler.config);assert scheduler.config.prediction_type=='epsilon';alpha=scheduler.alphas_cumprod.double().numpy()
  def unet(sample,times,tokens):
   n=len(sample);return pipe.unet(sample,times,encoder_hidden_states=prompts[0].expand(n,-1,-1).contiguous(),added_cond_kwargs=dict(text_embeds=prompts[1].expand(n,-1).contiguous(),time_ids=prompts[2].expand(n,-1).contiguous()),cross_attention_kwargs=dict(ip_hidden_states=tokens),return_dict=False)[0]
  def tokens_of(xb):
   with torch.autocast('cuda',enabled=False):cc=model(xb.float());cc=head(cc,gamma.expand(len(cc),-1),beta.expand(len(cc),-1))[0]
   return P(A(cc))
  optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01,betas=(.9,.999),eps=1e-8)
  noise_rng=torch.Generator(device='cpu').manual_seed(C.stable_seed(a.subject,0,'noise'));axial_rng=np.random.default_rng(C.stable_seed(a.subject,0,'axial'))
  coverage=np.zeros((3,13230),np.bool_);epoch=0;offset=0;updates=0;exposures=0;snapshot_id=0;total_training_seconds=0.;loss_sums=np.zeros(5,np.float64);restoration=None
  if a.segment:
   index=C.read(root/'resume_index.json');saved=torch.load(C.checked(index['checkpoint']),weights_only=False,map_location='cpu')
   assert saved['protocol']==C.ref(a.protocol) and saved['subject']==a.subject and saved['arm']==a.arm and saved['snapshot_id']==index['snapshot_id']
   assert saved['initial_model']==mp['model'] and saved['ordinary']==base['ordinary'];updates=saved['updates'];exposures=saved['primary_exposures'];epoch=saved['epoch'];offset=saved['pair_offset'];snapshot_id=saved['snapshot_id']+1
   coverage=saved['coverage'];assert coverage.shape==(3,13230);expected=np.zeros_like(coverage)
   for ep in range(epoch):expected[ep]=True
   if epoch<3:expected[epoch,[[positions[int(v)] for v in row[:2]] for row in pairs[epoch,:offset]]]=True
   assert np.array_equal(coverage,expected) and int(coverage.sum())==exposures and updates==(epoch*827+(offset+7)//8)
   logpin=saved['log_prefix'];logpath=Path(logpin['path']);assert logpath.stat().st_size>=logpin['bytes'];assert hashlib.sha256(logpath.read_bytes()[:logpin['bytes']]).hexdigest()==logpin['sha256']
   for key in ('block.layers.0.norm.weight','block.layers.0.norm.bias'):assert torch.equal(saved['model'][key],initial[key])
   model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
   for parameter in params:
    state=optimizer.state[parameter];assert int(state['step'])==updates and state['exp_avg'].shape==parameter.shape and state['exp_avg_sq'].shape==parameter.shape
    assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
   noise_rng.set_state(saved['noise_rng']);axial_rng.bit_generator.state=saved['axial_rng'];torch.set_rng_state(saved['torch_CPU_rng']);torch.cuda.set_rng_state_all(saved['torch_CUDA_rng'])
   assert C.rng_probe(noise_rng,axial_rng,torch)==saved['next_RNG_probe'];loss_sums=saved['loss_sums'];total_training_seconds=saved['training_seconds']
   restoration=dict(checkpoint=index['checkpoint'],updates=updates,primary_exposures=exposures,coverage_exact=True,all_Adam_steps_and_shapes=True,output_affine_exact=True,log_prefix_exact=True,next_noise_axial_draw_exact=True,CPU_CUDA_rng_restored=True)
   G.write(segment/'serialized_restore.json',restoration);del saved
  else:assert not (root/'resume_index.json').exists()
  stage='FORMAL_NATIVE_EEG_FULL_T_FIT';logpath=segment/'updates.jsonl';stream=logpath.open('xb');segment_training_started=time.monotonic();updates_at_start=updates;last_checkpoint=time.monotonic()
  def checkpoint():
   nonlocal snapshot_id,last_checkpoint
   stream.flush();os.fsync(stream.fileno());state=dict(protocol=C.ref(a.protocol),subject=a.subject,arm=a.arm,initial_model=mp['model'],ordinary=base['ordinary'],snapshot_id=snapshot_id,
    model={n:v.detach().cpu().clone() for n,v in model.state_dict().items()},optimizer=optimizer.state_dict(),epoch=epoch,pair_offset=offset,updates=updates,primary_exposures=exposures,coverage=coverage.copy(),
    noise_rng=noise_rng.get_state(),axial_rng=copy.deepcopy(axial_rng.bit_generator.state),torch_CPU_rng=torch.get_rng_state(),torch_CUDA_rng=torch.cuda.get_rng_state_all(),
    next_RNG_probe=C.rng_probe(noise_rng,axial_rng,torch),loss_sums=loss_sums.copy(),training_seconds=total_training_seconds+time.monotonic()-segment_training_started,log_prefix=C.ref(logpath),
    working_checkpoint_policy='Two rotating own A/B working states; immutable epoch/final models and failures remain. No baseline overwritten.')
   record=C.save_resume(root,state,torch);snapshot_id+=1;last_checkpoint=time.monotonic();C.atomic_json(root/'progress.json',dict(subject=a.subject,arm=a.arm,updates=updates,total_updates=2481,epoch=epoch,pair_offset=offset,primary_exposures=exposures,checkpoint=record,segment=a.segment,UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat()))
   print(json.dumps(dict(stage=stage,subject=a.subject,arm=a.arm,updates=updates,expected=2481,epoch=epoch,primary_exposures=exposures,segment=a.segment,seconds=time.monotonic()-started)),flush=True)
   return record
  while epoch<3:
   if time.monotonic()-started>=s['GPU_soft_seconds']:break
   budget(updates%32==0);batch=pairs[epoch,offset:offset+8];b=len(batch);n=2*b;assert b>0
   ids=np.array([[positions[int(v)] for v in row] for row in batch]);correct=ids[:,:2].reshape(-1);wrong=ids[:,2:].reshape(-1);assert not coverage[epoch,correct].any()
   zi=np.array(latents[ids[:,0]],np.float32);zj=np.array(latents[ids[:,1]],np.float32);times=torch.randint(0,1000,(b,),generator=noise_rng);eps=torch.randn((b,4,64,64),generator=noise_rng)
   common_np,attempts=C.harmonic_states(zi,zj,eps.numpy(),times.numpy(),alpha,axial_rng);zs=torch.from_numpy(np.stack((zi,zj),1).reshape(n,4,64,64)).cuda();noises=eps.repeat_interleave(2,0).cuda();tv=times.repeat_interleave(2,0).cuda()
   positive_state=scheduler.add_noise(zs,noises,tv);common=torch.from_numpy(common_np).repeat_interleave(2,0).cuda();ft=visual[correct].cuda();xb=x[correct].cuda();delta=torch.from_numpy((np.sqrt(alpha[times.numpy()]) / np.sqrt(1-alpha[times.numpy()]))[:,None,None,None]*(zj.astype(np.float64)-zi)).cuda();energy=delta.square().mean((1,2,3));assert (energy>0).all()
   optimizer.zero_grad(set_to_none=True)
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):teacher_tokens=P(A(ft));teacher=unet(common.half(),tv,teacher_tokens).reshape(b,2,4,64,64).double()
   with torch.autocast('cuda',dtype=torch.bfloat16):
    tk=tokens_of(xb);prediction=unet(positive_state.half(),tv,tk);noise_loss=(prediction.float()-noises).square().mean();token_loss=(1-F.cosine_similarity(tk.float(),teacher_tokens.float(),dim=-1)).mean();base_loss=noise_loss+.1*token_loss
   nv=float(noise_loss);tkv=float(token_loss);base_loss.backward();del prediction,base_loss,tk,noise_loss,token_loss
   if a.arm=='sameconcept_wrong_EEG':xb=x[wrong].cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):prediction=unet(common.half(),tv,tokens_of(xb)).reshape(b,2,4,64,64).double()
   ae=((prediction[:,0]-prediction[:,1])*delta).mean((1,2,3))/energy;af=((teacher[:,0]-teacher[:,1])*delta).mean((1,2,3))/energy;parallel=(ae-af).square().mean();extra=(0. if a.arm=='positive_only' else .1)*parallel
   pv=float(parallel);aev=float(ae.mean());afv=float(af.mean());extra.backward();del extra,parallel,prediction,teacher,teacher_tokens,ft,xb,delta,energy,ae,af
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));assert np.isfinite(norm)
   optimizer.step();assert all(p.grad is None for module in (head,A,P,pipe.unet) for p in module.parameters())
   for key in ('block.layers.0.norm.weight','block.layers.0.norm.bias'):assert torch.equal(model.state_dict()[key].cpu(),initial[key])
   coverage[epoch,correct]=True;offset+=b;updates+=1;exposures+=n;loss_sums+=n*np.array([nv,tkv,pv,aev,afv])
   record=dict(update=updates,epoch=epoch,pair_offset=offset,batch=n,primary_exposures=exposures,epsilon_MSE=nv,token_angular=tkv,parallel_loss=pv,student_projection=aev,teacher_projection=afv,preclip_gradient_norm=norm,t=times.tolist(),pair_rows=batch.tolist(),noise_SHA256=hashlib.sha256(eps.numpy().tobytes()).hexdigest(),common_state_SHA256=hashlib.sha256(common_np.tobytes()).hexdigest(),axial_rejection_attempts=attempts)
   stream.write((json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode());del zs,noises,positive_state,common,tv
   if offset==6615:
    assert coverage[epoch].all();epoch+=1;offset=0
    epfile=root/f'epoch_{epoch:01d}.pt';assert not epfile.exists();torch.save(dict(model={name:v.detach().cpu().clone() for name,v in model.state_dict().items()},subject=a.subject,arm=a.arm,epoch=epoch,updates=updates,primary_exposures=exposures,protocol=C.ref(a.protocol),ordinary=base['ordinary'],initial_model=mp['model']),epfile)
    checkpoint()
   elif updates==1 or updates%32==0 or time.monotonic()-last_checkpoint>120:checkpoint()
  assert updates>updates_at_start
  latest=checkpoint();stream.close();budget(True)
  assert tensor_digest(head)==head_digest and {name:tensor_digest(module,True) for name,module in dict(UNet=pipe.unet,A=A,P=P).items()}==host_digest
  assert tensor_digest(model)!=initial_digest
  complete=dict(status='COMPLETE_FIXED_EPOCH3_NATIVE_EEG_MODEL' if epoch==3 else 'RESUMABLE_FIXED_NATIVE_EEG_SEGMENT_SOFT_LIMIT',protocol=C.ref(a.protocol),source=C.ref(__file__),subject=a.subject,arm=a.arm,segment=a.segment,
   checkpoint=latest,updates=updates,updates_this_segment=updates-updates_at_start,total_updates=2481,epoch=epoch,pair_offset=offset,primary_exposures=exposures,coverage_count=int(coverage.sum()),loss_means=(loss_sums/exposures).tolist(),
   serialized_restore=restoration,ordinary_frozen=True,host_frozen=True,output_LN_affine_frozen=True,trainable_parameters=6450176,loader=loading,
   seconds=time.monotonic()-started,training_seconds=total_training_seconds+time.monotonic()-segment_training_started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=G.rss()-baseline_rss,
   resources_before=before,resources_after=C.resources(s,e),new_images=0,full_parent_T_fit_not_OOF=True,NEW_contacted=False)
  if epoch==3:
   assert updates==2481 and exposures==39690 and coverage.all();complete['final_model']=C.ref(root/'epoch_3.pt');G.write(root/'completion.json',complete)
  G.write(segment/'completion.json',complete);print(json.dumps(complete),flush=True)
 except BaseException as error:
  G.write(segment/'failure.json',dict(error=repr(error),stage=stage,traceback=traceback.format_exc(),updates=updates,seconds=time.monotonic()-started,automatic_retry=False));raise
if __name__=='__main__':main()
