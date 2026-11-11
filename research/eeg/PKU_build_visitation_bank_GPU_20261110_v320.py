"""Real four-step EEG trajectories and timestep/sigma-matched real-image forward states."""
import argparse,hashlib,json,os,socket,time,traceback
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');a=ap.parse_args()
from PKU_conditional_visitation_core_20261110_v320 import R,ROOT,sha,write,save,fixed_inputs
root=ROOT/'bank';root.mkdir(exist_ok=True);start=time.monotonic();jid=os.environ['SLURM_JOB_ID']
try:
    import numpy as np
    import torch
    from PKU_native_generator_20261109_v253 import load_native
    torch.set_num_threads(4);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==1 and not (root/'completion.json').exists()
    c,target,source=fixed_inputs('cuda');lp=R/'data/native_T_latents_v256/native_VAE_T_modes.npy';assert sha(lp)=='b1198b3e07361e3b941f87e3b60146b1cc7957293e813db69f7b59cfa8469a2d'
    lat=np.load(lp,mmap_mode='r');pipe,A,P,prompts,host=load_native();assert pipe._execution_device.type=='cuda'
    parts=root/'parts';parts.mkdir(exist_ok=True);records=sorted(parts.glob('*.json'));cursor=0
    for p in records:
        r=json.loads(p.read_text());assert r['lo']==cursor and (parts/r['file']).stat().st_size==r['bytes'];cursor=r['hi']
    assert cursor<13230 and (cursor==0)==a.profile
    beginning=cursor;limit=2 if a.profile else 512;captured=[]
    def hook(module,arguments):captured.append(arguments[0].detach().clone())
    handle=pipe.unet.register_forward_pre_hook(hook);torch.cuda.reset_peak_memory_stats();parity=None
    while cursor<13230 and cursor-beginning<limit and time.monotonic()-start<850:
        lo=cursor;n=min(32,13230-lo,limit-(cursor-beginning));ons=[];offs=[];seeds=[]
        for i in range(lo,lo+n):
            value=int(hashlib.sha256(('1729|sub-03|T|'+str(i)+'|visitation320').encode()).hexdigest()[:16],16)%(2**63-1);seeds.append(value)
            generator=torch.Generator(device='cuda').manual_seed(value);captured.clear()
            with torch.inference_mode():
                tokens=P(A(c[i:i+1].half()));result=pipe(prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],width=512,height=512,num_inference_steps=4,guidance_scale=0.,generator=generator,cross_attention_kwargs=dict(ip_hidden_states=tokens),output_type='latent').images
            assert len(captured)==4 and all(v.shape==(1,4,64,64) for v in captured)
            actual=torch.cat(captured).cpu();times=pipe.scheduler.timesteps.detach().float().cpu();sigmas=pipe.scheduler.sigmas[:4].detach().float().cpu();assert times.tolist()==[999.,749.,499.,249.]
            if i==0:
                original=actual.clone();last=result.clone();rng=generator.get_state();captured.clear();g2=torch.Generator(device='cuda').manual_seed(value)
                with torch.inference_mode():repeat=pipe(prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],width=512,height=512,num_inference_steps=4,guidance_scale=0.,generator=g2,cross_attention_kwargs=dict(ip_hidden_states=tokens),output_type='latent').images
                assert torch.equal(original,torch.cat(captured).cpu()) and torch.equal(last,repeat) and torch.equal(rng,g2.get_state());parity=dict(all4_native_UNet_inputs_exact=True,terminal_latent_exact=True,ancestral_RNG_exact=True,extra_engineering_UNet_calls=4);write(root/'native_trajectory_parity.json',parity)
            gcpu=torch.Generator().manual_seed(value);noise=torch.randn((4,4,64,64),generator=gcpu);s=sigmas[:,None,None,None];off=(torch.tensor(lat[i])[None]+s*noise)/torch.sqrt(1+s*s)
            assert torch.isfinite(actual).all() and torch.isfinite(off).all();ons.append(actual);offs.append(off.half())
        payload=dict(lo=lo,hi=lo+n,on=torch.stack(ons),forward=torch.stack(offs),times=times,sigmas=sigmas,seeds=seeds,source=source)
        path=parts/('%05d.pt'%lo);assert not path.exists();save(path,payload);write(path.with_suffix('.json'),dict(lo=lo,hi=lo+n,file=path.name,bytes=path.stat().st_size,sha256=sha(path),times=times.tolist(),sigmas=sigmas.tolist()))
        cursor+=n;assert sum(p.stat().st_size for p in parts.iterdir())<5<<30;assert torch.cuda.max_memory_allocated()<20<<30;write(root/'progress.json',dict(queries=cursor,total=13230,states=cursor*4,job_id=jid,seconds=time.monotonic()-start,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated()))
    handle.remove();assert cursor>beginning
    if cursor==13230:write(root/'completion.json',dict(status='COMPLETE_ALL13230_NATIVE_FOURSTEP_AND_MATCHED_FORWARD_STATE_BANK',queries=cursor,states_each_domain=cursor*4,new_updates=0,new_images=0,source=source,parts=len(list(parts.glob('*.pt'))),job_id=jid))
    write(root/('attempt_'+jid+'.json'),dict(lo=beginning,hi=cursor,profile=a.profile,new_updates=0,new_images=0,seconds=time.monotonic()-start));print('SAFE_BANK_PREFIX '+str(cursor),flush=True)
except BaseException as e:
    write(ROOT/('failure_bank_'+jid+'.json'),dict(error=repr(e),traceback=traceback.format_exc()));raise
