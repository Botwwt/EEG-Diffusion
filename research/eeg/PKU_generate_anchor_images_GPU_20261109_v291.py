"""Fixed all830 Dcheck x14 arms. One paired-noise image/query, safe finite prefixes."""
import argparse,datetime,functools,hashlib,json,os,shutil,socket,subprocess,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,required=True);args=ap.parse_args();assert 1<=args.stage<=20
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_anchor_images_v288';start=time.monotonic();jid=os.environ['SLURM_JOB_ID']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,r):
    b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)

try:
    import numpy as np
    import torch
    from PIL import Image
    from diffusers.utils.torch_utils import randn_tensor
    from PKU_native_generator_20261109_v253 import load_native
    torch.set_num_threads(4);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    policy_path=R/'code/EEG_anchor_image_comparison_v288.json';spec=json.loads(policy_path.read_text())
    proof=json.loads((out/'conditions_completion.json').read_text());assert proof['status']=='PASS_ALL3_ANCHORED_ENCODERS_830_OWN_EEG_FORWARDS_AND6_CONDITIONS'
    assert sha(out/'condition_bank.pt')==proof['condition_bank_sha256'] and sha(policy_path)==proof['policy_sha256']
    bank=torch.load(out/'condition_bank.pt',map_location='cpu',weights_only=True);assert bank['arms']==spec['arms'] and bank['condition'].shape==(6,830,1024)
    assert not (out/'completion.json').exists();records=out/'records';images=out/'images';stages=out/'stages'
    for folder in [records,images,stages]:folder.mkdir(exist_ok=True)
    assert not (stages/('stage_'+str(args.stage)+'.json')).exists()
    if args.stage>1:
        previous=json.loads((stages/('stage_'+str(args.stage-1)+'.json')).read_text());assert previous['next_job_id']==jid
    current=0
    while (records/('%05d.json'%current)).exists():current+=1
    assert current<4980 and len(list(records.glob('*.json')))==current
    beginning=current;assert shutil.disk_usage(out).free>10<<30
    pipe,A,P,prompts,host=load_native();assert host['IP_layers']==70
    # Keep original all-GPU pipeline placement so Diffusers infers the actual CUDA execution device.
    assert pipe._execution_device.type=='cuda';torch.cuda.empty_cache()
    calls={};processor=type(next(p for n,p in pipe.unet.attn_processors.items() if n.endswith('attn2.processor')));original=processor.__call__
    @functools.wraps(original)
    def checked(self,*a,**kw):
        token=kw.get('ip_hidden_states');assert token is not None and token.shape==(1,4,2048) and token.data_ptr()==calls['token_pointer'];calls['IP_calls']+=1
        return original(self,*a,**kw)
    processor.__call__=checked
    def unet_hook(module,arguments,result):
        calls['UNet_NFE']+=arguments[0].shape[0];assert torch.isfinite(result[0]).all()
    def vae_hook(module,arguments,result):
        calls['VAE_calls']+=1;assert torch.isfinite(result).all()
    handles=[pipe.unet.register_forward_hook(unet_hook),pipe.vae.decoder.register_forward_hook(vae_hook)]
    manifest_path=out/('inference_'+jid+'.jsonl');assert not manifest_path.exists();torch.cuda.reset_peak_memory_stats()
    with manifest_path.open('x',buffering=1) as log:
        while current<4980 and current-beginning<512 and time.monotonic()-start<900:
            qi,ai=divmod(current,6);arm=bank['arms'][ai];sid=bank['stimulus_ids'][qi];row=int(bank['array_rows'][qi])
            image_seed=int(hashlib.sha256(('1729|sub-03|'+sid+'|R4|EEG_Dcheck830_v271').encode()).hexdigest()[:16],16)%(2**63-1)
            generator=torch.Generator(device='cuda').manual_seed(image_seed)
            noise=randn_tensor((1,4,64,64),generator=generator,device=torch.device('cuda'),dtype=prompts[0].dtype)
            noise_sha=hashlib.sha256(noise.cpu().contiguous().numpy().tobytes()).hexdigest();rng_after_initial=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
            c=bank['condition'][ai,qi:qi+1].cuda();calls.clear();calls.update(UNet_NFE=0,IP_calls=0,VAE_calls=0)
            with torch.inference_mode():
                native=c.to(torch.float16);tokens=P(A(native));assert tokens.shape==(1,4,2048) and torch.isfinite(tokens).all();calls['token_pointer']=tokens.data_ptr()
                im=pipe(prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],width=512,height=512,num_inference_steps=4,guidance_scale=0.,generator=generator,
                    latents=noise,cross_attention_kwargs=dict(ip_hidden_states=tokens),output_type='pil').images[0].convert('RGB')
            assert im.size==(512,512) and calls['UNet_NFE']==4 and calls['IP_calls']==280 and calls['VAE_calls']==1
            calls.pop('token_pointer');measured_calls=dict(calls);assert torch.cuda.max_memory_allocated()<20<<30
            pixel_sha=hashlib.sha256(im.tobytes()).hexdigest();parity=None
            if current==0:
                # Check that explicitly recording the initial noise preserves the original pipeline RNG path.
                final_rng=generator.get_state().clone();reference_generator=torch.Generator(device='cuda').manual_seed(image_seed)
                calls.clear();calls.update(UNet_NFE=0,IP_calls=0,VAE_calls=0,token_pointer=tokens.data_ptr())
                with torch.inference_mode():
                    reference=pipe(prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],width=512,height=512,num_inference_steps=4,guidance_scale=0.,generator=reference_generator,
                        cross_attention_kwargs=dict(ip_hidden_states=tokens),output_type='pil').images[0].convert('RGB')
                assert hashlib.sha256(reference.tobytes()).hexdigest()==pixel_sha and torch.equal(reference_generator.get_state(),final_rng)
                assert calls['UNet_NFE']==4 and calls['IP_calls']==280 and calls['VAE_calls']==1
                parity=dict(original_native_pipeline_pixel_equal=True,original_native_pipeline_final_rng_equal=True,extra_engineering_UNet_calls=4)
                write(out/'original_pipeline_noise_parity.json',parity)
            path=images/('%05d_%s.webp'%(current,arm));assert not path.exists()
            temp=path.with_name(path.name+'.pending');assert not temp.exists();im.save(temp,format='WEBP',lossless=True,quality=90,method=4)
            with temp.open('rb') as f:os.fsync(f.fileno())
            with Image.open(temp) as check:assert check.size==(512,512) and hashlib.sha256(check.convert('RGB').tobytes()).hexdigest()==pixel_sha
            temp.replace(path)
            rec=dict(index=current,query_index=qi,arm=arm,array_row=row,stimulus_id=sid,subject='sub-03',repeat_protocol='R4',image_seed=image_seed,
                initial_noise_sha256=noise_sha,generator_after_initial_sha256=rng_after_initial,generator_final_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
                condition_sha256=hashlib.sha256(bank['condition'][ai,qi].contiguous().numpy().tobytes()).hexdigest(),actual_A_input_fp16_sha256=hashlib.sha256(native.cpu().contiguous().numpy().tobytes()).hexdigest(),
                condition_bank_sha256=proof['condition_bank_sha256'],image_path=str(path),file_sha256=sha(path),pixel_sha256=pixel_sha,bytes=path.stat().st_size,
                **measured_calls,noise_path_engineering=parity,normal_inference_target_access=False,one_image_per_query=True,selected_best_seed=False,job_id=jid)
            if ai:
                reference=json.loads((records/('%05d.json'%(current-ai))).read_text())
                for key in ['image_seed','initial_noise_sha256','generator_after_initial_sha256','generator_final_sha256']:assert rec[key]==reference[key]
            # Reuse exact original query noise, including every ancestral draw, across both studies.
            old=json.loads((R/'runs/EEG_images_v271/records'/('%05d.json'%(qi*14))).read_text())
            assert old['stimulus_id']==sid and old['array_row']==row
            for key in ['image_seed','initial_noise_sha256','generator_after_initial_sha256','generator_final_sha256']:assert rec[key]==old[key]
            write(records/('%05d.json'%current),rec);log.write(json.dumps(rec)+'\n');current+=1
            if current%28==0 or current==4980:
                write(out/'progress.json',dict(stage=args.stage,job_id=jid,images=current,expected_images=4980,completed_queries=current//6,seconds=time.monotonic()-start,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated()))
                assert sum(p.stat().st_size for p in images.iterdir())<4<<30 and shutil.disk_usage(out).free>1<<30
    for handle in handles:handle.remove()
    processor.__call__=original
    # Reopen and verify every newly committed image independently on CPU before dispatch.
    for i in range(beginning,current):
        rec=json.loads((records/('%05d.json'%i)).read_text());p=Path(rec['image_path']);assert p.parent==images and sha(p)==rec['file_sha256']
        with Image.open(p) as im:assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==rec['pixel_sha256']
    assert current>beginning
    stage=dict(stage=args.stage,job_id=jid,images=current,expected_images=4980,new_images=current-beginning,CPU_reopened_images=current-beginning,seconds=time.monotonic()-start,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),policy_sha256=sha(policy_path),condition_bank_sha256=proof['condition_bank_sha256'],source_sha256=sha(__file__),scoring_ready=False)
    if current==4980:
        stage['status']='COMPLETE_ALL830_QUERIES_6_ANCHOR_ARMS_IMAGES_SCORES_PENDING';write(out/'completion.json',stage)
    elif args.stage==20:
        stage['status']='FINITE_20_STAGE_CAP_SAFE_IMAGES_NOT_FULL_STUDY';write(out/'cap_stop.json',stage)
    else:
        claim=stages/('dispatch_after_'+str(args.stage)+'.json')
        write(claim,dict(parent_job=jid,next_stage=args.stage+1))
        p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+jid,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code/PKU_generate_anchor_images_GPU_20261109_v291.sbatch'),str(args.stage+1)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
        next_jid=p.stdout.strip().split(';')[0];assert p.returncode==0 and next_jid.isdigit(),(p.returncode,p.stdout,p.stderr)
        stage.update(status='SAFE_PREFIX_CPU_PIXELS_VERIFIED_NEXT_SUBMITTED',next_job_id=next_jid)
    write(stages/('stage_'+str(args.stage)+'.json'),stage);write(out/'progress.json',stage);print(json.dumps(stage),flush=True)
except BaseException as e:
    if out.exists():write(out/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED_NO_AUTO_RETRY',stage=args.stage,error=repr(e),traceback=traceback.format_exc()))
    raise
