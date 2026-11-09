"""Frozen native h94 generation in original baseline import context; no EEG fit."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,hashlib,io,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--amendment',required=True)
ap.add_argument('--mode',choices=('generate','encode'),required=True);ap.add_argument('--start',type=int,default=0)
args=ap.parse_args();sys.path.insert(0,str(Path(args.policy).parent))
import enigma_native_reference_contract_20261104_v1 as C
c=C.setup(args.policy);K=c['K'];q=c['q'];out=c['out'];parent=c['parent']
am=K.read(args.amendment);assert am['original_policy']==K.ref(args.policy)
assert am['status']=='ENGINEERING_FROZEN_GENERATOR_BASELINE_RUNTIME_RESTORATION'
assert K.ref(__file__)==am['sources']['GPU']
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;before=C.resources(c,starting=True)
suffix=args.mode+('_'+str(args.start) if args.mode=='generate' else '')
job=K.read(c['repo']/'audit'/(am['job_prefix']+'_'+suffix+'_job.json'))
assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==parent['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
# Deliberately use the proven baseline package context, not EEG-author fit paths.
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import tensor_sha
from brainhive_native_direction_contract_20261103_v1 import tensor_digest
def guard():
    assert time.monotonic()-started<850
    used=sum(p.stat().st_blocks*512 for p in Path(q['project_root']).rglob('*') if p.is_file())
    assert used<q['maximum_output_bytes']
    stat=os.statvfs(q['project_root']);assert stat.f_bavail*stat.f_frsize>=before['required_remaining_bytes']+q['maximum_output_bytes']-used
    assert torch.cuda.max_memory_allocated()<parent['maximum_GPU_bytes']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<parent['maximum_extra_CPU_bytes']
try:
    result={}
    if args.mode=='generate':
        assert args.start in (0,80,160,240,320)
        assert K.read(out/'conditions_CPU_audit.json')['status']=='PASS_NATIVE63_COMPLETE_OOF_AND_D_CPU_REPLAY'
        from atm_native_training_runtime_20261103_v1 import load_host
        prepared=K.read(out/'prepared.json');values=np.load(K.checked(prepared['conditions']),allow_pickle=False)
        loader=out/('loader_v7_'+str(args.start));loader.mkdir()
        pipe,projection,prompt,pooled,timeids,diffusion,proof=load_host(c['cv']['bq'],loader)
        unet=tensor_digest(pipe.unet,half=True);vae=tensor_digest(pipe.vae,half=True)
        import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
        old_noise,old_prepare=EA.randn_tensor,pipe.prepare_latents;capture={}
        def cn(*a,**kw):
            value=old_noise(*a,**kw);capture['ancestral'].append(tensor_sha(value));return value
        def ci(*a,**kw):
            value=old_prepare(*a,**kw);capture['initial'].append(tensor_sha(value));return value
        EA.randn_tensor,pipe.prepare_latents=cn,ci
        def generate(i,arm):
            value=torch.tensor(values[i,C.ARMS.index(arm)][None].copy(),device='cuda')
            with torch.inference_mode():
                token=projection(value.half())
                image=pipe(prompt_embeds=prompt,pooled_prompt_embeds=pooled,height=512,width=512,
                    ip_adapter_image_embeds=[value[:,None].half()],num_inference_steps=4,guidance_scale=0.,
                    generator=torch.Generator(device='cuda').manual_seed(c['queries'][i]['image_seed'])).images[0]
            return value,token,image
        try:
            if args.start==0:
                capture={'initial':[],'ancestral':[]};value,token,image=generate(0,'direct');ref=c['baseline']['direct'][0]
                import diffusers,transformers
                K.write(out/'native_replay_v7_actual.json',dict(status='ACTUAL_BEFORE_UNCHANGED_PIXEL_GATE',
                    condition_FP32_sha256=tensor_sha(value),tokens_FP16_sha256=tensor_sha(token),
                    actual_pixels_SHA256=hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest(),
                    expected_pixels_SHA256=ref['pixels_sha256'],actual_noise=capture,expected_noise=ref['noise'],
                    prompt_SHA256=tensor_sha(prompt),pooled_SHA256=tensor_sha(pooled),
                    modules={m.__name__:dict(version=m.__version__,path=m.__file__) for m in (torch,diffusers,transformers,np)},
                    source=K.ref(__file__),amendment=K.ref(args.amendment)))
                assert hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()==ref['pixels_sha256']
                assert capture==ref['noise'] and tensor_sha(token)==ref['tokens_FP16_sha256']
                image.save(out/'direct_q000_engineering_replay.webp',format='WEBP',lossless=True,quality=100)
            directory=out/'images/enigma_native63';directory.mkdir(parents=True,exist_ok=True);journal=directory/'images.jsonl'
            rows=[json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
            assert [r['query_index'] for r in rows]==list(range(args.start))
            for i in range(args.start,args.start+80):
                guard()
                if i%16==0:C.resources(c)
                capture={'initial':[],'ancestral':[]};value,token,image=generate(i,'enigma_native63')
                assert capture==c['baseline']['direct'][i]['noise']
                pixels=np.asarray(image.convert('RGB')).copy();assert pixels.shape==(512,512,3)
                data=io.BytesIO();image.save(data,format='WEBP',lossless=True,quality=100)
                path=directory/f'q{i:03d}.webp';assert not path.exists()
                used=sum(p.stat().st_blocks*512 for p in Path(q['project_root']).rglob('*') if p.is_file())
                assert used+len(data.getvalue())+65536<q['maximum_output_bytes'];path.write_bytes(data.getvalue())
                record={**c['queries'][i],'query_index':i,'arm':'enigma_native63','image':K.ref(path),
                    'pixels_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'noise':capture,
                    'condition_FP32_sha256':tensor_sha(value),'condition_FP16_sha256':tensor_sha(value.half()),
                    'tokens_FP16_sha256':tensor_sha(token),'normal_inference_only_EEG':True,
                    'query_image_or_concept_read_at_inference':False,'UNet_NFE':4,'prior_NFE':0,
                    'EEG_forward_calls_at_deployment':1,'prepared':K.ref(out/'prepared.json')}
                with journal.open('a') as f:f.write(json.dumps(record)+'\n')
                K.write(out/'generation_progress.json',dict(completed_queries=i+1,block_start=args.start))
        finally:EA.randn_tensor,pipe.prepare_latents=old_noise,old_prepare
        assert tensor_digest(pipe.unet,half=True)==unet and tensor_digest(pipe.vae,half=True)==vae
        result=dict(status='COMPLETE_NATIVE63_REFERENCE_80_IMAGE_BLOCK',start=args.start,new_images=80,
            completed_queries=args.start+80,all_frozen_host_weights_unchanged=True,loader=proof)
    else:
        import image_evaluation as E
        audit=K.read(out/'image_audit400.json');records=K.read(K.checked(audit['ordered_records']))
        base=np.load(K.checked(q['baseline_features']),allow_pickle=False);assert base.shape==(400,6,768)
        evaluator=E.dino_model();assert E.DINO_SHA256==q['DINO_weight_sha256']
        features=np.empty((400,7,768),np.float32);features[:,:6]=base
        replay=E.dino_features([K.checked(c['baseline']['direct'][0]['image'])],evaluator,batch_size=1,real_images=False).float().numpy()[0]
        error=float(abs(replay-base[0,0]).max());assert error<3e-6
        new=[r for r in records if r['arm']=='enigma_native63'];assert len(new)==400
        for lo in range(0,400,32):
            guard();C.resources(c);rows=new[lo:lo+32]
            encoded=E.dino_features([K.checked(r['image']) for r in rows],evaluator,batch_size=32,real_images=False).float().numpy()
            features[lo:lo+len(rows),6]=encoded
        assert np.isfinite(features).all() and np.array_equal(features[:,:6],base)
        assert np.allclose(np.linalg.norm(features,axis=2),1,atol=3e-5)
        K.save_array(out/'features400.npy',features)
        result=dict(status='COMPLETE_SEVEN_ARM_FROZEN_DINO_FEATURES',features=K.ref(out/'features400.npy'),
            new_images_encoded=400,reused_images=2400,native_single_replay_max_error=error,weight_sha256=E.DINO_SHA256)
    guard();result.update(source=K.ref(__file__),engineering_amendment=K.ref(args.amendment),
        seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_after=C.resources(c))
    K.write(out/(suffix+'_completion.json'),result);print(json.dumps(result),flush=True)
except BaseException as exc:
    K.write(out/('runtime_v7_'+suffix+'_failure.json'),dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',
        error=repr(exc),traceback=traceback.format_exc(),source=K.ref(__file__),amendment=K.ref(args.amendment)))
    raise
