"""Native unchanged renderer, fixed BN/direction/radius interventions; no fit."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,hashlib,io,json,resource,subprocess,time,traceback
from pathlib import Path
import numpy as np
import atm_BN_image_contract_20261104_v1 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('generate','encode'),required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stage',type=int,default=400);args=ap.parse_args()
c=C.setup(args.policy);K=c['K'];q=c['q'];out=c['out'];parent=c['parent'];started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;before=C.resources(c,starting=True)
assert K.ref(__file__)==q['sources']['GPU']
suffix=args.mode+('_'+str(args.start) if args.mode=='generate' else '_'+str(args.stage));job=K.read(c['repo']/'audit'/(q['job_prefix']+'_'+suffix+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==parent['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import tensor_sha
from brainhive_native_direction_contract_20261103_v1 import tensor_digest
def guard():
    assert time.monotonic()-started<850
    used=sum(p.stat().st_blocks*512 for p in Path(q['project_root']).rglob('*') if p.is_file());assert used<q['maximum_output_bytes']
    s=os.statvfs(q['project_root']);assert s.f_bavail*s.f_frsize>=before['required_remaining_bytes']+q['maximum_output_bytes']-used
    assert torch.cuda.max_memory_allocated()<parent['maximum_GPU_bytes'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<parent['maximum_extra_CPU_bytes']
try:
    if args.mode=='generate':
        assert args.start in (0,80,160,240,320);assert K.read(K.checked(q['BN_CPU_audit']))['status']=='PASS_FULL_T_BN_STATE_AND_ALL400_NATIVE_CPU_CONDITIONS_RADIUS_PROJECTOR_AUDIT'
        from atm_native_training_runtime_20261103_v1 import load_host
        prepared=K.read(out/'prepared.json');values=np.load(K.checked(prepared['conditions']),allow_pickle=False);loader_dir=out/('loader_'+str(args.start));loader_dir.mkdir()
        pipe,projection,prompt,pooled,timeids,diffusion,proof=load_host(c['cv']['bq'],loader_dir);unet=tensor_digest(pipe.unet,half=True);vae=tensor_digest(pipe.vae,half=True)
        import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
        old_noise=EA.randn_tensor;old_prepare=pipe.prepare_latents;capture={}
        def cn(*a,**kw):
            x=old_noise(*a,**kw);capture['ancestral'].append(tensor_sha(x));return x
        def ci(*a,**kw):
            x=old_prepare(*a,**kw);capture['initial'].append(tensor_sha(x));return x
        EA.randn_tensor=cn;pipe.prepare_latents=ci
        def generate(i,arm):
            value=torch.tensor(values[i,C.ARMS.index(arm)][None].copy(),device='cuda')
            with torch.inference_mode():
                token=projection(value.half());image=pipe(prompt_embeds=prompt,pooled_prompt_embeds=pooled,height=512,width=512,ip_adapter_image_embeds=[value[:,None].half()],num_inference_steps=4,guidance_scale=0.,generator=torch.Generator(device='cuda').manual_seed(c['queries'][i]['image_seed'])).images[0]
            return value,token,image
        try:
            if args.start==0:
                capture={'initial':[],'ancestral':[]};value,token,image=generate(0,'direct');ref=c['baseline']['direct'][0]
                assert hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()==ref['pixels_sha256'] and capture==ref['noise'] and tensor_sha(token)==ref['tokens_FP16_sha256']
                image.save(out/'direct_q000_engineering_replay.webp',format='WEBP',lossless=True,quality=100)
            for arm in C.MODELS:
                directory=out/'images'/arm;directory.mkdir(parents=True,exist_ok=True);journal=directory/'images.jsonl'
                rows=[json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
                assert [r['query_index'] for r in rows]==list(range(args.start))
            for i in range(args.start,args.start+80):
                guard()
                if i%16==0:C.resources(c)
                for arm in C.MODELS:
                    capture={'initial':[],'ancestral':[]};value,token,image=generate(i,arm);assert capture==c['baseline']['direct'][i]['noise']
                    pixels=np.asarray(image.convert('RGB')).copy();assert pixels.shape==(512,512,3)
                    data=io.BytesIO();image.save(data,format='WEBP',lossless=True,quality=100);path=out/'images'/arm/f'q{i:03d}.webp';assert not path.exists();guard()
                    used=sum(p.stat().st_blocks*512 for p in Path(q['project_root']).rglob('*') if p.is_file());assert used+len(data.getvalue())+65536<q['maximum_output_bytes'];path.write_bytes(data.getvalue())
                    record={**c['queries'][i],'query_index':i,'arm':arm,'image':K.ref(path),'pixels_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'noise':capture,
                        'condition_FP32_sha256':tensor_sha(value),'condition_FP16_sha256':tensor_sha(value.half()),'tokens_FP16_sha256':tensor_sha(token),'normal_inference_only_EEG':True,'query_image_or_concept_read_at_inference':False,'UNet_NFE':4,'prior_NFE':0,'EEG_encoder_forward_cost':1 if arm=='BN_only' else 2,'prepared':K.ref(out/'prepared.json')}
                    with (out/'images'/arm/'images.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                K.write(out/'generation_progress.json',dict(block_start=args.start,completed_queries_per_new_arm=i+1))
        finally:EA.randn_tensor=old_noise;pipe.prepare_latents=old_prepare
        assert tensor_digest(pipe.unet,half=True)==unet and tensor_digest(pipe.vae,half=True)==vae
        result=dict(status='COMPLETE_240_NEW_BN_RADIUS_IMAGES_BLOCK',start=args.start,completed_queries_per_arm=args.start+80,new_images=240,all_frozen_host_weights_unchanged=True,loader=proof)
    else:
        assert args.stage==400
        import image_evaluation as E
        audit=K.read(out/f'image_audit{args.stage}.json');records=K.read(K.checked(audit['ordered_records']));base=np.load(K.checked(q['baseline_features']),allow_pickle=False)
        model=E.dino_model();assert E.DINO_SHA256==q['DINO_weight_sha256'];features=np.empty((args.stage,6,768),np.float32);features[:,:3]=base[:args.stage,[0,2,3]];begin=0
        replay=E.dino_features([K.checked(c['baseline']['direct'][0]['image'])],model,batch_size=1,real_images=False).float().numpy()[0];error=float(abs(replay-base[0,0]).max());assert error<3e-6
        new=[r for r in records if r['arm'] in C.MODELS and r['query_index']>=begin]
        assert len(new)==(args.stage-begin)*3
        for lo in range(0,len(new),32):
            guard();C.resources(c);rows=new[lo:lo+32];encoded=E.dino_features([K.checked(r['image']) for r in rows],model,batch_size=32,real_images=False).float().numpy()
            for r,v in zip(rows,encoded):features[r['query_index'],C.ARMS.index(r['arm'])]=v
        assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features,axis=2),1,atol=3e-5) and np.array_equal(features[:,:3],base[:args.stage,[0,2,3]])
        K.save_array(out/f'features{args.stage}.npy',features);result=dict(status='COMPLETE_FROZEN_SIX_ARM_DINO_FEATURES',stage=args.stage,features=K.ref(out/f'features{args.stage}.npy'),new_images_encoded=len(new),old_baseline_features_reused=args.stage*3,weight_sha256=E.DINO_SHA256,native_single_replay_max_error=error)
    guard();result.update(source=K.ref(__file__),seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_after=C.resources(c))
    K.write(out/(suffix+'_completion.json'),result);print(json.dumps(result),flush=True)
except BaseException as exc:
    K.write(out/(suffix+'_failure.json'),dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',error=repr(exc),traceback=traceback.format_exc()));raise
