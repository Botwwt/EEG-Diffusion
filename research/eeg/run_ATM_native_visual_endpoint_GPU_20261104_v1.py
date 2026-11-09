"""Frozen native renderer and DINO; only the two privileged visual arms are new."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,hashlib,io,json,resource,subprocess,time,traceback
from pathlib import Path
import numpy as np
import atm_native_visual_endpoint_contract_20261104_v1 as V

ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('generate','encode'),required=True);args=ap.parse_args()
c=V.setup(args.policy);K=c['K'];q=c['q'];out=c['out'];parent=c['parent'];repo=c['repo']
assert K.ref(__file__)==q['sources']['GPU']
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
before=V.resources(c,starting=True)
name=q['job_prefix']+'_'+args.mode;job=K.read(repo/'audit'/(name+'_job.json'))
assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==parent['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
prepared=K.read(out/'prepared.json');assert prepared['policy']==K.ref(args.policy)
queries=K.read(K.checked(prepared['queries']));values=np.load(K.checked(prepared['conditions']),allow_pickle=False)

def guard():
    assert time.monotonic()-started<850
    used=sum(p.stat().st_blocks*512 for p in Path(q['project_root']).rglob('*') if p.is_file())
    s=os.statvfs(q['project_root']);assert used<q['maximum_output_bytes']
    assert s.f_bavail*s.f_frsize>=before['required_remaining_bytes']+q['maximum_output_bytes']-used
    assert torch.cuda.max_memory_allocated()<parent['maximum_GPU_bytes']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<parent['maximum_extra_CPU_bytes']

try:
    baseline=V.records(c)
    if args.mode=='generate':
        from atm_native_training_runtime_20261103_v1 import load_host,tensor_sha
        from brainhive_native_direction_contract_20261103_v1 import tensor_digest
        loader_path=out/'native_loader';loader_path.mkdir()
        pipe,projection,prompt,pooled,timeids,diffusion,proof=load_host(c['bq'],loader_path)
        unet_before=tensor_digest(pipe.unet,half=True);vae_before=tensor_digest(pipe.vae,half=True)
        import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
        old_noise=EA.randn_tensor;old_prepare=pipe.prepare_latents;capture={}
        def capture_noise(*a,**kw):
            result=old_noise(*a,**kw);capture['ancestral'].append(tensor_sha(result));return result
        def capture_initial(*a,**kw):
            result=old_prepare(*a,**kw);capture['initial'].append(tensor_sha(result));return result
        EA.randn_tensor=capture_noise;pipe.prepare_latents=capture_initial
        def generate(index,arm):
            value=torch.tensor(values[index,V.ARMS.index(arm)][None].copy(),device='cuda')
            with torch.inference_mode():
                tokens=projection(value.half())
                image=pipe(prompt_embeds=prompt,pooled_prompt_embeds=pooled,height=512,width=512,ip_adapter_image_embeds=[value[:,None].half()],num_inference_steps=4,guidance_scale=0.,generator=torch.Generator(device='cuda').manual_seed(queries[index]['image_seed'])).images[0]
            return value,tokens,image
        try:
            capture={'initial':[],'ancestral':[]};value,tokens,image=generate(0,'direct')
            pixelhash=hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest();reference=baseline['direct'][0]
            assert pixelhash==reference['pixels_sha256'] and capture==reference['noise']
            assert tensor_sha(value)==reference['condition_FP32_sha256'] and tensor_sha(tokens)==reference['tokens_FP16_sha256']
            image.save(out/'direct_q000_engineering_replay.webp',format='WEBP',lossless=True,quality=100)
            K.write(out/'engineering_replay.json',dict(status='PASS_NATIVE_DIRECT_PIXEL_NOISE_CONDITION_AND_TOKENS',reference=reference,loader=proof,new_fit=False))
            mapping=np.load(K.checked(prepared['same_concept_wrong_indices']),allow_pickle=False)
            for arm in V.NEW_ARMS:
                directory=out/'images'/arm;directory.mkdir(parents=True);journal=directory/'images.jsonl'
                assert not journal.exists()
                for i in range(80):
                    guard()
                    if i%16==0:V.resources(c)
                    capture={'initial':[],'ancestral':[]};value,tokens,image=generate(i,arm)
                    assert capture==baseline['direct'][i]['noise'] and len(capture['initial'])==1 and len(capture['ancestral'])==4
                    pixels=np.asarray(image.convert('RGB')).copy();assert pixels.shape==(512,512,3)
                    data=io.BytesIO();image.save(data,format='WEBP',lossless=True,quality=100);raw=data.getvalue()
                    used=sum(p.stat().st_blocks*512 for p in Path(q['project_root']).rglob('*') if p.is_file())
                    assert used+len(raw)+65536<q['maximum_output_bytes']
                    path=directory/f'q{i:03d}.webp';assert not path.exists();path.write_bytes(raw)
                    ti=i if arm=='oracle_CLIPH' else int(mapping[i])
                    record={**queries[i],'query_index':i,'arm':arm,'image':K.ref(path),
                            'pixels_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'noise':capture,
                            'condition_FP32_sha256':tensor_sha(value),'condition_FP16_sha256':tensor_sha(value.half()),'tokens_FP16_sha256':tensor_sha(tokens),
                            'visual_condition_query_index':ti,'visual_condition_stimulus_id':queries[ti]['stimulus_id'],
                            'privileged_true_image_diagnostic':True,'normal_EEG_method':False,'prior_NFE':0,'UNet_NFE':4,'prepared':K.ref(out/'prepared.json')}
                    with journal.open('a') as f:f.write(json.dumps(record)+'\n')
                    K.write(out/'generation_progress.json',dict(arm=arm,completed=i+1,total=80))
        finally:EA.randn_tensor=old_noise;pipe.prepare_latents=old_prepare
        assert tensor_digest(pipe.unet,half=True)==unet_before and tensor_digest(pipe.vae,half=True)==vae_before
        result=dict(status='COMPLETE_160_PRIVILEGED_VISUAL_ENDPOINT_IMAGES',new_research_images=160,engineering_replays=1,all_host_weights_unchanged=True,loader=proof)
    else:
        import image_evaluation as E
        audit=K.read(out/'image_audit.json');assert audit['status']=='PASS_ALL320_ENDPOINT_IMAGE_PROVENANCE'
        records=K.read(K.checked(audit['ordered_records']));old=np.load(K.checked(q['base_features']),allow_pickle=False)
        assert old.shape==(400,9,768)
        model=E.dino_model();assert E.DINO_SHA256==q['DINO_weight_sha256']
        features=np.empty((80,4,768),np.float32);features[:,:2]=old[:80,[2,8]]
        replay=E.dino_features([K.checked(baseline['direct'][0]['image'])],model,batch_size=1,real_images=False).float().numpy()[0]
        error=float(abs(replay-old[0,2]).max());assert error<3e-6
        new=[v for v in records if v['arm'] in V.NEW_ARMS];assert len(new)==160
        for lo in range(0,160,32):
            guard();V.resources(c);rows=new[lo:lo+32]
            got=E.dino_features([K.checked(r['image']) for r in rows],model,batch_size=32,real_images=False).float().numpy()
            for r,value in zip(rows,got):features[r['query_index'],V.ARMS.index(r['arm'])]=value
        assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features,axis=2),1,atol=3e-5)
        assert np.array_equal(features[:,:2],old[:80,[2,8]])
        K.save_array(out/'features80.npy',features)
        result=dict(status='COMPLETE_160_NEW_DINO_FEATURES_160_REUSED',features=K.ref(out/'features80.npy'),new_images_encoded=160,baseline_features_reused=160,
                    baseline_features=q['base_features'],native_DINO_single_replay_maximum_error=error,weight_sha256=E.DINO_SHA256)
    guard();result.update(seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
                          resources_before=before,resources_after=V.resources(c),source=K.ref(__file__))
    K.write(out/(args.mode+'_completion.json'),result);print(json.dumps(result),flush=True)
except BaseException as exc:
    K.write(out/(args.mode+'_failure.json'),dict(status='FAILED_PREFIX_RETAINED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc()));raise
