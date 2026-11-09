"""Frozen native h94/Turbo four-arm R80 images and complete metric suite."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,hashlib,io,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True)
ap.add_argument('--mode',choices=('profile','generate','encode','paper'),required=True)
ap.add_argument('--start',type=int,default=0);ap.add_argument('--arm',choices=('EMA_original','EMA_count','T_population','T_count'))
a=ap.parse_args();sys.path.insert(0,str(Path(a.policy).parent))
import enigma_R80_count_image_contract_20261104_v1 as C
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(a.policy);K=c['K'];q=c['q'];out=c['out'];before=C.resources(c,True)
assert K.ref(__file__)==q['sources']['GPU']
suffix=a.mode+('_'+str(a.start) if a.mode=='generate' else '_'+a.arm if a.mode=='paper' else '')
j=K.read(c['repo']/'audit'/(q['job_prefix']+'_'+suffix+'_job.json'));assert j['pid']==os.getpid() and j['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['parent']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(c['parent']['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import tensor_sha
def guard():
    assert time.monotonic()-started<850
    used=C.allocated(q['project_root']);assert used<q['maximum_output_bytes']
    st=os.statvfs(q['project_root']);assert st.f_bavail*st.f_frsize>=before['required_prior_bytes']+q['maximum_output_bytes']-used
    assert torch.cuda.max_memory_allocated()<c['parent']['maximum_GPU_bytes']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<c['parent']['maximum_extra_CPU_bytes']
try:
    result={}
    if a.mode in ('profile','generate'):
        from atm_native_training_runtime_20261103_v1 import load_host
        from brainhive_native_direction_contract_20261103_v1 import tensor_digest
        directory=out/('loader_'+suffix);directory.mkdir()
        pipe,projection,prompt,pooled,timeids,diffusion,proof=load_host(c['cv']['bq'],directory)
        import diffusers,transformers
        assert transformers.__version__=='4.36.0'
        frozen={k:tensor_digest(v,half=True) for k,v in [('UNet',pipe.unet),('VAE',pipe.vae)]}
        import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
        old_noise,old_prepare=EA.randn_tensor,pipe.prepare_latents;capture={}
        def cn(*args,**kw):
            x=old_noise(*args,**kw);capture['ancestral'].append(tensor_sha(x));return x
        def ci(*args,**kw):
            x=old_prepare(*args,**kw);capture['initial'].append(tensor_sha(x));return x
        EA.randn_tensor,pipe.prepare_latents=cn,ci
        def generate(value,seed):
            value=torch.tensor(value[None].copy(),device='cuda')
            with torch.inference_mode():
                token=projection(value.half())
                image=pipe(prompt_embeds=prompt,pooled_prompt_embeds=pooled,height=512,width=512,
                    ip_adapter_image_embeds=[value[:,None].half()],num_inference_steps=4,guidance_scale=0.,
                    generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
            return value,token,image
        try:
            if a.mode=='profile':
                ref=q['native_pixel_reference'];value=np.load(K.checked(q['native_pixel_conditions']),allow_pickle=False)[0,0]
                capture={'initial':[],'ancestral':[]};vv,tt,image=generate(value,ref['image_seed'])
                pixels=hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()
                K.write(out/'native_replay_actual.json',dict(condition=tensor_sha(vv),tokens=tensor_sha(tt),
                    pixels_SHA256=pixels,expected_pixels_SHA256=ref['pixels_sha256'],noise=capture,expected_noise=ref['noise'],
                    versions={v.__name__:dict(version=v.__version__,path=v.__file__) for v in (torch,diffusers,transformers,np)}))
                assert pixels==ref['pixels_sha256'] and capture==ref['noise'] and tensor_sha(tt)==ref['tokens_FP16_sha256']
                image.save(out/'native_R4_engineering_replay.webp',format='WEBP',lossless=True,quality=100)
                tokens=np.empty((200,4,4,2048),np.float16)
                with torch.inference_mode():
                    for i in range(200):
                        if i%16==0:guard()
                        for n in range(4):
                            v=torch.tensor(c['values'][i,n][None].copy(),device='cuda')
                            tokens[i,n]=projection(v.half()).cpu().numpy()[0]
                assert np.isfinite(tokens).all();K.save_array(out/'actual_IP_tokens.npy',tokens)
                result=dict(status='PASS_NATIVE_PIXEL_NOISE_AND_ALL800_IP_TOKEN_PROFILE',tokens=K.ref(out/'actual_IP_tokens.npy'),
                    original_pixels_and_noise_exact=True,EEG_condition_CPU_replay=q['measurement_CPU_replay'],new_research_images=0,engineering_images=1)
            else:
                assert a.start in (0,50,100,150)
                p=K.read(out/'profile_completion.json');assert p['status']=='PASS_NATIVE_PIXEL_NOISE_AND_ALL800_IP_TOKEN_PROFILE'
                tokens=np.load(K.checked(p['tokens']),allow_pickle=False)
                for arm in C.ARMS:
                    d=out/'images'/arm;d.mkdir(parents=True,exist_ok=True)
                    journal=d/'images.jsonl';old=[json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
                    assert [x['query_index'] for x in old]==list(range(a.start))
                for i in range(a.start,a.start+50):
                    guard()
                    if i%8==0:C.resources(c)
                    common=None
                    for n,arm in enumerate(C.ARMS):
                        capture={'initial':[],'ancestral':[]};value,token,image=generate(c['values'][i,n],c['rows'][i]['image_seed'])
                        assert np.array_equal(token.cpu().numpy()[0],tokens[i,n])
                        assert len(capture['initial'])==1 and len(capture['ancestral'])==4
                        if common is None:common=capture
                        else:assert capture==common
                        pixels=np.asarray(image.convert('RGB')).copy();assert pixels.shape==(512,512,3)
                        data=io.BytesIO();image.save(data,format='WEBP',lossless=True,quality=100)
                        path=out/'images'/arm/f'q{i:03d}.webp';assert not path.exists()
                        assert C.allocated(q['project_root'])+len(data.getvalue())+65536<q['maximum_output_bytes']
                        path.write_bytes(data.getvalue())
                        row={**c['rows'][i],'query_index':i,'arm':arm,'image':K.ref(path),
                            'pixels_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'noise':capture,
                            'condition_FP32_sha256':tensor_sha(value),'condition_FP16_sha256':tensor_sha(value.half()),
                            'tokens_FP16_sha256':tensor_sha(token),'condition_bank':q['measurement_conditions'],
                            'normal_inference_uses_only_EEG_and_known_R':True,'query_image_or_label_network_input':False,
                            'UNet_NFE':4,'prior_NFE':0,'EEG_deployment_forward_calls':1,'EEG_forwards_in_cached_generation':0}
                        with (path.parent/'images.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    K.write(out/'generation_progress.json',dict(completed_queries=i+1,research_images=4*(i+1),all4arms=True))
                result=dict(status='COMPLETE_ALL4_ARMS_50_R80_QUERIES',start=a.start,completed_queries=a.start+50,new_images=200)
        finally:EA.randn_tensor,pipe.prepare_latents=old_noise,old_prepare
        assert frozen=={k:tensor_digest(v,half=True) for k,v in [('UNet',pipe.unet),('VAE',pipe.vae)]}
        result.update(loader=proof,all_generator_weights_unchanged=True)
    elif a.mode=='encode':
        audit=K.read(out/'image_audit.json');assert audit['status']=='PASS_ALL800_R80_IMAGE_IDENTITY_CONDITIONS_TOKENS_NOISE_PIXELS'
        import image_evaluation as E
        assert K.ref(E.__file__)==q['runtime_sources']['image_evaluation'] and E.DINO_SHA256==q['DINO_weight_sha256']
        model=E.dino_model();target_paths=[K.checked(r['target']) for r in c['rows']]
        target=E.dino_features(target_paths,model,batch_size=32,real_images=True).float().numpy()
        features=np.empty((200,4,768),np.float32)
        records=K.read(K.checked(audit['ordered_records']))
        for n,arm in enumerate(C.ARMS):
            rows=[r for r in records if r['arm']==arm];assert len(rows)==200
            for lo in range(0,200,32):
                guard();C.resources(c)
                features[lo:lo+len(rows[lo:lo+32]),n]=E.dino_features([K.checked(r['image']) for r in rows[lo:lo+32]],model,batch_size=32,real_images=False).float().numpy()
        assert np.isfinite(features).all() and np.isfinite(target).all()
        assert np.allclose(np.linalg.norm(features,axis=2),1,atol=3e-5) and np.allclose(np.linalg.norm(target,axis=1),1,atol=3e-5)
        K.save_array(out/'features.npy',features);K.save_array(out/'target_features.npy',target)
        result=dict(status='COMPLETE_ALL800_AND200_TARGET_FROZEN_DINO',features=K.ref(out/'features.npy'),target=K.ref(out/'target_features.npy'),
            DINO_weight_sha256=E.DINO_SHA256,real_resize_CPU_generated_resize_CUDA_historical_routes_unchanged=True)
    else:
        assert a.arm in C.ARMS
        audit=K.read(out/'image_audit.json');assert audit['status']=='PASS_ALL800_R80_IMAGE_IDENTITY_CONDITIONS_TOKENS_NOISE_PIXELS'
        from brainhive_original_metrics_runtime_v1 import OriginalEvaluator,NAMES,transforms
        from PIL import Image
        assert tuple(NAMES)==C.PAPER
        pre=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512),transforms.ToTensor()])
        records=K.read(K.checked(audit['ordered_records']));rows=[r for r in records if r['arm']==a.arm]
        originals=[];generated=[]
        for r in rows:
            guard()
            with Image.open(K.checked(r['target'])) as im:originals.append(pre(im.convert('RGB')))
            with Image.open(K.checked(r['image'])) as im:generated.append(pre(im.convert('RGB')))
        ev=OriginalEvaluator()
        try:
            aggregate,per=ev.evaluate(torch.stack(originals),torch.stack(generated))
            values=np.stack([per[k] for k in C.PAPER],axis=1).astype(np.float64)
            K.save_array(out/('paper_'+a.arm+'.npy'),values)
            result=dict(status='COMPLETE_ALL200_ORIGINAL_AUTHOR_EIGHT_METRICS',arm=a.arm,values=K.ref(out/('paper_'+a.arm+'.npy')),
                metrics=list(C.PAPER),defined_counts=np.isfinite(values).sum(0).tolist(),aggregate={k:float(v) if np.isfinite(v) else None for k,v in aggregate.items()},
                numerical_source_checks=ev.checks,assets=q['original_metric_assets'],author_two_way_identification_orientation_unchanged=True)
        finally:ev.close()
    guard();result.update(source=K.ref(__file__),policy=K.ref(a.policy),seconds=time.monotonic()-started,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
        resources_after=C.resources(c))
    K.write(out/(suffix+'_completion.json'),result);print(json.dumps(result),flush=True)
except BaseException as exc:
    K.write(out/(suffix+'_failure.json'),dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',error=repr(exc),traceback=traceback.format_exc(),
        source=K.ref(__file__),policy=K.ref(a.policy),seconds=time.monotonic()-started));raise
