"""Whole native EEG encoder target experiment and frozen direct rendering."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,hashlib,io,json,resource,subprocess,time,traceback
from pathlib import Path
import numpy as np
import atm_conditional_stimulus_contract_20261104_v1 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('profile','fit','prepare','generate','encode'),required=True);ap.add_argument('--arm',default='');ap.add_argument('--start',type=int,default=0);ap.add_argument('--stage',type=int,default=80);args=ap.parse_args()
c=C.setup(args.policy);K=c['K'];q=c['q'];out=c['out'];parent=c['parent'];started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;before=C.resources(c,starting=True)
assert K.ref(__file__)==q['sources']['GPU']
suffix=args.mode+('_'+args.arm if args.mode=='fit' else '_'+str(args.start) if args.mode=='generate' else '_'+str(args.stage) if args.mode=='encode' else '')
name=q['job_prefix']+'_'+suffix;job=K.read(c['repo']/'audit'/(name+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==parent['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
import torch.nn.functional as F
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import tensor_sha
from brainhive_native_direction_contract_20261103_v1 import tensor_digest
def guard():
    assert time.monotonic()-started<850
    used=sum(p.stat().st_blocks*512 for p in Path(q['project_root']).rglob('*') if p.is_file());assert used<q['maximum_output_bytes']
    s=os.statvfs(q['project_root']);assert s.f_bavail*s.f_frsize>=before['required_remaining_bytes']+q['maximum_output_bytes']-used
    assert torch.cuda.max_memory_allocated()<parent['maximum_GPU_bytes'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<parent['maximum_extra_CPU_bytes']
def save_torch(path,data):
    temp=path.with_name(path.name+'.pending');assert not path.exists() and not temp.exists();torch.save(data,temp);temp.replace(path)
def native_class():
    import sys
    sys.path.insert(0,'/root/atm_official_20261016');from models.atms import ATMS
    return ATMS
def encoder_inputs():
    raw=torch.load(K.checked(q['EEG']),map_location='cpu',weights_only=True,mmap=True);assert raw.shape==(16540,63,250)
    return raw
def new_model(state=None):
    model=native_class()().cuda()
    model.load_state_dict(torch.load(K.checked(q['encoder']),map_location='cpu',weights_only=True,mmap=True) if state is None else state,strict=True)
    assert sum(p.numel() for p in model.parameters())==3200413
    return model
try:
    result={}
    if args.mode in ('profile','fit'):
        from atm_conditional_stimulus_objective_20261104_v1 import torch_loss,numpy_loss_gradient
        raw=encoder_inputs();targets=torch.load(K.checked(q['visual_target']),map_location='cpu',weights_only=True,mmap=True)
        assert targets.shape==(16540,1024);T=c['T'];concept_rows=T.reshape(-1,10)
        if args.mode=='profile':
            model=new_model().eval().requires_grad_(False);baseline=np.load(K.checked(c['oldprep']['conditions']),allow_pickle=False)[:,2];errs=[]
            with torch.inference_mode():
                for lo in range(0,400,128):
                    ids=[r['array_row'] for r in c['queries'][lo:lo+128]];pred=model(raw[ids].cuda(),torch.ones(len(ids),device='cuda',dtype=torch.long)).float().cpu().numpy()
                    errs.extend(np.max(abs(pred-baseline[lo:lo+len(ids)]),axis=1).tolist())
            assert max(errs)<5e-5;del model
            proofs=[]
            for arm in C.MODELS:
                model=new_model().train();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);ids=T[:160]
                torch.cuda.manual_seed(1729);pred=model(raw[ids].cuda(),torch.ones(160,device='cuda',dtype=torch.long));scores=model.logit_scale.exp().clamp_max(100)*F.normalize(pred,dim=1)@F.normalize(targets[ids].cuda(),dim=1).T
                loss,parts=torch_loss(scores,arm);reference=numpy_loss_gradient(scores.detach().cpu().numpy(),arm)
                derivative=torch.autograd.grad(loss,scores,retain_graph=True)[0].detach().cpu().numpy();error=float(abs(derivative-reference['gradient']).max())
                assert abs(float(loss)-reference['loss'])<3e-5 and error<3e-6 and float(parts['coarse_loss'])>=-1e-6
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(norm) and norm>0;opt.step()
                proofs.append(dict(arm=arm,real_batch=160,temporary_updates=1,loss=float(loss.detach()),gradient_norm=norm,NumPy_logits_gradient_max_error=error,numerical_parts={k:float(v.detach()) for k,v in parts.items()}))
                del model,opt,pred,scores,loss,parts,derivative;torch.cuda.empty_cache();guard()
            result=dict(status='PASS_REAL_NATIVE_ENCODER_THREE_TARGET_GRADIENTS_AND_ALL400_BASELINE_REPLAY',all400_baseline_max_error=max(errs),all400_baseline_errors=errs,
                proofs=proofs,temporary_profile_updates=3,temporary_models_discarded_not_final=True,parameters_each=3200413,T_visual_norm_max=float(targets[T].norm(dim=1).max()))
        else:
            assert args.arm in C.MODELS;profile=K.read(out/'profile_completion.json');assert profile['status']=='PASS_REAL_NATIVE_ENCODER_THREE_TARGET_GRADIENTS_AND_ALL400_BASELINE_REPLAY'
            directory=out/'fits'/args.arm;directory.mkdir(parents=True);assert not (directory/'final.pt').exists()
            torch.manual_seed(1729);torch.cuda.manual_seed(1729);model=new_model().train();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
            updates=exposures=0;epoch_log=directory/'epochs.jsonl';last_order=None
            for epoch in range(10):
                order=np.random.default_rng(1729+epoch).permutation(len(concept_rows));coverage=np.zeros(len(T),np.int8);sums=dict(global_loss=0.,correct_fine_loss=0.,supervised_fine_loss=0.,coarse_loss=0.,loss=0.);seen=0
                for bi,lo in enumerate(range(0,len(order),16)):
                    guard()
                    if bi%16==0:C.resources(c)
                    groups=order[lo:lo+16];positions=(groups[:,None]*10+np.arange(10)).reshape(-1);ids=T[positions];coverage[positions]+=1
                    torch.cuda.manual_seed(1729+epoch*10000+bi);opt.zero_grad(set_to_none=True)
                    pred=model(raw[ids].cuda(),torch.ones(len(ids),device='cuda',dtype=torch.long));scores=model.logit_scale.exp().clamp_max(100)*F.normalize(pred,dim=1)@F.normalize(targets[ids].cuda(),dim=1).T
                    loss,parts=torch_loss(scores,args.arm);assert torch.isfinite(loss) and float(parts['coarse_loss'])>=-1e-5
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();updates+=1;exposures+=len(ids);seen+=len(ids)
                    for key,value in dict(parts,loss=loss).items():sums[key]+=float(value.detach())*len(ids)
                assert np.all(coverage==1) and seen==13230
                last_order=hashlib.sha256(order.tobytes()).hexdigest();event=dict(epoch=epoch+1,arm=args.arm,updates=updates,exposures=exposures,concept_order_sha256=last_order,
                    full_T_coverage_min=int(coverage.min()),full_T_coverage_max=int(coverage.max()),training_means={k:v/seen for k,v in sums.items()},seconds=time.monotonic()-started,Dcheck_or_Dfit_used=False)
                with epoch_log.open('a') as stream:stream.write(json.dumps(event)+'\n');stream.flush();os.fsync(stream.fileno())
                print(json.dumps(event),flush=True)
            assert updates==830 and exposures==132300
            checkpoint=directory/'final.pt';model_cpu={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            opt_cpu=opt.state_dict()
            for state in opt_cpu['state'].values():
                for key,value in state.items():
                    if torch.is_tensor(value):state[key]=value.detach().cpu()
            save_torch(checkpoint,dict(model=model_cpu,optimizer=opt_cpu,CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state(),updates=updates,exposures=exposures,epoch=10,arm=args.arm,policy=K.ref(args.policy),initial_encoder=q['encoder'],last_concept_order_sha256=last_order))
            assert K.sha(q['encoder']['path'])==q['encoder']['sha256'];result=dict(status='COMPLETE_FULL_T10_NATIVE_ENCODER_TARGET',arm=args.arm,updates=updates,exposures=exposures,parameters=3200413,checkpoint=K.ref(checkpoint),epoch_log=K.ref(epoch_log),source_encoder_unchanged=True)
            K.write(directory/'completion.json',result)
    elif args.mode=='prepare':
        raw=encoder_inputs();old=np.load(K.checked(c['oldprep']['conditions']),allow_pickle=False);values=np.empty((400,5,1024),np.float32);values[:,:2]=old[:,[2,8]];models={}
        for j,arm in enumerate(C.MODELS,2):
            done=K.read(out/'fits'/arm/'completion.json');assert done['updates']==830 and done['exposures']==132300
            state=torch.load(K.checked(done['checkpoint']),map_location='cpu',weights_only=True,mmap=True);model=new_model(state['model']).eval().requires_grad_(False)
            with torch.inference_mode():
                for lo in range(0,400,128):
                    guard();ids=[r['array_row'] for r in c['queries'][lo:lo+128]];values[lo:lo+len(ids),j]=model(raw[ids].cuda(),torch.ones(len(ids),device='cuda',dtype=torch.long)).float().cpu().numpy()
            models[arm]=done;del state,model;torch.cuda.empty_cache()
        assert np.isfinite(values).all() and float(np.linalg.norm(values,axis=2).max())<10*K.read(out/'profile_completion.json')['T_visual_norm_max']
        K.save_array(out/'conditions400.npy',values);K.write(out/'queries400.json',c['queries'])
        result=dict(status='COMPLETE_ALL400_NATIVE_EEG_ONLY_CONDITIONS_LOCKED_PRE_IMAGES',policy=K.ref(args.policy),conditions=K.ref(out/'conditions400.npy'),queries=K.ref(out/'queries400.json'),models=models,
            target_metadata_ref=c['oldprep']['target_metadata_ref'],target_feature_ref=c['oldprep']['target_feature_ref'],baseline_features=c['cv']['q']['base_features'],normal_inference_only_EEG=True,
            concept_labels_and_true_images_NOT_inference_inputs=True,new_ordinary_used=False,original_HIVE_and_ATM_ordinary_unchanged=True)
        K.write(out/'prepared.json',result)
    elif args.mode=='generate':
        assert args.start in (0,80,160,240,320);assert K.read(out/'conditions_CPU_audit.json')['status']=='PASS_ALL1200_NATIVE_CPU_CONDITIONS'
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
                        'condition_FP32_sha256':tensor_sha(value),'condition_FP16_sha256':tensor_sha(value.half()),'tokens_FP16_sha256':tensor_sha(token),'normal_inference_only_EEG':True,'query_image_or_concept_read_at_inference':False,'UNet_NFE':4,'prior_NFE':0,'prepared':K.ref(out/'prepared.json')}
                    with (out/'images'/arm/'images.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                K.write(out/'generation_progress.json',dict(block_start=args.start,completed_queries_per_new_arm=i+1))
        finally:EA.randn_tensor=old_noise;pipe.prepare_latents=old_prepare
        assert tensor_digest(pipe.unet,half=True)==unet and tensor_digest(pipe.vae,half=True)==vae
        result=dict(status='COMPLETE_240_NEW_NATIVE_EEG_IMAGES_BLOCK',start=args.start,completed_queries_per_arm=args.start+80,new_images=240,all_frozen_host_weights_unchanged=True,loader=proof)
    else:
        assert args.stage in (80,400)
        import image_evaluation as E
        audit=K.read(out/f'image_audit{args.stage}.json');records=K.read(K.checked(audit['ordered_records']));base=np.load(K.checked(c['cv']['q']['base_features']),allow_pickle=False)
        model=E.dino_model();assert E.DINO_SHA256==q['DINO_weight_sha256'];features=np.empty((args.stage,5,768),np.float32);features[:,:2]=base[:args.stage,[2,8]];begin=0
        replay=E.dino_features([K.checked(c['baseline']['direct'][0]['image'])],model,batch_size=1,real_images=False).float().numpy()[0];error=float(abs(replay-base[0,2]).max());assert error<3e-6
        if args.stage==400:
            prefix=np.load(K.checked(K.read(out/'encode_80_completion.json')['features']),allow_pickle=False);features[:80]=prefix;begin=80
        new=[r for r in records if r['arm'] in C.MODELS and r['query_index']>=begin]
        assert len(new)==(args.stage-begin)*3
        for lo in range(0,len(new),32):
            guard();C.resources(c);rows=new[lo:lo+32];encoded=E.dino_features([K.checked(r['image']) for r in rows],model,batch_size=32,real_images=False).float().numpy()
            for r,v in zip(rows,encoded):features[r['query_index'],C.ARMS.index(r['arm'])]=v
        assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features,axis=2),1,atol=3e-5) and np.array_equal(features[:,:2],base[:args.stage,[2,8]])
        K.save_array(out/f'features{args.stage}.npy',features);result=dict(status='COMPLETE_FROZEN_FIVE_ARM_DINO_FEATURES',stage=args.stage,features=K.ref(out/f'features{args.stage}.npy'),new_images_encoded=len(new),old_baseline_features_reused=args.stage*2,weight_sha256=E.DINO_SHA256,native_single_replay_max_error=error)
    guard();result.update(source=K.ref(__file__),seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_after=C.resources(c))
    K.write(out/(suffix+'_completion.json'),result);print(json.dumps(result),flush=True)
except BaseException as exc:
    partial=None
    if args.mode=='fit' and 'model' in globals() and 'opt' in globals():
        try:
            partial_path=out/'fits'/args.arm/'failed_partial.pt';save_torch(partial_path,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),epoch=epoch,updates=updates,exposures=exposures,CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state()));partial=K.ref(partial_path)
        except BaseException as err:partial=dict(error=repr(err))
    K.write(out/(suffix+'_failure.json'),dict(status='FAILED_PREFIX_RETAINED_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc(),partial=partial));raise
