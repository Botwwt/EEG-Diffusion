"""Finite native63 reference fit/export/render. No author architecture changes."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,hashlib,io,json,resource,subprocess,sys,time,traceback,types
from pathlib import Path
import numpy as np
import enigma_native_reference_contract_20261104_v1 as C
from enigma_native_reference_math_20261104_v1 import author_namespace,torch_loss,numpy_loss_gradient
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('profile','fit','prepare','generate','encode'),required=True)
ap.add_argument('--role',default='');ap.add_argument('--end-epoch',type=int,default=0);ap.add_argument('--start',type=int,default=0)
args=ap.parse_args();c=C.setup(args.policy);K=c['K'];q=c['q'];out=c['out'];parent=c['parent']
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;before=C.resources(c,starting=True)
assert K.ref(__file__)==q['sources']['GPU']
suffix=args.mode+('_'+args.role+'_'+str(args.end_epoch) if args.mode=='fit' else '_'+str(args.start) if args.mode=='generate' else '')
job=K.read(c['repo']/'audit'/(q['job_prefix']+'_'+suffix+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==parent['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
sys.path[:0]=q['existing_runtime_paths']
import torch
import torch.nn.functional as F
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import tensor_sha
from brainhive_native_direction_contract_20261103_v1 import tensor_digest
scope=author_namespace(K.checked(q['original_models']).read_text(),K.checked(q['original_training']).read_text())
def guard():
    assert time.monotonic()-started<850
    used=sum(p.stat().st_blocks*512 for p in Path(q['project_root']).rglob('*') if p.is_file())
    assert used<q['maximum_output_bytes']
    stat=os.statvfs(q['project_root']);assert stat.f_bavail*stat.f_frsize>=before['required_remaining_bytes']+q['maximum_output_bytes']-used
    assert torch.cuda.max_memory_allocated()<parent['maximum_GPU_bytes']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<parent['maximum_extra_CPU_bytes']
def save_torch(path,state):
    temporary=path.with_name(path.name+'.pending');assert not path.exists() and not temporary.exists()
    torch.save(state,temporary);temporary.replace(path)
def model():
    value=scope['ENIGMA'](63,250,['sub-01']).cuda()
    assert sum(p.numel() for p in value.parameters())==1405242
    return value
def data():
    raw=torch.load(K.checked(q['EEG']),map_location='cpu',weights_only=True,mmap=True)
    images=torch.load(K.checked(q['visual_target']),map_location='cpu',weights_only=True,mmap=True)
    texts=torch.load(K.checked(q['text_target']),map_location='cpu',weights_only=True,mmap=True)
    assert raw.shape==(16540,63,250) and images.shape==(16540,1024) and texts.shape==(1654,1024)
    return raw,images,texts
def opt_scheduler(net,role):
    fit,_=c['roles'][role];batches=(len(fit)+511)//512
    return scope['prepare_optimizer'](net,3e-4,150,range(batches),.1,1000)
def owned_training_state(pin):
    path=K.checked(pin)
    assert path.is_relative_to(Path(q['project_root'])) and path.parent.name in C.ROLES
    assert path.name in ('epoch_50.pt','epoch_100.pt','epoch_150.pt')
    # Sealed receipts bind these checkpoints to this thread's own trainer.
    # PyTorch2.3 OneCycleLR includes a callable; this is not an external weight load.
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    allowed=[K.ref(args.policy)]
    if 'resume_origin_policy' in q:allowed.append(q['resume_origin_policy'])
    assert state['policy'] in allowed
    return state
def cpu_state(net,opt,scheduler,epoch,role,updates,exposures,log):
    state=opt.state_dict()
    for v in state['state'].values():
        for k,x in v.items():
            if torch.is_tensor(x):v[k]=x.detach().cpu()
    return dict(model={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},optimizer=state,
        scheduler=scheduler.state_dict(),epoch=epoch,role=role,updates=updates,exposures=exposures,
        CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state(),
        fit_rows_SHA256=hashlib.sha256(c['roles'][role][0].tobytes()).hexdigest(),
        epoch_order_rule='default_rng(1729+epoch_zero_based).permutation(number_of_fit_rows)',
        epoch_log=K.ref(log),policy=K.ref(args.policy),scratch_initialization_seed=1729)
try:
    result={}
    if args.mode=='profile':
        raw,images,texts=data();torch.manual_seed(1729);torch.cuda.manual_seed(1729)
        net=model().train();initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};opt,sched=opt_scheduler(net,'full')
        ids=c['T'][:512];x=raw[ids].cuda();y=images[ids].cuda();t=texts[ids//10].cuda()
        CPU_rng=torch.get_rng_state();CUDA_rng=torch.cuda.get_rng_state();tick=time.monotonic()
        pred=net(x,['sub-01']*len(ids));loss,parts=torch_loss(pred,y,t)
        reference=numpy_loss_gradient(pred.detach().cpu().numpy(),y.cpu().numpy(),t.cpu().numpy())
        derivative=torch.autograd.grad(loss,pred,retain_graph=True)[0]
        error=float(abs(derivative.detach().cpu().numpy()-reference['prediction_gradient']).max())
        assert abs(float(loss.detach())-reference['loss'])<5e-5 and error<4e-6
        values={k:float(v.detach()) for k,v in parts.items()}
        # Execute the author's actual loop, rather than requiring bitwise Adam parity
        # from a mathematically equivalent reordered loss graph near BN bias zeros.
        del net,opt,sched,pred,loss,parts,derivative;torch.cuda.empty_cache()
        batch=types.SimpleNamespace(eeg=x,class_id=torch.tensor(ids//10),img_vector=y,
            img_vector_norm=F.normalize(y,dim=-1),txt_vector_norm=F.normalize(t,dim=-1),subject=['sub-01']*len(ids))
        net=model().train();net.load_state_dict(initial,strict=True);opt,sched=opt_scheduler(net,'full')
        torch.set_rng_state(CPU_rng);torch.cuda.set_rng_state(CUDA_rng)
        scope['train_epoch'](net,[batch],opt,sched,'cuda',1.,.5,.05,False)
        torch.cuda.synchronize();step_seconds=time.monotonic()-tick
        reference_schedule=sched.state_dict();after={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        del net,opt,sched;torch.cuda.empty_cache()
        net2=model().train();net2.load_state_dict(initial,strict=True);opt2,sched2=opt_scheduler(net2,'full')
        torch.set_rng_state(CPU_rng);torch.cuda.set_rng_state(CUDA_rng)
        scope['train_epoch'](net2,[batch],opt2,sched2,'cuda',1.,.5,.05,False)
        parity=max(float((v-net2.state_dict()[k].detach().cpu()).abs().max()) for k,v in after.items())
        K.write(out/'profile_parity_numbers.json',dict(author_state_max_error=parity,
            schedule_equal=sched2.state_dict()==reference_schedule,
            largest_state_errors=sorted([(k,float((v-net2.state_dict()[k].detach().cpu()).abs().max())) for k,v in after.items()],key=lambda v:v[1],reverse=True)[:10],
            raw_loss_numpy_difference=abs(float(reference['loss'])-(values['raw_MSE']+.5*values['image_CE']+.05*values['text_CE'])),
            prediction_gradient_numpy_max_error=error))
        assert parity<2e-6 and sched2.state_dict()==reference_schedule
        net2.eval().requires_grad_(False)
        with torch.inference_mode():
            actual=net2(x[:8],['sub-01']*8)
            explicit=net2.mlp_proj(net2.tsencoder(net2.subject_wise_linear['sub-01'](x[:8])))
        assert torch.equal(actual,explicit)
        # This conservative profile includes NumPy proof time and is not a fit-time promise.
        result=dict(status='PASS_NATIVE63_REAL512_AUTHOR_TRAIN_STEP_AND_RAW_LOSS_GRADIENT',
            parameters=1405242,real_profile_batch=512,parts=values,prediction_gradient_max_error=error,
            author_train_epoch_one_update_state_max_error=parity,actual_training_executes_original_author_loop=True,
            original_vs_explicit_forward_max_error=0.,
            whole_proof_step_seconds=step_seconds,temporary_updates=2,temporary_models_discarded=True,
            loss_weights=dict(MSE=1.,image_CE=.5,text_CE=.05),raw_EEG_logits=True,logit_multiplier=1.)
    elif args.mode=='fit':
        assert args.role in C.ROLES and args.end_epoch in (50,100,150)
        assert K.read(out/'profile_completion.json')['status']=='PASS_NATIVE63_REAL512_AUTHOR_TRAIN_STEP_AND_RAW_LOSS_GRADIENT'
        raw,images,texts=data();fit,_=c['roles'][args.role];batch_count=(len(fit)+511)//512
        folder=out/'fits'/args.role;folder.mkdir(parents=True,exist_ok=True);log=folder/('epochs_'+str(args.end_epoch)+'.jsonl')
        assert not log.exists()
        torch.manual_seed(1729);torch.cuda.manual_seed(1729);net=model().train();opt,sched=opt_scheduler(net,args.role)
        begin=args.end_epoch-50;updates=exposures=0
        if begin:
            prev=K.read(folder/('segment_'+str(begin)+'_completion.json'))
            state=owned_training_state(prev['checkpoint'])
            assert state['epoch']==begin and state['role']==args.role
            assert state['fit_rows_SHA256']==hashlib.sha256(fit.tobytes()).hexdigest()
            net.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
            torch.set_rng_state(state['CPU_rng']);torch.cuda.set_rng_state(state['CUDA_rng'])
            updates,exposures=state['updates'],state['exposures'];K.checked(state['epoch_log']);del state
        assert updates==begin*batch_count and exposures==begin*len(fit)
        image_unit=F.normalize(images,dim=-1);text_unit=F.normalize(texts,dim=-1)
        for epoch in range(begin,args.end_epoch):
            order=np.random.default_rng(1729+epoch).permutation(len(fit));coverage=np.zeros(len(fit),np.int8)
            sums=dict(raw_MSE=0.,image_CE=0.,text_CE=0.,loss=0.,output_norm=0.)
            count=dict(seen=0,clipped=0,batches=0);current={}
            def batches():
                for lo in range(0,len(order),512):
                    guard();positions=order[lo:lo+512];ids=fit[positions];coverage[positions]+=1
                    current['ids']=ids
                    yield types.SimpleNamespace(eeg=raw[ids],class_id=torch.tensor(ids//10),img_vector=images[ids],
                        img_vector_norm=image_unit[ids],txt_vector_norm=text_unit[ids//10],subject=['sub-01']*len(ids))
            def observe_forward(module,arguments,pred):
                ids=current['ids'];n=len(ids)
                with torch.no_grad():
                    y=images[ids].cuda();yu=image_unit[ids].cuda();tu=text_unit[ids//10].cuda()
                    mse=F.mse_loss(pred.detach(),y)
                    ce=scope['ClipLoss']()(pred.detach(),yu,1)
                    tce=scope['ClipLoss']()(pred.detach(),tu,1)
                    parts=dict(raw_MSE=float(mse),image_CE=float(ce),text_CE=float(tce),
                        loss=float(mse+.5*ce+.05*tce),output_norm=float(pred.detach().norm(dim=-1).mean()))
                assert all(np.isfinite(v) for v in parts.values())
                for key,value in parts.items():sums[key]+=value*n
                count['seen']+=n;count['batches']+=1
            clip_original=torch.nn.utils.clip_grad_norm_
            def observe_clip(*values,**kw):
                norm=clip_original(*values,**kw);assert torch.isfinite(norm)
                count['clipped']+=int(float(norm)>1.);return norm
            hook=net.register_forward_hook(observe_forward);torch.nn.utils.clip_grad_norm_=observe_clip
            try:scope['train_epoch'](net,batches(),opt,sched,'cuda',1.,.5,.05,False)
            finally:hook.remove();torch.nn.utils.clip_grad_norm_=clip_original
            seen=count['seen'];clipped=count['clipped'];updates+=count['batches'];exposures+=seen
            assert count['batches']==batch_count
            assert np.all(coverage==1) and seen==len(fit)
            event=dict(role=args.role,epoch=epoch+1,updates=updates,exposures=exposures,
                fit_rows=len(fit),sample_order_SHA256=hashlib.sha256(order.tobytes()).hexdigest(),coverage_min=int(coverage.min()),coverage_max=int(coverage.max()),
                training_means={k:v/seen for k,v in sums.items()},gradient_clipped_batches=clipped,
                lr=[g['lr'] for g in opt.param_groups],scheduler_last_epoch=sched.last_epoch,
                seconds=time.monotonic()-started,heldout_concepts_or_D_used=False)
            with log.open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps(event),flush=True)
        assert updates==args.end_epoch*batch_count and exposures==args.end_epoch*len(fit)
        checkpoint=folder/('epoch_'+str(args.end_epoch)+'.pt');save_torch(checkpoint,cpu_state(net,opt,sched,args.end_epoch,args.role,updates,exposures,log))
        result=dict(status='COMPLETE_NATIVE63_REFERENCE_50_EPOCH_SEGMENT',role=args.role,epoch=args.end_epoch,updates=updates,exposures=exposures,
            checkpoint=K.ref(checkpoint),epoch_log=K.ref(log),parameters=1405242,initialization='SCRATCH_SEED1729_NO_FULL_TO_OOF_TRANSFER',
            role_train_rows=len(fit),role_train_concepts=len(fit)//10,no_D_or_held_concept_model_selection=True)
        K.write(folder/('segment_'+str(args.end_epoch)+'_completion.json'),result)
    elif args.mode=='prepare':
        raw,_,_=data();base=np.load(K.checked(q['baseline_conditions']),allow_pickle=False)
        assert base.shape==(400,6,1024);values=np.empty((400,7,1024),np.float32);values[:,:6]=base
        bank=np.empty((13230,1024),np.float32);D=np.empty((1650,1024),np.float32);models={};row_position={int(v):i for i,v in enumerate(c['T'])}
        for role in C.ROLES:
            done=K.read(out/'fits'/role/'segment_150_completion.json');state=owned_training_state(done['checkpoint'])
            assert state['epoch']==150 and state['role']==role
            net=model().eval().requires_grad_(False);net.load_state_dict(state['model'],strict=True)
            ids=c['D'] if role=='full' else c['roles'][role][1];predictions=[]
            with torch.inference_mode():
                for lo in range(0,len(ids),128):
                    guard();rows=ids[lo:lo+128];predictions.append(net(raw[rows].cuda(),['sub-01']*len(rows)).float().cpu().numpy())
            predictions=np.concatenate(predictions)
            if role=='full':
                D[:]=predictions;index={int(v):i for i,v in enumerate(ids)}
                values[:,6]=predictions[[index[r['array_row']] for r in c['queries']]]
            else:bank[[row_position[int(i)] for i in ids]]=predictions
            models[role]=done;del net,state;torch.cuda.empty_cache()
        assert np.isfinite(values).all() and np.isfinite(bank).all() and np.isfinite(D).all()
        K.save_array(out/'conditions400.npy',values);K.save_array(out/'T13230_encoder_OOF_conditions.npy',bank);K.save_array(out/'D1650_full_conditions.npy',D)
        K.write(out/'queries400.json',c['queries']);prepared0=K.read(K.checked(q['baseline_prepared']))
        result=dict(status='COMPLETE_NATIVE63_ENCODER_OOF_AND_FULL_D_CONDITIONS',policy=K.ref(args.policy),models=models,
            conditions=K.ref(out/'conditions400.npy'),OOF_conditions=K.ref(out/'T13230_encoder_OOF_conditions.npy'),D_conditions=K.ref(out/'D1650_full_conditions.npy'),
            queries=K.ref(out/'queries400.json'),target_metadata_ref=prepared0['target_metadata_ref'],target_feature_ref=prepared0['target_feature_ref'],
            exact_encoder_training_concept_exclusion=True,preexisting_preprocessing_MVNN_independent_replay=False,whole_pipeline_OOF_NOT_established=True,
            normal_inference_only_EEG=True,query_image_or_concept_used_at_inference=False,ordinary_used=False,original_ordinary_unchanged=True)
        K.write(out/'prepared.json',result)
    elif args.mode=='generate':
        assert args.start in (0,80,160,240,320)
        assert K.read(out/'conditions_CPU_audit.json')['status']=='PASS_NATIVE63_COMPLETE_OOF_AND_D_CPU_REPLAY'
        from atm_native_training_runtime_20261103_v1 import load_host
        prepared=K.read(out/'prepared.json');values=np.load(K.checked(prepared['conditions']),allow_pickle=False)
        loader=out/('loader_'+str(args.start));loader.mkdir()
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
    guard();result.update(source=K.ref(__file__),seconds=time.monotonic()-started,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
        resources_after=C.resources(c))
    K.write(out/(suffix+'_completion.json'),result);print(json.dumps(result),flush=True)
except BaseException as exc:
    partial=None
    if args.mode=='fit' and all(v in globals() for v in ('net','opt','sched','log')) and log.exists():
        try:
            path=out/'fits'/args.role/('failed_before_'+str(args.end_epoch)+'.pt')
            save_torch(path,cpu_state(net,opt,sched,epoch,args.role,updates,exposures,log));partial=K.ref(path)
        except BaseException as error:partial=dict(error=repr(error))
    K.write(out/(suffix+'_failure.json'),dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',error=repr(exc),
        traceback=traceback.format_exc(),partial=partial));raise
