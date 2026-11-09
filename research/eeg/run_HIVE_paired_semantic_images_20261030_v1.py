"""Changed EEG/P inference, fresh matched baselines, fixed native image screen."""
import argparse,datetime,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_paired_semantic_study_contract_CPU_20261030_v1 import ARMS,IMAGE_ARMS,atomic,checked,read,ref,image_decision
def byte_sha(x):return hashlib.sha256(x.tobytes()).hexdigest()
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024

def bindings(path):
    e=read(path);assert e['status']=='REGISTERED_EEG_P_PAIRED_SEMANTIC_NATIVE_IMAGE_EXECUTION_V1'
    s=read(checked(e['training_protocol']))
    for pin in e['source_pins']:checked(pin)
    d=read(checked(e['D09_completion']));inputs=read(checked(d['inputs']));panel=read(checked(inputs['panel']));q=panel['Dcheck']
    assert len(q)==400 and len({v['query_id'] for v in q})==400
    assert all(v['subject_id']=='sub-01' and v['partition']=='Dcheck' for v in q)
    return e,s,d,inputs,q

def sample_observed(pipe,prompts,tokens,seed,torch):
    import inspect
    counts=[0,0,0];initial={};module=inspect.getmodule(type(pipe.scheduler));old_randn=module.randn_tensor;old_prepare=pipe.prepare_latents;noise=[];hooks=[]
    def randn(*args,**kwargs):
        value=old_randn(*args,**kwargs);noise.append(dict(sha256=byte_sha(value.detach().cpu().contiguous().numpy()),shape=list(value.shape),dtype=str(value.dtype)));return value
    def prepare(*args,**kwargs):
        value=old_prepare(*args,**kwargs);initial['initial_noise_latents_sha256']=byte_sha(value.detach().cpu().contiguous().numpy());return value
    def forward(m,args,kwargs):
        assert kwargs['cross_attention_kwargs']['ip_hidden_states'].data_ptr()==tokens.data_ptr()
        counts[0]+=1
        if counts[0]==1:initial['initial_UNet_input_sha256']=byte_sha(args[0].detach().cpu().contiguous().numpy())
    def key(m,args):assert args[0].data_ptr()==tokens.data_ptr() and args[0].shape==(1,4,2048);counts[1]+=1
    def value(m,args):assert args[0].data_ptr()==tokens.data_ptr() and args[0].shape==(1,4,2048);counts[2]+=1
    hooks=[pipe.unet.register_forward_pre_hook(forward,with_kwargs=True)]
    processors=[v for v in pipe.unet.attn_processors.values() if type(v).__name__=='IPAttnProcessor2_0'];assert len(processors)==70
    for p in processors:hooks.extend([p.to_k_ip.register_forward_pre_hook(key),p.to_v_ip.register_forward_pre_hook(value)])
    module.randn_tensor=randn;pipe.prepare_latents=prepare
    try:
        image=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(seed),prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],
            cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,guidance_scale=0.,output_type='pil').images[0].convert('RGB')
    finally:
        module.randn_tensor=old_randn;pipe.prepare_latents=old_prepare
        for h in hooks:h.remove()
    assert counts==[4,280,280] and len(noise)==4
    return image,dict(**initial,ancestral_noise_observations=noise,UNet_NFE=4,native_IP_K_calls=280,native_IP_V_calls=280,scheduler_step_signature_unmodified=True)

def save_image(path,image,np,Image):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_name(path.name+'.pending');assert not path.exists() and not pending.exists()
    image.save(pending,format='WEBP',lossless=True,quality=90,method=4);assert pending.stat().st_size<=1<<20
    with Image.open(pending) as decoded:assert np.array_equal(np.asarray(image),np.asarray(decoded.convert('RGB')))
    with pending.open('rb') as f:os.fsync(f.fileno())
    pending.rename(path);return dict(generated_path=str(path),generated_sha256=ref(path)['sha256'],generated_pixel_sha256=byte_sha(np.asarray(image)))

def gpu(args):
    e,s,d,inputs,queries=bindings(args.execution);sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C;C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0,str(Path(args.execution).parent))
    import numpy as np,torch
    from PIL import Image
    import brainhive_native_models_v1 as N
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(e['output_root']);root.mkdir(exist_ok=True);attempt=root/f'{args.mode}_{args.stage}_{args.lo:03d}_{args.hi:03d}_{args.slot}'
    assert not attempt.exists();attempt.mkdir();start=time.monotonic();memory=rss()
    torch.cuda.set_per_process_memory_fraction(min(1.,s['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory))
    def budget():assert time.monotonic()-start<820 and rss()-memory<s['maximum_extra_CPU_bytes'] and torch.cuda.max_memory_allocated()<s['maximum_GPU_bytes'] and shutil.disk_usage(root).free>s['reserved_legacy_full1654_bytes']+s['disk_buffer_bytes']
    try:
        if args.mode=='features':
            import image_evaluation as E
            assert E.DINO_SHA256==e['DINO_weight']['sha256'];checked(e['DINO_weight']);model=E.dino_model();saved={}
            for arm in IMAGE_ARMS:
                rows=[read(root/'records'/f'{i:03d}_{arm}.json') for i in range(args.stage)]
                paths=[]
                for i,row in enumerate(rows):
                    assert row['query_id']==queries[i]['query_id'] and row['arm']==arm and row['execution']==ref(args.execution)
                    checked(dict(path=row['generated_path'],bytes=Path(row['generated_path']).stat().st_size,sha256=row['generated_sha256']))
                    with Image.open(row['generated_path']) as image:assert byte_sha(np.asarray(image.convert('RGB')))==row['generated_pixel_sha256']
                    paths.append(row['generated_path'])
                values=[];begin=80 if args.stage==400 else 0
                for lo in range(begin,args.stage,32):budget();values.append(E.dino_features(paths[lo:lo+32],model,batch_size=32,real_images=False).float().numpy())
                values=np.concatenate(values)
                if args.stage==400:values=np.concatenate([np.load(root/f'features80_{arm}.npy',allow_pickle=False),values])
                assert values.shape==(args.stage,768) and np.isfinite(values).all()
                p=root/f'features{args.stage}_{arm}.npy';assert not p.exists();np.save(p,values,allow_pickle=False);saved[arm]=ref(p)
            atomic(attempt/'completion.json',dict(status='COMPLETE_ALL7_FIXED_DINO_FEATURES',stage=args.stage,features=saved,seconds=time.monotonic()-start));return
        if args.mode=='export':
            certificate=read(e['fit_CPU_audit_path']);assert certificate['all5_fixed_finals_verified'] and certificate['registration']==ref(e['training_protocol']['path'])
            raw=torch.load(checked(e['raw_EEG']),weights_only=True,map_location='cpu',mmap=True);channels=read(checked(s['input_bindings']))['selected_indices']
            x=raw[[int(q['array_row']) for q in queries]][:,channels,:].numpy().copy();assert x.shape==(400,17,250);np.save(root/'inference_EEG400.npy',x,allow_pickle=False)
            original=torch.load(checked(s['parent']),weights_only=True,map_location='cpu')['model'];model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval().requires_grad_(False)
            prior=N.prior_models('cpu');A=prior['fusion_encoder'].adapter.cuda().half().eval().requires_grad_(False);P=prior['proj'].cuda().half().eval().requires_grad_(False)
            ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False);ordinary.load_state_dict(torch.load(checked(s['ordinary']),weights_only=True,map_location='cpu')['model'],strict=True)
            models={}
            for arm in IMAGE_ARMS:
                budget()
                if arm in ARMS:
                    completed=read(Path(s['output_root'])/arm/'completion.json');state=torch.load(checked(completed['model']),map_location='cpu',weights_only=False);model.load_state_dict(state['EEG'],strict=True)
                    P.load_state_dict({n:p.half() for n,p in state['P'].items()},strict=True);source=completed['model']
                else:
                    model.load_state_dict(original,strict=True);initial=N.prior_models('cpu')['proj'];P.load_state_dict({n:p.half() for n,p in initial.state_dict().items()},strict=True);source=s['parent']
                LN=next(v for v in model.modules() if isinstance(v,torch.nn.LayerNorm));raws=[];conditions=[];tokens=[]
                with torch.inference_mode():
                    for i in range(400):
                        c=model(torch.from_numpy(x[i:i+1].copy()).cuda());w=LN.weight[None];b=LN.bias[None];result=c if arm=='parent' else ordinary(c,w,b)[0]
                        tk=P(A(result.half()));raws.append(c.cpu().numpy());conditions.append(result.cpu().numpy());tokens.append(tk.cpu().numpy())
                for kind,values in [('raw_conditions',raws),('conditions',conditions),('tokens',tokens)]:
                    p=root/f'{kind}400_{arm}.npy';assert not p.exists();np.save(p,np.concatenate(values),allow_pickle=False)
                models[arm]=dict(source=source,raw=ref(root/f'raw_conditions400_{arm}.npy'),conditions=ref(root/f'conditions400_{arm}.npy'),tokens=ref(root/f'tokens400_{arm}.npy'))
            atomic(root/'export_completion.json',dict(status='COMPLETE_ALL7_EEG_AND_CHANGED_P_TOKENS400_EXPORT',models=models,EEG=ref(root/'inference_EEG400.npy'),execution=ref(args.execution),raw_EEG_source=e['raw_EEG'],normal_inference_target_access=False,new_images=0,seconds=time.monotonic()-start));return
        assert args.mode=='generate'
        certificate=read(e['tokens_CPU_audit_path']);assert certificate['all7_inference_paths_passed'] and certificate['execution']==ref(args.execution)
        assert certificate['export_completion']==ref(root/'export_completion.json')
        exported=read(root/'export_completion.json');banks={a:np.load(checked(exported['models'][a]['tokens']),allow_pickle=False) for a in IMAGE_ARMS}
        (attempt/'loader').mkdir();loader=ConvertedWeightAudit(torch,attempt/'loader');loader.install()
        try:pipe,A,P,prompts,host=load_native(training=False)
        finally:loader.restore()
        loading=loader.complete();(root/'records').mkdir(exist_ok=True);(root/'pixels').mkdir(exist_ok=True)
        count=0
        for i in range(args.lo,args.hi):
            q=queries[i];proof0=None
            for arm in IMAGE_ARMS:
                budget();record=root/'records'/f'{i:03d}_{arm}.json'
                if record.exists():
                    row=read(record);assert row['query_id']==q['query_id'] and row['arm']==arm and row['execution']==ref(args.execution)
                    assert row['tokens_sha256']==byte_sha(banks[arm][i:i+1]) and ref(row['generated_path'])['sha256']==row['generated_sha256']
                    with Image.open(row['generated_path']) as image:assert byte_sha(np.asarray(image.convert('RGB')))==row['generated_pixel_sha256']
                    proof={k:row[k] for k in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations')}
                else:
                    if time.monotonic()-start>650:
                        atomic(attempt/'completion.json',dict(status='COMPLETE_PRESERVED_IMAGE_PREFIX',full_range=False,range=[args.lo,args.hi],new_images=count,seconds=time.monotonic()-start));return
                    tk=torch.from_numpy(banks[arm][i:i+1].copy()).cuda()
                    with torch.inference_mode():image,proof=sample_observed(pipe,prompts,tk,q['image_seed'],torch)
                    saved=save_image(root/'pixels'/f'{i:03d}_{arm}.webp',image,np,Image)
                    atomic(record,dict(**q,query_index=i,arm=arm,execution=ref(args.execution),source_model=exported['models'][arm]['source'],
                        tokens_sha256=byte_sha(banks[arm][i:i+1]),condition_sha256=byte_sha(np.load(exported['models'][arm]['conditions']['path'],allow_pickle=False,mmap_mode='r')[i:i+1]),
                        normal_inference_target_access=False,one_image_per_query=True,**saved,**proof));count+=1
                    proof={k:proof[k] for k in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations')}
                if proof0 is None:proof0=proof
                else:assert proof0==proof
        atomic(attempt/'completion.json',dict(status='COMPLETE_ALL7_NATIVE_IMAGE_RANGE',full_range=True,range=[args.lo,args.hi],images=(args.hi-args.lo)*7,new_images=count,loader=loading,host=host,seconds=time.monotonic()-start))
    except BaseException as ex:atomic(attempt/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise

def score(args):
    e,s,d,inputs,q=bindings(args.execution);sys.path.insert(0,s['repository_root']);import numpy as np
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    root=Path(e['output_root']);targets=np.load(checked(d['check_target_features']),allow_pickle=False).astype(np.float64)
    metrics={a:S.metric_arrays(np.load(root/f'features{args.stage}_{a}.npy',allow_pickle=False).astype(np.float64),targets,q[:args.stage],q) for a in IMAGE_ARMS}
    contrasts={a:{b:S.compare(metrics[a],metrics[b],q[:args.stage]) for b in IMAGE_ARMS if b!=a} for a in ARMS}
    raw={a+'__'+k:v for a,m in metrics.items() for k,v in m.items()}
    raw.update({a+'_minus_'+b+'__'+k:metrics[a][k]-metrics[b][k] for a in ARMS for b in IMAGE_ARMS if a!=b for k in metrics[a]})
    np.savez_compressed(root/f'all_query_values{args.stage}.npz',**raw)
    result=dict(status='COMPLETE_FIXED_PAIRED_EEG_P_NATIVE_IMAGE_READOUT',stage=args.stage,queries=args.stage,gallery_targets=400,
        means={a:{k:float(v.mean()) for k,v in m.items()} for a,m in metrics.items()},contrasts=contrasts,decision=image_decision(contrasts,args.stage),
        matched_parent_and_ordinary_regenerated=True,independent_confirmation=False,SOTA=False,execution=ref(args.execution))
    atomic(root/f'score{args.stage}.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',required=True);p.add_argument('--mode',choices=['export','generate','features','score'],required=True);p.add_argument('--stage',type=int,choices=[80,400],default=80);p.add_argument('--lo',type=int,default=0);p.add_argument('--hi',type=int,default=80);p.add_argument('--slot',type=int,default=0);a=p.parse_args();score(a) if a.mode=='score' else gpu(a)
