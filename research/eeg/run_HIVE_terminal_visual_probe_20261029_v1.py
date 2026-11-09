"""Privileged visual last-step diagnostic, not a normal EEG method.

Native four-step sampler stays intact. Only fourth UNet IP tokens are swapped.
All80 ordinary replays must match existing pixels and all actual RNG draws.
"""
import argparse,datetime,gc,hashlib,inspect,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ARMS=('correct_visual_last','sameconcept_visual_last')
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):assert ref(x['path'])['sha256']==x['sha256'];return Path(x['path'])
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def native_switch(pipe,prompts,base,last,seed,torch):
    calls=[0,0,0];observed=[];noise=[];states={};expected=[base];handles=[];original_prepare=pipe.prepare_latents
    step_method=pipe.scheduler.step;step_module=inspect.getmodule(type(pipe.scheduler));randn=step_module.randn_tensor
    def capture_noise(*a,**kw):
        x=randn(*a,**kw);noise.append(dict(sha256=tensor_sha(x),shape=list(x.shape),dtype=str(x.dtype)));return x
    def capture_initial(*a,**kw):
        x=original_prepare(*a,**kw);states['initial_sha']=tensor_sha(x);return x
    def unet_pre(module,args,kwargs):
        i=len(observed);assert i<4 and len(args)>=2;tokens=base if i<3 else last;expected[0]=tokens
        assert kwargs['cross_attention_kwargs']['ip_hidden_states'].data_ptr()==base.data_ptr()
        observed.append(dict(step=i,timestep=float(args[1]),UNet_input_sha256=tensor_sha(args[0]),IP_tokens_sha256=tensor_sha(tokens)))
        if i==3:
            new=dict(kwargs);cross=dict(kwargs['cross_attention_kwargs']);cross['ip_hidden_states']=tokens;new['cross_attention_kwargs']=cross;return args,new
    def unet_post(module,args,result):
        calls[0]+=1;assert torch.isfinite(result[0]).all()
        if calls[0]==4:states['terminal_epsilon']=result[0].detach().cpu().clone()
    def key(module,args):assert args[0].data_ptr()==expected[0].data_ptr() and args[0].shape==(1,4,2048);calls[1]+=1
    def value(module,args):assert args[0].data_ptr()==expected[0].data_ptr() and args[0].shape==(1,4,2048);calls[2]+=1
    def callback(p,i,t,values):
        if i in (2,3):states['prefix' if i==2 else 'terminal']=values['latents'].detach().cpu().clone()
        return values
    assert 'callback_on_step_end' in inspect.signature(pipe.__call__).parameters
    handles=[pipe.unet.register_forward_pre_hook(unet_pre,with_kwargs=True),pipe.unet.register_forward_hook(unet_post)]
    processors=[p for p in pipe.unet.attn_processors.values() if type(p).__name__=='IPAttnProcessor2_0'];assert len(processors)==70
    for p in processors:handles.extend([p.to_k_ip.register_forward_pre_hook(key),p.to_v_ip.register_forward_pre_hook(value)])
    pipe.prepare_latents=capture_initial;step_module.randn_tensor=capture_noise
    try:
        image=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(seed),prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],cross_attention_kwargs={'ip_hidden_states':base},num_inference_steps=4,guidance_scale=0.,output_type='pil',callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents']).images[0].convert('RGB')
    finally:
        for h in handles:h.remove()
        pipe.prepare_latents=original_prepare;step_module.randn_tensor=randn
    assert pipe.scheduler.step==step_method and calls==[4,280,280] and len(noise)==4 and len(observed)==4
    assert [r['timestep'] for r in observed]==[999.,749.,499.,249.] and float(pipe.scheduler.sigmas[-1])==0
    proof=dict(initial_noise_latents_sha256=states['initial_sha'],initial_UNet_input_sha256=observed[0]['UNet_input_sha256'],ancestral_noise_observations=noise,UNet_NFE=4,native_IP_K_calls=280,native_IP_V_calls=280,all_step_observations=observed,terminal_sigma=float(pipe.scheduler.sigmas[-2]),scheduler_step_signature_unmodified=True)
    return image,proof,{k:v for k,v in states.items() if isinstance(v,torch.Tensor)}

def run(path,mode):
    s=read(path);assert s['status']=='REGISTERED_NATIVE_ORDINARY_LAST_STEP_PRIVILEGED_VISUAL_DIAGNOSTIC' and s['host']=='old'
    for x in s['source_pins']:checked(x)
    sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']));import brainhive_native_models_v1 as N
    import torch,numpy as np
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(s['output_root']);root.mkdir(exist_ok=True);start=time.monotonic();memory=rss()
    assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    def budget():assert time.monotonic()-start<800 and rss()-memory<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(root).free>2<<30 and sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<1<<30
    d09=read(checked(s['D09_completion']));inputs=read(checked(d09['inputs']));queries=read(checked(inputs['panel']))['Dcheck'];assert len(queries)==400
    attempt=root/mode;assert not attempt.exists();attempt.mkdir()
    try:
        if mode=='features':
            import image_evaluation as E
            assert E.DINO_SHA256==s['DINO_weight']['sha256'];checked(s['DINO_weight']);model=E.dino_model();budget();features={}
            from PIL import Image
            for arm in ARMS:
                rows=[read(root/'records'/f'{i:03d}_{arm}.json') for i in range(80)];paths=[]
                for i,r in enumerate(rows):
                    assert r['query_id']==queries[i]['query_id'] and r['privileged_query_image_F_used'] and r['normal_EEG_method_result'] is False
                    p=Path(r['generated_path']);assert ref(p)['sha256']==r['generated_sha256']
                    with Image.open(p) as image:assert hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()==r['generated_pixel_sha256']
                    paths.append(p)
                pieces=[]
                for lo in range(0,80,32):budget();pieces.append(E.dino_features(paths[lo:lo+32],model,batch_size=32,real_images=False).float().numpy())
                result=np.concatenate(pieces);assert result.shape==(80,768);dest=root/f'features80_{arm}.npy';np.save(dest,result,allow_pickle=False);features[arm]=ref(dest)
            write(attempt/'completion.json',dict(status='COMPLETE_PRIVILEGED_LAST_STEP_VISUAL_DINO_FEATURES',features=features,seconds=time.monotonic()-start,protocol=ref(path)));return
        assert mode=='generate';from brainhive_component_generator_v1 import load_native
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import save_image
        from PIL import Image
        bank=np.load(checked(d09['check_conditions']),allow_pickle=False);assert bank.shape==(400,3,1024)
        target=torch.load(checked(s['F_target_cache']),weights_only=True,map_location='cpu',mmap=True);assert target.shape==(16540,1024)
        old=[json.loads(x) for x in checked(d09['check_manifest']).read_text().splitlines()];refs={(r['query_id'],r['arm']):r for r in old}
        (attempt/'converted').mkdir();audit=ConvertedWeightAudit(torch,attempt/'converted');audit.install()
        try:pipe,A,P,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader=audit.complete();assert host==inputs['expected_host']
        def digest(module):
            h=hashlib.sha256()
            for n,v in module.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        modules=dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P);frozen={k:digest(v) for k,v in modules.items()}
        for name in ('records','pixels','ordinary_replays','terminal_tensors'):(root/name).mkdir()
        for i,q in enumerate(queries[:80]):
            budget();donor=s['same_concept_donor_indices'][i];assert donor!=i and queries[donor]['concept_id']==q['concept_id']
            reference=refs[(q['query_id'],'ordinary')]
            with torch.inference_mode():
                base=P(A(torch.from_numpy(bank[i:i+1,1].copy()).cuda().half()))
                visual={ARMS[0]:P(A(target[q['array_row']:q['array_row']+1].cuda().half())),ARMS[1]:P(A(target[queries[donor]['array_row']:queries[donor]['array_row']+1].cuda().half()))}
                image,baseline,states=native_switch(pipe,prompts,base,base,q['image_seed'],torch)
            with Image.open(reference['generated_path']) as old_image:assert np.array_equal(np.asarray(image),np.asarray(old_image.convert('RGB')))
            assert ref(reference['generated_path'])['sha256']==reference['generated_sha256']
            for k in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):assert baseline[k]==reference[k]
            write(root/'ordinary_replays'/f'{i:03d}.json',dict(**q,reference=reference,proof=baseline,pixel_exact=True))
            tensors={'ordinary_prefix':states['prefix'].numpy(),'ordinary_terminal':states['terminal'].numpy(),'ordinary_epsilon':states['terminal_epsilon'].numpy()}
            for arm in ARMS:
                budget()
                with torch.inference_mode():image,proof,tstate=native_switch(pipe,prompts,base,visual[arm],q['image_seed'],torch)
                for k in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):assert proof[k]==baseline[k]
                assert [r['UNet_input_sha256'] for r in proof['all_step_observations']]==[r['UNet_input_sha256'] for r in baseline['all_step_observations']]
                assert [r['IP_tokens_sha256'] for r in proof['all_step_observations'][:3]]==[r['IP_tokens_sha256'] for r in baseline['all_step_observations'][:3]]
                assert torch.equal(tstate['prefix'],states['prefix'])
                tensors[arm+'_prefix']=tstate['prefix'].numpy();tensors[arm+'_terminal']=tstate['terminal'].numpy();tensors[arm+'_epsilon']=tstate['terminal_epsilon'].numpy()
                saved=save_image(root/'pixels'/f'{i:03d}_{arm}.webp',image)
                write(root/'records'/f'{i:03d}_{arm}.json',dict(**q,query_index=i,arm=arm,protocol=ref(path),last_visual_source_query=queries[i if arm==ARMS[0] else donor],F_target_cache=s['F_target_cache'],last_F_condition_sha256=tensor_sha(target[q['array_row']:q['array_row']+1] if arm==ARMS[0] else target[queries[donor]['array_row']:queries[donor]['array_row']+1]),privileged_query_image_F_used=True,normal_EEG_method_result=False,training_updates=0,**proof,**saved))
            np.savez_compressed(root/'terminal_tensors'/f'{i:03d}.npz',**tensors)
            if (i+1)%10==0:print(json.dumps(dict(stage='PRIVILEGED_LAST_STEP_DIAGNOSTIC',queries=i+1,research_images=2*(i+1),ordinary_replays=i+1,seconds=time.monotonic()-start)),flush=True)
        assert frozen=={k:digest(v) for k,v in modules.items()};checked(s['ordinary']);budget()
        write(attempt/'completion.json',dict(status='COMPLETE_ALL80_LAST_STEP_PRIVILEGED_VISUAL_IMAGES_AND_ORDINARY_REPLAYS',protocol=ref(path),queries=80,research_images=160,ordinary_replay_images=80,total_UNet_calls=960,all80_ordinary_pixels_exact=True,all80x3_first4_UNet_inputs_same=True,loader=loader,host=host,all_weights_frozen=True,seconds=time.monotonic()-start,extra_CPU_peak=rss()-memory,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
    except BaseException as e:write(attempt/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise

def score(path):
    s=read(path);sys.path.insert(0,s['repository_root']);import numpy as np
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    root=Path(s['output_root']);d=read(checked(s['D09_completion']));inputs=read(checked(d['inputs']));queries=read(checked(inputs['panel']))['Dcheck'];targets=np.load(checked(d['check_target_features']),allow_pickle=False).astype(np.float64);base=np.load(checked(d['check_generated_features']),allow_pickle=False)
    metrics={'ordinary':S.metric_arrays(base[:80,1].astype(np.float64),targets,queries[:80],queries)}
    for arm in ARMS:metrics[arm]=S.metric_arrays(np.load(root/f'features80_{arm}.npy',allow_pickle=False).astype(np.float64),targets,queries[:80],queries)
    contrasts=dict(correct_minus_ordinary=S.compare(metrics[ARMS[0]],metrics['ordinary'],queries[:80]),sameconcept_minus_ordinary=S.compare(metrics[ARMS[1]],metrics['ordinary'],queries[:80]),correct_minus_sameconcept=S.compare(metrics[ARMS[0]],metrics[ARMS[1]],queries[:80]))
    a=contrasts['correct_minus_ordinary'];b=contrasts['correct_minus_sameconcept'];support=a['DINO2AFC']['difference']>=.02 and a['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and a['within_concept10_top1']['difference']>=0 and b['within_concept10_top1']['difference']>0 and b['within_concept10_top1']['descriptive_concept_bootstrap95'][0]>0
    np.savez_compressed(root/'per_query_metrics.npz',**{arm+'__'+k:v for arm,m in metrics.items() for k,v in m.items()})
    result=dict(status='COMPLETE_FIXED80_LAST_STEP_VISUAL_REFERENCE_READOUT',queries=80,gallery_targets=400,means={arm:{k:float(v.mean()) for k,v in m.items()} for arm,m in metrics.items()},contrasts=contrasts,predeclared_interface_support=support,teacher_uses_query_image=True,EEG_method_gain=False,new_training_admitted=False,SOTA=False,protocol=ref(path));write(root/'score80.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['generate','features','score'],required=True);a=p.parse_args();score(a.protocol) if a.mode=='score' else run(a.protocol,a.mode)
