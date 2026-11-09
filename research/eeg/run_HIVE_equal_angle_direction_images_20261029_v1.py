"""Equal-angle direction interventions on the existing fixed D09 panel."""
import argparse,datetime,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from prepare_HIVE_direction_interventions_CPU_20261029_v1 import ref
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending')
    with tmp.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
ARMS=('population_direction','varying_direction')
def read(p):return json.loads(Path(p).read_bytes())
def checked(x):
    p=Path(x['path']);assert p.is_file() and ref(p)['sha256']==x['sha256'];return p
def byte_sha(x):return hashlib.sha256(x.tobytes()).hexdigest()
def bindings(path):
    s=read(path);assert s['status']=='BOUND_ORDINARY_EQUAL_ANGLE_DIRECTION_IMAGE_EXECUTION'
    for x in s['source_pins']:checked(x)
    training=read(checked(s['scientific_protocol']));d09=read(checked(s['D09_completion']));inputs=read(checked(d09['inputs']));panel=read(checked(inputs['panel']));q=panel['Dcheck'];assert len(q)==400
    assert len({v['query_id'] for v in q})==400 and all(v['subject_id']=='sub-01' and v['partition']=='Dcheck' for v in q)
    return s,training,d09,inputs,q
def gpu(args):
    s,t,d09,inputs,queries=bindings(args.execution);sys.path.insert(0,t['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=t['runtime_manifest']))
    import brainhive_native_models_v1 as N
    import torch,numpy as np
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==t['GPU_UUID']
    root=Path(s['output_root']);root.mkdir(exist_ok=True);started=time.monotonic();base=rss()
    attempt=root/f'{args.mode}_{args.lo:03d}_{args.hi:03d}_{args.stage}';assert not attempt.exists();attempt.mkdir()
    def budget():assert time.monotonic()-started<800 and rss()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(root).free>2<<30
    try:
        if args.mode=='features':
            import image_evaluation as E
            assert E.DINO_SHA256==s['DINO_weight']['sha256'];checked(s['DINO_weight']);model=E.dino_model();budget()
            result={}
            for arm in args.arms:
                rows=[read(root/'records'/f'{i:03d}_{arm}.json') for i in range(args.stage)]
                paths=[]
                from PIL import Image
                for i,r in enumerate(rows):
                    assert r['query_id']==queries[i]['query_id'] and r['arm']==arm and r['execution']==ref(args.execution)
                    p=Path(r['generated_path']);assert ref(p)['sha256']==r['generated_sha256']
                    with Image.open(p) as im:assert im.size==(512,512) and byte_sha(np.asarray(im.convert('RGB')))==r['generated_pixel_sha256']
                    paths.append(p)
                pieces=[]
                for lo in range(0,len(paths),32):budget();pieces.append(E.dino_features(paths[lo:lo+32],model,batch_size=32,real_images=False).float().numpy())
                values=np.concatenate(pieces);assert values.shape==(args.stage,768) and np.isfinite(values).all()
                previous=root/f'features80_{arm}.npy'
                if args.stage==400:
                    old=np.load(previous,allow_pickle=False);assert np.max(np.abs(values[:80]-old))<2e-5
                    values[:80]=old  # frozen first80 features; no image regenerated
                dest=root/f'features{args.stage}_{arm}.npy';assert not dest.exists();np.save(dest,values,allow_pickle=False);result[arm]=ref(dest)
            atomic(attempt/'completion.json',dict(status='COMPLETE_EQUAL_ANGLE_DIRECTION_DINO_FEATURES',stage=args.stage,arms=args.arms,features=result,execution=ref(args.execution),seconds=time.monotonic()-started,new_model_updates=0));return
        assert args.mode=='generate' and 0<=args.lo<args.hi<=400 and args.hi-args.lo<=160
        from brainhive_component_generator_v1 import load_native
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
        def digest(module):
            h=hashlib.sha256()
            for name,value in module.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        bank=np.load(checked(d09['check_conditions']),allow_pickle=False);assert bank.shape==(400,3,1024) and bank.dtype==np.float32
        prepared=read(checked(s['prepared_completion']));assert prepared['status']=='COMPLETE_ALL400_EQUAL_ANGLE_DIRECTION_CONDITIONS_CPU_VERIFIED'
        assert prepared['protocol']==s['scientific_protocol'] and prepared['arms']==list(ARMS)
        candidate=np.load(checked(prepared['conditions']),allow_pickle=False);assert candidate.shape==(400,2,1024) and candidate.dtype==np.float32
        conditions={arm:candidate[:,ARMS.index(arm)] for arm in args.arms}
        corrections={arm:dict(ordinary_checkpoint=t['ordinary'],T_population_mean=t['T_mean_direction'],direction_intervention=arm,angle_matched_to_original_ordinary=True) for arm in args.arms}
        gc.collect();torch.cuda.empty_cache();(attempt/'converted').mkdir()
        audit=ConvertedWeightAudit(torch,attempt/'converted');audit.install()
        try:pipe,A,P,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader=audit.complete();assert host==inputs['expected_host'];modules=dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P);before={k:digest(v) for k,v in modules.items()}
        oldrows=[json.loads(x) for x in checked(d09['check_manifest']).read_text().splitlines()];assert len(oldrows)==1200
        refs={(r['query_id'],r['arm']):r for r in oldrows};from PIL import Image
        noise_keys=('initial_UNet_input_sha256','initial_noise_latents_sha256','ancestral_noise_observations')
        for j,arm in enumerate(('parent','ordinary')):
            r=refs[(queries[0]['query_id'],arm)];c=torch.from_numpy(bank[:1,j].copy()).cuda()
            with torch.inference_mode():image,proof=sample_observed(pipe,prompts,P(A(c.half())),r['image_seed'],torch)
            with Image.open(r['generated_path']) as im:assert np.array_equal(np.asarray(image),np.asarray(im.convert('RGB')))
            assert ref(r['generated_path'])['sha256']==r['generated_sha256'];assert all(proof[k]==r[k] for k in noise_keys)
            atomic(attempt/(arm+'_parity.json'),dict(pixel_exact=True,reference=r,observed=proof,**save_image(attempt/(arm+'_parity.webp'),image)))
        (root/'records').mkdir(exist_ok=True);(root/'pixels').mkdir(exist_ok=True)
        for i in range(args.lo,args.hi):
            q=queries[i];reference=refs[(q['query_id'],'parent')]
            for arm in args.arms:
                budget();record=root/'records'/f'{i:03d}_{arm}.json';assert not record.exists();c=torch.from_numpy(conditions[arm][i:i+1].copy()).cuda()
                with torch.inference_mode():tokens=P(A(c.half()));image,proof=sample_observed(pipe,prompts,tokens,q['image_seed'],torch)
                assert all(proof[k]==reference[k] for k in noise_keys)
                saved=save_image(root/'pixels'/f'{i:03d}_{arm}.webp',image)
                atomic(record,dict(**q,query_index=i,arm=arm,execution=ref(args.execution),condition_sha256=byte_sha(conditions[arm][i:i+1]),condition_FP16_sha256=byte_sha(c.half().cpu().numpy()),tokens_sha256=byte_sha(tokens.cpu().numpy()),normal_inference_target_access=False,one_image_per_query=True,**corrections[arm],**saved,**proof))
            if i%10==9:atomic(attempt/'progress.json',dict(query=i+1,arms=args.arms,seconds=time.monotonic()-started))
        assert before=={k:digest(v) for k,v in modules.items()};checked(t['ordinary']);budget()
        atomic(attempt/'completion.json',dict(status='COMPLETE_EQUAL_ANGLE_DIRECTION_NATIVE_IMAGE_SEGMENT',range=[args.lo,args.hi],arms=args.arms,images=(args.hi-args.lo)*len(args.arms),loader=loader,host=host,generator_frozen=True,seconds=time.monotonic()-started,extra_CPU_peak=rss()-base,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
    except BaseException as e:atomic(attempt/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
def score(args):
    s,t,d09,inputs,queries=bindings(args.execution);sys.path.insert(0,t['repository_root']);import numpy as np
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    root=Path(s['output_root']);targets=np.load(checked(d09['check_target_features']),allow_pickle=False).astype(np.float64);base=np.load(checked(d09['check_generated_features']),allow_pickle=False).astype(np.float64)
    assert targets.shape==(400,768) and base.shape==(400,3,768)
    metrics={a:S.metric_arrays(base[:args.stage,j],targets,queries[:args.stage],queries) for j,a in enumerate(('parent','ordinary','risk'))}
    for arm in args.arms:
        features=np.load(root/f'features{args.stage}_{arm}.npy',allow_pickle=False);metrics[arm]=S.metric_arrays(features.astype(np.float64),targets,queries[:args.stage],queries)
    contrasts={a:{b:S.compare(metrics[a],metrics[b],queries[:args.stage]) for b in ('parent','ordinary','risk')} for a in args.arms}
    decisions={a:S.decision(contrasts[a]['ordinary'],str(args.stage)) for a in args.arms}
    if len(args.arms)==2:contrasts['population_minus_varying']=S.compare(metrics[ARMS[0]],metrics[ARMS[1]],queries[:args.stage])
    result=dict(status='COMPLETE_EQUAL_ANGLE_DIRECTION_PAIRED_IMAGE_READOUT',stage=args.stage,queries=args.stage,gallery_targets=400,arms=args.arms,means={a:{k:float(v.mean()) for k,v in m.items()} for a,m in metrics.items()},contrasts=contrasts,decisions=decisions,mechanism_controls_not_new_method_claim=True,EEG_free_controls=False,independent_confirmation=False,SOTA=False,execution=ref(args.execution))
    atomic(root/f'score{args.stage}.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',required=True);p.add_argument('--mode',choices=['generate','features','score'],required=True);p.add_argument('--stage',type=int,choices=[80,400],required=True);p.add_argument('--lo',type=int,default=0);p.add_argument('--hi',type=int,default=80);p.add_argument('--arms',nargs='+',choices=list(ARMS),required=True);a=p.parse_args();score(a) if a.mode=='score' else gpu(a)
