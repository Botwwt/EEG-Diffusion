"""Frozen input-EEG diagnosis after the sealed efficacy/secondary studies."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_paired_semantic_study_contract_CPU_20261030_v1 import atomic,checked,read,ref
from brainhive_paired_EEG_intervention_CPU_20261030_v1 import BASES,GENERATED_ARMS,ALL_ARMS,METRICS,CONTRASTS,mapping,source,sampling_spec,comparisons
def checksum(x):return hashlib.sha256(x.tobytes()).hexdigest()
def bindings(path):
    s=read(path);assert s['status']=='REGISTERED_CONDITIONAL_FROZEN_EEG_INPUT_INTERVENTION400'
    for p in s['source_pins']:checked(p)
    up=read(checked(s['upstream_protocol']));e=read(checked(s['upstream_image_execution']));d=read(checked(e['D09_completion']));inputs=read(checked(d['inputs']));q=read(checked(inputs['panel']))['Dcheck'];maps=read(checked(s['fixed_maps']));setup(s);assert maps==mapping(q)
    return s,up,e,d,q,maps,Path(s['output_root']),Path(e['output_root'])
def setup(s):
    sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C;C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
def ready(s,images):
    core=read(Path(s['upstream_output_root'])/'controller_completion.json');assert core['registration']==s['upstream_protocol'] and core['final_audit']['stage']==400 and core['final_audit']['decision']['action']=='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION'
    secondary=read(Path(s['secondary_output_root'])/'controller_completion.json');assert secondary['status']=='COMPLETE_CONDITIONAL_ALL7_FULL400_SECONDARY_EVALUATION' and secondary['protocol']==s['secondary_protocol']
    independent=read(checked(secondary['independent_audit']));assert independent['protocol']==s['secondary_protocol'] and independent['all2800_images_and_source_joins_passed']
    return independent
def prepare(path):
    s,up,e,d,q,maps,root,images=bindings(path);ready(s,images);setup(s)
    import numpy as np
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    cert=read(images/'independent_tokens_CPU_audit.json');assert cert['execution']==s['upstream_image_execution'] and cert['all7_inference_paths_passed'];export=read(checked(cert['export_completion']));assert export['execution']==s['upstream_image_execution']
    target=np.load(checked(d['check_target_features']),allow_pickle=False).astype(float);raw=np.load(images/'all_query_values400.npz',allow_pickle=False);models={};record_refs={}
    for base in BASES:
        model=export['models'][base];bank=np.load(checked(model['tokens']),allow_pickle=False);assert bank.shape==(400,4,2048) and bank.dtype==np.float16
        p=images/f'features400_{base}.npy';metrics=S.metric_arrays(np.load(p,allow_pickle=False).astype(float),target,q,q)
        assert all(np.max(np.abs(metrics[k]-raw[base+'__'+k]))<2e-12 for k in METRICS)
        refs=[]
        for i,query in enumerate(q):
            p0=images/'records'/f'{i:03d}_{base}.json';r=read(p0);assert all(r[k]==v for k,v in query.items()) and r['execution']==s['upstream_image_execution']
            assert r['source_model']==model['source'] and r['tokens_sha256']==checksum(bank[i:i+1]);assert ref(r['generated_path'])['sha256']==r['generated_sha256'];refs.append(ref(p0))
        record_refs[base]=refs;models[base]=dict(source=model['source'],tokens=model['tokens'],features=ref(images/f'features400_{base}.npy'))
    root.mkdir(exist_ok=True);atomic(root/'prepared.json',dict(status='COMPLETE_EXACT_FROZEN_MODEL_REFERENCE_AND_SOURCE_BANK_JOIN',protocol=ref(path),models=models,correct_records=record_refs,EEG=export['EEG'],target_features=d['check_target_features'],
        export_completion=cert['export_completion'],raw_core_metrics=ref(images/'all_query_values400.npz'),upstream_image_audit=ref(images/'independent_images400_CPU_audit.json'),secondary_independent_audit=ref(Path(s['secondary_output_root'])/'independent_secondary_CPU_audit.json'),research_images=0,new_models=0))

def gpu(args):
    s,up,e,d,q,maps,root,images=bindings(args.protocol);ready(s,images);setup(s)
    import numpy as np,torch
    from PIL import Image
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_text();assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==up['GPU_UUID']
    torch.cuda.set_per_process_memory_fraction(min(1.,(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    attempt=root/'attempts'/f'{args.mode}_{args.lo}_{args.hi}_{args.slot}';attempt.mkdir(parents=True,exist_ok=False);start=time.monotonic();memory=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
    def budget():
        rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
        assert time.monotonic()-start<820 and rss-memory<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(root).free>58<<30
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<s['output_cap_bytes']
    prepared=read(root/'prepared.json');assert prepared['protocol']==ref(args.protocol)
    try:
        if args.mode=='features':
            import image_evaluation as E
            checked(s['DINO_weight']);assert E.DINO_SHA256==s['DINO_weight']['sha256'];model=E.dino_model();saved={}
            for arm in GENERATED_ARMS:
                paths=[]
                for i in range(400):
                    r=read(root/'records'/f'{i:03d}_{arm}.json');assert r['query_id']==q[i]['query_id'] and ref(r['generated_path'])['sha256']==r['generated_sha256'];paths.append(r['generated_path'])
                values=[]
                for lo in range(0,400,32):budget();values.append(E.dino_features(paths[lo:lo+32],model,batch_size=32,real_images=False).numpy())
                p=root/f'features400_{arm}.npy';assert not p.exists();np.save(p,np.concatenate(values),allow_pickle=False);saved[arm]=ref(p)
            atomic(root/'features_completion.json',dict(status='COMPLETE_ALL1600_FROZEN_DINO_FEATURES',features=saved,protocol=ref(args.protocol)));atomic(attempt/'completion.json',dict(status='COMPLETE_FEATURE_TASK',seconds=time.monotonic()-start));return
        from brainhive_component_generator_v1 import load_native
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        sys.path.insert(0,up['source_root']);from run_HIVE_paired_semantic_images_20261030_v1 import sample_observed,save_image
        (attempt/'loader').mkdir();loader=ConvertedWeightAudit(torch,attempt/'loader');loader.install()
        try:pipe,A,P,prompts,host=load_native(training=False)
        finally:loader.restore()
        audit=loader.complete();banks={base:np.load(checked(prepared['models'][base]['tokens']),allow_pickle=False) for base in BASES};EEG=np.load(checked(prepared['EEG']),allow_pickle=False)
        def proof_keys(r):return {k:r[k] for k in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations')}
        if not (root/'engineering_replay.json').exists():
            replays=[]
            with torch.inference_mode():
                for base in BASES:
                    for i in (0,399):
                        reference=read(checked(prepared['correct_records'][base][i]));im,proof=sample_observed(pipe,prompts,torch.from_numpy(banks[base][i:i+1].copy()).cuda(),q[i]['image_seed'],torch)
                        assert checksum(np.asarray(im))==reference['generated_pixel_sha256'] and proof_keys(proof)==proof_keys(reference);replays.append(dict(base=base,query_index=i,pixel_sha256=reference['generated_pixel_sha256']))
            atomic(root/'engineering_replay.json',dict(status='COMPLETE_FOUR_NATIVE_CORRECT_REFERENCE_PIXEL_REPLAYS',replays=replays,research_images=0,UNet_forward_evaluations=16))
        (root/'records').mkdir(exist_ok=True);(root/'pixels').mkdir(exist_ok=True);new=0
        for i in range(args.lo,args.hi):
            for arm in GENERATED_ARMS:
                budget();base,j,seed=sampling_spec(arm,i,q,maps);tk=banks[base][j:j+1];query=q[i];reference=read(checked(prepared['correct_records'][base][i]));p=root/'records'/f'{i:03d}_{arm}.json'
                if p.exists():
                    r=read(p);assert r['protocol']==ref(args.protocol) and r['source_query_index']==j and r['query_id']==query['query_id'] and r['input_tokens_sha256']==checksum(tk);assert ref(r['generated_path'])['sha256']==r['generated_sha256'];continue
                if time.monotonic()-start>650:
                    atomic(attempt/'completion.json',dict(status='COMPLETE_PRESERVED_INTERVENTION_IMAGE_PREFIX',full_range=False,range=[args.lo,args.hi],new_images=new));return
                with torch.inference_mode():im,proof=sample_observed(pipe,prompts,torch.from_numpy(tk.copy()).cuda(),seed,torch)
                assert proof_keys(proof)==proof_keys(reference);saved=save_image(root/'pixels'/f'{i:03d}_{arm}.webp',im,np,Image)
                atomic(p,dict(**query,arm=arm,base_method=base,query_index=i,source_query_index=j,EEG_source_query_id=q[j]['query_id'],EEG_source_stimulus_id=q[j]['stimulus_id'],EEG_source_array_row=q[j]['array_row'],source_EEG_values_sha256=checksum(EEG[j:j+1]),
                    input_tokens_sha256=checksum(tk),source_model=prepared['models'][base]['source'],protocol=ref(args.protocol),sampling_target_image_access=False,concept_metadata_used_only_for_control_assignment=True,not_a_candidate_inference_method=True,**saved,**proof));new+=1
        atomic(attempt/'completion.json',dict(status='COMPLETE_FIXED_ALL4_INTERVENTION_IMAGE_RANGE',full_range=True,range=[args.lo,args.hi],new_images=new,loader=audit,host=host,seconds=time.monotonic()-start))
    except BaseException as ex:atomic(attempt/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise

def score(path):
    s,up,e,d,q,maps,root,images=bindings(path);secondary=ready(s,images);setup(s)
    import numpy as np
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    prepared=read(root/'prepared.json');completion=read(root/'features_completion.json');target=np.load(checked(prepared['target_features']),allow_pickle=False).astype(float);features={base:np.load(checked(prepared['models'][base]['features']),allow_pickle=False).astype(float) for base in BASES}
    features.update({arm:np.load(checked(completion['features'][arm]),allow_pickle=False).astype(float) for arm in GENERATED_ARMS});metrics={arm:S.metric_arrays(f,target,q,q) for arm,f in features.items()};contrasts,deltas,interpret,error=comparisons(metrics,q);arrays={a+'__'+k:v for a,m in metrics.items() for k,v in m.items()}
    arrays.update({label+'__contrast':v for label,v in deltas.items()});source_gaps={}
    for arm in GENERATED_ARMS:
        source_queries=[q[source(arm,i,maps)[1]] for i in range(400)];v=S.metric_arrays(features[arm],target,source_queries,q)
        arrays.update({arm+'__source__'+k:value for k,value in v.items()});arrays[arm+'__source_minus_query_cosine']=v['target_cosine']-metrics[arm]['target_cosine'];source_gaps[arm]=float(arrays[arm+'__source_minus_query_cosine'].mean())
    np.savez_compressed(root/'all2400_frozen_EEG_input_metrics.npz',**arrays)
    atomic(root/'readout.json',dict(status='COMPLETE_FIXED400_FROZEN_EEG_INPUT_DIAGNOSIS',protocol=ref(path),reference_images_reused=800,new_research_images=1600,metrics={a:{k:float(v.mean()) for k,v in m.items()} for a,m in metrics.items()},
        contrasts=contrasts,interpretation=interpret,source_minus_query_cosine_mean=source_gaps,weighted_bootstrap_maxerror=error,per_query_arrays=ref(root/'all2400_frozen_EEG_input_metrics.npz'),secondary_decision_preserved=secondary['next_action'],
        intervention_uses_concept_metadata_NOT_normal_inference=True,no_training_or_calibration=True,not_an_independent_test=True,SOTA=False))

def audit(path):
    s,up,e,d,q,maps,root,images=bindings(path);ready(s,images);setup(s);os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np,torch
    from PIL import Image,ImageDraw
    sys.path.insert(0,up['source_root']);from audit_HIVE_paired_semantic_study_CPU_20261030_v1 import metric_values
    torch.set_num_threads(2);prepared=read(root/'prepared.json');readout=read(root/'readout.json');arrays=np.load(checked(readout['per_query_arrays']),allow_pickle=False);features=read(root/'features_completion.json')['features'];target=np.load(checked(prepared['target_features']),allow_pickle=False);EEG=np.load(checked(prepared['EEG']),allow_pickle=False)
    grid=np.array(sorted(range(400),key=lambda i:(q[i]['concept_id'],q[i]['stimulus_id']))).reshape(40,10);within=np.empty(400,int);across=within.copy();within[grid.ravel()]=np.roll(grid,-1,axis=1).ravel();across[grid.ravel()]=np.roll(grid,-1,axis=0).ravel();assert np.array_equal(within,maps['within']) and np.array_equal(across,maps['across'])
    bank={b:np.load(checked(prepared['models'][b]['tokens']),allow_pickle=False) for b in BASES};inodes=set();pixels=[];maximum=0.
    for arm in ALL_ARMS:
        f=np.load(checked(prepared['models'][arm]['features'] if arm in BASES else features[arm]),allow_pickle=False);m=metric_values(f,target,q,q,np)
        for j,k in enumerate(METRICS):maximum=max(maximum,float(np.max(np.abs(m[:,j]-arrays[arm+'__'+k]))));assert maximum<2e-12
        if arm in GENERATED_ARMS:
            src_indices=within if arm.endswith('within') else across;source_queries=[q[int(j)] for j in src_indices];src=metric_values(f,target,source_queries,q,np)
            for j,k in enumerate(METRICS):assert np.max(np.abs(src[:,j]-arrays[arm+'__source__'+k]))<2e-12
            gap=src[:,0]-m[:,0];assert np.max(np.abs(gap-arrays[arm+'__source_minus_query_cosine']))<2e-12 and abs(float(gap.mean())-readout['source_minus_query_cosine_mean'][arm])<2e-12
        for i in range(400):
            if arm in BASES:r=read(checked(prepared['correct_records'][arm][i]))
            else:
                r=read(root/'records'/f'{i:03d}_{arm}.json');base='ordinary' if arm.startswith('ordinary') else 'joint_correct';j=int(within[i] if arm.endswith('within') else across[i]);assert r['source_query_index']==j and r['base_method']==base and r['EEG_source_query_id']==q[j]['query_id'] and r['EEG_source_stimulus_id']==q[j]['stimulus_id'] and r['EEG_source_array_row']==q[j]['array_row']
                assert r['source_EEG_values_sha256']==checksum(EEG[j:j+1]) and r['input_tokens_sha256']==checksum(bank[base][j:j+1]) and r['source_model']==prepared['models'][base]['source'];assert r['protocol']==ref(path) and not r['sampling_target_image_access'] and r['not_a_candidate_inference_method']
                reference=read(checked(prepared['correct_records'][base][i]))
                for k in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):assert r[k]==reference[k]
                assert r['UNet_NFE']==4 and r['native_IP_K_calls']==r['native_IP_V_calls']==280
            assert all(r[k]==v for k,v in q[i].items());p=Path(r['generated_path']);assert not p.is_symlink()
            if arm in GENERATED_ARMS:assert p.resolve().is_relative_to((root/'pixels').resolve())
            assert ref(p)['sha256']==r['generated_sha256'];st=p.stat();assert (st.st_dev,st.st_ino) not in inodes;inodes.add((st.st_dev,st.st_ino))
            with Image.open(p) as im:value=np.array(im.convert('RGB'));assert value.shape==(512,512,3) and checksum(value)==r['generated_pixel_sha256'];pixels.append(checksum(value))
    concepts=sorted({v['concept_id'] for v in q});groups=[[i for i,v in enumerate(q) if v['concept_id']==c] for c in concepts];draws=np.random.default_rng(1729).integers(0,40,(1000,40));stats_error=0.
    for label,coeff in CONTRASTS.items():
        delta=np.stack([sum(v*arrays[a+'__'+k] for a,v in coeff.items()) for k in METRICS],axis=1);assert np.max(np.abs(delta-arrays[label+'__contrast']))<2e-12;clusters=np.stack([delta[g].mean(0) for g in groups]);boot=np.array([sum(clusters[i] for i in d)/40 for d in draws]);ci=np.quantile(boot,[.025,.975],axis=0)
        for j,k in enumerate(METRICS):v=readout['contrasts'][label][k];err=max(abs(float(delta[:,j].mean())-v['difference']),float(np.max(np.abs(ci[:,j]-v['descriptive_concept_bootstrap95']))));stats_error=max(stats_error,err);assert err<2e-12
    atomic(root/'independent_CPU_audit.json',dict(status='COMPLETE_INDEPENDENT_ALL2400_FROZEN_INPUT_IDENTITY_NOISE_AND_STATISTICS_AUDIT',protocol=ref(path),readout=ref(root/'readout.json'),new_images=1600,reused_reference_images=800,all400_source_permutations_independently_reconstructed=True,
        maximum_metric_error=maximum,maximum_bootstrap_error=stats_error,pixel_duplicates_retained=len(pixels)-len(set(pixels)),interpretation=readout['interpretation'],secondary_decision_preserved=readout['secondary_decision_preserved'],no_training_or_calibration=True,CUDA_initialized=torch.cuda.is_initialized(),SOTA=False));assert not torch.cuda.is_initialized()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['prepare','generate','features','score','audit'],required=True);p.add_argument('--lo',type=int,default=0);p.add_argument('--hi',type=int,default=80);p.add_argument('--slot',type=int,default=0);a=p.parse_args()
    prepare(a.protocol) if a.mode=='prepare' else gpu(a) if a.mode in ('generate','features') else score(a.protocol) if a.mode=='score' else audit(a.protocol)
