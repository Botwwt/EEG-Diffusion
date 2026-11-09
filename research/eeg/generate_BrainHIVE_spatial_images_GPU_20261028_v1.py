"""D40 matched native images and fixed DINO feature extraction, no fitting."""
import argparse,datetime,gc,hashlib,json,os,time,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import brainhive_spatial_images_20261028_v1 as U

def bootstrap(args):
    start=time.monotonic();execution,base,queries,prepared=U.prepared(args.execution);U.stage_gate(args.execution,args.stage)
    train=U.read(U.checked(execution['training_spec']));eng=U.read(U.checked(train['engineering']));es=U.read(U.checked(eng['spec']))
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=es['runtime_manifest']))
    import torch
    from verify_HIVE_second_host_native_GPU_20261028_v1 import wrapper,memory
    job=wrapper(args.job_name);torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized();baseCPU=memory()['VmRSS']
    def budget():
        assert time.monotonic()-start<850 and memory()['VmHWM']-baseCPU<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        U.evaluation_budget()
    def stats():return dict(seconds=time.monotonic()-start,peak_extra_CPU_bytes=memory()['VmHWM']-baseCPU,peak_GPU_bytes=torch.cuda.max_memory_allocated())
    return execution,base,queries,prepared,eng,es,torch,job,budget,stats

def rows(count,queries,execution_path):
    out=[]
    for i,q in enumerate(queries[:count]):
        for arm in U.ARMS:
            p=U.EVAL/'images/records'/f'{i:03d}_{arm}.json';r=U.read(p)
            assert all(r[k]==q[k] for k in U.IDENTITY) and r['arm']==arm and r['query_index']==i
            assert r['execution']['sha256']==U.sha(execution_path) and U.sha(r['generated_path'])==r['generated_sha256']
            assert r['UNet_NFE']==4 and r['native_IP_K_calls']==r['native_IP_V_calls']==280
            out.append(r)
    return out

def digest(module):
    h=hashlib.sha256()
    for name,t in module.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def generate(args):
    execution,base,queries,prepared,eng,es,torch,job,budget,stats=bootstrap(args)
    assert 0<=args.query_start<args.query_end<=args.stage and args.query_end-args.query_start<=20
    if args.query_start:rows(args.query_start,queries,args.execution)
    attempt=U.EVAL/'attempts'/args.job_name;assert not attempt.exists();attempt.mkdir(parents=True)
    from PIL import Image
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
    observer=ConvertedWeightAudit(torch,attempt/'converted');(attempt/'converted').mkdir();observer.install()
    try:pipe,adapter,projector,prompts,host=load_native(training=False)
    finally:observer.restore()
    loading=observer.complete();assert host==eng['host']
    frozen={n:digest(m) for n,m in [('UNet',pipe.unet),('A',adapter),('P',projector)]}
    # Replay the existing T engineering image through the exact same GPU
    # batch-one encoder/ordinary path, before any Dcheck research generation.
    cpu=U.read(U.checked(es['CPU_completion']));cs=U.read(U.checked(cpu['spec']));d30=U.read(U.checked(cs['D30_spec']))
    from brainhive_reversible_eeg_20261028_v1 import author_brain_class
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    cls=author_brain_class(U.checked(d30['author_models_source']),U.checked(d30['author_layers_source']))
    parent=cls(4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False).cuda()
    parent.load_state_dict(torch.load(U.checked(base['parent_checkpoint']),map_location='cpu',weights_only=True)['model'],strict=True)
    ordinary=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False).cuda()
    fs=U.read(U.checked(es['ordinary_spec']));ordinary.load_state_dict(torch.load(U.checked(fs['checkpoint']),map_location='cpu',weights_only=True)['model'],strict=True)
    rawref=next(x for x in cpu['files'] if x['path'].endswith('/T13230_mean4.npy'));raw=np.load(U.checked(rawref),mmap_mode='r',allow_pickle=False)
    reference=eng['parity_images'][0]
    with torch.inference_mode():
        c=parent(torch.from_numpy(np.array(raw[:1],copy=True)).cuda());gamma=parent.block.layers[0].norm.weight[None];beta=parent.block.layers[0].norm.bias[None]
        o=ordinary(c,gamma,beta,gate_mode='fixed')[0];tokens=projector(adapter(o.half()))
        image,proof=sample_observed(pipe,prompts,tokens,reference['image_seed'],torch)
    with Image.open(reference['generated_path']) as im:pixels=np.asarray(im.convert('RGB')).copy()
    assert U.sha(reference['generated_path'])==reference['generated_sha256'] and np.array_equal(np.asarray(image),pixels)
    for k in ['initial_UNet_input_sha256','initial_noise_latents_sha256','ancestral_noise_observations']:assert proof[k]==reference[k]
    U.write(attempt/'T_parity.json',dict(status='D40_EXISTING_T_ENGINEERING_IMAGE_PIXEL_EXACT',**proof,**save_image(attempt/'T_parity.webp',image),reference=reference))
    del parent,ordinary,raw;gc.collect();torch.cuda.empty_cache();budget()
    bank=np.load(U.checked(prepared['conditions']),mmap_mode='r',allow_pickle=False);assert bank.shape==(400,8,1024)
    old=[json.loads(s) for s in U.checked(base['baseline_manifest']).read_text().splitlines()];noise={r['query_id']:r for r in old if r['arm']=='parent'}
    (U.EVAL/'images/records').mkdir(parents=True,exist_ok=True);(U.EVAL/'images/pixels').mkdir(exist_ok=True)
    for i in range(args.query_start,args.query_end):
        q=queries[i];reference=noise[q['query_id']];assert all(reference[k]==q[k] for k in U.IDENTITY)
        for a,arm in enumerate(U.ARMS):
            budget();record=U.EVAL/'images/records'/f'{i:03d}_{arm}.json';assert not record.exists()
            with torch.inference_mode():
                c=torch.tensor(np.array(bank[i:i+1,a],copy=True),device='cuda');tokens=projector(adapter(c.half()))
                image,proof=sample_observed(pipe,prompts,tokens,q['image_seed'],torch)
            for k in ['initial_UNet_input_sha256','initial_noise_latents_sha256','ancestral_noise_observations']:assert proof[k]==reference[k]
            encoder,bg=arm.split('__')
            saved=save_image(U.EVAL/'images/pixels'/f'{i:03d}_{arm}.webp',image)
            row=dict(**q,arm=arm,query_index=i,execution=U.ref(args.execution),prepared=U.ref(U.EVAL/'prepared/completion.json'),
              **proof,**saved,condition_sha256=U.byte_sha(bank[i,a]),condition_FP16_sha256=U.byte_sha(c.half().cpu().numpy()),tokens_sha256=U.byte_sha(tokens.cpu().numpy()),
              encoder_checkpoint=prepared['models'][encoder],ordinary_checkpoint=prepared['ordinary_checkpoint'] if bg=='ordinary' else None,
              target_index=i,target_feature_binding=base['target_features'],normal_inference_target_access=False,one_image_per_query=True)
            U.write(record,row)
        U.write(attempt/f'query_{i:03d}.json',dict(query_index=i,records=8,**stats()))
    after={n:digest(m) for n,m in [('UNet',pipe.unet),('A',adapter),('P',projector)]};assert after==frozen
    for p in execution['source_pins']:U.checked(p)
    budget();U.write(attempt/'completion.json',dict(status='D40_ALL_EIGHT_ARM_MATCHED_IMAGE_SEGMENT_CLOSED',execution=U.ref(args.execution),job=job,
      query_start=args.query_start,query_end=args.query_end,new_research_images=(args.query_end-args.query_start)*8,engineering_replay_images=1,
      loader=loading,host=host,all_frozen_generator_A_P_weights_unchanged=True,**stats()))
    print(json.dumps(dict(status='D40_ALL_EIGHT_ARM_MATCHED_IMAGE_SEGMENT_CLOSED',query_end=args.query_end,**stats())),flush=True)

def encode(args):
    execution,base,queries,prepared,eng,es,torch,job,budget,stats=bootstrap(args)
    audit=U.read(U.EVAL/f'screen{args.stage}/image_audit.json');assert audit['images']==args.stage*8
    assert audit['execution']['sha256']==U.sha(args.execution)
    records=U.read(U.checked(audit['rows']));assert len(records)==args.stage*8
    folder=U.EVAL/f'screen{args.stage}/features';assert not folder.exists();folder.mkdir(parents=True)
    import image_evaluation as E
    assert E.DINO_SHA256==base['DINO_weight']['sha256'] and U.sha(E.DINO_WEIGHT)==E.DINO_SHA256
    model=E.dino_model();budget()
    def extract(paths,real):
        pieces=[]
        for lo in range(0,len(paths),32):
            budget();pieces.append(E.dino_features(paths[lo:lo+32],model,batch_size=32,real_images=real).float().numpy())
        return np.concatenate(pieces)
    prefix=np.empty((0,8,768),np.float32)
    if args.stage==80:
        target_rows=U.read(U.checked(prepared['target_rows']));assert len(target_rows)==400
        paths=[]
        for i,(q,r) in enumerate(zip(queries,target_rows)):
            assert r['target_index']==i and all(r[k]==q[k] for k in U.IDENTITY);paths.append(U.checked(r['image']))
        target=extract(paths,True);cached=np.load(U.checked(base['target_features']),allow_pickle=False)
        assert target.shape==cached.shape==(400,768)
        difference=float(np.abs(target-cached).max());assert difference<2e-5
        U.array(folder/'targets400.npy',target);targetref=U.ref(folder/'targets400.npy')
    else:
        previous=U.read(U.EVAL/'screen80/features/completion.json');assert previous['execution']['sha256']==U.sha(args.execution)
        prefix=np.load(U.checked(previous['features']),allow_pickle=False);assert prefix.shape==(80,8,768)
        targetref=previous['fresh_targets'];U.checked(targetref);difference=previous['original_cache_vs_all400_fresh_target_max_error']
    fresh=records[len(prefix)*8:];values=extract([Path(r['generated_path']) for r in fresh],False).reshape(-1,8,768)
    result=np.concatenate([prefix,values]);assert result.shape==(args.stage,8,768) and result.dtype==np.float32 and np.isfinite(result).all()
    assert np.array_equal(result[:len(prefix)],prefix) and np.allclose(np.linalg.norm(result.astype(np.float64),axis=-1),1,atol=2e-5)
    U.array(folder/'features.npy',result);budget()
    v=dict(status='D40_ALL_EIGHT_ARM_GENERATED_AND_ALL400_REAL_TARGET_DINO_BOUND',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      execution=U.ref(args.execution),stage=args.stage,arms=list(U.ARMS),features=U.ref(folder/'features.npy'),image_audit=U.ref(U.EVAL/f'screen{args.stage}/image_audit.json'),
      fresh_targets=targetref,original_target_cache=base['target_features'],target_identity_rows=prepared['target_rows'],
      original_cache_vs_all400_fresh_target_max_error=difference,target_images_encoded=400 if args.stage==80 else 0,
      new_generated_images_encoded=len(fresh),reused80_images=len(prefix)*8,DINO_weight=U.ref(E.DINO_WEIGHT),image_evaluation=U.ref(E.__file__),job=job,**stats())
    U.write(folder/'completion.json',v);print(json.dumps(v),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True);p.add_argument('--mode',choices=['generate','encode'],required=True)
    p.add_argument('--stage',type=int,choices=[80,400],required=True);p.add_argument('--query-start',type=int,default=0);p.add_argument('--query-end',type=int,default=20)
    p.add_argument('--job-name',required=True);a=p.parse_args()
    try:generate(a) if a.mode=='generate' else encode(a)
    except BaseException as e:
        U.write(U.EVAL/'failures'/(a.job_name+'.json'),dict(status='D40_IMAGE_JOB_FAILED_ALL_PREFIX_ARTIFACTS_RETAINED',error=repr(e),traceback=traceback.format_exc()))
        raise
