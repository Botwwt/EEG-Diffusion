"""Four-arm actual FP32 training engineering, all temporary states discarded."""
import argparse,datetime,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):
    r=ref(x['path']);assert r['sha256']==x['sha256']
    if 'bytes' in x:assert r['bytes']==x['bytes']
    return Path(x['path'])
def write(p,x):Path(p).write_bytes((json.dumps(x,indent=2,allow_nan=False)+'\n').encode())
def run(args):
    study=read(args.protocol);assert study['status']=='ENGINEERING_ONLY_SINGLE1729_OLD_COMMON_DESCENT_PROFILE' and study['host']=='old'
    for item in study['source_pins']:checked(item)
    s=read(checked(study['relative_study']));R=Path(s['repository_root']);code=Path(args.protocol).parent;out=Path(study['output_root']);out.mkdir(exist_ok=False)
    assert read(code/'CPU_review.json')['status']=='PASS_ACTUAL_COMMON_DIRECTION_MATH_AND_FIXED_INPUT_CPU_REVIEW'
    assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==study['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    sys.path.insert(0,str(R));import brainhive_paired_study_contract_20261025_v1 as C;C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']));sys.path.insert(0,str(code));sys.path.insert(0,str(Path(study['relative_study']['path']).parent))
    import torch,numpy as np
    import torch.nn.functional as F
    import brainhive_native_models_v1 as N
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_data_v1 import load_training
    from brainhive_conditional_relative_objective_20261030_v1 import relative_evidence_loss
    from brainhive_normalized_common_descent_20261030_v1 import ARMS,direction,interpolate_state
    from train_HIVE_relative_continuation_GPU_20261030_v1 import bank
    from train_BrainHIVE_bounded_component_v1 import seed
    from brainhive_functional_training_contract_v2 import resource_headroom
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    assert resource_headroom()['effective_headroom_bytes']>=10<<30;started=time.monotonic();phase='LOAD';base=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
    def budget():
        rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
        assert time.monotonic()-started<260 and rss-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(out).free>2<<30
    def digest(m):
        h=hashlib.sha256()
        for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    try:
        data=load_training();assert data['input_contract_sha256']==s['input_contract']['sha256'];contract=read(checked(s['input_contract']));banks,bank_hashes=bank(data,torch,np,contract,R);unit=F.normalize(data['target'].float(),dim=-1)
        prior=N.prior_models('cuda');A=prior['fusion_encoder'].adapter.float().eval().requires_grad_(False);P=prior['proj'].float().eval().requires_grad_(False);del prior;gc.collect();frozen=dict(A=digest(A),P=digest(P))
        with torch.no_grad():target_tokens=torch.cat([P(A(data['target'][i:i+256].float())) for i in range(0,13230,256)])
        old=torch.load(checked(s['ordinary']),weights_only=True,map_location='cpu')['model'];initial_norm=float(torch.sqrt(sum(v.double().square().sum() for v in old.values())));step=1e-4*initial_norm
        def fresh():
            m=BoundedConditionAdapter().cuda().eval();m.load_state_dict(old,strict=True);return m
        def gradients(m,ids,arm):
            ids=ids.cuda();v=ids%13230;aid=data['affine_ids'][ids];changed=m(data['condition'][ids],data['weight'][aid],data['bias'][aid])[0]
            slots=banks['mismatched_slot' if arm=='joint_mismatched_teacher' else 'true_slot'][v];gallery=unit[banks['same'][v]]
            rel=relative_evidence_loss(changed,gallery,slots,banks['temperatures'][aid]).mean();AP=.1*(1-F.cosine_similarity(P(A(changed)),target_tokens[v],dim=-1)).mean()
            params=list(m.parameters());gr=torch.cat([x.reshape(-1) for x in torch.autograd.grad(rel,params,retain_graph=True)]);ga=torch.cat([x.reshape(-1) for x in torch.autograd.grad(AP,params)])
            mode='relative' if arm=='relative_only' else 'AP' if arm=='AP_only' else 'joint';d,info=direction(gr,ga,mode)
            assert torch.isfinite(d).all() and torch.isfinite(rel) and torch.isfinite(AP)
            return gr,ga,d,dict(relative_loss=float(rel),AP_loss=float(AP),relative_norm=float(info['relative_norm']),AP_norm=float(info['AP_norm']),
                direction_norm=float(d.norm()),relative_derivative=float(info['relative_derivative']),AP_derivative=float(info['AP_derivative']))
        witness=np.load(checked(study['feasibility_readouts']),allow_pickle=False);reference=np.load(checked(study['feasibility_parameter_gradients']),allow_pickle=False);m=fresh();initial_hash=digest(m)
        gr,ga,d,cell=gradients(m,torch.from_numpy(witness['indices']),'joint_same');match={}
        for name,x in [('relative',gr),('AP',ga),('direction',d)]:
            got=x.detach().cpu().numpy().astype(np.float64);want=reference[name];relative=float(np.linalg.norm(got-want)/np.linalg.norm(want));scaled=float(np.max(abs(got-want)/(1+abs(want))));assert relative<2e-3 and scaled<2e-5
            match[name]=dict(relative_error=relative,scaled_error=scaled)
        del m,gr,ga,d;budget();arms={};order=torch.randperm(132300,generator=torch.Generator().manual_seed(seed('pass0')));order_sha=hashlib.sha256(order.numpy().tobytes()).hexdigest();assert order_sha==read(checked(s['starting_F_provenance']))['orders_sha256'][0]
        for arm in ARMS:
            phase='FOUR_ARMS_ALL20_ROLES_AND_32_DISCARDED_STEPS_'+arm;m=fresh();assert digest(m)==initial_hash;cells=[]
            for role in range(20):
                ids=torch.nonzero(data['affine_ids']==role).flatten()[:16].cpu();gr,ga,d,record=gradients(m,ids,arm);assert record['relative_norm']>0 and record['AP_norm']>0;cells.append(dict(role=role,**record));budget()
            times=[];temporary=[]
            for j in range(32):
                torch.cuda.synchronize();tick=time.monotonic();gr,ga,d,record=gradients(m,order[j*16:(j+1)*16],arm);offset=0
                with torch.no_grad():
                    for p in m.parameters():n=p.numel();p.add_(d[offset:offset+n].reshape_as(p),alpha=step);offset+=n
                assert offset==525568;torch.cuda.synchronize();times.append(time.monotonic()-tick);temporary.append(record);budget()
            raw={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};zero=interpolate_state(old,raw,0.);one=interpolate_state(old,raw,1.);assert all(torch.equal(old[k],zero[k]) and torch.equal(raw[k],one[k]) for k in old)
            # Parameter interpolation is distinct from radius scaling: radius stays .25.
            assert m.radius==.25;arms[arm]=dict(all20_role_gradients=cells,temporary_steps=32,temporary_step_records=temporary,
                median_step_seconds=float(np.median(times)),p95_step_seconds=float(np.quantile(times,.95)),estimated_24807_steps_p95_seconds=float(np.quantile(times,.95)*24807),
                original_and_final_fraction_state_exact=True,temporary_weights_discarded=True)
            del m,gr,ga,d,raw,zero,one;gc.collect()
        assert frozen==dict(A=digest(A),P=digest(P));checked(s['ordinary']);budget()
        result=dict(status='PASS_ACTUAL_FOUR_ARM_COMMON_DESCENT_FP32_GPU_ENGINEERING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(args.protocol),CPU64_to_actual_GPUFP32=match,
            initial_tensor_hash=initial_hash,initial_parameter_norm=initial_norm,fixed_parameter_step=step,all20_role_cells=80,bank_hashes=bank_hashes,first_order_sha256=order_sha,arms=arms,
            temporary_updates=128,scientific_updates=0,checkpoints_saved=0,new_research_images=0,generator_UNet_forwards=0,old_ordinary_unchanged=True,frozen_AP_unchanged=True,
            precision='FP32_head_and_AP_no_autocast_TF32_disabled; original prior stored weight values converted to FP32; not original BF16 training replay',
            seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),no_full_fit_or_image_admission=True)
        write(out/'completion.json',result)
    except BaseException as e:write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args())
