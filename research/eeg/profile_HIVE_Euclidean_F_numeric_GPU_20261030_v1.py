"""Frozen same-head actual CUDA trace checks, allT and singleton R80 forwards."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=digest)
def check(x):assert ref(x['path'])==x;return Path(x['path'])
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def memory():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024


def run(path):
    q=read(path);assert q['status']=='REGISTERED_SAME_FIXED_HEAD_FP32_TRACE_ENGINEERING_ONLY'
    assert q['host']=='old' and q['training_seed']==1729 and q['original_whole64_guard_failed_retained']
    for x in q['source_pins']+q['input_pins']:check(x)
    assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==q['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    sys.path.insert(0,q['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=q['runtime_manifest']));sys.path.insert(0,str(Path(path).parent))
    import torch,numpy as np
    from brainhive_component_data_v1 import load_training
    from brainhive_ordinary_geometry_control_20261030_v1 import EuclideanConditionControl
    from brainhive_FP32_trace_audit_CPU_20261030_v1 import verify,merge
    from brainhive_functional_training_contract_v2 import resource_headroom
    assert resource_headroom()['effective_headroom_bytes']>=10<<30
    assert torch.__version__=='2.3.0+cu121' and torch.version.cuda=='12.1'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    out=Path(q['output_root']);assert not out.exists();out.mkdir();started=time.monotonic();base=memory();phase='INIT'
    def budget():
        assert time.monotonic()-started<800 and memory()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        assert shutil.disk_usage(out).free>2<<30 and sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<64<<20
    try:
        state=torch.load(check(q['component']),map_location='cpu',weights_only=True)
        assert state['seed']==1729 and state['updates']==24807 and state['seen_queries']==396900
        assert state['calibration']['selected_strength']==q['selected_strength']
        model=EuclideanConditionControl().eval().requires_grad_(False).cuda();model.load_state_dict(state['model'],strict=True);model.radius=.25*q['selected_strength']
        weights={k:v.numpy().copy() for k,v in state['model'].items()};captured={};handles=[]
        def hook(name):
            def observe(module,args,result):
                captured[name]=result.detach().cpu().numpy().copy()
                if name=='z':captured['u']=args[0].detach().cpu().numpy().copy()
            return observe
        for name,module in [('z',model.trunk[0]),('h',model.trunk[1]),('raw',model.direction)]:handles.append(module.register_forward_hook(hook(name)))
        def forward(c,g,b):
            with torch.inference_mode():output=model(c,g,b)[0]
            trace={**captured,'output':output.cpu().numpy().copy()}
            result=verify(trace,weights,c.cpu().numpy(),g.cpu().numpy(),b.cpu().numpy(),model.radius)
            return trace,result
        phase='ALL132300_TRUE_OOF_T_CAPTURED_CUDA_FORWARD'
        inputs=load_training();assert inputs['input_contract_sha256']==q['input_contract']['sha256']
        T=[];probe=[];indices=set()
        for aid in range(20):indices.update(torch.nonzero(inputs['affine_ids']==aid).flatten()[:16].cpu().tolist())
        for start in range(0,132300,512):
            sl=slice(start,min(start+512,132300));aid=inputs['affine_ids'][sl];c=inputs['condition'][sl];g=inputs['weight'][aid];b=inputs['bias'][aid]
            trace,result=forward(c,g,b);T.append(result)
            selected=[i-start for i in range(start,min(start+512,132300)) if i in indices]
            if selected:probe.append(dict(index=np.array([start+i for i in selected],np.int64),c=c.cpu().numpy()[selected],g=g.cpu().numpy()[selected],b=b.cpu().numpy()[selected],**{k:v[selected] for k,v in trace.items()}))
            budget()
        assert sum(x['queries'] for x in T)==132300
        np.savez_compressed(out/'T320_fixed_affine_trace.npz',**{k:np.concatenate([x[k] for x in probe]) for k in probe[0]},**weights)
        del inputs;torch.cuda.empty_cache();phase='ALL2000_R80_SINGLE_QUERY_PRODUCTION_SHAPE'
        source=read(check(q['R80_input_completion']));bank=np.load(check(source['bank']),allow_pickle=False)
        R80=[];saved=[]
        for index in range(2000):
            sub,row=divmod(index,200);c=torch.from_numpy(bank['conditions'][sub,row:row+1].copy()).cuda();g=torch.from_numpy(bank['gammas'][sub:sub+1].copy()).cuda();b=torch.from_numpy(bank['betas'][sub:sub+1].copy()).cuda()
            trace,result=forward(c,g,b);R80.append(result);saved.append(trace)
            if index%100==0:budget()
        all_trace={k:np.concatenate([x[k] for x in saved]) for k in saved[0]}
        np.savez_compressed(out/'R80_all2000_FP32_trace.npz',**all_trace)
        for h in handles:h.remove()
        assert all(torch.equal(v.cpu(),state['model'][k]) for k,v in model.state_dict().items())
        for x in q['input_pins']:check(x)
        budget();result=dict(status='PASS_FINITE_CAPTURED_FP32_LAYER_CHECKS_ORIGINAL_FULL64_GUARD_STILL_FAILED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),component=q['component'],training_seed=1729,selected_strength=q['selected_strength'],T=merge(T),R80=merge(R80),T_probe=ref(out/'T320_fixed_affine_trace.npz'),R80_trace=ref(out/'R80_all2000_FP32_trace.npz'),weights_unchanged=True,training_updates=0,UNet_calls=0,research_images=0,original_full64_guard_passed=False,new_formal_image_admission_issued=False,erf_interval_is_empirically_checked_not_universal_guarantee=True,seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=memory()-base)
        write(out/'completion.json',result);print(json.dumps(result))
    except BaseException as error:
        write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),scientific_retries=0,seconds=time.monotonic()-started));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
