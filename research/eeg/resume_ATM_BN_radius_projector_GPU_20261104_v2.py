"""One fullT10 BN-only statistic fit, all400 conditions, native projector probe."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import fcntl,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
recovery=json.loads(Path(sys.argv[1]).read_bytes());sys.path.insert(0,str(Path(recovery['original_policy']['path']).parent))
import atm_BN_radius_contract_20261104_v1 as B
c=B.setup(recovery['original_policy']['path']);q=c['q'];K=c['K'];out=c['out'];up=c['up'];started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==recovery['sources']['GPU'];before=B.resources(c,starting=True)
job=K.read(c['repo']/'audit'/(recovery['job_name']+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['parent']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(c['parent']['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
sys.path.insert(0,'/root/atm_official_20261016');from models.atms import ATMS
def guard():
    assert time.monotonic()-started<850 and torch.cuda.max_memory_allocated()<c['parent']['maximum_GPU_bytes']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<c['parent']['maximum_extra_CPU_bytes']
def sha_tensor(t):return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
try:
    assert not (out/'GPU_completion.json').exists()
    for pin in recovery['completed_prefix'].values():K.checked(pin)
    checkpoint=K.checked(recovery['completed_prefix']['checkpoint']);values=np.load(K.checked(recovery['completed_prefix']['conditions']),allow_pickle=False)
    raw=torch.load(K.checked(up['q']['EEG']),map_location='cpu',weights_only=True,mmap=True)
    initial=torch.load(K.checked(up['q']['encoder']),map_location='cpu',weights_only=True,mmap=True)
    torch.manual_seed(1729);torch.cuda.manual_seed(1729);temporary=ATMS().cuda();temporary.load_state_dict(initial,strict=True);temporary.requires_grad_(False).train()
    bn={name:m for name,m in temporary.named_modules() if isinstance(m,torch.nn.BatchNorm2d)};assert tuple(bn)==B.BN_NAMES
    moments=np.load(K.checked(recovery['completed_prefix']['moments']),allow_pickle=False);input_checks=[];probe_id=0
    def capture(name,index):
        def callback(module,inputs):
            x=inputs[0];dims=(0,2,3);xx=x.detach().cpu().numpy().astype(float);mu=xx.mean(axis=dims);var=xx.var(axis=dims,ddof=1)
            gm=x.mean(dims).cpu().numpy();gv=x.var(dims,unbiased=True).cpu().numpy()
            errors=dict(CPU_mean=float(np.max(abs(mu-gm)/(1+abs(mu)))),CPU_variance=float(np.max(abs(var-gv)/(1+abs(var)))),saved_mean=float(np.max(abs(mu-moments['mean_'+str(index)][probe_id])/(1+abs(mu)))),saved_variance=float(np.max(abs(var-moments['variance_'+str(index)][probe_id])/(1+abs(var)))))
            assert max(errors.values())<5e-5;input_checks.append(dict(original_step=probe_id,BN=name,input_shape=list(x.shape),scaled_errors=errors,temporary_probe=True))
        return callback
    handles=[m.register_forward_pre_hook(capture(name,i)) for i,(name,m) in enumerate(bn.items())]
    for epoch,bi in [(0,0),(9,82)]:
        probe_id=epoch*83+bi;order=np.random.default_rng(1729+epoch).permutation(1323);ids=c['T'][(order[bi*16:bi*16+16,None]*10+np.arange(10)).reshape(-1)];torch.cuda.manual_seed(1729+epoch*10000+bi)
        with torch.inference_mode():temporary(raw[ids].cuda(),torch.ones(len(ids),device='cuda',dtype=torch.long))
    for h in handles:h.remove()
    del temporary,bn,raw,initial;torch.cuda.empty_cache()
    K.write(out/'fixed2_BN_input_probe_recovery_v2.json',dict(checks=input_checks,temporary_statistic_updates=2,formal_BN_updates_repeated=0))
    from safetensors import safe_open
    from diffusers.models.embeddings import ImageProjection
    mapping={'image_embeds.weight':'image_proj.proj.weight','image_embeds.bias':'image_proj.proj.bias','norm.weight':'image_proj.norm.weight','norm.bias':'image_proj.norm.bias'}
    with safe_open(str(K.checked(up['cv']['bq']['h94'])),framework='pt',device='cpu') as src:weights={name:src.get_tensor(source).half() for name,source in mapping.items()}
    projector=ImageProjection(image_embed_dim=1024,cross_attention_dim=2048,num_image_text_embeds=4).half().cuda().eval().requires_grad_(False);projector.load_state_dict(weights,strict=True)
    assert all(sha_tensor(value)==sha_tensor(weights[name]) for name,value in projector.state_dict().items())
    tokens=np.empty((400,6,4,2048),np.float16)
    with torch.inference_mode():
        for lo in range(0,400,16):tokens[lo:lo+16]=projector(torch.from_numpy(values[lo:lo+16].reshape(-1,1024)).cuda().half()).reshape(-1,6,4,2048).cpu().numpy();guard()
    K.save_array(out/'tokens400_FP16.npy',tokens)
    # Independent CPU audit later verifies these inherited-half weights in FP32.
    cpu_projector=ImageProjection(image_embed_dim=1024,cross_attention_dim=2048,num_image_text_embeds=4).float().eval().requires_grad_(False);cpu_projector.load_state_dict({k:v.float() for k,v in weights.items()},strict=True)
    fp32=projector.float();fp32_values=[];errors=[]
    with torch.inference_mode():
        for lo in range(0,400,16):
            v=torch.from_numpy(values[lo:lo+16].reshape(-1,1024));g=fp32(v.cuda()).cpu();reference=cpu_projector(v);error=float(((g-reference).abs()/(1+reference.abs())).max());errors.append(error);assert error<5e-5;fp32_values.append(g.numpy().reshape(-1,6,4,2048));guard()
    # Keep fixed8 rows rather than a second full token bank.
    selected=np.arange(0,80,10);K.save_array(out/'fixed8_tokens_FP32.npy',np.concatenate(fp32_values)[selected]);np.savez_compressed(out/'projector_weights_FP16.npz',**{k:v.numpy() for k,v in weights.items()})
    probes={}
    for arm in (1,2,3,4,5):
        x=tokens[:,arm].astype(float).reshape(400,-1);y=tokens[:,0].astype(float).reshape(400,-1)
        probes[B.ARMS[arm]]=dict(bit_equal_queries=int(np.all(x==y,axis=1).sum()),mean_token_cosine=float(np.mean(np.sum(x*y,1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)))),mean_token_squared_error=float(np.mean((x-y)**2)))
    K.write(out/'GPU_completion.json',dict(status='COMPLETE_T_ONLY_BN_AND_RADIUS_NATIVE_PROJECTOR_DIAGNOSTIC',policy=K.ref(recovery['original_policy']['path']),recovery_policy=K.ref(sys.argv[1]),original_failed_job=recovery['original_failed_job'],no_formal_BN_refit=True,temporary_fixed_input_probes=2,checkpoint=K.ref(checkpoint),conditions=K.ref(out/'conditions400.npy'),tokens=K.ref(out/'tokens400_FP16.npy'),
        statistic_updates=830,optimizer_updates=0,exposures=132300,all_parameters_and_nonBN_buffers_bit_exact=True,BN_input_moment_checks=input_checks,full2400_CPU_GPU_FP32_projector_scaled_errors=errors,
        native_projector_weight_source=up['cv']['bq']['h94'],probes=probes,BN_names=list(B.BN_NAMES),raw_condition_mean_norms={arm:float(np.linalg.norm(values[:,i],axis=1).mean()) for i,arm in enumerate(B.ARMS)},
        new_research_images=0,no_query_teacher_or_labels_used_at_inference=True,not_new_method_or_image_gain=True,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_after=B.resources(c)))
except BaseException as exc:K.write(out/'GPU_recovery_failure_v2.json',dict(error=repr(exc),traceback=traceback.format_exc(),partial_evidence_retained=True));raise
