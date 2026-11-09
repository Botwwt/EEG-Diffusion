"""One fullT10 BN-only statistic fit, all400 conditions, native projector probe."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import fcntl,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_BN_radius_contract_20261104_v1 as B
c=B.setup(sys.argv[1]);q=c['q'];K=c['K'];out=c['out'];up=c['up'];started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==q['sources']['GPU'];before=B.resources(c,starting=True)
job=K.read(c['repo']/'audit'/(q['job_name']+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
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
    raw=torch.load(K.checked(up['q']['EEG']),map_location='cpu',weights_only=True,mmap=True)
    initial=torch.load(K.checked(up['q']['encoder']),map_location='cpu',weights_only=True,mmap=True)
    torch.manual_seed(1729);torch.cuda.manual_seed(1729);model=ATMS().cuda();model.load_state_dict(initial,strict=True);model.requires_grad_(False).train()
    bns={name:m for name,m in model.named_modules() if isinstance(m,torch.nn.BatchNorm2d)};assert tuple(bns)==B.BN_NAMES
    assert all(m.momentum==.1 and m.track_running_stats and m.num_features==40 for m in bns.values())
    before_buffers={n:{k:v.detach().cpu().numpy().copy() for k,v in m.named_buffers()} for n,m in bns.items()};moments=[];input_checks=[];current={};step=0
    def hook(name):
        def capture(module,inputs):
            x=inputs[0];dims=(0,2,3);mu=x.mean(dims);var=x.var(dims,unbiased=True)
            current[name]=dict(mean=mu.detach().cpu().numpy().copy(),variance=var.detach().cpu().numpy().copy(),count=x.numel()//40)
            if step in (0,829):
                xx=x.detach().cpu().numpy().astype(np.float64);m=xx.mean(axis=dims);v=xx.var(axis=dims,ddof=1)
                errors=dict(mean=float(np.max(abs(m-current[name]['mean'])/(1+abs(m)))),variance=float(np.max(abs(v-current[name]['variance'])/(1+abs(v)))))
                assert max(errors.values())<5e-5;input_checks.append(dict(step=step,BN=name,input_shape=list(x.shape),FP64_input_moment_scaled_errors=errors))
        return capture
    handles=[m.register_forward_pre_hook(hook(n)) for n,m in bns.items()];epoch_events=[]
    for epoch in range(10):
        order=np.random.default_rng(1729+epoch).permutation(1323);coverage=np.zeros(13230,np.int8)
        for bi,lo in enumerate(range(0,1323,16)):
            guard();positions=(order[lo:lo+16,None]*10+np.arange(10)).reshape(-1);ids=c['T'][positions];coverage[positions]+=1;current={}
            torch.cuda.manual_seed(1729+epoch*10000+bi)
            with torch.inference_mode():model(raw[ids].cuda(),torch.ones(len(ids),device='cuda',dtype=torch.long))
            assert tuple(current)==B.BN_NAMES;moments.append(dict(epoch=epoch+1,batch=bi,queries=len(ids),array_rows_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),BN=current));step+=1
        assert np.all(coverage==1);B.resources(c)
        event=dict(epoch=epoch+1,statistic_updates=step,exposures=(epoch+1)*13230,optimizer_updates=0,concept_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),coverage_min=int(coverage.min()),coverage_max=int(coverage.max()))
        epoch_events.append(event);K.write(out/'fit_progress.json',event)
    for h in handles:h.remove()
    assert step==830
    bn_keys={name+'.'+kind for name in bns for kind in ('running_mean','running_var','num_batches_tracked')}
    state={name:t.detach().cpu().clone() for name,t in model.state_dict().items()}
    assert all(torch.equal(value,initial[name]) for name,value in state.items() if name not in bn_keys)
    assert all(int(state[name+'.num_batches_tracked'])==int(initial[name+'.num_batches_tracked'])+830 for name in bns)
    assert all(p.grad is None for p in model.parameters())
    checkpoint=out/'BN_only_state.pt';assert not checkpoint.exists();torch.save(dict(model=state,statistic_updates=830,optimizer_updates=0,exposures=132300,CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state(),initial_encoder=up['q']['encoder'],policy=K.ref(sys.argv[1])),checkpoint)
    K.write(out/'epochs.json',epoch_events)
    arrays={}
    for j,name in enumerate(bns):
        arrays['mean_'+str(j)]=np.stack([v['BN'][name]['mean'] for v in moments]);arrays['variance_'+str(j)]=np.stack([v['BN'][name]['variance'] for v in moments])
        for kind,v in before_buffers[name].items():arrays['initial_'+str(j)+'_'+kind]=v
        for kind in ('running_mean','running_var','num_batches_tracked'):arrays['final_'+str(j)+'_'+kind]=state[name+'.'+kind].numpy()
    np.savez_compressed(out/'BN_moments_and_buffers.npz',**arrays)
    K.write(out/'BN_batch_identities.json',[{k:v for k,v in row.items() if k!='BN'} for row in moments])
    prepared=K.read(up['out']/'prepared.json');old=np.load(K.checked(prepared['conditions']),allow_pickle=False);assert old.shape==(400,5,1024)
    values=np.empty((400,6,1024),np.float32);values[:,:3]=old[:,[0,2,3]];model.eval()
    with torch.inference_mode():
        for lo in range(0,400,128):
            ids=[r['array_row'] for r in c['queries'][lo:lo+128]];values[lo:lo+len(ids),3]=model(raw[ids].cuda(),torch.ones(len(ids),device='cuda',dtype=torch.long)).cpu().numpy()
    values[:,4],values[:,5]=B.decompose(values[:,0],values[:,1]);K.save_array(out/'conditions400.npy',values)
    K.write(out/'queries400.json',c['queries']);guard();del model,raw,initial,state;torch.cuda.empty_cache()
    from safetensors import safe_open
    from diffusers.models.embeddings import ImageProjection
    mapping={'image_embeds.weight':'image_proj.proj.weight','image_embeds.bias':'image_proj.proj.bias','norm.weight':'image_proj.norm.weight','norm.bias':'image_proj.norm.bias'}
    with safe_open(str(K.checked(c['parent']['h94'])),framework='pt',device='cpu') as src:weights={name:src.get_tensor(source).half() for name,source in mapping.items()}
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
    K.write(out/'GPU_completion.json',dict(status='COMPLETE_T_ONLY_BN_AND_RADIUS_NATIVE_PROJECTOR_DIAGNOSTIC',policy=K.ref(sys.argv[1]),checkpoint=K.ref(checkpoint),conditions=K.ref(out/'conditions400.npy'),tokens=K.ref(out/'tokens400_FP16.npy'),
        statistic_updates=830,optimizer_updates=0,exposures=132300,all_parameters_and_nonBN_buffers_bit_exact=True,BN_input_moment_checks=input_checks,full2400_CPU_GPU_FP32_projector_scaled_errors=errors,
        native_projector_weight_source=c['parent']['h94'],probes=probes,BN_names=list(bns),raw_condition_mean_norms={arm:float(np.linalg.norm(values[:,i],axis=1).mean()) for i,arm in enumerate(B.ARMS)},
        new_research_images=0,no_query_teacher_or_labels_used_at_inference=True,not_new_method_or_image_gain=True,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_after=B.resources(c)))
except BaseException as exc:K.write(out/'GPU_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),partial_evidence_retained=True));raise
