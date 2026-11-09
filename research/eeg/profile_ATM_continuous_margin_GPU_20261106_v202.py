"""Engineering only: persistent Adam, exact restart, rollback and baseline replay."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import datetime,fcntl,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
plan=json.loads(Path(sys.argv[1]).read_bytes());sys.path.insert(0,str(Path(plan['parent_policy']).parent))
import atm_response_repair_contract_20261105_v196 as C
c=C.setup(plan['parent_policy']);q,K,tc=c['q'],c['K'],c['train'];out=Path(plan['output_root']);started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==plan['sources']['profile']
for pin in plan['sources'].values():K.checked(pin)
before=C.resources(c,True);job=K.read(c['repo']/'audit'/(plan['job_name']+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==tc['physical']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sys.path[:0]=tc['physical']['existing_runtime_paths']+['/root/atm_official_20261016',str(Path(__file__).parent)]
import torch
import torch.nn.functional as F
from models.atms import ATMS
from safetensors import safe_open
import atm_continuous_margin_core_20261106_v202 as H
import atm_margin_checkpoint_20261106_v202 as CK
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory)

def guard():
    assert time.monotonic()-started<850 and time.time()<plan['absolute_deadline_epoch']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30 and torch.cuda.max_memory_allocated()<20<<30
    assert K.allocated(out)<plan['maximum_bytes'];used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
    fs=os.statvfs(out);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used

try:
    src={k:torch.load(K.checked(v),map_location='cpu',weights_only=True,mmap=True)['model'] for k,v in q['models'].items()}
    initial={n:src['NN'][n] if n.startswith('proj_eeg.') else v for n,v in src['CC'].items()}
    xall=torch.load(K.checked(tc['physical']['EEG']),map_location='cpu',weights_only=True,mmap=True)
    vall=torch.load(K.checked(tc['physical']['visual_target']),map_location='cpu',weights_only=True,mmap=True)
    anchor=ATMS().cuda().eval().requires_grad_(False);anchor.load_state_dict(src['NN'],strict=True)
    net=ATMS().cuda();net.load_state_dict(initial,strict=True)
    Qdone=K.read(Path(q['project_root'])/'Qbank/completion.json');bank=np.load(K.checked(Qdone['bank']));w={};kv=[]
    with safe_open(K.checked(c['cv']['bq']['h94']),framework='pt',device='cpu') as f:
        for key,name in [('P_weight','image_proj.proj.weight'),('P_bias','image_proj.proj.bias'),('P_gamma','image_proj.norm.weight'),('P_beta','image_proj.norm.bias')]:w[key]=f.get_tensor(name).half().float().cuda()
        for lay in Qdone['layouts']:
            ix=lay['ip_index'];kv.append({k:f.get_tensor(f'ip_adapter.{ix}.{n}.weight').half().float().cuda() for k,n in [('K','to_k_ip'),('V','to_v_ip')]})
    tr=H.Trainer(net,anchor,initial,xall,vall,tc['T'],bank,Qdone['layouts'],w,kv,guard);base=tr.snapshot()
    assert np.array_equal(tr.baseline,np.load(K.checked(plan['baseline_T']))['metrics'])
    pin=K.ref(sys.argv[1]);checkpoint_dir=out/'engineering_checkpoints';CK.commit(checkpoint_dir,base,pin)
    # Independent CPU64 derivatives of two actual pair margins at original CN.
    context=tr.context(0);scope=context['scope'];m=tr.margins(scope);gpu_grads=[H.flat(torch.autograd.grad(m[i],tr.params,retain_graph=True)).detach().cpu().double() for i in [0,len(m)-1]]
    with torch.no_grad():
        y=F.normalize(tr.predict(scope),dim=1);score=y@tr.visual.T;ix=torch.tensor(scope,device='cuda');ii=torch.arange(len(ix),device='cuda');score[ii,ix]=-torch.inf
        same=ix[:,None]//10==torch.arange(13230,device='cuda')[None,:]//10;wi=score.masked_fill(~same,-torch.inf).argmax(1);gi=score.argmax(1)
    cpu=ATMS().double();cpu.load_state_dict(initial,strict=True);cpu.eval();cp=[dict(cpu.named_parameters())[n] for n in tr.names]
    cy=F.normalize(cpu(xall[np.asarray(tc['T'])[scope]].double(),torch.ones(len(scope),dtype=torch.long)),dim=1);vv=F.normalize(vall[np.asarray(tc['T'])].double(),dim=1)
    pairs=[(0,int(wi[0])),(len(scope)-1,int(gi[-1]))];proofs=[]
    for g,(i,j) in zip(gpu_grads,pairs):
        value=(cy[i]*(vv[scope[i]]-vv[j])).sum();cg=H.flat(torch.autograd.grad(value,cp,retain_graph=True));err=float((cg-g).norm()/cg.norm().clamp_min(1e-20));assert err<1e-4;proofs.append(dict(query=int(scope[i]),negative=j,relative_L2=err))
    del cpu,cp,cy,vv,cg,gpu_grads,m,score,y,context
    first=tr.attempt('margin_constrained');r1=CK.commit(checkpoint_dir,tr.snapshot(),pin)
    saved1,check1=CK.load(checkpoint_dir,pin);H.assert_tree_equal(saved1,tr.snapshot());assert check1==r1
    second=tr.attempt('margin_constrained');uninterrupted=tr.snapshot();after2=tr.all_T()
    tr.restore(saved1);repeated=tr.attempt('margin_constrained');H.assert_tree_equal(uninterrupted,tr.snapshot());assert second==repeated and np.array_equal(after2,tr.all_T())
    r2=CK.commit(checkpoint_dir,tr.snapshot(),pin);loaded2,_=CK.load(checkpoint_dir,pin);H.assert_tree_equal(loaded2,tr.snapshot())
    K.write(out/'progress.json',dict(phase='TWO_CONTINUOUS_CONSTRAINED_STEPS_AND_EXACT_RESTART_PASS',accepted=[first['accepted'],second['accepted']]))
    # Same response-only objective reproduces the original raw Adam on fullT.
    tr.restore(base);plain_first=tr.attempt('response_only')
    assert np.array_equal(tr.all_T(),np.load(K.checked(plan['raw_Adam_T']))['raw_Adam'])
    plain_second=tr.attempt('response_only');assert tr.accepted==2
    # Explicitly injected numerical failure checks rollback, not efficacy.
    tr.restore(saved1);pre=tr.snapshot();old_solve=tr.solve
    def fail_solve(context,floor,proposal,an):return torch.zeros_like(proposal),dict(KKT_valid=False,injected_engineering_test=True)
    tr.solve=fail_solve
    rejected=tr.attempt('margin_constrained');post=tr.snapshot();tr.solve=old_solve
    assert not rejected['accepted'] and post['accepted']==pre['accepted'] and post['attempts']==pre['attempts']+1
    for key in ['trainable','optimizer','CPU_rng','CUDA_rng']:H.assert_tree_equal(pre[key],post[key])
    tr.restore(base);assert all(torch.equal(v.cpu(),initial[k]) for k,v in net.state_dict().items()) and np.array_equal(tr.all_T(),tr.baseline)
    # Atomic crash boundaries use tiny payloads; actual large checkpoint restart above.
    crash=[]
    for boundary in ['pending_durable','slot_replaced','pointer_committed']:
        dr=out/('transaction_fixture_'+boundary);s0=dict(trainable={'x':torch.tensor([1.])},names=['x'],optimizer={},CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state(),attempts=0,accepted=0,exposures=0,last_event=None)
        CK.commit(dr,s0,pin);s1={**s0,'trainable':{'x':torch.tensor([2.])},'attempts':1,'accepted':1,'exposures':1,'last_event':dict(attempt=1)}
        def hook(point):
            if point==boundary:raise RuntimeError('INJECTED_'+boundary)
        try:CK.commit(dr,s1,pin,hook=hook)
        except RuntimeError as exc:assert str(exc)=='INJECTED_'+boundary
        else:raise AssertionError('missing injected exception')
        readback,_=CK.load(dr,pin);expected=s1 if boundary=='pointer_committed' else s0;H.assert_tree_equal(readback,expected)
        crash.append(dict(boundary=boundary,loaded_committed_attempt=readback['attempts'],uncommitted_artifacts_retained=True))
    guard()
    for v in q['models'].values():K.checked(v)
    K.write(out/'completion.json',dict(status='PASS_CONTINUOUS_ADAM_RESTART_ROLLBACK_AND_NATIVE_RAW_REPLAY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=pin,first=first,second=second,plain_first=plain_first,plain_second=plain_second,CPU64_margin_gradients=proofs,checkpoint_receipts=[r1,r2],last_checkpoint_reloaded_exact=True,second_step_resumed_model_Adam_RNG_event_and_allT_byteexact=True,original_raw_first_step_allT_byteexact=True,forced_rejection_rollback_pass=True,crash_boundaries=crash,all_original_weights_buffers_fullT_restored=True,temporary_attempts=6,accepted_engineering_steps=sum(x['accepted'] for x in [first,second,repeated,plain_first,plain_second]),injected_failure_attempts=1,full_fit_models=0,new_images=0,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources=C.resources(c)))
except BaseException as exc:
    restored=False
    if 'net' in globals() and 'initial' in globals():net.load_state_dict(initial,strict=True);restored=all(torch.equal(v.cpu(),initial[k]) for k,v in net.state_dict().items())
    K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),weights_restored=restored,seconds=time.monotonic()-started));raise
