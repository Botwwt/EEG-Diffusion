"""All5 backbones x3 concept-isolated roles, fixed25 epochs, full1024 NCE via caching."""
import argparse,json,math,os,socket,time,traceback
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--role',required=True);ap.add_argument('--profile',action='store_true');args=ap.parse_args()
from PKU_backbone_core_20261110_v322 import *
start=time.monotonic();jid=os.environ['SLURM_JOB_ID'];folder=ROOT/args.case/'encoders'/args.role;folder.mkdir(parents=True,exist_ok=True)
try:
    from PKU_HIVE_native_classes_20261109_v239 import classes,teacher
    assert args.case in CASES and args.role in ROLES and torch.cuda.device_count()==1 and not (folder/'completion.json').exists()
    torch.set_num_threads(4);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    policy=R/'code/EEG_backbone_policy_v322.json';spec=json.loads(policy.read_text());assert spec['epochs']==25 and spec['effective_batch']==1024 and spec['microbatch']==128
    pins=json.loads((R/'code/EEG_backbone_pins_v322.json').read_text())
    for n,h in pins.items():assert sha(R/'code'/n)==h
    split,T,folds,post=input_contract();pos=torch.arange(13230) if args.role=='full' else torch.where(folds!=int(args.role[-1]))[0];allowed=torch.tensor(T)[pos];N=len(allowed);expected=25*math.ceil(N/1024)
    assert set(allowed.tolist()).isdisjoint(split['image_rows']['D']+split['image_rows']['V'])
    rawpath=R/'data/recovery_20261108_v236/sub-03_train_mean4.pt';assert sha(rawpath)=='85a25128f6b94e3739478c60e089d33f437a8dcbeb31cf136ce9772d2cc1615d';raw=torch.load(rawpath,map_location='cpu',weights_only=True,mmap=True).cuda()
    data=torch.load(R/'data/native_training_v239/train_visual_embeddings.pt',map_location='cpu',weights_only=True);data={k:v.cuda() for k,v in data.items()};vt=teacher(classes()).cuda().train().requires_grad_(False)
    for m in vt.modules():
        if isinstance(m,nn.Dropout):m.p=.1
    torch.manual_seed(1729);torch.cuda.manual_seed_all(1729);model=make(args.case).cuda().train();scale=nn.Parameter(torch.tensor(2.6592,device='cuda'))
    normids={id(p) for m in model.modules() if isinstance(m,nn.LayerNorm) for p in m.parameters()};decay=[p for n,p in model.named_parameters() if not n.endswith('bias') and id(p) not in normids]+[scale];nodecay=[p for n,p in model.named_parameters() if n.endswith('bias') or id(p) in normids]
    opt=torch.optim.AdamW([dict(params=decay,weight_decay=.05),dict(params=nodecay,weight_decay=0.)],lr=5e-4,betas=(.9,.995));sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:.5*(1+math.cos(math.pi*step/expected)))
    shuffle=torch.Generator().manual_seed(1730);orders=[torch.randperm(N,generator=shuffle) for ep in range(25)];teacher_rng=dict(CPU=torch.Generator().manual_seed(1732).get_state(),CUDA=torch.Generator(device='cuda').manual_seed(1732).get_state())
    provenance=dict(case=args.case,role=args.role,policy_sha256=sha(policy),train_array_rows=allowed.tolist(),train_positions=pos.tolist(),epochs=25,seed=1729,effective_batch=1024,microbatch=128,source_pins=pins,teacher_dropout=.1,teacher_rng_independent=True,parameters=sum(p.numel() for p in model.parameters()),orders_sha256=[hashlib.sha256(o.numpy().tobytes()).hexdigest() for o in orders])
    if (folder/'provenance.json').exists():assert json.loads((folder/'provenance.json').read_text())==provenance
    else:write(folder/'provenance.json',provenance)
    cursor=0;updates=0;history=[];coverage=torch.zeros(25,N,dtype=torch.bool);prior=0.
    if not args.profile:
        s=torch.load(folder/'state.pt',map_location='cpu',weights_only=False);assert s['provenance_sha256']==sha(folder/'provenance.json');model.load_state_dict(s['model'],strict=True);scale.data.copy_(s['scale'].cuda());opt.load_state_dict(s['optimizer']);sched.load_state_dict(s['scheduler']);cursor=s['cursor'];updates=s['updates'];history=s['history'];coverage=s['coverage'];prior=s['seconds'];teacher_rng=s['teacher_RNG'];set_rng(s['student_RNG'],torch.device('cuda'));del s
    else:assert not (folder/'state.pt').exists()
    def snapshot():
        save(folder/'state.pt',dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},scale=scale.detach().cpu(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),cursor=cursor,updates=updates,history=history,coverage=coverage,seconds=prior+time.monotonic()-start,teacher_RNG=teacher_rng,student_RNG=rng_state(torch.device('cuda')),provenance_sha256=sha(folder/'provenance.json')))
        write(folder/'checkpoint_receipt.json',dict(updates=updates,exposures=cursor,checkpoint_sha256=sha(folder/'state.pt'),job_id=jid))
    stop=False;torch.cuda.reset_peak_memory_stats()
    with (folder/('attempt_'+jid+'.jsonl')).open('x',buffering=1) as log:
        for ep in range(cursor//N,25):
            offset0=cursor%N if ep==cursor//N else 0
            for off in range(offset0,N,1024):
                if time.monotonic()-start>850 and updates>0:stop=True;break
                indices=orders[ep][off:off+1024];rows=allowed[indices].cuda();xx=raw[rows];opt.zero_grad(set_to_none=True)
                def target_fn(lo,hi):
                    global teacher_rng
                    with torch.random.fork_rng(devices=[0]):
                        set_rng(teacher_rng,torch.device('cuda'))
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):target=vt(**{k:v[rows[lo:hi]] for k,v in data.items()})
                        teacher_rng=rng_state(torch.device('cuda'))
                    return target
                loss=cached_backward(model,lambda x:encode(model,args.case,x,post),xx,scale,target_fn,micro=128,mixed=True);assert math.isfinite(loss)
                grad=torch.nn.utils.clip_grad_norm_(list(model.parameters())+[scale],1.,error_if_nonfinite=True);assert grad>0 and not any(p.grad is not None for p in vt.parameters());opt.step();sched.step()
                assert not coverage[ep,indices].any();coverage[ep,indices]=True;cursor=ep*N+off+len(indices);updates+=1
                e=dict(update=updates,epoch=ep+1,offset=off,n=len(indices),exposures=cursor,loss=loss,gradient_norm=float(grad),seconds=prior+time.monotonic()-start,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated());assert e['peak_GPU_allocated_bytes']<20<<30;history.append(e);log.write(json.dumps(e)+'\n')
                if updates==1 or updates%16==0 or off+len(indices)==N:write(folder/'progress.json',e);print(json.dumps(e),flush=True)
                if args.profile or updates%16==0:snapshot()
                if args.profile:stop=True;break
            if stop:break
        snapshot()
    if cursor==25*N:
        assert updates==expected and coverage.all();model.eval();af=affine(model,args.case)
        save(folder/'brain_final.pt',dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},affine=af,case=args.case,role=args.role,subject='sub-03',epochs=25,updates=updates,seed=1729))
        export=T+split['image_rows']['D'] if args.role=='full' else torch.tensor(T)[folds==int(args.role[-1])].tolist();values=[]
        with torch.inference_mode():
            for lo in range(0,len(export),64):values.append(encode(model,args.case,raw[export[lo:lo+64]],post).float().cpu())
        save(folder/'conditions.pt',dict(condition=torch.cat(values),array_rows=torch.tensor(export),model_sha256=sha(folder/'brain_final.pt')))
        write(folder/'completion.json',dict(status='COMPLETE_FIXED25_ONE_BACKBONE_ROLE',case=args.case,role=args.role,epochs=25,updates=updates,exposures=cursor,model_sha256=sha(folder/'brain_final.pt'),conditions_sha256=sha(folder/'conditions.pt'),parameters=provenance['parameters']))
    write(folder/('attempt_'+jid+'_completion.json'),dict(updates=updates,exposures=cursor,full_complete=cursor==25*N,profile=args.profile,new_images=0));print('SAFE_BACKBONE_PREFIX '+str(updates),flush=True)
except BaseException as e:write(ROOT/('failure_backbone_'+jid+'.json'),dict(case=args.case,role=args.role,error=repr(e),traceback=traceback.format_exc()));raise
