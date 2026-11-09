"""Full-T joint conditional marginal fitting; immutable prior, two matched arms.

Only complete epochs commit. A finite segment closes at an epoch boundary;
there is no automatic retry after exceptions or resource protection failures.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def check(x):
    p=Path(x['path']);assert ref(p)==x;return p
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def write(p,x,replace=False):
    raw=(json.dumps(x,indent=2,allow_nan=False)+'\n').encode();pending=p.with_name(p.name+'.pending')
    with pending.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if not replace:assert not p.exists()
    os.replace(pending,p)
def digest(state):
    import torch
    h=hashlib.sha256()
    for key,value in state.items():
        t=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def run(spec_path,job,mode):
    s=json.loads(Path(spec_path).read_bytes());assert s['status']=='REGISTERED_FULL_T_EEG_NATIVE_COARSE_MARGINAL_v2'
    assert s['arm'] in ('EEG_coupled_teacher','visual_only_teacher')
    assert s['recipe']==dict(seed=1729,epochs=10,batch=64,E_steps=4,tokens=5,q_lr=.03,p_lr=.0001,p_weight_decay=.01)
    for item in s['source_pins']:check(item)
    root=Path(s['output_root']);root.mkdir(exist_ok=True);out=root/(mode+'_'+job);out.mkdir();started=time.monotonic();base=peak();phase='INPUTS'
    try:
        expected='run_gpu_job_v2.py' if s['host']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
        assert expected in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode().replace('\0',' ')
        receipt=Path(s['repository_root'])/'audit'/(job+'_job.json')
        for _ in range(30):
            if receipt.is_file():break
            time.sleep(.1)
        rr=json.loads(receipt.read_bytes());assert rr['pid']==os.getpid() and rr['status']=='RUNNING'
        assert float(rr['max_seconds'] if s['host']=='old' else rr['maximum_seconds'])==900
        assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        sys.path.insert(0,s['training_code']);import AR_anchored_training_contract_CPU_20261029_v1 as C
        a,_=C.load_admission(s['training_admission']);C.validate_sources(a);C.verify_protected_files(a)
        sys.path.insert(0,a['vendor'])
        import numpy as np
        import torch
        from EEG_generative_backend_loader_20261029_v1 import build_and_load
        from eeg_labram_pretrained_20261028_v1 import build_encoder
        from eeg_labram_interface_20261028_v1 import CacheNativeEEGTransform
        from EEG_native_conditional_variational_core_20261029_v1 import EEGGaussianPrior,native_moments,diagonal_KL,token_ELBO,reparameterize
        from EEG_native_marginal_core_20261029_v2 import NativeMarginalDecoder,RowGaussianAdam,posterior_parameters
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def guard():
            assert time.monotonic()-started<800,'Finite body time protection'
            assert peak()-base<10<<30,'Extra CPU protection'
            assert torch.cuda.max_memory_allocated()<20<<30,'GPU allocation protection'
            assert shutil.disk_usage(root).free>=2<<30,'Disk buffer protection'
        identities=json.loads(check(a['data']['EEG_identities']).read_bytes());tid=json.loads(check(a['data']['token_identities']).read_bytes());C.identity_join(identities['T'],tid)
        raw=np.load(check(s['T_mean63']),mmap_mode='r',allow_pickle=False);all_tokens=np.load(check(a['data']['tokens']),mmap_mode='r',allow_pickle=False)
        assert raw.shape==(13230,63,250) and all_tokens.shape==(13230,680)
        role=s['full_EEG'];stats=json.loads(check(role['normalization']).read_bytes());assert stats['role']=='full' and stats['training_positions']==list(range(13230))
        transform=CacheNativeEEGTransform(tuple(stats['channel_names']),np.array(stats['center']),np.array(stats['scale']))
        N=64 if mode=='profile' else 13230
        x=np.empty((N,63,1,200),dtype=np.float32)
        for begin in range(0,N,128):
            x[begin:begin+128]=transform(np.array(raw[begin:min(begin+128,N)]),channel_names=tuple(stats['channel_names']))
        tokens=np.array(all_tokens[:N,:5],dtype=np.int64);assert tokens.min()>=0 and tokens.max()<4096
        torch.cuda.reset_peak_memory_stats();phase='FROZEN_NATIVE_PRIOR'
        vae,d,proof=build_and_load(a['author_source'],a['vendor'],check(a['public_VQ']),check(a['public_VAR']),seed=1729)
        public=torch.load(check(a['public_VAR']),weights_only=True,map_location='cpu',mmap=True);classes=public['class_emb.weight'].clone();del public
        before=digest({k:v for k,v in d.state_dict().items() if not k.startswith('embed_proj.')});vq_before=digest(vae.state_dict())
        native=NativeMarginalDecoder(d,classes).cuda().eval();vae.cuda().eval().requires_grad_(False);d.rng=torch.Generator(device='cuda')
        center,scale=native_moments(classes);center=center.cuda();scale=scale.cuda()
        encoder=build_encoder(a['author_source'],a['vendor'],seed=1729);ck=torch.load(check(role['model']),weights_only=True,map_location='cpu')
        assert ck['role']=='full' and ck['epoch']==50;encoder.load_state_dict(ck['model'],strict=True);del ck
        p=EEGGaussianPrior(encoder,center,scale).cuda().eval()
        from models.labram import get_input_chans
        channels=get_input_chans([name.upper() for name in stats['channel_names']])
        q=RowGaussianAdam(N,1024,'cuda');op=torch.optim.AdamW([v for v in p.parameters() if v.requires_grad],lr=1e-4,weight_decay=.01)
        rng=torch.Generator(device='cuda').manual_seed(1729);epochs=0;updates=0;events=[];coverage=[]
        initial_p=digest(p.state_dict());guard()
        # First two scales need only the first VQ code to construct their four
        # shifted features. No fine-scale target is used in training.
        def features(target):
            with torch.no_grad():
                values=vae.quantize.embedding(target[:,:1]).transpose(1,2).reshape(len(target),d.Cvae,1,1)
                fhat=torch.zeros(len(target),d.Cvae,16,16,device='cuda')
                _,following=vae.quantize.get_next_autoregressive_input(0,10,fhat,values)
                return following.flatten(2).transpose(1,2)
        if mode=='profile':
            target=torch.tensor(np.array(all_tokens[:4]),device='cuda',dtype=torch.long)
            with torch.no_grad():
                full_features=vae.quantize.idxBl_to_var_input(list(torch.split(target,[1,4,9,16,25,36,64,100,169,256],dim=1)))
                f=features(target[:,:5]);assert torch.equal(f,full_features[:,:4])
                z=classes[[0,11,287,1000]].cuda();full=native(full_features,z);coarse=native.marginal_logits(f,z)
                prefix_error=float((coarse-full[:,:5]).abs().max());assert torch.allclose(coarse,full[:,:5],atol=2e-5,rtol=2e-5)
            del full,coarse,full_features,z,target
        else:
            passed=json.loads(check(s['profile_completion']).read_bytes());assert passed['status']=='PASS_REAL_64_QUERY_NATIVE_MARGINAL_PROFILE_ALL_TEMPORARY_UPDATES_RESET'
            commit_path=root/'commit.json'
            if commit_path.is_file():
                commit=json.loads(commit_path.read_bytes());state=torch.load(check(commit['checkpoint']),weights_only=False,map_location='cpu')
                assert state['spec_sha256']==sha(spec_path) and state['epoch']==commit['epoch']
                p.load_state_dict(state['EEG_model'],strict=True);op.load_state_dict(state['EEG_optimizer']);q.load_state_dict(state['posterior'])
                rng.set_state(state['latent_RNG']);torch.set_rng_state(state['CPU_RNG']);torch.cuda.set_rng_state_all(state['CUDA_RNG'])
                epochs=state['epoch'];updates=state['updates'];events=state['events'];coverage=state['coverage'];del state
        phase='REAL_EEG_COUPLED_MARGINAL_E_AND_M';loop=time.monotonic();backbone_nonzero=False
        for epoch in range(epochs,1 if mode=='profile' else 10):
            order=torch.randperm(N,generator=torch.Generator().manual_seed(1729+epoch));observed=[];epoch_start=time.monotonic()
            for begin in range(0,N,64):
                guard();ids_cpu=order[begin:begin+64];ids=ids_cpu.cuda();target=torch.from_numpy(tokens[ids_cpu.numpy()]).cuda();eeg=torch.from_numpy(x[ids_cpu.numpy()]).cuda();prefix=features(target)
                with torch.no_grad():pm,ps=p(eeg,input_chans=channels)
                teacher_pm,teacher_ps=(pm,ps) if s['arm']=='EEG_coupled_teacher' else (center.expand_as(pm),scale.log().expand_as(ps))
                last=None
                for inner in range(4):
                    leaf=q.leaf(ids);qm,qs=posterior_parameters(leaf,center,scale);sample=reparameterize(qm,qs,generator=rng)
                    logits=native.marginal_logits(prefix,sample);loss,parts=token_ELBO(logits,target,qm,qs,teacher_pm.detach(),teacher_ps.detach());loss.backward()
                    assert all(v.grad is None for v in p.parameters()) and all(v.grad is None for v in native.parameters())
                    q.step(ids,leaf);last=dict(ELBO_nats_per_token=float(loss.detach()),NLL_nats_per_query=float(parts['token_NLL_nats_per_query'].mean()),KL_nats_per_query=float(parts['KL_nats_per_query'].mean()))
                    del leaf,qm,qs,sample,logits,loss
                op.zero_grad(set_to_none=True)
                with torch.no_grad():qm,qs=posterior_parameters(q.value[ids],center,scale)
                pm,ps=p(eeg,input_chans=channels);loss=diagonal_KL(qm,qs,pm,ps).mean()/5;loss.backward()
                backbone=sum(float(v.grad.square().sum()) for name,v in p.encoder.named_parameters() if v.grad is not None and not name.startswith(('head.','logit_scale')))**.5
                backbone_nonzero=backbone_nonzero or backbone>0
                grad=float(torch.nn.utils.clip_grad_norm_(p.parameters(),1.));assert torch.isfinite(loss) and all(v.grad is None for v in native.parameters())
                op.step();op.zero_grad(set_to_none=True);updates+=1;observed.extend(ids_cpu.tolist())
                events.append(dict(epoch=epoch+1,batch=begin//64+1,queries=len(ids),EEG_KL_nats_per_token=float(loss.detach()),preclip_EEG_gradient=grad,backbone_gradient=backbone,**last))
                del ids,ids_cpu,target,eeg,prefix,pm,ps,qm,qs,loss,teacher_pm,teacher_ps
            assert observed==order.tolist() and sorted(observed)==list(range(N))
            coverage.append(dict(epoch=epoch+1,queries=N,order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),seconds=time.monotonic()-epoch_start))
            epochs=epoch+1;torch.cuda.synchronize();guard()
            assert bool((q.count==4*epochs).all())
            if mode=='profile':break
            phase='ATOMIC_WHOLE_EPOCH_CHECKPOINT'
            state=dict(spec_sha256=sha(spec_path),epoch=epochs,updates=updates,EEG_model=p.state_dict(),EEG_optimizer=op.state_dict(),posterior=q.state_dict(),latent_RNG=rng.get_state(),CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),events=events,coverage=coverage)
            path=root/('state_A.pt' if epochs%2 else 'state_B.pt');pending=root/(path.name+'.pending');assert not pending.exists()
            torch.save(state,pending)
            with pending.open('rb') as f:os.fsync(f.fileno())
            os.replace(pending,path);del state
            commit=dict(status='WHOLE_EPOCH_COMMITTED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),epoch=epochs,updates=updates,query_exposures=epochs*N,posterior_row_updates=4*epochs,checkpoint=ref(path),spec_sha256=sha(spec_path))
            write(root/'commit.json',commit,replace=(root/'commit.json').exists());write(root/'events.json',events,replace=(root/'events.json').exists())
            write(root/'progress.json',dict(**commit,last=events[-1],coverage=coverage),replace=(root/'progress.json').exists())
            phase='REAL_EEG_COUPLED_MARGINAL_E_AND_M'
            if time.monotonic()-started>600:break
        torch.cuda.synchronize();loop_seconds=time.monotonic()-loop
        assert digest(d.state_dict())==before and digest(vae.state_dict())==vq_before
        C.verify_protected_files(a);guard()
        if mode=='profile':
            assert updates==1 and bool((q.count==4).all())
            # Explicitly reset all temporary objects, not a reusable trained head.
            del p,op,q
            status='PASS_REAL_64_QUERY_NATIVE_MARGINAL_PROFILE_ALL_TEMPORARY_UPDATES_RESET'
        else:
            status='FULL_T_NATIVE_MARGINAL_FIXED_FINAL_FIT_COMPLETE' if epochs==10 else 'WHOLE_EPOCH_SEGMENT_CLOSED_REQUIRES_NEXT_FINITE_SEGMENT'
            assert updates==207*epochs and bool((q.count==4*epochs).all()) and backbone_nonzero
            if epochs==10:
                final=root/'EEG_prior_final.pt';assert not final.exists()
                torch.save(dict(model=p.state_dict(),epoch=epochs,updates=updates,spec_sha256=sha(spec_path),arm=s['arm']),final)
        result=dict(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(spec_path),host=s['host'],arm=s['arm'],epoch=epochs,EEG_updates=updates,queries_per_epoch=N,posterior_updates_per_query=4*epochs,
            pretrained_VAR_VQ_unchanged=True,native_condition_identity_preserved=True,coarse_tokens=5,coarse_scales=[1,2],loop_seconds=loop_seconds,seconds=time.monotonic()-started,
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base,initial_EEG_model_digest=initial_p,
            new_research_images=0,method_benefit_or_SOTA_established=False,events=events if mode=='profile' else events[-5:],coverage=coverage)
        if mode=='profile':result['native_full_vs_marginal_logits_max_error']=prefix_error
        elif epochs==10:result['final_model']=ref(final);result['checkpoint']=json.loads((root/'commit.json').read_bytes())['checkpoint']
        write(out/'completion.json',result)
        if mode=='profile':write(root/'profile_completion.json',dict(**result,actual_completion=ref(out/'completion.json')))
        elif epochs==10:write(root/'completion.json',result)
    except BaseException as error:
        write(out/'failure.json',dict(status='NEW_ROUTE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--job-name',required=True);ap.add_argument('--mode',choices=['profile','fit'],required=True);v=ap.parse_args();run(v.spec,v.job_name,v.mode)
