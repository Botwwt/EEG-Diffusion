"""Registered six-case EEG-side study, finite same-state prefixes, no effect selection."""
import argparse,gc,hashlib,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--profile-all',action='store_true');ap.add_argument('--case');args=ap.parse_args()
R=Path('/dfs/share/maleiLab/wwt/eeg');ROOT=R/'runs/EEG_candidates_v267';ROOT.mkdir(exist_ok=True);started=time.monotonic()
def write(p,r):
    b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode();p=Path(p);t=p.with_name(p.name+'.pending')
    with t.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
def save(p,r):
    t=p.with_name(p.name+'.pending')
    with t.open('wb') as f:torch.save(r,f);f.flush();os.fsync(f.fileno())
    t.replace(p)
try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from diffusers import DDPMScheduler
    from PKU_eeg_candidate_core_20261109_v266 import CASES,seed,sha,posterior_indices,negative_cycle,spatial_statistics,make_model,encode
    from PKU_native_generator_20261109_v253 import load_native,denoise
    assert (args.profile_all and args.case is None) or (not args.profile_all and args.case in CASES)
    torch.set_num_threads(4);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    policy_path=R/'code/EEG_two_core_candidates_v266.json';policy=json.loads(policy_path.read_text());assert policy['passes_per_case']==1 and policy['updates_per_case']==827
    pins=json.loads((R/'code/EEG_candidate_source_pins_v267.json').read_text())
    for name,digest in pins.items():assert sha(R/'code'/name)==digest
    parent=R/'runs/HIVE_native_sub03_5096542';split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());ids=split['image_rows']['T'];post,other=posterior_indices()
    raw_path=R/'data/recovery_20261108_v236/sub-03_train_mean4.pt';assert sha(raw_path)=='85a25128f6b94e3739478c60e089d33f437a8dcbeb31cf136ce9772d2cc1615d'
    raw=torch.load(raw_path,map_location='cpu',weights_only=True,mmap=True);assert raw.shape==(16540,63,250);eeg=raw[ids].clone().cuda();del raw
    latent_path=R/'data/native_T_latents_v256/native_VAE_T_modes.npy';assert sha(latent_path)=='b1198b3e07361e3b941f87e3b60146b1cc7957293e813db69f7b59cfa8469a2d'
    a=np.load(latent_path,mmap_mode='r');latents=torch.tensor(a,device='cuda');del a
    allF=torch.load(parent/'deterministic_F_all16540.pt',weights_only=True,map_location='cpu');target=allF[ids].clone().cuda();del allF
    neg,probe=negative_cycle(ids);neg=neg.cuda();order=torch.randperm(13230,generator=torch.Generator().manual_seed(seed('order')))
    if (ROOT/'spatial_statistics.pt').exists():stats=torch.load(ROOT/'spatial_statistics.pt',weights_only=True,map_location='cpu')
    else:
        assert args.profile_all;stats=spatial_statistics(eeg,post,other);save(ROOT/'spatial_statistics.pt',stats)
    pipe,A,P,prompts,host=load_native();pipe.unet.train();pipe.unet.enable_gradient_checkpointing();assert float(pipe.unet.config.dropout)==0
    prompts=tuple(x.to(torch.bfloat16) for x in prompts)
    # These modules have no forward role in this latent training graph.
    for name in ['vae','text_encoder','text_encoder_2']:getattr(pipe,name).cpu()
    torch.cuda.empty_cache();scheduler=DDPMScheduler.from_config(pipe.scheduler.config)
    if not (ROOT/'energy_temperature.pt').exists():
        assert args.profile_all;base=make_model('A_positive',stats).requires_grad_(False);gen=torch.Generator(device='cuda').manual_seed(seed('temperature'));gaps=[]
        with torch.no_grad():
            for lo in range(0,256,8):
                ix=torch.tensor(probe[lo:lo+8],device='cuda');noise=torch.randn(latents[ix].shape,device='cuda',generator=gen);t=torch.randint(0,1000,(len(ix),),device='cuda',generator=gen);z=scheduler.add_noise(latents[ix],noise,t)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    cp=encode(base,'A_positive',eeg[ix],post);cn=encode(base,'A_positive',eeg[neg[ix]],post)
                    p=denoise(pipe,prompts,z,t,P(A(cp)));n=denoise(pipe,prompts,z,t,P(A(cn)))
                    lp=(p.float()-noise).square().mean((1,2,3));ln=(n.float()-noise).square().mean((1,2,3))
                gaps.append((ln-lp).cpu())
        gaps=torch.cat(gaps);tau=max(float(gaps.square().mean().sqrt()),1e-4)
        save(ROOT/'energy_temperature.pt',dict(tau=tau,paired_gaps=gaps,T_positions=probe,source='Fixed native fullT encoder,64Tconcepts x4 stimuli, uniformt, same image/noise',not_an_exact_log_likelihood=True))
        del base,p,n,lp,ln,cp,cn,gaps;gc.collect();torch.cuda.empty_cache()
    tau=float(torch.load(ROOT/'energy_temperature.pt',map_location='cpu',weights_only=True)['tau']);assert tau>=1e-4
    frozen=[p for mod in [pipe.unet,A,P] for p in mod.parameters()];versions=[p._version for p in frozen]
    common=dict(policy_sha256=sha(policy_path),source_pins=pins,raw_EEG_sha256=sha(raw_path),latent_sha256=sha(latent_path),native_parent_sha256=sha(parent/'full/brain_final.pt'),
        native_target_sha256=sha(parent/'deterministic_F_all16540.pt'),order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),negative_cycle_sha256=hashlib.sha256(neg.cpu().numpy().tobytes()).hexdigest(),
        statistics_sha256=sha(ROOT/'spatial_statistics.pt'),temperature_sha256=sha(ROOT/'energy_temperature.pt'),tau=tau,host=host,unused_VAE_and_text_offloaded=True,
        training_scope='13230 T only, no D/V inputs or losses; stored full feature container includes other training rows but only T selected',normal_query_inputs='Own EEG only',new_claim_or_SOTA=False)
    if (ROOT/'shared_provenance.json').exists():assert json.loads((ROOT/'shared_provenance.json').read_text())==common
    else:write(ROOT/'shared_provenance.json',common)
    # Case runner is defined below. All numerical work stays inside this Slurm job.
    def run_case(case,limit):
        folder=ROOT/case
        if args.profile_all:folder.mkdir()
        else:assert folder.is_dir() and (folder/'state.pt').exists()
        assert not (folder/'completion.json').exists();torch.manual_seed(1729);model=make_model(case,stats)
        pars=[p for p in model.parameters() if p.requires_grad];lr=1e-5 if case.startswith('A') else 1e-4;optimizer=torch.optim.AdamW(pars,lr=lr,weight_decay=.01)
        gen=torch.Generator(device='cuda').manual_seed(seed('training_noise'));cursor=0;updates=0;history=[];coverage=torch.zeros(13230,dtype=torch.bool);prior_seconds=0.
        if not args.profile_all:
            s=torch.load(folder/'state.pt',map_location='cpu',weights_only=False);assert s['provenance_sha256']==sha(ROOT/'shared_provenance.json') and s['case']==case
            model.load_state_dict(s['model'],strict=True);optimizer.load_state_dict(s['optimizer']);cursor=s['cursor'];updates=s['updates'];history=s['history'];coverage=s['coverage'];prior_seconds=s['seconds']
            gen.set_state(s['noise_RNG']);torch.set_rng_state(s['CPU_RNG']);torch.cuda.set_rng_state_all(s['CUDA_RNG']);del s
        else:
            # Same batch shape and FP32 execution as the original function.
            native=make_model('A_positive',stats).eval().requires_grad_(False)
            with torch.no_grad():
                old=native(eeg[:16,post,:]);new=encode(model,case,eeg[:16],post)
                assert torch.equal(old,new),'Zero initialization must preserve the actual native function'
            del native,old,new
        assert cursor==int(coverage.sum()) and updates==len(history) and updates<limit
        begin=time.monotonic();torch.cuda.reset_peak_memory_stats();last=time.monotonic()
        def snapshot():
            assert [p._version for p in frozen]==versions and all(p.grad is None for p in frozen)
            save(folder/'state.pt',dict(case=case,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),cursor=cursor,updates=updates,
                coverage=coverage,history=history,noise_RNG=gen.get_state(),CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),seconds=prior_seconds+time.monotonic()-begin,
                provenance_sha256=sha(ROOT/'shared_provenance.json')))
            write(folder/'checkpoint_receipt.json',dict(case=case,updates=updates,exposures=cursor,checkpoint_sha256=sha(folder/'state.pt'),job_id=os.environ['SLURM_JOB_ID']))
        with (folder/('attempt_'+os.environ['SLURM_JOB_ID']+'.jsonl')).open('x',buffering=1) as log:
            for offset in range(cursor,13230,16):
                if not args.profile_all and time.monotonic()-started>1000:break
                idx=order[offset:offset+16];ix=idx.cuda();wrong=neg[ix];n=len(ix);noise=torch.randn(latents[ix].shape,device='cuda',generator=gen);t=torch.randint(0,1000,(n,),device='cuda',generator=gen);z=scheduler.add_noise(latents[ix],noise,t)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    c=encode(model,case,eeg[ix],post,wrong=eeg[wrong] if case=='B_shuffle' else None);tokens=P(A(c));pred=denoise(pipe,prompts,z,t,tokens)
                    lp=(pred.float()-noise).square().mean((1,2,3))
                    with torch.no_grad():teacher=P(A(target[ix]))
                    token=(1-F.cosine_similarity(tokens.float(),teacher.float(),dim=-1)).mean();aux=lp.new_zeros(());ln=lp.detach()
                    if case.startswith('A'):
                        cn=encode(model,case,eeg[wrong],post);pn=denoise(pipe,prompts,z,t,P(A(cn)));ln=(pn.float()-noise).square().mean((1,2,3))
                        sign=-1. if case=='A_reverse' else 1.;weight=0. if case=='A_positive' else 1.
                        aux=weight*tau*F.softplus(sign*(lp-ln)/tau).mean()
                    loss=lp.mean()+.1*token+aux
                assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(pars,1.,error_if_nonfinite=True);assert grad>0
                optimizer.step();assert not coverage[idx].any();coverage[idx]=True;updates+=1;cursor=offset+n
                peak=torch.cuda.max_memory_allocated();assert peak<20<<30
                event=dict(case=case,update=updates,offset=offset,n=n,exposures=cursor,loss=float(loss.detach()),positive_energy=float(lp.detach().mean()),negative_energy=float(ln.detach().mean()),
                    token_error=float(token.detach()),contrastive=float(aux.detach()),grad_norm=float(grad),seconds=prior_seconds+time.monotonic()-begin,peak_GPU_allocated_bytes=peak)
                history.append(event);log.write(json.dumps(event)+'\n')
                if updates<=2 or updates%64==0 or cursor==13230:write(folder/'progress.json',event);print(json.dumps(event),flush=True)
                if updates>=limit:break
                if updates%64==0 or time.monotonic()-last>120:snapshot();last=time.monotonic()
        snapshot()
        if args.profile_all:
            import shutil
            shutil.copyfile(folder/'state.pt',folder/'prefix2.pt');assert sha(folder/'prefix2.pt')==sha(folder/'state.pt')
            if case.startswith('B'):
                with torch.no_grad():
                    W,b=model.folded_projection();x=eeg[:16];actual=model.base.proj_in(x[:,post,:].reshape(16,-1))+model.innovation(model.residual(x).reshape(16,-1));fold=F.linear(x.reshape(16,-1),W,b)
                    assert torch.allclose(actual,fold,atol=5e-5,rtol=3e-4);write(folder/'folding_check.json',dict(max_abs=float((actual-fold).abs().max()),pixels_replayed=False,inference_uses_factored_form=True))
        if cursor==13230:
            assert updates==827 and coverage.all();raw=torch.load(raw_path,map_location='cpu',weights_only=True,mmap=True);rows=ids+split['image_rows']['D'];values=[]
            with torch.no_grad():
                for lo in range(0,len(rows),256):values.append(encode(model,case,raw[rows[lo:lo+256]].cuda(),post).float().cpu())
            values=torch.cat(values);assert values.shape==(14880,1024) and torch.isfinite(values).all();save(folder/'full_conditions.pt',dict(condition=values,array_rows=torch.tensor(rows),subject='sub-03',role='full_T_candidate_not_OOF',case=case,checkpoint_sha256=sha(folder/'state.pt')))
        result=dict(status='COMPLETE_FIXED_FULL_T_CANDIDATE_FIT' if cursor==13230 else 'SAFE_COMMITTED_CANDIDATE_PREFIX',case=case,updates=updates,exposures=cursor,expected_updates=827,
            checkpoint_sha256=sha(folder/'state.pt'),seconds=prior_seconds+time.monotonic()-begin,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),new_images=0,ordinary_changed=False)
        write(folder/('attempt_'+os.environ['SLURM_JOB_ID']+'_completion.json'),result)
        if cursor==13230:write(folder/'completion.json',result)
        print(json.dumps(result),flush=True);return result
    results=[]
    for case in CASES if args.profile_all else [args.case]:
        results.append(run_case(case,2 if args.profile_all else 827));gc.collect();torch.cuda.empty_cache()
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<4<<30
    if args.profile_all:write(ROOT/'profile_completion.json',dict(status='PASS_ALL6_REAL_TWO_UPDATE_PREFIXES_RESOURCE_AND_ZERO_FUNCTION',job_id=os.environ['SLURM_JOB_ID'],cases=results,seconds=time.monotonic()-started,new_images=0,prefix_updates_count_once=True))
except BaseException as e:
    write(ROOT/('failure_'+os.environ['SLURM_JOB_ID']+'.json'),dict(status='FAILED_PRESERVED',case=args.case,profile=args.profile_all,error=repr(e),traceback=traceback.format_exc()));raise
