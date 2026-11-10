"""One fixed T pass, same graph across objective controls; no checkpoint selection."""
import argparse,copy,gc,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');ap.add_argument('--case',required=True);a=ap.parse_args()
from PKU_conditional_preservation_core_20261110_v317 import R,ROOT,CASES,flags,sha,write,save,states,load_states,fixed_inputs,constrain,check_fixture
ROOT.mkdir(exist_ok=True);start=time.monotonic();jid=os.environ['SLURM_JOB_ID'];folder=ROOT/a.case
try:
    import numpy as np
    import torch
    from torch.nn import functional as F
    from diffusers import DDPMScheduler
    from PKU_native_generator_20261109_v253 import load_native,denoise
    from PKU_eeg_candidate_core_20261109_v266 import seed,negative_cycle
    assert a.case in CASES and torch.cuda.device_count()==1;torch.set_num_threads(4);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    policy=R/'code/EEG_preservation_policy_v317.json';spec=json.loads(policy.read_text());assert spec['cases']==CASES
    pins=json.loads((R/'code/EEG_preservation_pins_v317.json').read_text())
    for n,h in pins.items():assert sha(R/'code'/n)==h
    if a.profile:folder.mkdir()
    else:assert folder.is_dir() and (folder/'state.pt').exists()
    assert not (folder/'completion.json').exists()
    c,target,source=fixed_inputs('cuda');latpath=R/'data/native_T_latents_v256/native_VAE_T_modes.npy';assert sha(latpath)=='b1198b3e07361e3b941f87e3b60146b1cc7957293e813db69f7b59cfa8469a2d'
    latent=torch.tensor(np.load(latpath,mmap_mode='r'),device='cuda');assert latent.shape==(13230,4,64,64)
    ids=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text())['image_rows']['T'];negative,_=negative_cycle(ids);negative=negative.cuda()
    order=torch.randperm(13230,generator=torch.Generator().manual_seed(seed('order')))
    taupath=R/'runs/EEG_candidates_v267/energy_temperature.pt';tau=float(torch.load(taupath,map_location='cpu',weights_only=True)['tau']);assert tau>=1e-4
    pipe,A,P,prompts,host=load_native();pipe.unet.train();pipe.unet.enable_gradient_checkpointing();assert float(pipe.unet.config.dropout)==0
    for name in ['vae','text_encoder','text_encoder_2']:getattr(pipe,name).cpu()
    prompts=tuple(x.to(torch.bfloat16) for x in prompts)
    P.float().eval().requires_grad_(False);P.proj.weight.requires_grad_(True)
    refP=copy.deepcopy(P).requires_grad_(False).eval();refprocs=None
    modules={'P':P};refs={'P':refP};pars=[P.proj.weight];initial=states(refs)
    assert sum(p.numel() for p in pars)==8388608
    fixture=check_fixture();sums=torch.zeros(1024,dtype=torch.float64)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for lo in range(0,13230,256):sums+=A(c[lo:lo+256]).double().cpu().sum(0)
    mu=(sums/13230).float();g_anchor=torch.Generator().manual_seed(1731)
    q_anchor=torch.randn(1024,generator=g_anchor);q_anchor=q_anchor/q_anchor.norm()*mu.norm()
    kind=flags(a.case)['anchor'];anchor_cpu=mu if kind=='mean' else q_anchor if kind=='random' else None
    anchor=None if anchor_cpu is None else anchor_cpu.cuda()
    w0=refP.proj.weight.detach()
    anchor_spec=dict(kind=kind,mu=mu,random_direction=q_anchor,fixture=fixture)
    if (folder/'anchor.pt').exists():
        prior_anchor=torch.load(folder/'anchor.pt',weights_only=True,map_location='cpu')
        assert prior_anchor['kind']==kind and torch.equal(prior_anchor['mu'],mu) and torch.equal(prior_anchor['random_direction'],q_anchor)
    else:save(folder/'anchor.pt',anchor_spec)
    targetids={id(p) for p in pars};frozen=[p for m in [pipe.unet,A,P]+list(refs.values()) for p in m.parameters() if id(p) not in targetids];versions=[p._version for p in frozen]
    assert all(not p.requires_grad for p in frozen);torch.cuda.empty_cache();optimizer=torch.optim.AdamW(pars,lr=1e-5,weight_decay=.01)
    gen=torch.Generator(device='cuda').manual_seed(seed('training_noise'));scheduler=DDPMScheduler.from_config(pipe.scheduler.config)
    common=dict(case=a.case,source=source,policy_sha256=sha(policy),source_pins=pins,latent_sha256=sha(latpath),tau_source_sha256=sha(taupath),tau=tau,order_sha256=__import__('hashlib').sha256(order.numpy().tobytes()).hexdigest(),anchor_sha256=sha(folder/'anchor.pt'),host=host,trainable_parameters=sum(p.numel() for p in pars),training_scope='T13230 only, fixed OOF EEG+ordinary conditions; no EEG/ordinary updates; no D inputs',model_family='P' if a.case.startswith('P_') else 'IP')
    if (folder/'provenance.json').exists():assert json.loads((folder/'provenance.json').read_text())==common
    else:write(folder/'provenance.json',common)
    cursor=0;updates=0;history=[];coverage=torch.zeros(13230,dtype=torch.bool);prior=0.;identity=None
    if not a.profile:
        s=torch.load(folder/'state.pt',map_location='cpu',weights_only=False);assert s['provenance_sha256']==sha(folder/'provenance.json');load_states(modules,s['model']);optimizer.load_state_dict(s['optimizer']);cursor=s['cursor'];updates=s['updates'];history=s['history'];coverage=s['coverage'];prior=s['seconds'];identity=s['identity'];gen.set_state(s['noise_RNG']);torch.set_rng_state(s['CPU_RNG']);torch.cuda.set_rng_state_all(s['CUDA_RNG']);del s
    else:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            z=latent[:1];t=torch.tensor([499],device='cuda');ts=P(A(c[:1]));pred=denoise(pipe,prompts,z,t,ts)
            if refprocs is not None:
                actual=dict(pipe.unet.attn_processors);pipe.unet.set_attn_processor(dict(refprocs));reference=denoise(pipe,prompts,z,t,refP(A(c[:1])));pipe.unet.set_attn_processor(dict(actual))
            else:reference=denoise(pipe,prompts,z,t,refP(A(c[:1])))
        assert torch.equal(pred,reference);identity=dict(native_target_zero_change_exact=True,probe_t=499,one_T_query=True)
    # load_state_dict increments version counters even for unchanged frozen values.
    # CPU checkpoint audit verifies frozen values; track new mutations after restore.
    versions=[p._version for p in frozen]
    f=flags(a.case);torch.cuda.reset_peak_memory_stats();begin=time.monotonic();last=time.monotonic()
    def snapshot():
        assert [p._version for p in frozen]==versions and all(p.grad is None for p in frozen)
        save(folder/'state.pt',dict(case=a.case,model=states(modules),optimizer=optimizer.state_dict(),cursor=cursor,updates=updates,coverage=coverage,history=history,seconds=prior+time.monotonic()-begin,identity=identity,noise_RNG=gen.get_state(),CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),provenance_sha256=sha(folder/'provenance.json')))
        write(folder/'checkpoint_receipt.json',dict(case=a.case,updates=updates,exposures=cursor,checkpoint_sha256=sha(folder/'state.pt'),job_id=jid))
    with (folder/('attempt_'+jid+'.jsonl')).open('x',buffering=1) as log:
        for offset in range(cursor,13230,16):
            if not a.profile and time.monotonic()-start>950:break
            ix=order[offset:offset+16].cuda();n=len(ix);wrong=negative[ix];noise=torch.randn(latent[ix].shape,device='cuda',generator=gen);t=torch.randint(0,1000,(n,),device='cuda',generator=gen);z=scheduler.add_noise(latent[ix],noise,t)
            optimizer.zero_grad(set_to_none=True);total={k:0. for k in ['positive_energy','negative_energy','token_error','visual_replay','contrastive','EEG_absolute','EEG_relative','EEG_preservation','loss']}
            for lo in range(0,n,4):
                j=ix[lo:lo+4];w=wrong[lo:lo+4];zt=z[lo:lo+4];tt=t[lo:lo+4];eps=noise[lo:lo+4];fraction=len(j)/n
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    with torch.no_grad():
                        ft=refP(A(target[j]))
                        if refprocs is not None:
                            actual=dict(pipe.unet.attn_processors);pipe.unet.set_attn_processor(dict(refprocs));refpred=denoise(pipe,prompts,zt,tt,ft);pipe.unet.set_attn_processor(dict(actual))
                        else:refpred=denoise(pipe,prompts,zt,tt,ft)
                        refpos=denoise(pipe,prompts,zt,tt,refP(A(c[j])))
                        refneg=denoise(pipe,prompts,zt,tt,refP(A(c[w])))
                    tokens=P(A(c[j]));pos=denoise(pipe,prompts,zt,tt,tokens);lp=(pos.float()-eps).square().mean((1,2,3));token=(1-F.cosine_similarity(tokens.float(),ft.float(),dim=-1)).mean()
                    neg=denoise(pipe,prompts,zt,tt,P(A(c[w])));ln=(neg.float()-eps).square().mean((1,2,3))
                    visual=denoise(pipe,prompts,zt,tt,P(A(target[j])));replay=(visual.float()-refpred.float()).square().mean()
                    pair=tau*F.softplus((lp-ln)/tau).mean()
                    dp=pos.float()-refpos.float();dn=neg.float()-refneg.float()
                    absolute=.5*(dp.square().mean()+dn.square().mean())
                    relative=.5*(dp-dn).square().mean()
                    preservation=relative if f['relative'] else absolute
                    loss=lp.mean()+.1*token+replay+pair+preservation
                assert torch.isfinite(loss);(loss*fraction).backward()
                vals=dict(positive_energy=lp.detach().mean(),negative_energy=ln.detach().mean(),token_error=token.detach(),visual_replay=replay.detach(),contrastive=pair.detach(),EEG_absolute=absolute.detach(),EEG_relative=relative.detach(),EEG_preservation=preservation.detach(),loss=loss.detach())
                for k,v in vals.items():total[k]+=float(v)*fraction
                del pos,neg,visual,refpred,refpos,refneg,tokens,loss,lp,ln,token,replay,pair,dp,dn,absolute,relative,preservation
            grad=torch.nn.utils.clip_grad_norm_(pars,1.,error_if_nonfinite=True);assert grad>0;optimizer.step();anchor_residual=constrain(P.proj.weight,w0,anchor);assert anchor_residual<2e-5;updates+=1;cursor=offset+n;assert not coverage[ix.cpu()].any();coverage[ix.cpu()]=True
            peak=torch.cuda.max_memory_allocated();assert peak<20<<30
            event=dict(case=a.case,anchor_constraint_max_abs=anchor_residual,update=updates,offset=offset,n=n,exposures=cursor,grad_norm=float(grad),peak_GPU_allocated_bytes=peak,seconds=prior+time.monotonic()-begin,**total);history.append(event);log.write(json.dumps(event)+'\n')
            if updates<=2 or updates%64==0 or cursor==13230:write(folder/'progress.json',event);print(json.dumps(event),flush=True)
            if a.profile and updates==2:break
            if updates%64==0 or time.monotonic()-last>120:snapshot();last=time.monotonic()
        snapshot()
    if a.profile:save(folder/'prefix2_model.pt',dict(model=states(modules),updates=updates,checkpoint_sha256=sha(folder/'state.pt')))
    if cursor==13230:
        total_shift=torch.zeros(4,2048,dtype=torch.float64);delta_energy=0.;samples=0
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for lo in range(0,13230,128):
                features=A(c[lo:lo+128]);delta=(P(features).float()-refP(features).float()).double().cpu()
                total_shift+=delta.sum(0);delta_energy+=float(delta.square().sum());samples+=len(delta)
        mean_shift=total_shift/samples
        write(folder/'T_token_drift.json',dict(T_queries=samples,token_mean_shift_squared=float(mean_shift.square().mean()),total_token_delta_squared=delta_energy/(samples*4*2048),anchor_does_not_guarantee_mean_after_LN=True))
        assert updates==827 and coverage.all();save(folder/'target_final.pt',dict(case=a.case,model=states(modules),updates=updates,checkpoint_sha256=sha(folder/'state.pt')))
        write(folder/'completion.json',dict(status='COMPLETE_FIXED_T_EXISTING_DOWNSTREAM_WEIGHT_FIT',case=a.case,updates=updates,exposures=cursor,checkpoint_sha256=sha(folder/'state.pt'),target_sha256=sha(folder/'target_final.pt'),new_images=0))
    result=dict(status='SAFE_DOWNSTREAM_TWO_UPDATE_PREFIX' if a.profile else 'COMPLETE_FIXED_T_FIT' if cursor==13230 else 'SAFE_FIXED_T_PREFIX',case=a.case,updates=updates,exposures=cursor,seconds=prior+time.monotonic()-begin,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),source_sha256=sha(__file__),EEG_and_ordinary_unchanged=True,new_images=0)
    write(folder/('attempt_'+jid+'_completion.json'),result);print(json.dumps(result),flush=True)
except BaseException as e:
    write(ROOT/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED_NO_AUTO_RETRY',case=a.case,error=repr(e),traceback=traceback.format_exc()));raise
