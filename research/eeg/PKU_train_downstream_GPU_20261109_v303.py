"""One fixed T pass, same graph across objective controls; no checkpoint selection."""
import argparse,gc,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--profile',action='store_true');ap.add_argument('--case',required=True);a=ap.parse_args()
from PKU_downstream_core_20261109_v303 import R,ROOT,CASES,flags,sha,write,save,states,load_states,setup_target,fixed_inputs
ROOT.mkdir(exist_ok=True);start=time.monotonic();jid=os.environ['SLURM_JOB_ID'];folder=ROOT/a.case
try:
    import numpy as np
    import torch
    from torch.nn import functional as F
    from diffusers import DDPMScheduler
    from PKU_native_generator_20261109_v253 import load_native,denoise
    from PKU_eeg_candidate_core_20261109_v266 import seed,negative_cycle
    assert a.case in CASES and torch.cuda.device_count()==1;torch.set_num_threads(4);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    policy=R/'code/EEG_downstream_policy_v303.json';spec=json.loads(policy.read_text());assert spec['cases']==CASES
    pins=json.loads((R/'code/EEG_downstream_pins_v303.json').read_text())
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
    prompts=tuple(x.to(torch.bfloat16) for x in prompts);modules,refs,refP,refprocs,pars=setup_target(pipe,P,a.case);initial=states(refs)
    targetids={id(p) for p in pars};frozen=[p for m in [pipe.unet,A,P]+list(refs.values()) for p in m.parameters() if id(p) not in targetids];versions=[p._version for p in frozen]
    assert all(not p.requires_grad for p in frozen);torch.cuda.empty_cache();optimizer=torch.optim.AdamW(pars,lr=1e-5,weight_decay=.01)
    gen=torch.Generator(device='cuda').manual_seed(seed('training_noise'));scheduler=DDPMScheduler.from_config(pipe.scheduler.config)
    common=dict(case=a.case,source=source,policy_sha256=sha(policy),source_pins=pins,latent_sha256=sha(latpath),tau_source_sha256=sha(taupath),tau=tau,order_sha256=__import__('hashlib').sha256(order.numpy().tobytes()).hexdigest(),host=host,trainable_parameters=sum(p.numel() for p in pars),training_scope='T13230 only, fixed OOF EEG+ordinary conditions; no EEG/ordinary updates; no D inputs',model_family='P' if a.case.startswith('P_') else 'IP')
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
    f=flags(a.case);torch.cuda.reset_peak_memory_stats();begin=time.monotonic();last=time.monotonic()
    def snapshot():
        assert [p._version for p in frozen]==versions and all(p.grad is None for p in frozen)
        save(folder/'state.pt',dict(case=a.case,model=states(modules),optimizer=optimizer.state_dict(),cursor=cursor,updates=updates,coverage=coverage,history=history,seconds=prior+time.monotonic()-begin,identity=identity,noise_RNG=gen.get_state(),CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),provenance_sha256=sha(folder/'provenance.json')))
        write(folder/'checkpoint_receipt.json',dict(case=a.case,updates=updates,exposures=cursor,checkpoint_sha256=sha(folder/'state.pt'),job_id=jid))
    with (folder/('attempt_'+jid+'.jsonl')).open('x',buffering=1) as log:
        for offset in range(cursor,13230,16):
            if not a.profile and time.monotonic()-start>950:break
            ix=order[offset:offset+16].cuda();n=len(ix);wrong=negative[ix];noise=torch.randn(latent[ix].shape,device='cuda',generator=gen);t=torch.randint(0,1000,(n,),device='cuda',generator=gen);z=scheduler.add_noise(latent[ix],noise,t)
            optimizer.zero_grad(set_to_none=True);total={k:0. for k in ['positive_energy','negative_energy','token_error','visual_replay','contrastive','loss']}
            for lo in range(0,n,4):
                j=ix[lo:lo+4];w=wrong[lo:lo+4];zt=z[lo:lo+4];tt=t[lo:lo+4];eps=noise[lo:lo+4];fraction=len(j)/n
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    with torch.no_grad():
                        ft=refP(A(target[j]))
                        if refprocs is not None:
                            actual=dict(pipe.unet.attn_processors);pipe.unet.set_attn_processor(dict(refprocs));refpred=denoise(pipe,prompts,zt,tt,ft);pipe.unet.set_attn_processor(dict(actual))
                        else:refpred=denoise(pipe,prompts,zt,tt,ft)
                    tokens=P(A(c[j]));pos=denoise(pipe,prompts,zt,tt,tokens);lp=(pos.float()-eps).square().mean((1,2,3));token=(1-F.cosine_similarity(tokens.float(),ft.float(),dim=-1)).mean()
                    neg=denoise(pipe,prompts,zt,tt,P(A(c[w])));ln=(neg.float()-eps).square().mean((1,2,3))
                    visual=denoise(pipe,prompts,zt,tt,P(A(target[j])));replay=(visual.float()-refpred.float()).square().mean()
                    pair=tau*F.softplus(f['sign']*(lp-ln)/tau).mean();loss=lp.mean()+.1*token+f['replay']*replay+f['pair']*pair
                assert torch.isfinite(loss);(loss*fraction).backward()
                vals=dict(positive_energy=lp.detach().mean(),negative_energy=ln.detach().mean(),token_error=token.detach(),visual_replay=replay.detach(),contrastive=pair.detach(),loss=loss.detach())
                for k,v in vals.items():total[k]+=float(v)*fraction
                del pos,neg,visual,refpred,tokens,loss,lp,ln,token,replay,pair
            grad=torch.nn.utils.clip_grad_norm_(pars,1.,error_if_nonfinite=True);assert grad>0;optimizer.step();updates+=1;cursor=offset+n;assert not coverage[ix.cpu()].any();coverage[ix.cpu()]=True
            peak=torch.cuda.max_memory_allocated();assert peak<20<<30
            event=dict(case=a.case,update=updates,offset=offset,n=n,exposures=cursor,grad_norm=float(grad),peak_GPU_allocated_bytes=peak,seconds=prior+time.monotonic()-begin,**total);history.append(event);log.write(json.dumps(event)+'\n')
            if updates<=2 or updates%64==0 or cursor==13230:write(folder/'progress.json',event);print(json.dumps(event),flush=True)
            if a.profile and updates==2:break
            if updates%64==0 or time.monotonic()-last>120:snapshot();last=time.monotonic()
        snapshot()
    if a.profile:save(folder/'prefix2_model.pt',dict(model=states(modules),updates=updates,checkpoint_sha256=sha(folder/'state.pt')))
    if cursor==13230:
        assert updates==827 and coverage.all();save(folder/'target_final.pt',dict(case=a.case,model=states(modules),updates=updates,checkpoint_sha256=sha(folder/'state.pt')))
        write(folder/'completion.json',dict(status='COMPLETE_FIXED_T_EXISTING_DOWNSTREAM_WEIGHT_FIT',case=a.case,updates=updates,exposures=cursor,checkpoint_sha256=sha(folder/'state.pt'),target_sha256=sha(folder/'target_final.pt'),new_images=0))
    result=dict(status='SAFE_DOWNSTREAM_TWO_UPDATE_PREFIX' if a.profile else 'COMPLETE_FIXED_T_FIT' if cursor==13230 else 'SAFE_FIXED_T_PREFIX',case=a.case,updates=updates,exposures=cursor,seconds=prior+time.monotonic()-begin,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),source_sha256=sha(__file__),EEG_and_ordinary_unchanged=True,new_images=0)
    write(folder/('attempt_'+jid+'_completion.json'),result);print(json.dumps(result),flush=True)
except BaseException as e:
    write(ROOT/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED_NO_AUTO_RETRY',case=a.case,error=repr(e),traceback=traceback.format_exc()));raise
