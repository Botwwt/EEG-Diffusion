"""Frozen trained native-P functional decomposition, full T statistics, parity images."""
import hashlib,json,os,shutil,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');OUT=R/'runs/EEG_P_intervention_tokens_v316';OUT.mkdir(exist_ok=False);start=time.monotonic()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,v):
    b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();p=Path(p);t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    t.replace(p)

try:
    import numpy as np
    import torch
    from PIL import Image
    from PKU_native_generator_20261109_v253 import load_native
    from PKU_downstream_core_20261109_v303 import fixed_inputs,load_states,target_modules
    assert torch.cuda.device_count()==1 and shutil.disk_usage(R).free>100<<30
    torch.set_num_threads(4);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    specpath=R/'code/EEG_P_intervention_policy_v316.json';spec=json.loads(specpath.read_text())
    cases=spec['fixed_trained_cases'];assert cases==['P_positive','P_replay_pair']
    done=json.loads((R/'runs/EEG_downstream_scores_v303/analysis_completion.json').read_text());assert done['queries']==830 and done['arms']==29
    oldroot=R/'runs/EEG_downstream_images_v303'
    old=torch.load(oldroot/'condition_bank.pt',weights_only=True,map_location='cpu');assert old['condition'].shape==(9,830,1024)
    assert all(torch.equal(old['condition'][i],old['condition'][0]) for i in range(9))
    cT,_,source=fixed_inputs('cuda');cD=old['condition'][0].cuda()
    pipe,A,P,prompts,host=load_native();assert pipe._execution_device.type=='cuda'
    base={k:v.detach().cpu().clone() for k,v in P.state_dict().items()}
    path=OUT/'all_T_native_and_trained_tokens.npy'
    table=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(3,13230,4,2048))
    d_tokens=[];models=[];sums=[]
    for mi,case in enumerate(['native']+cases):
        if case=='native':P.load_state_dict(base,strict=True)
        else:
            ck=R/'runs/EEG_downstream_v303'/case/'target_final.pt'
            proof=json.loads(ck.with_name('completion.json').read_text());assert sha(ck)==proof['target_sha256']
            state=torch.load(ck,weights_only=True,map_location='cpu');load_states(target_modules(pipe,P,case),state['model'])
            models.append(dict(case=case,path=str(ck),sha256=sha(ck)))
        total=np.zeros((4,2048),dtype=np.float64)
        with torch.inference_mode():
            for lo in range(0,13230,128):
                a=P(A(cT[lo:lo+128].half())).cpu().numpy();assert a.dtype==np.float16 and np.isfinite(a).all()
                table[mi,lo:lo+len(a)]=a;total+=a.astype(np.float64).sum(0)
                assert time.monotonic()-start<1000 and torch.cuda.max_memory_allocated()<20<<30
            # Use the original one-query P/A arithmetic for pixel parity.
            d=torch.cat([P(A(cD[lo:lo+1].half())).cpu() for lo in range(830)])
            assert d.shape==(830,4,2048) and torch.isfinite(d).all();d_tokens.append(d)
        table.flush();sums.append(total)
    means=torch.from_numpy(np.stack([(sums[i]-sums[0])/13230 for i in [1,2]]).astype(np.float32))
    target=[];arms=[];checks=[]
    for ci,case in enumerate(cases):
        original=d_tokens[0].float();full=d_tokens[ci+1].float();mean=means[ci]
        delta=full-original;residual=delta-mean
        error=float((original+mean+residual-full).abs().max());assert error<2e-6
        common=(original+mean).half();specific=(full-mean).half()
        assert torch.isfinite(common).all() and torch.isfinite(specific).all()
        target.extend([common,specific]);arms.extend([case+'_common',case+'_specific'])
        checks.append(dict(case=case,reconstruction_FP32_max_abs=error,delta_mean_squared=float(mean.square().mean()),mean_is_T_only=True,common_rounding_max_abs=float((common.float()-(original+mean)).abs().max()),specific_rounding_max_abs=float((specific.float()-(full-mean)).abs().max()),token_norm_base=float(original.norm(dim=-1).mean()),token_norm_full=float(full.norm(dim=-1).mean()),token_norm_common=float(common.float().norm(dim=-1).mean()),token_norm_specific=float(specific.float().norm(dim=-1).mean())))
    assert arms==spec['new_image_arms']
    # Zero change and complementary-factor algebra are fixtures, not efficacy.
    z=torch.zeros_like(d_tokens[0]);assert torch.equal((d_tokens[0].float()+z).half(),d_tokens[0])
    bank=dict(arms=arms,tokens=torch.stack(target),condition=old['condition'][0].unsqueeze(0).expand(4,-1,-1).clone(),array_rows=old['array_rows'],stimulus_ids=old['stimulus_ids'],means=means,full_D_tokens=torch.stack(d_tokens),models=models,T_source=source)
    torch.save(bank,OUT/'token_bank.pt')
    # Replay the exact old full trained-P path for the first fixed query of each case.
    parities=[]
    for ci,case in enumerate(cases):
        qi=0;ai=old['arms'].index(case);rec=json.loads((oldroot/'records'/('%05d.json'%(ai*830+qi))).read_text())
        g=torch.Generator(device='cuda').manual_seed(rec['image_seed'])
        with torch.inference_mode():
            im=pipe(prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],width=512,height=512,num_inference_steps=4,guidance_scale=0.,generator=g,cross_attention_kwargs=dict(ip_hidden_states=d_tokens[ci+1][qi:qi+1].cuda()),output_type='pil').images[0].convert('RGB')
        assert hashlib.sha256(im.tobytes()).hexdigest()==rec['pixel_sha256']
        assert hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest()==rec['generator_final_sha256']
        im.save(OUT/(case+'_parity.png'));parities.append(dict(case=case,query=qi,old_pixel_exact=True,old_final_RNG_exact=True,UNet_NFE=4))
    write(OUT/'completion.json',dict(status='PASS_FROZEN_T13230_DECOMPOSITION_AND_TWO_ORIGINAL_IMAGE_PARITIES',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,T_queries=13230,D_queries=830,new_arms=4,models=models,source=source,checks=checks,parities=parities,token_bank_sha256=sha(OUT/'token_bank.pt'),T_tokens_sha256=sha(path),policy_sha256=sha(specpath),source_sha256=sha(__file__),new_training_updates=0,new_research_images=0,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated()))
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())<2<<30
    print('PASS_FULL_T_MEANS_D830_TOKENS_AND_REAL_OLD_PIXELS',flush=True)
except BaseException as e:
    write(OUT/'failure.json',dict(status='FAILED_PRESERVED_NO_AUTO_RETRY',error=repr(e),traceback=traceback.format_exc()));raise
