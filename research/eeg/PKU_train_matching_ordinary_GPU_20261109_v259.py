"""Matching ordinary for rebuilt sub03: unchanged objective, auditable finite prefixes."""
import argparse,hashlib,json,math,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--stop-after-updates',type=int,default=2481);args=ap.parse_args()
assert args.stop_after_updates in [4,2481]
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/HIVE_matching_ordinary_sub03_v259';started=time.monotonic()
if args.resume:assert out.is_dir() and (out/'resume.pt').exists()
else:out.mkdir()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(n,v):
    b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();p=out/n;t=p.with_name(p.name+'.pending')
    with t.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
def seed(namespace):return int(hashlib.sha256(('1729|BrainHIVE_bounded_v1|'+namespace).encode()).hexdigest()[:16],16)%(2**63-1)
try:
    import shutil
    assert shutil.disk_usage(out).free>2<<30
    import numpy as np
    import torch
    from torch.nn import functional as F
    from diffusers import DDPMScheduler
    from PKU_native_generator_20261109_v253 import load_native,denoise
    from PKU_ordinary_component_source_v253 import BoundedConditionAdapter
    torch.set_num_threads(4);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and not (out/'training_completion.json').exists()
    policy_path=R/'code/matching_ordinary_policy_v259.json';policy=json.loads(policy_path.read_text())
    assert policy['subject']=='sub-03' and policy['passes']==3 and policy['batch_size']==16 and policy['expected_updates']==2481
    for n,digest in policy['source_sha256'].items():assert sha(R/'code'/n)==digest,n
    baseline=R/'runs/HIVE_native_sub03_5096542';bank=torch.load(baseline/'T_true_OOF_conditions.pt',map_location='cpu',weights_only=True)
    split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());ids=split['image_rows']['T']
    assert bank['subject']=='sub-03' and bank['array_rows'].tolist()==ids and bank['condition'].shape==(13230,1024)
    audit_marker=json.loads((R/'manifests/native_T_latents_audit_job_v258.json').read_text());audit_path=R/('runs/T_latent_independent_audit_'+audit_marker['job_id']+'/completion.json');proof=json.loads(audit_path.read_text())
    assert proof['status']=='PASS_ALL13230_NATIVE_LATENTS_SHA_COVERAGE_AND_TWO_CPU_REPLAYS'
    latent_path=R/'data/native_T_latents_v256/native_VAE_T_modes.npy';assert sha(latent_path)==proof['bank_sha256']
    latent=np.load(latent_path,mmap_mode='r');assert latent.shape==(13230,4,64,64) and latent.dtype==np.float32
    image_latents=torch.tensor(latent,device='cuda');del latent
    condition=bank['condition'].cuda();folds=bank['folds'].long().cuda();assert torch.equal(folds.unique().cpu(),torch.tensor([0,1]))
    fullF=torch.load(baseline/'deterministic_F_all16540.pt',map_location='cpu',weights_only=True);target=fullF[ids].clone().cuda();del fullF
    affine=[];parent_shas={}
    for role in ['oof0','oof1']:
        path=baseline/role/'brain_final.pt';parent_shas[role]=sha(path);state=torch.load(path,map_location='cpu',weights_only=True)['model']
        affine.append((state['block.layers.0.norm.weight'],state['block.layers.0.norm.bias']))
    weight=torch.stack([a[0] for a in affine]).cuda();bias=torch.stack([a[1] for a in affine]).cuda();del state,affine
    pipe,A,P,prompts,host=load_native();assert float(pipe.unet.config.dropout)==0
    pipe.unet.train();pipe.unet.enable_gradient_checkpointing();prompts=tuple(x.to(torch.bfloat16) for x in prompts)
    scheduler=DDPMScheduler.from_config(pipe.scheduler.config);assert scheduler.config.prediction_type=='epsilon' and scheduler.config.num_train_timesteps==1000
    torch.manual_seed(1729);model=BoundedConditionAdapter(predict_risk=False).cuda();assert sum(p.numel() for p in model.parameters())==525568
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    rng=torch.Generator(device='cuda').manual_seed(seed('shared_training_noise'))
    orders=[torch.randperm(13230,generator=torch.Generator().manual_seed(seed('pass'+str(i)))) for i in range(3)]
    provenance=dict(subject='sub-03',parent_root=str(baseline),parent_shas=parent_shas,policy_sha256=sha(policy_path),source_sha256=sha(__file__),
        OOF_condition_sha256=sha(baseline/'T_true_OOF_conditions.pt'),target_sha256=sha(baseline/'deterministic_F_all16540.pt'),latent_sha256=proof['bank_sha256'],latent_audit_sha256=sha(audit_path),
        orders_sha256=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in orders],host=host,train_queries=13230,updates=2481,exposures=39690,training_seed=1729,
        old_ordinary_untouched=True,single_subject_not_all10=True,whole_composite_not_OOF_because_head_trains_allT=True,no_D_V_or_official_test_reads=True)
    if (out/'input_provenance.json').exists():assert json.loads((out/'input_provenance.json').read_text())==provenance
    else:write('input_provenance.json',provenance)
    cursor=0;updates=0;seen=0;loss_sum=0.;prior_seconds=0.;history=[];coverage=torch.zeros((3,13230),dtype=torch.bool)
    if args.resume:
        s=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        assert s['input_provenance_sha256']==sha(out/'input_provenance.json')
        model.load_state_dict(s['model'],strict=True);optimizer.load_state_dict(s['optimizer'])
        cursor=s['cursor'];updates=s['updates'];seen=s['seen'];loss_sum=s['loss_sum'];coverage=s['coverage'];history=s['history'];prior_seconds=s['seconds']
        rng.set_state(s['noise_rng']);torch.set_rng_state(s['CPU_RNG']);torch.cuda.set_rng_state_all(s['CUDA_RNG']);del s
    assert cursor==seen and int(coverage.sum())==seen and len(history)==updates and updates<args.stop_after_updates
    frozen=[p for mod in [pipe.unet,A,P] for p in mod.parameters()];versions=[p._version for p in frozen]
    assert all(not p.requires_grad for p in frozen)
    torch.cuda.reset_peak_memory_stats();last_save=time.monotonic()
    def snapshot():
        assert [p._version for p in frozen]==versions and all(p.grad is None for p in frozen)
        state=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),cursor=cursor,updates=updates,seen=seen,
            coverage=coverage,loss_sum=loss_sum,history=history,noise_rng=rng.get_state(),CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),
            seconds=prior_seconds+time.monotonic()-started,input_provenance_sha256=sha(out/'input_provenance.json'))
        p=out/'resume.pt';tmp=out/'resume.pending';torch.save(state,tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        tmp.replace(p)
        write('checkpoint_receipt.json',dict(updates=updates,cursor=cursor,seen=seen,checkpoint_sha256=sha(p),checkpoint_bytes=p.stat().st_size,seconds=state['seconds'],job_id=os.environ['SLURM_JOB_ID']))
    reason='FINITE_TIME_PREFIX'
    with (out/('attempt_'+os.environ['SLURM_JOB_ID']+'.jsonl')).open('x',buffering=1) as log:
        stop=False
        for epoch in range(cursor//13230,3):
            start_offset=cursor%13230 if epoch==cursor//13230 else 0
            for offset in range(start_offset,13230,16):
                if time.monotonic()-started>1000:stop=True;break
                idx=orders[epoch][offset:offset+16];ix=idx.cuda();n=len(idx);w=weight[folds[ix]];b=bias[folds[ix]];c=condition[ix];clean=image_latents[ix]
                noise=torch.randn(clean.shape,device='cuda',generator=rng);t=torch.randint(0,1000,(n,),device='cuda',generator=rng);noisy=scheduler.add_noise(clean,noise,t)
                optimizer.zero_grad(set_to_none=True);changed,info=model(c,w,b)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    with torch.no_grad():real=P(A(target[ix]))
                    tokens=P(A(changed));pred=denoise(pipe,prompts,noisy,t,tokens)
                    image_loss=F.mse_loss(pred.float(),noise);token_loss=(1-F.cosine_similarity(tokens.float(),real.float(),dim=-1)).mean();loss=image_loss+.1*token_loss
                assert torch.isfinite(loss);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                assert model.direction.weight.grad is not None and all(p.grad is None for p in frozen)
                direction_grad=float(model.direction.weight.grad.norm());assert direction_grad>0
                optimizer.step();assert not coverage[epoch,idx].any();coverage[epoch,idx]=True;cursor=epoch*13230+offset+n;updates+=1;seen+=n;loss_sum+=float(loss.detach())*n
                peak=torch.cuda.max_memory_allocated();assert peak<20<<30
                event=dict(update=updates,pass_number=epoch+1,offset=offset,n=n,seen=seen,loss=float(loss.detach()),epsilon_loss=float(image_loss.detach()),token_loss=float(token_loss.detach()),gradient_norm=float(gradient),direction_weight_gradient_norm=direction_grad,
                    correction_RMS=float(info['correction'].square().mean().sqrt()),seconds=prior_seconds+time.monotonic()-started,peak_GPU_allocated_bytes=peak)
                history.append(event);log.write(json.dumps(event)+'\n')
                if updates<=4 or updates%64==0 or offset+n==13230:write('progress.json',dict(status='RUNNING',expected_updates=2481,expected_exposures=39690,**event));print(json.dumps(event),flush=True)
                if updates>=args.stop_after_updates:reason='REQUESTED_FIXED_PREFIX' if updates<2481 else 'COMPLETE_THREE_PASSES';stop=True;break
                if updates%64==0 or time.monotonic()-last_save>120:snapshot();last_save=time.monotonic()
            if stop:break
    snapshot()
    if updates==4:
        p=out/'first4_committed.pt';assert not p.exists();shutil.copyfile(out/'resume.pt',p);assert sha(p)==sha(out/'resume.pt')
    output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file());assert output_bytes<policy['output_cap_bytes']
    result=dict(status='COMPLETE_SINGLE_SUB03_T_THREE_PASSES_PRECALIBRATION' if updates==2481 else 'SAFE_COMMITTED_ORDINARY_TRAINING_PREFIX',reason=reason,
        job_id=os.environ['SLURM_JOB_ID'],updates=updates,expected_updates=2481,seen=seen,expected_seen=39690,seconds=prior_seconds+time.monotonic()-started,
        checkpoint_sha256=sha(out/'resume.pt'),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),output_bytes_before_receipt=output_bytes,old_ordinary_changed=False,new_efficacy_images=0,
        current_fit_is_matched_baseline_not_second_innovation=True,calibration_and_image_evaluation_complete=False)
    if updates==2481:
        assert seen==39690 and coverage.all();write('training_completion.json',result)
    write('attempt_'+os.environ['SLURM_JOB_ID']+'_completion.json',result);write('progress.json',result);print(json.dumps(result),flush=True)
except BaseException as e:
    write('failure_'+os.environ['SLURM_JOB_ID']+'.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
