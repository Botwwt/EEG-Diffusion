"""Full-T mechanism controls of immutable ordinary; no new method claim.

Same 525568-parameter head, initialization, 3 orders and 24807 AdamW steps.
Only the objective differs. No UNet/VAE or Dcheck/test target access here.
Each finite invocation commits one exact resumable transaction.
"""
import argparse, datetime, gc, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):assert ref(x['path'])==x;return Path(x['path'])
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending')
    with tmp.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def run(args):
    s=json.loads(Path(args.protocol).read_bytes());assert s['status']=='REGISTERED_ORDINARY_TWO_FULL_T_LOSS_MECHANISM_CONTROLS'
    assert s['host']=='old' and args.arm in ('F_only','token_only') and 0<=args.segment<8
    for x in s['source_pins']:checked(x)
    checked(s['ordinary']);checked(s['input_contract']);checked(s['ordinary_input_provenance'])
    assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    out=Path(s['output_root'])/args.arm;out.mkdir(parents=True,exist_ok=True);assert not (out/'completion.json').exists()
    start=time.monotonic();base=rss();phase='RUNTIME';sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    import brainhive_native_models_v1 as N
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_data_v1 import load_training
    from train_BrainHIVE_bounded_component_v1 import seed,development
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    from brainhive_functional_training_contract_v2 import resource_headroom
    resources=resource_headroom();assert resources['effective_headroom_bytes']>=10<<30
    def budget():
        assert rss()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        assert shutil.disk_usage(out).free>2<<30 and time.monotonic()-start<830
        assert sum(p.stat().st_size for p in Path(s['output_root']).rglob('*') if p.is_file())<2<<30
    def state_sha(module):
        h=hashlib.sha256()
        for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def save_pt(p,x):
        tmp=p.with_name(p.name+'.pending');torch.save(x,tmp);os.replace(tmp,p)
    try:
        phase='ALL10_REAL_OOF_AND_NATIVE_AP_LOAD';tick=time.monotonic();inputs=load_training()
        assert inputs['input_contract_sha256']==s['input_contract']['sha256'] and inputs['condition'].shape==(132300,1024)
        prior=N.prior_models('cuda');adapter=prior['fusion_encoder'].adapter.half().eval().requires_grad_(False)
        projector=prior['proj'].half().eval().requires_grad_(False);del prior;gc.collect();torch.cuda.empty_cache()
        frozen=dict(A=state_sha(adapter),P=state_sha(projector));load_seconds=time.monotonic()-tick
        torch.manual_seed(1729);model=BoundedConditionAdapter(predict_risk=False).cuda();assert sum(p.numel() for p in model.parameters())==525568
        initial_sha=state_sha(model);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        orders=[torch.randperm(132300,generator=torch.Generator().manual_seed(seed('pass'+str(i)))) for i in range(3)]
        order_sha=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in orders]
        original=json.loads(checked(s['ordinary_input_provenance']).read_bytes());assert order_sha==original['orders_sha256']
        provenance=dict(protocol=ref(args.protocol),arm=args.arm,parameters=525568,initial_model_tensor_sha256=initial_sha,
          orders_sha256=order_sha,input_contract=s['input_contract'],frozen_AP=frozen,seed=1729,
          expected_updates=24807,expected_query_exposures=396900,UNet_calls=0,diffusion_noise_draws=0,
          paired_information_and_optimizer_budget=True,matched_UNet_compute=False,new_innovation_claim=False)
        if (out/'provenance.json').exists():assert json.loads((out/'provenance.json').read_bytes())==provenance
        else:atomic(out/'provenance.json',provenance)
        updates=seen=cursor=0;total_loss=0.;seconds=0.;coverage=torch.zeros(3,132300,dtype=torch.bool)
        if args.segment:
            previous=out/f'segment_{args.segment-1:02d}.json';p=json.loads(previous.read_bytes());assert p['status']=='EXACT_PARTIAL_FULL_T_CONTROL_COMMITTED'
            old=torch.load(checked(p['checkpoint']),weights_only=False,map_location='cpu')
            assert old['protocol_sha256']==ref(args.protocol)['sha256'] and old['arm']==args.arm
            model.load_state_dict(old['model'],strict=True);optimizer.load_state_dict(old['optimizer'])
            updates,seen,cursor,total_loss,coverage,seconds=[old[k] for k in ('updates','seen','cursor','total_loss','coverage','training_seconds')]
            torch.set_rng_state(old['torch_rng']);torch.cuda.set_rng_state_all(old['cuda_rng']);del old
        else:assert not list(out.glob('checkpoint_segment_*.pt'))
        initial_updates=updates;tick=time.monotonic();phase='SCIENTIFIC_FULL_T_FIT';first_proof=None
        with (out/f'training_segment_{args.segment:02d}.jsonl').open('x',buffering=1) as log:
            while cursor<396900 and updates-initial_updates<8192 and time.monotonic()-start<710:
                epoch=cursor//132300;offset=cursor%132300;indices=orders[epoch][offset:offset+16];n=len(indices)
                idx=indices.cuda();pos=(indices%13230).cuda();aid=inputs['affine_ids'][idx]
                c=inputs['condition'][idx];w=inputs['weight'][aid];b=inputs['bias'][aid]
                optimizer.zero_grad(set_to_none=True);changed,info=model(c,w,b)
                if updates==0:assert torch.equal(changed,c)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    if args.arm=='F_only':loss=.1*(1-F.cosine_similarity(changed.float(),inputs['target'][pos].float(),dim=-1)).mean()
                    else:
                        with torch.no_grad():real=projector(adapter(inputs['target'][pos]))
                        tokens=projector(adapter(changed));loss=.1*(1-F.cosine_similarity(tokens.float(),real.float(),dim=-1)).mean()
                assert torch.isfinite(loss);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                assert model.direction.weight.grad is not None and all(p.grad is None for p in list(adapter.parameters())+list(projector.parameters()))
                if updates==0:
                    assert model.direction.weight.grad.norm()>0
                    first_proof=dict(zero_head_native_condition_bit_exact=True,direction_gradient_norm=float(model.direction.weight.grad.norm()),trunk_gradient_norm=float(model.trunk[0].weight.grad.norm()),first_loss=float(loss),frozen_AP_grads_none=True)
                optimizer.step();assert not coverage[epoch,indices].any();coverage[epoch,indices]=True
                cursor+=n;seen+=n;updates+=1;total_loss+=float(loss.detach())*n
                if updates%64==0 or cursor%132300==0:
                    event=dict(arm=args.arm,updates=updates,seen_queries=seen,pass_number=epoch+1,cumulative_mean_loss=total_loss/seen,last_loss=float(loss),gradient_norm=float(gradient),seconds=seconds+time.monotonic()-tick)
                    log.write(json.dumps(event)+'\n');atomic(out/'progress.json',dict(status='RUNNING',segment=args.segment,**event))
                    budget()
        torch.cuda.synchronize();seconds+=time.monotonic()-tick;budget();assert updates>initial_updates
        assert frozen==dict(A=state_sha(adapter),P=state_sha(projector))
        assert seen==cursor and int(coverage.sum())==seen
        checkpoint=out/f'checkpoint_segment_{args.segment:02d}.pt'
        state=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),updates=updates,seen=seen,cursor=cursor,total_loss=total_loss,coverage=coverage,training_seconds=seconds,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),protocol_sha256=ref(args.protocol)['sha256'],arm=args.arm)
        save_pt(checkpoint,state)
        committed=torch.load(checkpoint,weights_only=False,map_location='cpu');assert committed['updates']==updates and int(committed['coverage'].sum())==seen
        assert all(torch.equal(state['model'][k],committed['model'][k]) for k in state['model'])
        assert {int(v['step']) for v in committed['optimizer']['state'].values()}=={updates};del committed
        result=dict(status='EXACT_PARTIAL_FULL_T_CONTROL_COMMITTED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=args.arm,segment=args.segment,updates=updates,seen_queries=seen,checkpoint=ref(checkpoint),protocol=ref(args.protocol),first_scientific_step_proof=first_proof,training_seconds=seconds,load_seconds=load_seconds,segment_seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),peak_extra_CPU_bytes=rss()-base,resources_before=resources,frozen_AP_unchanged=True,ordinary_unchanged=ref(s['ordinary']['path'])==s['ordinary'])
        if cursor==396900:
            assert coverage.all() and updates==24807;phase='DFIT_ONLY_FIXED_STRENGTH_CALIBRATION'
            calibration=development(model.eval(),adapter,projector,'ordinary');atomic(out/'calibration.json',calibration)
            final=out/'component_final.pt';save_pt(final,dict(model=state['model'],arm=args.arm,seed=1729,updates=updates,seen_queries=seen,passes=3,calibration=calibration,protocol_sha256=ref(args.protocol)['sha256']))
            result.update(status='COMPLETE_ALL10_REAL_OOF_THREE_PASSES_LOSS_CONTROL',component=ref(final),calibration=ref(out/'calibration.json'),new_research_images=0,scientific_fit_complete_is_not_image_success=True)
            atomic(out/'completion.json',result)
        atomic(out/f'segment_{args.segment:02d}.json',result);atomic(out/'progress.json',result)
        print(json.dumps({k:result[k] for k in ('status','arm','updates','seen_queries','training_seconds','segment_seconds','peak_GPU_bytes')}),flush=True)
    except BaseException as e:
        atomic(out/f'failure_segment_{args.segment:02d}.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--arm',choices=['F_only','token_only'],required=True);p.add_argument('--segment',type=int,required=True);run(p.parse_args())
