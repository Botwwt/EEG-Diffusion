"""Four same-head continuations of immutable ordinary with normalized SGD."""
import argparse,gc,hashlib,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_common_continuation_contract_CPU_20261030_v1 import ARMS,FRACTIONS,atomic,checked,read,ref,rss,spec
def run(args):
    s=spec(args.study);assert args.mode in ('profile','fit') and (args.mode=='profile' or args.arm in ARMS)
    R=Path(s['repository_root']);root=Path(s['output_root']);root.mkdir(exist_ok=True);out=root/('fit_profile' if args.mode=='profile' else args.arm);out.mkdir(exist_ok=False)
    assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    sys.path.insert(0,str(R));import brainhive_paired_study_contract_20261025_v1 as C;C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']));sys.path.insert(0,str(Path(s['relative_study']['path']).parent))
    import torch,numpy as np
    import torch.nn.functional as F
    import brainhive_native_models_v1 as N
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_data_v1 import load_training,load_development
    from brainhive_conditional_relative_objective_20261030_v1 import relative_evidence_loss
    # These two implementations are the EXACT source pinned by the passed GPU profile.
    from brainhive_normalized_common_descent_20261030_v1 import direction,interpolate_state
    from train_HIVE_relative_continuation_GPU_20261030_v1 import bank
    from train_BrainHIVE_bounded_component_v1 import seed
    from brainhive_functional_training_contract_v2 import resource_headroom
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory));assert resource_headroom()['effective_headroom_bytes']>=10<<30
    started=time.monotonic();base=rss();phase='LOAD';stop=[False];updates=seen=nonzero=0;model=coverage=None
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def budget():
        assert not stop[0] and time.monotonic()-started<820 and rss()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        assert shutil.disk_usage(root).free>2<<30 and sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<2<<30
    def digest(m):
        h=hashlib.sha256()
        for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def save(p,x):tmp=p.with_name(p.name+'.pending');torch.save(x,tmp);os.replace(tmp,p)
    try:
        data=load_training();assert data['input_contract_sha256']==s['input_contract']['sha256'];contract=read(checked(s['input_contract']));banks,bank_hashes=bank(data,torch,np,contract,R);unit=F.normalize(data['target'].float(),dim=-1)
        prior=N.prior_models('cuda');A=prior['fusion_encoder'].adapter.float().eval().requires_grad_(False);P=prior['proj'].float().eval().requires_grad_(False);del prior;gc.collect();frozen=dict(A=digest(A),P=digest(P))
        with torch.no_grad():target_tokens=torch.cat([P(A(data['target'][i:i+256].float())) for i in range(0,13230,256)])
        original=torch.load(checked(s['ordinary']),weights_only=True,map_location='cpu')['model'];parameter_norm=float(torch.sqrt(sum(v.double().square().sum() for v in original.values())));step_size=parameter_norm*1e-4
        assert abs(step_size-s['fixed_parameter_step'])<1e-14
        engineering=read(checked(s['engineering_profile']));assert engineering['status']=='PASS_ACTUAL_FOUR_ARM_COMMON_DESCENT_FP32_GPU_ENGINEERING' and bank_hashes==engineering['bank_hashes']
        orders=[torch.randperm(132300,generator=torch.Generator().manual_seed(seed('pass'+str(i)))) for i in range(3)];orders_sha=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in orders]
        assert orders_sha==read(checked(s['starting_F_provenance']))['orders_sha256']
        def fresh():
            m=BoundedConditionAdapter().cuda().eval();m.load_state_dict(original,strict=True);return m
        def one_step(m,ids,arm,update):
            ids=ids.cuda();v=ids%13230;aid=data['affine_ids'][ids];changed=m(data['condition'][ids],data['weight'][aid],data['bias'][aid])[0]
            slots=banks['mismatched_slot' if arm=='joint_mismatched_teacher' else 'true_slot'][v];gallery=unit[banks['same'][v]]
            relative=relative_evidence_loss(changed,gallery,slots,banks['temperatures'][aid]).mean();AP=.1*(1-F.cosine_similarity(P(A(changed)),target_tokens[v],dim=-1)).mean()
            params=list(m.parameters());gr=torch.cat([x.reshape(-1) for x in torch.autograd.grad(relative,params,retain_graph=True)]);ga=torch.cat([x.reshape(-1) for x in torch.autograd.grad(AP,params)])
            mode='relative' if arm=='relative_only' else 'AP' if arm=='AP_only' else 'joint';d,info=direction(gr,ga,mode)
            assert torch.isfinite(d).all() and torch.isfinite(relative) and torch.isfinite(AP) and len(d)==525568
            record=dict(relative_loss=float(relative),AP_loss=float(AP),relative_gradient_norm=float(info['relative_norm']),AP_gradient_norm=float(info['AP_norm']),
                gradient_cosine=float((gr@ga)/(gr.norm()*ga.norm()).clamp_min(1e-30)),direction_norm=float(d.norm()),relative_derivative=float(info['relative_derivative']),AP_derivative=float(info['AP_derivative']))
            if update:
                offset=0
                with torch.no_grad():
                    for p in params:n=p.numel();p.add_(d[offset:offset+n].reshape_as(p),alpha=step_size);offset+=n
                assert offset==525568
            assert all(p.grad is None for p in list(A.parameters())+list(P.parameters()))
            return record
        def calibrate(m,caldata,label):
            # Five PARAMETER fractions, never radius fractions. At0, use exact O tensors.
            raw={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};errors=[];n=len(caldata['condition'])
            with torch.no_grad():ctarget=torch.cat([P(A(caldata['target'][i:i+256].float())) for i in range(0,n,256)])
            for fraction in FRACTIONS:
                m.load_state_dict(interpolate_state(original,raw,fraction),strict=True);total=0.
                with torch.no_grad():
                    for i in range(0,n,256):
                        c=m(caldata['condition'][i:i+256],caldata['weight'][i:i+256],caldata['bias'][i:i+256])[0];tokens=P(A(c));total+=float((1-F.cosine_similarity(tokens,ctarget[i:i+256],dim=-1)).mean(-1).sum())
                        budget()
                errors.append(total/n)
            chosen=min(range(5),key=lambda j:(errors[j],j));fraction=FRACTIONS[chosen];selected=interpolate_state(original,raw,fraction);m.load_state_dict(raw,strict=True)
            return dict(label=label,fractions=list(FRACTIONS),Dfit_token_errors=errors,selected_parameter_fraction=fraction,radius=.25,parameter_interpolation_not_radius_scaling=True,
                fraction0_is_original_ordinary=True,precision='same_original_weight_values_FP32_noautocast_TF32disabled'),selected
        if args.mode=='profile':
            phase='ACTUAL_EXECUTOR_ALL20_GRADIENTS_AND_DISCARDED_UPDATE_CALIBRATION_CONTRACT';proof={}
            witness=np.load(checked(s['feasibility_readouts']),allow_pickle=False);wi=torch.from_numpy(witness['indices']).cuda();wa=data['affine_ids'][wi]
            tiny=dict(condition=data['condition'][wi],weight=data['weight'][wa],bias=data['bias'][wa],target=data['target'][wi%13230])
            for arm in ARMS:
                m=fresh();assert digest(m)==engineering['initial_tensor_hash'];cells=[]
                for role in range(20):
                    ids=torch.nonzero(data['affine_ids']==role).flatten()[:16].cpu();cells.append(one_step(m,ids,arm,False));budget()
                tmp=[one_step(m,orders[0][j*16:(j+1)*16],arm,True) for j in range(2)];raw={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};cal,selected=calibrate(m,tiny,'T80_ENGINEERING_ONLY_NOT_DFIT')
                assert all(torch.equal(raw[k],m.state_dict()[k].cpu()) for k in raw);assert all(torch.equal(interpolate_state(original,raw,0.)[k],original[k]) for k in raw)
                proof[arm]=dict(all20_role_gradients=cells,discarded_updates=2,temporary_step_records=tmp,parameter_fraction_engineering=cal,temporary_weights_discarded=True)
                del m,raw,selected;gc.collect()
            assert frozen==dict(A=digest(A),P=digest(P));checked(s['ordinary']);budget()
            atomic(out/'completion.json',dict(status='PASS_ACTUAL_FINAL_COMMON_EXECUTOR_PROFILE_AND_FRACTION_CONTRACT',study=ref(args.study),initial_tensor_sha256=engineering['initial_tensor_hash'],arms=proof,orders_sha256=orders_sha,bank_hashes=bank_hashes,
                scientific_updates=0,temporary_updates=8,new_research_images=0,Dfit_effect_read=False,seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=rss()-base));return
        profile=read(root/'fit_profile/completion.json');assert profile['status']=='PASS_ACTUAL_FINAL_COMMON_EXECUTOR_PROFILE_AND_FRACTION_CONTRACT' and profile['study']==ref(args.study)
        model=fresh();assert digest(model)==profile['initial_tensor_sha256'];coverage=torch.zeros(3,132300,dtype=torch.bool);tick=time.monotonic();last_save=tick;totals=[0.,0.]
        provenance=dict(study=ref(args.study),arm=args.arm,initial_checkpoint=s['ordinary'],initial_tensor_sha256=digest(model),orders_sha256=orders_sha,bank_hashes=bank_hashes,
            optimizer='MANUAL_NORMALIZED_SGD_NO_MOMENTUM_NO_ADAM',fixed_parameter_step=step_size,parameters=525568,additional_passes=3,additional_updates=24807,additional_primary_exposures=396900,
            initial_updates=24807,total_historical_updates=49614,total_historical_exposures=793800,both_parameter_gradients_computed_for_every_control=True,training_conditions='TrueOOF',full_chain_OOF=False,frozen_AP=frozen)
        atomic(out/'provenance.json',provenance)
        def checkpoint():
            save(out/'resume.pt',dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},SGD=dict(kind='MANUAL_NORMALIZED_SGD',momentum=0.,parameter_step=step_size,iterations=updates,nonzero_steps=nonzero),
                coverage=coverage,updates=updates,seen=seen,orders_sha256=orders_sha,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                objective_totals=totals,training_seconds=time.monotonic()-tick,study_sha256=ref(args.study)['sha256'],arm=args.arm,seed=1729))
        phase='THREE_FULL_OOF_PASSES_FIXED_NORMALIZED_SGD'
        with (out/'training.jsonl').open('x',buffering=1) as log:
            for epoch in range(3):
                for offset in range(0,132300,16):
                    if stop[0]:checkpoint();raise InterruptedError('Owned stop with exact SGD state retained')
                    ids=orders[epoch][offset:offset+16];cell=one_step(model,ids,args.arm,True);assert not coverage[epoch,ids].any();coverage[epoch,ids]=True;updates+=1;seen+=len(ids);nonzero+=int(cell['direction_norm']>0)
                    totals[0]+=cell['relative_loss']*len(ids);totals[1]+=cell['AP_loss']*len(ids)
                    if updates%64==0 or offset+len(ids)==132300:
                        event=dict(arm=args.arm,updates=updates,seen_queries=seen,nonzero_steps=nonzero,pass_number=epoch+1,cumulative_mean_relative=totals[0]/seen,cumulative_mean_AP=totals[1]/seen,training_seconds=time.monotonic()-tick,**cell)
                        log.write(__import__('json').dumps(event)+'\n');atomic(out/'progress.json',event);budget()
                    if time.monotonic()-last_save>120:checkpoint();last_save=time.monotonic()
        torch.cuda.synchronize();training_seconds=time.monotonic()-tick;assert coverage.all() and updates==24807 and seen==396900;checkpoint()
        rawstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};rawfile=out/'raw_last.pt';save(rawfile,dict(model=rawstate,arm=args.arm,seed=1729,updates=updates,seen_queries=seen))
        phase='FIXED_LAST_ONLY_DFIT_PARAMETER_FRACTION_CALIBRATION';development=load_development('Dfit','cuda');cal,selected=calibrate(model,development,'ALL8200_DFIT_AFTER_FIXED_LAST');atomic(out/'calibration.json',cal)
        final=out/'component_final.pt';save(final,dict(model=selected,raw_last=ref(rawfile),arm=args.arm,seed=1729,updates=updates,seen_queries=seen,
            initial_checkpoint=s['ordinary'],initial_updates=24807,total_historical_updates=49614,total_historical_exposures=793800,calibration=cal,study_sha256=ref(args.study)['sha256'],radius=.25))
        identity=all(torch.equal(selected[k],original[k]) for k in original);assert frozen==dict(A=digest(A),P=digest(P));checked(s['ordinary']);budget()
        atomic(out/'completion.json',dict(status='COMPLETE_THREE_EXTRA_OOF_PASSES_NORMALIZED_SGD_PARAMETER_CALIBRATED',study=ref(args.study),arm=args.arm,component=ref(final),raw_last=ref(rawfile),checkpoint=ref(out/'resume.pt'),
            provenance=ref(out/'provenance.json'),calibration=ref(out/'calibration.json'),updates=updates,query_exposures=seen,nonzero_steps=nonzero,selected_parameter_fraction=cal['selected_parameter_fraction'],
            selected_parameters_exact_original_ordinary=identity,total_historical_updates=49614,total_historical_exposures=793800,training_seconds=training_seconds,seconds=time.monotonic()-started,
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=rss()-base,new_research_images=0))
    except BaseException as e:
        atomic(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),updates=updates,seen_queries=seen,seconds=time.monotonic()-started));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--mode',choices=['profile','fit'],required=True);p.add_argument('--arm',choices=ARMS);run(p.parse_args())
