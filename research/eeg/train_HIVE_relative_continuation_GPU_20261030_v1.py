"""Four separately continued identical F heads; extra3 fullOOF passes each."""
import argparse,csv,gc,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_relative_continuation_contract_CPU_20261030_v1 import ARMS,atomic,checked,read,ref,rss,spec

def bank(inputs,torch,np,contract,R):
    metadata={int(r['array_row']):r['concept_id'] for r in csv.DictReader((R/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    rows=np.asarray(inputs['array_rows']);concepts=sorted({metadata[int(r)] for r in rows});assert len(concepts)==1323
    groups=np.asarray([[j for j,r in enumerate(rows) if metadata[int(r)]==c] for c in concepts]);assert groups.shape==(1323,10)
    ci=np.empty(13230,dtype=np.int64);slot=np.empty_like(ci)
    for i,g in enumerate(groups):ci[g]=i;slot[g]=np.arange(10)
    same=groups[ci];different_ci=(ci[:,None]+1+137*np.arange(9)[None,:])%1323;different_slot=(slot[:,None]+1+3*np.arange(9)[None,:])%10
    different=np.concatenate([np.arange(13230)[:,None],groups[different_ci,different_slot]],axis=1)
    assert np.all(same[np.arange(13230),slot]==np.arange(13230))
    assert np.all(ci[different[:,1:]]!=ci[:,None]) and np.all((slot+1)%10!=slot)
    temperatures=[]
    for subject in contract['subjects']:
        for x in subject['sources']:
            if x['role'].startswith('oof'):
                state=torch.load(x['checkpoint_path'],weights_only=True,map_location='cpu');temperatures.append(float(state['logit_scale'].exp()))
    assert len(temperatures)==20
    arrays=dict(same=same,different=different,true_slot=slot,mismatched_slot=(slot+1)%10,temperatures=np.asarray(temperatures,dtype=np.float32))
    hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
    return {k:torch.from_numpy(v).cuda() for k,v in arrays.items()},hashes

def run(args):
    s=spec(args.study);assert args.mode in ('profile','fit') and (args.mode=='profile' or args.arm in ARMS)
    R=Path(s['repository_root']);root=Path(s['output_root']);root.mkdir(exist_ok=True)
    out=root/('fit_profile' if args.mode=='profile' else args.arm);out.mkdir(exist_ok=False)
    assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    sys.path.insert(0,str(R));import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    import torch,numpy as np
    import torch.nn.functional as F
    import brainhive_native_models_v1 as N
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_data_v1 import load_training
    from brainhive_conditional_relative_objective_20261030_v1 import relative_evidence_loss
    from train_BrainHIVE_bounded_component_v1 import seed,development
    from brainhive_functional_training_contract_v2 import resource_headroom
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    assert resource_headroom()['effective_headroom_bytes']>=10<<30
    started=time.monotonic();base=rss();phase='LOAD';stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def budget():
        assert not stop[0] and time.monotonic()-started<820
        assert rss()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        assert shutil.disk_usage(root).free>2<<30 and sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<2<<30
    def digest(m):
        h=hashlib.sha256()
        for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def save(p,x):
        tmp=p.with_name(p.name+'.pending');torch.save(x,tmp);os.replace(tmp,p)
    updates=seen=0;model=optimizer=coverage=None;tick=None
    try:
        inputs=load_training();assert inputs['input_contract_sha256']==s['input_contract']['sha256']
        contract=read(s['input_contract']['path']);banks,bank_hashes=bank(inputs,torch,np,contract,R)
        unit_targets=F.normalize(inputs['target'].float(),dim=-1)
        prior=N.prior_models('cuda');A=prior['fusion_encoder'].adapter.half().eval().requires_grad_(False);P=prior['proj'].half().eval().requires_grad_(False);del prior;gc.collect();torch.cuda.empty_cache()
        frozen=dict(A=digest(A),P=digest(P));initial_state=torch.load(checked(s['starting_F']),weights_only=True,map_location='cpu')
        assert initial_state['updates']==24807 and initial_state['seen_queries']==396900
        orders=[torch.randperm(132300,generator=torch.Generator().manual_seed(seed('pass'+str(i)))) for i in range(3)]
        order_sha=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in orders]
        assert order_sha==read(checked(s['starting_F_provenance']))['orders_sha256']
        def fresh():
            torch.manual_seed(1729);m=BoundedConditionAdapter().cuda();m.load_state_dict(initial_state['model'],strict=True)
            assert sum(v.numel() for v in m.parameters())==525568
            return m,torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01)
        def step(m,opt,arm,indices,update=True):
            idx=indices.cuda();v=(indices%13230).cuda();aid=inputs['affine_ids'][idx];c=inputs['condition'][idx];w=inputs['weight'][aid];b=inputs['bias'][aid]
            opt.zero_grad(set_to_none=True);changed,info=m(c,w,b)
            if arm=='continued_F_cosine':loss=.1*(1-F.cosine_similarity(changed.float(),inputs['target'][v].float(),dim=-1)).mean()
            else:
                gallery=unit_targets[banks['different' if arm=='conditional_different' else 'same'][v]]
                target=torch.zeros(len(idx),dtype=torch.long,device='cuda') if arm=='conditional_different' else banks['mismatched_slot' if arm=='conditional_same_mismatched_teacher' else 'true_slot'][v]
                loss=relative_evidence_loss(changed.float(),gallery,target,banks['temperatures'][aid]).mean()
            assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
            assert all(v.grad is None for v in list(A.parameters())+list(P.parameters()))
            record=dict(loss=float(loss),gradient_norm=float(gn),direction_gradient=float(m.direction.weight.grad.norm()),trunk_gradient=float(m.trunk[0].weight.grad.norm()))
            if update:opt.step()
            return record
        selected=torch.stack([torch.nonzero(inputs['affine_ids']==i)[0,0] for i in range(20)])
        if args.mode=='profile':
            phase='ACTUAL_FOUR_ARM_GRADIENT_PROFILE';proof={};initial_hashes=[]
            zero=BoundedConditionAdapter().cuda().eval()
            with torch.no_grad():assert torch.equal(zero(inputs['condition'][selected],inputs['weight'],inputs['bias'])[0],inputs['condition'][selected])
            del zero
            for arm in ARMS:
                model,optimizer=fresh();initial_hashes.append(digest(model))
                with torch.no_grad():assert torch.equal(model(inputs['condition'][selected],inputs['weight'],inputs['bias'],enabled=False)[0],inputs['condition'][selected])
                # Actual nonzero-head GPU objective against independent NumPy64
                # loss/condition gradient, before the disposable Adam updates.
                with torch.no_grad():changed=model(inputs['condition'][selected],inputs['weight'],inputs['bias'])[0]
                if arm!='continued_F_cosine':
                    vv=(selected%13230).cuda();gallery=unit_targets[banks['different' if arm=='conditional_different' else 'same'][vv]]
                    slots=torch.zeros(20,dtype=torch.long,device='cuda') if arm=='conditional_different' else banks['mismatched_slot' if arm=='conditional_same_mismatched_teacher' else 'true_slot'][vv]
                    k=banks['temperatures'][inputs['affine_ids'][selected]];variable=changed.detach().clone().requires_grad_(True)
                    actual_loss=relative_evidence_loss(variable,gallery,slots,k);actual_gradient=torch.autograd.grad(actual_loss.sum(),variable)[0].cpu().numpy().astype(np.float64)
                    cc=changed.cpu().numpy().astype(np.float64);ff=gallery.cpu().numpy().astype(np.float64);kk=k.cpu().numpy().astype(np.float64);ss=slots.cpu().numpy();qq=cc/np.linalg.norm(cc,axis=1,keepdims=True)
                    logits=np.einsum('bd,bkd->bk',qq,ff)*kk[:,None];exp=np.exp(logits-logits.max(axis=1,keepdims=True));pp=exp/exp.sum(axis=1,keepdims=True)
                    ideal_loss=.1*(np.log(exp.sum(axis=1))+logits.max(axis=1)-logits[np.arange(20),ss])/kk
                    dif=np.einsum('bk,bkd->bd',pp,ff)-ff[np.arange(20),ss];ideal_gradient=.1*(dif-qq*np.sum(dif*qq,axis=1,keepdims=True))/np.linalg.norm(cc,axis=1,keepdims=True)
                    loss_error=float(np.max(np.abs(actual_loss.detach().cpu().numpy()-ideal_loss)));gradient_error=float(np.max(np.abs(actual_gradient-ideal_gradient)));gradient_relative=float(np.linalg.norm(actual_gradient-ideal_gradient)/np.linalg.norm(ideal_gradient))
                    assert loss_error<2e-6 and gradient_error<1e-7 and gradient_relative<1e-4
                    independent=dict(actual_role_queries=20,loss_error=loss_error,gradient_error=gradient_error,gradient_relative=gradient_relative)
                else:independent=dict(legacy_F_cosine_objective_reused=True)
                records=[]
                for i in range(20):
                    idx=torch.nonzero(inputs['affine_ids']==i).flatten()[:16].cpu();x=step(model,optimizer,arm,idx,False);assert x['direction_gradient']>0 and x['trunk_gradient']>0;records.append(dict(affine_id=i,**x));budget()
                temporary=[step(model,optimizer,arm,orders[0][j*16:(j+1)*16]) for j in range(4)]
                proof[arm]=dict(all20_affine_gradients=records,independent_GPU_math=independent,temporary_updates=4,temporary_records=temporary,temporary_weights_discarded=True)
                del model,optimizer;model=optimizer=None;gc.collect()
            assert len(set(initial_hashes))==1 and frozen==dict(A=digest(A),P=digest(P));budget()
            for x in (s['ordinary'],s['starting_F']):checked(x)
            atomic(out/'completion.json',dict(status='PASS_ACTUAL_FOUR_ARM_NONZERO_INITIAL_HEAD_GRADIENT_PROFILE',study=ref(args.study),initial_tensor_sha256=initial_hashes[0],orders_sha256=order_sha,bank_hashes=bank_hashes,arms=proof,
                all20_zero_head_and_disabled_exact=True,baseline_files_unchanged=True,temporary_updates=16,scientific_updates=0,weights_saved=False,generator_pixel_profile_not_yet_run=True,seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=rss()-base));return
        engineering=read(root/'fit_profile/completion.json');assert engineering['status']=='PASS_ACTUAL_FOUR_ARM_NONZERO_INITIAL_HEAD_GRADIENT_PROFILE' and engineering['study']==ref(args.study)
        model,optimizer=fresh();assert digest(model)==engineering['initial_tensor_sha256'] and bank_hashes==engineering['bank_hashes']
        provenance=dict(study=ref(args.study),arm=args.arm,initial_tensor_sha256=digest(model),initial_checkpoint=s['starting_F'],orders_sha256=order_sha,bank_hashes=bank_hashes,frozen_AP=frozen,
            additional_updates=24807,additional_exposures=396900,initial_updates=24807,total_historical_updates=49614,seed=1729,parameters=525568,training_conditions='TrueOOF',full_chain_OOF=False,new_modules=0)
        atomic(out/'provenance.json',provenance);coverage=torch.zeros(3,132300,dtype=torch.bool);total=0.;tick=time.monotonic();last_save=tick
        def checkpoint():
            save(out/'resume.pt',dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),coverage=coverage,updates=updates,seen=seen,
                orders_sha256=order_sha,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),total_loss=total,training_seconds=time.monotonic()-tick,study_sha256=ref(args.study)['sha256'],arm=args.arm,seed=1729))
        phase='ALL132300_THREE_ADDITIONAL_PASSES'
        with (out/'training.jsonl').open('x',buffering=1) as log:
            for epoch in range(3):
                for offset in range(0,132300,16):
                    if stop[0]:checkpoint();raise InterruptedError('Owned user stop with state retained')
                    indices=orders[epoch][offset:offset+16];x=step(model,optimizer,args.arm,indices)
                    assert not coverage[epoch,indices].any();coverage[epoch,indices]=True;updates+=1;seen+=len(indices);total+=x['loss']*len(indices)
                    if updates%64==0 or offset+len(indices)==132300:
                        event=dict(arm=args.arm,updates=updates,seen_queries=seen,pass_number=epoch+1,cumulative_mean_loss=total/seen,training_seconds=time.monotonic()-tick,**x)
                        log.write(json.dumps(event)+'\n');atomic(out/'progress.json',event);budget()
                    if time.monotonic()-last_save>120:checkpoint();last_save=time.monotonic()
        torch.cuda.synchronize();training_seconds=time.monotonic()-tick;assert coverage.all() and updates==24807 and seen==396900;checkpoint()
        phase='DFIT_ONLY_ORIGINAL_AP_FIVE_STRENGTHS';calibration=development(model.eval(),A,P,'ordinary');atomic(out/'calibration.json',calibration)
        final=out/'component_final.pt';save(final,dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},arm=args.arm,seed=1729,updates=updates,seen_queries=seen,
            additional_passes=3,initial_checkpoint=s['starting_F'],initial_updates=24807,total_historical_updates=49614,total_historical_exposures=793800,calibration=calibration,study_sha256=ref(args.study)['sha256']))
        assert frozen==dict(A=digest(A),P=digest(P));budget()
        for x in (s['ordinary'],s['starting_F']):checked(x)
        atomic(out/'completion.json',dict(status='COMPLETE_EXTRA_THREE_FULL_OOF_PASSES_FIXED_LAST_RELATIVE_CONTROL',study=ref(args.study),arm=args.arm,component=ref(final),checkpoint=ref(out/'resume.pt'),
            provenance=ref(out/'provenance.json'),calibration=ref(out/'calibration.json'),updates=24807,query_exposures=396900,total_historical_updates=49614,total_historical_exposures=793800,
            selected_strength=calibration['selected_strength'],orders_sha256=order_sha,initial_tensor_sha256=engineering['initial_tensor_sha256'],bank_hashes=bank_hashes,frozen_AP_unchanged=True,
            training_seconds=training_seconds,seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=rss()-base,new_research_images=0))
    except BaseException as e:
        atomic(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),updates=updates,seen=seen,seconds=time.monotonic()-started));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--mode',choices=['profile','fit'],required=True);p.add_argument('--arm',choices=ARMS);run(p.parse_args())
