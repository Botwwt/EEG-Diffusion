"""Fresh native sub03 full/oof0/oof1 fits, stochastic frozen F, fixed final25."""
import csv,json,math,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ROOT=Path('/dfs/share/maleiLab/wwt/eeg');out=ROOT/'runs'/('HIVE_native_sub03_'+os.environ['SLURM_JOB_ID']);out.mkdir(mode=0o700);start=time.monotonic()
def write(p,v):
    p=Path(p);t=p.with_name(p.name+'.pending');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def save(p,v):
    p=Path(p);t=p.with_name(p.name+'.pending');import torch;torch.save(v,t);t.replace(p)
try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from PKU_HIVE_native_classes_20261109_v239 import classes,teacher,sha
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    policy=json.loads((ROOT/'code/HIVE_native_sub03_protocol_v239.json').read_text())
    assert policy['epochs']==25 and policy['seed']==1729 and policy['stochastic_teacher_dropout']==.1
    cpu=json.loads((ROOT/'data/native_training_v239/completion.json').read_text());assert cpu['status']=='COMPLETE_NATIVE_HIVE_TEACHER_TRAIN_INPUTS_RESTORED'
    assert sha(ROOT/'data/native_training_v239/train_visual_embeddings.pt')==cpu['visual_embeddings_sha256']
    identity=dict(job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),GPU=torch.cuda.get_device_name(0),GPU_count=1,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,policy_sha256=sha(ROOT/'code/HIVE_native_sub03_protocol_v239.json'))
    write(out/'registration.json',dict(identity=identity,policy=policy,new_fresh_fit_not_exact_old_resume=True))
    raw_path=ROOT/'data/recovery_20261108_v236/sub-03_train_mean4.pt'
    assert sha(raw_path)=='85a25128f6b94e3739478c60e089d33f437a8dcbeb31cf136ce9772d2cc1615d'
    raw=torch.load(raw_path,map_location='cpu',weights_only=True,mmap=True)
    layout=json.loads((ROOT/'data/recovery_20261108_v236/sub-03_layout.json').read_text())
    names=policy['electrodes'];indices=[layout['channels'].index(n) for n in names]
    eeg=raw[:,indices,:].clone().cuda();del raw
    assert eeg.shape==(16540,17,250) and torch.isfinite(eeg).all()
    split=json.loads((ROOT/'data/recovery_20261108_v236/current_split_manifest.json').read_text());T=split['image_rows']['T'];assert len(T)==13230
    with (ROOT/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:
        rows={int(r['array_row']):r for r in csv.DictReader(f) if r['dataset_split']=='train'}
    fold=np.array([split['oof_fold_by_concept'][rows[i]['concept_id']] for i in T]);assert set(fold)=={0,1}
    scope=classes();visual_teacher=teacher(scope).cuda()
    embeddings={k:v.cuda() for k,v in torch.load(ROOT/'data/native_training_v239/train_visual_embeddings.pt',weights_only=True).items()}
    def guard():
        assert time.monotonic()-start<1100,'bounded GPU job time exceeded'
        assert torch.cuda.max_memory_allocated()<20<<30,'original20GiB GPU allocated cap'
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<1<<30,'owned1GiB output cap'
    # Deterministic reference only for replay/evaluation. Training uses fresh stochastic F.
    fullF=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for lo in range(0,16540,256):fullF.append(visual_teacher(**{k:v[lo:lo+256] for k,v in embeddings.items()}).float().cpu())
    fullF=torch.cat(fullF);save(out/'deterministic_F_all16540.pt',fullF)
    old=torch.from_numpy(np.load(ROOT/'data/full_T_recovery_20261108_v238/T_gallery_F.npy',allow_pickle=False))
    replay=fullF[T];cos=F.cosine_similarity(replay,old);rel=(replay-old).norm(dim=-1)/old.norm(dim=-1)
    assert float(cos.min())>.99998 and float(rel.max())<.005,('full native F replay mismatch',float(cos.min()),float(rel.max()))
    write(out/'teacher_replay.json',dict(rows=13230,minimum_cosine=float(cos.min()),maximum_relative_L2=float(rel.max()),cached_F_used_for_training=False))
    for layer in visual_teacher.modules():
        if isinstance(layer,nn.Dropout):layer.p=.1
    visual_teacher.train().requires_grad_(False)
    assert not any(p.requires_grad for p in visual_teacher.parameters())
    OOF=torch.empty(13230,1024);seen=np.zeros(13230,dtype=np.int8);fits=[];total_updates=0
    for role in ['full','oof0','oof1']:
        guard();folder=out/role;folder.mkdir()
        held=np.flatnonzero(fold==int(role[-1])).tolist() if role!='full' else []
        train_positions=list(range(13230)) if role=='full' else np.flatnonzero(fold!=int(role[-1])).tolist()
        allowed=[T[i] for i in train_positions];assert set(allowed).isdisjoint(split['image_rows']['D']+split['image_rows']['V'])
        if held:assert not set(allowed)&{T[i] for i in held}
        torch.manual_seed(1729);torch.cuda.manual_seed_all(1729);shuffle=torch.Generator().manual_seed(1730)
        model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda();assert sum(p.numel() for p in model.parameters())==6452224
        scale=nn.Parameter(torch.tensor(2.6592,device='cuda'))
        norm_ids={id(p) for layer in model.modules() if isinstance(layer,nn.LayerNorm) for p in layer.parameters()}
        decay=[p for n,p in model.named_parameters() if not n.endswith('bias') and id(p) not in norm_ids]+[scale]
        no_decay=[p for n,p in model.named_parameters() if n.endswith('bias') or id(p) in norm_ids]
        pars=list(model.parameters())+[scale]
        opt=torch.optim.AdamW([dict(params=decay,weight_decay=.05),dict(params=no_decay,weight_decay=0.)],lr=5e-4,betas=(.9,.995))
        expected=25*math.ceil(len(allowed)/1024);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:.5*(1+math.cos(math.pi*step/expected)))
        ids=torch.tensor(allowed,device='cuda');updates=0;role_start=time.monotonic()
        write(folder/'input_contract.json',dict(role=role,subject='sub-03',train_array_rows=allowed,heldout_array_rows=[T[i] for i in held],epochs=25,
            source_forward_unchanged=True,actual_teacher_train_mode=True,visual_dropout=.1,deterministic_F_cache_used_as_training_target=False))
        with (folder/'epochs.jsonl').open('x',buffering=1) as log:
            for epoch in range(25):
                model.train();total=0.;nseen=0
                order=ids[torch.randperm(len(ids),generator=shuffle).cuda()]
                for b in order.split(1024):
                    guard();opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        brain=F.normalize(model(eeg[b]),dim=-1)
                        with torch.no_grad():target=visual_teacher(**{k:v[b] for k,v in embeddings.items()})
                        target=F.normalize(target.to(brain.dtype),dim=-1)
                        logits=brain@target.T*scale.exp();labels=torch.arange(len(b),device='cuda')
                        loss=.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))
                    assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(pars,1.,error_if_nonfinite=True)
                    opt.step();sched.step();updates+=1;total_updates+=1;total+=float(loss)*len(b);nseen+=len(b)
                assert nseen==len(allowed)
                row=dict(role=role,epoch=epoch+1,updates=updates,expected_updates=expected,unique_epoch_queries=nseen,
                    mean_train_InfoNCE=total/nseen,grad_norm_last=float(grad),seconds=time.monotonic()-role_start)
                log.write(json.dumps(row)+'\n');write(out/'progress.json',dict(stage='TRAINING_NATIVE_ENCODER',**row,total_updates=total_updates))
                if epoch==0 or (epoch+1)%5==0:print(json.dumps(row),flush=True)
        assert updates==expected
        model.eval();weights={k:v.detach().cpu() for k,v in model.state_dict().items()}
        save(folder/'brain_final.pt',dict(model=weights,logit_scale=scale.detach().cpu(),role=role,subject='sub-03',seed=1729,epochs=25,updates=updates))
        save(folder/'final_training_state.pt',dict(model=weights,logit_scale=scale.detach().cpu(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
            CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all(),shuffle_RNG=shuffle.get_state(),epoch=25,updates=updates))
        export_rows=T+split['image_rows']['D'] if role=='full' else [T[i] for i in held]
        values=[]
        with torch.inference_mode():
            for lo in range(0,len(export_rows),256):values.append(model(eeg[export_rows[lo:lo+256]]).float().cpu())
        values=torch.cat(values);assert torch.isfinite(values).all()
        save(folder/'conditions.pt',dict(condition=values,array_rows=torch.tensor(export_rows),model_sha256=sha(folder/'brain_final.pt')))
        if role=='full':fullD=values[13230:]
        else:OOF[held]=values;seen[held]+=1
        fits.append(dict(role=role,updates=updates,training_queries=len(allowed),exposures=25*len(allowed),model_sha256=sha(folder/'brain_final.pt')))
        if role=='full':
            # Actual trained native architecture support, not an end-to-end speed claim.
            import eeg_margin_matrix_free_20261106_v233 as M
            from check_eeg_cross_host_gpu_runtime_CPU_20261106_v231 import serial
            xx=eeg[T[:2]];gallery=F.normalize(fullF[T].cuda(),dim=-1);own=torch.arange(2,device='cuda');groups=torch.arange(13230,device='cuda')//10
            _,J,_,_=serial(model(xx),gallery,own,groups,tuple(model.parameters()))
            encoder,theta=M.functional_encoder_snapshot(model,list(dict(model.named_parameters())),(xx,))
            operator=M.MatrixFreeMarginMap(encoder,theta,gallery,own,groups)
            tangent=tuple(torch.randn_like(t)*.001 for t in theta);vf=torch.cat([v.reshape(-1) for v in tangent]).double()
            w=torch.randn(4,device='cuda');got=operator.forward(tangent).double();ref=J@vf
            jt=torch.cat([v.reshape(-1) for v in operator.transpose(w)]).double();jref=J.T@w.double()
            g=operator.gram(w).double();gref=J@(J.T@w.double())
            errors={n:float((a-b).abs().max()/(1+b.abs().max())) for n,a,b in [('Jv',got,ref),('JT_w',jt,jref),('Gram',g,gref)]}
            assert max(errors.values())<1e-4,errors
            write(out/'trained_native_operator_support.json',dict(status='PASS_REAL_TRAINED_BRAINMLP_REAL_EEG_AND_T_GALLERY_OPERATOR_SUPPORT',
                queries=2,gallery=13230,parameters=6452224,scaled_max_errors=errors,speedup_or_nonlinear_training_trajectory_proved=False))
            del J,encoder,theta,operator,tangent,vf,got,ref,jt,jref,g,gref,gallery
        del model,opt,sched,weights,scale,pars,decay,no_decay;torch.cuda.empty_cache()
    assert (seen==1).all();save(out/'T_true_OOF_conditions.pt',dict(condition=OOF,array_rows=torch.tensor(T),subject='sub-03',folds=torch.tensor(fold)))
    # Fixed final encoder readout; no images, no V, no epoch or hyperparameter selection.
    def metrics(value,target,batch=256):
        y=F.normalize(target.cuda(),dim=-1);outputs=[]
        with torch.inference_mode():
            for lo in range(0,len(value),batch):
                pred=F.normalize(value[lo:lo+batch].cuda(),dim=-1);s=pred@y.T
                ii=torch.arange(len(pred),device='cuda');own=torch.arange(lo,lo+len(pred),device='cuda');correct=s[ii,own]
                global_correct=s.argmax(-1)==own
                mask=(own[:,None]//10)==(torch.arange(len(y),device='cuda')[None,:]//10)
                within=s.masked_fill(~mask,-torch.inf).argmax(-1)==own
                outputs.append(torch.stack([correct,global_correct.float(),within.float()],dim=1).cpu())
        a=torch.cat(outputs);return dict(queries=len(a),matched_cosine=float(a[:,0].mean()),global_Top1=float(a[:,1].mean()),within_concept_Top1=float(a[:,2].mean()))
    readout=dict(T_true_OOF=metrics(OOF,fullF[T]),D_full_fixed_final=metrics(fullD,fullF[split['image_rows']['D']]),
        interpretation='Encoder baseline diagnostics only, not image efficacy; one subject, source-stochastic teacher, no V/R80 accessed')
    write(out/'encoder_readout.json',readout);guard()
    result=dict(status='COMPLETE_NATIVE_SUB03_THREE_ROLE_STOCHASTIC_TEACHER_REBUILD',identity=identity,roles=fits,total_updates=total_updates,
        total_exposures=sum(r['exposures'] for r in fits),seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        readout=readout,new_images=0,new_ordinary_fit=0,old_ordinary_changed=False,exact_old_checkpoint_resumed=False,new_method_gain_or_SOTA=False)
    write(out/'completion.json',result);print(json.dumps(result),flush=True)
except BaseException as e:
    write(out/'failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
