"""Independent all-query CPU replay and fit/OOF/checkpoint audit, no new fitting."""
import json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ROOT=Path('/dfs/share/maleiLab/wwt/eeg');parent=ROOT/'runs/HIVE_native_sub03_5096542'
out=ROOT/'runs'/('HIVE_native_independent_audit_'+os.environ['SLURM_JOB_ID']);out.mkdir(mode=0o700);start=time.monotonic()
def write(name,v):
    p=out/name;t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
try:
    import numpy as np
    import torch
    from PKU_HIVE_native_classes_20261109_v239 import classes,sha
    torch.set_num_threads(4)
    done=json.loads((parent/'completion.json').read_text());assert done['status']=='COMPLETE_NATIVE_SUB03_THREE_ROLE_STOCHASTIC_TEACHER_REBUILD'
    split=json.loads((ROOT/'data/recovery_20261108_v236/current_split_manifest.json').read_text());T=split['image_rows']['T'];D=split['image_rows']['D']
    identities=json.loads((ROOT/'data/full_T_recovery_20261108_v238/T_identities.json').read_text());fold=np.array(identities['folds'])
    assert [r['array_row'] for r in identities['T']]==T
    layout=json.loads((ROOT/'data/recovery_20261108_v236/sub-03_layout.json').read_text())
    reg=json.loads((parent/'registration.json').read_text());policy=reg['policy'];assert policy['stochastic_teacher_dropout']==.1
    indices=[layout['channels'].index(n) for n in policy['electrodes']]
    raw=torch.load(ROOT/'data/recovery_20261108_v236/sub-03_train_mean4.pt',weights_only=True,map_location='cpu',mmap=True)
    combined=torch.load(parent/'T_true_OOF_conditions.pt',weights_only=True,map_location='cpu')
    assert combined['subject']=='sub-03' and combined['array_rows'].tolist()==T
    allOOF=combined['condition'];assert torch.isfinite(allOOF).all() and allOOF.shape==(13230,1024)
    seen=np.zeros(13230,np.int8);roles=[];replayed=0;maximum=0.;counted_updates=0;exposures=0;backup_files=[]
    scope=classes()
    for item in done['roles']:
        role=item['role'];folder=parent/role;contract=json.loads((folder/'input_contract.json').read_text())
        assert sha(folder/'brain_final.pt')==item['model_sha256']
        for n in ['brain_final.pt','final_training_state.pt']:
            p=folder/n;backup_files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
        checkpoint=torch.load(folder/'brain_final.pt',weights_only=True,map_location='cpu')
        state=torch.load(folder/'final_training_state.pt',weights_only=True,map_location='cpu')
        assert checkpoint['role']==role and checkpoint['subject']=='sub-03' and checkpoint['seed']==1729 and checkpoint['epochs']==25
        assert state['epoch']==25 and state['updates']==item['updates']==checkpoint['updates']
        assert all(torch.equal(v,state['model'][k]) and torch.isfinite(v).all() for k,v in checkpoint['model'].items())
        model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval();model.load_state_dict(checkpoint['model'],strict=True)
        assert len(state['optimizer']['state'])==len(list(model.parameters()))+1
        assert all(float(s['step'])==item['updates'] for s in state['optimizer']['state'].values())
        assert len(state['CUDA_RNG'])==1 and state['CPU_RNG'].dtype==torch.uint8 and state['shuffle_RNG'].dtype==torch.uint8
        fit=contract['train_array_rows'];assert set(fit).isdisjoint(D+split['image_rows']['V'])
        held=contract['heldout_array_rows']
        if role=='full':assert fit==T and not held;selected=D
        else:
            held_positions=np.flatnonzero(fold==int(role[-1]));expected_held=[T[i] for i in held_positions]
            assert held==expected_held and not set(fit)&set(held) and set(fit)|set(held)==set(T)
            assert {r['concept_id'] for r in identities['T'] if r['array_row'] in set(fit)}.isdisjoint({r['concept_id'] for r in identities['T'] if r['array_row'] in set(held)})
            seen[held_positions]+=1;selected=held
        epochs=[json.loads(x) for x in (folder/'epochs.jsonl').read_text().splitlines()]
        assert len(epochs)==25 and [r['epoch'] for r in epochs]==list(range(1,26))
        per_epoch=(len(fit)+1023)//1024
        assert all(r['updates']==(i+1)*per_epoch and r['unique_epoch_queries']==len(fit) for i,r in enumerate(epochs))
        counted_updates+=epochs[-1]['updates'];exposures+=25*len(fit)
        exported=torch.load(folder/'conditions.pt',weights_only=True,map_location='cpu')
        assert exported['model_sha256']==item['model_sha256']
        if role=='full':assert exported['array_rows'].tolist()==T+D;values=exported['condition'][13230:];fullD=values
        else:
            assert exported['array_rows'].tolist()==held;values=exported['condition']
            assert torch.equal(values,allOOF[held_positions])
        max_error=0.
        with torch.inference_mode():
            for lo in range(0,len(selected),128):
                which=selected[lo:lo+128];pred=model(raw[which][:,indices,:]);ref=values[lo:lo+128]
                error=float((pred-ref).abs().max());max_error=max(max_error,error)
                assert torch.allclose(pred,ref,atol=3e-5,rtol=3e-4),(role,lo,error)
                replayed+=len(which)
        maximum=max(maximum,max_error);roles.append(dict(role=role,model_SHA=item['model_sha256'],train=len(fit),held=len(held),updates=item['updates'],CPU_replay_rows=len(selected),max_abs_CPU_GPU_error=max_error))
        del model,state,checkpoint
    assert (seen==1).all() and counted_updates==675 and exposures==661500 and replayed==14880
    target=torch.load(parent/'deterministic_F_all16540.pt',weights_only=True,map_location='cpu').numpy().astype(np.float64)
    def score(value,rows,label):
        x=value.numpy().astype(np.float64);y=target[rows];x/=np.linalg.norm(x,axis=1,keepdims=True);y/=np.linalg.norm(y,axis=1,keepdims=True)
        cosine=[];global_count=0;within_count=0
        for lo in range(0,len(x),128):
            a=x[lo:lo+128]@y.T;ix=np.arange(lo,min(lo+128,len(x)));own=a[np.arange(len(ix)),ix]
            cosine.extend(own.tolist());global_count+=int(np.sum(np.argmax(a,axis=1)==ix))
            mask=ix[:,None]//10==np.arange(len(x))[None,:]//10
            within_count+=int(np.sum(np.argmax(np.where(mask,a,-np.inf),axis=1)==ix))
        prior=done['readout'][label]
        assert global_count==round(prior['global_Top1']*len(x)),(label,'global',global_count,prior)
        assert within_count==round(prior['within_concept_Top1']*len(x)),(label,'within',within_count,prior)
        assert abs(float(np.mean(cosine))-prior['matched_cosine'])<1e-6
        return dict(queries=len(x),global_correct=global_count,within_correct=within_count,matched_cosine=float(np.mean(cosine)),FP64_full_gallery_recomputed=True)
    scores=dict(T_true_OOF=score(allOOF,T,'T_true_OOF'),D_full_fixed_final=score(fullD,D,'D_full_fixed_final'))
    result=dict(status='PASS_INDEPENDENT_ALL14880_CPU_FORWARD_OOF_AND_FULL_SCORE_AUDIT',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),
        seconds=time.monotonic()-start,roles=roles,CPU_forward_queries=replayed,max_abs_CPU_GPU_error=maximum,optimizer_updates_verified=counted_updates,
        exposures_verified=exposures,OOF_all13230_rows_exactly_once=True,concept_exclusion_all_roles=True,scores=scores,backup_files=backup_files,new_updates=0,new_images=0)
    write('completion.json',result);print(json.dumps(result),flush=True)
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
