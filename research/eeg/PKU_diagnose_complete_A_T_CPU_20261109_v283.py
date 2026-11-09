"""Fixed complete-T mechanism readout; no D/V/images, no training or checkpoint selection."""
import csv,hashlib,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_A_complete_T_diagnostic_v283';out.mkdir();started=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(name,r):
    p=out/name;t=p.with_name(p.name+'.pending');b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode()
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
try:
    import numpy as np
    import torch
    from PKU_eeg_candidate_core_20261109_v266 import make_model,posterior_indices,encode
    torch.set_num_threads(4);torch.manual_seed(1729);study=R/'runs/EEG_candidates_v267';parent=R/'runs/HIVE_native_sub03_5096542';split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());ids=split['image_rows']['T'];assert len(ids)==13230
    with (R/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:identity={int(r['array_row']):r for r in csv.DictReader(f) if r['dataset_split']=='train'}
    concepts=np.asarray([identity[i]['concept_id'] for i in ids]);keys=sorted(set(concepts));assert len(keys)==1323 and all((concepts==k).sum()==10 for k in keys)
    group_index=torch.tensor(np.stack([np.flatnonzero(concepts==k) for k in keys]));concept_position={k:j for j,k in enumerate(keys)};group_for_query=torch.tensor([concept_position[k] for k in concepts]);own=torch.tensor([list(group_index[concept_position[concepts[i]]].tolist()).index(i) for i in range(13230)])
    allF=torch.load(parent/'deterministic_F_all16540.pt',map_location='cpu',weights_only=True);target=allF[ids].clone().double();del allF;target_unit=target/target.norm(dim=1,keepdim=True)
    raw=torch.load(R/'data/recovery_20261108_v236/sub-03_train_mean4.pt',map_location='cpu',weights_only=True,mmap=True);positions=[0,4409,8820,13229];probe=raw[[ids[i] for i in positions]].clone();del raw
    post,_=posterior_indices();stats=torch.load(study/'spatial_statistics.pt',map_location='cpu',weights_only=True);arms=['native_fullT','A_positive','A_pair'];all_values=[];histories={};proofs=[]
    for arm in arms:
        if arm=='native_fullT':
            bank_path=parent/'full/conditions.pt';cache=torch.load(bank_path,map_location='cpu',weights_only=True);digest=sha(parent/'full/brain_final.pt');model=make_model('A_positive',stats,device='cpu').requires_grad_(False)
        else:
            folder=study/arm;complete=json.loads((folder/'completion.json').read_text());assert complete['updates']==827 and complete['exposures']==13230;digest=sha(folder/'state.pt');assert digest==complete['checkpoint_sha256']
            audits=[json.loads(p.read_text()) for p in (study/'audits').glob(arm+'_*.json')];assert any(v['full_T_complete'] and v['checkpoint_sha256']==digest for v in audits)
            state=torch.load(folder/'state.pt',map_location='cpu',weights_only=False);assert state['coverage'].all();histories[arm]=state['history'];model=make_model(arm,stats,device='cpu').requires_grad_(False);model.load_state_dict(state['model'],strict=True)
            bank_path=folder/'full_conditions.pt';cache=torch.load(bank_path,map_location='cpu',weights_only=True);assert cache['checkpoint_sha256']==digest and cache['subject']=='sub-03';del state
        lookup={int(v):j for j,v in enumerate(cache['array_rows'])};c=cache['condition'][[lookup[i] for i in ids]].clone().double()
        with torch.inference_mode():actual=encode(model,'A_positive',probe,post).double()
        error=float((actual-c[positions]).abs().max());assert torch.allclose(actual,c[positions],atol=5e-5,rtol=5e-5)
        unit=c/c.norm(dim=1,keepdim=True);metrics=[]
        for lo in range(0,13230,256):
            hi=min(lo+256,13230);gallery=target_unit[group_index[group_for_query[lo:hi]]];score=(unit[lo:hi,None,:]*gallery).sum(-1);positive=score.gather(1,own[lo:hi,None])[:,0];negative=score.clone();negative[torch.arange(hi-lo),own[lo:hi]]=-torch.inf;wrong=negative.max(1).values;top=(score.argmax(1)==own[lo:hi]).double();afc=((score<positive[:,None]).sum(1)+.5*((score==positive[:,None]).sum(1)-1))/9
            mse=(c[lo:hi]-target[lo:hi]).square().mean(1);metrics.append(torch.stack([positive,wrong,positive-wrong,top,afc,mse],1))
        values=torch.cat(metrics).numpy();assert values.shape==(13230,6) and np.isfinite(values).all();all_values.append(values)
        mean_energy=float(c.mean(0).square().sum()/c.square().sum(1).mean());proofs.append(dict(arm=arm,checkpoint_sha256=digest,cache_sha256=sha(bank_path),fixed4_actual_CPU_forward_max_abs=error,mean_energy_fraction=mean_energy,training_exposure='All T seen by warm-start encoder; current A arms also fine-tuned T, not OOF or validation'))
        del model,cache,c,unit
    assert len(histories['A_positive'])==len(histories['A_pair'])==827
    logs={}
    for arm,history in histories.items():
        assert [r['update'] for r in history]==list(range(1,828)) and sum(r['n'] for r in history)==13230
        logs[arm]={k:float(sum(r[k]*r['n'] for r in history)/13230) for k in ['positive_energy','negative_energy','token_error','contrastive','grad_norm']};logs[arm]['mean_paired_energy_gap']=logs[arm]['negative_energy']-logs[arm]['positive_energy']
    assert [(r['offset'],r['n']) for r in histories['A_positive']]==[(r['offset'],r['n']) for r in histories['A_pair']]
    values=np.stack(all_values);names=['correct_F_cosine','hardest_sameconcept_F_cosine','sameconcept_margin','sameconcept_Top1','sameconcept_2AFC','F_MSE'];rng=np.random.default_rng(1729);sample=rng.integers(0,1323,size=(1000,1323));comparisons=[]
    for label,left,right in [('A_positive-native',1,0),('A_pair-native',2,0),('A_pair-A_positive',2,1)]:
        for mi,name in enumerate(names):
            delta=values[left,:,mi]-values[right,:,mi];per=np.asarray([delta[concepts==k].mean() for k in keys]);ci=np.quantile(per[sample].mean(1),[.025,.975]);comparisons.append(dict(comparison=label,metric=name,mean=float(delta.mean()),concept95_low=float(ci[0]),concept95_high=float(ci[1])))
    flips={}
    for left,right in [(1,0),(2,0),(2,1)]:
        a=values[left,:,3]>0;b=values[right,:,3]>0;flips[arms[left]+'-'+arms[right]]=dict(wrong_to_right=int((a&~b).sum()),right_to_wrong=int((~a&b).sum()),net=int(a.sum()-b.sum()))
    np.savez_compressed(out/'all13230_three_encoders.npz',values=values,arms=np.asarray(arms),metrics=np.asarray(names),array_rows=np.asarray(ids),concepts=concepts)
    write('completion.json',dict(status='COMPLETE_FIXED_FINAL_MODELS_ALL13230_T_GEOMETRY_AND_MATCHED_TRAIN_LOGS',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-started,queries=13230,concepts=1323,arms=arms,
        means={arm:{name:float(values[i,:,j].mean()) for j,name in enumerate(names)} for i,arm in enumerate(arms)},comparisons=comparisons,flips=flips,model_proofs=proofs,weighted_training_logs=logs,
        pairing_and_order_provenance_sha256=sha(study/'shared_provenance.json'),values_sha256=sha(out/'all13230_three_encoders.npz'),new_images=0,new_optimizer_updates=0,D_V_or_official_queries_used=False,
        no_effect_or_epoch_gate=True,independent_generalization=False,training_diagnostic_only=True,image_gain_or_SOTA_established=False,source_sha256=sha(__file__)))
    print('COMPLETE_ALL_T_FINAL_TWO_A_GEOMETRY_AND_TRAIN_LOGS',flush=True)
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
