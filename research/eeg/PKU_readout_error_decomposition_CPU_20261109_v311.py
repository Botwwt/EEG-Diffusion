"""Frozen full-T counterfactual analysis, no new learned model or image claim."""
import csv,gzip,hashlib,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
R=Path('/dfs/share/maleiLab/wwt/eeg');source=R/'runs/EEG_conjugate_target_decision_v310';out=R/'runs/EEG_readout_error_decomposition_v311';out.mkdir(exist_ok=False);start=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(name,r):
    p=out/name;t=p.with_name(p.name+'.pending');b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode()
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    t.replace(p)
try:
    import numpy as np
    import torch
    from PKU_conjugate_coordinates_20261109_v310 import OrdinaryCapacityControl
    torch.set_num_threads(4)
    done=json.loads((source/'completion.json').read_text());assert done['queries']==13230 and done['fits']==6
    assert done['table_sha256']==sha(source/'all13230_4controls.csv.gz')
    parent=R/'runs/HIVE_native_sub03_5096542';bp=parent/'T_true_OOF_conditions.pt';fp=parent/'deterministic_F_all16540.pt'
    assert sha(bp)==done['input_sha256']['OOF'] and sha(fp)==done['input_sha256']['F']
    bank=torch.load(bp,map_location='cpu',weights_only=True);rows=bank['array_rows'].tolist();x=bank['condition'].float();F=torch.load(fp,map_location='cpu',weights_only=True)[rows].float();row_index={r:i for i,r in enumerate(rows)}
    original={};concepts={}
    with gzip.open(source/'all13230_4controls.csv.gz','rt') as f:
        for v in csv.DictReader(f):
            i=row_index[int(v['array_row'])];concepts[i]=v['concept_id'];original[(v['case'],i)]={k:float(v[k]) for k in ['original_F_MSE','original_F_cosine','same_concept_margin','same_concept_strict_Top1']}
    assert len(original)==52920 and len(concepts)==13230
    groups={key:sorted([i for i,c in concepts.items() if c==key],key=lambda i:rows[i]) for key in sorted(set(concepts.values()))};assert len(groups)==1323 and all(len(v)==10 for v in groups.values())
    neighbor=torch.tensor([groups[concepts[i]] for i in range(13230)])
    own=torch.tensor([groups[concepts[i]].index(i) for i in range(13230)])
    cases=['ridge','full_head','fit_mean_update_only','individual_update_only','radial_update_only','tangent_update_only','sameconcept_shuffled_update']
    fields=['F_MSE','F_cosine','sameconcept_margin','sameconcept_Top1','norm_error_MSE','angle_error_MSE']
    values=torch.zeros(7,13230,6,dtype=torch.float64);covered=torch.zeros(13230,dtype=torch.bool);checks=[]
    for role in [0,1]:
        for fold in range(3):
            assert time.monotonic()-start<1000
            proof=json.loads((source/('fit_role'+str(role)+'_fold'+str(fold)+'.json')).read_text());mp=source/('chart_role'+str(role)+'_fold'+str(fold)+'.pt');assert sha(mp)==proof['model_sha256']
            saved=torch.load(mp,map_location='cpu',weights_only=True);train=torch.tensor([row_index[r] for r in saved['train_rows']]);test=torch.tensor([row_index[r] for r in saved['test_rows']]);assert not covered[test].any()
            head=OrdinaryCapacityControl().eval().requires_grad_(False);head.load_state_dict(saved['head'],strict=True)
            def feature(index):
                z=(x[index]-saved['mu_x'])/saved['scale_x'];return torch.cat([z,torch.ones(len(z),1)],-1)
            xt,xv=feature(train),feature(test);yt=(F[train]-saved['mu_y'])/saved['scale_y'];penalty=torch.eye(1025)*(.01*len(train));penalty[-1,-1]=0
            weight=torch.linalg.solve(xt.T@xt+penalty,xt.T)@yt
            with torch.inference_mode():
                zt=xt@weight;zv=xv@weight;base=saved['mu_y']+saved['scale_y']*zv;full=saved['mu_y']+saved['scale_y']*head(zv)
                mean_update=(saved['scale_y']*(head(zt)-zt)).mean(0).double();base=base.double();full=full.double();delta=full-base
                radial=(delta*base).sum(-1,keepdim=True)/base.square().sum(-1,keepdim=True)*base;tangent=delta-radial
                local={int(i):j for j,i in enumerate(test.tolist())};shift=[]
                for i in test.tolist():
                    members=groups[concepts[i]];other=members[(members.index(i)+1)%10];assert other in local;shift.append(local[other])
                alternatives=[base,full,base+mean_update,base+delta-mean_update,base+radial,base+tangent,base+delta[shift]]
                valid_scale=(1+(delta*base).sum(-1)/base.square().sum(-1));positive_scaling=bool((valid_scale>0).all())
                max_identity_error=0.;max_old_error=0.;rank_count_errors=0
                target=F[test].double();target_norm=target.norm(dim=-1)
                for ai,predicted in enumerate(alternatives):
                    norm=predicted.norm(dim=-1);unit=predicted/norm[:,None];target_unit=target/target_norm[:,None]
                    cosine=(unit*target_unit).sum(-1);mse=(predicted-target).square().mean(-1)
                    radial_error=(norm-target_norm).square()/1024;angle_error=2*norm*target_norm*(1-cosine)/1024
                    max_identity_error=max(max_identity_error,float((mse-radial_error-angle_error).abs().max()));assert max_identity_error<1e-10
                    margins=[]
                    for lo in range(0,len(test),128):
                        idx=test[lo:lo+128];candidate=F[neighbor[idx]].double();candidate=candidate/candidate.norm(dim=-1,keepdim=True)
                        scores=(unit[lo:lo+128,None,:]*candidate).sum(-1);correct=scores[torch.arange(len(idx)),own[idx]];scores[torch.arange(len(idx)),own[idx]]=-float('inf');margins.append(correct-scores.max(-1).values)
                    margin=torch.cat(margins);top1=(margin>0).double();values[ai,test]=torch.stack([mse,cosine,margin,top1,radial_error,angle_error],-1)
                    if ai in [0,1]:
                        previous='identity_ridge' if ai==0 else 'generic_output_head'
                        ref=torch.tensor([[original[(previous,int(i))][k] for k in ['original_F_MSE','original_F_cosine','same_concept_margin','same_concept_strict_Top1']] for i in test],dtype=torch.float64)
                        max_old_error=max(max_old_error,float((values[ai,test,:3]-ref[:,:3]).abs().max()));rank_count_errors+=int((values[ai,test,3]!=ref[:,3]).sum());assert max_old_error<2e-5 and rank_count_errors==0
                if positive_scaling:assert torch.allclose(values[4,test,1],values[0,test,1],atol=1e-12,rtol=0) and torch.equal(values[4,test,3],values[0,test,3])
                covered[test]=True;checks.append(dict(role=role,fold=fold,queries=len(test),mean_update_norm=float(mean_update.norm()),mean_delta_norm=float(delta.norm(dim=-1).mean()),max_MSE_decomposition_error=max_identity_error,max_previous_score_replay_error=max_old_error,previous_Top1_mismatches=rank_count_errors,all_radial_scales_positive=positive_scaling,model_sha256=proof['model_sha256']))
    assert covered.all() and torch.isfinite(values).all();keys=sorted(groups);rng=np.random.default_rng(1729);samples=rng.integers(0,1323,size=(1000,1323));comparisons=[]
    for ai,case in enumerate(cases[1:],1):
        for mi,metric in enumerate(fields):
            d=(values[ai,:,mi]-values[0,:,mi]).numpy();per=np.asarray([d[groups[k]].mean() for k in keys]);ci=np.quantile(per[samples].mean(1),[.025,.975]);comparisons.append(dict(case=case,reference='ridge',metric=metric,mean=float(d.mean()),concept95=[float(v) for v in ci]))
    with gzip.open(out/'all13230_7interventions.csv.gz','xt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case','array_row','concept_id']+fields);w.writeheader()
        for ai,case in enumerate(cases):
            for i in range(13230):w.writerow(dict(case=case,array_row=rows[i],concept_id=concepts[i],**{k:float(values[ai,i,j]) for j,k in enumerate(fields)}))
    result=dict(status='COMPLETE_FROZEN_ALL13230_SEVEN_COUNTERFACTUALS_NO_NEW_FIT_OR_IMAGES',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,queries=13230,concepts=1323,subjects=1,cases=cases,means={case:{k:float(values[ai,:,j].mean()) for j,k in enumerate(fields)} for ai,case in enumerate(cases)},comparisons=comparisons,checks=checks,source_completion_sha256=sha(source/'completion.json'),table_sha256=sha(out/'all13230_7interventions.csv.gz'),source_sha256=sha(__file__),new_training_updates=0,new_images=0,interpretation='Frozen feature-level interventions can attribute current readout score changes, not image-level causal benefits. Fit-derived common update uses no heldout answers. Sameconcept shuffled update is a diagnostic using known concepts, not a deployable inference method. Norm-angle MSE identity is classical, not novelty.',confirmed_second_method=False,SOTA=False)
    write('completion.json',result);assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<16<<20;print(json.dumps(dict(status=result['status'],means=result['means'],seconds=result['seconds'])),flush=True)
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED_NO_AUTO_RETRY',error=repr(e),traceback=traceback.format_exc()));raise
