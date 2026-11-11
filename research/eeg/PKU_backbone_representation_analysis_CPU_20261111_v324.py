"""All-query final representation diagnostics, after the fixed image study. No refit."""
import csv,json,os,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
from PKU_backbone_core_20261110_v322 import *
from PKU_ordinary_component_source_v253 import BoundedConditionAdapter
torch.set_num_threads(4);start=time.monotonic();out=R/'runs/EEG_backbone_representation_v324';out.mkdir();split,T,folds,post=input_contract()
assert json.loads((R/'runs/EEG_backbone_scores_v322/analysis_completion.json').read_text())['status']=='COMPLETE_FULL830_ALL5_BACKBONES_OWN_ORDINARY_THIRTEEN_ARMS_ALL14METRICS'
with (R/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as stream:metadata={int(r['array_row']):r for r in csv.DictReader(stream) if r['dataset_split']=='train'}
fit=[r for r in split['image_rows']['D'] if metadata[r]['concept_id'] in split['d_fit_concepts']]
check=[r for r in split['image_rows']['D'] if metadata[r]['concept_id'] not in split['d_fit_concepts']]
assert len(fit)==820 and len(check)==830
Fbank=torch.load(R/'runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt',map_location='cpu',weights_only=True)
def measure(c,rows):
    y=Fbank[rows].double();z=c.double();yn=F.normalize(y,dim=-1);zn=F.normalize(z,dim=-1);groups={}
    for i,r in enumerate(rows):groups.setdefault(metadata[r]['concept_id'],[]).append(i)
    assert all(len(g)==10 for g in groups.values())
    candidates=torch.tensor([groups[metadata[r]['concept_id']] for r in rows]);correct=torch.arange(len(rows));expected=(candidates==correct[:,None]).long().argmax(1);parts=[]
    for lo in range(0,len(rows),128):parts.append(torch.einsum('nd,nkd->nk',zn[lo:lo+128],yn[candidates[lo:lo+128]]))
    scores=torch.cat(parts);value=scores[correct,expected];wrong=candidates!=correct[:,None]
    # Full ten actual stimuli per concept, no nearest training-image replacement.
    result=dict(F_MSE=(z-y).square().mean(1),F_cosine=value,F_same_concept_Top1=(scores.argmax(1)==expected).double(),F_same_concept_2AFC=(((scores<value[:,None])&wrong).sum(1)+.5*((scores==value[:,None])&wrong).sum(1))/9,condition_norm=z.norm(dim=1))
    energy=float(z.square().sum(1).mean());common=float(z.mean(0).square().sum())
    return result,dict(condition_mean_energy=common,condition_total_second_moment=energy,condition_mean_energy_fraction=common/energy,Top1_tie_queries=int(((scores==scores.max(1).values[:,None]).sum(1)>1).sum()))
tables={};aggregate=[];parents={};metric_names=None
with (out/'per_query.csv').open('x',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['split','case','variant','array_row','concept_id','F_MSE','F_cosine','F_same_concept_Top1','F_same_concept_2AFC','condition_norm']);writer.writeheader()
    for case in CASES:
        parent=ROOT/case/'encoders';bank=torch.load(parent/'T_true_OOF_conditions.pt',map_location='cpu',weights_only=True);assert bank['array_rows'].tolist()==T
        full=torch.load(parent/'full/conditions.pt',map_location='cpu',weights_only=True);positions={r:i for i,r in enumerate(full['array_rows'].tolist())}
        affine_states=[torch.load(parent/role/'brain_final.pt',map_location='cpu',weights_only=True)['affine'] for role in ['oof0','oof1','full']]
        final=ROOT/case/'calibration/component_final.pt';head_state=torch.load(final,map_location='cpu',weights_only=True);head=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);head.load_state_dict(head_state['model'],strict=True);head.radius=.25*head_state['calibration']['selected_strength'];parents[case]=dict(full=sha(parent/'full/brain_final.pt'),ordinary=sha(final))
        for name,rows in [('T_upstream_OOF',T),('Dfit_calibration',fit),('Dcheck_exposed_development',check)]:
            c=bank['condition'] if name=='T_upstream_OOF' else full['condition'][[positions[r] for r in rows]]
            role_ids=folds if name=='T_upstream_OOF' else torch.full((len(rows),),2,dtype=torch.long);g=torch.stack([v['gamma'] for v in affine_states])[role_ids];b=torch.stack([v['beta'] for v in affine_states])[role_ids]
            with torch.no_grad():changed=torch.cat([head(c[lo:lo+128],g[lo:lo+128],b[lo:lo+128],gate_mode='fixed')[0] for lo in range(0,len(rows),128)])
            for variant,condition in [('base',c),('ordinary',changed)]:
                metrics,geometry=measure(condition,rows);tables[(name,case,variant)]=metrics;metric_names=list(metrics)
                aggregate.append(dict(split=name,case=case,variant=variant,queries=len(rows),means={k:float(v.mean()) for k,v in metrics.items()},geometry=geometry))
                for i,r in enumerate(rows):writer.writerow(dict(split=name,case=case,variant=variant,array_row=r,concept_id=metadata[r]['concept_id'],**{k:float(v[i]) for k,v in metrics.items()}))
        del head,bank,full,changed
comparisons=[]
for name,rows in [('T_upstream_OOF',T),('Dfit_calibration',fit),('Dcheck_exposed_development',check)]:
    keys=sorted({metadata[r]['concept_id'] for r in rows});index={k:[i for i,r in enumerate(rows) if metadata[r]['concept_id']==k] for k in keys};draw=np.random.default_rng(1729).integers(0,len(keys),size=(1000,len(keys)))
    contrasts=[(c+'_ordinary_minus_base',(c,'ordinary'),(c,'base')) for c in CASES]
    contrasts += [(c+'_'+v+'_minus_MLP17_'+v,(c,v),('MLP17',v)) for c in CASES[1:] for v in ['base','ordinary']]
    contrasts += [('Joint63_'+v+'_minus_'+c+'_'+v,('Joint63',v),(c,v)) for c in ['Separated63','Joint17'] for v in ['base','ordinary']]
    for label,left,right in contrasts:
        for metric in metric_names:
            d=tables[(name,)+left][metric]-tables[(name,)+right][metric];per=np.array([float(d[index[k]].mean()) for k in keys]);ci=np.quantile(per[draw].mean(1),[.025,.975]);comparisons.append(dict(split=name,comparison=label,metric=metric,mean=float(d.mean()),concept95=ci.tolist(),positive_concepts=int((per>0).sum()),negative_concepts=int((per<0).sum())))
result=dict(status='COMPLETE_ALL5_BASE_OWN_ORDINARY_FULL_T_DFIT_DCHECK_REPRESENTATION_ANALYSIS',rows=10*(13230+820+830),aggregate=aggregate,comparisons=comparisons,parents=parents,T_ordinary_is_training_not_heldout=True,Dfit_has_calibration_exposure=True,Dcheck_is_repeatedly_exposed_development=True,not_image_efficacy_or_SOTA=True,source_sha256=sha(__file__),per_query_sha256=sha(out/'per_query.csv'),seconds=time.monotonic()-start)
write(out/'completion.json',result);assert sum(p.stat().st_size for p in out.iterdir())<48<<20;print(json.dumps({k:v for k,v in result.items() if k not in ['aggregate','comparisons','parents']}),flush=True)
