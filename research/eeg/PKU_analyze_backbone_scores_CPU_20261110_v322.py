"""Full fixed intervention effects versus reused ordinary and full P controls."""
import csv,json,os,socket,time,traceback,sys
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');OUT=R/'runs/EEG_backbone_scores_v322';start=time.monotonic()
sys.path.insert(0,str(R/'package_cache/evaluation_runtime_v275'))
try:
    import numpy as np
    import torch
    from PKU_EEG_evaluation_core_20261109_v279 import METRICS,sha
    torch.set_num_threads(4)
    assert json.loads((OUT/'scoring_completion.json').read_text())['status']=='COMPLETE_DINO_ALL7_PAPER_METRICS_AND_EFFNET_SEED_STILL_PENDING'
    spec=json.loads((R/'code/EEG_backbone_policy_v322.json').read_text())
    new=spec['image_arms'];refs=['parent','matched_ordinary','P_abs_free'];arms=refs+new
    old=R/'runs/EEG_downstream_scores_v303';first=R/'runs/EEG_scores_v279'
    manifest=json.loads((OUT/'all_image_records.json').read_text());gallery=json.loads((OUT/'gallery.json').read_text())['rows']
    concepts=np.asarray([v['concept_id'] for v in gallery]);keys=sorted(set(concepts));assert len(keys)==83
    tables={arm:{} for arm in arms};replays=[]
    for arm in arms:
        root=OUT if arm in new else first if arm in ['parent','matched_ordinary'] else (R/'runs/EEG_preservation_scores_v317')
        assert [v['image_sha256'] for v in json.loads((root/'gallery.json').read_text())['rows']]==[v['image_sha256'] for v in gallery]
        for metric in METRICS:
            p=root/metric/(arm+'.pt');proof=json.loads(p.with_suffix('.json').read_text());assert sha(p)==proof['data_sha256']
            payload=torch.load(p,map_location='cpu',weights_only=True);assert payload['array_rows']==[v['array_row'] for v in gallery];f=payload['fields']
            if metric in ['DINO','Alex2','Alex5','Inception','CLIP']:
                s=f['full_gallery_scores'].numpy();correct=np.diag(s)[:,None];mask=~np.eye(830,dtype=bool);less=(s<correct)&mask;ties=(s==correct)&mask;same=concepts[:,None]==concepts[None,:]
                afc=(less.sum(1)+ties.sum(1)*(.5 if metric=='DINO' else 0))/829
                within=((less&same).sum(1)+(ties&same).sum(1)*(.5 if metric=='DINO' else 0))/9
                assert np.allclose(afc,f['two_afc'].numpy(),atol=1e-7,rtol=0) and np.allclose(within,f['same_concept_two_afc'].numpy(),atol=1e-7,rtol=0)
                name={'DINO':'DINO2AFC','Alex2':'AlexNet2_2AFC','Alex5':'AlexNet5_2AFC','Inception':'Inception2AFC','CLIP':'CLIP_L14_2AFC'}[metric];tables[arm][name]=afc
                if metric=='DINO':tables[arm].update(DINOTop1=f['top1'].numpy(),DINOTop5=f['top5'].numpy(),DINOCosine=np.diag(s),DINOSameConceptTop1=f['same_concept_top1'].numpy(),DINOSameConcept2AFC=within)
                replays.append(dict(arm=arm,metric=metric,full_matrix_pairs=830*830))
            elif metric in ['EffNet','SwAV']:tables[arm][metric+'Distance']=f['distance'].numpy()
            else:
                for k,v in f.items():tables[arm][k]=v.numpy()
    names=list(tables[arms[0]]);assert len(names)==14 and all(set(v)==set(names) for v in tables.values())
    rng=np.random.default_rng(1729);samples=rng.integers(0,83,size=(1000,83));summaries=[]
    effects=[]
    cases=spec['cases']
    for case in cases:
        effects.append((case+'_ordinary_minus_base',[(case+'_ordinary',1),(case+'_base',-1)]))
        for suffix in ['_base','_ordinary']:
            arm=case+suffix
            for ref in ['matched_ordinary','MLP17'+suffix]:
                if arm!=ref:effects.append((arm+'_minus_'+ref,[(arm,1),(ref,-1)]))
    for left,right in [('Joint63','Separated63'),('Joint63','Joint17')]:
        for suffix in ['_base','_ordinary']:effects.append((left+suffix+'_minus_'+right+suffix,[(left+suffix,1),(right+suffix,-1)]))
    for case in cases[1:]:effects.append((case+'_x_ordinary_interaction',[(case+'_ordinary',1),(case+'_base',-1),('MLP17_ordinary',-1),('MLP17_base',1)]))
    for label,terms in effects:
        for metric in names:
            delta=sum(tables[arm][metric]*coefficient for arm,coefficient in terms)
            per=np.asarray([delta[concepts==key].mean() for key in keys]);ci=np.quantile(per[samples].mean(1),[.025,.975])
            summaries.append(dict(comparison=label,metric=metric,mean=float(delta.mean()),concept95=[float(v) for v in ci],positive_concepts=int((per>0).sum()),negative_concepts=int((per<0).sum()),per_concept=per.tolist()))
    with (OUT/'per_query.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['arm','query_index','array_row','concept_id']+names);writer.writeheader()
        for arm in arms:
            for i,row in enumerate(gallery):writer.writerow(dict(arm=arm,query_index=i,array_row=row['array_row'],concept_id=row['concept_id'],**{name:float(tables[arm][name][i]) for name in names}))
    summary=dict(queries=830,subjects=1,repeat='R4',concepts=83,arms=arms,means={arm:{k:float(v.mean()) for k,v in values.items()} for arm,values in tables.items()},comparisons=summaries,interpretation='Five fixed25-epoch backbones, each full/oof0/oof1 and separately fitted original ordinary, full830 image matrix. No isolated novelty or SOTA conclusion; one exposed R4 subject and seed.')
    (OUT/'all_paired_summaries.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    done=dict(status='COMPLETE_FULL830_ALL5_BACKBONES_OWN_ORDINARY_THIRTEEN_ARMS_ALL14METRICS',CPU_job=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,queries=830,new_images=8300,arms=13,metric_cells=13*830*14,full_rank_replays=replays,summary_sha256=sha(OUT/'all_paired_summaries.json'),per_query_sha256=sha(OUT/'per_query.csv'),source_sha256=sha(__file__),EEG_encoder_replaced=True,UNet_NFE_per_image=4,new_encoder_updates=3375,new_ordinary_updates=12405,SEED_pending=True,automatic_new_fit=False,novel_method_established=False,SOTA_established=False)
    (OUT/'analysis_completion.json').write_text(json.dumps(done,indent=2)+'\n');print(json.dumps(done),flush=True)
except BaseException as e:
    (OUT/('analysis_failure_'+os.environ['SLURM_JOB_ID']+'.json')).write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2)+'\n');raise
