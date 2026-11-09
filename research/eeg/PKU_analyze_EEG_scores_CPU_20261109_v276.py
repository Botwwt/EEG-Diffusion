"""Independent full-gallery rank replay, paired concept summaries and fixed image panels."""
import csv,hashlib,json,os,socket,sys,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_scores_v276';start=time.monotonic();sys.path.insert(0,str(R/'package_cache/evaluation_runtime_v275'))
def write(p,r):
    b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
try:
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from PKU_EEG_evaluation_core_20261109_v276 import METRICS,sha
    torch.set_num_threads(4);done=json.loads((out/'scoring_completion.json').read_text());assert done['status']=='COMPLETE_DINO_ALL7_PAPER_METRICS_AND_EFFNET_SEED_STILL_PENDING'
    assert not (out/'analysis_completion.json').exists();records=json.loads((out/'all_image_records.json').read_text());arms=records['arms'];records=records['records'];gallery=json.loads((out/'gallery.json').read_text())['rows']
    concepts=np.asarray([r['concept_id'] for r in gallery]);keys=sorted(set(concepts));assert len(keys)==83 and all((concepts==key).sum()==10 for key in keys)
    tables={arm:{} for arm in arms};replays=[];metric_files=[]
    for metric in METRICS:
        for arm in arms:
            p=out/metric/(arm+'.pt');proof=json.loads(p.with_suffix('.json').read_text());assert sha(p)==proof['data_sha256']
            saved=torch.load(p,map_location='cpu',weights_only=True);fields=saved['fields'];assert saved['array_rows']==[r['array_row'] for r in gallery]
            if metric in ['DINO','Alex2','Alex5','Inception','CLIP']:
                s=fields['full_gallery_scores'].numpy();assert s.shape==(830,830);correct=np.diag(s)[:,None];mask=~np.eye(830,dtype=bool)
                strict=(s<correct)&mask;ties=(s==correct)&mask;same=concepts[:,None]==concepts[None,:]
                afc=(strict.sum(1)+ties.sum(1)*(.5 if metric=='DINO' else 0))/829
                within=((strict&same).sum(1)+(ties&same).sum(1)*(.5 if metric=='DINO' else 0))/9
                assert np.allclose(afc,fields['two_afc'].numpy(),atol=1e-7,rtol=0) and np.allclose(within,fields['same_concept_two_afc'].numpy(),atol=1e-7,rtol=0)
                assert np.allclose(np.diag(s),fields['correct_similarity'].numpy(),atol=0,rtol=0)
                assert np.array_equal(ties.sum(1),fields['exact_negative_ties'].numpy())
                # Top1/5 may be ambiguous at an exact tie; verify the admissible interval instead of inventing a tie-breaker.
                above=(s>correct).sum(1);equal=(s==correct).sum(1)
                top1=fields['top1'].numpy();top5=fields['top5'].numpy();assert np.all(top1[above>0]==0) and np.all(top1[(above==0)&(equal==1)]==1)
                assert np.all(top5[above>=5]==0) and np.all(top5[(above+equal)<=5]==1)
                name={'DINO':'DINO2AFC','Alex2':'AlexNet2_2AFC','Alex5':'AlexNet5_2AFC','Inception':'Inception2AFC','CLIP':'CLIP_L14_2AFC'}[metric];tables[arm][name]=afc
                if metric=='DINO':
                    tables[arm].update(DINOTop1=top1,DINOTop5=top5,DINOCosine=np.diag(s),DINOSameConceptTop1=fields['same_concept_top1'].numpy(),DINOSameConcept2AFC=within)
                replays.append(dict(metric=metric,arm=arm,rows=830,all_pairs=688900,exact_negative_ties=int(ties.sum()),primary_rank_replayed=True,top1_exact_tie_queries=int(((above==0)&(equal>1)).sum())))
            elif metric in ['EffNet','SwAV']:tables[arm][metric+'Distance']=fields['distance'].numpy()
            else:
                for k,v in fields.items():tables[arm][k]=v.numpy()
            metric_files.append(dict(path=str(p),sha256=proof['data_sha256']))
    names=list(tables[arms[0]]);assert len(names)==14 and all(set(tables[arm])==set(names) for arm in arms)
    cube=np.stack([np.stack([tables[arm][name] for name in names]) for arm in arms]);assert cube.shape==(14,14,830) and np.isfinite(cube).all()
    comparisons=[('ordinary_minus_parent','matched_ordinary','parent')]
    cases=['A_positive','A_pair','A_reverse','B_residual','B_raw','B_shuffle']
    comparisons.extend((case+'_minus_'+ref,case,ref) for case in cases for ref in ['parent','matched_ordinary'])
    comparisons.extend([('A_pair_minus_positive','A_pair','A_positive'),('A_pair_minus_reverse','A_pair','A_reverse'),('B_residual_minus_raw','B_residual','B_raw'),('B_residual_minus_shuffle','B_residual','B_shuffle')])
    comparisons.extend((case+'_frozen_O_effect',case+'_plus_frozen_ordinary',case) for case in cases)
    rng=np.random.default_rng(1729);samples=rng.integers(0,83,size=(1000,83));summaries=[]
    def summarize(label,name,difference,kind):
        per=np.asarray([difference[concepts==key].mean() for key in keys]);boot=per[samples].mean(1);ci=np.quantile(boot,[.025,.975]);summaries.append(dict(comparison=label,metric=name,kind=kind,mean=float(difference.mean()),concept95_low=float(ci[0]),concept95_high=float(ci[1]),positive_concepts=int((per>0).sum()),negative_concepts=int((per<0).sum()),per_concept=per.tolist(),higher_is_better=not name.endswith('Distance')))
    for label,left,right in comparisons:
        for name in names:summarize(label,name,tables[left][name]-tables[right][name],'paired_method_difference')
    for case in cases:
        for name in names:summarize(case+'_O_interaction',name,(tables[case+'_plus_frozen_ordinary'][name]-tables[case][name])-(tables['matched_ordinary'][name]-tables['parent'][name]),'2x2_frozen_postprocessor_interaction')
    np.savez_compressed(out/'all830_all14arms_all14metrics.npz',values=cube,arms=np.asarray(arms),metrics=np.asarray(names),array_rows=np.asarray([r['array_row'] for r in gallery]),concept_ids=concepts,bootstrap_concept_indices=samples)
    with (out/'per_query.csv').open('x',newline='') as f:
        fields=['arm','query_index','array_row','stimulus_id','concept_id','image_sha256','initial_noise_sha256']+names;writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for ai,arm in enumerate(arms):
            for qi,r in enumerate(gallery):
                image=records[qi*14+ai];writer.writerow(dict(arm=arm,query_index=qi,array_row=r['array_row'],stimulus_id=r['stimulus_id'],concept_id=r['concept_id'],image_sha256=image['file_sha256'],initial_noise_sha256=image['initial_noise_sha256'],**{n:float(tables[arm][n][qi]) for n in names}))
    write(out/'all_paired_summaries.json',dict(queries=830,subjects=1,concepts=83,bootstrap=1000,seed=1729,means={arm:{n:float(tables[arm][n].mean()) for n in names} for arm in arms},comparisons=summaries,interpretation='Descriptive concept resampling, not trainingseed/subject uncertainty; all fixed comparisons exploratory, no multiplicity correction or SOTA claim'))
    panels=[];query_groups=[[0,1,270,271],[550,551,820,821]]
    for index,queries in enumerate(query_groups):
        canvas=Image.new('RGB',(15*128,4*158),'white');draw=ImageDraw.Draw(canvas)
        for line,qi in enumerate(queries):
            draw.text((0,line*158),gallery[qi]['stimulus_id'][:22],fill='black')
            for col in range(15):
                path=gallery[qi]['path'] if col==0 else records[qi*14+col-1]['image_path'];label='target' if col==0 else arms[col-1]
                with Image.open(path) as im:canvas.paste(im.convert('RGB').resize((128,128),Image.Resampling.BILINEAR),(col*128,line*158+30))
                draw.text((col*128,line*158+15),label[:19],fill='black')
        p=out/('fixed_panel_'+str(index+1)+'.png');canvas.save(p);panels.append(dict(path=str(p),sha256=sha(p),query_indices=queries,selected_by_scores=False))
    write(out/'analysis_completion.json',dict(status='COMPLETE_ALL830_14ARMS_DINO_PAPER7_EFFNET_PAIRED_ANALYSIS_SEED_PENDING',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,queries=830,subjects=1,arms=14,metrics=14,
        rank_replay=replays,metric_files=metric_files,comparison_rows=len(summaries),panels=panels,all_values_sha256=sha(out/'all830_all14arms_all14metrics.npz'),per_query_csv_sha256=sha(out/'per_query.csv'),summary_sha256=sha(out/'all_paired_summaries.json'),
        SEED_ObjectF1_CapSim_not_yet_available=True,independent_unexposed_H=False,SOTA_or_CCFA_established=False,automatic_stop_on_one_metric=False,promotion_or_next_fit_automatic=False,source_sha256=sha(__file__)))
    print('COMPLETE_ALL_FIXED_COMPARISONS_AND_INDEPENDENT_FULL_RANK_REPLAY',flush=True)
except BaseException as e:
    write(out/('analysis_failure_'+os.environ['SLURM_JOB_ID']+'.json'),dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
