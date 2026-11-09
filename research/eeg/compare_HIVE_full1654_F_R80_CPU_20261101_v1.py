"""Fixed three fullscope methods; all16 metrics and no control-winner selection."""
import argparse,csv,gzip,json
from pathlib import Path
import brainhive_full1654_F_study_contract_20261101_v1 as G

def run(path):
    s=G.setup(path)
    import numpy as np
    from brainhive_loss_control_R80_statistics_CPU_20261030_v1 import COMMON,PAPER,paired,fixed_draws
    root=Path(s['output_root']);data={};manifests={};pins=[]
    def csvrows(pin):
        with G.check(pin).open(newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==2000 and len({r['query_id'] for r in rows})==2000
        return {r['query_id']:r for r in rows}
    data['parent']={k:csvrows(v) for k,v in s['evaluation']['parent_CSVs'].items()}
    manifests['parent']={r['query_id']:r for r in [json.loads(x) for x in G.check(s['evaluation']['baseline_scoring_manifest']).read_text().splitlines()]}
    for arm in G.ARMS:
        out=root/arm;artifact=G.read(out/'loss_control/artifact_audit.json');assert artifact['queries']==2000
        assert G.read(out/'evaluation/scores/summary.json')['status']=='COMPLETED'
        assert G.read(out/'evaluation/original_paper_metrics/completion.json')['queries']==2000
        files=dict(common=G.ref(out/'evaluation/scores/per_query.csv'),paper=G.ref(out/'evaluation/original_paper_metrics/per_query.csv'));pins+=list(files.values())
        data[arm]={k:csvrows(v) for k,v in files.items()};manifests[arm]={r['query_id']:r for r in [json.loads(x) for x in G.check(artifact['manifest']).read_text().splitlines()]}
    keys=sorted(data['parent']['common']);reference=data['parent']['common'];subjects=sorted({r['subject'] for r in reference.values()});concepts=sorted({r['concept_id'] for r in reference.values()});assert len(subjects)==10 and len(concepts)==200
    cells={(r['subject'],r['concept_id']):key for key,r in reference.items()};assert len(cells)==2000
    inodes=set()
    for role,parts in data.items():
        for rows in parts.values():
            assert sorted(rows)==keys
            for key in keys:
                assert all(rows[key][field]==reference[key][field] for field in ('subject','stimulus_id','concept_id','image_seed','score_target_path'))
        for key in keys:
            r=manifests[role][key];p=Path(r['generated_path']);st=p.stat();identity=st.st_dev,st.st_ino
            assert identity not in inodes and G.ref(p)['sha256']==r['generated_sha256'];inodes.add(identity)
            assert str(p)==parts['common'][key]['generated_path']
    assert len(inodes)==6000
    contrasts=(('bounded_F','parent'),('Euclidean_F','parent'),('bounded_F','Euclidean_F'));draws=fixed_draws();results={};undefined=[];out=root/'joint_comparison';out.mkdir()
    with gzip.open(out/'all3x16_paired_differences.csv.gz','xt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['comparison','metric','subject','concept_id','difference']);writer.writeheader()
        for left,right in contrasts:
            name=left+'_minus_'+right;results[name]={}
            for kind,metrics in [('common',COMMON),('paper',PAPER)]:
                for metric in metrics:
                    if any(data[r][kind][key][metric] in ('','None') for r in (left,right) for key in keys):
                        results[name][metric]=dict(status='UNKNOWN_UNDEFINED_FULL_DENOMINATOR_RETAINED',expected=2000);undefined.append([name,metric]);continue
                    arrays={r:np.array([[float(data[r][kind][cells[sub,con]][metric]) for con in concepts] for sub in subjects]) for r in (left,right)}
                    v=paired(arrays[left],arrays[right],*draws);v['status']='COMPLETE';v['lower_is_better']=metric in ('eval_effnet','eval_swav');results[name][metric]=v
                    for i,sub in enumerate(subjects):
                        for j,con in enumerate(concepts):writer.writerow(dict(comparison=name,metric=metric,subject=sub,concept_id=con,difference=float(arrays[left][i,j]-arrays[right][i,j])))
    support={}
    for name in ('bounded_F_minus_parent','bounded_F_minus_Euclidean_F'):
        r=results[name]
        support[name]=all(r[k].get('status')=='COMPLETE' for k in ('DINO2AFC','SEED','ObjectF1')) and r['DINO2AFC']['paired_left_minus_right']>=.01 and r['DINO2AFC']['descriptive_subject_concept_bootstrap95'][0]>0 and r['SEED']['descriptive_subject_concept_bootstrap95'][0]>0 and r['ObjectF1']['paired_left_minus_right']>=0
    G.write(out/'summary.json',dict(status='COMPLETE_ALL3_FIXED_FULL1654_R80_COMPARISONS_ALL16_METRICS',study=G.ref(path),results=results,undefined=undefined,predeclared_practical_support=support,all6000_image_inodes_and_hashes_verified=True,all96k_difference_cells_reported=True,input_new_CSVs=pins,training_seed=1729,independent_H=False,official200_previously_exposed=True,no_control_winner_selected=True,SOTA=False,CCF_A=False))
    print(json.dumps(dict(status='COMPLETE_ALL3_FIXED_FULL1654_R80_COMPARISONS_ALL16_METRICS',predeclared_practical_support=support,undefined=len(undefined))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);run(p.parse_args().study)
