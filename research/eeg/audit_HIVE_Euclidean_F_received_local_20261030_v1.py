"""Independent full tables/bootstrap/raw-cell arithmetic for three contrasts."""
import argparse,csv,gzip,json
from pathlib import Path
import numpy as np
COMMON=('DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED')
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
PAIRS=(('bounded_F','Euclidean_F'),('Euclidean_F','parent'),('ordinary','Euclidean_F'))
def run(root):
    root=Path(root);summary=json.loads((root/'actual/joint_comparison/summary.json').read_bytes());assert summary['queries_per_method']==2000
    data={}
    for role in ('parent','ordinary','bounded_F','Euclidean_F'):
        data[role]={}
        for kind in ('common','paper'):
            with (root/'tables'/role/(kind+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
            assert len(rows)==2000 and len({r['query_id'] for r in rows})==2000;data[role][kind]={r['query_id']:r for r in rows}
    reference=data['parent']['common'];subs=sorted({r['subject'] for r in reference.values()});cons=sorted({r['concept_id'] for r in reference.values()});assert len(subs)==10 and len(cons)==200
    cells={(r['subject'],r['concept_id']):key for key,r in reference.items()};assert len(cells)==2000
    for parts in data.values():
        for table in parts.values():
            assert set(table)==set(reference)
            for key,row in table.items():assert all(row[f]==reference[key][f] for f in ('subject','concept_id','stimulus_id','image_seed','score_target_path'))
    # Independent count-weight resampling, preserving interleaved subject then
    # concept RNG draws; do not import production paired()/fixed_draws().
    rng=np.random.default_rng(1729);sw=[];cw=[]
    for unused in range(1000):sw.append(np.bincount(rng.integers(10,size=10),minlength=10));cw.append(np.bincount(rng.integers(200,size=200),minlength=200))
    sw=np.asarray(sw);cw=np.asarray(cw);expected={};unknown=[];err=0.;means={};results={}
    for left,right in PAIRS:
        pair=left+'_minus_'+right;results[pair]={}
        for kind,names in (('common',COMMON),('paper',PAPER)):
            for metric in names:
                record=summary['results'][pair][metric]
                undefined=any(data[role][kind][key][metric] in ('','None') for role in (left,right) for key in reference)
                if undefined:
                    assert record['status']=='UNKNOWN_UNDEFINED_FULL_DENOMINATOR_RETAINED';unknown.append([pair,metric]);continue
                a,b=[np.asarray([[float(data[role][kind][cells[sub,con]][metric]) for con in cons] for sub in subs]) for role in (left,right)]
                assert np.isfinite(a).all() and np.isfinite(b).all();diff=a-b
                draws=np.einsum('rs,sc,rc->r',sw,diff,cw)/2000
                ci=np.quantile(draws,[.025,.975]);value=float(diff.mean())
                errors=[abs(value-record['paired_left_minus_right']),np.max(abs(ci-record['descriptive_subject_concept_bootstrap95'])),np.max(abs(diff.mean(axis=1)-record['per_subject_difference'])),abs(a.mean()-record['left_mean']),abs(b.mean()-record['right_mean'])]
                err=max(err,float(max(errors)));assert max(errors)<1e-12
                means.setdefault(left,{})[metric]=float(a.mean());means.setdefault(right,{})[metric]=float(b.mean())
                results[pair][metric]=dict(difference=value,descriptive_CI95=ci.tolist())
                for i,sub in enumerate(subs):
                    for j,con in enumerate(cons):expected[pair,metric,cells[sub,con]]=(sub,con,float(diff[i,j]))
    seen=set();rawerr=0.
    with gzip.open(root/'actual/joint_comparison/all3x16_full_paired_differences.csv.gz','rt',newline='') as f:
        for row in csv.DictReader(f):
            key=row['comparison'],row['metric'],row['query_id'];assert key in expected and key not in seen;seen.add(key)
            sub,con,value=expected[key];assert (row['subject'],row['concept_id'])==(sub,con);rawerr=max(rawerr,abs(float(row['difference'])-value))
    assert set(expected)==seen and rawerr==0
    report=dict(status='PASS_FULL_THREE_CONTRAST_ALL16_MEANS_BOOTSTRAP_AND_EVERY_STORED_RAW_CELL',methods=4,queries=2000,contrasts=3,metric_groups=len(results)*16-len(unknown),independently_verified_pairs=len(expected),maximum_statistic_error=err,maximum_raw_cell_error=rawerr,undefined_groups_retained=unknown,means=means,comparisons=results,training_seed=1729,seed_stability_assessed=False,SOTA_established=False)
    p=root/'independent_local_statistics.json';assert not p.exists();p.write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in report.items() if k not in ('means','comparisons')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
