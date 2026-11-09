"""Three predeclared contrasts, unchanged full10x200/all16 paired arithmetic."""
import argparse, csv, datetime, gzip, hashlib, json
from pathlib import Path
import numpy as np
from brainhive_loss_control_R80_statistics_CPU_20261030_v1 import COMMON, PAPER, paired, fixed_draws

COMPARISONS=(('bounded_F','Euclidean_F'),('Euclidean_F','parent'),('ordinary','Euclidean_F'))
def ref(p):
    p=Path(p)
    with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=digest)
def read(p): return json.loads(Path(p).read_bytes())
def checked(x): assert ref(x['path'])==x; return Path(x['path'])
def table(p):
    with Path(p).open(newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==len({x['query_id'] for x in rows})==2000
    return {x['query_id']:x for x in rows}


def run(path):
    s=read(path); assert s['status']=='REGISTERED_SINGLE1729_EUCLIDEAN_F_GEOMETRY_FULL_LIFECYCLE'
    root=Path(s['output_root']); candidate=root/'R80'; data={}; manifests={}; pins=[]
    for role in ('parent','ordinary','bounded_F'):
        parts=s['reference_CSVs'][role]; data[role]={kind:table(checked(item)) for kind,item in parts.items()};pins+=list(parts.values())
        rows=[json.loads(x) for x in checked(s['reference_manifests'][role]).read_text().splitlines()]
        assert len(rows)==2000; manifests[role]={x['query_id']:x for x in rows};assert len(manifests[role])==2000
    artifact=read(candidate/'loss_control/artifact_audit.json')
    assert artifact['queries']==2000 and artifact['actual_corrected_scalar_count']==2048000
    rows=[json.loads(x) for x in checked(artifact['manifest']).read_text().splitlines()]
    manifests['Euclidean_F']={x['query_id']:x for x in rows};assert len(manifests['Euclidean_F'])==2000
    paths=dict(common=candidate/'evaluation/scores/per_query.csv',paper=candidate/'evaluation/original_paper_metrics/per_query.csv')
    assert read(candidate/'evaluation/scores/summary.json')['status']=='COMPLETED'
    assert read(candidate/'evaluation/original_paper_metrics/completion.json')['queries']==2000
    data['Euclidean_F']={kind:table(p) for kind,p in paths.items()};pins+=[ref(p) for p in paths.values()]
    reference=data['parent']['common'];keys=sorted(reference);subjects=sorted({x['subject'] for x in reference.values()});concepts=sorted({x['concept_id'] for x in reference.values()})
    assert len(subjects)==10 and len(concepts)==200
    cells={(x['subject'],x['concept_id']):key for key,x in reference.items()};assert len(cells)==2000
    inodes=set();file_paths=set()
    for role,parts in data.items():
        for kind,rows in parts.items():
            assert sorted(rows)==keys
            for key in keys:
                assert all(rows[key][field]==reference[key][field] for field in ('subject','stimulus_id','concept_id','image_seed','score_target_path'))
        for key in keys:
            row=manifests[role][key];p=Path(row['generated_path']);st=p.stat()
            assert str(p)==parts['common'][key]['generated_path'] and ref(p)['sha256']==row['generated_sha256']
            assert str(p) not in file_paths and (st.st_dev,st.st_ino) not in inodes
            file_paths.add(str(p));inodes.add((st.st_dev,st.st_ino))
    assert len(inodes)==8000
    for key in keys:
        for field in ('initial_noise_latents_sha256','ancestral_noise_observations'):
            assert manifests['Euclidean_F'][key][field]==manifests['ordinary'][key][field]==manifests['bounded_F'][key][field]
    out=root/'joint_comparison';out.mkdir();draws=fixed_draws();results={};undefined=[]
    with gzip.open(out/'all3x16_full_paired_differences.csv.gz','xt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['comparison','metric','query_id','subject','concept_id','difference']);writer.writeheader()
        for left,right in COMPARISONS:
            key=left+'_minus_'+right;results[key]={}
            for kind,names in (('common',COMMON),('paper',PAPER)):
                for name in names:
                    if any(data[role][kind][query][name] in ('','None') for role in (left,right) for query in keys):
                        results[key][name]=dict(status='UNKNOWN_UNDEFINED_FULL_DENOMINATOR_RETAINED',expected=2000);undefined.append([key,name]);continue
                    values={role:np.asarray([[float(data[role][kind][cells[sub,con]][name]) for con in concepts] for sub in subjects]) for role in (left,right)}
                    result=paired(values[left],values[right],*draws);result.update(status='COMPLETE',direction='lower_is_better' if name in ('eval_effnet','eval_swav') else 'higher_is_better');results[key][name]=result
                    for i,sub in enumerate(subjects):
                        for j,con in enumerate(concepts): writer.writerow(dict(comparison=key,metric=name,query_id=cells[sub,con],subject=sub,concept_id=con,difference=float(values[left][i,j]-values[right][i,j])))
    result=dict(status='COMPLETE_EUCLIDEAN_F_ALL16_THREE_FIXED_COMPARISONS' if not undefined else 'COMPLETE_WITH_UNDEFINED_METRICS_RETAINED_UNKNOWN',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),study=ref(path),results=results,undefined=undefined,
        queries_per_method=2000,subjects=subjects,concepts=200,training_seed=1729,bootstrap_rounds=1000,bootstrap_seed=1729,
        all8000_image_file_hashes_and_inodes_verified=True,all2000_candidate_noise_observations_match_cached_references=True,
        input_CSVs=pins,boundedF_and_EuclideanF_same_objective_and_updates=True,
        original_ordinary_training_compute_not_matched=True,whole_geometry_only_not_suboperation_attribution=True,
        R80_exposed_and_used_for_direction_selection=True,independent_H=False,training_seed_stability_assessed=False,SOTA_established=False)
    with (out/'summary.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],queries=2000,metrics=16,comparisons=3,unknown=len(undefined))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();run(a.study)
