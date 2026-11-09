"""All four fixed methods, all16 metrics and all five declared contrasts."""
import argparse,csv,datetime,gzip,hashlib,json
from pathlib import Path
import numpy as np
from brainhive_loss_control_R80_statistics_CPU_20261030_v1 import COMMON,PAPER,COMPARISONS,paired,fixed_draws


def ref(path):
    path=Path(path)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest)


def readcsv(path):
    with Path(path).open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==2000 and len({row['query_id'] for row in rows})==2000
    return {row['query_id']:row for row in rows}


def run(study_path):
    s=json.loads(Path(study_path).read_bytes());root=Path(s['output_root']);data={};pins=[];manifests={}
    for role,item in s['reference_manifests'].items():
        assert ref(item['path'])==item
        rows=[json.loads(line) for line in Path(item['path']).read_text().splitlines()];assert len(rows)==2000
        manifests[role]={row['query_id']:row for row in rows};assert len(manifests[role])==2000
    for role,paths in s['reference_CSVs'].items():
        for item in paths.values():assert ref(item['path'])==item
        data[role]={name:readcsv(item['path']) for name,item in paths.items()};pins+=list(paths.values())
    for arm in ('F_only','token_only'):
        arm_root=root/arm;artifact=json.loads((arm_root/'loss_control/artifact_audit.json').read_bytes());assert artifact['queries']==2000
        assert ref(artifact['manifest']['path'])==artifact['manifest']
        rows=[json.loads(line) for line in Path(artifact['manifest']['path']).read_text().splitlines()];assert len(rows)==2000
        manifests[arm]={row['query_id']:row for row in rows};assert len(manifests[arm])==2000
        common=json.loads((arm_root/'evaluation/scores/summary.json').read_bytes());paper=json.loads((arm_root/'evaluation/original_paper_metrics/completion.json').read_bytes())
        assert common['status']=='COMPLETED' and paper['queries']==2000
        paths=dict(common=arm_root/'evaluation/scores/per_query.csv',paper=arm_root/'evaluation/original_paper_metrics/per_query.csv')
        data[arm]={kind:readcsv(path) for kind,path in paths.items()};pins+=[ref(path) for path in paths.values()]
    keys=sorted(data['parent']['common']);reference=data['parent']['common']
    for role,parts in data.items():
        for part,rows in parts.items():
            assert sorted(rows)==keys
            for key in keys:
                for field in ('subject','stimulus_id','concept_id','image_seed','score_target_path'):assert rows[key][field]==reference[key][field]
    subjects=sorted({row['subject'] for row in reference.values()});concepts=sorted({row['concept_id'] for row in reference.values()})
    assert len(subjects)==10 and len(concepts)==200
    cells={(row['subject'],row['concept_id']):key for key,row in reference.items()};assert len(cells)==2000
    inodes=set();paths=set()
    for key in keys:
        assert len({data[role]['common'][key]['generated_path'] for role in data})==4
        for role in data:
            row=manifests[role][key];path=Path(row['generated_path']);stat=path.stat()
            assert str(path)==data[role]['common'][key]['generated_path'] and ref(path)['sha256']==row['generated_sha256']
            assert str(path) not in paths and (stat.st_dev,stat.st_ino) not in inodes
            paths.add(str(path));inodes.add((stat.st_dev,stat.st_ino))
        for arm in ('F_only','token_only'):
            for field in ('initial_noise_latents_sha256','ancestral_noise_observations'):
                assert manifests[arm][key][field]==manifests['ordinary'][key][field]
    assert len(inodes)==8000
    draws=fixed_draws();results={};out=root/'joint_comparison';out.mkdir();undefined=[]
    with gzip.open(out/'all5x16_full_paired_differences.csv.gz','xt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['comparison','metric','query_id','subject','concept_id','difference']);writer.writeheader()
        for left,right in COMPARISONS:
            contrast=left+'_minus_'+right;results[contrast]={}
            for kind,metrics in (('common',COMMON),('paper',PAPER)):
                for metric in metrics:
                    if any(data[role][kind][key][metric] in ('','None') for role in (left,right) for key in keys):
                        results[contrast][metric]=dict(status='UNKNOWN_UNDEFINED_FULL_DENOMINATOR_RETAINED',expected=2000);undefined.append([contrast,metric]);continue
                    arrays={role:np.asarray([[float(data[role][kind][cells[sub,con]][metric]) for con in concepts] for sub in subjects]) for role in (left,right)}
                    result=paired(arrays[left],arrays[right],*draws);result.update(status='COMPLETE',direction='lower_is_better' if metric in ('eval_effnet','eval_swav') else 'higher_is_better');results[contrast][metric]=result
                    for i,sub in enumerate(subjects):
                        for j,con in enumerate(concepts):writer.writerow(dict(comparison=contrast,metric=metric,query_id=cells[sub,con],subject=sub,concept_id=con,difference=float(arrays[left][i,j]-arrays[right][i,j])))
    result=dict(status='COMPLETE_SINGLE_SEED_R80_FIVE_FIXED_COMPARISONS_ALL16_METRICS' if not undefined else 'COMPLETE_WITH_UNDEFINED_METRICS_RETAINED_UNKNOWN',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),study=ref(study_path),queries_per_method=2000,subjects=subjects,concepts=200,training_seed=1729,bootstrap_rounds=1000,bootstrap_seed=1729,results=results,undefined=undefined,input_CSVs=pins,all_comparisons_reported=True,all8000_image_file_hashes_and_inodes_verified=True,all4000_control_noise_observations_equal_original_ordinary=True,source=ref(__file__),ordinary_unchanged=True,compute_matched=False,training_seed_stability_assessed=False,independent_H=False,SOTA_established=False,limits=['ExposedR80; singletrainingseed.','OrdinaryandcontrolheadsfitT1323,notfinalofficial1654componentfit.','OriginalgeneratorsupervisionusesmoretrainingcomputethanF/token-only.','Descriptivepairedbootstrapdoesnotcovertrainingseeduncertaintyorselection.','PositiveCIisnotbyitselfnoveltyorexclusivemechanismproof.'])
    with (out/'summary.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],queries_per_method=2000,comparisons=5,metrics=16,unknown=len(undefined))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',required=True);run(parser.parse_args().study)
