"""All4 fullscope methods, fixed5 comparisons/all16 metrics, no selected winner."""
import argparse
import csv
import gzip
import math
from pathlib import Path
import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F
E=F.E;K=F.K
CONTRASTS=(('ordinary_full1654','parent'),('bounded_F','parent'),('Euclidean_F','parent'),
           ('bounded_F','Euclidean_F'),('ordinary_full1654','bounded_F'))


def finite_columns(data,left,right,kind,metric,keys):
    try:return all(math.isfinite(float(data[role][kind][key][metric])) for role in (left,right) for key in keys)
    except (ValueError,TypeError):return False


def comparison_values(data,subjects,concepts,cells):
    import numpy as np
    from brainhive_loss_control_R80_statistics_CPU_20261030_v1 import COMMON,PAPER,paired,fixed_draws
    draws=fixed_draws();results={};differences={};undefined=[]
    for left,right in CONTRASTS:
        name=left+'_minus_'+right;results[name]={}
        for kind,metrics in (('common',COMMON),('paper',PAPER)):
            for metric in metrics:
                if not finite_columns(data,left,right,kind,metric,cells.values()):
                    results[name][metric]=dict(status='UNKNOWN_UNDEFINED_FULL_DENOMINATOR_RETAINED',expected=2000)
                    undefined.append([name,metric]);continue
                arrays={role:np.array([[float(data[role][kind][cells[sub,con]][metric]) for con in concepts] for sub in subjects])
                        for role in (left,right)}
                value=paired(arrays[left],arrays[right],*draws)
                value.update(status='COMPLETE',lower_is_better=metric in ('eval_effnet','eval_swav'))
                results[name][metric]=value;differences[name,metric]=arrays[left]-arrays[right]
    support={}
    for name in ('ordinary_full1654_minus_parent','ordinary_full1654_minus_bounded_F'):
        values=results[name]
        support[name]=(all(values[m].get('status')=='COMPLETE' for m in ('DINO2AFC','SEED','ObjectF1')) and
                       values['DINO2AFC']['paired_left_minus_right']>=.01 and
                       values['DINO2AFC']['descriptive_subject_concept_bootstrap95'][0]>0 and
                       values['SEED']['descriptive_subject_concept_bootstrap95'][0]>0 and
                       values['ObjectF1']['paired_left_minus_right']>=0)
    return results,differences,undefined,support


def run(protocol):
    s=F.setup_fit(protocol);root=Path(s['output_root']);old=K.read(K.check(s['completed_F_study']))
    old_study=K.read(K.check(old['study']));oldroot=Path(old_study['output_root'])
    def table(path):
        with Path(path).open(newline='') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==2000 and len({r['query_id'] for r in rows})==2000
        return {r['query_id']:r for r in rows}
    data={role:{kind:table(K.check(pin)) for kind,pin in pins.items()} for role,pins in s['reference_CSVs'].items()}
    manifests={role:[__import__('json').loads(x) for x in K.check(pin).read_text().splitlines()]
               for role,pin in s['reference_manifests'].items()}
    for role,folder in [('ordinary_full1654',root/'ordinary_full1654')]:
        artifact=K.read(folder/'loss_control/artifact_audit.json');assert artifact['queries']==2000
        assert K.read(folder/'evaluation/scores/summary.json')['status']=='COMPLETED'
        assert K.read(folder/'evaluation/original_paper_metrics/completion.json')['queries']==2000
        closure=K.read(folder/'evaluation/evaluation_closure.json')
        assert closure['status']=='COMPLETE_ACTUAL_NEW_ORDINARY_ALL16_SCORE_CALLS_AND_FILES'
        data[role]={kind:table(K.check(pin)) for kind,pin in closure['score_CSVs'].items()}
        manifests[role]=[__import__('json').loads(x) for x in K.check(artifact['manifest']).read_text().splitlines()]
    reference=data['parent']['common'];subjects=sorted({r['subject'] for r in reference.values()});concepts=sorted({r['concept_id'] for r in reference.values()})
    assert len(subjects)==10 and len(concepts)==200
    cells={(r['subject'],r['concept_id']):key for key,r in reference.items()};assert len(cells)==2000
    inodes=set()
    for role,tables in data.items():
        for kind,rows in tables.items():
            assert set(rows)==set(reference)
            for key,row in rows.items():
                assert all(row[field]==reference[key][field] for field in ('subject','stimulus_id','concept_id','image_seed','score_target_path'))
        for row in manifests[role]:
            path=Path(row['generated_path']);st=path.stat();inode=st.st_dev,st.st_ino
            assert inode not in inodes and K.ref(path)['sha256']==row['generated_sha256'];inodes.add(inode)
            assert str(path)==tables['common'][row['query_id']]['generated_path']
    assert len(inodes)==8000
    results,differences,undefined,support=comparison_values(data,subjects,concepts,cells)
    out=root/'joint_comparison';out.mkdir();count=0
    with gzip.open(out/'all5x16_paired_differences.csv.gz','xt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['comparison','metric','subject','concept_id','difference']);writer.writeheader()
        for (name,metric),delta in differences.items():
            for i,sub in enumerate(subjects):
                for j,con in enumerate(concepts):
                    writer.writerow(dict(comparison=name,metric=metric,subject=sub,concept_id=con,difference=float(delta[i,j])));count+=1
    K.write(out/'summary.json',dict(status='COMPLETE_ALL4_FULL1654_METHODS_FIXED5_ALL16_COMPARISONS',protocol=K.ref(protocol),
        results=results,undefined=undefined,actual_defined_difference_cells=count,full_expected_difference_cells=160000,
        predeclared_practical_support=support,all8000_image_inodes_and_hashes_verified=True,
        original_ordinary_practical_gate_passed=all(support.values()),
        training_seed=1729,training_compute_matched=False,no_control_winner_selected=True,
        official200_previously_exposed=True,independent_H=False,SOTA=False,CCF_A=False))
    print(__import__('json').dumps(dict(status='COMPLETE_ALL4_FULL1654_METHODS_FIXED5_ALL16_COMPARISONS',support=support,undefined=undefined)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);run(parser.parse_args().protocol)
