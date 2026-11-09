"""Fixed source-minus-mean and mean-minus-parent comparisons, full2000/all16."""
import argparse
import csv
import gzip
import math
from pathlib import Path
import brainhive_training_mean_R80_contract_CPU_20261101_v1 as M


def comparison_values(data,subjects,concepts,cells):
    import numpy as np
    from brainhive_loss_control_R80_statistics_CPU_20261030_v1 import COMMON,PAPER,paired,fixed_draws
    draws=fixed_draws();results={};differences={};undefined=[];support={}
    for left,right in M.CONTRASTS:
        name=left+'_minus_'+right;results[name]={}
        for kind,metrics in (('common',COMMON),('paper',PAPER)):
            for metric in metrics:
                try:valid=all(math.isfinite(float(data[role][kind][key][metric])) for role in (left,right) for key in cells.values())
                except (ValueError,TypeError):valid=False
                if not valid:
                    results[name][metric]=dict(status='UNKNOWN_UNDEFINED_FULL_DENOMINATOR_RETAINED',expected=2000)
                    undefined.append([name,metric]);continue
                arrays={role:np.array([[float(data[role][kind][cells[sub,con]][metric]) for con in concepts] for sub in subjects])
                        for role in (left,right)}
                result=paired(arrays[left],arrays[right],*draws)
                result.update(status='COMPLETE',lower_is_better=metric in ('eval_effnet','eval_swav'))
                results[name][metric]=result;differences[name,metric]=arrays[left]-arrays[right]
        values=results[name]
        support[name]=(all(values[m].get('status')=='COMPLETE' for m in ('DINO2AFC','SEED','ObjectF1')) and
            values['DINO2AFC']['paired_left_minus_right']>=.01 and values['DINO2AFC']['descriptive_subject_concept_bootstrap95'][0]>0 and
            values['SEED']['descriptive_subject_concept_bootstrap95'][0]>0 and values['ObjectF1']['paired_left_minus_right']>=0)
    return results,differences,undefined,support


def run(protocol):
    s,parent,F=M.setup(protocol);binding=M.read(M.OUT/'actual_source_data_closure.json')
    def table(pin):
        with M.checked(pin).open(newline='') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==2000 and len({row['query_id'] for row in rows})==2000
        return {row['query_id']:row for row in rows}
    data={role:{kind:table(pin) for kind,pin in values.items()} for role,values in binding['source_CSVs'].items()}
    manifests={role:[__import__('json').loads(x) for x in M.checked(pin).read_text().splitlines()]
               for role,pin in binding['source_manifests'].items()}
    for role in M.ROLES:
        root=M.OUT/role;artifact=M.read(root/'loss_control/artifact_audit.json')
        assert artifact['status']=='PASS_TRAIN_MEAN_ALL2000_R80_ARTIFACTS_AND_NUMPY_GEOMETRY'
        closure=M.read(root/'evaluation/evaluation_closure.json')
        assert closure['status']=='COMPLETE_TRAIN_MEAN_ACTUAL_ALL16_SCORE_CALLS_AND_FILES' and closure['queries']==2000
        assert closure['manifest']==artifact['manifest'] and closure['all16_metrics_included'] is True
        data[role]={kind:table(pin) for kind,pin in closure['score_CSVs'].items()}
        manifests[role]=[__import__('json').loads(x) for x in M.checked(artifact['manifest']).read_text().splitlines()]
    assert set(data)=={'parent','bounded_F','ordinary_full1654',*M.ROLES}
    reference=data['parent']['common'];subjects=sorted({row['subject'] for row in reference.values()})
    concepts=sorted({row['concept_id'] for row in reference.values()});assert len(subjects)==10 and len(concepts)==200
    cells={(row['subject'],row['concept_id']):key for key,row in reference.items()};assert len(cells)==2000
    inodes=set()
    for role,tables in data.items():
        for kind,rows in tables.items():
            assert set(rows)==set(reference)
            for key,row in rows.items():
                assert all(row[field]==reference[key][field] for field in ('subject','stimulus_id','concept_id','image_seed','score_target_path'))
        assert len(manifests[role])==2000
        for row in manifests[role]:
            path=Path(row['generated_path']);st=path.stat();inode=st.st_dev,st.st_ino
            assert inode not in inodes and M.ref(path)['sha256']==row['generated_sha256'];inodes.add(inode)
            assert str(path)==tables['common'][row['query_id']]['generated_path']
    assert len(inodes)==10000
    results,differences,undefined,support=comparison_values(data,subjects,concepts,cells)
    output=M.OUT/'fixed4_comparison';output.mkdir();count=0
    with gzip.open(output/'all4x16_paired_differences.csv.gz','xt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['comparison','metric','subject','concept_id','difference']);writer.writeheader()
        for (name,metric),delta in differences.items():
            for i,sub in enumerate(subjects):
                for j,con in enumerate(concepts):
                    writer.writerow(dict(comparison=name,metric=metric,subject=sub,concept_id=con,difference=float(delta[i,j])));count+=1
    result=dict(status='COMPLETE_TRAIN_MEAN_FIXED4_ALL16_FULL2000_COMPARISONS',protocol=M.ref(protocol),
        results=results,undefined=undefined,predeclared_practical_support=support,actual_defined_difference_cells=count,
        full_expected_difference_cells=128000,all10000_source_and_control_image_inodes_and_hashes_verified=True,
        no_equivalence_conclusion_from_crossing_zero=True,no_control_winner_selected=True,
        mean_uses_R80_inputs_or_scores=False,source_training_cost_inherited=True,new_training_updates=0,
        training_seed=1729,official200_previously_exposed=True,independent_H=False,second_method=False,SOTA=False,CCF_A=False)
    M.write(output/'summary.json',result);print(__import__('json').dumps(dict(status=result['status'],support=support,undefined=undefined)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
