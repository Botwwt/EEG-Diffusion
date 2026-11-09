"""Paired saved-image checks for two fixed ideas, with no result-based search."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_native_models_v1 import ROOT,WORK,write,torch
from score_full_manifest import interval


def read_run(run):
    folder=WORK/'evaluation'/run
    summary=json.loads((folder/'scores/summary.json').read_text())
    low=json.loads((folder/'low_level/summary.json').read_text())
    assert summary['status']=='COMPLETED' and summary['queries']==16600
    assert summary['gallery_images']==1660 and summary['repeat_protocol']=='R4'
    assert summary['subjects']==10 and summary['intervention']=='correct'
    assert low['verified_and_decoded']==16600
    rows=list(csv.DictReader((folder/'scores/per_query.csv').open()))
    values={r['query_id']:r for r in rows};assert len(rows)==len(values)==16600
    return summary,low,values


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',required=True);a=parser.parse_args()
    recipe=json.loads(Path(a.recipe).read_text());torch.set_num_threads(2)
    candidate=recipe['candidate'];run=recipe['run_name']
    out=WORK/'comparison/two_ideas_v1'/candidate;assert not out.exists();out.mkdir(parents=True)
    summary,low,rows=read_run(run);metrics=['DINO2AFC','DINO_Top1','SEED','CapSim','ObjectF1']
    controls={'parent':'protected_R4_parent_bounded_v1'}
    if candidate=='response_rank_v1':controls['ordinary']='protected_R4_ordinary_bounded_v1'
    results={};missing={}
    for name,reference_run in controls.items():
        if not (WORK/'evaluation'/reference_run/'scores/summary.json').exists():
            missing[name]='Matched reference incomplete; no alternative protocol substituted';continue
        base_summary,base_low,base=read_run(reference_run);assert set(rows)==set(base)
        for field in ['metric_version','metric_source_sha256','gallery_images','repeat_protocol','intervention','subjects']:
            assert summary[field]==base_summary[field],field
        paired=[]
        for query,row in rows.items():
            ref=base[query]
            for field in ['subject','stimulus_id','concept_id','image_seed','score_target_path','EEG_source_stimulus_id']:
                assert row[field]==ref[field],(query,field)
            assert row['generated_path']!=ref['generated_path']
            paired.append(dict(subject=row['subject'],concept_id=row['concept_id'],
                               **{k:float(row[k])-float(ref[k]) for k in metrics}))
        delta={k:interval(paired,k) for k in metrics}
        signal=(delta['DINO2AFC']['mean']>=.01 and delta['DINO2AFC']['CI95_subject_and_concept_bootstrap'][0]>0
            and delta['SEED']['mean']>0 and delta['SEED']['CI95_subject_and_concept_bootstrap'][0]>0
            and delta['ObjectF1']['mean']>=0)
        results[name]=dict(reference_run=reference_run,paired_differences=delta,practical_image_gain_signal=signal)
    report=dict(status='FULL_IMAGES_AND_SCORES_SAVED_PAIRED_CONTROLS_CHECKED',candidate=candidate,queries=16600,
        seed=1729,images_per_query=1,repeat='R4',gallery=1660,
        metrics={k:summary['summary'][k]['mean'] for k in metrics},
        low_level=low['summary_over_valid_images'],comparisons=results,missing_controls=missing,
        practical_image_gain_signal=not missing and all(r['practical_image_gain_signal'] for r in results.values()),
        no_automatic_retraining_or_strength_changes=True,independent_confirmation=False,mechanism_confirmed=False,
        required_if_positive=(['Same fixed attribution deletion/insertion vs random/low-importance and native MLP comparator',
                              'Image-level EEG permutation/missing controls; independent frozen protocol confirmation']
                             if candidate=='brainkan_full_v1' else
                             ['Same trained branch shuffled positive/negative teacher ranking control',
                              'Matched EEG permutation/missing controls; independent frozen protocol confirmation']),
        limitations=['Single training seed; previously exposed development V, descriptive intervals.',
                     'KAN attribution is exact only before final LayerNorm; no physiological causality claim.',
                     'DINO2AFC is not object recognition accuracy; ObjectF1 is a COCO/gender proxy.',
                     'Mechanism controls and blinded quality assessment are not supplied by this main comparison.'])
    write(out/'summary.json',report);print(json.dumps(report),flush=True)
    # Each closure saves a fresh five-method table, including explicit missing
    # arms, without replacing an earlier closure or borrowing another protocol.
    names={'parent':'protected_R4_parent_bounded_v1','ordinary':'protected_R4_ordinary_bounded_v1',
        'risk':'protected_R4_risk_bounded_v1','brainkan_full_v1':'protected_R4_brainkan_two_ideas_v1',
        'response_rank_v1':'protected_R4_response_rank_two_ideas_v1'}
    table=[]
    for name,method_run in names.items():
        item=dict(method=name,run_name=method_run,status='INCOMPLETE_NO_SCORE_IMPUTATION',queries=0,
                  **{k:None for k in metrics})
        if (WORK/'evaluation'/method_run/'scores/summary.json').exists():
            s,_,_=read_run(method_run)
            for field in ['metric_version','metric_source_sha256','gallery_images','repeat_protocol']:
                assert s[field]==summary[field],field
            item.update(status='FULL16600_SCORED',queries=16600,**{k:s['summary'][k]['mean'] for k in metrics})
        table.append(item)
    with (out/'unified_five_method_table.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)


if __name__=='__main__':main()
