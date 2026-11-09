"""Close the completed native benchmark from saved query records only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv
import json
from pathlib import Path
import time

from brainhive_native_models_v1 import ROOT,WORK,sha,write,torch
from score_full_manifest import interval


def main():
    torch.set_num_threads(2)
    out=ROOT/'audit/BrainHIVE_native_R80_full_closure_20261020_v10.json';assert not out.exists()
    arms=['correct','permuted','missing'];common={};original={};summaries={};proofs={}
    native_names=['eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav']
    common_names=['DINO2AFC','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED']
    table=[]
    for arm in arms:
        folder=WORK/'evaluation'/('native_R80_'+arm+'_v1');source=WORK/'images'/('native_R80_'+arm+'_v1')
        pairing=json.loads((source/'pairing_audit.json').read_text());assert sha(source/'manifest.jsonl')==pairing['manifest_sha256']
        c=list(csv.DictReader((folder/'scores/per_query.csv').open()));summary=json.loads((folder/'scores/summary.json').read_text())
        n=list(csv.DictReader((folder/'original_paper_metrics_v1/per_query.csv').open()));ns=json.loads((folder/'original_paper_metrics_v1/completion.json').read_text())
        assert sha(folder/'original_paper_metrics_v1/per_query.csv')==ns['per_query_sha256']
        assert summary['status']=='COMPLETED' and ns['status']=='ALL2000_AUTHOR_CODE_QUERY_METRICS_COMPLETE'
        c={r['query_id']:r for r in c};n={r['query_id']:r for r in n};assert len(c)==len(n)==2000 and set(c)==set(n)
        for key in c:
            for field in ['subject','stimulus_id','concept_id','generated_path','score_target_path','image_seed']:
                assert c[key][field]==n[key][field],(arm,key,field)
        for metrics,rows,means in [(common_names,c,{k:summary['summary'][k]['mean'] for k in common_names}),
            (native_names,n,ns['means'])]:
            for metric in metrics:
                mean=sum(float(r[metric]) for r in rows.values())/2000
                assert abs(mean-means[metric])<1e-6,(arm,metric)
        common[arm]=c;original[arm]=n;summaries[arm]=summary
        proofs[arm]=dict(common_query_sha256=sha(folder/'scores/per_query.csv'),native_query_sha256=sha(folder/'original_paper_metrics_v1/per_query.csv'),
            native_completion_sha256=sha(folder/'original_paper_metrics_v1/completion.json'),pairing_sha256=sha(source/'pairing_audit.json'))
        table.append(dict(arm=arm,subjects=10,queries=2000,repeat_protocol='R80',gallery=200,
            **{k:summary['summary'][k]['mean'] for k in common_names},**ns['means']))
    differences={}
    for other in ['permuted','missing']:
        differences['correct_minus_'+other]={}
        assert set(common[other])==set(common['correct'])
        records=[]
        for key in common['correct']:
            a=common['correct'][key];b=common[other][key]
            for field in ['subject','stimulus_id','concept_id','score_target_path','image_seed']:assert a[field]==b[field]
            assert a['generated_path']!=b['generated_path']
            row=dict(subject=a['subject'],concept_id=a['concept_id'])
            row.update({k:float(a[k])-float(b[k]) for k in common_names})
            row.update({k:float(original['correct'][key][k])-float(original[other][key][k]) for k in native_names})
            records.append(row)
        for metric in common_names+native_names:differences['correct_minus_'+other][metric]=interval(records,metric)
    self_check=json.loads((WORK/'audit/original_real_image_checks_v1/completion.json').read_text())
    assert self_check['status']=='ALL200_REAL_IMAGE_IDENTITY_AND_FIXED_DERANGED_CHECKS_PASSED'
    report=dict(status='ALL6000_NATIVE_IMAGES_COMMON_AND_AUTHOR_EIGHT_QUERY_RESULTS_CLOSED_CPU',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),table=table,paired_differences=differences,
        proofs=proofs,original_real_image_checks=self_check,entry_sha256=sha(__file__),
        EEG_pairing_dependent_image_content_established=True,component_gain_established=False,
        exact_paper_multiple_seed_ten_image_mean=False,independent_H=False,
        inference_seed_or_checkpoint_selected=False,new_images_generated=0,
        limits=['R80 and 200 real-gallery only; do not rank against older R4/1660-gallery runs.',
            'ObjectF1 is a COCO+gender proxy, not THINGS object accuracy.',
            'EffNet/SwAV author metrics are distances; smaller is better.',
            'Single train seed; resampling subjects/concepts does not cover train seed variability.',
            'No blinded human image-quality ratings; no image-quality noninferiority claim.',
            'Native SSIM is higher for missing EEG; low-level similarity does not establish target content.'])
    write(out,report)
    with (ROOT/'reports/BrainHIVE_native_R80_comparison_20261020_v10.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(dict(status=report['status'],table=table,
        DINO_correct_minus_permuted=differences['correct_minus_permuted']['DINO2AFC'],
        SEED_correct_minus_permuted=differences['correct_minus_permuted']['SEED'])),flush=True)


if __name__=='__main__':main()
