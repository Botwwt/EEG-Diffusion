"""Full saved-query comparison; no selection, image regeneration, or test unlock."""
import argparse
import csv
import html
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_native_models_v1 import ROOT,WORK,sha,write,torch
from score_full_manifest import interval


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',required=True);a=parser.parse_args()
    recipe=json.loads(Path(a.recipe).read_text());torch.set_num_threads(2)
    out=WORK/'comparison/bounded_v1';assert not out.exists();out.mkdir(parents=True)
    scores={};manifests={};table=[];metrics=['DINO2AFC','DINO_Top1','SEED','CapSim','ObjectF1']
    for arm in ['parent','ordinary','risk']:
        run=recipe['run_names'][arm];folder=WORK/'evaluation'/run
        summary=json.loads((folder/'scores/summary.json').read_text());assert summary['status']=='COMPLETED' and summary['queries']==16600
        low=json.loads((folder/'low_level/summary.json').read_text());assert low['verified_and_decoded']==16600
        rows=list(csv.DictReader((folder/'scores/per_query.csv').open()));assert len(rows)==16600
        scores[arm]={r['query_id']:r for r in rows};assert len(scores[arm])==16600
        manifests[arm]={r['query_id']:r for r in map(json.loads,(WORK/'images'/run/'manifest.jsonl').read_text().splitlines())}
        for q,row in scores[arm].items():assert row['generated_path']==manifests[arm][q]['generated_path']
        table.append(dict(arm=arm,subjects=10,queries=16600,repeat='R4',gallery=1660,
            **{k:summary['summary'][k]['mean'] for k in metrics},
            PixCorr=low['summary_over_valid_images']['PixCorr_RGB'],SSIM=low['summary_over_valid_images']['SSIM_Y']))
    differences={}
    for reference in ['parent','ordinary']:
        assert set(scores['risk'])==set(scores[reference])
        paired=[]
        for q,row in scores['risk'].items():
            base=scores[reference][q]
            for field in ['subject','stimulus_id','concept_id','image_seed','score_target_path','EEG_source_stimulus_id']:assert row[field]==base[field]
            assert row['generated_path']!=base['generated_path']
            paired.append(dict(subject=row['subject'],concept_id=row['concept_id'],**{k:float(row[k])-float(base[k]) for k in metrics}))
        differences['risk_minus_'+reference]={k:interval(paired,k) for k in metrics}
    signals=[]
    for reference in ['parent','ordinary']:
        d=differences['risk_minus_'+reference]
        signals.append(d['DINO2AFC']['mean']>=.01 and d['DINO2AFC']['CI95_subject_and_concept_bootstrap'][0]>0
            and d['SEED']['mean']>0 and d['SEED']['CI95_subject_and_concept_bootstrap'][0]>0 and d['ObjectF1']['mean']>=0)
    report=dict(status='ALL_THREE_FULL16600_QUERY_IMAGE_ARMS_PAIRED_COMPARISON_COMPLETE',table=table,paired_differences=differences,
        practical_numerical_gain_signal=all(signals),mechanism_confirmed=False,independent_SOTA_established=False,
        quality_noninferiority_established=False,human_blind_ratings_performed=False,
        required_contingent_confirmation=['Same trained risk branch with fixed Dfit mean gate','Fixed shuffled-risk ablation',
            'Correct/permuted/missing EEG at this R4 protocol','Fixed no-reference quality proxy and actual blinded ratings if collected'],
        no_automatic_hyperparameter_search=True,official_test_used_to_tune=False,single_training_seed=1729,
        limits=['Public R80 reproduction and these R4 development scores are separate protocols.',
            'V was previously exposed; these intervals are descriptive, not independent confirmation.',
            'ObjectF1 is a COCO+gender proxy; DINO2AFC is not object accuracy.',
            'A positive three-arm result needs registered gate and EEG controls before a risk-mechanism claim.'])
    write(out/'summary.json',report)
    with (out/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    # Full paged gallery references actual originals; no duplicate thumbnails.
    gallery=WORK/'gallery/bounded_component_V_v1';gallery.mkdir(exist_ok=False);assets=gallery/'assets';assets.mkdir()
    storage=json.loads(Path(recipe['storage_lifecycle_receipt']).read_text())
    for i,root in enumerate(storage['allowed_private_roots']):(assets/('volume'+str(i))).symlink_to(root,target_is_directory=True)
    (assets/'targets').symlink_to(WORK/'gallery/V',target_is_directory=True)
    def relative(path):
        path=Path(path)
        for i,root in enumerate(storage['allowed_private_roots']):
            try:return 'assets/volume'+str(i)+'/'+path.relative_to(root).as_posix()
            except ValueError:pass
        return 'assets/targets/'+path.name
    keys=sorted(scores['parent']);pages=[]
    for page,start in enumerate(range(0,len(keys),100)):
        body=['<!doctype html><meta charset="utf-8"><title>Complete BrainHIVE V comparison</title>',
            '<style>body{font-family:system-ui}table{border-collapse:collapse}td{border:1px solid #ddd;padding:5px}img{width:160px;height:160px;object-fit:contain}</style>',
            '<p>All 16600 queries; fixed seed1729, one image/query. Not blinded human ratings.</p><a href="index.html">Index</a>',
            '<table><tr><th>Query</th><th>Target</th><th>Parent</th><th>Ordinary</th><th>Risk</th></tr>']
        for q in keys[start:start+100]:
            ref=scores['parent'][q];cells=['<td>'+html.escape(q)+'</td>']
            paths=[ref['score_target_path']]+[scores[arm][q]['generated_path'] for arm in ['parent','ordinary','risk']]
            for j,path in enumerate(paths):
                assert Path(path).is_file()
                extra='' if j==0 else '<br>DINO '+format(float(scores[['parent','ordinary','risk'][j-1]][q]['DINO2AFC']),'.4f')
                cells.append('<td><img loading="lazy" decoding="async" src="'+html.escape(relative(path),quote=True)+'">'+extra+'</td>')
            body.append('<tr>'+''.join(cells)+'</tr>')
        body.append('</table>');name=f'page_{page:03d}.html';(gallery/name).write_text('\n'.join(body));pages.append(name)
    (gallery/'index.html').write_text('<meta charset="utf-8"><h1>Complete 16600-query gallery</h1>'+' '.join('<a href="'+p+'">'+str(i+1)+'</a>' for i,p in enumerate(pages)))
    write(gallery/'coverage.json',dict(rows=len(keys),generated_images=49800,pages=len(pages),new_images_generated=False,
        original_images_referenced=True,all_links_checked_on_server=True,independent_human_ratings=False))
    print(json.dumps(dict(status=report['status'],table=table,practical_numerical_gain_signal=all(signals))),flush=True)


if __name__=='__main__':main()
