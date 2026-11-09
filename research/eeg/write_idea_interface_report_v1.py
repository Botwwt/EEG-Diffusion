"""Full-population same-encoder report; required missing arms stay visible."""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import time
from prepare_idea_interface_arm_v1 import load,sha

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
PY='/root/miniconda3/bin/python'
ARMS=['Frozen','RealPaired','Conditional','Moment','Iso','Clean']
METRICS=['DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=ARMS);args=parser.parse_args()
    runtime_path=ROOT/'configs/idea_interface_runtime_v1.json';runtime=load(runtime_path)
    results={};rows=[]
    for arm in ARMS:
        root=DATA/'evaluation/V'/f'IDEA1_{arm}_correct_R4';summary=root/'metrics/summary.json'
        row=dict(arm=arm,status='NOT_RUN_OR_INCOMPLETE',expected_queries=16600,actual_images=0,
            full_training_config_sha256=runtime['full_training_config_sha256'],runtime_sha256=sha(runtime_path),
            DINO2AFC='',SEED='',ObjectF1='',PixCorr_RGB='',SSIM_Y='',source='',source_sha256='')
        images=DATA/'images/IDEA1_interface_v1/V'/f'IDEA1_{arm}_correct_R4'
        if (images/'progress.json').exists():row['actual_images']=load(images/'progress.json').get('completed',0)
        if summary.exists() and (root/'identity_after/summary.json').exists() and (root/'low_level/summary.json').exists():
            report=load(summary);complete=load(images/'completion.json');identity=load(root/'identity_after/summary.json')
            assert report['status']==complete['status']=='COMPLETED' and report['queries']==complete['queries']==16600
            assert identity['verified_queries']==identity['score_rows_checked']==16600 and identity['failed_queries']==0
            assert complete['full_training_config_sha256']==runtime['full_training_config_sha256']
            assert complete['image_interface_config_sha256']==sha(runtime_path)
            assert sha(images/'manifest.jsonl')==complete['image_manifest_sha256']
            row.update(status='COMPLETE_FULL_IMAGE_METRICS_AND_IDENTITY',actual_images=16600,
                source=str(summary),source_sha256=sha(summary),**{m:report['summary'][m]['mean'] for m in METRICS})
            low=load(root/'low_level/summary.json');row.update(low['summary_over_valid_images'])
            results[arm]=dict(summary=report,root=root,images=images)
        else:
            if summary.exists():row['status']='FULL_METRICS_PENDING_COMPLETE_POSTSCORING_AUDITS'
            decisions=sorted((ROOT/'audit').glob(f'idea_interface_{arm}_budget_*.json'))
            if decisions and not load(decisions[-1])['arm_lifecycle_funded']:
                row.update(status='NOT_RUN_COMPLETE_LIFECYCLE_BUDGET_NOT_FUNDED',source=str(decisions[-1]),source_sha256=sha(decisions[-1]))
        rows.append(row)
    paired=[]
    for candidate in ['RealPaired','Conditional','Moment','Iso','Clean']:
        references=['Frozen']+(['RealPaired','Moment','Iso','Clean'] if candidate=='Conditional' else [])
        for reference in references:
            if candidate not in results or reference not in results:continue
            a=results[candidate];b=results[reference]
            am=[json.loads(x) for x in (a['images']/'manifest.jsonl').read_text().splitlines()]
            bm={x['query_id']:x for x in map(json.loads,(b['images']/'manifest.jsonl').read_text().splitlines())}
            assert len(am)==len(bm)==16600
            for x in am:
                y=bm[x['query_id']]
                for key in ['encoder_sha256','full_training_config_sha256','initial_image_noise_sha256','image_seed','image_interface_config_sha256','target_image_sha256']:
                    assert x[key]==y[key],('Mismatched comparison contract',candidate,reference,key)
            out=DATA/'evaluation/paired'/f'IDEA1_{candidate}_minus_{reference}_V_R4'
            if not (out/'summary.json').exists():
                assert not out.exists(),'Preserve incomplete paired result and inspect before rerunning'
                command=[PY,str(ROOT/'paired_analysis.py'),'--reference-manifest',str(b['images']/'manifest.jsonl'),
                    '--candidate-manifest',str(a['images']/'manifest.jsonl'),'--reference-metrics',str(b['root']/'metrics/per_query.csv'),
                    '--candidate-metrics',str(a['root']/'metrics/per_query.csv'),'--output',str(out)]
                result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=180)
                assert result.returncode==0,result.stderr[-2000:]
            pair=load(out/'summary.json');assert pair['status']=='COMPLETED_PAIRED_FULL_PARTITION' and pair['queries']==16600
            paired.append(dict(candidate=candidate,reference=reference,path=str(out/'summary.json'),sha256=sha(out/'summary.json'),metrics=pair['metrics']))
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+str(time.time_ns());folder=ROOT/'reports/idea_interface_snapshots'/stamp;folder.mkdir(parents=True)
    fields=['arm','status','expected_queries','actual_images',*METRICS,'PixCorr_RGB','SSIM_Y','full_training_config_sha256','runtime_sha256','source','source_sha256']
    with (folder/'comparison.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    text=['# EEG 误差适配：同编码器完整图片比较','',
        '仅比较新 patience50 编码器和相同 SDXL512/20步/CFG5/IP1 下的结果。旧 patience20 图片另表保存，不混入此处差值。',
        '一个训练 seed1729；每方法全部十被试、16,600 个 V query、每 query 一张。V 是开发数据，bootstrap 区间不覆盖训练 seed 变异。','',
        '| 方法 | 图片/预期 | 状态 | DINO2AFC | SEED | ObjectF1 |','|---|---:|---|---:|---:|---:|']
    for row in rows:
        num=lambda x:f'{x:.6f}' if isinstance(x,(float,int)) else '—'
        text.append(f'| {row["arm"]} | {row["actual_images"]}/16600 | {row["status"]} | {num(row["DINO2AFC"])} | {num(row["SEED"])} | {num(row["ObjectF1"])} |')
    text+=['','## 完整配对差值','']
    for pair in paired:
        value=pair['metrics']['DINO2AFC'];ci=value['CI95_subject_concept']
        text.append(f'- {pair["candidate"]} − {pair["reference"]}：DINO2AFC {value["paired_delta"]:+.6f}，95% 被试—概念区间 [{ci[0]:+.6f}, {ci[1]:+.6f}]。')
    if not paired:text.append('尚无同编码器完整配对图片比较。')
    missing=[a for a in ARMS if a not in results]
    text+=['','## 解释边界','',
        '- 必需缺项：'+('、'.join(missing) if missing else '此表六个图像身份均已有完整结果；仍需审查 EEG 依赖、质量及独立确认。'),
        '- 五组只训练已有原生投影器8,400,896参数；这不是完整349,188,096参数IP分支训练或完整ATM公开复现。',
        '- 固定配方为一个完整132,300训练query pass，真实覆盖每个被试—刺激一次；完成该预算不自动证明收敛或容量足够。',
        '- Conditional 的原生/投影相对能量判据通过，仍不证明模拟分布等于真实分布；强分类器仍能区分两者。',
        '- DINO2AFC不是对象准确率。ObjectF1使用COCO+gender82类及原空检测约定，需要同时看检测覆盖和真实图片。',
        '- 尚缺普通噪声或RealPaired控制时，不能把增益归因为独立EEG误差机制；缺失/未收敛/失败不能填零或写成已完成负结果。',
        '- 正确、置换、缺失EEG控制及完整盲评材料仍是任务要求；人工评分未实施不能声称画质非劣。',
        '- 官方200刺激曾暴露；本表不宣布SOTA、未暴露最终确认或训练seed稳定性。','']
    body='\n'.join(text);(folder/'REPORT.md').write_text(body)
    receipt=dict(status='EVIDENCE_ONLY_SAME_ENCODER_IMAGE_REPORT',snapshot=str(folder),source_rows=rows,paired_comparisons=paired,
        completed_arms=list(results),missing_required_arms=missing,scientific_goal_complete=False,
        independent_mechanism_confirmation=False,SOTA_established=False,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'reports/IDEA_INTERFACE_COMPARISON.md').write_text(body)
    evaluation=DATA/'evaluation/V'/f'IDEA1_{args.arm}_correct_R4';assert args.arm in results
    (evaluation/'REPORT.md').write_text(body)
    (evaluation/'REPORT_receipt.json').write_text(json.dumps(dict(arm=args.arm,queries=16600,snapshot=str(folder),
        source_metric_sha256=sha(evaluation/'metrics/summary.json'),manifest_sha256=sha(results[args.arm]['images']/'manifest.jsonl'),
        full_training_config_sha256=runtime['full_training_config_sha256'],scientific_goal_complete=False),indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_rows','paired_comparisons']}),flush=True)


if __name__=='__main__':main()
