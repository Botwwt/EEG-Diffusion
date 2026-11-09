"""Join target metadata after inference and verify all released-feature queries."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import time


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--mode',choices=['join','verify','report'],required=True)
    a=p.parse_args()
    cfg=json.loads(a.registration.read_text())
    out=Path(cfg['output_root'])
    started=time.monotonic()
    targets={x['query_id']:x for x in map(json.loads,Path(cfg['target_mapping']).read_text().splitlines())}
    assert len(targets)==2000
    gallery={x['stimulus_id']:x for x in map(json.loads,Path(cfg['gallery_manifest']).read_text().splitlines())}
    assert len(gallery)==200
    if a.mode=='join':
        import torch
        from PIL import Image
        import numpy as np
        torch.set_num_threads(2)
        source=out/'inference_manifest.jsonl'
        completion=json.loads((out/'completion.json').read_text())
        assert completion['status']=='COMPLETED_ALL2000_PUBLISHED_FEATURE_DECODER_IMAGES'
        assert sha(source)==completion['inference_manifest_sha256']
        rows=[json.loads(x) for x in source.read_text().splitlines()]
        assert len(rows)==len({x['query_id'] for x in rows})==2000
        assert {x['query_id'] for x in rows}==set(targets)
        assert len({x['generated_path'] for x in rows})==2000
        ce={s:torch.load(v['features'],weights_only=True,map_location='cpu') for s,v in cfg['subjects'].items()}
        duplicates=collections.defaultdict(list)
        merged=[]
        for r in rows:
            t=targets[r['query_id']]
            for key in ['subject','stimulus_id','array_row','image_seed','prior_seed']:
                assert r[key]==t[key],(r['query_id'],key)
            x=ce[r['subject']][r['array_row']:r['array_row']+1].float().contiguous()
            digest=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()
            assert digest==r['EEG_native_condition_sha256']
            assert sha(r['generated_path'])==r['generated_sha256']
            assert r['prior_NFE']==100 and r['UNet_NFE']==4
            assert r['projection_calls']==4 and r['VAE_decoder_calls']==1
            assert not r['target_image_or_visual_feature_read_at_inference']
            with Image.open(r['generated_path']) as image:
                image.load()
                assert image.size==(512,512) and image.mode=='RGB'
                pixels=np.asarray(image)
                assert np.isfinite(pixels).all()
            g=gallery[r['stimulus_id']]
            assert g['image_sha256']==t['target_image_sha256'] and sha(g['image_path'])==g['image_sha256']
            duplicates[r['generated_sha256']].append(r['query_id'])
            merged.append(r|t|dict(method='ATM_published_feature_decoder_v1',intervention='correct',
                EEG_source_stimulus_id=r['stimulus_id'],
                repeat_protocol='R80_author_release_convention_not_independent_raw_replay',
                target_mapping_attached_after_inference=True))
        final=out/'manifest.jsonl'
        assert not final.exists()
        final.write_text(''.join(json.dumps(x)+'\n' for x in merged))
        proof=dict(status='PASSED_ALL2000_FEATURE_FILE_ROW_AND_SAVED_IMAGE_TARGET_JOIN',
                   manifest_sha256=sha(final),queries=2000,subjects=10,
                   duplicate_hash_groups={k:v for k,v in duplicates.items() if len(v)>1},
                   source_EEG_encoder_replay_performed=False,upstream_EEG_stimulus_identity_independently_proven=False,
                   mapping_basis='Fixed author released-feature ordering plus audited canonical stimulus metadata and target ZIP hashes',
                   target_images_or_labels_used_as_model_conditions=False,seconds=time.monotonic()-started)
        (out/'identity_before_score.json').write_text(json.dumps(proof,indent=2)+'\n')
    elif a.mode=='verify':
        rows={r['query_id']:r for r in map(json.loads,(out/'manifest.jsonl').read_text().splitlines())}
        scores=list(csv.DictReader((out/'scores/per_query.csv').open()))
        assert len(scores)==len({x['query_id'] for x in scores})==2000
        assert {x['query_id'] for x in scores}==set(rows)==set(targets)
        names=['DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED']
        means=collections.defaultdict(float)
        for s in scores:
            r=rows[s['query_id']]
            for key in ['subject','stimulus_id','concept_id','generated_path','EEG_source_stimulus_id']:
                assert s[key]==r[key],(s['query_id'],key)
            assert s['target_image_path']==s['score_target_path']==r['score_target_path']
            assert sha(s['generated_path'])==r['generated_sha256']
            assert sha(s['score_target_path'])==r['target_image_sha256']
            assert int(s['image_seed'])==r['image_seed']
            for name in names:
                value=float(s[name])
                assert math.isfinite(value)
                means[name]+=value/2000
            assert abs(float(s['SEED'])-(float(s['EffNet'])+float(s['CapSim'])+float(s['ObjectF1']))/3)<1e-12
        summary=json.loads((out/'scores/summary.json').read_text())
        assert summary['queries']==2000 and summary['subjects']==10 and summary['gallery_images']==200
        for name in names:
            assert abs(means[name]-summary['summary'][name]['mean'])<1e-10
        low=json.loads((out/'lowlevel/summary.json').read_text())
        assert low['verified_and_decoded']==2000 and low['missing_count']==low['corrupt_or_mismatched_count']==0
        proof=dict(status='PASSED_ALL2000_SCORED_IMAGE_AND_TARGET_IDENTITY',queries=2000,
                   means=dict(means),score_csv_sha256=sha(out/'scores/per_query.csv'),
                   manifest_sha256=sha(out/'manifest.jsonl'),seconds=time.monotonic()-started,
                   EEG_encoder_reproduced=False,component_gain_proven=False,independent_confirmation=False)
        (out/'identity_after_score.json').write_text(json.dumps(proof,indent=2)+'\n')
    else:
        verified=json.loads((out/'identity_after_score.json').read_text())
        assert verified['status']=='PASSED_ALL2000_SCORED_IMAGE_AND_TARGET_IDENTITY'
        root=a.registration.parent.parent
        path=root/'reports/ATM_PUBLISHED_DECODER_2000_20261020.md'
        assert not path.exists()
        lines=['# ATM 公开特征到图片的解码复现','',
               '已完成全部十被试 2,000 个 query 的实际图片、逐 query 评价和保存文件配对核验。此次使用作者公开 EEG 特征与匹配 prior，尚未复现上游 EEG 编码器。',
               '', '作者原生路径使用 prior 50 步 CFG5、开启 x0 裁剪；SDXL Turbo512 四步 CFG0、IP1。采样 seed 规则固定为1729，每 query 一张 PNG，不筛选图片或种子。作者预训练 seed 未知，本轮未重训。',
               '', '| 指标 | 全量均值 |','| --- | --- |']
        lines += [f'| {k} | {v:.8f} |' for k,v in verified['means'].items()]
        lines += ['', '这里的 DINO 来自固定 DINOv3 ViT S16 评价器；SEED 使用现有锁定实现。未声称逐项复现论文所有评价器或表格数值。',
                  '', '评价数据是已暴露官方 benchmark，作者特征按 R80 源码约定解释；上游原始 EEG 与特征行序未独立重算。结果与当前 R4、16,600 query、不同训练范围的机制实验分轨报告，不能直接比较组件增益。',
                  '', '推理进程仅打开 EEG 特征、固定权重与不含目标类别/图片路径的 query 清单。目标元数据在推理后由 CPU 独立合并。全部目标字节、生成文件、评分目标和实际条件 tensor 行已核对；这不替代上游 EEG 编码器复现。',
                  '', '未训练新组件，未进行人工评分，未证明画质非劣、独立确认或 SOTA。',
                  '', f'完整产物：`{out}`。逐 query 指标：`{out}/scores/per_query.csv`。', '']
        path.write_text('\n'.join(lines))
        proof=dict(status='COMPLETED_PUBLIC_FEATURE_DECODER_IMAGES_AND_UNIFORM_SCORES_ONLY',
                   report=str(path),report_sha256=sha(path),queries=2000,full_EEG_reproduction=False,
                   independent_component_gain=False,seconds=time.monotonic()-started)
        (out/'report_completion.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    main()
