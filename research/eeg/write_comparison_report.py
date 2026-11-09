"""Evidence-only comparison snapshots; never score a partial image partition.

Legacy V64 and current full V/benchmark protocols remain separate panels.
CSV exports preserve source paths and hashes. This does not unlock or select.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
OLD=ROOT.parent/'eeg_diffusion_next_20261018_v1'
METRICS=['DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED']


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def load(path):return json.loads(Path(path).read_text())


def number(value):return f'{value:.6f}' if isinstance(value,(int,float)) else '—'


def csv_export(path,rows,fields):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def completed_result(path):
    report=load(path);arm_root=path.parent.parent;feature_config=load(arm_root/'features/config.json')
    manifest=Path(feature_config['manifest']);generation=load(manifest.parent/'completion.json')
    n=report['queries'];assert report['status']=='COMPLETED' and n in [16600,2000] and report['subjects']==10
    assert generation['status']=='COMPLETED' and generation['queries']==n
    assert sha(manifest)==feature_config['manifest_sha256']==generation['image_manifest_sha256']
    feature=load(arm_root/'features/completion.json');assert feature['status']=='COMPLETED' and feature['queries']==n
    assert feature['manifest_sha256']==feature_config['manifest_sha256']
    records=[json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(records)==n and len({r['query_id'] for r in records})==n
    with (path.parent/'per_query.csv').open() as file:queries=list(csv.DictReader(file))
    assert len(queries)==n and {q['query_id'] for q in queries}=={r['query_id'] for r in records}
    counts=collections.Counter(q['subject'] for q in queries)
    assert set(counts)==set(report['per_subject']) and len(counts)==10 and set(counts.values())=={n//10}
    first=records[0]
    for field in ['partition','repeat_protocol','method','intervention','generator_revision','steps','CFG','UNet_NFE','image_interface_version']:
        assert len({r[field] for r in records})==1,(path,field)
    track=f'{first["partition"]}/{first["repeat_protocol"]}/gallery{report["gallery_images"]}/steps{first["steps"]}/CFG{first["CFG"]}'
    row=dict(scope='CURRENT_FULL_PARTITION',track=track,arm=arm_root.name,status='COMPLETE_IMAGE_METRICS',subjects=10,
        expected_queries=n,actual_images=n,gallery_images=report['gallery_images'],
        method=first['method'],intervention=first['intervention'],repeat_protocol=first['repeat_protocol'],
        generator_revision=first['generator_revision'],image_steps=first['steps'],image_CFG=first['CFG'],UNet_NFE=first['UNet_NFE'],
        semantic_prior_forward_budget=100 if first['prior_checkpoint'] else 0,
        generation_seconds_per_query_including_load_IO=generation['seconds']/n,
        evaluator_seconds_per_query_including_load_IO=feature['seconds']/n,
        blank_proxy_count=report['blank_proxy_count'],GT_detector_empty_count=report['GT_object_detector_empty_count'],
        generated_detector_empty_count=report['generated_object_detector_empty_count'],
        source=str(path),source_sha256=sha(path),manifest=str(manifest),manifest_sha256=sha(manifest),
        source_per_query_csv=str(path.parent/'per_query.csv'),PixCorr_RGB='',SSIM_Y='',human_ratings_performed=False)
    for metric in METRICS:row[metric]=report['summary'][metric]['mean']
    pauses=[]
    if first['method']=='REF_REG' and first['intervention']=='correct' and first['partition']=='V':
        job=load(ROOT/'audit/S1_V_generate_REF_REG_attempt2_job.json')
        for receipt in sorted((ROOT/'audit').glob('*slot.json')):
            item=load(receipt)
            if item.get('REG_pid')==job['pid']:
                assert item['status'].startswith('REG_RESUMED_AFTER_')
                pauses.append(dict(path=str(receipt),sha256=sha(receipt),seconds=item['pause_seconds']))
    seconds=sum(p['seconds'] for p in pauses);assert 0<=seconds<generation['seconds']
    row.update(accounted_nested_diagnostic_pause_seconds=seconds,
        generation_seconds_per_query_excluding_only_nested_pauses=(generation['seconds']-seconds)/n,
        nested_pause_receipts=json.dumps(pauses))
    low=arm_root/'low_level/summary.json'
    if low.exists():
        quality=load(low);row.update(saved_files_verified=quality['verified_and_decoded'],missing_images=quality['missing_count'],
            corrupt_images=quality['corrupt_or_mismatched_count'],duplicate_hash_images_beyond_first=quality['duplicate_images_beyond_first'])
        if quality['status']=='COMPLETED' and quality['verified_and_decoded']==n:
            row.update({metric:quality['summary_over_valid_images'][metric] for metric in ['PixCorr_RGB','SSIM_Y']})
    subjects=[dict(scope=row['scope'],track=track,arm=row['arm'],subject=s,queries=counts[s],
        repeat_protocol=row['repeat_protocol'],gallery_images=row['gallery_images'],**scores,source=str(path),source_sha256=row['source_sha256'])
        for s,scores in report['per_subject'].items()]
    return row,subjects,report


def main():
    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    rows=[];subjects=[];raw={}
    for path in sorted((DATA/'evaluation').glob('*/*/metrics/summary.json')):
        row,part,report=completed_result(path);rows.append(row);subjects+=part;raw[row['arm']]=report
    interface=load(ROOT/'configs/image_evaluation_resolved_batchsafe_v3.json')
    expected_arms={f'{method}_correct_R4' for method in interface['formal_methods']}
    valid_image_root=DATA/'images/S1_full_native_batchsafe_v2/V'
    expected_arms|={p.name for p in valid_image_root.glob('*') if p.is_dir()}
    done={(row['arm'],row['track'].split('/')[0]) for row in rows}
    for arm in sorted(expected_arms):
        if (arm,'V') in done:continue
        progress=valid_image_root/arm/'progress.json';complete=valid_image_root/arm/'completion.json'
        saved=load(complete)['queries'] if complete.exists() else (load(progress)['completed'] if progress.exists() else 0)
        rows.append(dict(scope='CURRENT_FULL_PARTITION',track='V/R4/gallery1660/steps20/CFG5.0',arm=arm,
            status='FULL_IMAGE_METRICS_PENDING',subjects=10,expected_queries=16600,actual_images=saved,gallery_images=1660,
            source=str(progress) if progress.exists() else str(ROOT/'configs/image_evaluation_resolved_batchsafe_v3.json')))
    hypothesis_path=ROOT/'research/hypothesis_status.json'
    mechanism_hypothesis=next(h for h in load(hypothesis_path)['hypotheses'] if h['id']=='H_ERROR_ADAPTED_IMAGE_INTERFACE')
    implemented_methods={r.get('method') for r in rows if r['status']=='COMPLETE_IMAGE_METRICS'}
    implemented_methods|={p.name[:-len('_correct_R4')] for p in valid_image_root.glob('*_correct_R4') if p.is_dir()}
    mechanism_rows=[dict(scope='REQUIRED_MECHANISM_ARMS_NOT_RUN',track='planned_full_V/R4/16600_queries',arm=arm,
        status=mechanism_hypothesis['status'],subjects=10,expected_queries=16600,actual_images=0,
        gallery_images=1660,source=str(hypothesis_path),source_sha256=sha(hypothesis_path),
        reason_at_that_time=mechanism_hypothesis['scope'])
        for arm in ['CLEAN_BRIDGE','ISO','MOMENT','CONDITIONAL','REAL_PAIRED'] if arm not in implemented_methods]
    legacy_path=OLD/'reports/final_summary.json';legacy=load(legacy_path)
    legacy_rows=[]
    for arm in legacy['arms']:
        row=dict(scope='LEGACY_V64_ONLY',track='legacy/sub-01/R4/gallery64/Turbo4stepsCFG0',arm=arm['name'],
            status=arm['status'],subjects=1,expected_queries=64,actual_images=arm.get('V_images',''),gallery_images=64,
            DINO2AFC=arm.get('DINO2AFC',''),DINO_cosine=arm.get('DINO_cosine',''),DINO_Top1=arm.get('DINO_Top1',''),
            PixCorr_RGB='',SSIM_Y='',legacy_PixCorr=arm.get('PixCorr',''),legacy_SSIM=arm.get('SSIM',''),
            reason_at_that_time=arm.get('reason',''),source=str(legacy_path),source_sha256=sha(legacy_path))
        legacy_rows.append(row)
    out=ROOT/'reports';out.mkdir(exist_ok=True);snapshots=out/'snapshots';snapshots.mkdir(exist_ok=True)
    fields=['scope','track','arm','status','subjects','expected_queries','actual_images','gallery_images',*METRICS,
        'PixCorr_RGB','SSIM_Y','legacy_PixCorr','legacy_SSIM','blank_proxy_count','saved_files_verified','missing_images','corrupt_images',
        'duplicate_hash_images_beyond_first','GT_detector_empty_count','generated_detector_empty_count','UNet_NFE','semantic_prior_forward_budget',
        'generation_seconds_per_query_including_load_IO','accounted_nested_diagnostic_pause_seconds',
        'generation_seconds_per_query_excluding_only_nested_pauses','nested_pause_receipts',
        'evaluator_seconds_per_query_including_load_IO','source','source_sha256','manifest',
        'manifest_sha256','source_per_query_csv','reason_at_that_time']
    csv_export(out/'unified_comparison.csv',rows+mechanism_rows+legacy_rows,fields)
    csv_export(out/'per_subject.csv',subjects,['scope','track','arm','subject','queries','repeat_protocol','gallery_images',*METRICS,'source','source_sha256'])
    text=[f'更新时间：{utc}。这是有来源的阶段比较；缺少完整结果的行保持空值，不根据部分图片或训练 loss 填成绩。',
        '', '本轮完整分区', '',
        '| 协议 | 方法/干预 | 已保存/预期图片 | 状态 | DINO2AFC | SEED | ObjectF1 | 空白代理数 |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f'| {r["track"]} | {r["arm"]} | {r["actual_images"]}/{r["expected_queries"]} | {r["status"]} | {number(r.get("DINO2AFC"))} | {number(r.get("SEED"))} | {number(r.get("ObjectF1"))} | {r.get("blank_proxy_count","—")} |')
    text+=['','完整数值与来源哈希见 [统一 CSV](unified_comparison.csv)；已完成方法的各被试绝对分数见 [per_subject.csv](per_subject.csv)。完整逐 query 文件与图片均在服务器，CSV 中保留实际路径。']
    if mechanism_rows:
        text+=['','必需接口对照的未完成项','',
            '以下是任务要求的实验身份；尚未产生训练检查点或正式图片结果，不表示已有完整运行接口。',
            '', '| 对照 | 已保存/预期图片 | 状态 |', '|---|---:|---|']
        for r in mechanism_rows:text.append(f'| {r["arm"]} | 0/16600 | {r["status"]} |')
        text+=['','先完成真实概念折外条件及独立 Dcheck 校验，再用实测吞吐核算全部匹配训练、图片、控制与最终确认成本。未运行的指标为空，不能当作负结果。']
    text+=[
        '', '历史 V64：单被试、17 电极、旧 BrainMLP/Turbo', '',
        '| 历史方法 | 当时状态 | DINO2AFC | 图数 |', '|---|---|---:|---:|']
    for r in legacy_rows:text.append(f'| {r["arm"]} | {r["status"]} | {number(r.get("DINO2AFC"))} | {r["actual_images"] or "未运行"} |')
    text+=['','历史 REJECTED 只表示该固定配置没有通过当时的筛选，不是所有 A/C 机制的全数据否定。历史 gallery64 与本轮 gallery1660/200、通道、被试和生成器不同，不能跨表按原分数排名；旧 PixCorr/SSIM 与新补充定义也不直接混合。',
        'B1/B2 的历史缺 SDXL 原因保留原文；当前标准 SDXL 已到位，当前仍需验证密度比/额外 EEG 证据等前置条件。',
        '', '配对差值', '']
    comparisons=[]
    for path in sorted((DATA/'evaluation/paired').glob('*/summary.json')):
        report=load(path)
        if report.get('status')!='COMPLETED_PAIRED_FULL_PARTITION':continue
        comparisons.append(dict(path=str(path),sha256=sha(path),queries=report['queries']))
        value=report['metrics']['DINO2AFC'];ci=value['CI95_subject_concept']
        text.append(f'- {path.parent.name}：candidate − reference 的 DINO2AFC 差值 {value["paired_delta"]:.6f}，被试/概念区间 [{ci[0]:.6f}, {ci[1]:.6f}]；{report["queries"]} query。EEG 干预比较需注意此处方向为“干预减正确”。')
    if not comparisons:text.append('尚无本轮完整配对图片比较。')
    progress=load(ROOT/'audit/current_progress.json')
    references_complete=all((f'{method}_correct_R4','V') in done for method in interface['formal_methods'])
    reference_uncertainty=('两套参照的完整数值已在上表记录；图片增益需结合配对差值、独立语义和质量检查解释。' if references_complete else '两套新参照的完整图片比较尚未完成。')
    text+=['',f'资源：上限40 GPU小时、训练seed1729。作业墙钟加保守历史/S0额度累计 {progress["charged_job_hours_plus_prior_conservative_allowances"]:.3f} 小时；更保守剩余授权 {progress["remaining_authorized_hours"]:.3f} 小时。供应商账单未独立核验，失败作业计入。',
        '', '解释与未完成项', '',
        '- DINO2AFC 是固定 gallery 相对偏好，不是对象正确率；独立语义同时报告 ObjectF1/SEED及检测覆盖。',
        '- 一次训练seed；区间重采样被试/概念，不覆盖训练随机性。开发区间属于描述性；官方200曾暴露，不是独立H。',
        '- 原始事件级 stimulus 标签不在发布 EEG 字典；已核验发布顺序、显式元数据、ZIP成员和图片SHA，不能宣称独立采集事件重放。',
        '- 本轮encoder patience20来自CLI，官方benchmark为50；完整T覆盖不等于收敛或完整官方训练配置复现。',
        '- 原生投影CPU诊断与检索不等于图片成功。当前没有受控扩容证据。',
        '- 人工评分未开展；完整盲评材料不等于已有评分或画质非劣证明。正式每作业峰值显存尚未逐作业测量，T吞吐profile不替代该项。',
        '- OOF、误差模型、完整matched接口训练及最终确认的状态以实际回执为准，不能把未运行视为假设被否定。',
        '- 生成墙钟包含装载/I/O及嵌套诊断暂停；CSV另列诊断暂停和仅扣除该暂停的耗时。总资源账本始终计入这些诊断，不能把扣除后的吞吐当成总花费。',
        '', '已否定/未确定/独立确认', '',
        '- 已有不利证据：历史A1/A2/C1/C2固定配置未通过筛选；新epsilon prior在修正批处理后未通过T图就绪检查，未作全V全局否定。',
        '- '+reference_uncertainty+' 容量/停止规则的因果影响及新的误差适配机制仍需各自的实际验证。',
        '- 本自动报告不宣布SOTA或独立机制确认；这还需要完整图片、独立语义、质量、EEG依赖、matched参照与未暴露确认数据的逐项审查。',
        '', f'完整假设状态见 [hypothesis_status.json](../research/hypothesis_status.json)。最终测试是否解锁：{progress["final_test_unlocked"]}。']
    body='\n'.join(text)+'\n';(out/'current_comparison.md').write_text(body)
    # Every snapshot has its own directory so its linked CSVs are immutable too.
    archive=snapshots/stamp;archive.mkdir(exist_ok=False)
    archived_body=body.replace('../research/hypothesis_status.json','../../../research/hypothesis_status.json')
    (archive/'current_comparison.md').write_text(archived_body)
    for filename in ['unified_comparison.csv','per_subject.csv']:(archive/filename).write_bytes((out/filename).read_bytes())
    receipt=dict(status='EVIDENCE_ONLY_PROGRESS_REPORT_WRITTEN',UTC=utc,current_complete_image_methods=sum(r['status']=='COMPLETE_IMAGE_METRICS' for r in rows),
        partial_image_partitions_scored=False,current_rows=len(rows),required_mechanism_arms_not_run=len(mechanism_rows),historical_rows=len(legacy_rows),
        per_subject_rows=len(subjects),paired_comparisons=comparisons,report=str(out/'current_comparison.md'),snapshot=str(archive),
        report_sha256=sha(out/'current_comparison.md'),source_sha256=sha(__file__),
        scientific_goal_complete=False,SOTA_established=False,independent_confirmation=False)
    (out/'report_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
