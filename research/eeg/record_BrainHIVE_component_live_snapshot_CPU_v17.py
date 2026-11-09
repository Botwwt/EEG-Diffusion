"""Evidence-driven small documentation snapshot; never edits experiment sources."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    now=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    closure=read(ROOT/'audit/BrainHIVE_native_R80_full_closure_20261020_v10.json')
    recipe=read(ROOT/'configs/BrainHIVE_bounded_main_recipe_v1.json')
    queue=read(WORK/'audit/bounded_full_lifecycle_queue_v1.json')
    job=read(ROOT/'audit/BrainHIVE_bounded_full_T_risk_v1_job.json')
    progress=read(WORK/'components/bounded_v1/risk/progress.json')
    assert progress['updates']>=64 and progress['status']=='RUNNING' and job['status']=='RUNNING'
    replay=read(WORK/'audit/component_native_pixel_replay_v1.json')
    assert replay['max_absolute_channel_difference']==0
    latent=read(WORK/'audit/component_T_latent_bank_v2.json')
    assert latent['status']=='COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS'
    before=ROOT/'audit/before_BrainHIVE_component_live_v17'
    assert not before.exists();before.mkdir()
    documents=['STATUS.md','FINAL_REPORT.md','budget.yaml','campaign_state.json']
    for name in documents:shutil.copy2(ROOT/name,before/name)
    write(before/'index.json',dict(UTC=now,files={name:sha(before/name) for name in documents}))
    report='reports/BRAINHIVE_NATIVE_CLOSURE_AND_COMPONENT_EXECUTION_20261021.md'
    status=f'''# 当前实际状态

更新：{now}。主线为 Brain-HIVE（ICLR 2026）H14+B32+VAE，已完成单 seed、单图版本的原生 R80 复现与评价；现在执行这一个父架构上的组件实验。

正确、固定打乱、缺失 EEG 各 2000 张真实图片，共 6000 张；配对、噪声和实际条件注入检查通过，统一指标与作者八项原生指标已全部闭合。正确 EEG 的 DINO2AFC/SEED 为 0.872769/0.182698，打乱为 0.488869/0.068504，缺失为 0.498915/0.061316。正确相对打乱的 DINO 配对差 +0.383899，95% 被试—概念区间 [0.348507,0.417132]。这证明 EEG 配对相关内容，不表示 87% 物体正确。固定 20 样本仍有明显物体错误；没有盲评质量非劣或独立 SOTA 结论。R80/200 图库与旧 R4/1660 图库分轨。

单组件是有界条件方向修正与预测风险门控；普通适配器为匹配对照。原生父模型、A/P/IP、Turbo 全部冻结。组件工程反向、真实 token/UNet 响应、输入支持集、真实 OOF 零分支和新入口原生图片逐像素重放检查通过。全部 13230 张 T 图的分块原生 VAE 缓存已完成。

正式风险组件训练已实际开始：本快照 {progress['updates']}/24807 更新、{progress['seen_queries']}/396900 次训练暴露。训练 seed1729，十被试全部 132300 条真实 OOF query 固定三遍，不根据 V/测试选 checkpoint。每组完整训练、16600 张 V/R4 图片及逐 query 评价后继续下一组，顺序风险组件→普通适配器→原生父模型。原生512/4步/CFG0/IP1、每query一图，三组使用相同输入权限及噪声。完整主阶段预计30–36GPU小时；KAN、阶段路由及第二架构延后。**组件图片增益尚未确定。**

三组共49800张原图的主生命周期存储预测已登记，运行时持续检查；固定平均门控、打乱风险及R4 EEG机制控制尚未完整预留，主三组阳性也不能直接宣布机制或通用性成立。总GPU时长上限已由用户取消；仅使用现有单卡，不购买、续费或扩容；每项作业有有限超时和检查点。候选12个/结构修订2次限制未重置。

原始EEG、学习权重、历史原图保留。可重算缓存迁入私有RAM并保留原路径，RAM不是持久备份，重启后需要重建；详细SHA/来源日志留在服务器。旧AVDE检查点连同优化器无损归档，使用旧路径前需运行相应恢复入口。C/D不保存大文件，不使用共享公共盘。

完整证据、成本、失败记录及恢复入口见 [{report}]({report})。最新队列为服务器 `/root/brainhive_reproduction_20261020_v1/audit/bounded_full_lifecycle_queue_v1.json`；每10分钟的只读监视为 `bounded_10min_monitor_latest_v1.json`，不改变模型、不反复查看V/测试得分。正式训练日志与检查点在 `/root/brainhive_reproduction_20261020_v1/components/bounded_v1/risk/`。本文件是时间点快照，实时数字以服务器进度为准。

历史原始报告保留在 FINAL_REPORT.md；本次更新前四份文档精确副本在 `audit/before_BrainHIVE_component_live_v17/`。目标尚未达成。
'''
    (ROOT/'STATUS.md').write_text(status)
    old=(ROOT/'FINAL_REPORT.md').read_text()
    prefix=f'''# Brain-HIVE复现已完成，单组件全量实验运行中

更新：{now}。Brain-HIVE正确/打乱/缺失EEG各2000张原生R80图片及统一评价、作者八项指标已全部完成。DINO2AFC为0.872769/0.488869/0.498915，SEED为0.182698/0.068504/0.061316；正确EEG有明确配对相关内容，但固定样本的对象错误仍明显，不能把2AFC当对象正确率。

有界方向修正+风险门控组件已通过真实输入、梯度、完整原生UNet及精确像素重放检查，正式风险组正在训练（本快照{progress['updates']}/24807更新）。固定最后checkpoint后自动完成十被试16600张V/R4实际图片与逐query评价，再继续普通适配器及原生父模型；总计49800张主实验图片。配置、数据权限、逐query噪声和评价定义已锁定，不依据中途V/测试分数扩展候选。

详细结果和执行依据见 [{report}]({report})，实际当前状态见 STATUS.md。用户取消总时长上限，现有单卡主阶段预计30–36小时；没有购买扩容。主三组生命周期存储预测已登记，RAM派生缓存非持久，运行时保留存储缓冲；原始EEG、学习权重、历史原图保留。组件增益、风险机制、质量非劣、跨架构性与SOTA均未建立。

---

以下全文是本次更新前的历史研究报告，保留原文供追溯。其中“尚未完成原生评价/组件未实施/空间尚未预留”等描述属于当时快照；当前状态由上文、STATUS.md和服务器最新回执确定。

'''
    (ROOT/'FINAL_REPORT.md').write_text(prefix+old)
    state=read(ROOT/'campaign_state.json')
    state.update(stage='BRAINHIVE_NATIVE_CLOSED_BOUNDED_FULL_IMAGE_TRAINING_RUNNING',last_update_utc=now,
        component_implemented=True,component_training_started=True,component_training_updates_at_snapshot=progress['updates'],
        component_unique_real_OOF_training_queries=132300,component_training_expected_updates=24807,
        BrainHIVE_original_metrics_GPU_complete=True,BrainHIVE_original_metric_queue='ALL_THREE_R80_ARMS_CLOSED',
        BrainHIVE_native_EEG_pairing_image_signal_established=True,new_EEG_image_gain_established=False,
        component_image_gain_established=False,main_three_lifecycle_storage_reserved=True,
        all_contingent_mechanism_controls_storage_reserved=False,
        active_controller_pid=read(ROOT/'audit/BrainHIVE_bounded_full_lifecycle_launch_v17.json')['pid'],
        active_GPU_pid=job['pid'],active_state_path=str(WORK/'audit/bounded_full_lifecycle_queue_v1.json'),
        ten_minute_monitor=str(WORK/'audit/bounded_10min_monitor_latest_v1.json'),
        current_component_evidence_report=report,scientific_goal_complete=False,
        resource_budget='No user total time cap; existing single GPU, finite jobs, no purchase or expansion',
        storage_status='MAIN_THREE_ARMS_FORECAST_RESERVED_WITH_RUNTIME_GUARD_DERIVED_RAM_CACHES_VOLATILE')
    (state.setdefault('verified_wait',[])).append(dict(UTC=now,queue_pid=state['active_controller_pid'],GPU_pid=job['pid'],
        job=job['name'],training_updates=progress['updates'],finite_exact_recipe=True))
    write(ROOT/'campaign_state.json',state)
    budget=(ROOT/'budget.yaml').read_text()
    changes={
        'next_gate: Finish existing native R80 common and original eight image metrics then one BrainHIVE component with matched ordinary adapter':
            'next_gate: Complete fixed full-T risk component, all16600 V images and query evaluation, then ordinary and native references',
        'actual_EEG_image_generation: all_correct_permuted_missing_R80_6000_complete_evaluation_running':
            'actual_EEG_image_generation: all6000_native_R80_images_and_common_and_original_eight_metrics_complete',
        'component_implemented_at_this_snapshot: false':'component_implemented_at_this_snapshot: true',
        'full_component_lifecycle_storage_funded: false':'full_component_lifecycle_storage_funded: true',
        'original_eight_metric_GPU_evaluation_complete: false':'original_eight_metric_GPU_evaluation_complete: true',
        'component_training_started: false':'component_training_started: true'}
    for a,b in changes.items():assert a in budget,a;budget=budget.replace(a,b)
    budget+='\ncurrent_bounded_component_main_phase:\n  recipe: configs/BrainHIVE_bounded_main_recipe_v1.json\n  fixed_arm_order: [risk, ordinary, parent]\n  unique_real_OOF_T_queries: 132300\n  passes_per_trainable_arm: 3\n  updates_per_trainable_arm: 24807\n  planned_actual_V_images_per_arm: 16600\n  total_main_actual_images: 49800\n  measured_batch16_steady_seconds_per_update: 1.11\n  estimated_training_GPU_hours_per_arm: 7.7\n  estimated_main_GPU_hours_range: [30, 36]\n  persistent_main_forecast_bytes: 18457741824\n  storage_reservation_receipt: audit/BrainHIVE_bounded_three_main_lifecycle_reservation_v1.json\n  runtime_remaining_lifecycle_guard: true\n  all_contingent_mechanism_controls_storage_reserved: false\n  risk_fixed_mean_shuffled_and_R4_interventions_complete: false\n  derived_RAM_caches_are_persistent_backup: false\n  original_EEG_learned_checkpoints_historical_original_images_deleted: false\n  scientific_goal_complete: false\n'
    (ROOT/'budget.yaml').write_text(budget)
    event=dict(event_id='BrainHIVE_native6000_closed_bounded_full_training_live_v17',UTC=now,
        type='native_query_result_closure_and_actual_full_real_OOF_image_training',
        native_closure_sha256=sha(ROOT/'audit/BrainHIVE_native_R80_full_closure_20261020_v10.json'),
        recipe_sha256=sha(ROOT/'configs/BrainHIVE_bounded_main_recipe_v1.json'),
        source_and_zero_native_pixel_replay_sha256=sha(WORK/'audit/component_native_pixel_replay_v1.json'),
        main_candidate_configurations=1,cap_not_reset=True,training_progress=progress,
        formal_V_images_generated=0,component_gain_established=False,scientific_goal_complete=False)
    ledger=ROOT/'experiment_ledger.jsonl'
    assert event['event_id'] not in ledger.read_text()
    with ledger.open('a') as f:f.write(json.dumps(event)+'\n')
    receipt=dict(UTC=now,status='ACTUAL_FORMAL_RISK_TRAINING_PROGRESS_AND_NATIVE_CLOSURE_DOCUMENTED',
        queue=queue,job=job,progress=progress,
        actual_native_replay=read(WORK/'audit/component_native_pixel_replay_v1.json'),
        T_latent_receipt_sha256=sha(WORK/'audit/component_T_latent_bank_v2.json'),
        main_lifecycle_reserved=True,all_contingent_controls_reserved=False,
        updated_documents={n:sha(ROOT/n) for n in documents},goal_complete=False)
    out=ROOT/'audit/BrainHIVE_component_live_snapshot_v17.json';assert not out.exists();write(out,receipt)
    print(json.dumps(dict(status=receipt['status'],updates=progress['updates'],job_pid=job['pid'])))


if __name__=='__main__':main()
