"""Preserve old records, then record the observed queue without inventing runs."""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import time
from brainhive_two_ideas_storage_v1 import ROOT,WORK,read,sha


def main():
    queue=read(WORK/'audit/two_ideas_queue_v1.json')
    launch=read(WORK/'audit/two_ideas_queue_launch_v1.json')
    assert queue['pid']==launch['pid'];os.kill(queue['pid'],0)
    assert queue['status']=='WAITING_FOR_EXISTING_MAIN_LIFECYCLE_AND_EXCLUSIVE_GPU'
    progress=read(WORK/'components/bounded_v1/risk/progress.json')
    storage=read(ROOT/'audit/BrainHIVE_two_ideas_actual_storage_forecast_v1.json')
    registration=read(ROOT/'audit/BrainHIVE_two_ideas_registration_v1.json')
    assert read(ROOT/'audit/BrainHIVE_two_ideas_server_preflight_v1.json')['status']=='PASSED'
    assert sha(ROOT/'configs/BrainHIVE_bounded_main_recipe_v1.json')=='55d92772f3e76240799cf69df12f590229a408c7b7a48050b9a0c7ec01f31db3'
    before=ROOT/'audit/before_BrainHIVE_two_ideas_v1';assert not before.exists();before.mkdir()
    for name in ['STATUS.md','campaign_state.json','results.tsv','experiment_ledger.jsonl','budget.yaml']:
        shutil.copyfile(ROOT/name,before/name)
    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    record=dict(status='REGISTERED_TWO_IDEAS_QUEUE_LIVE_WAITING_NOT_FORMALLY_TRAINING',UTC=utc,
        queue=queue,queue_process_live=True,existing_training=progress,registration=registration,
        complete_lifecycle_storage=storage,formal_new_training_started=False,new_images=0,
        component_image_gain_established=False,local_large_files_created=False)
    receipt=ROOT/'audit/BrainHIVE_two_ideas_verified_launch_v1.json';assert not receipt.exists()
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    status=ROOT/'STATUS.md'
    opening=f'''# 当前执行状态

更新 {utc}。此前 Brain-HIVE 单 seed 原生 R80 复现已闭合；正在执行的风险组件为 {progress['updates']}/24807 更新，正式组件图片增益尚未确定。

根据最新指令新增且仅新增两个固定实验：① BrainKAN 真正替换 BrainMLP，6456512 参数，对照6452224参数；② 同容量525568参数的原生 IP 注意力响应正负目标排序修正。二者训练／生成／审计入口及实际服务器 CPU 检查通过，真实权重 GPU 前后向检查尚待执行。KAN 的17电极×10时间块贡献仅在 LayerNorm 前精确可加，未建立生理因果解释。

新队列 PID {queue['pid']} 已核验存活，状态 `{queue['status']}`。原风险→普通→父模型队列保持运行；新队列随后按 KAN→响应排序顺序，逐组完成训练、16600张V/R4原图、逐query评价及五方法统一比较表。seed1729、单图、512分辨率、4步、CFG0、IP1、全部10被试与保护划分不变；不按图片得分选checkpoint或追加搜索。

两组新增完整生命周期尚缺约 {storage['minimum_total_additional_bytes']/1e9:.2f} GB 持久空间（这是当前预测，非物理全量预分配）。新队列会先等现有三组闭合，再重新核验空间；未准入时明确等待并停止正式训练启动。现有空闲保护原三组；C/D没有新增大文件。三份legacy mean4文件是当前EEG实际源，不删除。

登记见 configs/BrainHIVE_two_ideas_sequence_v1.json；实际回执见 audit/BrainHIVE_two_ideas_verified_launch_v1.json。两项目前均无新训练或图片提升结论；通用跨架构、独立确认及解释忠实性仍需专门验证。

此前状态保留如下，旧的“KAN延后”描述由以上最新授权取代：

'''
    status.write_text(opening+status.read_text())
    campaign=read(ROOT/'campaign_state.json')
    campaign.update(last_update_utc=utc,next_two_ideas_queue=queue,
        next_two_ideas=['brainkan_full_v1','response_rank_v1'],next_two_ideas_admitted=False,
        next_two_ideas_formal_training_started=False,next_two_ideas_image_gain=False,
        next_two_ideas_storage_additional_bytes=storage['minimum_total_additional_bytes'],
        latest_user_KAN_and_two_fixed_ideas_authorization=True)
    (ROOT/'campaign_state.json').write_text(json.dumps(campaign,indent=2,ensure_ascii=False)+'\n')
    table=ROOT/'results.tsv'
    with table.open() as file:reader=csv.DictReader(file,delimiter='\t');rows=list(reader);fields=reader.fieldnames
    additions=[]
    for candidate,path in zip(['brainkan_full_v1','response_rank_v1'],registration['templates']):
        config=read(path);assert not any(r['run_id']==config['run_name'] for r in rows)
        row={key:'' for key in fields}
        row.update(run_id=config['run_name'],parent_run='BrainHIVE_full_source_dropout_v2',
            hypothesis=config['hypothesis'],status='registered_waiting_existing_main_and_storage',
            config_hash=sha(path),protocol_version='existing_protected_V_R4',train_seed='1729',subject_scope='all10',R='4',
            trainable_params=str(config.get('trainable_encoder_parameters',config.get('trainable_parameters'))),
            train_updates='0',expected_images='16600',generated_images='0',
            keep_decision='unresolved_not_trained',reason='Complete fixed recipe registered; actual GPU preflight and storage admission pending; no result imputation')
        additions.append(row)
    with table.open('a',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,delimiter='\t');writer.writerows(additions)
    with (ROOT/'experiment_ledger.jsonl').open('a') as file:
        file.write(json.dumps(dict(event_id='BrainHIVE_two_ideas_actual_queue_live_v1',UTC=utc,receipt=str(receipt),
            queue_pid=queue['pid'],new_GPU_jobs=0,new_images=0,scientific_goal_complete=False))+'\n')
    with (ROOT/'budget.yaml').open('a') as file:
        file.write('\nBrainHIVE_two_requested_ideas_v1:\n  fixed_new_configurations: 2\n  complete_lifecycle_storage_admitted: false\n'
            f"  additional_persistent_bytes_forecast: {storage['minimum_total_additional_bytes']}\n"
            '  large_local_C_D_artifacts: false\n  total_GPU_hours_cap: null\n  finite_job_seconds_per_training: 86400\n')
    print(json.dumps(dict(status=record['status'],queue_pid=queue['pid'],existing_training_updates=progress['updates'],
        storage_deficit=storage['minimum_total_additional_bytes'])),flush=True)


if __name__=='__main__':main()
