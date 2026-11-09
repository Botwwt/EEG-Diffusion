"""Summarize actual receipts, full-coverage curves, live jobs, and charged compute."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import shutil
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def idea_first_status():
    def load(path):return json.loads(Path(path).read_text())
    def sha(path):
        with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
    now=time.time();utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    charged=sum(x.get('charged_seconds',0) for x in map(json.loads,(ROOT/'audit/compute_ledger.jsonl').read_text().splitlines()))
    jobs=[]
    for path in sorted((ROOT/'audit').glob('*_job.json')):
        item=load(path)
        if item.get('status')!='RUNNING' or not item.get('start_utc'):continue
        command=Path('/proc',str(item.get('wrapper_pid',-1)),'cmdline')
        alive=(Path('/proc',str(item.get('pid',-1))).exists() and command.exists()
            and b'run_gpu_job.py' in command.read_bytes() and item['name'].encode() in command.read_bytes())
        if not alive:continue
        elapsed=max(0,now-datetime.datetime.strptime(item['start_utc'],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp())
        charged+=elapsed;jobs.append(dict(name=item['name'],pid=item['pid'],wrapper_pid=item['wrapper_pid'],seconds_this_job=elapsed,alive=True))
    config=load(ROOT/'configs/idea_first_ATMS_training_v1.json');run=Path(config['output_root'])
    roles={role:sum(load(p)['status']=='COMPLETED' for p in (run/role).glob('sub-*/completion.json')) for role in ['full','oof0','oof1']}
    cache_config=ROOT/'configs/idea_first_native_cache_v1.json';cache=None
    if cache_config.exists():
        receipt=Path(load(cache_config)['condition_cache_output_root'])/'completion.json'
        if receipt.exists():cache=dict(path=str(receipt),sha256=sha(receipt),**load(receipt))
    numerical=DATA/'error_validation/idea_first_ATMS_patience50_v1/summary.json'
    response=DATA/'error_validation/idea_first_full_UNet_response_v1/summary.json'
    validations={}
    for name,path in [('CPU',numerical),('full_UNet',response)]:
        if path.exists():
            data=load(path);validations[name]=dict(path=str(path),sha256=sha(path),status=data['status'],
                gate=data.get('gate',data.get('gate_passed')),queries=data.get('Dcheck_queries',data.get('queries')))
    aux_path=ROOT/'audit/idea_image_preparation_queue_state_v1.json';aux=load(aux_path) if aux_path.exists() else {}
    queue_path=ROOT/'audit/idea_interface_campaign_state_v1.json'
    if (ROOT/'audit/idea_interface_campaign_state_v2.json').exists():queue_path=ROOT/'audit/idea_interface_campaign_state_v2.json'
    queue=load(queue_path) if queue_path.exists() else {}
    arms={}
    for path in sorted((ROOT/'configs').glob('idea_interface_*_resolved_v1.json')):
        value=load(path);arm=value['arm'];state=ROOT/'audit'/f'idea_interface_{arm}_state_v1.json'
        entry=dict(state=load(state) if state.exists() else dict(status=value['status']))
        for kind,folder in [('training',value['training_root']),('images',value['image_root'])]:
            for filename in ['completion.json','progress.json']:
                receipt=Path(folder)/filename
                if receipt.exists():entry[kind]=dict(path=str(receipt),**load(receipt));break
        arms[arm]=entry
    bounds=load(ROOT/'configs/compute_time_bounds.json');occupied=max(0,(bounds['conservative_occupied_time_deadline_epoch']-now)/3600)
    remaining=max(0,min(40-charged/3600,occupied))
    stage='IDEA_FIRST_PREREQUISITES'
    if aux.get('status')=='RUNNING':stage='IDEA_FIRST_'+aux.get('phase','PREPARATION').upper()
    if queue.get('status')=='RUNNING':stage='IDEA_INTERFACE_'+queue.get('phase','RUNNING')
    elif queue.get('status','').startswith(('TERMINAL_','STOPPED_')):stage='IDEA_INTERFACE_'+queue['status']
    progress=dict(stage=stage,utc=utc,user_priority='Validate idea first, with strictly matched necessary references',
        seed=1729,user_confirmed_total_GPU_hour_cap=40,charged_job_hours_plus_prior_conservative_allowances=charged/3600,
        remaining_compute_hours=max(0,40-charged/3600),conservative_occupied_time_remaining_hours=occupied,
        remaining_authorized_hours=remaining,provider_billing_hours_independently_verified=False,live_GPU_jobs=jobs,
        new_aligned_ATMS_completed_by_role=roles,complete_OOF_subject_role_receipts=roles['oof0']+roles['oof1'],
        complete_scientific_OOF_cache_runs=int(cache is not None),complete_all10_CPU_error_validation_runs=int(numerical.exists()),
        native_cache_receipt=cache,validation_receipts=validations,preparation_queue=aux,idea_interface_queue=queue,
        actual_interface_arms=arms,formal_queries_per_arm=16600,training_queries_per_arm=132300,
        formal_EEG_image_gain_established=False,final_test_unlocked=(ROOT/'configs/FINAL_EVAL_UNLOCK.json').exists(),
        exact_full_ATM_reproduction_claimed=False,data_disk_free_GiB=shutil.disk_usage(DATA).free/1024**3,
        local_large_artifacts_created=False)
    diagnosis=ROOT/'audit/idea_first_new_encoder_diagnosis_compact_v1.json'
    if diagnosis.exists():
        value=load(diagnosis);progress['new_encoder_native_diagnosis']=dict(source=str(diagnosis),sha256=sha(diagnosis),
            status=value['status'],full_V_native_metrics=value['macro_metrics']['new_max100_patience50']['V'])
    gallery=ROOT/'audit/existing_REG_gallery_CPU_state_v1.json'
    if gallery.exists():
        value=load(gallery);progress['historical_full_gallery']=dict(status=value['status'],source=str(gallery),
            output=value['output'],check=value.get('materials_check'),historical_patience20_only=True)
        browser_proof=Path(value['output'])/'actual_browser_UI_QA_v1.json'
        if browser_proof.exists():
            proof=load(browser_proof);progress['historical_full_gallery']['browser_UI_QA']=dict(
                status=proof['status'],source=str(browser_proof),sha256=sha(browser_proof))
    (ROOT/'audit/current_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    state=load(ROOT/'campaign_state.json');state.update(stage=stage,last_update_utc=utc,
        verified_wait=jobs,active_controller_pid=queue.get('controller_pid'),active_state_path=str(queue_path),
        aligned_encoder_completion_by_role=roles,remaining_GPU_cap_hours=remaining,
        remaining_authorized_hours=remaining,new_EEG_image_gain_established=False,exact_full_ATM_reproduction_claimed=False)
    (ROOT/'campaign_state.json').write_text(json.dumps(state,indent=2)+'\n')
    if (ROOT/'STATUS.md').exists():
        archive=ROOT/'audit/status_history';archive.mkdir(exist_ok=True)
        previous=(ROOT/'STATUS.md').read_text();digest=hashlib.sha256(previous.encode()).hexdigest()
        saved=archive/(digest+'.md')
        if not saved.exists():saved.write_text(previous)
    lines=['# 当前实际状态：以 idea 验证为主','',f'更新时间：{utc}。阶段：{stage}。',
        f'新 ATMS 完成：full {roles["full"]}/10，oof0 {roles["oof0"]}/10，oof1 {roles["oof1"]}/10。完整折外缓存：{int(cache is not None)}；独立 CPU 校验：{int(numerical.exists())}；完整 UNet 响应校验：{int(response.exists())}。','',
        f'当前 GPU 作业：{json.dumps(jobs,ensure_ascii=False)}。',
        f'实际后续队列：{queue.get("status","尚未启动")}；阶段 {queue.get("phase",queue.get("preparation_phase","待前置检查"))}。','',
        '下一步依照最新 configs/idea_interface_campaign_plan_v2.json：完整生命周期预算允许时，优先承诺 RealPaired 真实配对参照及 Conditional 等匹配组；Frozen 仍必需，但其资源释放放到 idea 轮次之后。逐组完成全部图片、评价、核查与报告。',
        '所有组固定同一批新编码器、R4、划分、条件接口、SDXL512/20 Euler/CFG5/IP1、种子1729与每 query 一张图片。正式 V 为全部十被试16,600 query；T 的40张诊断图不能替代 V。',
        '当前公开 ATMS 类的 max100/patience50 设置已对齐；这不等于完整 ATM 生成链复现。适配器固定训练原生投影器8,400,896参数、一遍132,300 query；不把固定预算完成当作已经收敛。','',
        'CPU 相对条件保真度通过不代表模拟分布等于真实分布，也不代表图片改善。普通噪声、真实配对及正确/打乱/缺失 EEG 对照未完整时，机制结论保留未确定。尚未证实容量不足或新方法图片增益。',
        f'总上限40 GPU小时；已计费作业及先前保守额度 {charged/3600:.3f} 小时；更严格剩余额度 {remaining:.3f} 小时。供应商实际租用账单未独立核实。',
        f'服务器数据盘剩余 {progress["data_disk_free_GiB"]:.2f} GiB。大模型、数据、图片和检查点均留在服务器，未向 C/D 下载大文件。',
        '全部五组仍是完整机制判断的必需对照。预算不足记为未执行/未确定，不当作否定，不宣称完整任务成功或 SOTA。官方200刺激已暴露，无独立 H；历史实验和失败现场保留。','',
        '实际证据：audit/current_progress.json、model_registry.csv、experiment_ledger.jsonl、research/hypothesis_status.json。此前状态保留在 audit/status_history。','']
    if diagnosis.exists():lines+=['新编码器已完成全 T/V 原生条件诊断：reports/IDEA_FIRST_NEW_ENCODER_DIAGNOSIS_20261018.md。V 正确刺激 Top1 1.3193%、Top5 4.4217%（1660张 gallery）；这不是图片准确率或容量不足证明。','']
    for arm,entry in arms.items():
        train=entry.get('training',{})
        if train.get('stage')=='actual_image_denoising_training':
            updates=train['updates'];seen=train['seen_queries'];seconds=train['seconds']
            eta=(8269-updates)*seconds/max(1,updates)/3600
            lines += [f'实际运行：{arm} 图像去噪训练 {updates}/8269 次更新，覆盖 {seen}/132300 个训练 query。按当前累计吞吐粗估，剩余训练约 {eta:.2f} 小时；这不是图片完成时间或收敛保证。','']
    if gallery.exists() and progress['historical_full_gallery']['status']=='COMPLETED_EXISTING_REFERENCE_GALLERY_CPU_ONLY':
        lines+=['历史版本完整离线画廊已完成：16600行、415页；配对和盲评文件结构已核对。图片留在服务器，未进行人工评分；不属于新编码器或新 idea 的结果。','']
        if 'browser_UI_QA' in progress['historical_full_gallery']:
            lines+=['画廊实际浏览器检查已通过：筛选、搜索、分页及边界、图片加载、盲评标签隐藏均已核对；首个稳定页面已目视检查，未进行人工科学评分。见 reports/GALLERY_BROWSER_QA_20261018.md。','']
    (ROOT/'STATUS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(UTC=utc,stage=stage,completed_encoder_roles=roles,charged_hours=charged/3600,
        remaining_authorized_hours=remaining,live_GPU_jobs=jobs,idea_queue=queue.get('status'))))


def main():
    if (ROOT/'configs/idea_first_ATMS_training_v1.json').exists():return idea_first_status()
    base=DATA/'runs/S1_full_atm_native_v1/full';training=[]
    epochs=[json.loads(line) for line in (base/'epochs.jsonl').read_text().splitlines()] if (base/'epochs.jsonl').exists() else []
    for p in sorted(base.glob('sub-*/completion.json')):
        record=json.loads(p.read_text());enc=record['encoder'];prior=record['prior']
        best=next(row for row in epochs if row.get('stage')=='encoder' and row.get('subject')==int(record['subject'][-2:]) and row['epoch']==enc['best_epoch'])
        training.append(dict(subject=record['subject'],encoder_class='models.atms.ATMS()',encoder_parameters=enc['parameters'],
            encoder_gradient_tensor_active_parameters=enc['gradient_active_parameters'],encoder_epochs=enc['epochs'],encoder_updates=enc['updates'],
            encoder_seen_R4_examples=enc['seen_samples'],unique_T_stimuli_per_subject=enc['unique_training_rows'],
            encoder_best_epoch=enc['best_epoch'],encoder_T_native_MSE_at_best=best['mean_native_mse'],
            encoder_V_native_MSE_at_best=best['validation']['native_mse'],
            encoder_V_native_stimulus_Top1_at_best=best['validation']['native_stimulus_top1'],
            encoder_V_native_concept_Top1_at_best=best['validation']['native_concept_top1'],
            V_gallery_unique_stimuli=1660,prior_class='DiffusionPriorUNet(cond_dim1024,dropout.1)',prior_parameters=prior['parameters'],
            prior_epochs=prior['epochs'],prior_updates=prior['updates'],prior_seen_R4_examples=prior['seen_samples'],
            prior_best_epoch=prior['best_epoch'],prior_V_fixed_noise_epsilon_MSE_at_best=prior['best_validation_loss'],
            encoder_checkpoint=enc['checkpoint'],encoder_sha256=enc['checkpoint_sha256'],prior_checkpoint=prior['checkpoint'],prior_sha256=prior['checkpoint_sha256'],
            initialized_from='scratch_both_EEG_encoder_and_prior',inference_weights_frozen=True))
    if training:
        with (ROOT/'audit/full_reference_training_summary.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(training[0]));writer.writeheader();writer.writerows(training)
    jobs=[];charged=0.
    ledger=ROOT/'audit/compute_ledger.jsonl'
    if ledger.exists():charged=sum(json.loads(line).get('charged_seconds',0) for line in ledger.read_text().splitlines())
    for p in sorted((ROOT/'audit').glob('*_job.json')):
        d=json.loads(p.read_text())
        if d.get('status')=='RUNNING' and d.get('start_utc'):
            pid=d['pid'];live=Path('/proc',str(pid)).exists()
            elapsed=max(0,time.time()-datetime.datetime.strptime(d['start_utc'],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp())
            if live:charged+=elapsed
            jobs.append(dict(name=d['name'],pid=pid,wrapper_pid=d.get('wrapper_pid'),alive=live,seconds_this_job=elapsed))
    queue=ROOT/'audit/S1_queue_state.json';q=json.loads(queue.read_text()) if queue.exists() else dict(status='WAITING_FULL_REFERENCE_TRAINING')
    images={}
    for p in (DATA/'images/S1_full_native_batchsafe_v2/V').glob('*/progress.json'):
        images[p.parent.name]=json.loads(p.read_text())
    progress=dict(stage='S1_FULL_DATA_REFERENCE_AND_ACTUAL_IMAGES',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        training_subjects_completed=len(training),training_subjects_required=10,formal_training_started=True,
        all_training_and_R1_R4_R80_test_inputs_verified=True,seed=1729,user_confirmed_total_GPU_hour_cap=40,
        charged_job_hours_plus_prior_conservative_allowances=charged/3600,remaining_compute_hours=40-charged/3600,
        provider_billing_hours_independently_verified=False,live_GPU_jobs=jobs,queue=q,image_progress=images,
        formal_EEG_image_gain_established=False,final_test_unlocked=(ROOT/'configs/FINAL_EVAL_UNLOCK.json').exists(),
        data_disk_free_GiB=shutil.disk_usage(DATA).free/1024**3,
        local_large_artifacts_created=False,new_native_prior_clipping_contract_issue_proven=True)
    progress['prior_full_image_prerequisite_pending']=(ROOT/'audit/prior_sampling_prerequisite_pending.json').exists() and not (ROOT/'audit/prior_sampling_prerequisite_check.json').exists()
    progress['velocity_prior_subjects_completed']=len(list((DATA/'runs/S1_prior_velocity_repair_v1/full').glob('sub-*/completion.json')))
    postprocessing=ROOT/'audit/S1_postprocessing_state.json'
    if postprocessing.exists():progress['CPU_postprocessing_queue']=json.loads(postprocessing.read_text())
    controls=ROOT/'audit/S1_EEG_control_queue_state.json'
    if controls.exists():progress['EEG_control_queue']=json.loads(controls.read_text())
    oof_states=[]
    for path in sorted((ROOT/'audit').glob('S3_native_oof_*_state.json')):
        item=json.loads(path.read_text())
        oof_states.append(dict(state_path=str(path),**item))
    if oof_states:
        progress['native_OOF_stages']=oof_states
        latest=max(oof_states,key=lambda x:x.get('last_update_utc',''))
        if latest['status']=='RUNNING':progress['stage']='S3_REAL_CONCEPT_OOF_CACHE_AND_CPU_DCHECK'
        elif latest['status']=='COMPLETED_ALL10_OOF_CACHE_AND_CPU_DCHECK':
            progress['stage']='S3_CPU_FIDELITY_COMPLETE_FULL_UNET_AND_IMAGE_MECHANISM_UNCONFIRMED'
        else:progress['stage']='S3_INCOMPLETE_INSPECT_REGISTERED_STATE_AND_PRESERVED_OUTPUTS'
    completed_oof_receipts=set();complete_oof_caches=set()
    for path in sorted((ROOT/'configs').glob('S3_native_oof_*_resolved.json')):
        resolved=json.loads(path.read_text());config=json.loads(Path(resolved['training_config_path']).read_text())
        cache_config=json.loads(Path(resolved['cache_config_path']).read_text())
        for role in ['oof0','oof1']:
            for receipt in (Path(config['output_root'])/role).glob('sub-*/completion.json'):
                if json.loads(receipt.read_text())['status']=='COMPLETED':completed_oof_receipts.add(str(receipt))
        receipt=Path(cache_config['condition_cache_output_root'])/'completion.json'
        if receipt.exists() and json.loads(receipt.read_text())['status']=='COMPLETED_ALL10_CONCEPT_OOF_T_AND_MATCHED_D_CHAINS':
            complete_oof_caches.add(str(receipt))
    progress['complete_OOF_subject_role_receipts']=len(completed_oof_receipts)
    progress['complete_scientific_OOF_cache_runs']=len(complete_oof_caches)
    progress['complete_all10_CPU_error_validation_runs']=sum(s['status']=='COMPLETED_ALL10_OOF_CACHE_AND_CPU_DCHECK' for s in oof_states)
    controls_deferred=progress.get('EEG_control_queue',{}).get('status','').startswith('DEFERRED_')
    controls_description=('Controller was stopped while waiting, before any control GPU job. '
        'Review the complete mechanism/control/final budget before restarting; no intervention is running.'
        if controls_deferred else 'The finite controller waits for both references and CPU audits, '
        'then applies the locked selection rule and complete-control budget gate.')
    replay=ROOT/'audit/V_original_batch_replay_and_isolation.json'
    if replay.exists():
        p=json.loads(replay.read_text())
        progress['original_batch_replay']=dict(status=p['status'],queries=len(p['queries']),
            formal_JPEG_exact=p['exact_original_formal_JPEG_replay_all_queries'],
            unchanged_query_count=p['unchanged_query_count'],
            all_unchanged_query_latent_deltas_zero=p['all_unchanged_query_latent_deltas_zero'])
    native=ROOT/'audit/frozen_encoder_native_diagnosis.json'
    if native.exists():
        diagnosis=json.loads(native.read_text())
        progress['full_T_V_native_condition_diagnosis']=diagnosis['status']
        progress['condition_diagnosis_establishes_capacity_limit']=False
    invalid=DATA/'images/S1_full_native/V/REF_REG_correct_R4'
    progress['invalid_batch_mixed_images_preserved']=len(list(invalid.glob('sub-*/*.jpg'))) if invalid.exists() else 0
    progress['batch_condition_isolation_verified']=(ROOT/'audit/batched_native_condition_check.json').exists()
    bounds=ROOT/'configs/compute_time_bounds.json'
    if bounds.exists():
        b=json.loads(bounds.read_text())
        progress['conservative_occupied_time_remaining_hours']=max(0,(b['conservative_occupied_time_deadline_epoch']-time.time())/3600)
        progress['remaining_authorized_hours']=min(progress['remaining_compute_hours'],progress['conservative_occupied_time_remaining_hours'])
    (ROOT/'audit/current_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    state=json.loads((ROOT/'campaign_state.json').read_text());state.update(stage=progress['stage'],formal_training_started=True,
        training_subjects_completed=len(training),budget_question_pending=False,confirmed_total_gpu_hours=40,
        verified_wait=jobs,last_update_utc=progress['utc'],last_goal_turn_classification='progress')
    (ROOT/'campaign_state.json').write_text(json.dumps(state,indent=2)+'\n')
    status=f'''Full-data single-seed EEG autoresearch: {progress['stage']}. Updated {progress['utc']}.

User confirmed total40 GPUh. Accounted GPU-job hours plus prior conservative allowances: {charged/3600:.3f}; ledger remaining {40-charged/3600:.3f}; tighter authorized remaining {progress.get('remaining_authorized_hours',40-charged/3600):.3f}. Provider rental billing is not independently verified. No resource purchase/renewal.

All10 subjects have verified training inputs and genuine test R1/R4/R80. All16740 image identities checked. Full V16600 queries; real gallery1660 images; official benchmark2000 queries per R. One image/query, seed1729. Historical official200 exposure is disclosed; no independent H established.

Formal full ATMS/native-prior subjects completed: {len(training)}/10. ATMS default3200413 parameters; prior9675648. Encoder up to100 epochs with registered patience20; official benchmark patience50 differs. Prior full400 epochs. Gradients use T only; V selects checkpoints. Actual counts/checkpoints: audit/full_reference_training_summary.csv. Full T exposure is not proof of convergence or exact official reproduction, and loss/retrieval curves do not establish image success.

Fixed formal image track: standard SDXL base1.0,512,20 Euler steps,CFG5,IP scale1,one fixed query seed. Uniform JPEG90/subsampling0. Actual source and weights are audited; all learned/image files stay on server. Native prior x0 clipping disabled after exact-epsilon contract failure; not claimed as the cause of all legacy failures.

The first batched generation attempt is INVALID: upstream IP helper repeated already-batched EEG conditions, mixing four conditions into each query. All {progress['invalid_batch_mixed_images_preserved']} images/receipts preserved under images/S1_full_native. Training and historical single-query generation unaffected. Fixed helper has actual UNet isolation proof: other query delta0, changed query delta0.5159, batch/single relative epsilon RMSE0.00591. Valid reruns use images/S1_full_native_batchsafe_v2 only.

Corrected T oracle images recover four distinct objects. The fully trained epsilon prior still yields grids/unrelated T content; high-noise epsilon errors amplify to implied native x0 MSE~700. This is a specific readiness failure, not a full-V rejection of epsilon priors or capacity proof. One same9675648-parameter v-objective repair is registered: all10 frozen encoders reused,400epochs5200updates per subject, same data/seed/sampling, max0.5GPUh. Completed subjects {progress['velocity_prior_subjects_completed']}/10. No novelty claim.

SEED frozen GIT/MiniLM/EffNet/GroundingDINO weights loaded and actual T oracle evaluation passed. Streamed EffNet preprocessing exact; ObjectF1 equal to pinned official function on self/oracle/shuffle. Full per-query evaluation code uses explicit stimulus joins.

Finite two-reference queue status: {q['status']}. Full V image arm progress: {json.dumps(images)}. Prior full-image prerequisite pending: {progress['prior_full_image_prerequisite_pending']}. The corrected four-T-image v diagnostic passed operational readiness only; the epsilon configuration remains a preserved T-readiness failure. Same-capacity v conditions restored some recognizable content, but full-image improvement remains unconfirmed. Separate finite EEG-control queue: {progress.get('EEG_control_queue',{}).get('status','not registered')}. {controls_description} It cannot start training or unlock final benchmark. Any new mechanism requires a matched full-control budget gate. Current EEG adaptation gains/SOTA are not established.

Fixed-eight V true-condition oracle and exact original four-query replay are diagnostic only. All four formal JPEGs replayed exactly. Replacing only the badger condition changed its content; the other three latent trajectories and JPEGs stayed identical. Native Top5 can coexist with wrong generated objects. This supports condition responsiveness/isolation and an EEG-condition content/distribution bottleneck, not a capacity claim or full image accuracy. See audit/V_original_batch_replay_and_isolation.json and audit/V_fixed8_oracle_diagnosis.json. Both bounded pauses resumed the REG worker; elapsed time is charged once inside that job.

OOF/cache, native error-channel estimators and the complete CPU Dcheck runner are implemented. Analytical/synthetic checks passed, including classifier shifted/null controls and Dcheck not altering fitted weights. The initial random-linear-initialization control failure is retained. Source/seed/draw/classifier/energy-score/decision rules are locked in configs/native_error_channel_validation_plan_v1.json. Actual complete OOF subject-role receipts: {len(completed_oof_receipts)}; complete scientific OOF cache runs: {len(complete_oof_caches)}; completed all10 CPU error-validation runs: {progress['complete_all10_CPU_error_validation_runs']}. The runner checks full cache/model provenance; CPU/projector fidelity cannot replace full denoiser response or full images.

Actual T-only deployed v-prior cache profiling measured0.027725seconds/query at batch4,100 prior sample-evaluations/query. The complete181800-query OOF/D cache sampling alone projects1.400h, excluding teacher training, loads/hash/I/O,validation and margin. The old whole-stage1.5h reservation is insufficient evidence of complete funding for a velocity chain; recompute the selected-chain budget while preserving controls/final and total40h. The bounded profile paused REG9.235seconds and resumed it; charged once inside REG. REF_REG has no semantic-prior sampling term.

Actual registered OOF stage states (empty means none started): {json.dumps(oof_states)}. A completed CPU channel check is not an image gain or permission to start an unfunded interface experiment.

Full T/V frozen-condition CPU audit is saved in audit/frozen_encoder_native_diagnosis.json when complete. It compares prediction errors against a T-only constant mean to avoid interpreting raw CLIP anisotropy as decoding quality; native retrieval is not image performance. Paired statistics are locked in configs/paired_statistics_plan.json. Separate CPU-only postprocessing will produce full paired differences, supplementary PixCorr/SSIM, saved-image integrity counts, complete offline pagination and blinded materials. Human ratings have not been collected; perceptual quality noninferiority is not established.

Server data disk free: {progress['data_disk_free_GiB']:.2f}GiB. Earlier134 unrelated directories cleaned; current EEG datasets/code/checkpoints/images/negative evidence preserved. No large C/D downloads.
'''
    (ROOT/'STATUS.md').write_text(status)
    print(json.dumps(dict(UTC=progress['utc'],training_subjects_completed=len(training),queue=q['status'],charged_hours=charged/3600,live_jobs=jobs,image_progress=images)))


if __name__=='__main__':main()
