"""Record the actual finite queue handles and preserve all prior research history."""
import datetime,hashlib,json
from pathlib import Path
from zoneinfo import ZoneInfo
L=Path(__file__).resolve().parent;O=L/'audit/HIVE_noordinary_factorial_queue_20261031_v1'
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):Path(p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def ref(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
actual=read(O/'actual_verified_waiting_observation.json');main=read(O/'actual_launch.json');secondary=read(O/'secondary_actual_launch.json')
assert actual['waiter']['exact_live'] and actual['secondary_process']['exact_live']
assert len(read(O/'independent_local_registration_audit.json')['verified_sources'])==18
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S')
path=L/'audit/OVERNIGHT_active_execution_handoff_20261031_v1.json';assert not path.exists()
handoff=dict(status='ACTUAL_FINITE_OVERNIGHT_OLD_ONLY_QUEUES_VERIFIED_LIVE',snapshot_UTC=now.isoformat(),actual_server_observation_UTC=actual['UTC'],previous_handoff='audit/AR_active_execution_handoff_20261029_v165.json',
    goal_status='paused',automatic_unbounded_goal_not_resumed=True,user_authorization='给我挂一堆你需要的实验，我得明天才能检查了',task_scope='Concrete finite overnight experiments only',
    OLD_only=True,NEW_reserved=True,single_training_seed=1729,original_ordinary_immutable=True,formal_D_and_old_stopped_families_remain_stopped=True,
    original_three_controllers=actual['original_upstreams'],original_stage=actual['original_upstreams'][0]['state']['controller_progress.json'],
    original_completed_fits=['joint_correct','EEG_only_correct','P_only_correct'],original_active_fit='joint_swapped_targets',original_pending_fit=['joint_absolute_only'],
    original_scientific_gates_sources_and_output_caps_unchanged=True,new_main_waiter=main,new_secondary_waiter=secondary,actual_new_main_state=actual['state'],actual_new_secondary_state=actual['secondary_state'],
    fixed_new_experiment=dict(training_arms=['joint_correct','joint_swapped_targets','joint_absolute_only'],ordinary_in_training=False,subject='sub-01',T_stimuli=13230,T_concepts=1323,physical_EEG='R4',
        per_arm_complete_T_passes=1,per_arm_pair_updates=6615,per_arm_EEG_exposures=13230,trainable='existing EEG weights except final LN affine + native P FP32 master weights',
        sampling='native512/foursteps/CFG0/IP1/same query noise/singleimage',image_paths=8,new_image_maximum=3200,new_inference_parameters=0,
        predeclared_hypotheses=['standalone noO-trained main vs native and its two trained controls','same fixed noO-trained main composed with original O vs O and composed controls'],
        interaction='(new_plus_ordinary-ordinary)-(new-parent); descriptive conditional inference interaction, not biological causality',
        gate80='Both modes clearly harmful (DINO delta<=-.02, CI upper<0) stops all; otherwise all8 to400',gate400='Each mode independently requires DINO>=+.01/CI lower>0/no withinconcept Top1 loss plus positive DINO CIs vs its two controls',
        secondary='Only after actual positive400 and noO waiter exit: all8 EffNet/CapSim/ObjectF1/SEED/PixCorrRGB/SSIMY and42 comparisons; each mode retains its own original gate',
        seed_stability_independent_H_cross_subject_SOTA_NOT_established=True),
    actual_checks=dict(server_main_CPU_checks=11,server_secondary_CPU_checks=6,local_exact_source_files=18,local_registration_audit=ref(O/'independent_local_registration_audit.json'),
        new_actual_GPU_profile='PENDING; first GPU task AFTER original three exits, three temporary updates discarded, no-O call hooks and E/P gradients/serialization/native latent parity required before training',new_training_updates=0,new_research_images=0),
    storage=dict(new_main_cap_bytes=32<<30,new_secondary_cap_bytes=2<<30,latest_actual_resource_review=secondary['resource_review'],checkpoint_every_updates=512,checkpoint_also_at_each_normal_segment_end=True,maximum_checkpoints_per_arm=45,
        old_checkpoints_not_deleted=True,not_physical_reservation=True),
    runtime_limits=dict(OLD_GPU_bytes=20<<30,extra_CPU_bytes=10<<30,job_seconds=900,normal_prefix_seconds=650,new_main_wait_max_hours=32,new_main_lifecycle_max_hours=48,new_secondary_wait_max_hours=81,new_secondary_controller_max_hours=96,no_failed_job_auto_retry=True),
    initial_launch_guard_failure=dict(reason='Local Windows write_text converted LF to CRLF; hash mismatch rejected BEFORE Popen',preserved=True,actual_exact_server_metadata_received=True,old_science_changed=False),
    confirmed_new_method=False,SOTA=False,CCF_A=False,requires_user_input=False,
    next_safe_actions=[
      'Use bounded OLD RAM SSH helper + read_HIVE_noordinary_factorial_OLD_CPU_20261031_v1.py with host:old to freshly verify exact new413043/start105377481 and413172/start105390496 plus original3. Actual08:24UTC snapshot is historical; observer expiry/SSH failure is never training terminal.',
      'Do not restart or edit original core364356/start99329292, original SEED366479/start99515864, original input368698/start99767468, or either new queue. All five server controllers are independent of local chat/observers.',
      'New noO waiter must see original complete all5 fit audit, final80/400 independent image audit and all original3 exact exits before its noO GPU profile. A valid scientific STOP remains STOP but does not cancel the separately registered independence experiment. Any upstream runtime failure fails the new waiter closed.',
      'If new profile fails or finite slots/resources expire, retain failure/prefix, do not change thresholds or automatically retry. Do not stop unrelated GPU work. NEW stays reserved. No multiseeds.',
      'After original core completes use existing terminal collector/local independent reader/render. After new core completes inspect its separate all3 fit audit/all8 token+image audit/composition_interaction; original terminal collector hardcodes5/7 and CANNOT be reused unmodified for3/8.',
      'New all8 secondary evaluator only starts after its exact new noO waiter exits and positive400 original gates. STOP withdraws it with GPU0. Follow actual secondary independent audit; no selecting a control/epoch/strength to rescue failed science.',
      'All new work is a single-subject development validation, not full10 subjects, full1654 fitting, independentH, confirmed novelty or SOTA. The user paused the automatic goal; do not infer that the finite queue resumed an unbounded loop.'
    ])
write(path,handoff)
for rel in ['configs/research_resource_allocation_latest_20261029.json','configs/research_priority_single_seed_latest_20261030.json']:
    p=L/rel;v=read(p);v['latest_handoff']='audit/OVERNIGHT_active_execution_handoff_20261031_v1.json';v['latest_authoritative_handoff']=v['latest_handoff'];v['snapshot_UTC']=now.isoformat();v['finite_overnight_queue']=dict(main=main,secondary=secondary,actual_observation_UTC=actual['UTC'],automatic_unbounded_goal_resumed=False,OLD_only=True,NEW_reserved=True,seed=1729);write(p,v)
ag=L/'AGENTS.md';text=ag.read_text(encoding='utf8');text=text.replace('Latest handoff: audit/AR_active_execution_handoff_20261029_v165.json','Latest handoff: audit/OVERNIGHT_active_execution_handoff_20261031_v1.json',1)
text+='\n\n- AUTHORITATIVE FINITE OVERNIGHT TASK '+now.isoformat()+': user requested unattended finite experiments. Original3 handles remain LIVE and unchanged; joint/E-only/P-only fits6615 complete, swapped active, absolute pending. NEW noO waiter413043/start105377481 and all8-secondary waiter413172/start105390496 VERIFIED LIVE at'+actual['UTC']+'. Three new noO-trained fullT sub01 fits, native/O×each fixed new model =8image paths, fixed80->400/DINO+withinconcept/42contrasts/composition analysis; conditional all8 SEED/ObjectF1/lowlevel onlyafterpositive400. CPU11+6checks and18local sourceSHA passed; real newGPUprofile/fit/images PENDING. New32GiB+2GiB caps include old remaining caps/50GiBlegacy/8GiBbuffer; no deletions. Ordinaryimmutable/OLDonly/NEWreserved/single1729/nofailedautoretry. New queue waits original complete scientific closure and all3 exits, but retains original STOP. Automatic unbounded goal remains paused; this new finite user task does not resume it. Follow new handoff; all earlier paragraphs are historical.\n'
ag.write_bytes(text.encode())
status=L/'STATUS.md';previous=status.read_text(encoding='utf8')
note=f'''# 最新有限夜间队列：{stamp}（America/Los_Angeles）

用户已授权挂接无人值守实验。服务器实际核验于 {actual['UTC']}：原五臂主队列及两个原条件评价控制器仍在线；joint、EEG-only、P-only三个完整T拟合已完成，正在运行错配目标对照。原任务、源码、门槛和ordinary不变。

新增训练/图片队列 **PID 413043 / start 105377481** 已在线等待原三个控制器全部闭合退出；新增全八组语义评价队列 **PID 413172 / start 105390496** 已在线条件等待。两者均在服务器独立运行，关闭本地聊天不影响排队。

- 三个新增拟合均不使用ordinary：正确配对、同概念错配目标、只用绝对语义目标。每臂sub-01完整T13230刺激、6615配对更新、单seed1729，更新现有EEG权重和原生P，冻结最终LN affine与原ordinary。
- 三个最终模型分别单独使用或组合原版ordinary，加原生与ordinary两个基线，共八条原生四步图片路径。固定80→400、最多3200张；单独效果、组合增量、同概念Top1及两项训练对照全部报告。交互分析不会挽救失败的主门槛。
- 任一事前主假设过完整400门槛后，再自动计算八组EffNet、CapSim、ObjectF1、SEED、PixCorr_RGB、SSIM_Y及42项配对比较，独立复算图像身份、指标公式和区间；未通过则撤下此条件评价。

实际服务器CPU检查11+6项通过，18份新源码与注册文件已取回并逐项核对。**新GPU工程检查、三组新增拟合与图片效果尚未执行**；GPU工程是接续后的第一项任务，失败即停止。32GiB训练/图片与2GiB评价上限已连同原任务剩余额度核算，最后规划余量约7.25GB；这不是文件系统预留。只用旧卡，第二张卡保留给用户，不跑多seed、不覆盖旧ordinary、不删除旧结果。

这是一组单被试开发验证，不是十被试最终验证、确认创新或SOTA。新增训练按现有速度估计约14小时，另加原有剩余任务、出图和条件评价，整队可能超过一夜。所有作业有有限时长、正常前缀断点和失败停止；不会自动重试失败作业。

入口与准确状态：`audit/OVERNIGHT_active_execution_handoff_20261031_v1.json`、`audit/HIVE_noordinary_factorial_queue_20261031_v1/actual_verified_waiting_observation.json`。自动无限迭代目标仍暂停，本次仅执行用户新授权的有限夜间队列。以下均为历史快照。

---

'''
status.write_bytes((note+previous).encode())
write(O/'delivery.json',dict(status='COMPLETE_ACTUAL_FINITE_OVERNIGHT_QUEUE_DELIVERY',handoff=ref(path),main=main,secondary=secondary,actual_server_UTC=actual['UTC'],actual_checks=handoff['actual_checks'],task_complete='finite queue implemented, checked, launched and observed waiting; scientific results remain pending',automatic_goal_resumed=False))
print(json.dumps(dict(handoff=str(path),updated_STATUS=True,main_PID=main['PID'],secondary_PID=secondary['controller_PID'])))
