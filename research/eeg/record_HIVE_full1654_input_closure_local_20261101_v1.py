"""Close the actual input stage; preserve the full active research objective."""
import hashlib,json
from pathlib import Path

root=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p);raw=p.read_bytes()
    return dict(path=str(p.relative_to(root)).replace('\\','/'),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
assetdir=root/'audit/HIVE_exposed_OOF_asset_complete_20261101_v1'
asset=read(assetdir/'local_independent_scope_and_receiving.json')
assert asset['actual_fits']==20 and asset['updates']==4500 and asset['exported_queries']==165400 and asset['all500_epoch_audit_entries_locally_checked']
bank=read(assetdir/'actual/input_bank_completion.json');closed=read(assetdir/'actual/recovery1_controller_completion.json')
consumerdir=root/'audit/HIVE_exposed_fullfit_component_inputs_preparation_20261101_v3'
consumer=read(consumerdir/'actual_input_consumer_CPU_review.json');checks=read(consumerdir/'actual_local_CPU_checks.json')
assert consumer['all165400_source_raw_output_joins_bit_exact'] and consumer['all20_actual_model_LN_affines_bit_exact'] and consumer['raw_condition_join_cells']==169369600
assert checks['count']==18 and not consumer['component_or_benchmark_admitted']
profiledir=root/'audit/HIVE_exposed_full1654_F_geometry_profile_20261101_v1';p=read(profiledir/'code/profile_completion.json');observation=read(profiledir/'actual_observation.json')
assert p['status']=='PASS_ACTUAL_FULL1654_F_GEOMETRY_ENGINEERING_ONLY' and not p['learned_weights_saved'] and not p['final_fits_admitted']
assert [a['temporary_updates'] for a in p['arms']]==[4,4] and all(a['initial_zero_head_and_disabled_native_bit_exact'] and a['serialization_model_Adam_RNG_bytes_exact'] for a in p['arms'])
assert not observation['wrapper']['exact_live'] and not observation['worker']['exact_live'] and observation['GPU_receipt']['exit_code']==0
previous='audit/GOAL_active_execution_handoff_20261101_v7.json';new='audit/GOAL_active_execution_handoff_20261101_v8.json';target=root/new;assert not target.exists()
next_actions=[
    'Do not restart completed464771/466643 or profile475746/475747. All20 real folds and bank are complete. OLD GPU was empty at the dated03:56:14UTC observation, not a permanent idle guarantee; recheck before a new job. NEW remains reserved.',
    'Implement actual full1654 bounded_F/Euclidean_F trainers, exact31014-update/496200-exposure coverage and optimizer/state/CPU forward audit using the verified bank and own20affines. Reuse immutable head formulas, fresh seed1729 initialization and fixed last update. Profile weights were discarded and must never become fits.',
    'Bind a separate truthful exposed-benchmark method/calibration/resource protocol and the entire generation/scoring/audit lifecycle before admitting final fits. Proposed strength1 is predeclared from disclosed prior evidence, not recalibrated on oldDfit(nowtrain) or new official200 outcomes. Draft alone is not admission.',
    'For full1654 ordinary epsilon+token, implement/review exposure-aware full16540 native VAE-mode latent construction from exact train identities, full native generator profile and matched resources; keep old13230T bank/originalordinary unchanged. Do not fake legacy false-exposure gates.',
    'All proposed final arms need actual full2000 R80 single-image/512/native4step generation, source/noise/pixel joins and all16 locked metrics. Published full1654 parent checkpoints may be reused after exact source/affine checks; originalT-trained heads are not new fullscope heads. Controls and external strong systems retain equal information/cost opportunities.',
    'Continue substantial EEG-grounded method research using actual paired images. Prior jointE/P and noO families remain closed negative; do not promote controls or repeat generic pretrainedLaBraM+VAR as new innovation. First-method completeness and a second confirmed mechanism are still unachieved. Goal stays active.'
]
h=dict(status='ACTIVE_GOAL_ACTUAL_FULL1654_OOF_BANK_AND_CONSUMER_COMPLETE_F_GEOMETRY_PROFILE_PASS_FINAL_FITS_PENDING',snapshot_UTC=observation['UTC'],
    objective='你一直迭代 每次方向都要思考创新变化大一点 可以参考借鉴根据上一个运行的日志进一步改善 仔细思考！然后如果有好效果就继续改进直到达到ccfa水准',
    goal_status='active',goal_complete=False,previous_handoff=previous,previous_goal_turn_classification='no progress: status-only answer',
    current_goal_turn_classification='progress: remaining17 actual full folds completed with independent audits, complete165400 bank and independent consumer verified, new actual fullscope engineering probe passed',
    OLD_only=True,NEW_reserved=True,single_training_seed=1729,ordinary_SHA256='a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03',ordinary_immutable=True,
    stopped_families_remain_stopped=['formal_HIVE_D','withdrawn_AR','native_categorical','terminal_real_prefix','paired_semantic_joint5','noO_joint3'],
    active_research_handles=[],completed_handles=[dict(PID=466643,start_ticks=112273987),dict(PID=475746),dict(PID=475747)],
    full1654_input_stage=dict(status='COMPLETE_REAL20FOLDS_AND165400_RAW_BANK',protocol=bank['protocol'],input_bank=closed['input_bank'],subjects=10,concepts=1654,images=16540,physical_repeats='R4',
        actual_new_native_models=20,epochs_per_model=25,updates_per_model=225,total_updates=4500,total_EEG_exposures=4135000,held_OOF_queries=165400,
        independent_all8270_CPU_FP64_forward_checks_per_fold=True,independent_all25_epoch_Adam_counter_moment_shuffle_checks_per_fold=True,
        maximum_manual_CPU_GPU_absolute_error=asset['independent_CPU_forward_max_absolute_error'],not_all_gradient_transitions_replayed=True,
        local_receiving_and_scope=ref(assetdir/'local_independent_scope_and_receiving.json'),terminal_observation=ref(assetdir/'terminal_observation.json'),
        stage_controller_completion=ref(assetdir/'actual/recovery1_controller_completion.json'),original_failure_preserved=True,
        owned_cache_reviews=18,no_files_deleted=True,original24GiB_memory_floor_unchanged=True,original_wall_deadline_unchanged=True),
    actual_consumer=dict(status=consumer['status'],review=ref(consumerdir/'actual_input_consumer_CPU_review.json'),source=consumer['source'],local18checks=ref(consumerdir/'actual_local_CPU_checks.json'),
        all169369600_join_values_bit_exact=True,all20_own_LN_affines_bit_exact=True,all16540_F_targets_finite=True,geometry=consumer['own_affine_FP64_geometry'],
        CPU_only=True,seconds=consumer['seconds']),
    actual_F_geometry_profile=dict(status=p['status'],protocol=p['protocol'],completion=ref(profiledir/'code/profile_completion.json'),observation=ref(profiledir/'actual_observation.json'),
        two_arms=[a['arm'] for a in p['arms']],temporary_updates_per_arm=4,total_temporary_updates=8,learned_weights_saved=False,research_images=0,
        CPU_FP64_max_absolute_errors=[a['same_formula_CPU_FP64_max_absolute_error'] for a in p['arms']],future_GPU_bit_exactness_not_tested=True,
        program_seconds=p['seconds'],GPU_wrapper_seconds=observation['GPU_receipt']['charged_seconds'],peak_GPU_bytes=p['peak_GPU_bytes'],full31014_fit_not_yet_started=True),
    failures_and_corrections=ref(consumerdir/'engineering_errata.json'),
    next_full1654_completion_DRAFT='configs/HIVE_exposed_full1654_single_seed_completion_DRAFT_20261101_v1.json',
    old_Dfit_Dcheck_V_now_train=True,official200_previously_exposed=True,official_results_used_for_previous_direction=True,legacy_false_exposure_gate_not_edited=True,
    independent_H=False,final_component_or_benchmark_admitted=False,confirmed_second_method=False,SOTA=False,CCF_A=False,
    previous_method_results=dict(noO_all8='audit/HIVE_noordinary_factorial_complete_20261101_v1/scientific_readout.json',original5='audit/HIVE_paired_semantic_full_study_complete_20261030_v1',
        ordinary_R80='reports/BRAINHIVE_ORDINARY_R80_FULL_RESULTS_20261029_v1.md',independent_positive_VAR_target_line='reports/EEG_NATIVE_MARGINAL_TARGET_FULL400_RESULTS_20261029_v3.md'),
    primary_and_existing_experiment_review='audit/HIVE_EEG_pretraining_direction_recheck_20261101_v1/actual_primary_and_local_evidence_review.json',next_safe_actions=next_actions)
target.write_text(json.dumps(h,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
agents=root/'AGENTS.md';text=agents.read_text(encoding='utf8').replace('Latest handoff: '+previous,'Latest handoff: '+new,1)
old='- Authoritative current state:';lines=text.splitlines()
for n,line in enumerate(lines):
    if line.startswith(old):
        lines[n]='- Authoritative current state: prior5fit/noO studies are CLOSED NEGATIVE. Actual full1654 train-only stage is now COMPLETE:20 new native concept-OOF models/165400 raw conditions and independent consumer all169369600 joins/all20 LN affines passed. Actual F-geometry4 temporary updates/arm profile passed/discarded. All queues exited; final31014-step component fits and full2000-image lifecycle remain UNIMPLEMENTED/NOT ADMITTED. Keep goal active; OLD only, NEW reserved, ordinary immutable, single1729.';break
text='\n'.join(lines)+'\n'
text+=f"\n- AUTHORITATIVE {observation['UTC']}: full20fold/165400 OOF asset stage COMPLETE, exact466643 absent; all121files/500epoch audit entries locallychecked, actual169369600 joins/all20affines CPUconsumer PASS5.015s. Corrected runtime ownership and preserved all failures. F-geometry actual profile475746/475747 COMPLETE/EXITED,4discardedupdates each, GPUwrapper7.025s/CPUFP64max3.866e-7 and1.252e-7; no learnedweights/images/finalfits. Final1654 component/generation stage still needs implementation/source-data-resource review; DRAFT not admission. Do not restart terminal jobs or fake oldofficial exposure flags. Follow{new}; OLDonly/NEWreserved/Oimmutable/single1729/goalactiveunachieved.\n"
agents.write_text(text,encoding='utf8')
for name in ('research_resource_allocation_latest_20261029.json','research_priority_single_seed_latest_20261030.json'):
    path=root/'configs'/name;v=read(path);v['snapshot_UTC']=h['snapshot_UTC'];v['latest_handoff']=v['latest_authoritative_handoff']=new;v['goal_status']='active'
    v['current_full1654_stage']=h['full1654_input_stage'];v['latest_execution_observation']=h['actual_F_geometry_profile']
    if 'next_priority' in v:v['next_priority']=next_actions[1]
    path.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
note='> **完整训练输入与工程检查已实际闭合：2026-10-31 20:56 America/Los_Angeles**：全1654概念、10被试、20个新原生概念折外模型各25轮/225更新，合计4500更新；165400条原始条件及20套真实LN affine组装完成。全部留出CPU前向、500个逐轮Adam/RNG审计条目及实际1.69亿数值拼接核验通过。另两个匹配头各4次临时GPU更新、零头回退与CPU/GPU前向检查通过，临时权重丢弃。原内存保护及后续运行环境/路径失败均保留，未改训练数学、门槛或旧ordinary。**最终全范围组件拟合与2000条图片尚未启动，这不是新图片收益或第二创新。** 旧Dfit/V现属训练，官方200已有曝光，不能恢复独立H。详见[实际交接](audit/GOAL_active_execution_handoff_20261101_v8.json)。\n\n'
for name in ('STATUS.md','BRAINHIVE_CURRENT_METHOD_EXPLAINED_20261020.md'):
    path=root/name;oldtext=path.read_text(encoding='utf8')
    if oldtext.startswith('# '):first,rest=oldtext.split('\n',1);path.write_text(first+'\n\n'+note+rest.lstrip('\n'),encoding='utf8')
    else:path.write_text(note+oldtext,encoding='utf8')
print(json.dumps(dict(handoff=ref(target),actual20fold_bank_complete=True,actual_consumer_pass=True,actual_profile_pass=True,final_fits_started=False,goal_active=True)))
