"""Persist a real exact-handle snapshot, without reading partial effect scores."""
import argparse,hashlib,json
from pathlib import Path

def main(receipt,version):
    root=Path(__file__).resolve().parent
    def read(p):return json.loads(Path(p).read_bytes())
    def ref(p):
        p=Path(p).resolve();raw=p.read_bytes();return dict(path=str(p.relative_to(root)).replace('\\','/'),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    envelope=read(receipt);assert envelope['exit_code']==0;d=json.loads(envelope['stdout'])
    assert (d['launch']['controller_PID'],d['launch']['start_ticks'])==(476654,112533823)
    assert d['controller']['exact_live'] or 'failure.json' in d['state'] or 'completion.json' in d['state']
    previous=root/f'audit/GOAL_active_execution_handoff_20261101_v{version-1}.json';h=read(previous)
    target=root/f'audit/GOAL_active_execution_handoff_20261101_v{version}.json';assert not target.exists()
    out=root/'audit/HIVE_full1654_F_lifecycle_preparation_20261101_v2';snapshot=out/f'actual_observation_v{version}.json';assert not snapshot.exists()
    d.pop('metadata_files',None);snapshot.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
    components={}
    for arm,values in d['arms'].items():
        fit=values.get('fit_completion.json');audit=values.get('fit_CPU_audit.json');profile=values.get('native_profile')
        if fit:
            assert fit['updates']==31014 and fit['exposures']==496200 and fit['study']==d['launch']['study']
            assert audit and audit['component']==fit['component'] and audit['all165400_CPU64_forwards_verified']
        components[arm]=dict(final_component=fit['component'] if fit else None,updates=fit['updates'] if fit else values.get('fit_progress.json',{}).get('updates'),
            exposures=fit['exposures'] if fit else values.get('fit_progress.json',{}).get('exposures'),
            fit_seconds=fit['fit_seconds'] if fit else None,fit_CPU_audit=audit,native_profile=profile,
            generated_query_records=values.get('generated_query_records',0),completed_200_image_segments=values['completed_200_image_segments'])
    images=sum(v['generated_query_records'] for v in components.values())
    h.update(status='ACTIVE_FULL1654_TWO_F_FITS_AUDITED_FIXED4000_R80_IMAGE_LIFECYCLE_RUNNING',snapshot_UTC=d['UTC'],previous_handoff=str(previous.relative_to(root)).replace('\\','/'),
        previous_goal_turn_classification='progress: actual20fold165400 input bank, consumer and temporary engineering completed',
        current_goal_turn_classification='progress: complete exposed fullscope source/data/resource protocol and full lifecycle implemented; actual two31014-update fits and all165400 forwards perhead audited; finalhead native profiles and real research images started',
        goal_status='active',goal_complete=False,active_research_handles=[d['controller']]+([d['worker']] if d['worker'] and d['worker']['exact_live'] else []))
    h['current_full1654_F_component_study']=dict(study=d['launch']['study'],launch=d['launch'],controller=d['controller'],active=d['active'],worker=d['worker'],
        phase=d['state'].get('progress.json',{}).get('phase'),state=d['state'],components=components,research_images_saved_at_observation=images,planned_research_images=4000,
        actual_source_and_registration='audit/HIVE_full1654_F_lifecycle_preparation_20261101_v2/actual',
        local12CPUchecks='audit/HIVE_full1654_F_lifecycle_preparation_20261101_v2/actual_local_CPU_checks.json',
        actual_observation=ref(snapshot),raw_receipt=ref(receipt),source_files_frozen=True,all4000_before_effect_scores=True,new_effect_scores_read=False,
        original_ordinary_epsilon_token_full1654_branch_not_yet_fitted=True,first_zero_update_runtime_failure_preserved='audit/HIVE_full1654_F_lifecycle_preparation_20261101_v1/actual/first_actual_observation.json',
        v2_fix='Only perform CPU-only input verification before CUDA initialization; original v1 has no arm folder/model/update. Same math/data/seed/updates/strength/numerical guard; new job names and directories.',
        ordinary_unchanged=True,OLD_only=True,NEW_contacted=False,independent_H=False,SOTA=False,CCF_A=False)
    h['final_component_or_benchmark_admitted']='ONLY_BOUNDED_F_AND_EUCLIDEAN_F_FULL1654_FIXED_COMPARISON_ADMITTED'
    h['next_safe_actions']=[
        'Follow exact controller476654/start112533823 on OLD with read_HIVE_full1654_F_lifecycle_OLD_CPU_20261101_v1.py and compact request. Revalidate current worker from the same progress/receipt; do not restart models or controller because an observer expires. Do not edit frozen running sources/study/calibration/gates.',
        'Both actual full1654 F heads are now31014 updates/496200 exposures with7 checkpoint Adam/coverage audits and all165400 CPU64 forwards each. Preserve actual final models, source affines, seeds, training outputs and failures. Numeric guard is elementwise atol1e-5 plus rtol5e-6, not a max-absolute-only threshold; Euclidean max absolute2.30e-5 passed relative-aware rule without amendment.',
        'Queue completes both native all2000-condition/10subject pixel profiles, all4000 new single-image R80 outputs, allidentity/noise/pixel/condition audits, frozen common8 plusauthor8 metrics, then fixed bounded-parent/Euclidean-parent/bounded-Euclidean comparisons. No partial method effects or control winner selection. Same cached2000 native reference images and actual published full1654 parent affines.',
        'Implement terminal-only bounded metadata/CSV/manifest/fixed-image collector and independent local all96k paired-difference/1000 subject-concept bootstrap checks before interpreting final results. Fetch only after the same controller is terminal and complete; no source/model/test recalibration.',
        'After this lifecycle closes, continue the still-required originalordinary epsilon+token full1654 branch: separate truthful full16540 VAE latent stage, complete frozen-host profile/trainer/all2000 metrics and source/data/resource admission. The current F-only comparison does not complete or replace that objective.',
        'Use final actual image results and training logs to choose substantive EEG-grounded changes. Prior jointE/P/noO/terminal/AR failures remain stopped. No newseed, no NEW GPU, no oldordinary overwrite, no fake independentH or oldofficial false-exposure flags. Novelty/SOTA/CCF A remain unproved; keep full goal active.'
    ]
    target.write_text(json.dumps(h,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    relative=str(target.relative_to(root)).replace('\\','/');agents=root/'AGENTS.md';text=agents.read_text(encoding='utf8');text=text.replace('Latest handoff: '+str(previous.relative_to(root)).replace('\\','/'),'Latest handoff: '+relative,1)
    lines=text.splitlines()
    for n,line in enumerate(lines):
        if line.startswith('- Authoritative current state:'):
            lines[n]=f"- Authoritative current state at{d['UTC']}: full1654 OOF20model/165400 bank complete; NEW actual two F heads both31014 updates/496200exposures and all165400CPU forwards audited. Finite controller476654/start112533823 exactlive={d['controller']['exact_live']}, phase{h['current_full1654_F_component_study']['phase']}, research records{images}/4000. Complete frozen all16R80 pipeline queued; no new effect scores read. Originalordinary epsilon+token full1654 branch still pending, not replaced by F-only. Prior failed families STOPPED. OLDonly/NEWreserved/Oimmutable/single1729/goalactiveunachieved.";break
    text='\n'.join(lines)+'\n';text+=f"\n- AUTHORITATIVE {d['UTC']}: two actual full1654 F-head fits COMPLETE/AUDITED, source-bound all4000 R80 lifecycle476654/start112533823 LIVE={d['controller']['exact_live']}, records{images}/4000. v1 stopped beforemodel/update by CPU-only loader guard; v2 moves CUDA init after CPU bank verification only, retains originaldata/math/seed/strength/guard. Full7states/head andall165400forwards audited; not allgradient transitions replayed. Native profile perarm all2000conditions/10pixelreplays; current details in{relative}. No partial neweffects, no controlwinner, no SOTA/CCF A. Oldordinary unchanged, epsilon+tokenfullscope remainsfuture. Neverrestart onobserverexpiry.\n"
    agents.write_text(text,encoding='utf8')
    for name in ('research_resource_allocation_latest_20261029.json','research_priority_single_seed_latest_20261030.json'):
        p=root/'configs'/name;v=read(p);v['snapshot_UTC']=d['UTC'];v['latest_handoff']=v['latest_authoritative_handoff']=relative;v['goal_status']='active'
        v['latest_execution_observation']=dict(status=h['status'],observation=ref(snapshot),controller=d['controller'],active=d['active'],worker=d['worker'],research_images_saved=images,new_effect_scores_read=False)
        v['current_full1654_component_study']=h['current_full1654_F_component_study']
        if 'next_priority' in v:v['next_priority']=h['next_safe_actions'][0]
        p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    draft=root/'configs/HIVE_exposed_full1654_single_seed_completion_DRAFT_20261101_v1.json';v=read(draft);v['actual_two_F_head_substage']=dict(study=d['launch']['study'],handoff=relative,scope='Only bounded_F and Euclidean_F; originalordinaryepsilon+token fullscope remains DRAFT',both_actual_fits_complete=True,all4000_images_and_metrics_pending=True);draft.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    note=f"> **全1654 F目标几何比较已真实训练并出图：{d['UTC']}**：两组各完成31014更新/496200次EEG暴露，7个断点的Adam/覆盖及全部165400条件CPU前向审计通过。原生2000条件/十被试像素回放按队列核验；本快照研究图片{images}/4000，控制器476654继续运行，全部新图后统一16指标及固定三比较。首次零更新的CUDA初始化顺序失败保留，v2未改数学、数据、seed、强度或数值门槛。**这是受限F头与同容量对照的全范围比较；原ordinary的ε+token全范围分支尚未拟合，当前没有新图片收益读数。** 单1729、旧卡、原ordinary固定及第二卡预留继续。详见[实际交接]({relative})。\n\n"
    for name in ('STATUS.md','BRAINHIVE_CURRENT_METHOD_EXPLAINED_20261020.md'):
        p=root/name;old=p.read_text(encoding='utf8')
        if old.startswith('# '):first,rest=old.split('\n',1);p.write_text(first+'\n\n'+note+rest.lstrip('\n'),encoding='utf8')
        else:p.write_text(note+old,encoding='utf8')
    print(json.dumps(dict(handoff=ref(target),actual_UTC=d['UTC'],both_fits_audited=True,images_saved=images,controller_live=d['controller']['exact_live'],goal_active=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--version',type=int,required=True);a=p.parse_args();main(a.receipt,a.version)
