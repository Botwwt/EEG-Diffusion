"""Record only a real bounded OLD observation, not intent or stale locks."""
import argparse,hashlib,json
from pathlib import Path


def main(receipt,version):
    root=Path(__file__).resolve().parent
    def read(p):return json.loads(Path(p).read_bytes())
    def ref(p):
        p=Path(p).resolve();raw=p.read_bytes()
        return dict(path=str(p.relative_to(root)).replace('\\','/'),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    r=read(receipt);assert r['exit_code']==0
    d=json.loads(r['stdout']);assert d['launch']['controller_PID']==466643 and d['launch']['start_ticks']==112273987
    assert d['controller']['exact_live'] or any('completion' in key or 'failure' in key for key in d['state'])
    previous=root/f'audit/GOAL_active_execution_handoff_20261101_v{version-1}.json';h=read(previous)
    target=root/f'audit/GOAL_active_execution_handoff_20261101_v{version}.json';assert not target.exists()
    failure=d['state'].get('recovery1_controller_failure.json')
    complete=d['state'].get('recovery1_controller_completion.json')
    h.update(status='ACTIVE_GOAL_FULL1654_OOF_SOURCE_BOUND_CACHE_RECOVERY_PROGRESS' if not complete else 'ACTIVE_GOAL_FULL1654_OOF_INPUT_ASSET_STAGE_COMPLETE_NOT_COMPONENT_SUCCESS',
        snapshot_UTC=d['UTC'],previous_handoff=ref(previous)['path'],previous_goal_turn_classification='no progress: user-facing status answer only; no authoritative state changed',
        current_goal_turn_classification='progress: source-bound remaining17 continuation actually launched and revalidated; real full-fold fits and audits advanced; input consumer implemented and15 full-scope local checks passed',
        goal_status='active',goal_complete=False,confirmed_second_method=False,SOTA=False,CCF_A=False)
    stage=h['full1654_current_stage']
    stage.update(actual_observation=ref(receipt),original_controller_terminal=True,original_failure_preserved=True,recovery_not_yet_started=False,
        recovery_launch=d['launch'],recovery_controller=d['controller'],recovery_active=d['active'],recovery_worker=d['worker'],recovery_state=d['state'],
        completed_fits=d['completed_fits'],completed_independent_audits=d['completed_independent_audits'],trained_updates=225*d['completed_fits'],
        exported_OOF_queries=8270*d['completed_fits'],not_complete165400=complete is None,controller_terminal=not d['controller']['exact_live'],
        failure_phase=failure['phase'] if failure else None,recovery_failure=failure,recovery_completion=complete,
        owned_cache_proof='audit/HIVE_exposed_OOF_assets_preparation_20261101_v2/actual_initial_owned_cache_release.json',
        recovery_source_metadata='audit/HIVE_exposed_OOF_assets_preparation_20261101_v2/actual_recovery1/observation.json',
        input_consumer_source='brainhive_exposed_fullfit_component_inputs_CPU_20261101_v1.py',
        input_consumer15checks='audit/HIVE_exposed_fullfit_component_inputs_preparation_20261101_v1/actual_local_CPU_checks.json',
        input_consumer_actual_fullbank_executed=False,final_component_or_benchmark_admitted=False)
    h['next_full1654_stage_context']['stage']='HISTORICAL_INITIAL_DRAFT_SUPERSEDED_BY_FULL1654_CURRENT_STAGE'
    h['next_safe_actions']=[
        'Follow exact recovery controller466643/start112273987 using bounded OLD reads. Original464771 is terminal; never restart it or completed folds. Observer exhaustion is not task termination.',
        'Recovery uses the same immutable training/auditor/protocol/admission and original wall deadline; only remaining17 are fitted. Each audit is followed by own-task clean-file cache advice. Preserve the original fourth preflight failure and all output bytes. No lower24GiB gate, newseed, loss or epoch changes.',
        'After actual20 fits/20 audits/aggregate completion and exact controller exit, fetch bounded terminal metadata/20 audits/assembly certificate/cache reviews. Then execute and source-review the new CPU-only input consumer against all165400 raw join values and all20 actual affine tensors. Local15checks alone are not real-bank proof or component admission.',
        'Implement a separate transparent full1654 component method/calibration/resource protocol. Old Dfit/Dcheck/V become train, official200 has been exposed, and no independentH can be restored. Do not fake legacy false-exposure flags or reuse old validation as independent.',
        'Prior noO/five-arm joint E/P studies remain closed negative. Originalordinary unchanged; OLD only, NEW reserved, seed1729. Complete positive first-method evidence while evaluating substantial EEG-grounded alternatives; do not rename controls or asset preparation as a second innovation.'
    ]
    target.write_text(json.dumps(h,indent=2)+'\n',encoding='utf8')
    agents=root/'AGENTS.md';text=agents.read_text(encoding='utf8')
    pointer=str(previous.relative_to(root)).replace('\\','/');text=text.replace('Latest handoff: '+pointer,'Latest handoff: '+str(target.relative_to(root)).replace('\\','/'),1)
    note=f"\n- AUTHORITATIVE full1654 recovery at{d['UTC']}: original464771 terminal/failure retained, new466643/start112273987 exactlive={d['controller']['exact_live']}; actual fits{d['completed_fits']}/20, independent audits{d['completed_independent_audits']}/20. Own clean-output cache released without byte/deletion/math changes; same24GiB gate/source/seed1729/originaldeadline. Reuse3 completed folds; remaining17 only. New CPU consumer15local full165400metadata/batch checks PASS, actual fullbank validation still pending. Not secondidea/component/benchmark admission. OLDonly/NEWreserved/Oimmutable/goalACTIVEunachieved. Follow{target.name}.\n"
    agents.write_text(text+note,encoding='utf8')
    for name in ('research_resource_allocation_latest_20261029.json','research_priority_single_seed_latest_20261030.json'):
        p=root/'configs'/name;v=read(p);v['snapshot_UTC']=d['UTC'];v['latest_handoff']=str(target.relative_to(root)).replace('\\','/');v['latest_authoritative_handoff']=v['latest_handoff']
        v['latest_execution_observation']=dict(status=h['status'],observation=ref(receipt),controller=d['controller'],active=d['active'],completed_fits=d['completed_fits'],completed_independent_audits=d['completed_independent_audits'],final_component_or_benchmark_admitted=False)
        v['current_full1654_stage']=stage
        if 'next_priority' in v:v['next_priority']=h['next_safe_actions'][0]
        p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
    status=root/'STATUS.md';old=status.read_text(encoding='utf8')
    intro=f"此条为{d['UTC']}实测：全1654训练专用折外输入队列接续466643在线={d['controller']['exact_live']}，完整模型{d['completed_fits']}/20、独立审计{d['completed_independent_audits']}/20。原内存保护失败保留；仅释放本阶段清洁页缓存后接续，不重训前三折。新加载器15项本地检查通过，真实全银行核验及最终组件拟合仍待完成。此项不是第二创新或新图片收益，目标继续。详见[audit/{target.name}](audit/{target.name})。\n\n"
    status.write_text(intro+old,encoding='utf8')
    print(json.dumps(dict(handoff=ref(target),actual_UTC=d['UTC'],completed_fits=d['completed_fits'],completed_audits=d['completed_independent_audits'],goal_active=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--version',type=int,required=True);a=p.parse_args();main(a.receipt,a.version)
