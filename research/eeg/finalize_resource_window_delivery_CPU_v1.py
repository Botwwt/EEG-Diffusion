"""Reconcile documentation and traceability after the resource stop; no scoring."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def write_csv(path,fields,rows):
    with path.open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    started=time.monotonic();utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    index_path=ROOT/'audit/final_artifact_index_20261020_v1.json'
    assert not index_path.exists()
    stop=load(ROOT/'audit/current_idea_closure_and_unlaunched_decoder_20261020_v1.json')
    assert stop['tighter_remaining_authorized_hours']==0
    closure=load(ROOT/'audit/idea_Frozen_RealPaired_Conditional_full_score_closure_v1.json')
    assert closure['status']=='PASSED_ALL16600_COMPLETED_SCORE_IDENTITY_AND_PAIRED_ARITHMETIC'
    registry=load(ROOT/'audit/current_model_registry_CPU_20261020_v1.json')
    assert sha(registry['inventory'])==registry['inventory_sha256'] and registry['strict_loaded_count']==60
    generator=load(ROOT/'audit/current_generator_parameter_count_CPU_20261020_v1.json')
    assert generator['status']=='EXACT_CLASS_META_PARAMETER_COUNTS_MATCH_SAVED_WEIGHT_HEADERS'
    checks=load(ROOT/'audit/final_delivery_checks_20261020_v1.json')
    assert checks['status']=='PASSED_READ_ONLY_REPRODUCTION_ENTRY_AND_RESOURCE_REFUSAL'
    assert sha(ROOT/'reproduce.py')==checks['source_sha256']
    assert sha(ROOT/'audit/query_manifest_R4.csv')==load(ROOT/'configs/idea_interface_runtime_v1.json')['query_manifest_sha256']
    assert (ROOT/'FINAL_REPORT.md').is_file()
    archive=ROOT/'audit/final_delivery_before_20261020_v1';archive.mkdir()
    for name in ['budget.yaml','results.tsv','research/novelty_comparison.md']:
        p=archive/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    budget=(ROOT/'budget.yaml').read_text()
    changes=dict(status='REGISTERED_COMPUTE_WINDOW_EXPIRED_NO_NEW_GPU_TASK',
        formal_long_training_allowed='false',formal_adapter_training_allowed='false',
        next_gate='No new GPU execution under expired bound; preserve complete evidence and registered recovery entries')
    for key,value in changes.items():
        budget,count=re.subn(r'^'+re.escape(key)+r':.*$',key+': '+value,budget,flags=re.M)
        assert count==1,key
    budget+='\ncurrent_resource_closure:\n'
    budget+=f'  UTC: {utc}\n  charged_GPU_job_hours_including_historical_allowances: {stop["charged_GPU_job_hours_including_historical_allowances"]}\n'
    budget+=f'  ledger_remaining_hours_is_not_current_authorization: {stop["ledger_remaining_hours"]}\n'
    budget+=f'  occupied_deadline_UTC: {stop["conservative_occupied_time_deadline_utc"]}\n'
    budget+='  tighter_remaining_authorized_hours: 0\n  resource_limit_changed: false\n  completed_matching_image_arms: [Frozen, RealPaired, Conditional]\n  queries_per_completed_arm: 16600\n  author_decoder_images: 0\n  scientific_goal_complete: false\n'
    (ROOT/'budget.yaml').write_text(budget)
    novelty=(ROOT/'research/novelty_comparison.md').read_text()
    novelty+='\n\n2026-10-20完整结果更新：Frozen/RealPaired/Conditional各完成16,600张图片及逐query核验；Frozen的主要内容指标最好，当前投影器微调没有图片增量。旧相对误差响应信号不能覆盖q-v2绝对分布门槛失败。作者公开特征解码已准备但未提交，跨架构组件尚未训练；先前“当前先执行”的段落为历史状态。完整证据和未完成项见 FINAL_REPORT.md 与 reports/THREE_ARM_CLOSURE_AND_REPRODUCTION_NEXT_20261020.md。\n'
    (ROOT/'research/novelty_comparison.md').write_text(novelty)

    metrics=['DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED']
    saved=load(ROOT/'audit/idea_completed_three_arm_saved_summaries_20261020_v1.json')
    rows=[];subjects=[]
    for arm in closure['completed_arms']:
        summary=saved['arms'][arm]['metrics']
        assert summary['queries']==16600 and len(summary['per_subject'])==10
        m={k:summary['summary'][k]['mean'] for k in metrics}
        for k in metrics:
            assert math.isclose(sum(x[k] for x in summary['per_subject'].values())/10,m[k],rel_tol=1e-10,abs_tol=1e-12)
        rows.append(dict(scope='MATCHED_PATIENCE50_V_R4',method=arm,status='COMPLETED_FULL_IMAGES_METRICS_IDENTITY',
            subjects=10,R=4,expected_queries=16600,actual_images=16600,gallery_images=1660,
            source=closure['sources'][arm]['summary']['path'],source_sha256=closure['sources'][arm]['summary']['sha256'],
            per_query=closure['sources'][arm]['metrics']['path'],manifest=closure['sources'][arm]['manifest']['path'],
            notes='One seed1729; matched conditions/sampling; development data; no SOTA or independent confirmation',**m))
        for subject,values in summary['per_subject'].items():
            reference=saved['arms']['Frozen']['metrics']['per_subject'][subject]
            subjects.append(dict(arm=arm,subject=subject,queries=1660,
                delta_DINO2AFC_vs_Frozen=values['DINO2AFC']-reference['DINO2AFC'],
                delta_SEED_vs_Frozen=values['SEED']-reference['SEED'],**values))
    for old in csv.DictReader((ROOT/'reports/unified_comparison.csv').open()):
        if old['scope']=='LEGACY_V64_ONLY' or (old['scope']=='CURRENT_FULL_PARTITION' and old['arm']=='REF_REG_correct_R4'):
            row=dict(scope='LEGACY_SUB01_V64' if old['scope']=='LEGACY_V64_ONLY' else 'OLDER_PATIENCE20_V_R4',
                method=old['arm'],status=old['status'],subjects=old['subjects'],R=4,
                expected_queries=old['expected_queries'],actual_images=old['actual_images'],gallery_images=old['gallery_images'],
                source=old['source'],source_sha256=old['source_sha256'],per_query=old['source_per_query_csv'],manifest=old['manifest'],
                notes='Preserved historical/unmatched protocol; not included in new matched deltas')
            row.update({k:old[k] for k in metrics});rows.append(row)
    for method in ['Clean','Iso','Moment']:
        rows.append(dict(scope='MATCHED_PATIENCE50_REQUIRED_NOT_RUN',method=method,status='NOT_RUN_COMPLETE_LIFECYCLE_UNFUNDED',
            subjects=10,R=4,expected_queries=16600,actual_images=0,gallery_images=1660,
            notes='Missing comparison; blank metrics do not mean zero score or scientific rejection'))
    rows.append(dict(scope='EXPOSED_PUBLISHED_FEATURE_DECODER_R80_CONVENTION',method='ATM_published_feature_decoder',
        status='PREPARED_NOT_SUBMITTED_OCCUPIED_WINDOW_EXPIRED',subjects=10,R='80_author_convention_not_raw_replayed',
        expected_queries=2000,actual_images=0,gallery_images=200,
        source=str(ROOT/'configs/ATM_public_decoder_registration_20261020_v1.json'),
        source_sha256=sha(ROOT/'configs/ATM_public_decoder_registration_20261020_v1.json'),
        notes='No image result; no upstream EEG encoder reproduction; no direct comparison to R4 V'))
    fields=['scope','method','status','subjects','R','expected_queries','actual_images','gallery_images']+metrics+['source','source_sha256','per_query','manifest','notes']
    write_csv(ROOT/'reports/final_unified_comparison_20261020_v1.csv',fields,rows)
    write_csv(ROOT/'reports/final_per_subject_20261020_v1.csv',
        ['arm','subject','queries']+metrics+['delta_DINO2AFC_vs_Frozen','delta_SEED_vs_Frozen'],subjects)

    table=ROOT/'results.tsv';runs=list(csv.DictReader(table.open(),delimiter='\t'));fields=list(runs[0])
    extras=['training_queries','parameter_count_scope','peak_vram_stage','seconds_per_image_scope','per_query_metrics','image_manifest','evidence_closure']
    fields+= [x for x in extras if x not in fields]
    registry_rows=list(csv.DictReader(Path(registry['inventory']).open()))
    encoders=[dict(subject=x['module'].rsplit('_',1)[-1],checkpoint=x['checkpoint'],sha256=x['sha256']) for x in registry_rows if x['record_scope']=='CURRENT_DIRECT_NATIVE_CONDITION_full']
    assert len(encoders)==10
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    for arm in closure['completed_arms']:
        run=[x for x in runs if x['run_id']=='IDEA1_'+arm+'_full_V_R4_16600'];assert len(run)==1
        row=run[0];cfg=ROOT/'configs'/f'idea_interface_{arm}_resolved_v1.json'
        row.update(git_commit=commit,config_hash=sha(cfg),
            data_manifest_hash=load(ROOT/'configs/idea_interface_runtime_v1.json')['query_manifest_sha256'],
            encoder_ckpt=json.dumps(encoders,separators=(',',':')),prior_ckpt='NONE_DIRECT_ATMS_BRANCH',
            generator_ckpt='/dev/shm/eeg_sota_autoresearch_20261018_v1/sdxl_base_46216598',
            total_params=generator['complete_generator_parameters_including_text_VAE_IP']+3200413,
            parameter_count_scope='One subject EEG encoder plus complete image generator; projection included once; offline teacher excluded',
            training_queries=0 if arm=='Frozen' else 132300,
            seconds_per_image=stop['per_arm_GPU_job_seconds'][arm]['IDEA1_'+arm+'_generate']/16600,
            seconds_per_image_scope='Charged generation wrapper seconds/16600; includes load and IO; excludes training and evaluation',
            per_query_metrics=closure['sources'][arm]['metrics']['path'],image_manifest=closure['sources'][arm]['manifest']['path'],
            evidence_closure=str(ROOT/'audit/idea_Frozen_RealPaired_Conditional_full_score_closure_v1.json'))
        row['reason']+=' Code commit identifies preserved registered source bytes, not an assertion that launch occurred at that commit. Shared EEG/OOF prerequisites remain separately charged.'
        if arm=='Frozen':row['adapter_ckpt']='/dev/shm/eeg_diffusion_next_20261018_v1/adapter.safetensors'
        else:
            training=load(DATA/'runs'/('IDEA1_interface_'+arm+'_v1/completion.json'))
            row['peak_vram']=training['peak_cuda_allocated'];row['peak_vram_stage']='Adapter training, allocated bytes; not a new generation memory measurement'
    with table.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(runs)

    artifacts=dict(current_status=ROOT/'STATUS.md',current_budget=ROOT/'budget.yaml',
        locked_protocol_equivalent=ROOT/'configs/split_manifest.json',query_manifest=ROOT/'audit/query_manifest_R4.csv',
        data_manifest_equivalent=ROOT/'audit/data_manifest.csv',image_manifest=ROOT/'audit/image_manifest.csv',
        actual_training_config=ROOT/'configs/idea_first_ATMS_training_v1.json',
        actual_adapter_recipe=ROOT/'configs/idea_interface_training_recipe_v1.json',
        actual_image_runtime=ROOT/'configs/idea_interface_runtime_v1.json',
        current_architecture_inventory=Path(registry['inventory']),generator_count=ROOT/'audit/current_generator_parameter_count_CPU_20261020_v1.json',
        literature=ROOT/'research/literature_matrix.csv',skills=ROOT/'research/skills_inventory.md',novelty=ROOT/'research/novelty_comparison.md',
        result_ledger=ROOT/'results.tsv',comparison=ROOT/'reports/final_unified_comparison_20261020_v1.csv',
        per_subject=ROOT/'reports/final_per_subject_20261020_v1.csv',report=ROOT/'FINAL_REPORT.md',
        recovery_entry=ROOT/'reproduce.py',recovery_checks=ROOT/'audit/final_delivery_checks_20261020_v1.json',
        RP_Conditional_gallery=DATA/'galleries/IDEA1_Conditional_RealPaired_V_complete_20261019T231408Z_v1/index.html')
    for key,path in artifacts.items():assert path.is_file(),(key,path)
    requirements=[
        ('single_seed_and_full_current_partition','VERIFIED_CURRENT_THREE_ARMS','One seed1729;10subjects;132300trainingqueries for each adapter;16600images per arm'),
        ('historical_and_current_model_audit','VERIFIED_WITH_RECORDED_SOURCE_LIMITS','84registry rows;60CPU strict loads;full generator exact architecture counts;original event replay unavailable'),
        ('full_author_strong_baseline','INCOMPLETE','ATM released-feature decoder prepared/not submitted;upstream EEG encoder unreproduced;D2FOSA executable not located'),
        ('native_error_absolute_fidelity','FAILED_QV2_PREREQUISITE','Relative response signals retained;absolute classifier/support gate fails'),
        ('complete_required_adaptation_controls','INCOMPLETE','Frozen/RealPaired/Conditional complete;Clean/Iso/Moment missing'),
        ('current_projector_image_increment','REJECTED_ON_COMPLETE_V','Both trained projections worse than Frozen on protected primary/semantic metrics'),
        ('full_correct_shuffle_missing_EEG','INCOMPLETE','Bounded diagnostic evidence only;full per-query intervention arms not run'),
        ('full_unified_gallery','PARTIAL','Complete RP/Conditional gallery;Frozen and full intervention columns missing'),
        ('quality_noninferiority','UNESTABLISHED','No human ratings;low-level/integrity metrics do not prove perceptual equivalence'),
        ('official_R1_R4_R80_benchmark','INCOMPLETE','No complete formal image comparisons;published released features remain unrun'),
        ('final_all_official_train_fit_and_independent_H','INCOMPLETE','Development T/D/V only;official200 exposed;independentH empty'),
        ('controlled_capacity_expansion','NOT_RUN','No evidence establishing capacity as the cause'),
        ('portable_component_independent_increment','UNESTABLISHED','New native-condition residual design untrained;two IP/Turbo parents would not prove all generator families'),
        ('comparable_SOTA','UNESTABLISHED','Required reproducible competitors,controls,quality and confirmation incomplete')]
    index=dict(UTC=utc,status='RESOURCE_WINDOW_REPORT_AND_RECOVERY_ARTIFACTS_COMPLETE_SCIENTIFIC_GOAL_UNMET',
        artifacts={k:dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for k,p in artifacts.items()},
        requirements=[dict(requirement=k,status=s,evidence_scope=e) for k,s,e in requirements],
        protected_protocol_metrics_or_query_seeds_changed=False,experiment_outputs_recomputed=False,
        source_sha256=sha(__file__),GPU_used=False,new_GPU_jobs=0,new_images=0,
        scientific_goal_complete=False,comparable_SOTA_established=False,
        seconds=time.monotonic()-started)
    index_path.write_text(json.dumps(index,indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
        file.write(json.dumps(dict(event_id='final_resource_window_delivery_20261020_v1',GPU_used=False,
            seconds=index['seconds'],receipt=str(index_path),receipt_sha256=sha(index_path)))+'\n')
    with (ROOT/'events.jsonl').open('a') as file:
        file.write(json.dumps(dict(classification='progress',time_utc=utc,event_id='final_delivery_registry_budget_and_recovery',
            artifact_index=str(index_path),artifact_index_sha256=sha(index_path),scientific_goal_complete=False))+'\n')
    print(json.dumps(dict(status=index['status'],artifact_count=len(artifacts),comparison_rows=len(rows),per_subject_rows=len(subjects),seconds=index['seconds'])))


if __name__=='__main__':
    main()
