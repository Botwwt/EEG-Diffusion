"""Index actual checkpoints and append receipt changes without inventing results.

This is a retrospective index. Original frozen configs/logs remain the evidence
for timing and preregistration; writing an index cannot create preregistration.
"""
from pathlib import Path
import csv
import hashlib
import json
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
OLD=ROOT.parent/'eeg_diffusion_next_20261018_v1'


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    records=[]
    for directory,prediction in [('S1_full_atm_native_v1','epsilon'),('S1_prior_velocity_repair_v1','v_prediction')]:
        for path in sorted((DATA/'runs'/directory/'full').glob('sub-*/completion.json')):
            p=json.loads(path.read_text());subject=path.parent.name
            for component in (['encoder','prior'] if prediction=='epsilon' else ['prior']):
                model=p[component];checkpoint=Path(model['checkpoint']);assert checkpoint.exists()
                records.append(dict(model_id=f'{directory}/{subject}/{component}',subject=subject,
                    class_name='models.atms.ATMS()' if component=='encoder' else 'diffusion_prior.DiffusionPriorUNet(cond_dim1024,dropout.1)',
                    is_ATM=component=='encoder',is_shrunken_ATM=False,parameters=model['parameters'],
                    input='63x250 physical mean4 EEG' if component=='encoder' else '1024 native EEG condition +1024 noisy native visual vector +t',
                    output_dim=1024,initialization='scratch_in_this_campaign',frozen_at_inference=True,
                    prediction_type='native_MSE_plus_CLIP' if component=='encoder' else prediction,
                    actual_epochs=model['epochs'],actual_updates=model['updates'],seen_R4_examples=model['seen_samples'],
                    unique_T_stimuli=model['unique_training_rows'],checkpoint=str(checkpoint),checkpoint_sha256=model['checkpoint_sha256'],
                    receipt=str(path),receipt_sha256=sha(path),
                    status=('trained; patience20 REF_REG full V image metrics completed, not a new mechanism result'
                        if component=='encoder' and (DATA/'evaluation/V/REF_REG_correct_R4/metrics/summary.json').exists()
                        else 'trained; full V image evaluation pending' if component=='encoder' or prediction=='v_prediction'
                        else 'trained; failed corrected T image readiness, not full V hypothesis rejection')))
    for resolved_path in sorted((ROOT/'configs').glob('S3_native_oof_*_resolved.json')):
        resolved=json.loads(resolved_path.read_text())
        assert sha(resolved['training_config_path'])==resolved['training_config_sha256']
        config=json.loads(Path(resolved['training_config_path']).read_text())
        for role in ['oof0','oof1']:
            for path in sorted((Path(config['output_root'])/role).glob('sub-*/completion.json')):
                proof=json.loads(path.read_text());subject=path.parent.name
                assert proof['status']=='COMPLETED' and proof['role']==role and proof['subject']==subject
                assert proof['training_seed']==1729 and proof['split_sha256']==config['split_sha256']
                for component in ['encoder','prior']:
                    model=proof[component]
                    if model is None:continue
                    checkpoint=Path(model['checkpoint']);assert sha(checkpoint)==model['checkpoint_sha256']
                    assert model['parameters']==(3200413 if component=='encoder' else 9675648)
                    assert model['seen_samples']==model['epochs']*model['unique_training_rows']
                    if component=='prior':assert model['prediction_type']=='v_prediction' and model['epochs']==400
                    records.append(dict(model_id=f'{resolved["run_id"]}/{role}/{subject}/{component}',subject=subject,
                        class_name='models.atms.ATMS()' if component=='encoder' else 'diffusion_prior.DiffusionPriorUNet(cond_dim1024,dropout.1)',
                        is_ATM=component=='encoder',is_shrunken_ATM=False,parameters=model['parameters'],
                        input='63x250 physical mean4 EEG; concept-held-out role '+role if component=='encoder' else '1024 native EEG condition +1024 noisy native visual vector +t',
                        output_dim=1024,initialization='scratch_in_this_campaign_concept_fold',frozen_at_inference=True,
                        prediction_type='native_MSE_plus_CLIP' if component=='encoder' else 'v_prediction',
                        actual_epochs=model['epochs'],actual_updates=model['updates'],seen_R4_examples=model['seen_samples'],
                        unique_T_stimuli=model['unique_training_rows'],checkpoint=str(checkpoint),checkpoint_sha256=model['checkpoint_sha256'],
                        receipt=str(path),receipt_sha256=sha(path),status='completed OOF teacher; not a formal image-method result'))
    idea_config=ROOT/'configs/idea_first_ATMS_training_v1.json'
    if idea_config.exists():
        config=json.loads(idea_config.read_text())
        assert config['study']=='idea_first_direct_ATMS_v1' and config['encoder_patience']==50
        assert config['condition_chain']=='REF_REG' and not config['prior_enabled']
        for role in ['full','oof0','oof1']:
            for path in sorted((Path(config['output_root'])/role).glob('sub-*/completion.json')):
                proof=json.loads(path.read_text());subject=path.parent.name;model=proof['encoder']
                assert proof['status']=='COMPLETED' and proof['role']==role and proof['subject']==subject
                assert proof['training_seed']==1729 and proof['split_sha256']==config['split_sha256']
                assert proof['prior'] is None and model['parameters']==3200413
                assert model['seen_samples']==model['epochs']*model['unique_training_rows']
                assert sha(model['checkpoint'])==model['checkpoint_sha256']
                assert sha(proof['condition_cache'])==proof['condition_cache_sha256']
                records.append(dict(model_id=f'idea_first_ATMS_patience50_v1/{role}/{subject}/encoder',subject=subject,
                    class_name='models.atms.ATMS()',is_ATM=True,is_shrunken_ATM=False,parameters=3200413,
                    input='63x250 physical mean4 EEG; idea-first '+role,output_dim=1024,
                    initialization='scratch_in_this_campaign',frozen_at_inference=True,
                    prediction_type='native_MSE_plus_CLIP',actual_epochs=model['epochs'],actual_updates=model['updates'],
                    seen_R4_examples=model['seen_samples'],unique_T_stimuli=model['unique_training_rows'],
                    checkpoint=model['checkpoint'],checkpoint_sha256=model['checkpoint_sha256'],
                    receipt=str(path),receipt_sha256=sha(path),
                    status='Completed actual '+role+' encoder with official max100/patience50; not full ATM pipeline reproduction or idea image gain'))
    legacy=json.loads((OLD/'forensics/model_inventory.json').read_text())
    items=[('EEG',legacy['EEG']),('PRIOR',legacy['PRIOR']),('point_mapper',legacy['point_mapper'])]+list(legacy['adapters'].items())
    for name,m in items:
        records.append(dict(model_id='legacy/'+name,subject='sub-01',class_name=m.get('class_name','Controller'),
            is_ATM=False,is_shrunken_ATM=False,parameters=m['parameters'],input='legacy17x250 EEG/joint native bridge; see inventory',
            output_dim=1024 if name in ['EEG','PRIOR','point_mapper'] else '4xHxW epsilon correction',
            initialization='preexisting_task_checkpoint' if name=='EEG' else 'scratch_in_legacy_campaign',frozen_at_inference=True,
            prediction_type=m.get('objective','legacy distillation/energy objective; see inventory'),
            actual_epochs='',actual_updates=m.get('updates',m.get('original_training_updates')),
            seen_R4_examples='',unique_T_stimuli=m.get('original_fit_images',''),checkpoint=m['checkpoint'],checkpoint_sha256=m['sha256'],
            receipt=str(OLD/'forensics/model_inventory.json'),receipt_sha256=sha(OLD/'forensics/model_inventory.json'),
            status='historical evidence retained; unmatched to new all10/full V protocol'))
    projection=ROOT/'audit/native_projection_full_V_diagnosis.json'
    if projection.exists():
        p=json.loads(projection.read_text())
        records.append(dict(model_id='standard_SDXL/frozen_native_ImageProjection',subject='all10',
            class_name='diffusers.models.embeddings.ImageProjection',is_ATM=False,is_shrunken_ATM=False,
            parameters=p['projector_parameters'],input='1024 raw pooled CLIP-H native vector',output_dim='4x2048',
            initialization='existing_public_pretrained_IP_Adapter_weights',frozen_at_inference=True,
            prediction_type='Linear1024_to8192_then_reshape4x2048_and_LayerNorm',actual_epochs='',actual_updates=0,
            seen_R4_examples='',unique_T_stimuli='',checkpoint='/dev/shm/eeg_diffusion_next_20261018_v1/adapter.safetensors',
            checkpoint_sha256=p['adapter_sha256'],receipt=str(projection),receipt_sha256=sha(projection),
            status='8400896-parameter projection subset inside the full IP asset; CPU diagnostic is not exact BF16 replay or image evidence'))
    for path in sorted((DATA/'runs').glob('IDEA1_interface_*_v1/completion.json')):
        proof=json.loads(path.read_text());arm=proof['arm']
        assert proof['status']=='COMPLETED_REGISTERED_MATCHED_FULL_QUERY_PASS'
        assert arm in ['Clean','Iso','Moment','Conditional','RealPaired']
        assert proof['trainable_parameters']==8400896 and proof['subjects']==10 and proof['seed']==1729
        assert proof['seen_training_queries']==132300 and proof['each_subject_query_coverage']==[13230]*10
        assert proof['updates']==8269 and proof['unique_training_images']==13230
        assert proof['recipe_sha256']==sha(ROOT/'configs/idea_interface_training_recipe_v1.json')
        assert sha(proof['checkpoint'])==proof['checkpoint_sha256']
        assert sha(path.parent/'input_provenance.json')==proof['source_input_provenance_sha256']
        metrics=DATA/'evaluation/V'/f'IDEA1_{arm}_correct_R4/metrics/summary.json'
        records.append(dict(model_id=f'IDEA1_interface_{arm}_v1/native_ImageProjection',subject='all10',
            class_name='diffusers.models.embeddings.ImageProjection',is_ATM=False,is_shrunken_ATM=False,
            parameters=8400896,input=f'1024 native condition; {arm} registered R4 training contract',output_dim='4x2048',
            initialization='existing_public_pretrained_native_projection_then_finetune_this_campaign',frozen_at_inference=True,
            prediction_type='Image latent epsilon MSE through frozen SDXL/IP attention; not semantic-vector regression',
            actual_epochs='1 complete query pass;10 unique-image exposures',actual_updates=proof['updates'],
            seen_R4_examples=proof['seen_training_queries'],unique_T_stimuli=13230,
            checkpoint=proof['checkpoint'],checkpoint_sha256=proof['checkpoint_sha256'],
            receipt=str(path),receipt_sha256=sha(path),status=('Trained; full V metrics exist, mechanism gain still requires matched controls'
                if metrics.exists() else 'Trained; full V images/metrics pending; a completed pass does not establish convergence')))
    fields=list(records[0])
    with (ROOT/'model_registry.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(records)
    events=[]
    for path in sorted((ROOT/'audit').glob('*_job.json')):
        data=json.loads(path.read_text())
        events.append(dict(record_type='actual_job_receipt',source=str(path),source_sha256=sha(path),
            status=data.get('status'),name=data.get('name',path.stem),actual_receipt=data,
            scientific_outcome='Not inferred from process exit or training loss. See linked diagnostic/full-image results.'))
    for name in ['batched_native_condition_check.json','epsilon_prior_readiness_failure.json',
                 'prior_sampling_prerequisite_check.json','frozen_encoder_native_diagnosis.json','native_projection_full_V_diagnosis.json',
                 'encoder_stop_rule_audit.json','S3_OOF_implementation_preflight.json',
                 'native_oof_cache_software_checks.json','native_error_channel_software_checks.json',
                 'V_fixed8_partial_visual_QA.json','V_fixed8_oracle_diagnosis.json',
                 'V_fixed8_native_condition_scores.json','V_original_batch_replay_and_isolation.json',
                 'error_channel_classifier_initial_positive_control_failure.json',
                 'error_channel_validation_software_checks.json','native_error_validator_prerequisite_negative_check.json',
                 'native_prior_cache_cost_profile.json','S3_native_prior_cache_budget_implication.json',
                 'V_fixed8_oracle_norm_diagnosis.json','velocity_config_text_consistency.json',
                 'native_oof_stage_software_checks.json','native_oof_preparer_prerequisite_check.json','S3_OOF_budget_gate.json',
                 'compile_runtime_prerequisite_check.json','full_gallery_format_check.json',
                 'static_unet_graph_cost_profile.json','REG_static_unet_graph_cost_slot.json',
                 'native_oof_stage_queue_preflight.json','CPU_postprocessing_streaming_revision_preparation.json',
                 'full_EEG_controls_stage_v2_software_checks.json','final_benchmark_preparation_software_checks.json',
                 'idea_first_remote_preflight_v1.json','idea_first_prerequisite_budget_gate_v1.json',
                 'idea_first_actual_training_verified_v1.json','native_projector_training_cost.json',
                 'idea_first_response_and_batch_preflight_v1.json','idea_first_batch_cost_profile_v1.json',
                 'idea_first_Dcheck_summary_v1.json','idea_interface_software_preflight_v1.json',
                 'idea_interface_campaign_software_preflight_v1.json','idea_interface_campaign_software_preflight_v2.json',
                 'idea_first_new_encoder_diagnosis_compact_v1.json','idea_first_selected_vs_final_logged_diagnosis_v1.json',
                 'idea_first_interface_actual_training_started_v1.json','gallery_browser_QA_v1/report.json']:
        path=ROOT/'audit'/name
        if path.exists():
            data=json.loads(path.read_text());events.append(dict(record_type='diagnostic_receipt',source=str(path),
                source_sha256=sha(path),status=data.get('status'),name=path.stem,
                scientific_outcome='Limited to the diagnostic scope in the source; not a full image comparison or SOTA claim.'))
    for partition in ['V','benchmark']:
        generation_receipts=list((DATA/'images/S1_full_native_batchsafe_v2'/partition).glob('*/completion.json'))
        generation_receipts+=list((DATA/'images/IDEA1_interface_v1'/partition).glob('*/completion.json'))
        for path in sorted(generation_receipts):
            data=json.loads(path.read_text());manifest=path.parent/'manifest.jsonl'
            assert data['status']=='COMPLETED' and sha(manifest)==data['image_manifest_sha256']
            assert data['queries']==(16600 if partition=='V' else 2000)
            events.append(dict(record_type='complete_image_generation',source=str(path),source_sha256=sha(path),
                status=data['status'],name=path.parent.name,partition=partition,queries=data['queries'],
                image_manifest_sha256=data['image_manifest_sha256'],
                scientific_outcome='Complete generated population only; image success is determined by separate full metrics and controls.'))
        for path in sorted((DATA/'evaluation'/partition).glob('*/identity_*/summary.json')):
            data=json.loads(path.read_text());assert data['expected_queries']==(16600 if partition=='V' else 2000)
            events.append(dict(record_type='complete_image_identity_audit',source=str(path),source_sha256=sha(path),
                status=data['status'],name=path.parent.parent.name,partition=partition,
                expected_queries=data['expected_queries'],verified_queries=data['verified_queries'],
                failed_queries=data['failed_queries'],score_rows_checked=data['score_rows_checked'],
                native_EEG_cache_hash_verified_queries=data['native_EEG_cache_hash_verified_queries'],
                scientific_outcome='Release-order identity, actual cached condition ownership and saved-file checks; no acquisition-event replay or image-quality claim.'))
        for path in sorted((DATA/'evaluation'/partition).glob('*/low_level/summary.json')):
            data=json.loads(path.read_text());assert data['expected_images']==(16600 if partition=='V' else 2000)
            events.append(dict(record_type='complete_low_level_image_audit',source=str(path),source_sha256=sha(path),
                status=data['status'],name=path.parent.parent.name,partition=partition,
                expected_images=data['expected_images'],verified_and_decoded=data['verified_and_decoded'],
                missing_count=data['missing_count'],corrupt_or_mismatched_count=data['corrupt_or_mismatched_count'],
                scientific_outcome='Full saved-image audit and supplementary pixel/structure metrics; not object success, primary-metric superiority or human quality ratings.'))
    for path in sorted((ROOT/'audit').glob('S3_OOF_budget_check_*.json')):
        data=json.loads(path.read_text());events.append(dict(record_type='complete_scope_budget_decision',source=str(path),
            source_sha256=sha(path),status=data['status'],name=path.stem,
            funded=data['full_two_fold_training_and_condition_sampling_funded'],
            scientific_outcome='Resource gate only; an unfunded stage is not a rejected EEG hypothesis.'))
    for path in sorted((ROOT/'audit').glob('S3_native_oof_*_state.json')):
        data=json.loads(path.read_text())
        events.append(dict(record_type='actual_OOF_stage_state',source=str(path),source_sha256=sha(path),
            status=data['status'],name=data['run_id'],completed_stages=data['completed_stages'],
            scientific_outcome='Actual execution state, not evidence of an image-method gain.'))
        if data['status']=='COMPLETED_ALL10_OOF_CACHE_AND_CPU_DCHECK':
            summary=Path(data['CPU_validation_summary_path'])
            assert sha(summary)==data['CPU_validation_summary_sha256']
            result=json.loads(summary.read_text())
            assert result['Dcheck_queries']==8300 and result['Dfit_queries']==8200 and result['subjects']==10
            events.append(dict(record_type='complete_CPU_error_channel_validation',source=str(summary),
                source_sha256=sha(summary),status=result['status'],name=data['run_id'],
                Dcheck_queries=8300,Dfit_queries=8200,subjects=10,gate=result['gate'],
                scientific_outcome='Conditional-error fidelity at native/CPU-projector level only; not full UNet response or image gain.'))
    for path in sorted((DATA/'evaluation/V').glob('*/metrics/summary.json')):
        data=json.loads(path.read_text());events.append(dict(record_type='full_image_evaluation',source=str(path),source_sha256=sha(path),
            status=data['status'],name=path.parent.parent.name,queries=data['queries'],gallery=data['gallery_images'],
            summary=data['summary'],scientific_outcome='Full V descriptive result; requires paired comparison and independent controls.'))
    idea_state=ROOT/'audit/idea_first_prerequisite_state_v1.json'
    if idea_state.exists():
        state=json.loads(idea_state.read_text())
        events.append(dict(record_type='actual_idea_first_prerequisite_state',source=str(idea_state),
            source_sha256=sha(idea_state),status=state['status'],name='idea_first_full_and_concept_OOF',
            scientific_outcome='Actual training/cache/validation state; not evidence of a full image mechanism gain.'))
    for folder,name,scope in [('idea_first_ATMS_patience50_v1','idea_first_CPU_Dcheck','Native/projected relative fidelity only'),
                              ('idea_first_full_UNet_response_v1','idea_first_UNet_Dcheck','Frozen denoiser response fidelity only')]:
        path=DATA/'error_validation'/folder/'summary.json'
        if path.exists():
            result=json.loads(path.read_text())
            events.append(dict(record_type='complete_idea_first_fidelity_validation',source=str(path),source_sha256=sha(path),
                status=result['status'],name=name,
                scientific_outcome=scope+'; not actual generated image superiority, distribution equivalence, capacity proof or SOTA.'))
    ledger=ROOT/'experiment_ledger.jsonl'
    # Manual user-priority/progress records use their own typed schema. Keep
    # them unchanged; only automatically indexed receipts carry event_id.
    seen={item['event_id'] for item in map(json.loads,ledger.read_text().splitlines())
          if 'event_id' in item} if ledger.exists() else set()
    added=0
    with ledger.open('a') as file:
        for event in events:
            event_id=hashlib.sha256(json.dumps(event,sort_keys=True).encode()).hexdigest()
            if event_id in seen:continue
            event.update(event_id=event_id,indexed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                declaration_status='Retrospective receipt index; original locked configs/logs determine preregistration, not this entry')
            file.write(json.dumps(event)+'\n');added+=1
    print(json.dumps(dict(models_indexed=len(records),new_ledger_events=added,experiment_ledger=str(ledger),model_registry=str(ROOT/'model_registry.csv'))))


if __name__=='__main__':main()
