"""Resolve audited source/inputs and reserve the confirmed 40-hour envelope."""
from pathlib import Path
import hashlib
import json
import math
import time

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/atm_official_20261016')
FEATURES=Path('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2/train_image_embeds.pt')


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    data=json.loads((ROOT/'audit/data_audit.json').read_text())
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    sampler=json.loads((ROOT/'audit/budget_sampler_profile.json').read_text())
    metric=json.loads((ROOT/'audit/semantic_evaluation_profile.json').read_text())
    assert data['complete_for_formal_training'] and len(protocol['subjects'])==10
    assert sampler['status']=='PASSED' and metric['status']=='COMPLETED'
    assert sampler['steps']==20 and sampler['resolution']==512
    assert 'user_confirmed_total_GPU_hour_limit: 40' in (ROOT/'budget.yaml').read_text()
    # Conservative envelope includes saving/DINO margin above the measured GPU calls.
    generation=max(1.,sampler['measured_generation_seconds_per_image'])
    semantics=max(.5,sum(x['seconds_per_image'] for x in metric['cases'].values()))
    v=len(protocol['image_rows']['V'])*10;assert v==16600
    final_queries=12000  # R4 two references + winner correct/permuted/missing; separate winner R1/R80.
    # Correct winner R4 is reused from the two references, so this is a conservative upper bound.
    gallery=1860
    allocations=dict(historical_recorded_compute=1.4351579818408937,
        S0_compute_conservative_allowance=.25,full_reference_training_max=2.,
        two_OOF_roles_training_max=1.5,
        two_reference_full_V_generation_and_semantics=2*v*(generation+semantics)/3600,
        selected_reference_two_EEG_controls_full_V_generation_and_semantics=2*v*(generation+semantics)/3600,
        locked_benchmark_all10_R4_and_separate_R1_R80_reserved=final_queries*(generation+semantics)/3600,
        shared_real_gallery_semantics=gallery*semantics/3600)
    total=sum(allocations.values());assert total<40
    budget=dict(status='LOCKED_USER_CONFIRMED_40_HOURS_S1_AND_FINAL_RESERVED',total_limit_GPU_hours=40,
        accounting='Sum of GPU job wall time, conservatively counting loading/I/O within GPU jobs; provider rental billing not independently verified',
        allocations_GPU_hours=allocations,unallocated_margin_GPU_hours=40-total,
        generation_seconds_per_query_reserved=generation,semantic_seconds_per_image_reserved=semantics,
        full_V_queries_per_method=v,final_query_upper_bound=final_queries,
        generation_settings=dict(model='standard SDXL base1.0',resolution=512,steps=20,CFG=5.,IP_scale=1.,batch=4),
        canonical_image_encoding=dict(format='JPEG',quality=90,subsampling=0,all_methods_identical=True),
        mechanism_adapter_training_and_full_image_evaluation_reserved=False,
        mechanism_gate='Do not launch S4 adapter arms unless measured remaining budget covers every required matched arm plus reserved final/controls. Record NOT_RUN_BUDGET if it does not.',
        hypotheses_not_falsified_by_budget=True,resource_purchase=False,local_large_artifacts=False,
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    cfg=dict(version='full_atms_native_reference_v1',seed=1729,subjects=protocol['subjects'],
        official_source_commit='efb6b1d5d7dd14bea44ee5a952b72a951fa6da54',
        official_atms_source_sha256=sha(SOURCE/'models/atms.py'),
        official_prior_source_sha256=sha(SOURCE/'Generation/diffusion_prior.py'),
        trainer_source_sha256=sha(ROOT/'train_full_reference.py'),split_sha256=sha(ROOT/'configs/split_manifest.json'),
        visual_features_path=str(FEATURES),visual_features_sha256=sha(FEATURES),
        output_root='/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runs/S1_full_atm_native_v1',
        encoder_class='models.atms.ATMS',encoder_kwargs='official ATMS() default; separate model per subject; original raw subject IDs',
        encoder_epochs_max=100,encoder_patience=20,encoder_batch=64,encoder_learning_rate=.0003,
        prior_class='diffusion_prior.DiffusionPriorUNet',prior_cond_dim=1024,prior_dropout=.1,
        prior_epochs_max=400,prior_batch=1024,prior_learning_rate=.001,
        prior_prediction_type='epsilon',prior_schedule='1000 DDPM linear beta1e-4 to.02',
        prior_sample_clipping=False,prior_sampling_steps=50,prior_CFG=5.,
        role_max_wall_seconds=dict(full=7200,oof0=2700,oof1=2700),
        formal_training_authorized_within_user_cap=True,
        explicit_deviations_from_official_develop_entrypoint=[
            'Use predeclared concept-disjoint T/D/V, with all10 subjects and physical mean4 training.',
            'Prior trains only T; validation never merged into prior train loader.',
            'Keep final batch to cover every row; repair double-stride retrieval diagnostic.',
            'Encoder checkpoint uses fixed V combined generation loss alpha.99; training alpha.90 as author source.',
            'Prior checkpoint uses fixed-noise full V epsilon validation; run full400 epochs, no train-loss selection.',
            'Optimizer step before LR scheduler step; no deprecated first-step ordering.',
            'Disable DDPM x0 clipping because unbounded native CLIP targets exceed[-1,1]; exact-epsilon contract test proves distortion.',
            'Single seed1729, one image/query; standard SDXL512/20step CFG5 cost track, not exact paper full multi-pipeline reproduction.'],
        formal_official_test_access_during_training=False)
    for path,value in [(ROOT/'configs/full_reference_resolved.json',cfg),(ROOT/'audit/locked_budget_allocation.json',budget)]:
        assert not path.exists(),'Do not silently replace a frozen resolved config'
        path.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='LOCKED',reserved_GPU_hours=total,margin_GPU_hours=40-total,config=str(ROOT/'configs/full_reference_resolved.json'))))


if __name__=='__main__':main()
