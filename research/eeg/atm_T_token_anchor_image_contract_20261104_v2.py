"""Fixed T-token calibration controls; one current EEG encoder, all400 before scores."""
import json
import sys
from pathlib import Path
import numpy as np

BASES = ('direct', 'nce_continued', 'conditional', 'restore_old_mean',
         'old_mean_new_center', 'new_mean_old_center', 'token_ensemble', 'wrong_old_mean')
MODELS = ('fixed_old_T_token_mean', 'fixed_new_T_token_mean', 'fixed_visual_T_token_mean')
ANCHORS = ('old_EEG_T_mean', 'new_EEG_T_mean', 'visual_T_mean')
ARMS = BASES + MODELS
IDENTITY = ('query_id', 'subject_id', 'concept_id', 'stimulus_id', 'array_row', 'partition', 'image_seed')
METRICS = ('target_cosine', 'DINO2AFC', 'gallery_top1', 'within_concept10_top1',
           'within_concept10_two_afc', 'PixCorr_RGB', 'SSIM_Y', 'mean_RGB_absolute_error')
CONTRASTS = tuple((MODELS[0], b) for b in ('nce_continued', 'old_mean_new_center', 'token_ensemble',
                  'restore_old_mean', 'direct', 'conditional', 'new_mean_old_center')) + (
    (MODELS[1], 'nce_continued'), (MODELS[1], 'old_mean_new_center'),
    (MODELS[2], 'nce_continued'), (MODELS[2], 'old_mean_new_center'),
    (MODELS[0], MODELS[1]), (MODELS[0], MODELS[2]))
_context = None

def setup(path):
    global _context
    q = json.loads(Path(path).read_bytes())
    assert q['status'] == 'LOCKED_FULL400_T_ONLY_NATIVE_TOKEN_ANCHOR_IMAGE_CONTROLS'
    assert q['host'] == 'old' and q['seed'] == 1729 and q['arms'] == list(ARMS)
    assert q['anchor_names'] == list(ANCHORS) and q['contrasts'] == [list(v) for v in CONTRASTS]
    assert q['new_images'] == 1200 and q['reused_images'] == 3200 and q['new_training_updates'] == 0
    assert q['maximum_output_bytes'] == 512 << 20 and q['maximum_queue_seconds'] == 10800
    if _context is None:
        sys.path.insert(0, str(Path(q['parent_policy']['path']).parent))
        import atm_token_split_image_contract_20261104_v1 as P
        up = P.setup(q['parent_policy']['path']); K = up['K']
        for key in ('parent_policy', 'parent_completion', 'anchor_policy', 'anchor_completion',
                    'baseline_features', 'baseline_metrics', 'baseline_ordered_records', 'fixed_anchor_tokens',
                    'anchor_statistics', 'native_P400', 'old_new_conditions400', 'source_metrics', 'source_low_level'):
            K.checked(q[key])
        for pin in q['sources'].values(): K.checked(pin)
        done = K.read(q['parent_completion']['path'])
        assert done['status'] == 'COMPLETE_EIGHT_ARM_NATIVE_IP_SPLIT_FULL400_IMAGE_STUDY'
        anchor_q = K.read(q['anchor_policy']['path']); anchor = K.read(q['anchor_completion']['path'])
        assert anchor['status'] == 'COMPLETE_T_ONLY_NATIVE_TOKEN_ANCHORS_NO_FIT_OR_IMAGES'
        assert anchor['policy'] == q['anchor_policy'] and anchor['means_fitted_T_only']
        assert anchor['Dcheck_targets_features_and_scores_NOT_used_for_moment_fit']
        assert anchor['statistics'] == q['anchor_statistics'] and anchor['fixed_future_tokens'] == q['fixed_anchor_tokens']
        for policy in (q['parent_policy'], q['anchor_policy']):
            launch = K.read(Path(policy['path']).parent / 'actual_launch.json')
            assert not K.process(launch['PID'], launch['start_ticks'])['live']
            if 'job_receipt' in launch:
                job = K.read(launch['job_receipt'])
                assert job['status'] == 'COMPLETED' and job['exit_code'] == 0
                assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
        for event in done['events']:
            if event['GPU']:
                job = K.read(K.checked(event['receipt']))
                assert job['status'] == 'COMPLETED' and job['exit_code'] == 0
                assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
        baseline = P.records(up, 400)
        assert K.read(q['baseline_ordered_records']['path']) == [baseline[a][i] for i in range(400) for a in BASES]
        _context = dict(up, P=P, parent_ctx=up, q=q, K=K, out=Path(q['output_root']),
                        baseline=baseline, anchor_q=anchor_q, anchor_done=anchor)
    assert _context['q'] == q
    return _context

def resources(c, starting=False):
    parent = c['P'].resources(c['parent_ctx'], starting=starting)
    allocated = c['up']['cv']['ctx']['base']['ctx']['C'].allocated
    anchor_left = c['anchor_q']['maximum_output_bytes'] - allocated(c['anchor_q']['project_root'])
    own = allocated(c['q']['project_root']) if Path(c['q']['project_root']).exists() else 0
    assert anchor_left >= 0 and 0 <= own <= c['q']['maximum_output_bytes']
    prior = parent['required_including_this_project'] + anchor_left
    required = prior + c['q']['maximum_output_bytes'] - own
    assert parent['free_bytes'] >= required
    return dict(required_remaining_bytes=prior, required_including_this_project=required,
                free_bytes=parent['free_bytes'], own_used_bytes=own,
                own_remaining_bytes=c['q']['maximum_output_bytes'] - own,
                separate_T_anchor128MiB_remaining=anchor_left, inherited=parent,
                all_prior_forecasts_retained=True)

def fixed_tokens(native, means):
    assert native.shape == (400, 4, 2048) and native.dtype == np.float16
    assert means.shape == (3, 2048) and np.isfinite(native).all() and np.isfinite(means).all()
    value = native.astype(np.float32)
    return np.stack([(value - value.mean(1, keepdims=True) + mu.astype(np.float32)[None, None]).astype(np.float16)
                     for mu in means], axis=1)

def prepare(c, path):
    K, out, q = c['K'], c['out'], c['q']
    assert not (out / 'prepared.json').exists()
    values = np.load(K.checked(q['old_new_conditions400']), allow_pickle=False)
    native = np.load(K.checked(q['native_P400']), allow_pickle=False)
    stats = np.load(K.checked(q['anchor_statistics']), allow_pickle=False)
    tokens = np.load(K.checked(q['fixed_anchor_tokens']), allow_pickle=False)
    assert values.shape == (400, 2, 1024) and values.dtype == np.float32
    assert native.shape == (400, 2, 4, 2048) and native.dtype == np.float16
    assert tokens.shape == (400, 3, 4, 2048) and tokens.dtype == np.float16
    assert np.array_equal(fixed_tokens(native[:, 1], stats['token_means']), tokens)
    K.save_array(out / 'current_EEG_conditions400.npy', np.ascontiguousarray(values[:, 1]))
    K.write(out / 'prepared.json', dict(status='COMPLETE_T_ONLY_FIXED_TOKEN_ANCHOR_IMAGE_INPUTS',
        policy=K.ref(path), conditions=K.ref(out / 'current_EEG_conditions400.npy'),
        baseline_conditions=c['parent_ctx']['parent_ctx']['mean_done']['conditions'],
        native=q['native_P400'], anchors=q['fixed_anchor_tokens'], statistics=q['anchor_statistics'],
        queries=K.ref(c['up']['out'] / 'queries400.json'),
        target_metadata_ref=c['oldprep']['target_metadata_ref'], target_feature_ref=c['oldprep']['target_feature_ref'],
        all1200_tokens_rebuilt_BIT_EXACT=True, all_means_T_only=True,
        normal_current_EEG_encoder_forwards=1, extra_native_P_forwards=0, UNet_NFE=4,
        query_image_or_concept_used_at_inference=False, original_ordinary_unchanged=True,
        new_training_updates=0))

def records(c, n):
    assert n == 400
    result = dict(c['baseline'])
    for arm in MODELS:
        journal = c['out'] / 'images' / arm / 'images.jsonl'
        result[arm] = [json.loads(s) for s in journal.read_text().splitlines()]
        assert len(result[arm]) == 400
    return result

def paired(arrays, queries):
    concepts = sorted({r['concept_id'] for r in queries})
    groups = [[i for i, r in enumerate(queries) if r['concept_id'] == g] for g in concepts]
    assert len(groups) == 40 and all(len(v) == 10 for v in groups)
    draw = np.random.default_rng(1729).integers(0, 40, (1000, 40))
    result = {}
    for a, b in CONTRASTS:
        result[a + '-' + b] = {}
        for key in METRICS:
            d = arrays[a][key] - arrays[b][key]
            means = np.array([d[g].mean() for g in groups])
            result[a + '-' + b][key] = dict(difference=float(d.mean()),
                descriptive_concept_bootstrap95=np.quantile(means[draw].mean(1), [.025, .975]).tolist(),
                positive_concepts=int((means > 0).sum()), concepts=40)
    return result

def interpret(pairs, n):
    assert n == 400
    return dict(action='COMPLETE_ALL_FIXED_T_TOKEN_ANCHOR_CONTROLS',
                report_all13_comparisons8_metrics=True, no_single_metric_early_rejection=True,
                no_best_anchor_selection=True, no_equivalence_or_noninferiority_from_crossing_CI=True,
                no_automatic_followup_fit_or_seed=True, fixed_T_calibration_NOT_claimed_novel=True,
                wrong_mean_baseline_is_privileged_control_not_deployable=True,
                second_independent_method_or_SOTA=False)
