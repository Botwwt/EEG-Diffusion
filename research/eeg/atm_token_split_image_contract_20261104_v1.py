"""Fixed full400 native IP two-EEG mean/center intervention and fusion controls."""
import json, sys
from pathlib import Path
import numpy as np

ARMS = ('direct', 'nce_continued', 'conditional', 'restore_old_mean',
        'old_mean_new_center', 'new_mean_old_center', 'token_ensemble', 'wrong_old_mean')
BASES, MODELS = ARMS[:4], ARMS[4:]
IDENTITY = ('query_id', 'subject_id', 'concept_id', 'stimulus_id', 'array_row', 'partition', 'image_seed')
METRICS = ('target_cosine', 'DINO2AFC', 'gallery_top1', 'within_concept10_top1',
           'within_concept10_two_afc', 'PixCorr_RGB', 'SSIM_Y', 'mean_RGB_absolute_error')
CONTRASTS = (('old_mean_new_center', 'nce_continued'), ('old_mean_new_center', 'restore_old_mean'),
             ('old_mean_new_center', 'token_ensemble'), ('old_mean_new_center', 'new_mean_old_center'),
             ('old_mean_new_center', 'wrong_old_mean'), ('old_mean_new_center', 'direct'),
             ('token_ensemble', 'nce_continued'), ('new_mean_old_center', 'nce_continued'))
_context = None

def setup(path):
    global _context
    q = json.loads(Path(path).read_bytes())
    assert q['status'] == 'LOCKED_NATIVE_IP_TOKEN_SPLIT_FIXED_FULL400_IMAGE_STUDY'
    assert q['host'] == 'old' and q['seed'] == 1729 and q['arms'] == list(ARMS)
    assert q['new_training_updates'] == 0 and q['new_images'] == q['reused_images'] == 1600
    assert q['maximum_output_bytes'] == 1024 << 20 and q['contrasts'] == [list(v) for v in CONTRASTS]
    if _context is None:
        sys.path.insert(0, str(Path(q['parent_policy']['path']).parent))
        import atm_mean_image_contract_20261104_v1 as P
        parent = P.setup(q['parent_policy']['path']); K = parent['K']
        K.checked(q['parent_policy']); K.checked(q['profile_policy']); K.checked(q['profile_completion'])
        done = K.read(K.checked(q['parent_completion']))
        assert done['status'] == 'COMPLETE_SIX_ARM_T_MEAN_SHIFT_FULL400_IMAGE_DIAGNOSIS'
        profile = K.read(q['profile_completion']['path'])
        assert profile['status'] == 'PASS_NATIVE_IP_TOKEN_SPLIT_ENGINEERING_ONLY' and profile['new_optimizer_updates'] == 0
        for code_path in (Path(q['parent_policy']['path']).parent, Path(q['profile_policy']['path']).parent):
            launch = K.read(code_path / 'actual_launch.json')
            assert not K.process(launch['PID'], launch['start_ticks'])['live']
        for pin in [*q['sources'].values(), q['split_math_source'], q['baseline_features'], q['baseline_metrics'], q['source_metrics'], q['source_low_level']]:
            K.checked(pin)
        for e in done['events']:
            if e['GPU']:
                job = K.read(K.checked(e['receipt']))
                assert job['status'] == 'COMPLETED' and job['exit_code'] == 0
                assert not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
        old_records = P.records(parent, 400)
        _context = dict(parent, P=P, parent_ctx=parent, q=q, K=K, out=Path(q['output_root']),
                        baseline={arm: old_records[arm] for arm in BASES}, parent_done=done,
                        profile_q=K.read(q['profile_policy']['path']))
    assert _context['q'] == q
    return _context

def resources(c, starting=False):
    parent = c['P'].resources(c['parent_ctx'], starting=starting)
    allocated = c['up']['cv']['ctx']['base']['ctx']['C'].allocated
    profile_left = c['profile_q']['maximum_output_bytes'] - allocated(c['profile_q']['project_root'])
    assert profile_left >= 0
    own = allocated(c['q']['project_root']) if Path(c['q']['project_root']).exists() else 0
    assert 0 <= own <= c['q']['maximum_output_bytes']
    prior = parent['required_including_this_project'] + profile_left
    required = prior + c['q']['maximum_output_bytes'] - own
    assert parent['free_bytes'] >= required
    return dict(required_remaining_bytes=prior, required_including_this_project=required,
                free_bytes=parent['free_bytes'], own_used_bytes=own,
                own_remaining_bytes=c['q']['maximum_output_bytes'] - own,
                inherited=parent, separate_profile128MiB_remaining=profile_left,
                all_prior_forecasts_retained=True)

def donor_indices(queries):
    groups = {}
    for i, row in enumerate(queries):
        groups.setdefault(row['concept_id'], []).append(i)
    assert len(groups) == 40 and all(len(v) == 10 for v in groups.values())
    mapping = np.empty(400, np.int64)
    for ids in groups.values():
        for i, j in zip(ids, ids[1:] + ids[:1]):
            mapping[i] = j
    assert np.all(mapping != np.arange(400)) and len(set(mapping.tolist())) == 400
    return mapping

def prepare(c, path):
    K, out = c['K'], c['out']
    assert not (out / 'prepared.json').exists()
    values = np.load(K.checked(c['parent_ctx']['mean_done']['conditions']), allow_pickle=False)
    assert values.shape == (400, 6, 1024) and values.dtype == np.float32 and np.isfinite(values).all()
    raw = np.ascontiguousarray(values[:, :2])
    K.save_array(out / 'old_new_conditions400.npy', raw)
    donor = donor_indices(c['queries'])
    K.save_array(out / 'donor_indices400.npy', donor)
    result = dict(status='COMPLETE_NATIVE_IP_SPLIT_FIXED400_EEG_INPUTS', policy=K.ref(path),
                  conditions=K.ref(out / 'old_new_conditions400.npy'), donors=K.ref(out / 'donor_indices400.npy'),
                  baseline_conditions=c['parent_ctx']['mean_done']['conditions'],
                  queries=K.ref(c['up']['out'] / 'queries400.json'),
                  target_metadata_ref=c['oldprep']['target_metadata_ref'], target_feature_ref=c['oldprep']['target_feature_ref'],
                  deployable_arms_only_current_query_EEG=list(MODELS[:3]),
                  wrong_mean_donor_is_privileged_diagnostic_control_not_deployable=True,
                  old_ordinary_unchanged=True, new_training_updates=0,
                  normal_split_EEG_forwards=2, normal_split_extra_P_forwards=2, UNet_NFE=4)
    K.write(out / 'prepared.json', result)

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
    return dict(action='COMPLETE_ALL_FIXED400_NATIVE_IP_TOKEN_SPLIT_CONTROLS',
                report_all_semantic_identity_structure_and_color_tradeoffs=True,
                no_early_single_metric_rejection=True, no_posthoc_best_branch_selection=True,
                no_equivalence_from_crossing_CI=True, no_automatic_new_seed_or_fit=True,
                no_pure_style_mean_or_semantic_center_claim=True,
                known_affine_projection_and_softmax_shift_identity_NOT_original=True,
                second_method_or_SOTA=False)
