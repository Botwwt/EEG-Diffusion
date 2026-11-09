"""Receive bounded T-prefix numbers and independently audit fixed feature samples."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.receipt.read_bytes())
    assert record['exit_code'] == 0
    result = json.loads(record['stdout'])
    assert result['status'] == 'COMPLETE_MATCHED_IMMUTABLE_T_PREFIX_TWO_POLICY_LOG_ANALYSIS'
    assert result['prefix_updates'] == 2816
    assert result['new_GPU_jobs'] == result['new_neural_updates'] == result['new_images'] == 0
    assert result['active_sources_seed_losses_gates_budgets_changed'] is False
    blob = result.pop('numeric_archive')
    raw = base64.b64decode(blob.pop('base64'), validate=True)
    assert len(raw) == blob['bytes'] < 4 * 1024**2
    assert hashlib.sha256(raw).hexdigest() == blob['sha256']
    data = np.load(io.BytesIO(raw), allow_pickle=False)
    expected = {'joint_table', 'EEG_only_table', 'updates', 'pair_indices',
                'concept_ids', 'sample_indices', 'joint_sample_features',
                'EEG_only_sample_features', 'sample_target_features'}
    assert set(data.files) == expected
    columns = result['columns']
    assert data['joint_table'].shape == data['EEG_only_table'].shape == (2816, 9)
    assert np.array_equal(data['updates'], np.arange(1, 2817))
    assert np.array_equal(data['sample_indices'], np.r_[0:8, 2808:2816])
    root = Path(__file__).resolve().parent
    order_path = root / 'audit/HIVE_paired_semantic_full_study_registration_20261030_v1/fixed_training_order.npy'
    pair_path = root / 'audit/HIVE_paired_semantic_fullT_inputs_complete_20261030_v1/code/fixed_T_pairs.json'
    order = np.load(order_path, allow_pickle=False)
    pairs = json.loads(pair_path.read_bytes())
    assert np.array_equal(data['pair_indices'], order[:2816])
    concepts = np.asarray([pairs[int(i)]['concept_id'] for i in order[:2816]])
    assert np.array_equal(concepts, data['concept_ids'])
    assert len(set(concepts)) == result['represented_T_concepts'] == 1249
    errors = []
    sample_errors = {}
    for name, table_key, feature_key, window_key in [
        ('joint_correct', 'joint_table', 'joint_sample_features', 'joint'),
        ('EEG_only_correct', 'EEG_only_table', 'EEG_only_sample_features', 'EEG_only')]:
        table = data[table_key]
        assert np.isfinite(table).all()
        for j, column in enumerate(columns):
            errors.append(abs(float(table[:, j].mean()) - result['summary'][name]['means'][column]))
        flags = result['summary'][name]
        errors.append(abs(float(np.mean(table[:, 3] > 0)) - flags['positive_training_assignment_margin_fraction']))
        assert flags['EEG_total_gradient_clipped_fraction'] == float(np.mean(table[:, -2] > 1))
        assert flags['P_total_gradient_clipped_fraction'] == float(np.mean(table[:, -1] > 1))
        for k, window in enumerate(result['windows']):
            assert window['start_update'] == 128*k+1 and window['end_update'] == 128*(k+1)
            for j, column in enumerate(columns):
                errors.append(abs(float(table[k*128:(k+1)*128,j].mean()) - window[window_key][column]))
        # Independent 2x2 assignment matrix, rather than importing producer formulas.
        g = np.array(data[feature_key], dtype=np.float64)
        v = np.array(data['sample_target_features'], dtype=np.float64)
        g /= np.linalg.norm(g, axis=2, keepdims=True)
        v /= np.linalg.norm(v, axis=2, keepdims=True)
        similarity = g @ np.swapaxes(v, 1, 2)
        margin = similarity[:,0,0] + similarity[:,1,1] - similarity[:,0,1] - similarity[:,1,0]
        absolute = 1 - .5 * (similarity[:,0,0]+similarity[:,1,1])
        pair_loss = .07 * np.logaddexp(0, -margin/.07)
        dg, dv = g[:,0]-g[:,1], v[:,0]-v[:,1]
        norms = np.linalg.norm(dg,axis=1), np.linalg.norm(dv,axis=1)
        alignment = np.sum(dg*dv,axis=1)/(norms[0]*norms[1])
        independent = np.column_stack([absolute+pair_loss, absolute, pair_loss, margin,
                                       norms[0],alignment,norms[1]])
        err = float(np.abs(independent-table[data['sample_indices'],:7]).max())
        sample_errors[name] = err
        assert err < 2e-12
    delta = data['joint_table'] - data['EEG_only_table']
    for j, column in enumerate(columns):
        claimed = result['summary']['joint_minus_EEG_only'][column]
        errors.extend([abs(float(delta[:,j].mean())-claimed['mean']),
                       abs(float(np.median(delta[:,j]))-claimed['median']),
                       float(np.abs(np.quantile(delta[:,j],[.05,.95])-claimed['quantiles05_95']).max())])
    assert max(errors) < 2e-12
    # These are varying-state training observations, deliberately without CIs or winners.
    by_concept = []
    for concept in sorted(set(concepts)):
        mask = concepts == concept
        by_concept.append(dict(concept_id=concept, pairs=int(mask.sum()),
            joint_minus_EEG_only_mean={c:float(delta[mask,j].mean()) for j,c in enumerate(columns)}))
    window_sign_counts = {c:int(sum(w['joint'][c] > w['EEG_only'][c] for w in result['windows']))
                          for c in columns}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'numeric_tables_and_fixed_samples.npz').write_bytes(raw)
    result['numeric_archive'] = blob
    result['receipt'] = dict(path=str(args.receipt.resolve()),
                            sha256=hashlib.sha256(args.receipt.read_bytes()).hexdigest())
    (args.output/'actual_training_prefix_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (args.output/'all1249_concept_training_differences.json').write_text(json.dumps(by_concept,indent=2)+'\n',encoding='utf-8')
    audit = dict(status='PASS_BOUNDED_ARCHIVE_TABLE_ORDER_CONCEPT_AND_FIXED_FEATURE_REPLAY',
        training_updates_per_arm=2816,T_stimulus_exposures_per_arm=5632,concepts=1249,
        all_numeric_rows_aggregated=True,all22_windows_recomputed=True,
        local_feature_pairs_per_arm=16,all2816_raw_features_recomputed_locally=False,
        fixed_sample_2x2_assignment_loss_error=sample_errors,table_summary_error_max=max(errors),
        joint_greater_than_EEG_only_window_counts_out_of22=window_sign_counts,
        no_generalization_CIs_or_control_winner_selected=True,
        interpretation='Joint policy has worse mean training alignment/margin on this matched evolving-state prefix. This does not identify pure P causality or heldout performance, and does not change the sealed five-arm study.',
        original_image_gates_still_required=True)
    (args.output/'independent_local_numeric_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
