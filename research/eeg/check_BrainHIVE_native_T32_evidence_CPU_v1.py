"""Synthetic-only CPU contracts for the independent T32 evidence reader.

No imports of Torch/native runtime, actual diagnostic artifacts, EEG, images,
weights, real scores, SSH or queue execution. Temporary JSON/CSV fixtures only.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import time

import read_BrainHIVE_native_T32_evidence_CPU_v1 as reader


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Fixture:
    """Exact production-sized denominators with small entirely invented values."""
    def __init__(self, root):
        self.root = Path(root)
        self.repo = self.root / 'repo'; self.repo.mkdir()
        self.out = self.root / 'out'; self.out.mkdir()
        self.files = {}
        self.policy = dict(schema='BrainHIVE_native_T32_evidence_policy_v1', execution_mode='SYNTHETIC_FIXTURE_ONLY',
            reader_source_sha256=digest(reader.__file__), repository_root=str(self.repo),
            automatic_admission=False, rules=copy.deepcopy(reader.RULES),
            diagnostic_output_directory=str(self.out),
            expected=dict(queries=32, timesteps=4, rows=128, native_IP_layers=70,
                full_UNet_calls=160, full_UNet_samples=640, conditional_forwards_per_query_time=5))
        for name in ('main_state', 'diagnostic_queue_state', 'diagnostic_job_state'):
            self.policy[name] = str(self.root / (name + '.json'))
        self.base_states = {
            'main_state': dict(status=reader.MAIN_SUCCESS, pid=21000001),
            'diagnostic_queue_state': dict(status=reader.DIAG_SUCCESS, pid=21000002),
            'diagnostic_job_state': dict(status='COMPLETED', pid=21000003, wrapper_pid=21000004, exit_code=0)}
        source_names = ['diagnose_BrainHIVE_native_function_T32_v2.py', 'brainhive_native_models_v1.py',
            'brainhive_component_data_v1.py', 'brainhive_component_generator_v1.py', 'brainhive_response_rank_v1.py',
            'train_BrainHIVE_response_rank_v1.py', 'train_BrainHIVE_bounded_component_v1.py']
        source_sha = {}
        for name in source_names + ['queue_BrainHIVE_native_function_T32_v2.py', 'run_gpu_job_v2.py']:
            path = self.repo / name; path.write_text('# SYNTHETIC metadata fixture, not an executable host\n')
            source_sha[name] = digest(path)
        self.csv_rows = {i: dict(dataset_split='train', array_row=i,
            stimulus_id=f'synthetic_{i:05d}.jpg', concept_id=f'synthetic_concept_{i // 10:04d}') for i in range(13230)}
        allowed = list(self.csv_rows)
        concepts = sorted({r['concept_id'] for r in self.csv_rows.values()})
        protocol = dict(version='SYNTHETIC_NOT_ACTUAL', image_rows=dict(T=allowed, D=[20000], V=[20001]),
            train_concepts=dict(T=concepts, D=['D_SYNTHETIC'], V=['V_SYNTHETIC']),
            oof_fold_by_concept={concept: i % 2 for i, concept in enumerate(concepts)})
        split = self.root / 'split.json'; split.write_bytes(encode(protocol))
        csv_path = self.root / 'image_manifest.csv'
        csv_path.write_text('dataset_split,array_row,stimulus_id,concept_id\n' + '\n'.join(
            ','.join(str(r[k]) for k in ('dataset_split', 'array_row', 'stimulus_id', 'concept_id'))
            for r in self.csv_rows.values()) + '\n')
        oracle = [dict(stimulus_id=self.csv_rows[i]['stimulus_id'], array_row=i) for i in range(32)]
        oracle_path = self.root / 'oracle.json'; oracle_path.write_bytes(encode(oracle))
        sources = []
        for fold in (0, 1):
            role = 'oof' + str(fold)
            ids = [i for i in allowed if protocol['oof_fold_by_concept'][self.csv_rows[i]['concept_id']] == fold]
            sources.append(dict(role=role, condition_cache_path=str(self.root / (role + '_NO_TENSOR.pt')),
                condition_cache_sha256=('a' if fold == 0 else 'b') * 64,
                checkpoint_path=str(self.root / (role + '_NO_WEIGHTS.pt')),
                checkpoint_sha256=('c' if fold == 0 else 'd') * 64,
                actual_native_output_affine_keys=['block.layers.0.norm.weight', 'block.layers.0.norm.bias'],
                groups=dict(component_train=[dict(array_row=i, cache_row=j) for j, i in enumerate(ids)])))
        contract = dict(status='ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT',
            target_cache_path=str(self.root / 'NO_TARGET_TENSOR.pt'), target_cache_sha256='e' * 64,
            subjects=[dict(subject='sub-01', sources=sources)])
        contract_path = self.root / 'contract.json'; contract_path.write_bytes(encode(contract))
        latent_receipt = self.root / 'latent_receipt.json'; latent_receipt.write_bytes(encode({'synthetic': True}))
        verification = self.root / 'native_asset_verification.json'; verification.write_bytes(encode({'synthetic': True}))
        metadata_paths = [split, csv_path, oracle_path, contract_path, latent_receipt, verification]
        input_sha = {str(path): digest(path) for path in metadata_paths}
        reg = dict(diagnostic='native_function_T32_v2', subject='sub-01', queries=32,
            component_training_enabled=False, KAN_enabled=False, optimizer_created=False, generated_images=0,
            output_directory=str(self.out), oracle_inference_inputs=str(oracle_path),
            source_sha256={k: source_sha[k] for k in source_names},
            queue_source_sha256={k: source_sha[k] for k in source_sha if k not in source_names}, input_sha256=input_sha)
        reg_path = self.root / 'registration.json'; reg_path.write_bytes(encode(reg))
        self.policy.update(diagnostic_registration=dict(path=str(reg_path), sha256=digest(reg_path)),
            registered_metadata_input_sha256=input_sha, split_manifest_path=str(split),
            image_manifest_path=str(csv_path), condition_contract_path=str(contract_path))
        manifest = dict(queries=32, execution=dict(admitted=False), rows=[])
        self.fixed = []
        self.negative = reader.fixed_old_negative(allowed, self.csv_rows)
        for index in range(32):
            row = self.csv_rows[index]; role = 'oof' + str(protocol['oof_fold_by_concept'][row['concept_id']])
            source = next(s for s in sources if s['role'] == role)
            entry = dict(query_index=index, oracle_query_index=index, T_position=index,
                array_row=index, stimulus_id=row['stimulus_id'], concept_id=row['concept_id'],
                subject='sub-01', query_id='sub-01/' + row['stimulus_id'] + '/R4', source_role=role,
                cache_row=next(r['cache_row'] for r in source['groups']['component_train'] if r['array_row'] == index),
                condition_cache_path=source['condition_cache_path'], condition_cache_sha256=source['condition_cache_sha256'],
                parent_checkpoint_path=source['checkpoint_path'], parent_checkpoint_sha256=source['checkpoint_sha256'])
            manifest['rows'].append(entry)
            wrong_position = self.negative[index]; wrong = self.csv_rows[wrong_position]
            self.fixed.append(dict(oracle_query_index=index, T_position=index, array_row=index,
                stimulus_id=row['stimulus_id'], concept_id=row['concept_id'], wrong_T_position=wrong_position,
                wrong_array_row=wrong_position, wrong_stimulus_id=wrong['stimulus_id'], wrong_concept_id=wrong['concept_id']))
        manifest_path = self.root / 'fixed_metadata.json'; manifest_path.write_bytes(encode(manifest))
        self.policy['fixed_queries_metadata'] = dict(path=str(manifest_path), sha256=digest(manifest_path))
        self.provenance = dict(registration_sha256=digest(reg_path), source_sha256=reg['source_sha256'], input_sha256=input_sha,
            input_contract_sha256=digest(contract_path), oracle_metadata_sha256=digest(oracle_path),
            training_target_cache_path=contract['target_cache_path'], training_target_cache_sha256=contract['target_cache_sha256'],
            Q_reference=reader.Q_REFERENCE, missing_definition=reader.MISSING, diagnostic_noise_scheduler='DDPMScheduler',
            all_conditions_share_same_real_latent_noise_and_timestep=True, optimizer_created=False,
            all_native_weights_frozen=True, protected_D_V_or_test_tensors_or_scores_loaded=False,
            generated_images=0, parameter_updates=0, native_scheduler_timesteps=[999, 749, 499, 249],
            negative_mapping=dict(negative_position_sha256=hashlib.sha256(struct.pack('<13230q', *self.negative)).hexdigest(),
                seed=1729, different_concept_verified=True, stimulus_or_array_index_used_as_position=False,
                mapping_array_rows_sha256=hashlib.sha256(struct.pack('<13230q', *allowed)).hexdigest(),
                image_manifest_sha256=digest(csv_path), split_manifest_sha256=digest(split)))
        required_rows = {q[k] for q in self.fixed for k in ('array_row', 'wrong_array_row')}
        self.provenance['actual_sub01_OOF_sources'] = [dict(role=s['role'], checkpoint_path=s['checkpoint_path'],
            checkpoint_sha256=s['checkpoint_sha256'], condition_cache_path=s['condition_cache_path'],
            condition_cache_sha256=s['condition_cache_sha256'],
            selected_mapping=[r for r in s['groups']['component_train'] if r['array_row'] in required_rows]) for s in sources]
        self.build()

    def build(self, raw=(.2, .2, .2, .2), permuted=(.8, .8, .8, .8), missing=(.6, .6, .6, .6), near_zero=False, teacher_advantage=.4):
        self.rows = []
        times = self.provenance['native_scheduler_timesteps']
        for q, fixed in enumerate(self.fixed):
            for ti, timestep in enumerate(times):
                negative = math.sqrt(1e-7) if near_zero and ti == 0 else 1.
                separation = negative ** 2
                values = [raw[ti], permuted[ti], 0., negative, missing[ti]]
                metrics = {}
                for arm, value in zip(reader.ARMS, values):
                    value, orthogonal = value if isinstance(value, tuple) else (value, 0.)
                    positive_distance = value ** 2 + orthogonal ** 2
                    negative_distance = (value - negative) ** 2 + orthogonal ** 2
                    margin = (positive_distance - negative_distance) / (separation + 1e-6)
                    metrics[arm] = dict(d_positive=positive_distance, d_negative=negative_distance,
                        teacher_separation=separation, normalized_margin=margin, J=-margin,
                        rank_loss=max(margin, 0.) + math.log1p(math.exp(-abs(margin))),
                        near_zero_teacher_fraction=float(separation <= 1e-6))
                epsilon = dict(zip(reader.ARMS, [.3, .4, .1, .1 + teacher_advantage, .45]))
                key = f"1729|native_function_T32_v1|sub-01|{fixed['stimulus_id']}|R4|DDPM|{timestep}"
                self.rows.append(dict(fixed, subject='sub-01', repeat_protocol='R4', timestep=timestep,
                    native_time_index=ti, noise_seed=int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**63 - 1),
                    epsilon_MSE=epsilon, EEG_epsilon_MSE_minus_correct_F=.2,
                    permuted_EEG_epsilon_MSE_minus_EEG=.1, wrong_F_epsilon_MSE_minus_correct_F=teacher_advantage,
                    missing_epsilon_MSE_minus_EEG=.15, native_IP_response_by_arm=metrics,
                    EEG_J_minus_permuted_EEG_J=metrics[reader.ARMS[0]]['J'] - metrics[reader.ARMS[1]]['J'],
                    EEG_J_minus_missing_J=metrics[reader.ARMS[0]]['J'] - metrics[reader.ARMS[4]]['J'],
                    noise_sha256=hashlib.sha256(key.encode()).hexdigest(),
                    noised_real_T_latent_sha256=hashlib.sha256(('SYNTHETIC_LATENT' + key).encode()).hexdigest(),
                    conditional_full_UNet_forwards=5, captured_IP_layers=70,
                    Q_reference_arm=reader.ARMS[4], no_generated_image=True))
        self.summary = dict(status='FROZEN_T_ONLY_FUNCTIONAL_MEASUREMENTS_SAVED', by_timestep=[],
            teacher_advantage_required_for_engineering_success=False, final_test_inference_performed=False)
        self.layers = []
        for timestep in times:
            rows = [r for r in self.rows if r['timestep'] == timestep]
            item = dict(timestep=timestep, queries=32,
                mean_epsilon_MSE={a: reader.mean([r['epsilon_MSE'][a] for r in rows]) for a in reader.ARMS},
                mean_native_IP_response_by_arm={a: {m: reader.mean([r['native_IP_response_by_arm'][a][m] for r in rows])
                    for m in reader.METRICS} for a in reader.ARMS})
            for field in ('EEG_epsilon_MSE_minus_correct_F', 'permuted_EEG_epsilon_MSE_minus_EEG',
                          'wrong_F_epsilon_MSE_minus_correct_F', 'missing_epsilon_MSE_minus_EEG',
                          'EEG_J_minus_permuted_EEG_J', 'EEG_J_minus_missing_J'):
                item['mean_' + field] = reader.mean([r[field] for r in rows])
            item['fraction_correct_F_lower_epsilon_MSE_than_EEG'] = reader.mean([float(r['EEG_epsilon_MSE_minus_correct_F'] > 0) for r in rows])
            item['fraction_EEG_IP_response_closer_to_correct_than_wrong'] = reader.mean([float(r['native_IP_response_by_arm'][reader.ARMS[0]]['J'] > 0) for r in rows])
            self.summary['by_timestep'].append(item)
            for arm in reader.ARMS:
                for layer in range(70):
                    self.layers.append(dict(layer=f'SYNTHETIC.native_layer_{layer:02d}.processor', timestep=timestep,
                        arm=arm, queries=32, query_positions=16, **item['mean_native_IP_response_by_arm'][arm]))
        self.completion = dict(status=reader.COMPLETION, registration_sha256=self.policy['diagnostic_registration']['sha256'],
            actual_full_UNet_forwards=dict(calls=160, samples=640), real_T_queries=32, rows=128, native_IP_layers=70,
            generated_images=0, fitted_parameters=0, engineering_checks_passed=True, protected_scores_read=False,
            component_effectiveness_established=False, native_defect_mechanism_established=False, timesteps=times)
        self.commit()

    def commit(self):
        artifacts = dict(provenance=self.provenance, fixed_queries=self.fixed, per_layer_timestep=self.layers, summary=self.summary)
        for name, value in artifacts.items():
            path = self.out / (name + '.json'); path.write_bytes(encode(value))
            self.completion[name + '_sha256'] = digest(path)
        raw_path = self.out / 'per_query_timestep.jsonl'
        raw_path.write_bytes(b''.join((json.dumps(row, allow_nan=False) + '\n').encode() for row in self.rows))
        self.completion['per_query_timestep_sha256'] = digest(raw_path)
        completion_path = self.out / 'completion.json'; completion_path.write_bytes(encode(self.completion))
        for name, base in self.base_states.items():
            value = dict(base)
            if name == 'diagnostic_queue_state':
                value.update(completion_path=str(completion_path), completion_sha256=digest(completion_path))
            Path(self.policy[name]).write_bytes(encode(value))


def must_refuse(fixture, mutate, label, reads_expected=None, live=lambda pid: False):
    fixture.build()
    mutate(fixture)
    fs = reader.Files()
    try:
        reader.read_evidence(fixture.policy, files=fs, live=live, synthetic=True)
    except reader.EvidenceRefused:
        pass
    else:
        raise AssertionError('Invalid fixture accepted: ' + label)
    if reads_expected is not None:
        assert fs.reads == [fixture.policy[name] for name in reads_expected], (label, fs.reads)
    return label


def checks():
    started = time.monotonic(); results = []
    with tempfile.TemporaryDirectory(prefix='brainhive_T32_evidence_synthetic_') as directory:
        fixture = Fixture(directory)
        read = lambda: reader.read_evidence(fixture.policy, live=lambda pid: False, synthetic=True)
        result = read()
        assert result['scope'] == 'SYNTHETIC_CONTRACT_FIXTURE_ONLY'
        assert result['evidence']['status'] == 'DIRECTIONAL_SUPPORT_THIS_PROBE_ONLY'
        assert result['automatic_admission'] is result['contract_consumable_review'] is result['fitting_authorized'] is False
        assert len(result['evidence']['all128_query_timestep_details']) == 128
        assert 'EEG_pairing_signal_present' not in result and 'teacher_response_nondegenerate' not in result
        results.append('Complete fixed-denominator directional fixture; no admission/review booleans produced')
        fixture.build(raw=(.6,) * 4, permuted=(.6,) * 4, missing=(.6,) * 4)
        result = read()
        assert abs(fixture.rows[0]['native_IP_response_by_arm']['correct_F']['J'] - 1) < 2e-6
        assert result['evidence']['status'] == 'UNKNOWN'
        results.append('Beautiful algebraic teacher self-J does not turn zero EEG pairing contrast into support')
        fixture.build(raw=((.2, 2.),) * 4)
        result = read()['evidence']
        assert result['aggregate']['EEG_J_minus_permuted_EEG_J'] > 0
        assert result['aggregate']['permuted_d_positive_minus_EEG_d_positive'] < 0
        assert result['status'] == 'UNKNOWN'
        results.append('Ranking can improve while positive response MSE worsens; conflicting evidence stays unknown')
        fixture.build(teacher_advantage=0.)
        assert read()['evidence']['status'] == 'UNKNOWN'
        results.append('Teacher self identities plus favorable EEG directions cannot replace the nontrivial full-UNet teacher comparison')
        fixture.build(raw=(.05, .9, .9, .9), permuted=(.5,) * 4, missing=(.6,) * 4)
        result = read()['evidence']
        assert result['all_four_timestep_details'][0]['EEG_J_minus_permuted_EEG_J'] > 0
        assert result['aggregate']['EEG_J_minus_permuted_EEG_J'] < 0
        assert result['status'] == 'NOT_SUPPORTED_THIS_PROBE_ONLY'
        results.append('One attractive time is not selected over the four-time/full32 mean')
        fixture.build(near_zero=True)
        result = read()['evidence']
        assert result['status'] == 'UNKNOWN' and result['functional_v2_masked_correspondence'].startswith('UNKNOWN')
        results.append('Any near-zero teacher item without joint query-by-layer output makes masked-v2 correspondence unknown')
        results.append(must_refuse(fixture, lambda f: (f.rows.pop(), f.commit()), 'Missing row rejected despite refreshed artifact SHA'))
        results.append(must_refuse(fixture, lambda f: (f.rows[0].update(stimulus_id='WRONG.jpg'), f.commit()), 'Wrong query identity rejected despite refreshed SHA'))
        results.append(must_refuse(fixture, lambda f: (f.out / 'summary.json').write_text('{}'), 'Actual artifact SHA tampering rejected'))
        results.append(must_refuse(fixture, lambda f: (f.completion.update(engineering_checks_passed='UNKNOWN'), f.commit()), 'Truthy string cannot replace a literal completion boolean'))
        def state_change(f, key, **fields):
            path = Path(f.policy[key]); state = json.loads(path.read_bytes()); state.update(fields); path.write_bytes(encode(state))
        results.append(must_refuse(fixture, lambda f: state_change(f, 'main_state', status='RUNNING'),
            'Running main refuses before queue/raw/summary reads', ['main_state']))
        results.append(must_refuse(fixture, lambda f: None, 'Live main PID refuses before result reads', ['main_state'], live=lambda pid: pid == 21000001))
        results.append(must_refuse(fixture, lambda f: state_change(f, 'diagnostic_queue_state', status='RUNNING'),
            'Running diagnostic refuses before job/raw/summary reads', ['main_state', 'diagnostic_queue_state']))
        results.append(must_refuse(fixture, lambda f: None, 'Live diagnostic PID refuses before result reads',
            ['main_state', 'diagnostic_queue_state'], live=lambda pid: pid == 21000002))
        results.append(must_refuse(fixture, lambda f: state_change(f, 'diagnostic_job_state', status='RUNNING'),
            'Running actual diagnostic wrapper refuses before raw/summary reads', ['main_state', 'diagnostic_queue_state', 'diagnostic_job_state']))
        results.append(must_refuse(fixture, lambda f: None, 'Live diagnostic wrapper PID refuses before result reads',
            ['main_state', 'diagnostic_queue_state', 'diagnostic_job_state'], live=lambda pid: pid == 21000004))
        results.append(must_refuse(fixture, lambda f: (f.out / 'per_query_timestep.jsonl').unlink(),
            'Missing result refuses after only terminal metadata reads', ['main_state', 'diagnostic_queue_state', 'diagnostic_job_state']))
        fixture.build()
        bad_policy = copy.deepcopy(fixture.policy); bad_policy['automatic_admission'] = 'UNKNOWN'
        try:
            reader.read_evidence(bad_policy, synthetic=True)
        except reader.EvidenceRefused:
            results.append('Truthy policy UNKNOWN is rejected before result reads')
        else:
            raise AssertionError('Truthy string policy was accepted')
        assert 'torch' not in sys.modules and 'diffusers' not in sys.modules
    return dict(schema='BrainHIVE_native_T32_evidence_synthetic_CPU_contract_v1',
        status='PASS_SYNTHETIC_ONLY_NO_ACTUAL_DIAGNOSTIC_READ', checks=results, check_count=len(results),
        reader_source_sha256=digest(reader.__file__), test_source_sha256=digest(__file__),
        actual_T32_V_or_test_score_reads=0, actual_EEG_or_weights_loaded=False,
        torch_imported=False, GPU_used=False, SSH_used=False, fitting_started=False,
        automatic_admission=False, seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output:
        assert not args.output.exists(), 'Preserve previous synthetic receipts'
    result = checks()
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
