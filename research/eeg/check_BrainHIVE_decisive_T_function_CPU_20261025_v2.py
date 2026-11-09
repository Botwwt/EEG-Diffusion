"""Synthetic CPU checks; no server, real EEG/image/score, fitting, or CUDA."""
import copy
import json
import unittest

import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as G
import analyze_BrainHIVE_functional_T_logs_CPU_20261025_v1 as L
from brainhive_functional_training_contract_v2 import teacher_mappings


def synthetic_selection():
    allowed = list(range(13230))
    protocol = dict(image_rows=dict(T=allowed, D=[20000], V=[20001]))
    metadata = {index: dict(dataset_split='train', stimulus_id='s%05d.jpg' % index,
                            concept_id='concept%04d' % (index // 10)) for index in allowed}
    subjects = []
    for si in range(1, 11):
        sources = []
        for role in (0, 1):
            rows = [dict(array_row=index, cache_row=index) for index in allowed if index // 10 % 2 == role]
            sources.append(dict(role='oof%d' % role, groups=dict(component_train=rows)))
        subjects.append(dict(subject='sub-%02d' % si, sources=sources))
    return protocol, metadata, dict(subjects=subjects)


def synthetic_log(arm='PN'):
    records = []
    for update in L.EXPECTED_UPDATES:
        number = 1 if update <= 8269 else (2 if update <= 16538 else 3)
        loss = 2 - update / 24807
        records.append(dict(stage='BRAINHIVE_FUNCTIONAL_ALIGNMENT_V2_REAL_OOF_TRAINING', arm=arm,
            pass_number=number, updates=update, expected_updates=24807,
            seen_queries=396900 if update == 24807 else min(update * 16, 396899), expected_queries=396900,
            last_loss=loss, epsilon_loss=loss, token_loss=.5, functional_auxiliary=loss,
            cumulative_mean_loss=loss + .1, gradient_norm=.5, correction_RMS=.1,
            functional_diagnostics=dict(positive_alignment=loss), UNet_forwards=2, mean_gate=1., seconds=update))
    return records


def synthetic_measurements():
    conditions = ['parent', 'O', 'P', 'PN', 'S']
    conditions += [condition + '_permuted' for condition in conditions] + ['correct_F', 'wrong_F', 'missing']
    errors = dict(parent=2., O=1.8, P=1.5, PN=1., S=1.6, correct_F=0., wrong_F=1., missing=2.2)
    errors.update({condition + '_permuted': 3. for condition in ('parent', 'O', 'P', 'PN', 'S')})
    margins = {condition: 2 - error for condition, error in errors.items()}
    cells, network = {}, []
    rows = [dict(subject='sub-%02d' % (index // 8 + 1)) for index in range(80)]
    for domain in G.DOMAINS:
        for query in range(80):
            for ti in range(4):
                for condition in conditions:
                    cells[(domain, query, ti, condition)] = dict(layers=70, valid=70,
                        error=errors[condition] * 70, margin=margins[condition] * 70)
                for condition, error in dict(parent=2., PN=1., correct_F=.5, wrong_F=3., missing=2., parent_permuted=3.).items():
                    network.append(dict(domain=domain, query_index=query, timestep_index=ti,
                        condition=condition, known_training_noise=domain == G.DOMAINS[0],
                        epsilon_MSE=error if domain == G.DOMAINS[0] else None,
                        prediction_to_correct_F_MSE=error, prediction_to_wrong_F_MSE=error + 1))
    gradients = [dict(arm=arm, subject='sub-%02d' % subject, timestep_index=ti,
        ranking_active_in_training=arm in ('PN', 'S'),
        cosines=dict(epsilon__positive_weighted=-.1, epsilon__ranking_weighted=-.2),
        compound_gradient=dict(base_vs_active_auxiliary_cosine=-.1,
            summed_gradient_reverses_base_descent=False, active_auxiliary_to_base_norm_ratio=.1))
        for arm in G.ARMS for subject in range(1, 11) for ti in range(4)]
    saturation = [dict(arm=arm, query_index=qi, radius_kind=kind, radius=.25, disabled=False, utilization=.995)
                  for arm in G.ARMS for qi in range(80) for kind in ('training_radius', 'deployment_radius')]
    return rows, cells, network, gradients, saturation


class ContractChecks(unittest.TestCase):
    def test_actual_FP32_head_precedes_generator_autocast(self):
        import torch
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        G.torch = torch
        handle = None
        try:
            torch.manual_seed(1729)
            head = BoundedConditionAdapter(predict_risk=False)
            with torch.no_grad():
                head.direction.weight.normal_(std=.001)
            condition = torch.randn(2, 1024)
            weight, bias = torch.ones_like(condition), torch.zeros_like(condition)
            observed = []
            handle = head.trunk[0].register_forward_hook(lambda module, args, value: observed.append(value.dtype))
            generator_projection = torch.nn.Linear(1024, 4)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                changed, _ = G.training_head_forward(head, condition, weight, bias)
                tokens = generator_projection(changed)
                loss = tokens.float().square().mean()
            gradients = torch.autograd.grad(loss, list(head.parameters()))
            self.assertEqual(observed, [torch.float32])
            self.assertEqual(changed.dtype, torch.float32)
            self.assertEqual(tokens.dtype, torch.bfloat16)
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
            self.assertGreater(sum(float(value.square().sum()) for value in gradients), 0)
        finally:
            if handle is not None:
                handle.remove()
            del G.torch

    def test_active_composite_overrules_individual_branch_negative_cosines(self):
        import torch
        G.torch = torch
        try:
            # Both auxiliary branches conflict with epsilon, yet the ACTUAL
            # base includes token and aligns with their sum.
            gradients = dict(epsilon=torch.tensor([1., 0.]), token_weighted=torch.tensor([-3., 0.]),
                positive_weighted=torch.tensor([-1., 0.]), ranking_weighted=torch.tensor([-.5, 0.]))
            result = G.compound_gradient_diagnostics(gradients, 'PN')
            self.assertGreater(result['base_vs_active_auxiliary_cosine'], .99)
            self.assertGreater(result['base_dot_base_plus_auxiliary'], 0)
            self.assertFalse(result['summed_gradient_reverses_base_descent'])
            ordinary = G.compound_gradient_diagnostics(gradients, 'O')
            self.assertIsNone(ordinary['base_vs_active_auxiliary_cosine'])
            self.assertEqual(ordinary['active_auxiliary_norm'], 0)
        finally:
            del G.torch
    def test_plain_import_is_GPU_free(self):
        self.assertNotIn('torch', G.__dict__)

    def test_pure_contract_entry(self):
        self.assertEqual(G.contract_check()['status'], 'PURE_CPU_SYNTHETIC_CONTRACT_CHECK_PASSED')

    def test_selection_balances_all_subjects_and_source_roles(self):
        protocol, metadata, contracts = synthetic_selection()
        rows = G.choose_rows(protocol, metadata, contracts)
        self.assertEqual(len(rows), 80)
        self.assertEqual(len({row['array_row'] for row in rows}), 8)
        for subject in range(10):
            selected = rows[subject * 8:(subject + 1) * 8]
            self.assertEqual(sum(row['source_role'] == 'oof0' for row in selected), 4)
            self.assertEqual(sum(row['source_role'] == 'oof1' for row in selected), 4)
            self.assertEqual(len({row['concept_id'] for row in selected}), 8)
            for row in selected:
                negative = rows[row['negative_query_index']]
                self.assertEqual(negative['subject'], row['subject'])
                self.assertEqual(negative['source_role'], row['source_role'])
                self.assertNotEqual(negative['concept_id'], row['concept_id'])

    def test_selection_rejects_cross_subject_OOF_role_disagreement(self):
        protocol, metadata, contracts = synthetic_selection()
        contracts['subjects'][1]['sources'][0]['role'] = 'broken'
        with self.assertRaises(AssertionError):
            G.choose_rows(protocol, metadata, contracts)

    def test_real_helper_teacher_mapping_API(self):
        rows = list(range(40))
        concepts = {row: row // 10 for row in rows}
        mapping = teacher_mappings(rows, concepts)
        self.assertEqual(set(mapping), {'negative', 'rank_positive_shuffled', 'rank_negative_shuffled', 'receipt'})
        for name in ('negative', 'rank_positive_shuffled', 'rank_negative_shuffled'):
            self.assertEqual(sorted(mapping[name]), rows)
            self.assertTrue(all(concepts[row] != concepts[mapping[name][row]] for row in rows))

    def test_complete_log_schedule(self):
        result = L.analyze_records(synthetic_log(), 'PN')
        self.assertEqual(result['unique_logged_updates'], 390)
        self.assertLess(result['metrics']['epsilon_loss']['last_minus_first'], 0)
        self.assertTrue(result['logged_update_count_is_not_loss_for_all_24807_updates'])

    def test_missing_log_event_cannot_be_dropped(self):
        with self.assertRaises(AssertionError):
            L.analyze_records(synthetic_log()[1:], 'PN')

    def test_nonfinite_log_loss_rejected(self):
        records = synthetic_log(); records[10]['epsilon_loss'] = float('nan')
        with self.assertRaises(AssertionError):
            L.analyze_records(records, 'PN')

    def test_resumed_duplicate_and_rollback_reported(self):
        records = synthetic_log()
        resumed = records[:100] + records[90:]
        result = L.analyze_records(resumed, 'PN')
        self.assertEqual(result['duplicate_updates'], 10)
        self.assertEqual(result['rollback_boundaries'], 1)

    def test_mask_and_full_coverage_required(self):
        data = synthetic_measurements()
        result = G.summarize(*data)
        self.assertEqual(result['coverage']['observed_query_timestep_domain_cells'], 640)
        self.assertEqual(result['diagnostic_decisions']['teacher_discrimination']['result'], 'SUPPORTED')
        self.assertEqual(result['diagnostic_decisions']['paired_EEG_signal']['result'], 'SUPPORTED')
        data[1].pop(next(iter(data[1])))
        with self.assertRaises(AssertionError):
            G.summarize(*data)

    def test_all_three_controls_required_for_transfer(self):
        data = synthetic_measurements()
        self.assertEqual(G.summarize(*data)['diagnostic_decisions']['trajectory_transfer']['result'], 'SUPPORTED')
        for (domain, query, ti, condition), value in data[1].items():
            if condition == 'PN' and domain == G.DOMAINS[1]:
                value.update(error=2.5 * 70, margin=-.5 * 70)
        self.assertEqual(G.summarize(*data)['diagnostic_decisions']['trajectory_transfer']['result'], 'NOT_SUPPORTED')

    def test_absent_DDPM_gain_cannot_be_state_mismatch(self):
        data = synthetic_measurements()
        for (domain, query, ti, condition), value in data[1].items():
            if condition == 'PN':
                value.update(error=1.8 * 70, margin=.2 * 70)
        self.assertEqual(G.summarize(*data)['diagnostic_decisions']['trajectory_transfer']['result'], 'UNKNOWN')

    def test_near_zero_teachers_preserve_query_and_force_unknown(self):
        data = synthetic_measurements()
        for condition in ('PN', 'O'):
            data[1][(G.DOMAINS[0], 0, 0, condition)].update(valid=0, error=0., margin=0.)
        result = G.summarize(*data)
        self.assertEqual(result['coverage']['expected_queries'], 80)
        self.assertEqual(result['diagnostic_decisions']['PN_over_O']['result'], 'UNKNOWN')

    def test_common_shift_does_not_become_paired_signal(self):
        data = synthetic_measurements()
        # Correct and permuted PN move equally: absolute teacher fit improves,
        # but paired stimulus gap remains exactly that of O.
        for (domain, query, ti, condition), value in data[1].items():
            if condition == 'PN_permuted':
                value.update(error=2.2 * 70, margin=-.2 * 70)
        result = G.summarize(*data)
        self.assertNotEqual(result['diagnostic_decisions']['paired_EEG_signal']['result'], 'SUPPORTED')
        self.assertEqual(result['diagnostic_decisions']['trajectory_transfer']['result'], 'UNKNOWN')

    def test_missing_whole_UNet_controls_or_gradient_rows_cannot_be_dropped(self):
        data = synthetic_measurements()
        data[2].pop()
        with self.assertRaises(AssertionError):
            G.summarize(*data)
        data = synthetic_measurements()
        data[3].pop()
        with self.assertRaises(AssertionError):
            G.summarize(*data)

    def test_trajectory_cannot_receive_a_training_noise_target(self):
        data = synthetic_measurements()
        next(row for row in data[2] if row['domain'] == G.DOMAINS[1])['epsilon_MSE'] = 0.
        with self.assertRaises(AssertionError):
            G.summarize(*data)

    def test_nonfinite_compound_cosine_is_unknown_not_no_conflict(self):
        data = synthetic_measurements()
        next(row for row in data[3] if row['arm'] == 'PN')['compound_gradient']['base_vs_active_auxiliary_cosine'] = None
        result = G.summarize(*data)
        self.assertEqual(result['diagnostic_decisions']['gradient_conflict']['result'], 'UNKNOWN')

    def test_gradient_local_conflict_and_radius_have_separate_decisions(self):
        result = G.summarize(*synthetic_measurements())
        self.assertEqual(result['diagnostic_decisions']['gradient_conflict']['result'], 'SUPPORTED')
        self.assertEqual(result['diagnostic_decisions']['radius_saturation']['result'], 'SUPPORTED')
        self.assertEqual(result['parameter_updates'], 0)
        self.assertFalse(result['candidate_auto_started'])
        self.assertFalse(result['trajectory_noise_error_available'])

    def test_separate_gradient_probe_releases_shared_graph_last(self):
        import torch
        G.torch = torch
        try:
            weight = torch.tensor([.5, -.2], requires_grad=True)
            shared = (weight.square() + weight).sin()
            losses = dict(token_weighted=shared.square().mean(),
                positive_weighted=(shared - 1).square().mean(),
                ranking_weighted=torch.nn.functional.softplus(shared).mean(),
                epsilon=shared.cos().square().mean())
            vectors = G.separate_gradient_vectors(losses, [weight])
            self.assertEqual(list(vectors), ['token_weighted', 'positive_weighted', 'ranking_weighted', 'epsilon'])
            self.assertTrue(all(torch.isfinite(value).all() and not value.requires_grad for value in vectors.values()))
            self.assertIsNone(weight.grad)
            with self.assertRaises(RuntimeError):
                torch.autograd.grad(losses['epsilon'], [weight])
        finally:
            del G.torch


if __name__ == '__main__':
    unittest.main(verbosity=2)
