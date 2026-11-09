"""Meaningful CPU algebra, off-training-edge identity and status-only checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import diagnose_BrainHIVE_paired_direction_gain_T80_20261026_v1 as P


def response(student,teacher):
    s=torch.tensor(student,dtype=torch.float32).reshape(1,1,1,-1)
    t=torch.tensor(teacher,dtype=torch.float32).reshape(1,1,1,-1)
    return P.response_measures(s,torch.zeros_like(s),t,torch.zeros_like(t),torch)


def identity_fixture():
    subjects=['sub-%02d'%i for i in range(1,11)];allowed=list(range(16));rows=[]
    for subject in subjects:
        base=len(rows)
        for i in range(8):
            rows.append(dict(subject=subject,array_row=i,concept_id='concept-%d'%i,source_role='oof%d'%(i//4),
                negative_query_index=base+(i//4)*4+(i+1)%4))
    # Each selected row's training partner is an UNSELECTED row. This is a
    # bijective plan over 160 identities; the real preflight verifies 132300.
    partner=[subject*16+(i+8)%16 for subject in range(10) for i in range(16)]
    return rows,allowed,subjects,partner


class Algebra(unittest.TestCase):
    def test_perfect_pair_and_gain_definitions(self):
        values,masks,flags=response([1,0],[1,0])
        self.assertAlmostEqual(float(values['cosine'][0]),1)
        self.assertEqual(float(values['raw_difference_MSE'][0]),0)
        self.assertAlmostEqual(float(values['projection_gain'][0]),1)
        self.assertAlmostEqual(float(values['student_along_teacher_gain'][0]),1)
        self.assertTrue(all(bool(m[0]) for m in masks.values()))

    def test_gap_can_improve_while_direction_unchanged_and_error_worsens(self):
        small,_,_=response([.5,.5],[1,0]);large,_,_=response([1,1],[1,0])
        self.assertAlmostEqual(float(small['cosine'][0]),float(large['cosine'][0]),places=6)
        self.assertAlmostEqual(float(large['pair_gap'][0]),2*float(small['pair_gap'][0]),places=6)
        self.assertGreater(float(large['normalized_difference_MSE'][0]),float(small['normalized_difference_MSE'][0]))

    def test_zero_student_keeps_energy_error_and_masks_undefined_direction(self):
        values,masks,flags=response([0,0],[1,0])
        self.assertTrue(bool(flags['teacher_valid'][0]))
        self.assertTrue(bool(flags['student_degenerate'][0]))
        self.assertFalse(bool(masks['cosine'][0]));self.assertFalse(bool(masks['projection_gain'][0]))
        self.assertTrue(bool(masks['normalized_difference_MSE'][0]))
        self.assertEqual(float(values['student_difference_energy'][0]),0)
        self.assertEqual(float(values['difference_dot'][0]),0)
        self.assertEqual(float(values['student_along_teacher_gain'][0]),0)
        self.assertAlmostEqual(float(values['normalized_difference_MSE'][0]),float(values['amplitude_error'][0]),places=6)
        self.assertEqual(float(values['angular_error'][0]),0)

    def test_zero_teacher_has_explicit_degeneracy_not_spurious_direction(self):
        values,masks,flags=response([1,0],[0,0])
        self.assertTrue(bool(flags['teacher_degenerate'][0]))
        self.assertFalse(bool(masks['cosine'][0]))
        self.assertFalse(bool(masks['normalized_difference_MSE'][0]))
        self.assertTrue(bool(masks['raw_difference_MSE'][0]))
        self.assertEqual(float(values['raw_difference_MSE'][0]),.5)

    def test_nonfinite_row_is_explicitly_undefined(self):
        _,masks,flags=response([float('inf'),0],[1,0])
        self.assertFalse(bool(flags['finite_inputs'][0]))
        self.assertTrue(all(not bool(v[0]) for v in masks.values()))

    def test_basic_amplitude_angular_and_absolute_decompositions(self):
        torch.manual_seed(1729)
        tensors=[torch.randn(3,2,4,5) for _ in range(4)]
        values,_,_=P.response_measures(*tensors,torch)
        self.assertTrue(torch.allclose(values['normalized_difference_MSE'],values['amplitude_error']+values['angular_error'],rtol=2e-5,atol=2e-5))
        self.assertTrue(torch.allclose(values['absolute'],values['common']+values['contrast'],rtol=2e-5,atol=2e-5))

    def test_common_response_shift_does_not_change_pair_diagnostics(self):
        s=torch.tensor([1.,2.]).reshape(1,1,1,2);z=torch.zeros_like(s);t=torch.tensor([2.,3.]).reshape_as(s)
        before,_,_=P.response_measures(s,z,t,z,torch)
        after,_,_=P.response_measures(s+7,z+7,t,z,torch)
        for key in ('cosine','projection_gain','student_along_teacher_gain','normalized_difference_MSE','pair_gap','amplitude_error','angular_error'):
            self.assertTrue(torch.allclose(before[key],after[key]),key)
        self.assertGreater(float(after['common'][0]),float(before['common'][0]))

    def test_antialignment_is_angular_error(self):
        values,_,_=response([-1,0],[1,0])
        self.assertAlmostEqual(float(values['cosine'][0]),-1)
        self.assertEqual(float(values['amplitude_error'][0]),0)
        self.assertAlmostEqual(float(values['normalized_difference_MSE'][0]),float(values['angular_error'][0]),places=5)

    def test_two_gain_directions_are_not_conflated(self):
        values,_,_=response([.5,0],[1,0])
        self.assertAlmostEqual(float(values['projection_gain'][0]),2)
        self.assertAlmostEqual(float(values['student_along_teacher_gain'][0]),.5)
        self.assertAlmostEqual(float(values['amplitude_ratio'][0]),.5)


class PairIdentities(unittest.TestCase):
    def test_original_off_training_edge_pairs_are_kept(self):
        rows,allowed,subjects,partner=identity_fixture();before=copy.deepcopy(rows)
        result=P.choose_probe_pairs(rows,allowed,subjects,partner)
        self.assertEqual(result['probe_permutation'],result['original_permutation'])
        self.assertFalse(result['replaced_mapping']);self.assertEqual(rows,before)
        self.assertEqual((result['queries'],result['subjects'],result['unique_stimuli']),(80,10,8))

    def test_forward_training_edge_causes_identity_only_replacement(self):
        rows,allowed,subjects,partner=identity_fixture();partner[0],partner[9]=partner[9],partner[0]
        result=P.choose_probe_pairs(rows,allowed,subjects,partner)
        self.assertEqual(result['original_forward_training_edge_overlaps'],1)
        self.assertTrue(result['replaced_mapping']);self.assertEqual(result['final_forward_training_edge_overlaps'],0)
        self.assertEqual(result['final_reverse_training_edge_overlaps'],0)
        self.assertEqual(result,P.choose_probe_pairs(rows,allowed,subjects,partner))
        self.assertTrue(result['selection_used_only_identities']);self.assertFalse(result['selection_used_features_or_scores'])

    def test_reverse_training_edge_also_excluded(self):
        rows,allowed,subjects,partner=identity_fixture();partner[1],partner[8]=partner[8],partner[1]
        result=P.choose_probe_pairs(rows,allowed,subjects,partner)
        self.assertEqual(result['original_reverse_training_edge_overlaps'],1)
        self.assertTrue(result['replaced_mapping']);self.assertEqual(result['final_reverse_training_edge_overlaps'],0)

    def test_bad_permutation_or_same_concept_is_rejected(self):
        rows,allowed,subjects,partner=identity_fixture();rows[0]['negative_query_index']=4
        with self.assertRaises(AssertionError):P.choose_probe_pairs(rows,allowed,subjects,partner)
        rows,allowed,subjects,partner=identity_fixture();rows[1]['concept_id']=rows[0]['concept_id']
        with self.assertRaises(AssertionError):P.choose_probe_pairs(rows,allowed,subjects,partner)

    def test_cross_subject_or_role_cannot_be_selected(self):
        rows,allowed,subjects,partner=identity_fixture()
        rows[0]['source_role']='invalid'
        with self.assertRaises(AssertionError):P.choose_probe_pairs(rows,allowed,subjects,partner)


class CompletionStatus(unittest.TestCase):
    def test_only_first_status_field_is_parsed(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'comparison.json'
            # Deliberately invalid JSON after the first status: no score content
            # is parsed. Real upstream also requires the queue-bound file SHA.
            path.write_text('{\n  "status": "EXPECTED",\n  INVALID_SCORE_CONTENT',encoding='utf-8')
            P.status_prefix(path,'EXPECTED')

    def test_wrong_or_late_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'comparison.json'
            for content in ('{\n  "status": "WRONG",\n}', '{\n  "scores": {},\n  "status": "EXPECTED"\n}'):
                path.write_text(content,encoding='utf-8')
                with self.assertRaises(AssertionError):P.status_prefix(path,'EXPECTED')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (Algebra,PairIdentities,CompletionStatus)))
    assert result.wasSuccessful()
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PAIRED_DIRECTION_GAIN_CPU_CHECKS_PASSED',CPU_checks=result.testsRun,
        GPU_used=False,new_fit=False,real_study_files_changed=False)))
