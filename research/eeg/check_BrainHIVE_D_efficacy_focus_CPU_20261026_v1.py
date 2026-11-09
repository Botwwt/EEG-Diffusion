"""Independent scope, producer/consumer and process-ownership CPU checks."""
import ast
import copy
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import Mock,patch,sentinel

import queue_BrainHIVE_D_efficacy_focus_20261026_v1 as Q
import compare_BrainHIVE_DO_efficacy_CPU_20261026_v1 as M


def document():
    return dict(schema=Q.SCHEMA,retained_candidate='D',completed_baseline='O',withdrawn_arms=['A','U'],
        withdrawn_diagnostics=['original_T80','direction_gain_addon'],candidate_ordinal=13,candidate_limit=13,
        original_scientific_core=Q.CORE,unchanged_D_training_and_inference=True,original_four_arm_protocol_completed=False,
        automatic_method_search=False,maximum_metadata_bytes=64<<20,comparison_maximum_seconds=1800,
        comparison_maximum_bytes=64<<20,efficacy_gate=dict(DINO_minimum=.01,DINO_CI_lower_positive=True,
            SEED_CI_lower_positive=True,ObjectF1_mean_nonnegative=True,SSIM_Y_mean_nonnegative=True,PixCorr_RGB_mean_nonnegative=True),
        mechanism_claim='WITHDRAWN_NOT_TESTED',SOTA_established=False)


def config():
    return dict(registration_path='/original/study_review_v3.json',training_job_names={'D':['Dtrain','Dtrain_resume']},
        generation_job_names={'D':dict(T32=['D_T32'],V=['D_V','D_V_resume'])})


def ownership_objects():
    process=Mock();process.pid=111;process.poll.return_value=None
    job=Mock(spec=['exists']);job.exists.return_value=True
    spec=dict(name='registered_D_job',command=['python','D_script.py'])
    record=dict(wrapper_pid=111,pid=112,name=spec['name'],command=spec['command'],status='RUNNING')
    return process,spec,job,record


class Scope(unittest.TestCase):
    def test_valid_scope(self):Q.scope(document())

    def test_new_O_fit_or_A_U_retention_rejected(self):
        for arm in ('O','A','U'):
            candidate=document();candidate['retained_candidate']=arm
            with self.assertRaises(AssertionError):Q.scope(candidate)
        for arms in (['U'],['A'],[],['A','D','U']):
            candidate=document();candidate['withdrawn_arms']=arms
            with self.assertRaises(AssertionError):Q.scope(candidate)

    def test_new_candidate_or_changed_core_rejected(self):
        for key,value in (('candidate_ordinal',14),('candidate_limit',14),('original_scientific_core','0'*64),
                          ('unchanged_D_training_and_inference',False),('automatic_method_search',True)):
            candidate=document();candidate[key]=value
            with self.assertRaises(AssertionError):Q.scope(candidate)

    def test_every_registered_effect_or_low_level_threshold_is_protected(self):
        for key in document()['efficacy_gate']:
            candidate=document();candidate['efficacy_gate'][key]=.009 if key=='DINO_minimum' else False
            with self.assertRaises(AssertionError):Q.scope(candidate)

    def test_original_four_arm_success_or_mechanism_claim_is_rejected(self):
        for key,value in (('original_four_arm_protocol_completed',True),('mechanism_claim','SUPPORTED'),('SOTA_established',True)):
            candidate=document();candidate[key]=value
            with self.assertRaises(AssertionError):Q.scope(candidate)

    def test_removed_probe_cannot_be_silently_restored(self):
        for removed in ([],['original_T80'],['direction_gain_addon']):
            candidate=document();candidate['withdrawn_diagnostics']=removed
            with self.assertRaises(AssertionError):Q.scope(candidate)


class StageContracts(unittest.TestCase):
    def test_exactly_D_six_stages_plus_independent_comparison(self):
        stages=Q.specs(config())
        self.assertEqual([s['stage'] for s in stages],['train','postfit','generate','audit','features','score','compare'])
        self.assertEqual(len({s['name'] for s in stages}),7)
        self.assertTrue(all(s['arm']=='D' for s in stages[:-1]))
        self.assertTrue(all('--arm' in s['command'] and s['command'][s['command'].index('--arm')+1]=='D' for s in stages[:-1]))
        self.assertTrue(all('T80' not in ' '.join(s['command']) and 'direction_gain' not in ' '.join(s['command']) for s in stages))
        self.assertNotIn('--resume',stages[0]['command'])
        self.assertEqual(stages[0]['maximum_seconds'],86400)
        self.assertEqual(stages[-1]['maximum_seconds'],1800)
        self.assertFalse(stages[-1]['GPU'])
        self.assertIn('compare_BrainHIVE_DO_efficacy_CPU_20261026_v1.py',' '.join(stages[-1]['command']))

    def test_comparison_producer_status_matches_queue_consumer(self):
        # Read the actual producer declaration, not a duplicate expected string.
        tree=ast.parse(inspect.getsource(M.compare))
        statuses=[kw.value.value for node in ast.walk(tree) if isinstance(node,ast.Call)
            for kw in node.keywords if kw.arg=='status' and isinstance(kw.value,ast.Constant)
            and isinstance(kw.value.value,str) and 'DO_EFFICACY_COMPARISON_COMPLETE' in kw.value.value]
        self.assertEqual(len(statuses),1)
        self.assertEqual(Q.specs(config())[-1]['expected_status'],statuses[0])


class Ownership(unittest.TestCase):
    def test_valid_live_job_returns_identity_adapter(self):
        process,spec,job,record=ownership_objects()
        with patch.object(Q.C,'read',return_value=record),patch.object(Q.Q,'live',return_value=True),patch.object(Q,'AdoptedProcess',return_value=sentinel.owned) as create:
            self.assertIs(Q.owned_after_registration(process,spec,job),sentinel.owned)
            create.assert_called_once_with(job,spec['name'],spec['command'],111,112)

    def test_wrong_job_name_command_or_wrapper_is_rejected_without_adoption(self):
        for key,value in (('name','foreign_job'),('command',['python','foreign.py']),('wrapper_pid',999)):
            process,spec,job,record=ownership_objects();record[key]=value
            with patch.object(Q.C,'read',return_value=record),patch.object(Q.Q,'live',return_value=True),patch.object(Q,'AdoptedProcess') as create:
                with self.assertRaises(AssertionError):Q.owned_after_registration(process,spec,job)
                create.assert_not_called()

    def test_dead_child_or_closed_wrapper_is_never_adopted(self):
        for wrapper_dead,child_dead in ((True,False),(False,True)):
            process,spec,job,record=ownership_objects()
            if wrapper_dead:process.poll.return_value=1
            clock=iter(range(20))
            with patch.object(Q.C,'read',return_value=record),patch.object(Q.Q,'live',return_value=not child_dead),patch.object(Q,'AdoptedProcess') as create,patch.object(Q.time,'monotonic',side_effect=lambda:float(next(clock))),patch.object(Q.time,'sleep'):
                self.assertIsNone(Q.owned_after_registration(process,spec,job));create.assert_not_called()

    def test_closed_receipt_is_not_mistaken_for_running_ownership(self):
        process,spec,job,record=ownership_objects();record.update(status='COMPLETED',exit_code=0);process.poll.return_value=0
        with patch.object(Q.C,'read',return_value=record),patch.object(Q.Q,'live',return_value=False),patch.object(Q,'AdoptedProcess') as create:
            self.assertIsNone(Q.owned_after_registration(process,spec,job));create.assert_not_called()

    def test_missing_receipt_with_exited_wrapper_is_not_guessed(self):
        process,spec,job,_=ownership_objects();job.exists.return_value=False;process.poll.return_value=1
        clock=iter(range(20))
        with patch.object(Q.C,'read',side_effect=FileNotFoundError('not registered')) as read,patch.object(Q,'AdoptedProcess') as create,patch.object(Q.time,'monotonic',side_effect=lambda:float(next(clock))),patch.object(Q.time,'sleep'):
            self.assertIsNone(Q.owned_after_registration(process,spec,job));self.assertGreater(read.call_count,0);create.assert_not_called()

    def test_missing_live_receipt_wait_is_finite(self):
        process,spec,job,_=ownership_objects();job.exists.return_value=False
        clock=iter(range(20))
        with patch.object(Q.time,'monotonic',side_effect=lambda:float(next(clock))),patch.object(Q.time,'sleep'),patch.object(Q.C,'read',side_effect=FileNotFoundError('not registered')) as read,patch.object(Q,'AdoptedProcess') as create:
            self.assertIsNone(Q.owned_after_registration(process,spec,job));self.assertGreater(read.call_count,0);create.assert_not_called()

    def test_partial_initial_JSON_is_retried_before_binding(self):
        process,spec,job,record=ownership_objects()
        partial=json.JSONDecodeError('partial wrapper write','',0)
        with patch.object(Q.C,'read',side_effect=[partial,record]) as read,patch.object(Q.Q,'live',return_value=True),patch.object(Q,'AdoptedProcess',return_value=sentinel.owned),patch.object(Q.time,'sleep'):
            self.assertIs(Q.owned_after_registration(process,spec,job),sentinel.owned)
            self.assertEqual(read.call_count,2)

    def test_live_PID_identity_failure_is_never_suppressed(self):
        process,spec,job,record=ownership_objects()
        with patch.object(Q.C,'read',return_value=record),patch.object(Q.Q,'live',return_value=True),patch.object(Q,'AdoptedProcess',side_effect=AssertionError('PID identity mismatch')):
            with self.assertRaisesRegex(AssertionError,'PID identity mismatch'):
                Q.owned_after_registration(process,spec,job)

    def test_receipt_closing_during_identity_binding_is_not_live_adoption(self):
        process,spec,job,record=ownership_objects();closed=record|dict(status='COMPLETED',exit_code=0)
        process.poll.side_effect=[None,0]
        with patch.object(Q.C,'read',side_effect=[record,closed,closed]),patch.object(Q.Q,'live',return_value=True),patch.object(Q,'AdoptedProcess',side_effect=AssertionError('receipt closed during bind')),patch.object(Q.time,'sleep'):
            self.assertIsNone(Q.owned_after_registration(process,spec,job))

    def test_closed_receipt_waits_for_wrapper_poll(self):
        process,spec,job,record=ownership_objects();record.update(status='COMPLETED',exit_code=0)
        process.poll.side_effect=[None,0]
        with patch.object(Q.C,'read',return_value=record),patch.object(Q,'AdoptedProcess') as create,patch.object(Q.time,'sleep'):
            self.assertIsNone(Q.owned_after_registration(process,spec,job));create.assert_not_called()


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (Scope,StageContracts,Ownership)))
    assert result.wasSuccessful()
    print(json.dumps(dict(status='D_EFFICACY_FOCUS_CPU_CHECKS_PASSED',CPU_checks=result.testsRun,
        GPU_used=False,SSH_used=False,new_fit=False,real_study_files_changed=False)))
