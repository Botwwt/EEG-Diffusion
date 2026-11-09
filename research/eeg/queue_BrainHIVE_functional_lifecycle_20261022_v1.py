"""Finite source-bound execution of the existing four full formal lifecycles.

Profiles and formal work are separate modes. Formal mode requires admission
already verified against actual new-core profiles, all inputs, frozen evaluator
readiness and all-four persistent output budgets. Training/generation can use
only their predeclared second exact-resume slot. Other failed/partial stages
stop and preserve evidence; no metric-driven branch, seed change or retry.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import brainhive_functional_training_contract_v2 as C
import prepare_BrainHIVE_functional_lifecycle_CPU_20261022_v1 as P

ROOT, WORK, BASE, CONFIG = P.ROOT, P.WORK, P.BASE, P.CONFIG
TRAIN_DONE = 'FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE'
GENERATE_DONE = 'FUNCTIONAL_ALIGNMENT_V2_ALL16600_IMAGES_AND_TARGET_JOIN_COMPLETE'
POSTFIT_DONE = 'CURRENT_ARM_FIXED_POSTFIT_T32_ENGINEERING_IMAGES_SAVED'


def write_state(path, value):
    value.update(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    temporary = path.with_suffix('.pending')
    temporary.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
    temporary.replace(path)


def recipe_set():
    prep = P.read(BASE / 'preparation.json')
    assert prep['status'] == 'COMPLETE_LIFECYCLE_RECIPES_READY_FOR_NEW_GPU_ENGINEERING_PROFILES'
    plan = P.load_ref(prep['plan'])
    recipes = {a: P.read(CONFIG / (a + '.json')) for a in C.ARMS}
    for name, digest in prep['source_sha256'].items():
        assert C.sha(ROOT / name) == digest, name
    C.validate_group_identity(plan, recipes)
    for recipe in recipes.values():
        C.validate_recipe(recipe)
    assert plan['execution']['fixed_arm_order'] == list(C.ARMS)
    return prep, plan, recipes


def command(recipe, stage, *, index=0):
    """One real existing entry point per stage; no arbitrary shell commands."""
    arm = recipe['arm']
    own = WORK / 'images' / C.run_name(arm)
    if stage == 'profile':
        return recipe['profile_job_name'], 1800, 'train_BrainHIVE_functional_alignment_v2.py', ['--recipe', str(CONFIG / (arm + '.json')), '--profile-only']
    if stage == 'train':
        assert index in (0, 1) and index < len(recipe['formal_job_names'])
        return recipe['formal_job_names'][index], recipe['training_job_seconds'], 'train_BrainHIVE_functional_alignment_v2.py', ['--recipe', str(CONFIG / (arm + '.json'))] + (['--resume'] if index else [])
    if stage == 'postfit':
        path = CONFIG / (arm + '_postfit_T32.json')
        value = P.read(path)
        return value['job_name'], 1200, 'generate_BrainHIVE_functional_postfit_T32_v1.py', ['--recipe', str(path)]
    if stage == 'generate':
        assert index in (0, 1) and index < len(recipe['generation_job_names'])
        return recipe['generation_job_names'][index], recipe['generation_job_seconds'], 'generate_BrainHIVE_functional_alignment_V_v2.py', ['--recipe', str(CONFIG / (arm + '.json'))]
    if stage == 'features':
        name = 'BrainHIVE_functional_v2_' + arm + '_expanded_lifecycle_' + P.TAG + '_features'
        return name, 14400, 'extract_image_features.py', ['--manifest', str(own / 'manifest.jsonl'), '--output', str(WORK / 'evaluation' / C.run_name(arm) / 'features')]
    raise AssertionError('Unknown fixed GPU stage ' + stage)


def training_can_resume(recipe, code):
    folder = Path(recipe['output_directory'])
    return code != 0 and (folder / 'resume.pt').is_file() and (folder / 'progress.json').is_file() and (
        P.read(folder / 'progress.json').get('status') == 'FUNCTIONAL_V2_FINITE_RESOURCE_STOP_RESUMABLE')


def generation_can_resume(recipe, code):
    folder = WORK / 'images' / C.run_name(recipe['arm'])
    return code != 0 and (folder / 'inference_manifest.jsonl').is_file() and (folder / 'progress.json').is_file() and (
        P.read(folder / 'progress.json').get('status') == 'FUNCTIONAL_V2_GENERATION_FINITE_STOP_EXACT_PREFIX_PRESERVED')


def exact_completion(path, status, *, arm=None, core=None, count=None):
    value = P.read(path)
    assert value['status'] == status, (str(path), value.get('status'))
    if arm is not None:
        assert value['arm'] == arm
    if core is not None:
        assert value['recipe_core_sha256'] == core
    if count is not None:
        assert value['queries'] == count
    return P.ref(path)


def check_output_budgets(plan):
    """These are real per-arm caps, including the generic feature extractor."""
    for entry in plan['arms']:
        for root, budget in (('training_root', 'training_budget_bytes'),
                ('feature_root', 'feature_budget_bytes'), ('metadata_root', 'metadata_budget_bytes')):
            used = C.directory_bytes(entry[root])
            assert used <= entry[budget], ('Sealed output cap exceeded; preserve and stop', entry['arm'], root, used, entry[budget])


def stop_owned_process(process, gpu, job):
    """Always wait the actual handle; an unobserved exit is not a stopped job."""
    registered = None
    if gpu and job.exists():
        record = P.read(job)
        if record.get('wrapper_pid') == process.pid:
            registered = record.get('pid')
            if C.process_live(registered):
                try:
                    os.killpg(registered, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    elif not gpu and process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    # Even if the GPU child has already exited, the wrapper may still be
    # closing its compute ledger. Missing job metadata is not terminal proof.
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if gpu and registered is not None and C.process_live(registered):
            try:
                os.killpg(registered, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            return False
    return process.poll() is not None


class Runner:
    def __init__(self, state_path, mode, preparation):
        assert not state_path.exists(), 'Preserve previous queue/attempt; no automatic restart'
        self.path = state_path
        self.metadata_before = C.directory_bytes(BASE)
        self.plan = P.load_ref(preparation['plan'])
        self.deadline = time.monotonic() + (7200 if mode == 'profiles' else 6 * 86400)
        self.state = dict(status='RUNNING_NEW_CORE_ENGINEERING_PROFILES' if mode == 'profiles' else 'RUNNING_FIXED_FOUR_ARM_FULL_LIFECYCLE',
            mode=mode, pid=os.getpid(), source_sha256=C.sha(__file__), preparation=P.ref(BASE / 'preparation.json'),
            plan=preparation['plan'], attempts=[], completed=[], current_stage=None,
            formal_training_started=False, image_gain_established=False, independent_SOTA_established=False)
        self.save()

    def save(self):
        write_state(self.path, self.state)

    def run(self, name, entry, arguments, seconds, gpu=False):
        assert math.isfinite(seconds) and 0 < seconds <= 86400
        assert time.monotonic() + seconds + 45 < self.deadline, 'Global finite queue boundary; preserve before starting stage'
        assert Path(name).name == name and entry in P.SOURCES
        assert C.sha(ROOT / entry) == P.read(BASE / 'preparation.json')['source_sha256'][entry]
        log = BASE / (name + '.log')
        assert not log.exists()
        child = [sys.executable, '-B', '-u', str(ROOT / entry), *arguments]
        job = ROOT / 'audit' / (name + '_job.json')
        if gpu:
            assert not job.exists(), 'Preserve every previous actual GPU attempt'
            argv = [sys.executable, '-B', str(ROOT / 'run_gpu_job_v2.py'), '--name', name,
                '--max-seconds', str(seconds), '--', *child]
        else:
            argv = child
        environment = os.environ.copy()
        if not gpu:
            environment['CUDA_VISIBLE_DEVICES'] = ''
        for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[variable] = '2'
        attempt = dict(name=name, argv=argv, log=str(log), maximum_seconds=seconds, GPU=gpu)
        self.state['attempts'].append(attempt)
        self.state.update(current_stage=name)
        self.save()
        started = time.monotonic()
        with log.open('xb') as stream:
            process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, shell=False, start_new_session=True)
            self.state.update(child_pid=process.pid)
            self.save()
            try:
                while process.poll() is None:
                    assert log.stat().st_size <= 16 << 20, 'Finite per-stage log cap'
                    assert C.directory_bytes(BASE) - self.metadata_before < 64 << 20, 'Finite complete-controller metadata cap'
                    check_output_budgets(self.plan)
                    if time.monotonic() - started > seconds + (35 if gpu else 0):
                        raise TimeoutError('Fixed stage time limit; outputs retained')
                    time.sleep(2)
            except BaseException:
                # The GPU wrapper's child owns another process group. Stop that
                # registered child, and let the still-live wrapper close ledger.
                terminal = stop_owned_process(process, gpu, job)
                self.state.update(unresolved_live_child=not terminal,
                    child_pid=None if terminal else process.pid,
                    observed_wrapper_terminal=terminal)
                self.save()
                raise
        attempt.update(exit_code=process.returncode, seconds=time.monotonic() - started, log_sha256=C.sha(log))
        self.state.update(child_pid=None)
        if gpu:
            record = P.read(job)
            assert record['status'] != 'RUNNING' and not C.process_live(record.get('pid'))
            attempt['job'] = P.ref(job)
            assert (record['exit_code'] == 0) == (process.returncode == 0)
        check_output_budgets(self.plan)
        assert C.directory_bytes(BASE) - self.metadata_before <= 64 << 20, 'Finite complete-controller metadata cap'
        self.save()
        return process.returncode

    def gpu(self, recipe, stage, index=0):
        name, seconds, entry, arguments = command(recipe, stage, index=index)
        return self.run(name, entry, arguments, seconds, True)

    def cpu(self, name, entry, arguments, seconds):
        code = self.run(name, entry, arguments, seconds)
        assert code == 0, 'CPU stage failed; preserved exact invocation and log'


def profiles():
    P.assert_host()
    preparation, plan, recipes = recipe_set()
    P.gpu_free()
    runner = Runner(BASE / 'new_core_profile_queue.json', 'profiles', preparation)
    try:
        for arm, recipe in recipes.items():
            C.verify_admission(recipe, 'profile')
            assert not Path(recipe['profile_output']).exists(), 'Profile path must have been explicitly archived during preparation'
            assert runner.gpu(recipe, 'profile') == 0
            item = exact_completion(recipe['profile_output'], 'ACTUAL_PARENT_Q_TWO_FORWARD_ENGINEERING_PROFILE_COMPLETE', arm=arm, core=C.core_sha(recipe))
            runner.state['completed'].append(dict(arm=arm, profile=item))
            runner.save()
        runner.state.update(status='ALL_FOUR_NEW_CORE_ACTUAL_GPU_PROFILES_COMPLETE_NO_FORMAL_FIT', current_stage=None)
        runner.save()
    except BaseException as error:
        runner.state.update(status=('NEW_CORE_PROFILE_STOP_REQUESTED_WRAPPER_STILL_LIVE' if runner.state.get('unresolved_live_child')
            else 'NEW_CORE_PROFILE_FAILED_PRESERVED_NO_AUTOMATIC_RETRY'), error=repr(error))
        runner.save()
        raise


def formal():
    P.assert_host()
    preparation, plan, recipes = recipe_set()
    closure = P.read(BASE / 'formal_admission_closure.json')
    assert closure['status'] == 'FOUR_ARM_FULL_LIFECYCLE_FORMAL_ADMISSION_ACTUALLY_VERIFIED'
    assert closure['plan'] == preparation['plan']
    for item in closure['recipes']:
        assert C.sha(item['path']) == item['sha256']
    for recipe in recipes.values():
        C.verify_admission(recipe, 'formal')
        assert not Path(recipe['output_directory']).exists(), 'Fresh formal queue only; diagnose previous attempts explicitly'
    # Verify all model assets and retained gallery shards before the first fit.
    ready = closure['evaluation']['readiness']
    assert C.sha(ready['path']) == ready['sha256']
    P.evaluation_ready(ready['path'], plan)
    P.gpu_free()
    runner = Runner(BASE / 'formal_queue.json', 'formal', preparation)
    entries = {entry['arm']: entry for entry in plan['arms']}
    try:
        for arm, recipe in recipes.items():
            core = C.core_sha(recipe)
            C.verify_admission(recipe, 'formal')
            runner.state['formal_training_started'] = True
            runner.save()
            code = runner.gpu(recipe, 'train')
            if training_can_resume(recipe, code):
                code = runner.gpu(recipe, 'train', 1)
            assert code == 0, 'Training failure beyond fixed exact-resume policy'
            final = exact_completion(Path(recipe['output_directory']) / 'completion.json', TRAIN_DONE, arm=arm, core=core)
            # This creates a separate wrapper from this actual final and leaves
            # the trained recipe core/checkpoint untouched.
            postfit = P.postfit(arm)
            runner.state['current_postfit_wrapper'] = P.ref(postfit['path'])
            runner.save()
            assert runner.gpu(recipe, 'postfit') == 0
            exact_completion(WORK / 'audit/functional_postfit_T32_v1' / arm / 'completion.json', POSTFIT_DONE, arm=arm, count=32)
            code = runner.gpu(recipe, 'generate')
            if generation_can_resume(recipe, code):
                code = runner.gpu(recipe, 'generate', 1)
            assert code == 0, 'Generation failure beyond fixed exact-prefix policy'
            generated = exact_completion(Path(entries[arm]['metadata_root']) / 'completion.json', GENERATE_DONE, arm=arm, core=core, count=16600)
            prefix = 'BrainHIVE_functional_v2_' + arm + '_expanded_lifecycle_' + P.TAG
            runner.cpu(prefix + '_artifact', 'audit_BrainHIVE_functional_full_V_artifacts_CPU_v1.py',
                ['--recipe', str(CONFIG / (arm + '.json')), '--output', entries[arm]['artifact_audit_root']], 1800)
            artifact = P.read(Path(entries[arm]['artifact_audit_root']) / 'summary.json')
            assert artifact['status'] == 'COMPLETE_FULL_V_ARTIFACT_AUDIT' and artifact['actual_full_V16600_verified'] is True
            assert artifact['complete_technical_coverage'] is True and artifact['full_denominator'] == 16600
            assert not artifact['failures'] and not artifact['source_errors']
            assert runner.gpu(recipe, 'features') == 0
            feature = P.read(Path(entries[arm]['feature_root']) / 'completion.json')
            assert feature['status'] == 'COMPLETED' and feature['queries'] == 16600 and feature['all_queries_used'] is True
            runner.cpu(prefix + '_evaluation', 'run_BrainHIVE_functional_evaluation_CPU_v1.py',
                ['--recipe', str(CONFIG / (arm + '.json'))], 14400)
            evaluation = exact_completion(Path(entries[arm]['metadata_root']) / 'evaluation_closure.json',
                'FUNCTIONAL_ALIGNMENT_V2_FULL16600_EVALUATION_CLOSED', arm=arm, count=16600)
            runner.state['completed'].append(dict(arm=arm, final=final, generated=generated, evaluation=evaluation))
            runner.save()
        assert [item['arm'] for item in runner.state['completed']] == list(C.ARMS)
        runner.cpu('BrainHIVE_functional_v2_expanded_lifecycle_' + P.TAG + '_fixed_comparison',
            'compare_BrainHIVE_functional_V_CPU_v2.py', ['--recipe', str(CONFIG / 'PN.json')], 1800)
        comparison = exact_completion(Path(plan['comparison']['output_root']) / 'summary.json',
            'ALL_FOUR_FULL16600_V_PAIRED_COMPARISONS_COMPLETE')
        runner.state.update(status='ALL_FOUR_FULL_FIXED_LIFECYCLES_AND_PREREGISTERED_COMPARISON_COMPLETE',
            comparison=comparison, current_stage=None, scientific_goal_complete=False,
            independent_mechanism_confirmation=False, independent_SOTA_established=False)
        runner.save()
    except BaseException as error:
        runner.state.update(status=('FORMAL_LIFECYCLE_STOP_REQUESTED_WRAPPER_STILL_LIVE' if runner.state.get('unresolved_live_child')
            else 'FORMAL_LIFECYCLE_STOPPED_WITH_PRESERVED_EVIDENCE_NO_AUTOMATIC_RETRY'), error=repr(error))
        runner.save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['profiles', 'formal'])
    args = parser.parse_args()
    P.assert_host()
    import fcntl
    with (BASE / 'lifecycle_controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        profiles() if args.mode == 'profiles' else formal()


if __name__ == '__main__':
    main()
