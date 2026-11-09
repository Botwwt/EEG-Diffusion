"""CPU routing/guard contracts only; no native GPU or actual image claim."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack

import generate_BrainHIVE_functional_postfit_T32_v1 as entry


def rejects(action):
    try: action()
    except (AssertionError, KeyError): return
    raise AssertionError('Invalid or premature engineering path accepted')


def positive_prepare(draft):
    """Exercise the entire real prepare body with explicit synthetic admission inputs.

Only native recipe/admission/memory facts are injected. All source/metadata,
query joins, current-final SHA reads and the actual disk-budget branch execute.
This is not a reviewed admission or a real profile/head/condition payload.
"""
    with tempfile.TemporaryDirectory(prefix='hive_postfit_prepare_CPU_') as temporary:
        base = Path(temporary); repo = base/'repo'; work = base/'work'; repo.mkdir(); work.mkdir()
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding='utf-8')
            return dict(path=str(path), sha256=entry.sha(path))
        source_pins = {}
        for name in entry.SOURCES:
            p = repo/name; p.write_text('SYNTHETIC_SOURCE_IDENTITY_ONLY\n'); source_pins[name] = entry.sha(p)
        protocol = json.loads((Path(__file__).parent/'configs/split_manifest.json').read_text())
        write(repo/'configs/split_manifest.json', protocol)
        document = copy.deepcopy(draft); sources = {}
        for role in ('oof0', 'oof1'):
            folder = work/'runs'/role; folder.mkdir(parents=True)
            cache, checkpoint = folder/'conditions.pt', folder/'brain_final.pt'
            cache.write_bytes(b'SYNTHETIC CACHE BYTES NOT A TENSOR'); checkpoint.write_bytes(b'SYNTHETIC PARENT BYTES NOT WEIGHTS')
            registration = write(folder/'registration.json', dict(training_array_rows=[]))
            mapping = [dict(array_row=row['array_row'], cache_row=row['cache_row']) for row in document['rows'] if row['source_role'] == role]
            sources[role] = dict(role=role, condition_cache_path=str(cache), condition_cache_sha256=entry.sha(cache),
                checkpoint_path=str(checkpoint), checkpoint_sha256=entry.sha(checkpoint),
                parent_registration_path=registration['path'], parent_registration_sha256=registration['sha256'],
                groups=dict(component_train=mapping))
        for row in document['rows']:
            source = sources[row['source_role']]
            for new, old in (('condition_cache', 'condition_cache'), ('parent_checkpoint', 'checkpoint'), ('parent_registration', 'parent_registration')):
                for suffix in ('_path', '_sha256'): row[new+suffix] = source[old+suffix]
        oracle = [dict(stimulus_id=row['stimulus_id'], array_row=row['array_row']) for row in document['rows']]
        oracle_binding = write(work/'oracle_inputs.json', oracle)
        document['source_selection']['oracle_inference_inputs_path'] = oracle_binding['path']
        document['source_selection']['oracle_inference_inputs_sha256'] = oracle_binding['sha256']
        query_binding = write(repo/'configs/BrainHIVE_functional_postfit_T32_queries_DRAFT_v1.json', document)
        contract = write(work/'data/component_input_contract_v1.json', dict(status='ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT',
            subjects=[dict(subject='sub-01', sources=list(sources.values()))]))
        image_manifest = repo/'audit/image_manifest.csv'; image_manifest.parent.mkdir(parents=True)
        image_manifest.write_text('dataset_split,array_row,stimulus_id,concept_id\n'+''.join(
            f"train,{row['array_row']},{row['stimulus_id']},{row['concept_id']}\n" for row in document['rows']))
        protocol['image_manifest_sha256'] = entry.sha(image_manifest); write(repo/'configs/split_manifest.json', protocol)
        main = work/'main.json'; diag = work/'diag.json'
        write(main, dict(status='ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED', pid=12345))
        write(diag, dict(status='FROZEN_FUNCTIONAL_DIAGNOSTIC_SAVED_NO_FORMAL_COMPONENT_TRAINING', pid=12346))
        terminal_pins = {str(path):entry.sha(path) for path in (main,diag)}
        recipes = {}
        formal_receipt_path = work/'formal_admission.json'
        for arm in entry.ARMS:
            recipes[arm] = dict(arm=arm, candidate='functional_alignment_v2_'+arm, source_sha256=source_pins,
                input_contract_sha256=contract['sha256'], input_sha256={str(image_manifest):entry.sha(image_manifest)},
                strength_policy='Dfit_only_global_strength_calibration', output_directory=(work/'components/functional_alignment_v2'/arm).as_posix(),
                admission_receipt=str(formal_receipt_path), admission_sha256='pending', formal_job_names=[arm+'_training'], profile_job_name=arm+'_profile')
        lifecycle = write(work/'lifecycle.json', dict(recipes=[dict(arm=arm, recipe_core_sha256=entry.core_sha(recipes[arm])) for arm in entry.ARMS]))
        review = write(work/'functional_review.json', dict(actual_GPU_diagnostic=True, native_fixed_Q_reused=True,
            teacher_response_nondegenerate=True, EEG_pairing_signal_present=True))
        profiles = [dict(arm=arm, **write(work/(arm+'_profile.json'), dict(actual_GPU=True, engineering_checks_passed=True,
            actual_parent_Q_engineering_verified=True))) for arm in entry.ARMS]
        formal_receipt = write(formal_receipt_path, dict(functional_prerequisite_review=review, actual_GPU_profiles=profiles, four_arm_lifecycle_plan=lifecycle))
        for recipe in recipes.values(): recipe['admission_sha256'] = formal_receipt['sha256']
        formal_bindings = [dict(arm=arm, **write(repo/'configs'/('formal_'+arm+'.json'), recipe)) for arm,recipe in recipes.items()]
        own = recipes['O']; folder = Path(own['output_directory'])
        calibration = dict(kind=own['strength_policy'], strengths=[0.,.25,.5,.75,1.], Dfit_token_errors=[.5,.4,.3,.35,.36],
            selected_strength=.5, selected_without_V_or_test_images=True)
        cal = write(folder/'calibration.json', calibration)
        prov = write(folder/'input_provenance.json', dict(recipe_core_sha256=entry.core_sha(own), arm='O', source_sha256=source_pins,
            input_contract_sha256=contract['sha256'], risk_head_or_gate=False, all_frozen_host_weights=True))
        checkpoint = folder/'component_final.pt'; checkpoint.write_bytes(b'SYNTHETIC CURRENT FINAL, NEVER TORCH LOADED')
        done = write(folder/'completion.json', dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE',
            arm='O', candidate=own['candidate'], updates=24807, seen_queries=396900, passes=3, subjects=10, parameters=525568,
            explicit_UNet_forwards_per_update=2, recipe_core_sha256=entry.core_sha(own), checkpoint_path=checkpoint.as_posix(),
            checkpoint_sha256=entry.sha(checkpoint), calibration_sha256=cal['sha256'], input_provenance_sha256=prov['sha256']))
        input_pins = {item['path']:item['sha256'] for item in (oracle_binding,contract,cal,prov,done)}
        input_pins.update({str(checkpoint):entry.sha(checkpoint)})
        for source in sources.values():
            for a,b in [('condition_cache_path','condition_cache_sha256'),('checkpoint_path','checkpoint_sha256'),('parent_registration_path','parent_registration_sha256')]:
                input_pins[source[a]]=source[b]
        plan = dict(schema=entry.SCHEMA,status=entry.READY,arm='O',candidate=own['candidate'],generator=entry.GENERATOR,training_seed=1729,
            subject='sub-01',queries=32,maximum_seconds=1200,maximum_arm_output_bytes=64<<20,maximum_four_arm_output_bytes=256<<20,
            output_directory=str(work/'audit/functional_postfit_T32_v1/O'),query_manifest=query_binding,formal_recipes=formal_bindings,
            source_sha256=source_pins,job_name='O_fixed_postfit_only',engineering_only=True,semantic_evaluation_enabled=False,automatic_retry=False,input_sha256=input_pins)
        admission = write(work/'postfit_admission.json', dict(schema='BrainHIVE_functional_postfit_T32_v1_reviewed_admission',
            status='CURRENT_ARM_FINAL_POSTFIT_T32_ENGINEERING_ADMITTED',arm='O',plan_core_sha256=entry.plan_core(plan),
            reviewed_internal_engineering_facts=True,image_gain_claimed=False,issued_epoch=entry.time.time()-1,expires_epoch=entry.time.time()+120,
            terminal_state_sha256=terminal_pins,maximum_seconds=1200,maximum_arm_output_bytes=64<<20,maximum_four_arm_output_bytes=256<<20,
            final_completion_sha256=done['sha256'],resource_limits=dict(minimum_CPU_headroom_bytes=8<<30,minimum_free_GPU_bytes=16<<30)))
        plan['postfit_admission']=admission
        reads=[]; actual_read=entry.read
        def tracked_read(path): reads.append(Path(path)); return actual_read(path)
        def local_bound(item):
            assert entry.sha(item['path'])==item['sha256']; return tracked_read(item['path'])
        checked_recipes=[]
        def recipe_check(recipe): checked_recipes.append(recipe['arm']); return entry.core_sha(recipe)
        formal = dict(remaining_storage=[dict(root=str(work), required=123456)])
        with ExitStack() as stack:
            for name,value in [('ROOT',repo),('REPO',repo),('WORK',work),('MAIN',main),('DIAG',diag),('BASE',work/'audit/functional_postfit_T32_v1'),('QUERY_SHA',query_binding['sha256'])]:
                stack.enter_context(patch.object(entry,name,value))
            stack.enter_context(patch.object(entry,'read',tracked_read)); stack.enter_context(patch.object(entry,'bound_read',local_bound))
            stack.enter_context(patch.object(entry,'process_live',return_value=False))
            stack.enter_context(patch.object(entry,'validate_recipe',recipe_check))
            stack.enter_context(patch.object(entry,'verify_admission',return_value=formal))
            stack.enter_context(patch.object(entry,'resource_headroom',return_value=dict(effective_headroom_bytes=16<<30)))
            usage = stack.enter_context(patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=(256<<20)+(64<<20)+123456+4096)))
            result=entry.prepare(plan)
            assert result['completion']['arm']=='O' and result['formal_check']['remaining_storage'][0]['required']==123456
            assert set(checked_recipes)==set(entry.ARMS) and usage.call_count==1
            assert not Path(plan['output_directory']).exists()
            assert all(not Path(recipes[arm]['output_directory']).exists() for arm in ('P','PN','S'))
            assert not any('/components/functional_alignment_v2/'+arm+'/' in p.as_posix() for p in reads for arm in ('P','PN','S'))
            usage.return_value=SimpleNamespace(free=123455)
            rejects(lambda:entry.prepare(plan))


def check():
    assert 'torch' not in sys.modules
    draft_path = Path(__file__).parent / 'configs/BrainHIVE_functional_postfit_T32_queries_DRAFT_v1.json'
    draft = json.loads(draft_path.read_text())
    with patch.object(entry, 'terminal_check', side_effect=AssertionError('No terminal or GPU work before DRAFT rejection')) as terminal, \
         patch.object(entry, 'execute', side_effect=AssertionError('No execution before DRAFT rejection')) as execute:
        rejects(lambda: entry.run(draft_path))
        terminal.assert_not_called(); execute.assert_not_called()
    assert 'torch' not in sys.modules
    positive_prepare(draft)
    assert 'torch' not in sys.modules
    for row in draft['rows']:
        identity, seed = entry.query_identity('sub-01', row, draft['protocol_version'])
        assert row['query_id'] == identity and row['image_seed'] == seed
    assert draft['rows'][0]['image_seed'] == 5533825044790129053
    rejects(lambda: entry.strict('UNKNOWN'))
    source = ast.parse(Path(entry.__file__).read_text())
    top_imports = [node for node in source.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any('torch' in ast.unparse(node) or 'brainhive_component_generator' in ast.unparse(node) for node in top_imports)
    run_node = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    calls = [node.func.id for node in ast.walk(run_node) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert calls.index('prepare') < calls.index('exclusive_wrapper') < calls.index('execute')
    execute_node = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    text = ast.unparse(execute_node)
    assert text.index('check_state(') < text.index('component.cuda()') < text.index('load_native(training=False)')
    assert 'sample_native(' in text and 'enabled=False' not in text
    # Only synthetic CPU tensors below this point. No native models/assets loaded.
    import torch
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    torch.set_num_threads(1); torch.manual_seed(1729)
    sources, payloads = {}, {}
    x = torch.sin(torch.arange(1024).float() * .11)[None]
    for index, role in enumerate(('oof0', 'oof1')):
        cache, checkpoint = role + '_cache', role + '_checkpoint'
        sources[role] = dict(condition_cache_path=cache, checkpoint_path=checkpoint, checkpoint_sha256=role + '_SHA')
        payloads[cache] = dict(model_sha256=role + '_SHA', array_rows=torch.tensor([900 + index, 1900 + index]),
            condition=torch.cat([x + .01 * index, x + .02 * index]))
        payloads[checkpoint] = dict(model={'block.layers.0.norm.weight': torch.full((1024,), 1. + index),
            'block.layers.0.norm.bias': torch.full((1024,), .1 * index)})
    loaded_paths = []
    def reader(path, **kwargs):
        assert kwargs['weights_only'] is True and kwargs['map_location'] == 'cpu'
        loaded_paths.append(path)
        return payloads[path]
    loaded = entry.load_oof(sources, reader)
    assert set(loaded_paths) == set(payloads)
    query = dict(source_role='oof1', cache_row=1, array_row=1901)
    condition, weight, bias = entry.route(query, loaded)
    assert torch.equal(condition, payloads['oof1_cache']['condition'][1:2])
    assert torch.equal(weight, torch.full((1, 1024), 2.)) and torch.equal(bias, torch.full((1, 1024), .1))
    rejects(lambda: entry.route(dict(query, cache_row=1901), loaded))
    rejects(lambda: entry.route(dict(query, array_row=900), loaded))
    head = BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False); head.radius = .125
    with torch.no_grad(): head.direction.bias.copy_(torch.cos(torch.arange(1024).float() * .03))
    recorded = []
    class RecordingHead:
        def __call__(self, c, w, b, **kwargs):
            recorded.append((w.clone(), b.clone(), kwargs))
            return head(c, w, b, **kwargs)
    def adapter(value):
        assert value.dtype == torch.float16 and value.shape == (1, 1024)
        return value
    def projector(value): return value.repeat(1, 8).reshape(1, 4, 2048).contiguous()
    with torch.inference_mode():
        raw, native, tokens, info = entry.encode(torch, RecordingHead(), adapter, projector, condition, weight, bias, device='cpu')
        wrong, _, _, _ = entry.encode(torch, head, adapter, projector, condition,
            loaded['oof0']['weight'], loaded['oof0']['bias'], device='cpu')
    assert torch.equal(recorded[0][0], weight) and torch.equal(recorded[0][1], bias)
    assert recorded[0][2] == {'gate_mode': 'fixed'} and torch.all(info['gate'] == 1)
    assert not torch.allclose(raw, wrong, atol=1e-6), 'Own-affine check must exercise an active correction'
    assert native.dtype == tokens.dtype == torch.float16
    # Inspect actual native-call arguments without allocating CUDA or making pixels.
    generators, invocations = [], []
    class Generator:
        def __init__(self, device): assert device == 'cuda'; generators.append(self)
        def manual_seed(self, seed): self.seed = seed; return self
    def pipe(**kwargs): invocations.append(kwargs); return SimpleNamespace(images=['NO_PIXEL_INJECTION_SENTINEL'])
    fake_torch = SimpleNamespace(Generator=Generator)
    for _ in range(2):
        assert entry.sample_native(pipe, ('prompt', 'pooled'), tokens, draft['rows'][0]['image_seed'], fake_torch) == 'NO_PIXEL_INJECTION_SENTINEL'
    assert generators[0] is not generators[1] and generators[0].seed == generators[1].seed == 5533825044790129053
    for arguments in invocations:
        assert arguments['height'] == arguments['width'] == 512 and arguments['num_inference_steps'] == 4
        assert arguments['guidance_scale'] == 0 and arguments['cross_attention_kwargs']['ip_hidden_states'] is tokens
        assert set(arguments) == {'height', 'width', 'generator', 'prompt_embeds', 'pooled_prompt_embeds', 'cross_attention_kwargs', 'num_inference_steps', 'guidance_scale', 'output_type'}
    # Exercise the real final-state contract without inventing another arm's head.
    recipe = dict(arm='O', candidate='functional_alignment_v2_O', output_directory=str(entry.WORK / 'components/functional_alignment_v2/O'),
        source_sha256={'actual_source': 'pinned'}, input_contract_sha256='contract', strength_policy='Dfit_only_global_strength_calibration')
    calibration = dict(kind=recipe['strength_policy'], strengths=[0., .25, .5, .75, 1.],
        Dfit_token_errors=[.5, .4, .3, .35, .36], selected_strength=.5, selected_without_V_or_test_images=True)
    done = dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE', arm='O', candidate=recipe['candidate'],
        updates=24807, seen_queries=396900, passes=3, subjects=10, parameters=525568, explicit_UNet_forwards_per_update=2,
        recipe_core_sha256=entry.core_sha(recipe), checkpoint_path=recipe['output_directory']+'/component_final.pt', input_provenance_sha256='provenance')
    provenance = dict(recipe_core_sha256=entry.core_sha(recipe), arm='O', source_sha256=recipe['source_sha256'],
        input_contract_sha256='contract', risk_head_or_gate=False, all_frozen_host_weights=True)
    entry.final_metadata(recipe, done, calibration, provenance)
    rejects(lambda: entry.final_metadata(recipe, dict(done, updates=1024), calibration, provenance))
    rejects(lambda: entry.final_metadata(recipe, done, dict(calibration, selected_strength=1.), provenance))
    rejects(lambda: entry.final_metadata(recipe, done, calibration, dict(provenance, all_frozen_host_weights='UNKNOWN')))
    state = dict(arm='O', candidate=recipe['candidate'], recipe_core_sha256=entry.core_sha(recipe), seed=1729,
        updates=24807, seen_queries=396900, passes=3, risk_gate=False, trained_from_scratch=True,
        calibration=calibration, input_provenance_sha256='provenance')
    entry.check_state(state, recipe, done, calibration)
    rejects(lambda: entry.check_state(dict(state, arm='PN'), recipe, done, calibration))
    folder = Path(recipe['output_directory'])
    done.update(calibration_sha256='calibration_SHA', checkpoint_sha256='final_SHA')
    small_files = {str(folder/'completion.json'): done, str(folder/'calibration.json'): calibration,
        str(folder/'input_provenance.json'): provenance}
    digests = {str(folder/'completion.json'): 'completion_SHA', str(folder/'calibration.json'): 'calibration_SHA',
        str(folder/'input_provenance.json'): 'provenance', str(Path(done['checkpoint_path'])): 'final_SHA'}
    opened = []
    def own_read(path):
        opened.append(str(path)); return small_files[str(path)]
    with patch.object(entry, 'read', own_read), patch.object(entry, 'sha', side_effect=lambda p: digests[str(p)]):
        entry.current_final(dict(input_sha256=digests), recipe, dict(final_completion_sha256='completion_SHA'))
    assert set(opened) == set(small_files) and not any('/P/' in p or '/PN/' in p or '/S/' in p for p in opened)
    with tempfile.TemporaryDirectory(prefix='hive_postfit_contract_') as temporary:
        base = Path(temporary); out = base / 'O'; out.mkdir()
        with patch.object(entry, 'BASE', base), patch.object(entry, 'ARM_BYTES', 16), patch.object(entry, 'TOTAL_BYTES', 32), patch.object(entry, 'METADATA_RESERVE', 0):
            entry.save_bytes(out, out/'prefix', b'one\n', append=True)
            entry.save_bytes(out, out/'prefix', b'two\n', append=True)
            rejects(lambda: entry.save_bytes(out, out/'too_large', b'x'*17))
            assert (out/'prefix').read_bytes() == b'one\ntwo\n' and not (out/'too_large').exists()
    return dict(schema='BrainHIVE_functional_postfit_T32_CPU_contract_v1', status='PASS_CPU_AST_AND_INJECTED_ROUTING_ONLY',
        source_sha256={Path(entry.__file__).name: entry.sha(entry.__file__), Path(__file__).name: entry.sha(__file__)},
        checks=['actual_DRAFT_rejected_before_Torch_or_outputs', 'all32_frozen_seeds_match_query_identity',
            'positive_full_prepare_injected_metadata_reaches_real_disk_budget_return', 'unfunded_full_lifecycle_storage_rejected',
            'UNKNOWN_boolean_refused', 'admission_wrapper_before_runtime_AST', 'final_state_before_CUDA_AST',
            'real_OOF_cache_row_not_array_or_T_position', 'own_OOF_checkpoint_affine_active_correction',
            'FP32_head_to_FP16_native_tokens', 'injected_native512_fourNFE_CFG0_same_query_generator',
            'current_arm_only_fixed24807_final_and_existing_Dfit_grid', 'no_other_arm_final_file_reads',
            'wrong_arm_head_rejected', 'finite_bytes_and_failure_prefix'],
        synthetic_CPU_tensors_only=True, actual_native_UNet_executed=False, actual_images_generated=0,
        actual_pixel_equivalence_verified=False, GPU_used=False, real_EEG_or_condition_caches_loaded=False,
        future_heads_assumed_present=False, runtime_admission_issued=False, real_1200s_budget_profiled=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output'); args = parser.parse_args()
    result = check()
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream: stream.write(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__': main()
