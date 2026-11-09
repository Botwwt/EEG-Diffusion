"""Exercise exact manifest installation and writer scope; no fake efficacy claims."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import types

L=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_bytes())


def ref(path):
    raw=Path(path).read_bytes();return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def checked(pin):assert ref(pin['path'])==pin;return Path(pin['path'])


def reject(call):
    try:call()
    except (AssertionError,KeyError,ValueError):return
    raise AssertionError('Expected remaining-controller refusal')


def functions(path,names,namespace):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


def main(output):
    output=Path(output);assert not output.exists();output.parent.mkdir(exist_ok=True,parents=True)
    controller=L/'queue_HIVE_training_mean_remaining_lifecycle_CPU_20261101_v3.py';tree=ast.parse(controller.read_text())
    checks=[];source=L/'audit/HIVE_training_mean_R80_recovery_preparation_20261101_v2/actual/brainhive_training_mean_R80_contract_CPU_20261101_v1.py'
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='EXPLICIT_SYNTHETIC_INSTALL_') as directory:
        root=Path(directory);legacy=root/'old_generation';legacy.mkdir();coord=root/'remaining';coord.mkdir();code=root/'code';code.mkdir();(code/'protocol.json').write_text('{}')
        old_write={'Path':Path,'OUT':legacy,'json':json,'os':__import__('os')};functions(source,{'write'},old_write)
        m=types.SimpleNamespace(OUT=legacy,ref=ref,checked=checked,read=read,write=old_write['write'],process=lambda *args:dict(exact_live=False))
        namespace={'Path':Path,'M':m,'COORD_CODE':code,'COORD_OUT':coord,'os':__import__('os')}
        functions(controller,{'write_own','install_first_role'},namespace)
        first=legacy/'mean_bounded_F';first.mkdir();cpu=root/'completed_CPU';cpu.mkdir()
        manifest=cpu/'manifest.jsonl';manifest.write_bytes(b''.join((json.dumps(dict(EXPLICIT_SYNTHETIC_INSTALL_ONLY=True,index=i))+'\n').encode() for i in range(2000)))
        generation=cpu/'generation_protocol.json';generation.write_text('{}');auditor=cpu/'auditor.py';auditor.write_text('# SYNTHETIC ONLY\n')
        descriptor=dict(status='PASS_TRAIN_MEAN_ALL2000_R80_ARTIFACTS_AND_NUMPY_GEOMETRY',queries=2000,role='mean_bounded_F',
            protocol=ref(generation),audit_source=ref(auditor),all2048000_generation_condition_values_bit_exact_with_profile=True,
            all_noise_equal_own_source_head=True,all_new_file_pixel_identities_verified=True,
            profile50_and_segment20_engineering_image_files_independently_checked=True,manifest=ref(manifest),SYNTHETIC_FIXTURE_ONLY=True)
        artifact=cpu/'artifact_audit.json';artifact.write_text(json.dumps(descriptor));completion=cpu/'completion.json'
        completion.write_text(json.dumps(dict(status='COMPLETE_CPU_ONLY_REAUDIT_EXISTING2000_ARTIFACTS_NOT_NEW_EFFICACY',exit_code=0,artifact_audit=ref(artifact))))
        marker=cpu/'other_pin.json';marker.write_text('{}')
        q=dict(required_exited_handles=[dict(PID=1,start_ticks=2)],first_role_CPU_completion=ref(completion),
            completed_first_role_auditor=ref(auditor),generation_protocol=ref(generation),
            first_role_generation_completions=[ref(marker)],first_role_profile=ref(marker),preserved_generation_failure=ref(marker))
        called=[]
        m.process=lambda *args:dict(exact_live=True)
        original_checked=m.checked;m.checked=lambda p:called.append(p)
        reject(lambda:namespace['install_first_role'](q));assert called==[]
        m.process=lambda *args:dict(exact_live=False);m.checked=original_checked
        checks.append('live_completed_CPU_handle_rejected_before_reading_or_writing_any_installation_data')
        raw=manifest.read_bytes();manifest.write_bytes(raw+b'bad')
        reject(lambda:namespace['install_first_role'](q));assert not (first/'loss_control').exists();manifest.write_bytes(raw)
        checks.append('tampered_source_manifest_refused_before_destination_creation')
        receipt=namespace['install_first_role'](q)
        installed=first/'loss_control/manifest.jsonl';assert installed.read_bytes()==raw and manifest.read_bytes()==raw
        result=read(first/'loss_control/artifact_audit.json')
        assert result['installed_from_completed_CPU_audit']==ref(artifact) and result['installed_manifest_source']==ref(manifest)
        assert receipt['new_GPU_jobs']==receipt['new_images']==0 and receipt['reused_research_images']==2000
        checks.append('explicit_SYNTHETIC_all2000_row_installation_preserves_source_bytes_and_records_real_source_copy_provenance')
        before=installed.read_bytes();reject(lambda:namespace['install_first_role'](q));assert installed.read_bytes()==before
        checks.append('already_installed_destination_refused_without_overwriting_either_manifest')
        namespace['write_own'](coord/'writer_test.json',dict(SYNTHETIC_FIXTURE_ONLY=True))
        reject(lambda:namespace['write_own'](legacy/'outside_new_writer.json',{}))
        reject(lambda:m.write(coord/'outside_legacy_writer.json',{}))
        checks.append('new_and_legacy_metadata_writer_scopes_both_enforced_and_new_writer_roundtrip_passes')
    # Confirm the active generation loop is only the previously unstarted role.
    tuples=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='role' and isinstance(n.iter,ast.Tuple)]
    assert len(tuples)==1 and [v.value for v in tuples[0].iter.elts]==['mean_ordinary_full1654']
    evaluated=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='role' and isinstance(n.iter,ast.Attribute) and n.iter.attr=='ROLES']
    assert len(evaluated)==1
    checks.append('only_unstarted_second_role_generates_but_both_roles_receive_original_full_evaluation')
    legacy_writes=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id=='M' and n.func.attr=='write']
    for node in legacy_writes:
        names={n.id for n in ast.walk(node.args[0]) if isinstance(n,ast.Name)}
        assert not names&{'logs','COORD_CODE','COORD_OUT'}
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='write_own'
        and any(isinstance(v,ast.Name) and v.id=='logs' for v in ast.walk(n.args[0])) for n in ast.walk(tree))
    checks.append('new_controller_event_logs_use_new_writer_and_no_legacy_write_targets_new_metadata_roots')
    old=L/'audit_HIVE_training_mean_R80_schema_repair_CPU_20261101_v4.py';new=L/'audit_HIVE_training_mean_R80_schema_repair_CPU_20261101_v5.py'
    def body(p,name):return ast.dump(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    for name in ('constant_geometry_numpy','check_replay_reference_fields','target_binding_fields','write_own_audit_result'):assert body(old,name)==body(new,name)
    checks.append('second_role_auditor_preserves_actual_passed_v4_geometry_noise_target_and_writer_functions_exactly')
    # Verify the other role against real ordinary source fields as well.
    namespace={};functions(new,{'target_binding_fields'},namespace)
    actual=L/'audit/HIVE_full1654_original_ordinary_complete_20261101_v1'
    source_row=json.loads((actual/'actual/ordinary_full1654/loss_control/manifest.jsonl').open(encoding='utf-8').readline())
    target=json.loads((actual/'references/gallery_manifest.jsonl').open(encoding='utf-8').readline())
    baseline=dict(stimulus_id=source_row['stimulus_id']);mapped=namespace['target_binding_fields'](baseline,source_row,target)
    assert mapped['target_image_sha256']==target['image_sha256'] and 'target_image_sha256' not in baseline
    checks.append('actual_ordinary_source_and_gallery_bind_without_fabricating_missing_native_fields')
    value=dict(status='PASS_REMAINING_MEAN_CONTROLLER_REUSE_SCOPE_AND_SCHEMA_CHECKS',checks=checks,
        copy_fixture_explicitly_synthetic=True,real_GPU_or_new_images_used=False,
        source_pins=[ref(controller),ref(new)],scientific_recipe_or_old_source_changed=False)
    with output.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(value))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
