"""Real historical reference rows, explicit synthetic replay records, schema refusals."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from audit_HIVE_training_mean_R80_schema_repair_CPU_20261101_v4 import check_replay_reference_fields,target_binding_fields,write_own_audit_result

L=Path(__file__).resolve().parent;A=L/'audit/crossdisciplinary_20261028_v1'


def reject(call):
    try:call()
    except (AssertionError,KeyError,ValueError):return
    raise AssertionError('Invalid schema evidence accepted')


def main(output):
    probe=json.loads(json.loads((A/'RAMroutes_old_mean_recovery_real_schema_deadline_20261101_v2_20261101T194643516021Z.json').read_bytes())['stdout'])
    parent=probe['actual_native_first_row']
    original=L/'audit/HIVE_full1654_F_complete_20261101_v1'
    source=json.loads((original/'actual/bounded_F/loss_control/manifest.jsonl').open(encoding='utf-8').readline())
    gallery=L/'audit/HIVE_full1654_original_ordinary_complete_20261101_v1/references/gallery_manifest.jsonl'
    target=json.loads(gallery.open(encoding='utf-8').readline())
    assert all(key not in parent for key in ('initial_noise_latents_sha256','ancestral_noise_observations','target_image_sha256','concept_id'))
    unchanged=json.dumps(parent,sort_keys=True)
    contract=L/'audit/HIVE_training_mean_R80_recovery_preparation_20261101_v2/actual/brainhive_training_mean_R80_contract_CPU_20261101_v1.py'
    tree=ast.parse(contract.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='noise_reference')
    namespace={'json':json};exec(compile(ast.Module(body=[function],type_ignores=[]),str(contract),'exec'),namespace)
    reference=namespace['noise_reference'];checks=[]
    pin=dict(path='/EXPLICIT_SYNTHETIC/actual_bound_source_manifest',bytes=1,sha256='a'*64)
    for label in ('native_replay','source_head_replay','disabled_replay','zero_head_replay'):
        image=source if label=='source_head_replay' else parent
        item=dict(index=0,path=label,pixel_exact=True,expected_image_path=image['generated_path'],
            expected_file_SHA256=image['generated_sha256'],noise_reference_manifest=pin,noise_reference_query_id=source['query_id'],
            UNet_NFE=4,native_IP_K_calls=280,native_IP_V_calls=280,
            **{key:copy.deepcopy(source[key]) for key in ('initial_noise_latents_sha256','ancestral_noise_observations')})
        check_replay_reference_fields(item,0,label,image,source,pin,reference)
        for key,value in [('index',1),('path','bad'),('pixel_exact',False),('expected_image_path','bad'),
            ('expected_file_SHA256','b'*64),('noise_reference_manifest',dict(pin,sha256='b'*64)),('noise_reference_query_id','bad'),
            ('UNet_NFE',8),('native_IP_K_calls',279),('native_IP_V_calls',279),('initial_noise_latents_sha256','b'*64)]:
            bad=copy.deepcopy(item);bad[key]=value
            reject(lambda:check_replay_reference_fields(bad,0,label,image,source,pin,reference))
        contradiction=copy.deepcopy(image);contradiction['initial_noise_latents_sha256']='b'*64
        reject(lambda:check_replay_reference_fields(item,0,label,contradiction,source,pin,reference))
        checks.append(label+'_real_old_image_and_separate_noise_source_pass_twelve_refusals')
    fields=target_binding_fields(parent,source,target)
    assert fields['target_image_sha256']==target['image_sha256'] and fields['concept_id']==source['concept_id']
    assert json.dumps(parent,sort_keys=True)==unchanged
    checks.append('real_target_gallery_and_bound_full_head_supply_missing_old_fields_without_mutating_native_manifest')
    for key,value in [('target_image_sha256','b'*64),('concept_id','wrong')]:
        changed=copy.deepcopy(parent);changed[key]=value
        reject(lambda:target_binding_fields(changed,source,target))
    for key,value in [('stimulus_id','wrong'),('target_image_sha256','b'*64),('target_image_path','wrong'),('score_target_path','wrong')]:
        changed=copy.deepcopy(source);changed[key]=value
        reject(lambda:target_binding_fields(parent,changed,target))
    checks.append('contradictory_existing_native_target_or_concept_and_wrong_source_target_binding_refused')
    old=L/'audit/HIVE_training_mean_R80_recovery_preparation_20261101_v2/actual/audit_HIVE_training_mean_R80_CPU_20261101_v1.py'
    new=L/'audit_HIVE_training_mean_R80_schema_repair_CPU_20261101_v4.py'
    extract=lambda p:ast.dump(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='constant_geometry_numpy'),include_attributes=False)
    assert extract(old)==extract(new)
    checks.append('independent_geometry_function_AST_identical_to_original_registered_auditor')
    import tempfile
    with tempfile.TemporaryDirectory(dir=output.parent if output.parent.exists() else L/'audit') as directory:
        root=Path(directory);target_file=root/'artifact_audit.json'
        write_own_audit_result(target_file,dict(status='EXPLICIT_SYNTHETIC_WRITER_ONLY'),root)
        assert json.loads(target_file.read_bytes())['status']=='EXPLICIT_SYNTHETIC_WRITER_ONLY'
        reject(lambda:write_own_audit_result(root/'wrong.json',{},root))
        reject(lambda:write_own_audit_result(target_file,{},root/'elsewhere'))
    checks.append('new_audit_directory_writer_round_trip_and_outside_root_refusals_do_not_use_legacy_OUT_guard')
    result=dict(status='PASS_CPU_ONLY_ARTIFACT_SCHEMA_REPAIR_LOCAL_REFERENCE_AND_REFUSAL_CHECKS',checks=checks,
        historical_native_source_and_gallery_rows=3,replay_receipts_explicitly_synthetic=True,
        native_original_fields_not_fabricated=True,old_pinned_auditor_SHA=hashlib.sha256(old.read_bytes()).hexdigest(),
        revised_auditor=dict(path=str(new),bytes=new.stat().st_size,sha256=hashlib.sha256(new.read_bytes()).hexdigest()),
        real_full2000_artifacts_not_yet_reaudited=True,new_GPU_jobs=0,new_optimizer_updates=0,new_images=0)
    output=Path(output);assert not output.exists();output.parent.mkdir(exist_ok=True,parents=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
