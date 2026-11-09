"""Exercise the actual dispatcher recipe construction without creating a review."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F

def run():
    local=Path(__file__).resolve().parent
    script=local/'audit/crossdisciplinary_20261028_v1/prepare_launch_HIVE_full1654_original_ordinary_OLD_CPU_20261101_v1.py'
    tree=ast.parse(script.read_bytes());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignments=[n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='proposal' for t in n.targets)]
    updates=[n for n in main.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
        isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='proposal' and n.value.func.attr=='update']
    assert len(assignments)==len(updates)==1
    profile=json.loads((local/'audit/HIVE_full1654_original_ordinary_profile_preparation_20261101_v1/actual/protocol.json').read_bytes())
    original=json.loads((local/'audit/HIVE_full1654_F_lifecycle_preparation_20261101_v2/actual/study.json').read_bytes())
    request=json.loads((local/'audit/crossdisciplinary_20261028_v1/HIVE_full1654_original_ordinary_prepare_launch_request_20261101_v1.json').read_bytes())
    code=PurePosixPath('/CPU_FIXTURE_ONLY/not_created/code');output=PurePosixPath('/CPU_FIXTURE_ONLY/not_created/output')
    refs={role:{kind:dict(path='/CPU_FIXTURE_ONLY/'+role+'/'+kind,bytes=1,sha256='0'*64) for kind in ('common','paper')}
          for role in ('parent','bounded_F','Euclidean_F')}
    source_pins=[dict(path=str(code/name),bytes=0,sha256=item['sha256']) for name,item in request['new_sources'].items()]
    namespace=dict(profile_protocol=profile,original=original,OUT=output,source_pins=source_pins,
        reference_csvs=refs,reference_manifests={role:dict(path='/CPU_FIXTURE_ONLY/'+role+'/manifest') for role in refs},
        chain=dict(ordinary_engineering={'CPU_FIXTURE_ONLY':True},F_evaluation={'CPU_FIXTURE_ONLY':True}))
    body=ast.fix_missing_locations(ast.Module(body=assignments+updates,type_ignores=[]))
    exec(compile(body,str(script),'exec'),namespace)
    proposal=namespace['proposal'];checks=[]
    assert proposal['status']=='PENDING_ACTUAL_COMPLETE_ORDINARY_LIFECYCLE_REVIEW'
    assert 'actual_complete_lifecycle_review' not in proposal
    checks.append('actual_dispatcher_constructs_pending_recipe_not_fake_actual_review')
    spec=importlib.util.spec_from_file_location('future_dispatcher',script)
    dispatcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(dispatcher)
    old_pin=request['recovery_protocol'];new_pin=dict(path='/CPU_FIXTURE_ONLY/input_recovery',bytes=1,sha256='1'*64)
    dispatcher.check_completed_F_recovery_pin(dict(recovery_protocol=old_pin),dict(recovery_protocol=new_pin,original_evaluation_recovery_protocol=old_pin))
    dispatcher.check_completed_F_recovery_pin(dict(recovery_protocol=old_pin),dict(recovery_protocol=old_pin))
    try:dispatcher.check_completed_F_recovery_pin(dict(recovery_protocol=old_pin),dict(recovery_protocol=new_pin))
    except AssertionError:checks.append('later_input_recovery_cannot_replace_closed_F_evaluation_provenance')
    else:raise AssertionError('Closed F recovery pin rewritten')
    fixture=dict(proposal,status=F.STATUS)
    F.validate_scope(fixture);checks.append('actual_recipe_matches_full31014_single_seed_resource_and_R80_contract')
    for name,mutation in [
        ('second_seed',lambda s:s['method'].update(seed=2027)),
        ('old132300_input_count',lambda s:s['method'].update(queries=132300)),
        ('test_payload_training',lambda s:s.update(training_R80_payloads_allowed=True)),
        ('larger_segment_quota',lambda s:s.update(maximum_training_segments=129)),
        ('missing_strong_reference',lambda s:s['reference_CSVs'].pop('bounded_F')),
        ('overwriting_oldordinary',lambda s:s.update(old_ordinary_immutable=False)),
        ('wrong_number_of_images',lambda s:s['R80'].update(queries=1999)),
        ('reduced_start_CPU_margin',lambda s:s.update(minimum_start_CPU_headroom_bytes=10<<30)),
        ('engineering_status_as_formal',lambda s:s.update(status=F.E.PROFILE_STATUS))]:
        bad=copy.deepcopy(fixture);mutation(bad)
        try:F.validate_scope(bad)
        except AssertionError:checks.append(name+'_rejected')
        else:raise AssertionError(name+' should be rejected')
    return dict(status='PASS_ACTUAL_DISPATCHER_PENDING_RECIPE_CPU_CHECKS_ONLY',checks=checks,
        source=dict(path=script.name,bytes=script.stat().st_size,sha256=hashlib.sha256(script.read_bytes()).hexdigest()),
        new_GPU_jobs=0,actual_complete_lifecycle_admission=False,
        fixture_only_no_remote_or_local_study_directories_created=True,new_R80_outcomes_read=0)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();value=run();assert not args.receipt.exists()
    args.receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');print(json.dumps(value))
