"""Exercise the real consumer against full-size metadata and boundary examples."""
import copy,gzip,hashlib,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
import brainhive_exposed_fullfit_component_inputs_CPU_20261101_v1 as I


def run():
    root=Path(__file__).resolve().parent
    plan=root/'audit/official_fullfit_metadata_plan_20261022_v1'
    task_path=plan/'official_fullfit_encoder_tasks.json'
    tasks={t['task_id']:t for t in json.loads(task_path.read_bytes())['tasks']}
    ids=I.expected_affine_ids(tasks,np);checks=['full165400_consumer_affine_map']
    runtime=json.loads((root/'audit/diagnostic_memory_recovery_20261025_v3/isolated_runtime_manifest.json').read_bytes())
    I.validate_runtime_owner(runtime,str(Path(runtime['path'])/'accelerate/__init__.py'),'0.34.2');checks.append('actual_manifest_identifies_accelerate_not_diffusers_owner')
    for label,path,version in [('wrong_diffusers_owner_rejected',str(Path(runtime['path'])/'diffusers/__init__.py'),'0.34.2'),('wrong_accelerate_version_rejected',str(Path(runtime['path'])/'accelerate/__init__.py'),'0.34.1')]:
        try:I.validate_runtime_owner(runtime,path,version)
        except AssertionError:checks.append(label)
        else:raise AssertionError(label)
    querymap=plan/'official_fullfit_OOF_query_map.jsonl.gz'
    keys=sorted(tasks);count=0
    with gzip.open(querymap,'rt',encoding='utf8') as stream:
        for n,line in enumerate(stream):
            q=json.loads(line)
            assert q['query_index']==n and q['array_row']==n%16540 and q['subject']==I.SUBJECTS[n//16540]
            assert keys[int(ids[n])]==q['encoder_task_id'];count+=1
    assert count==165400;checks.append('all165400_original_query_map_indices_match_consumer')
    pin=dict(path='/synthetic/schema_only_protocol.json',bytes=1,sha256='0'*64)
    c=dict(status=I.STATUS,protocol=pin,subjects=10,concepts=1654,images=16540,queries=165400,raw_shape=[165400,1024],training_seed=1729,
        official200_previously_exposed=True,independent_H=False,official_test_payloads_opened=False,final_component_or_benchmark_admitted=False,
        all20_folds_and_all165400_join_values_verified=True,query_order='subject sub-01..sub-10, then exact global array_row0..16539',folds=[dict(task_id=k) for k in keys])
    assert I.validate_header(c,pin)==keys
    for label,change in [('legacy132300_rejected',dict(queries=132300)),('fake_unexposed_rejected',dict(official200_previously_exposed=False)),
        ('independent_H_claim_rejected',dict(independent_H=True)),('test_payload_access_rejected',dict(official_test_payloads_opened=True)),
        ('component_admission_claim_rejected',dict(final_component_or_benchmark_admitted=True)),('incomplete_fold_list_rejected',dict(folds=c['folds'][:-1]))]:
        invalid=copy.deepcopy(c);invalid.update(change)
        try:I.validate_header(invalid,pin)
        except AssertionError:checks.append(label)
        else:raise AssertionError(label)
    for label,field in [('row_overlap_rejected','training_array_rows'),('concept_overlap_rejected','training_concepts')]:
        bad=copy.deepcopy(tasks);t=bad[keys[0]];t[field][0]=t['exported_OOF_array_rows' if field.endswith('rows') else 'held_concepts'][0]
        try:I.expected_affine_ids(bad,np)
        except AssertionError:checks.append(label)
        else:raise AssertionError(label)
    # Markers keep all1024 coordinates and165400 query identities without huge allocation.
    raw=np.broadcast_to(np.arange(165400,dtype=np.float32)[:,None],(165400,1024))
    weight=np.broadcast_to(np.arange(20,dtype=np.float32)[:,None],(20,1024))
    bias=weight+100;target=torch.arange(16540,dtype=torch.float32)[:,None].expand(16540,1024)
    inputs=dict(condition=raw,affine_ids=ids,weight=weight,bias=bias,target=target)
    indices=np.array([0,8270,16539,16540,33079,165399],dtype=np.int64);batch=I.cpu_batch(inputs,indices)
    assert np.array_equal(batch['condition'][:,0].numpy(),indices.astype(np.float32))
    assert np.array_equal(batch['target'][:,0].numpy(),indices%16540)
    assert np.array_equal(batch['weight'][:,0].numpy(),ids[indices])
    assert np.array_equal(batch['bias'][:,0].numpy(),ids[indices]+100)
    assert np.array_equal(batch['subject_index'],indices//16540)
    checks.append('subject_boundary_source_affine_stimulus_target_and_raw_scale_batch_identity')
    for label,bad in [('negative_index_rejected',[-1]),('test_or_out_of_range_index_rejected',[165400]),('fractional_index_rejected',[.5]),('empty_batch_rejected',[])]:
        try:I.cpu_batch(inputs,bad)
        except AssertionError:checks.append(label)
        else:raise AssertionError(label)
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_FULL165400_METADATA_AND_CONSUMER_BOUNDARY_CHECKS',checks=checks,count=len(checks),query_map_rows=count,
        sources=[I.ref(Path(__file__)),I.ref(Path(I.__file__))],metadata=[I.ref(task_path),I.ref(querymap)],
        full_model_or_condition_payloads_loaded=False,fixture_header_not_admission=True,CUDA_initialized=False,new_training_updates=0,new_images=0)
    out=root/'audit/HIVE_exposed_fullfit_component_inputs_preparation_20261101_v3';out.mkdir(exist_ok=False)
    (out/'actual_local_CPU_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
