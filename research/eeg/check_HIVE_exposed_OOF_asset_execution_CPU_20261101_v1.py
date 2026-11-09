"""Numerical/reference/schema/full-query-map checks, not learned model results."""
import argparse,ast,copy,gzip,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import brainhive_exposed_OOF_assets_contract_CPU_20261101_v1 as G

def main(path,references):
    s=G.read(path);refs=G.read(references);tasks,_=G.load_scope(s)
    sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']));sys.path.insert(0,str(Path(path).parent))
    import numpy as np,torch
    import torch.nn.functional as F
    import audit_HIVE_exposed_OOF_assets_CPU_20261101_v1 as A
    torch.set_num_threads(2);checks=[]
    profile=G.read(G.checked(refs['profile']));assert profile['status']=='PASS_ACTUAL_TRAIN_ONLY_OOF_ASSET_ENGINEERING' and profile['temporary_updates']==9 and not profile['learned_weights_saved'] and profile['final_model'] is None
    checks.append('actual9_update_profile_is_not_an_admissible_final_fold')
    source=G.checked(refs['validated_native_manual']).read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='manual_EEG')
    scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen_previous_native_manual>','exec'),scope)
    torch.manual_seed(1729)
    w={'proj_in.weight':torch.randn(1024,4250)*.02,'proj_in.bias':torch.randn(1024)*.02,'block.layers.0.ff.w1.weight':torch.randn(1024,1024)*.02,'block.layers.0.ff.w2.weight':torch.randn(1024,1024)*.02,'block.layers.0.norm.weight':1+torch.randn(1024)*.03,'block.layers.0.norm.bias':torch.randn(1024)*.03}
    x=torch.randn(3,17,250);observed=A.manual(x,{k:v.double() for k,v in w.items()},torch)
    expected=torch.cat([scope['manual_EEG'](x[i:i+1],w,torch,F) for i in range(3)])
    maximum=float((observed-expected).abs().max());assert maximum<2e-12
    checks.append('new_manual_formula_matches_frozen_previously_real400_validated_native_formula')
    lookup=np.full(165400,-1,dtype=np.int16)
    for i,key in enumerate(sorted(tasks)):
        t=tasks[key];where=G.SUBJECTS.index(t['subject'])*16540+np.asarray(t['exported_OOF_array_rows'])
        assert (lookup[where]==-1).all();lookup[where]=i
    count=0;sorted_keys=sorted(tasks)
    with gzip.open(G.checked(refs['query_map']),'rt') as f:
        for i,line in enumerate(f):
            q=json.loads(line);assert q['query_index']==i and q['subject']==G.SUBJECTS[i//16540] and q['array_row']==i%16540
            assert sorted_keys[int(lookup[i])]==q['encoder_task_id'] and q['dataset_split']=='official_train' and not q['official_benchmark_query'];count+=1
    assert count==165400 and (lookup>=0).all();checks.append('all165400_metadata_join_indices_independently_match_original_query_map')
    certificate=dict(status='COMPLETE_NATIVE_FULL1654_TRAIN_ONLY_FOLD_AND_HELD_EXPORT',protocol=G.ref(path),task_id=sorted_keys[0],updates=225,epochs=25,temporary_updates=0,learned_weights_saved=True,official_test_payloads_loaded=False,final_model=dict(path='synthetic_schema_only'),held_queries=8270)
    A.validate_final_certificate(certificate,G.ref(path),sorted_keys[0]);checks.append('valid_full_fold_certificate_schema')
    for label,change in [('transient9_update_rejected',dict(updates=9,epochs=1,temporary_updates=9)),('24_epoch_rejected',dict(epochs=24)),('wrong_protocol_rejected',dict(protocol={})),('test_payload_access_rejected',dict(official_test_payloads_loaded=True)),('missing_final_model_rejected',dict(final_model=None))]:
        c=copy.deepcopy(certificate);c.update(change)
        try:A.validate_final_certificate(c,G.ref(path),sorted_keys[0])
        except AssertionError:checks.append(label)
        else:raise AssertionError(label)
    assert G.ref(Path(path).parent/'brainhive_official_encoder_training_v1.py')['sha256']==G.ref(Path(s['repository_root'])/'brainhive_official_encoder_training_v1.py')['sha256'];checks.append('native_training_kernel_byte_identity_preserved')
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_TRAIN_ONLY_FULL_SCOPE_EXECUTION_CPU_PREPARATION',protocol=G.ref(path),checks=checks,count=len(checks),maximum_manual_reference_error=maximum,full_query_map_rows=count,synthetic_weights_and_certificate_not_learned_results=True,actual25_epoch_and_full_final_fits_still_pending=True,CUDA_initialized=False,new_GPU_jobs=0)
    G.atomic(Path(path).parent/'actual_execution_CPU_checks.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--references',required=True);a=p.parse_args();main(a.protocol,a.references)
