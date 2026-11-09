"""Engineering-only driver: preserve a verified budget-stopped v1 image prefix.

The immutable v1 renderer/sampler runs unchanged. This driver supplies a verified
prefix to its existing reuse path, adds explicit recovery provenance and error
tracebacks, and records itself separately from the v1 algorithm source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from PIL import Image
import generate_BrainHIVE_rapid_candidate_20261027_v1 as V1

sha,ref,read,rows=V1.sha,V1.ref,V1.read,V1.rows
V1_SHA='df29f16ce712eeaaa48acaa97ee1c2a1c50389b09982081d88dca50954ea5509'
RESUME_SCHEMA='BrainHIVE_rapid_incomplete_resume_20261027_v2'
ORIGINAL_SOURCES=V1.runtime_sources
ORIGINAL_VALIDATE=V1.validate
ORIGINAL_ATOMIC=V1.atomic


def runtime_sources():
    value=ORIGINAL_SOURCES()
    assert value['generator']['sha256']==V1_SHA
    return value|{'recovery_driver':ref(__file__)}


def verify_resume(directory,binding_path,args,panel,pack,baseline_rows,index):
    directory=Path(directory);binding=read(binding_path)
    assert binding['schema']==RESUME_SCHEMA
    assert Path(binding['incomplete_directory']).resolve()==directory.resolve()
    assert not (directory/'completion.json').exists(),'Resume only an incomplete attempt'
    expected_paths={'input_provenance':directory/'input_provenance.json','failure':directory/'failure.json',
                    'inference_manifest':directory/'inference_manifest.jsonl'}
    for name,expected in expected_paths.items():
        assert Path(binding[name]['path']).resolve()==expected.resolve()
        V1.C.verified(binding[name])
    V1.C.verified(binding['log']);V1.C.verified(binding['original_driver'])
    assert binding['original_driver']['sha256']==V1_SHA==sha(V1.__file__)
    provenance=read(binding['input_provenance']['path']);failure=read(binding['failure']['path'])
    assert provenance['schema']==V1.P.GEN_SCHEMA and provenance['stage']=='80' and args.stage=='80'
    assert provenance['runtime_sources']==ORIGINAL_SOURCES(),'Every original v1 helper must match'
    for source in provenance['runtime_sources'].values():V1.C.verified(source)
    for key,path in [('pack',args.pack),('panel',args.panel),('baseline_completion',args.baseline_completion),('native_registration',args.native_registration)]:
        assert provenance[key]['sha256']==sha(path);V1.C.verified(provenance[key])
    assert provenance['host_family']=='BrainHIVE' and provenance['host_cache_key']==pack['host_cache_key']
    assert failure['status']=='INCOMPLETE_RAPID_CANDIDATE_NO_AUTOMATIC_RETRY' and failure['stage']=='80'
    assert failure['error']=='AssertionError()' and failure['zero_module_parity_passed'] is True
    log=Path(binding['log']['path']).read_text(encoding='utf-8',errors='replace')
    assert 'time.monotonic()-invoked<args.max_seconds-120' in log and 'AssertionError' in log and Path(V1.__file__).name in log,'Require the original reserve-budget traceback'
    receipts=[]
    for line in log.splitlines():
        if not line.startswith('{'):continue
        try:value=json.loads(line)
        except ValueError:continue
        if {'command','max_seconds','exit_code','charged_seconds'}<=set(value):receipts.append(value)
    assert len(receipts)==1;receipt=receipts[0];command=receipt['command']
    assert receipt['exit_code']==1 and receipt['status']=='FAILED_OR_JOB_TIMEOUT_RESUMABLE'
    assert str(Path(binding['original_driver']['path'])) in command
    def argument(flag):return command[command.index(flag)+1]
    maximum=float(argument('--max-seconds'));work=maximum-120
    assert maximum==float(receipt['max_seconds']) and work>0
    assert work<=float(failure['seconds'])<work+5,'Not the recorded fixed-reserve stop'
    assert float(failure['seconds'])<=float(receipt['charged_seconds'])<float(failure['seconds'])+15
    assert float(receipt['charged_seconds'])<maximum
    assert Path(argument('--output')).resolve()==directory.resolve()
    for flag,path in [('--pack',args.pack),('--panel',args.panel),('--baseline-completion',args.baseline_completion),('--native-registration',args.native_registration)]:
        assert sha(argument(flag))==sha(path)
    source_rows=rows(binding['inference_manifest']['path']);n=len(source_rows)
    assert n==binding['expected_prefix_rows']==failure['retained_candidate_rows'] and 0<n<80
    assert failure['actual_new_images_generated']==n+1 and provenance['reused_candidate_images']==0 and provenance['new_query_images']==80
    positions={item['row']:i for i,item in enumerate(index)}
    payload=pack['payload'];array_path=payload['conditions']['path'] if pack['mode']=='native_conditions' else None
    conditions=np.load(array_path,mmap_mode='r',allow_pickle=False) if array_path else None
    native=read(pack['native_parent_binding']['path'])
    native_values=np.load(native['predictions_path'],mmap_mode='r',allow_pickle=False)
    recovered=[];image_bindings=[]
    for i,(row,query) in enumerate(zip(source_rows,panel['queries'][:n])):
        assert all(row[key]==query[key] for key in V1.C.IDENTITY_KEYS),'Missing/reordered prefix query'
        assert row['global_query_index']==i and row['pack_sha256']==sha(args.pack) and row['host_cache_key']==pack['host_cache_key']
        assert row['UNet_NFE']==4 and row['normal_inference_target_access'] is False and row['arm']==pack['arm']
        base=baseline_rows[query['query_id']]
        for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256'):assert row[key]==base[key]
        raw=np.array((conditions if conditions is not None else native_values)[positions[query['row']]:positions[query['row']]+1],copy=True)
        assert hashlib.sha256(raw.tobytes()).hexdigest()==row['condition_sha256']
        assert hashlib.sha256(raw.astype(np.float16).tobytes()).hexdigest()==row['condition_FP16_sha256']
        assert len(row['tokens_sha256'])==64
        path=Path(row['generated_path']);assert path.resolve().is_relative_to(directory.resolve())
        assert sha(path)==row['generated_sha256']
        with Image.open(path) as image:
            assert image.size==(512,512) and hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()==row['generated_pixel_sha256']
        image_bindings.append(dict(path=str(path),sha256=row['generated_sha256']))
        target={key:base[key] for key in ('target_image_path','score_target_path','target_image_sha256','target_index')}
        recovered.append(row|target|dict(target_mapping_attached_after_inference=True,
            resumed_original_v1_row=dict(original_driver_sha256=V1_SHA,input_provenance=binding['input_provenance'],
                inference_manifest=binding['inference_manifest'],line=i+1,original_row_sha256=V1.P.digest(row))))
    parity=read(directory/'zero_module_pixel_parity.json')
    assert parity['status']=='ACTUAL_ZERO_MODULE_NATIVE_PIXEL_PARITY_PASSED'
    parity_image=parity['image'];assert sha(parity_image['generated_path'])==parity_image['generated_sha256']
    assert parity_image['generated_pixel_sha256']==baseline_rows[panel['queries'][0]['query_id']]['generated_pixel_sha256']
    audit=dict(schema=RESUME_SCHEMA,status='VERIFIED_V1_BUDGET_PREFIX_PRESERVED',binding=ref(binding_path),original_directory=str(directory.resolve()),
        original_v1_driver=binding['original_driver'],original_input_provenance=binding['input_provenance'],original_failure=binding['failure'],
        original_inference_manifest=binding['inference_manifest'],original_log=binding['log'],original_wrapper_receipt=receipt,
        preserved_rows=n,remaining_query_images=80-n,source_image_bindings=image_bindings,
        old_files_modified=False,algorithm_or_model_or_seed_changed=False)
    return recovered,audit


def main():
    extra=argparse.ArgumentParser(add_help=False)
    extra.add_argument('--resume-incomplete',type=Path);extra.add_argument('--resume-binding',type=Path)
    options,remaining=extra.parse_known_args()
    assert bool(options.resume_incomplete)==bool(options.resume_binding)
    recovery=[None]
    def validate(args):
        result=ORIGINAL_VALIDATE(args)
        if options.resume_incomplete:
            panel,pack,baseline,index,native,base_rows,sources,previous,prefix=result
            assert args.stage=='80' and not prefix
            prefix,audit=verify_resume(options.resume_incomplete,options.resume_binding,args,panel,pack,base_rows,index)
            recovery[0]=audit
            return panel,pack,baseline,index,native,base_rows,sources,previous,prefix
        return result
    def atomic(path,value):
        if Path(path).name in ('input_provenance.json','completion.json','failure.json'):
            value=value|dict(execution_driver=ref(__file__),algorithm_driver=ref(V1.__file__),engineering_resume=recovery[0])
            if Path(path).name=='failure.json':
                value['runtime_error_traceback']=traceback.format_exc()
                value['runtime_error_message']='Fixed120s reserve left insufficient working time' if 'time.monotonic()-invoked<args.max_seconds-120' in value['runtime_error_traceback'] else value.get('error')
            if Path(path).name=='completion.json' and recovery[0]:
                audit=recovery[0]
                for key in ('binding','original_input_provenance','original_failure','original_inference_manifest','original_log','original_v1_driver'):
                    V1.C.verified(audit[key])
                for item in audit['source_image_bindings']:V1.C.verified(item)
        return ORIGINAL_ATOMIC(path,value)
    V1.runtime_sources=runtime_sources;V1.validate=validate;V1.atomic=atomic
    sys.argv=[sys.argv[0]]+remaining
    return V1.main()


if __name__=='__main__':raise SystemExit(main())
