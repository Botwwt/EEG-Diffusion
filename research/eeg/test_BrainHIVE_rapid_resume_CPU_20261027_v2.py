"""Three CPU engineering checks: valid prefix, corruption rejection, no writes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
from PIL import Image
import generate_BrainHIVE_rapid_candidate_20261027_v2 as R


def test():
    with tempfile.TemporaryDirectory(prefix='brainhive_resume_CPU_') as temporary:
        root=Path(temporary);old=root/'images80';old.mkdir()
        bank=root/'conditions.npy';np.save(bank,np.zeros((13230,1024),np.float32),allow_pickle=False)
        native=root/'native.json';R.ORIGINAL_ATOMIC(native,dict(predictions_path=str(bank)))
        pack_path=root/'pack.json';pack=dict(mode='native_conditions',arm='candidate',host_cache_key='fixture',
            payload={'conditions':{'path':str(bank)}},native_parent_binding={'path':str(native)})
        R.ORIGINAL_ATOMIC(pack_path,pack)
        panel_path=root/'panel.json';baseline_path=root/'base.json';registration=root/'registration.json'
        for path in (panel_path,baseline_path,registration):R.ORIGINAL_ATOMIC(path,{'fixture':True})
        args=SimpleNamespace(pack=pack_path,panel=panel_path,baseline_completion=baseline_path,native_registration=registration,stage='80')
        index=[dict(row=i) for i in range(13230)]
        panel={'queries':[]};source=[];base={}
        raw_sha=hashlib.sha256(np.zeros((1,1024),np.float32).tobytes()).hexdigest()
        half_sha=hashlib.sha256(np.zeros((1,1024),np.float16).tobytes()).hexdigest()
        for i in range(2):
            query=dict(row=i,source_row=i,array_row=i,subject_id='sub-01',concept_id='c',stimulus_id=f's{i}',fold=0,query_id=f'q{i}',image_seed=1729+i)
            panel['queries'].append(query);path=old/f'{i}.png';image=Image.new('RGB',(512,512),(7,11,13));image.save(path)
            pixel_sha=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
            source.append(query|dict(global_query_index=i,pack_sha256=R.sha(pack_path),host_cache_key='fixture',UNet_NFE=4,
                normal_inference_target_access=False,arm='candidate',initial_noise_latents_sha256='a'*64,initial_UNet_input_sha256='b'*64,
                condition_sha256=raw_sha,condition_FP16_sha256=half_sha,tokens_sha256='c'*64,
                generated_path=str(path),generated_sha256=R.sha(path),generated_pixel_sha256=pixel_sha))
            base[f'q{i}']=dict(initial_noise_latents_sha256='a'*64,initial_UNet_input_sha256='b'*64,generated_pixel_sha256=pixel_sha,
                target_image_path='not_opened',score_target_path='not_opened',target_image_sha256='d'*64,target_index=i)
        manifest=old/'inference_manifest.jsonl';manifest.write_text(''.join(json.dumps(item)+'\n' for item in source),encoding='utf-8')
        provenance=old/'input_provenance.json';R.ORIGINAL_ATOMIC(provenance,dict(schema=R.V1.P.GEN_SCHEMA,stage='80',runtime_sources=R.ORIGINAL_SOURCES(),
            **{key:R.ref(path) for key,path in [('pack',pack_path),('panel',panel_path),('baseline_completion',baseline_path),('native_registration',registration)]},
            host_family='BrainHIVE',host_cache_key='fixture',reused_candidate_images=0,new_query_images=80))
        failure=old/'failure.json';R.ORIGINAL_ATOMIC(failure,dict(status='INCOMPLETE_RAPID_CANDIDATE_NO_AUTOMATIC_RETRY',stage='80',error='AssertionError()',
            zero_module_parity_passed=True,seconds=10.2,retained_candidate_rows=2,actual_new_images_generated=3))
        R.ORIGINAL_ATOMIC(old/'zero_module_pixel_parity.json',dict(status='ACTUAL_ZERO_MODULE_NATIVE_PIXEL_PARITY_PASSED',image={
            key:source[0][key] for key in ('generated_path','generated_sha256','generated_pixel_sha256')}))
        command=['python',str(Path(R.V1.__file__).resolve()),'--max-seconds','130','--output',str(old)]
        for flag,path in [('--pack',pack_path),('--panel',panel_path),('--baseline-completion',baseline_path),('--native-registration',registration)]:command.extend([flag,str(path)])
        log=root/'old.log';log.write_text('generate_BrainHIVE_rapid_candidate_20261027_v1.py\nassert time.monotonic()-invoked<args.max_seconds-120\nAssertionError\n'+
            json.dumps(dict(command=command,max_seconds=130,exit_code=1,status='FAILED_OR_JOB_TIMEOUT_RESUMABLE',charged_seconds=10.5))+'\n',encoding='utf-8')
        binding_path=root/'binding.json';binding=dict(schema=R.RESUME_SCHEMA,incomplete_directory=str(old),expected_prefix_rows=2,
            original_driver=R.ref(R.V1.__file__),input_provenance=R.ref(provenance),failure=R.ref(failure),inference_manifest=R.ref(manifest),log=R.ref(log))
        R.ORIGINAL_ATOMIC(binding_path,binding)
        before={str(path):R.sha(path) for path in old.rglob('*') if path.is_file()}
        restored,audit=R.verify_resume(old,binding_path,args,panel,pack,base,index)
        assert len(restored)==2 and audit['remaining_query_images']==78
        assert all(all(restored[i][key]==value for key,value in source[i].items()) for i in range(2))
        assert before=={str(path):R.sha(path) for path in old.rglob('*') if path.is_file()}
        original=manifest.read_bytes();manifest.write_bytes(original+b'{}\n')
        try:R.verify_resume(old,binding_path,args,panel,pack,base,index);raise RuntimeError('corrupted manifest accepted')
        except AssertionError:pass
        manifest.write_bytes(original.splitlines(keepends=True)[0])
        bad=copy.deepcopy(binding);bad['inference_manifest']=R.ref(manifest);bad_path=root/'missing_binding.json';R.ORIGINAL_ATOMIC(bad_path,bad)
        try:R.verify_resume(old,bad_path,args,panel,pack,base,index);raise RuntimeError('missing prefix row accepted')
        except AssertionError:pass
        manifest.write_bytes(original)
        assert before=={str(path):R.sha(path) for path in old.rglob('*') if path.is_file()}
    assert not R.V1.torch.cuda.is_initialized()
    return dict(status='THREE_ENGINEERING_RESUME_CPU_CHECKS_PASSED',checks=3,valid_contiguous_prefix=True,
        corrupt_or_missing_rows_rejected=True,source_pixels_and_all_original_files_unchanged=True,GPU_started=False,
        old14_sampler_fixtures_reused=True,scientific_algorithm_changed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args()
    result=test();result['source_pins']={name:R.ref(path) for name,path in [('v2_driver',R.__file__),('v1_algorithm',R.V1.__file__),
        ('v2_feature',Path(__file__).with_name('cache_BrainHIVE_rapid_candidate_DINO_20261027_v2.py')),('checks',__file__)]}
    args.receipt.parent.mkdir(parents=True,exist_ok=True);R.ORIGINAL_ATOMIC(args.receipt,result);print(json.dumps(result))
