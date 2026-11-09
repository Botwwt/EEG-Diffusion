"""All D16 images/noise/identity/readout and complete400 FP32 condition replay."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
from contextlib import ExitStack
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_response_image_diagnostic_20261027_v1 as V
from analyze_BrainHIVE_visual_support_CPU_20261027_v1 import F_metrics

def main():
    p=argparse.ArgumentParser()
    for key in ('completion','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();torch.set_num_threads(2)
    assert not torch.cuda.is_initialized() and not a.output.exists();a.output.mkdir(parents=True)
    done=U.read(a.completion);assert done['status']=='COMPLETE_D16_FIXED80_IMAGE_DIAGNOSTIC' and done['diagnostic_only'] is True
    assert done['original_prediction_gate_passed'] is False and done['original_candidate_admitted'] is False and done['automatic_expansion_allowed'] is False
    fit=U.read(U.checked(done['inputs']));spec=U.read(U.checked(done['spec']));d09=U.read(U.checked(spec['D09_completion']))
    inputs=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(inputs['panel']));qs=panel['Dcheck'];queries=qs[:80]
    lock_path=U.checked(done['condition_lock']);lock=U.read(lock_path);bank=np.load(U.checked(lock['conditions']),allow_pickle=False)
    assert bank.shape==(400,4,1024) and bank.dtype==np.float32 and np.isfinite(bank).all()
    assert lock['query_ids']==[q['query_id'] for q in qs] and tuple(lock['arms'])==V.ARMS
    assert all(lock[k] is False for k in ('Dcheck_target_F_or_image_used_for_condition','target_rank_or_DINO_used_for_choices','original_prediction_gate_passed','original_candidate_admitted','automatic_expansion_allowed'))
    old=U.read(U.checked(fit['spec']));assert old['epochs']==10
    assert U.ref(lock['model']['path'])==fit['model']==spec['fixed_final_model']
    rows=[json.loads(x) for x in U.checked(done['manifest']).read_text().splitlines()]
    baseline=[json.loads(x) for x in U.checked(d09['check_manifest']).read_text().splitlines()]
    assert len(rows)==320 and len(baseline)==1200
    paths=set();inodes=set();files=set();pixels=set();constant=0
    def check_image(row):
        path=Path(row['generated_path']);st=path.stat();assert path.resolve() not in paths and (st.st_dev,st.st_ino) not in inodes
        paths.add(path.resolve());inodes.add((st.st_dev,st.st_ino));assert U.sha(path)==row['generated_sha256'];files.add(row['generated_sha256'])
        with Image.open(path) as image:
            assert image.size==(512,512) and image.mode=='RGB';array=np.asarray(image)
            digest=hashlib.sha256(array.tobytes()).hexdigest();assert digest==row['generated_pixel_sha256'];pixels.add(digest)
            return int(np.all(array==array[0,0]))
    for ordinal,row in enumerate(rows):
        i,j=divmod(ordinal,4);arm=V.ARMS[j]
        for key,value in queries[i].items():assert row[key]==value
        assert row['arm']==arm and row['query_index']==i and row['UNet_NFE']==row['per_query_policy_NFE']==4
        assert row['native_IP_K_calls']==row['native_IP_V_calls']==280 and row['extra_UNet_evaluations']==0
        assert row['diagnostic_only'] is True and row['normal_inference_target_access'] is False and row['scheduler_step_signature_unmodified'] is True
        assert row['response_source_query_index']==(i if arm=='paired_responses' else lock['shuffled_response_source_query_permutation'][i] if arm=='shuffled_responses' else None)
        for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):assert row[key]==baseline[i*3][key]
        raw=np.ascontiguousarray(bank[i,j:j+1]);assert hashlib.sha256(raw.tobytes()).hexdigest()==row['condition_sha256']
        assert hashlib.sha256(raw.astype(np.float16).tobytes()).hexdigest()==row['condition_FP16_sha256']
        assert lock_path.stat().st_mtime_ns<Path(row['generated_path']).stat().st_mtime_ns
        constant+=check_image(row)
    parity=U.read(U.checked(done['pixel_parity']));assert parity['status']=='BOTH_T_NATIVE_AND_FULL_DC_PARENT_EXACT_BYTES_PIXELS_PASSED'
    for row,expected in zip(parity['rows'],[inputs['parity_reference'],baseline[0]]):
        for key in ('generated_sha256','generated_pixel_sha256'):assert row['image'][key]==expected[key]
        for key in ('condition_sha256','condition_FP16_sha256','tokens_sha256'):assert row['proof'][key]==expected[key]
        for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256'):assert row['observations'][key]==expected[key]
        if 'ancestral_noise_observations' in expected:assert row['observations']['ancestral_noise_observations']==expected['ancestral_noise_observations']
        else:assert row['name']=='T_native'
        check_image(row['image'])
    generated=np.load(U.checked(done['generated_features']),allow_pickle=False);target=np.load(U.checked(d09['check_target_features']),allow_pickle=False)
    base=np.load(U.checked(d09['check_generated_features']),allow_pickle=False)
    assert generated.shape==(80,4,768) and np.isfinite(generated).all()
    rebuilt=V.score(generated,target,base,queries,qs,80);assert rebuilt==done['readout']
    measures={arm:U.S.metric_arrays(base[:80,j],target,queries,qs) for j,arm in enumerate(U.ARMS)}|{arm:U.S.metric_arrays(generated[:,j],target,queries,qs) for j,arm in enumerate(V.ARMS)}
    csv_rows=list(csv.DictReader((Path(a.completion).parent/'per_query.csv').open()));assert len(csv_rows)==560
    names=list(measures)
    for ordinal,row in enumerate(csv_rows):
        i,j=divmod(ordinal,7);arm=names[j];assert row['query_id']==queries[i]['query_id'] and row['concept_id']==queries[i]['concept_id'] and row['arm']==arm
        for key,value in measures[arm].items():assert float(row[key])==float(value[i])
    selfpair=U.S.metric_arrays(target,target,qs,qs);assert np.all(selfpair['DINO2AFC']==1) and np.all(selfpair['gallery_top1']==1)
    # Recreate ALL400 conditions from byte-bound native c/A/R and final heads,
    # using FP32 CPU arithmetic. No fitting or GPU, no target F supplied.
    replay_dir=a.output/'CPU_all400_conditions';replay_dir.mkdir();cpu,cpu_lock=V.build_conditions(fit,spec,inputs,panel,replay_dir,device='cpu')
    relative=np.linalg.norm(cpu-bank,axis=-1)/np.linalg.norm(bank,axis=-1)
    cosine=(cpu*bank).sum(-1)/(np.linalg.norm(cpu,axis=-1)*np.linalg.norm(bank,axis=-1))
    assert relative.max()<.03 and cosine.min()>.999
    numeric=dict(queries=400,arms=4,maximum_relative_L2=float(relative.max()),minimum_cosine=float(cosine.min()),
        original_dtype='BF16 head/FP32 native geometry',replay_dtype='FP32 CPU',all400_conditions_replayed=True)
    U.atomic(a.output/'CPU_all400_replay.json',numeric)
    cache=torch.load(U.checked(inputs['native_cache']),weights_only=True,map_location='cpu',mmap=True);lookup={int(row):i for i,row in enumerate(cache['array_rows'])}
    raw=cache['condition'][[lookup[q['array_row']] for q in qs]].numpy();checkpoint=torch.load(U.checked(inputs['native_checkpoint']),weights_only=True,map_location='cpu')['model']
    gamma=checkpoint['block.layers.0.norm.weight'].numpy();beta=checkpoint['block.layers.0.norm.bias'].numpy();u=(raw-beta)/gamma;modified=(bank-beta)/gamma
    norms=np.linalg.norm(u,axis=-1);changed=np.linalg.norm(modified,axis=-1);norm_error=float(np.abs(changed/norms[:,None]-1).max())
    direction=(modified*u[:,None]).sum(-1)/(changed*norms[:,None]);assert norm_error<1e-5 and direction.min()>=1/np.sqrt(1+.25**2)-1e-5
    # Post-outcome geometry, not an input to locked samples or image selection.
    contract=U.read('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json')
    raw_F=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
    target_F=raw_F[[q['array_row'] for q in qs]].numpy();geometry={'parent':F_metrics(raw,target_F,80)}
    for j,arm in enumerate(V.ARMS):geometry[arm]=F_metrics(bank[:,j],target_F,80)
    metadata=U.read(U.checked(inputs['target_metadata']))['Dcheck'];canvas=Image.new('RGB',(8*144,34+8*164),'white');draw=ImageDraw.Draw(canvas)
    for j,label in enumerate(('target','parent','ordinary','risk','c-only','final A','paired R','shuffled R')):draw.text((144*j+3,6),label,fill='black')
    with ExitStack() as stack:
        archives={}
        for line,i in enumerate(range(0,80,10)):
            item=metadata[i]
            if item['image_zip'] not in archives:archives[item['image_zip']]=stack.enter_context(zipfile.ZipFile(item['image_zip']))
            payload=archives[item['image_zip']].read(item['image_member']);assert hashlib.sha256(payload).hexdigest()==item['image_sha256']
            with Image.open(io.BytesIO(payload)) as image:stimulus=image.convert('RGB')
            for j,row in enumerate([None]+baseline[i*3:i*3+3]+rows[i*4:i*4+4]):
                if row is None:image=stimulus
                else:
                    with Image.open(row['generated_path']) as opened:image=opened.convert('RGB')
                y=34+164*line;canvas.paste(ImageOps.contain(image,(140,140)),(144*j+2,y))
                if j==0:draw.text((2,y+141),item['stimulus_id'][:23],fill='black')
    sheet=a.output/'fixed_first_stimulus_8concepts.png';canvas.save(sheet)
    for reference in spec['source_pins']:U.checked(reference)
    assert not torch.cuda.is_initialized()
    summary=dict(status='D16_ALL320_IMAGES_NOISE_PIXELS_560_SCORES_AND_ALL400_CONDITIONS_CPU_VERIFIED',source=U.ref(__file__),completion=U.ref(a.completion),
        diagnostic_only=True,original_prediction_gate_passed=False,original_candidate_admitted=False,automatic_expansion_allowed=False,
        all320_image_queries_native_conditions_initial_and_ancestral_noise_verified=True,all_images_size512_RGB_hashes_verified=True,
        unique_paths=len(paths),unique_inodes=len(inodes),unique_file_hashes=len(files),unique_pixel_hashes=len(pixels),constant_research_images=constant,
        original_T_and_full_Dcheck_parent_parities_exact=True,legacy_T_ancestral_noise_comparison_available=False,
        all560_perquery_scores_verified=True,full_image_readout_and_bootstrap_recomputed_exactly=True,condition_lock_precedes_all_new_images=True,
        CPU_all400_replay=numeric,all400_native_norm_max_relative_error=norm_error,all400_native_direction_min_cosine=float(direction.min()),
        postresult_F_geometry=geometry,F_geometry_is_not_image_quality=True,contact_sheet=U.ref(sheet),fixed_sheet_selection='first registered stimulus in each8concept, all7 methods, no score sorting',
        new_GPU_jobs=0,new_parameter_updates=0,new_images=0,no_model_or_locked_conditions_modified=True,seconds=time.monotonic()-start,
        independent_confirmation=False,SOTA_established=False,perceptual_noninferiority_established=False)
    U.atomic(a.output/'summary.json',summary);print(json.dumps(dict(status=summary['status'],seconds=summary['seconds'],CPU_all400_replay=numeric,F_geometry=geometry)))

if __name__=='__main__':main()
