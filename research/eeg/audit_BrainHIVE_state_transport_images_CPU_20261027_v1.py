"""D18 complete native image identity, noise, byte/pixel and paired readout audit."""
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
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_state_transport_image_readout_20261027_v1 as V

def main():
    p=argparse.ArgumentParser()
    for key in ('completion','fit-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();assert not a.output.exists();a.output.mkdir(parents=True)
    done=U.read(a.completion);n=done['stage'];assert n in (80,400) and done['status']==f'COMPLETE_D18_STAGE{n}'
    fit=U.read(U.checked(done['inputs']));spec=U.read(U.checked(done['spec']));lock_path=U.checked(done['condition_lock']);lock=U.read(lock_path)
    cpu=U.read(a.fit_audit);assert cpu['status']=='D18_FULL_SOURCE_NUMERICAL_AND_ALL24600_ENDPOINT_SCORES_WITH_SELECTED_CPU_ODE_REPLAYS_VERIFIED' and cpu['completion']==done['inputs']
    assert lock['model']==fit['model']==spec['fixed_final_model'];U.checked(fit['model'])
    bank=np.load(U.checked(lock['conditions']),allow_pickle=False);assert bank.shape==(400,2,1024) and np.isfinite(bank).all()
    d09=U.read(U.checked(spec['D09_completion']));inputs=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(inputs['panel']));qs=panel['Dcheck'];queries=qs[:n]
    assert lock['query_ids']==[q['query_id'] for q in qs] and tuple(lock['arms'])==V.ARMS
    assert lock['query_target_F_or_images_used'] is False and lock['target_rank_or_DINO_used_for_choices'] is False and lock['numerical_gate_passed'] is True
    rows=[json.loads(x) for x in U.checked(done['manifest']).read_text().splitlines()];baseline=[json.loads(x) for x in U.checked(d09['check_manifest']).read_text().splitlines()]
    assert len(rows)==2*n and len(baseline)==1200
    paths=set();inodes=set();files=set();pixels=set();constant=0
    def check_image(row):
        path=Path(row['generated_path']);st=path.stat();assert path.resolve() not in paths and (st.st_dev,st.st_ino) not in inodes
        paths.add(path.resolve());inodes.add((st.st_dev,st.st_ino));assert U.sha(path)==row['generated_sha256'];files.add(row['generated_sha256'])
        with Image.open(path) as image:
            assert image.size==(512,512) and image.mode=='RGB';x=np.asarray(image);digest=hashlib.sha256(x.tobytes()).hexdigest();assert digest==row['generated_pixel_sha256'];pixels.add(digest)
            return int(np.all(x==x[0,0]))
    for ordinal,row in enumerate(rows):
        i,j=divmod(ordinal,2);arm=V.ARMS[j]
        for key,value in queries[i].items():assert row[key]==value
        assert row['arm']==arm and row['query_index']==i
        assert row['frozen_transport_model']==fit['model'] and row['field_parameters']==row['additional_parameters_at_inference']==4483072
        assert row['fixed_Heun_steps']==64 and row['small_field_NFE']==128 and row['independent_Gaussian_source_noise'] is False
        assert row['normal_inference_other_subject_EEG'] is False and row['normal_inference_only_current_query_EEG'] is True
        assert row['normal_inference_target_access'] is False and row['UNet_NFE']==row['per_query_policy_NFE']==4
        assert row['native_IP_K_calls']==row['native_IP_V_calls']==280 and row['extra_UNet_evaluations']==0 and row['scheduler_step_signature_unmodified'] is True
        for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):assert row[key]==baseline[i*3][key]
        c=np.ascontiguousarray(bank[i,j:j+1]);assert hashlib.sha256(c.tobytes()).hexdigest()==row['condition_sha256']
        assert hashlib.sha256(c.astype(np.float16).tobytes()).hexdigest()==row['condition_FP16_sha256']
        assert lock_path.stat().st_mtime_ns<Path(row['generated_path']).stat().st_mtime_ns;constant+=check_image(row)
    parity=U.read(U.checked(done['pixel_parity']));assert parity['status']=='BOTH_T_NATIVE_AND_FULL_DC_PARENT_EXACT_BYTES_PIXELS_PASSED'
    for row,expected in zip(parity['rows'],[inputs['parity_reference'],baseline[0]]):
        for key in ('generated_sha256','generated_pixel_sha256'):assert row['image'][key]==expected[key]
        for key in ('condition_sha256','condition_FP16_sha256','tokens_sha256'):assert row['proof'][key]==expected[key]
        for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256'):assert row['observations'][key]==expected[key]
        if 'ancestral_noise_observations' in expected:assert row['observations']['ancestral_noise_observations']==expected['ancestral_noise_observations']
        else:assert row['name']=='T_native'
        check_image(row['image'])
    generated=np.load(U.checked(done['generated_features']),allow_pickle=False);target=np.load(U.checked(d09['check_target_features']),allow_pickle=False);base=np.load(U.checked(d09['check_generated_features']),allow_pickle=False)
    assert generated.shape==(n,2,768) and np.isfinite(generated).all();rebuilt=V.score(generated,target,base,queries,qs,n);assert rebuilt==done['readout']
    values={arm:U.S.metric_arrays(base[:n,j],target,queries,qs) for j,arm in enumerate(U.ARMS)}|{arm:U.S.metric_arrays(generated[:,j],target,queries,qs) for j,arm in enumerate(V.ARMS)}
    names=list(values);csv_rows=list(csv.DictReader((a.completion.parent/'per_query.csv').open()));assert len(csv_rows)==n*5
    for ordinal,row in enumerate(csv_rows):
        i,j=divmod(ordinal,5);arm=names[j];assert row['query_id']==queries[i]['query_id'] and row['concept_id']==queries[i]['concept_id'] and row['arm']==arm
        for key,value in values[arm].items():assert float(row[key])==float(value[i])
    if n==400:
        previous=U.read(U.checked(done['previous']));assert rows[:160]==[json.loads(x) for x in U.checked(previous['manifest']).read_text().splitlines()]
        assert np.array_equal(generated[:80],np.load(U.checked(previous['generated_features'])))
    parent=np.load(U.checked(d09['check_conditions']),allow_pickle=False)[:,0]
    geometry={arm:dict(condition_norm_mean=float(np.linalg.norm(bank[:,j],axis=-1).mean()),
        relative_condition_change_from_parent_mean=float((np.linalg.norm(bank[:,j]-parent,axis=-1)/np.linalg.norm(parent,axis=-1)).mean())) for j,arm in enumerate(V.ARMS)}
    metadata=U.read(U.checked(inputs['target_metadata']))['Dcheck'];canvas=Image.new('RGB',(6*144,34+8*164),'white');draw=ImageDraw.Draw(canvas)
    for j,label in enumerate(('target','parent','ordinary','risk','paired flow','wrong target flow')):draw.text((144*j+3,6),label,fill='black')
    with ExitStack() as stack:
        archives={}
        for line,i in enumerate(range(0,80,10)):
            item=metadata[i]
            if item['image_zip'] not in archives:archives[item['image_zip']]=stack.enter_context(zipfile.ZipFile(item['image_zip']))
            payload=archives[item['image_zip']].read(item['image_member']);assert hashlib.sha256(payload).hexdigest()==item['image_sha256']
            with Image.open(io.BytesIO(payload)) as opened:stimulus=opened.convert('RGB')
            for j,row in enumerate([None]+baseline[i*3:i*3+3]+rows[i*2:i*2+2]):
                if row is None:image=stimulus
                else:
                    with Image.open(row['generated_path']) as opened:image=opened.convert('RGB')
                y=34+164*line;canvas.paste(ImageOps.contain(image,(140,140)),(144*j+2,y))
                if j==0:draw.text((2,y+141),item['stimulus_id'][:23],fill='black')
    sheet=a.output/'fixed_first_stimulus_8concepts.png';canvas.save(sheet)
    for item in spec['source_pins']:U.checked(item)
    result=dict(status=f'D18_ALL{2*n}_NATIVE_IMAGES_NOISE_PIXELS_AND{5*n}_SCORES_CPU_VERIFIED',source=U.ref(__file__),completion=U.ref(a.completion),fit_forward_audit=U.ref(a.fit_audit),
        all_image_identities_conditions_initial_and4_ancestral_noises_verified=True,all512_RGB_bytes_and_pixels_verified=True,all_paired_metrics_and_bootstraps_recomputed_exactly=True,
        all_per_query_scores_verified=True,original_T_and_full_Dcheck_host_parities_exact=True,condition_lock_precedes_all_images=True,
        unique_paths=len(paths),unique_inodes=len(inodes),unique_file_hashes=len(files),unique_pixel_hashes=len(pixels),constant_new_research_images=constant,
        original80_image_and_feature_prefix_exactly_preserved=(n==400),condition_geometry=geometry,contact_sheet=U.ref(sheet),
        display_selection='first registered stimulus of each8concept, all5methods, no score sorting',new_GPU_jobs=0,new_images=0,new_parameter_updates=0,
        seconds=time.monotonic()-start,normal_inference_other_subject_EEG=False,independent_confirmation=False,SOTA_established=False,perceptual_noninferiority_established=False)
    U.atomic(a.output/'summary.json',result);print(json.dumps({k:result[k] for k in ('status','seconds','unique_file_hashes','constant_new_research_images','condition_geometry')}))

if __name__=='__main__':main()
