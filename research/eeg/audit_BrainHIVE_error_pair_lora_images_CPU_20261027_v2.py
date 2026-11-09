"""D20 v2: preserve all v1 checks; read display-only targets from the actual D09 metadata."""
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_error_pair_lora_20261027_v2 as V
import brainhive_error_pair_lora_readout_20261027_v1 as M
R=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--completion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.monotonic();assert not a.output.exists();a.output.mkdir(parents=True)
    d=U.read(a.completion);assert d['status']=='COMPLETE_D20_FIXED_NATIVE_LORA_IMAGES_CHUNK' and d['end'] in (80,400);stage=d['end'];spec=U.read(U.checked(d['spec']))
    for x in spec['source_pins']:U.checked(x)
    fixed=U.read(U.checked(spec['fit_spec']));old=U.read(U.checked(spec['D09_completion']));inp=U.read(U.checked(old['inputs']));panel=U.read(U.checked(inp['panel']));queries=panel['Dcheck'];rows=[json.loads(x) for x in U.checked(d['manifest']).read_text().splitlines()];assert len(rows)==3*stage
    baseline_rows=[json.loads(x) for x in U.checked(old['check_manifest']).read_text().splitlines()];conditions=np.load(U.checked(old['check_conditions']),allow_pickle=False)
    files=set();inodes=set();hashes=set();pixels=set();constants=0
    for i,q in enumerate(queries[:stage]):
        reference=baseline_rows[3*i+1]
        for j,arm in enumerate(V.ARMS):
            row=rows[3*i+j]
            for k,value in q.items():assert row[k]==reference[k]==value
            assert row['arm']==arm and row['query_index']==i and row['normal_inference_target_access'] is False
            assert row['model']==U.read(U.checked(spec['fits'][arm]))['model']
            for key in ['initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations','condition_sha256','tokens_sha256']:assert row[key]==reference[key]
            assert row['condition_sha256']==hashlib.sha256(np.ascontiguousarray(conditions[i:i+1,1]).tobytes()).hexdigest()
            assert row['new_UNet_NFE']==4 and row['extra_UNet_NFE']==0 and row['UNet_NFE']==4 and row['native_IP_K_calls']==row['native_IP_V_calls']==280
            path=Path(row['generated_path']);assert path not in files;files.add(path);inodes.add(path.stat().st_ino)
            assert U.sha(path)==row['generated_sha256'];hashes.add(row['generated_sha256'])
            with Image.open(path) as im:assert im.size==(512,512) and im.mode=='RGB';px=np.asarray(im)
            h=hashlib.sha256(px.tobytes()).hexdigest();assert h==row['generated_pixel_sha256'];pixels.add(h)
            constants+=int(float(px.std(axis=(0,1)).max())==0.)
            assert Path(d['spec']['path']).stat().st_mtime<=path.stat().st_mtime
    assert len(inodes)==len(files)
    prefix_status=True
    if d['previous']:
        previous=U.read(U.checked(d['previous']));previous_rows=[json.loads(x) for x in U.checked(previous['manifest']).read_text().splitlines()]
        assert rows[:len(previous_rows)]==previous_rows
        assert np.array_equal(np.load(U.checked(d['generated_features']))[:previous['end']],np.load(U.checked(previous['generated_features'])))
    generated=np.load(U.checked(d['generated_features']),allow_pickle=False);base=np.load(U.checked(old['check_generated_features']),allow_pickle=False);target=np.load(U.checked(old['check_target_features']),allow_pickle=False)
    assert M.score(generated,target,base,queries[:stage],queries,stage)==d['readout']
    allarrays={arm:U.S.metric_arrays(base[:stage,j],target,queries[:stage],queries) for j,arm in enumerate(U.ARMS)}
    allarrays|={arm:U.S.metric_arrays(generated[:,j],target,queries[:stage],queries) for j,arm in enumerate(V.ARMS)}
    csvrows=list(csv.DictReader((a.completion.parent/'per_query.csv').open()));assert len(csvrows)==stage*6
    for i,q in enumerate(queries[:stage]):
        for j,arm in enumerate(allarrays):
            row=csvrows[6*i+j];assert row['query_id']==q['query_id'] and row['concept_id']==q['concept_id'] and row['arm']==arm
            for key,value in allarrays[arm].items():assert float(row[key])==float(value[i])
    parity=U.read(U.checked(d['pixel_parity']));r=parity['image'];assert U.sha(r['generated_path'])==r['generated_sha256']==baseline_rows[1]['generated_sha256']
    # Fixed first registered image of each of the original eight concepts.
    metadata=U.read(U.checked(inp['target_metadata']))['Dcheck']
    columns=['target','old parent','old ordinary','old risk','paired error','positive only','wrong error'];tile=150;rowheight=180;sheet=Image.new('RGB',(7*tile,8*rowheight+24),'white');draw=ImageDraw.Draw(sheet)
    for j,name in enumerate(columns):draw.text((j*tile+4,5),name,fill='black')
    for k,i in enumerate(range(0,80,10)):
        item=metadata[i]
        assert item['stimulus_id']==queries[i]['stimulus_id'] and item['concept_id']==queries[i]['concept_id']
        with zipfile.ZipFile(item['image_zip']) as f:b=f.read(item['image_member'])
        assert hashlib.sha256(b).hexdigest()==item['image_sha256'];ims=[Image.open(io.BytesIO(b)).convert('RGB')]
        ims+=[Image.open(baseline_rows[3*i+j]['generated_path']).convert('RGB') for j in range(3)]
        ims+=[Image.open(rows[3*i+j]['generated_path']).convert('RGB') for j in range(3)]
        for j,im in enumerate(ims):sheet.paste(ImageOps.fit(im,(tile-3,tile-3)),(j*tile,k*rowheight+24))
        draw.text((3,k*rowheight+24+tile),queries[i]['stimulus_id'],fill='black')
    sheetpath=a.output/'fixed_first_stimulus_8concepts.png';sheet.save(sheetpath)
    result=dict(status=f'D20_ALL{stage*3}_ACTUAL_IMAGES_AND{stage*6}_SCORES_CPU_VERIFIED',completion=U.ref(a.completion),source=U.ref(__file__),all_noise_condition_token_model_and_query_identities_verified=True,
        all_file_pixel_and512RGB_verified=True,all_actual_UNet_NFE4_and_native_IP_calls280=True,all_paired_metrics_and_bootstraps_exactly_recomputed=True,all_per_query_scores_recomputed=True,
        prefix_images_and_features_unchanged=prefix_status,unique_paths=len(files),unique_inodes=len(inodes),unique_file_hashes=len(hashes),unique_pixel_hashes=len(pixels),constant_new_images=constants,
        exact_pixel_duplicates_are_recorded_not_silently_excluded=True,contact_sheet=U.ref(sheetpath),display_selection='first registered stimulus of each8concept, all6methods, no score sorting',
        new_GPU_jobs=0,new_images=0,new_optimizer_updates=0,seconds=time.monotonic()-started,normal_inference_target_access=False,formal_D_resumed=False,perceptual_noninferiority_established=False,SOTA_established=False)
    U.atomic(a.output/'summary.json',result);print(json.dumps(result))
if __name__=='__main__':main()
