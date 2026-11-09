"""D19 all decoded image identities, source noise, pixels and exact cached metrics."""
import argparse,csv,hashlib,io,json,time,zipfile
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_image_domain_readout_20261027_v1 as V

def main():
    p=argparse.ArgumentParser()
    for key in ('completion','fit-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();assert not a.output.exists();a.output.mkdir(parents=True)
    done=U.read(a.completion);n=done['stage'];assert done['status']==f'COMPLETE_D19_STAGE{n}' and n in (80,400)
    fit=U.read(U.checked(done['fit']));spec=U.read(U.checked(done['spec']));audit=U.read(a.fit_audit)
    assert audit['status']=='D19_ALL4800_HELD_LOGITS_ALL1200_EIGHT_STEP_EDITS_AND_FOUR_NATIVE_VAE_CPU_REPLAYS_VERIFIED' and audit['completion']==done['fit']
    lock_path=U.checked(done['edit_lock']);lock=U.read(lock_path);assert lock['model']==fit['model']==spec['fixed_model'];U.checked(fit['model'])
    latents=np.load(U.checked(lock['latents']),allow_pickle=False,mmap_mode='r');source_latents=np.load(U.checked(lock['source_latents']),allow_pickle=False)
    assert latents.shape==(400,4,4,64,64) and np.array_equal(latents[:,0],source_latents) and lock['Dcheck_target_pixels_F_or_scores_used'] is False
    prepared=U.read(U.checked(fit['prepared_inputs']));mapping=U.read(U.checked(prepared['mapping']));sources=mapping['Dcheck400']
    d09=U.read(U.checked(spec['D09_completion']));inputs=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(inputs['panel']));gallery=panel['Dcheck'];queries=gallery[:n]
    assert lock['query_ids']==[q['query_id'] for q in gallery] and tuple(lock['arms'])==V.ARMS
    rows=[json.loads(line) for line in U.checked(done['manifest']).read_text().splitlines()];old=[json.loads(line) for line in U.checked(d09['check_manifest']).read_text().splitlines()]
    assert len(rows)==4*n and len(old)==1200;paths=set();inodes=set();files=set();pixels=set();constant=0
    def check_image(row):
        path=Path(row['generated_path']);st=path.stat();assert path.resolve() not in paths and (st.st_dev,st.st_ino) not in inodes
        paths.add(path.resolve());inodes.add((st.st_dev,st.st_ino));assert U.sha(path)==row['generated_sha256'];files.add(row['generated_sha256'])
        with Image.open(path) as image:
            assert image.mode=='RGB' and image.size==(512,512);x=np.asarray(image);digest=hashlib.sha256(x.tobytes()).hexdigest();assert digest==row['generated_pixel_sha256'];pixels.add(digest)
            return int(np.all(x==x[0,0]))
    for ordinal,row in enumerate(rows):
        i,j=divmod(ordinal,4);arm=V.ARMS[j]
        for key,value in queries[i].items():assert row[key]==sources[i][key]==old[i*3][key]==value
        assert row['arm']==arm and row['query_index']==i and row['edit_model']==fit['model']
        assert row['source_image']==sources[i]['generated'];U.checked(row['source_image'])
        assert row['source_image']['sha256']==old[i*3]['generated_sha256'] and row['source_pixel_sha256']==old[i*3]['generated_pixel_sha256']
        assert row['source_initial_noise_sha256']==old[i*3]['initial_noise_latents_sha256'] and row['source_initial_UNet_input_sha256']==old[i*3]['initial_UNet_input_sha256']
        assert row['source_ancestral_noises']==old[i*3]['ancestral_noise_observations'] and len(row['source_ancestral_noises'])==4
        assert row['latent_sha256']==hashlib.sha256(np.ascontiguousarray(latents[i,j:j+1]).tobytes()).hexdigest()
        assert row['endpoint_radius']==(0. if j==0 else .1) and row['small_CNN_gradient_calls']==(0 if j==0 else 8)
        assert row['original_source_UNet_NFE']==4 and row['new_UNet_NFE']==0 and row['additional_VAE_encode_images']==row['additional_VAE_decode_images']==1
        assert row['no_new_sampling_noise'] is True and row['normal_inference_target_access'] is False and row['normal_inference_only_current_query_EEG_and_generated_image'] is True
        assert lock_path.stat().st_mtime_ns<Path(row['generated_path']).stat().st_mtime_ns;constant+=check_image(row)
    parity=U.read(U.checked(done['pixel_parity']));reference=U.read(U.checked(fit['VAE_parity']));assert parity['status']=='EXACT_FIXED_FP32_COMMON_VAE_DECODE_BYTES_AND_PIXELS_PASSED'
    for key in ('generated_sha256','generated_pixel_sha256'):assert parity['image'][key]==reference['image'][key]
    check_image(parity['image'])
    generated=np.load(U.checked(done['generated_features']));target=np.load(U.checked(d09['check_target_features']));base=np.load(U.checked(d09['check_generated_features']))
    assert generated.shape==(n,4,768) and np.isfinite(generated).all();rebuilt=V.score(generated,target,base,queries,gallery,n);assert rebuilt==done['readout']
    values={arm:U.S.metric_arrays(base[:n,j],target,queries,gallery) for j,arm in enumerate(U.ARMS)}|{arm:U.S.metric_arrays(generated[:,j],target,queries,gallery) for j,arm in enumerate(V.ARMS)}
    csvrows=list(csv.DictReader((a.completion.parent/'per_query.csv').open()));assert len(csvrows)==n*7;names=list(values)
    for ordinal,row in enumerate(csvrows):
        i,j=divmod(ordinal,7);arm=names[j];assert row['query_id']==queries[i]['query_id'] and row['concept_id']==queries[i]['concept_id'] and row['arm']==arm
        for key,value in values[arm].items():assert float(row[key])==float(value[i])
    if n==400:
        previous=U.read(U.checked(done['previous']));assert rows[:320]==[json.loads(line) for line in U.checked(previous['manifest']).read_text().splitlines()]
        assert np.array_equal(generated[:80],np.load(U.checked(previous['generated_features'])))
    edit_diagnostics=U.read(U.checked(fit['edit_diagnostics']));proxy={}
    for arm in V.ARMS[1:]:
        row=edit_diagnostics[arm];gain=np.asarray(row['edited_logits'][:n])-np.asarray(row['source_logits'][:n])
        delta=values[arm]['DINO2AFC']-values['vae_roundtrip']['DINO2AFC']
        correlation=float(np.corrcoef(gain,delta)[0,1]) if gain.std()>0 and delta.std()>0 else None
        proxy[arm]=dict(mean_optimized_logit_gain=float(gain.mean()),fraction_positive_logit_gain=float((gain>0).mean()),
            actual_DINO_change_from_zero_edit_mean=float(delta.mean()),logit_gain_vs_DINO_change_Pearson=correlation,
            descriptive_post_result_only=True,used_for_model_or_strength_choice=False)
    metadata=U.read(U.checked(inputs['target_metadata']))['Dcheck'];canvas=Image.new('RGB',(8*144,34+8*164),'white');draw=ImageDraw.Draw(canvas)
    for j,label in enumerate(('target','parent','ordinary','risk','VAE roundtrip','conditional','no EEG','wrong c')):draw.text((144*j+3,6),label,fill='black')
    with ExitStack() as stack:
        archives={}
        for line,i in enumerate(range(0,80,10)):
            item=metadata[i]
            if item['image_zip'] not in archives:archives[item['image_zip']]=stack.enter_context(zipfile.ZipFile(item['image_zip']))
            raw=archives[item['image_zip']].read(item['image_member']);assert hashlib.sha256(raw).hexdigest()==item['image_sha256']
            with Image.open(io.BytesIO(raw)) as opened:stimulus=opened.convert('RGB')
            for j,row in enumerate([None]+old[i*3:i*3+3]+rows[i*4:i*4+4]):
                if row is None:image=stimulus
                else:
                    with Image.open(row['generated_path']) as opened:image=opened.convert('RGB')
                y=34+164*line;canvas.paste(ImageOps.contain(image,(140,140)),(144*j+2,y))
                if j==0:draw.text((2,y+141),item['stimulus_id'][:23],fill='black')
    sheet=a.output/'fixed_first_stimulus_8concepts.png';canvas.save(sheet)
    for ref in spec['source_pins']:U.checked(ref)
    result=dict(status=f'D19_ALL{4*n}_DECODED_IMAGES_SOURCE_NOISE_PIXELS_AND{7*n}_SCORES_CPU_VERIFIED',completion=U.ref(a.completion),fit_audit=U.ref(a.fit_audit),source=U.ref(__file__),
        all_original_source_images_and_noise_observations_verified=True,all_locked_latents_and_query_identities_exact=True,all512_RGB_file_and_pixel_SHAs_verified=True,
        all_paired_metrics_and_bootstraps_recomputed_exactly=True,all_per_query_scores_recomputed=True,same_original_sampler_images_and_fixed_noise_reused=True,
        no_new_UNet_or_sampling_noise=True,all_VAE_and_CNN_extra_costs_kept_explicit=True,condition_lock_precedes_all_images=True,zero_edit_latents_bitwise_source_identity=True,
        unique_paths=len(paths),unique_inodes=len(inodes),unique_file_hashes=len(files),unique_pixel_hashes=len(pixels),constant_new_research_images=constant,
        original80_image_and_feature_prefix_exactly_preserved=(n==400),logit_ascent_vs_actual_image_change=proxy,
        contact_sheet=U.ref(sheet),display_selection='first registered stimulus of each8concept, all7methods, no score sorting',
        new_GPU_jobs=0,new_images=0,new_parameter_updates=0,seconds=time.monotonic()-start,perceptual_noninferiority_established=False,
        exact_diffusion_score_or_posterior=False,independent_confirmation=False,SOTA_established=False)
    U.atomic(a.output/'summary.json',result);print(json.dumps({key:result[key] for key in ['status','seconds','unique_file_hashes','constant_new_research_images']}))

if __name__=='__main__':main()
