"""D19 native T400 plus Dfit200 image pairs; unchanged Dcheck400 generated sources."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_image_domain_20261027_v1 as V

R=Path(__file__).resolve().parent
Q=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/rapid_iteration_20261027_v1')
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic()
    assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(2);checks=V.self_test();assert not torch.cuda.is_initialized()
    native_path=Q/'native400/completion.json';native=U.read(native_path);panel=U.read(U.checked(native['panel']));training=panel['queries'];assert len(training)==400
    source_rows=rows(U.checked(native['inference_manifest']));assert len(source_rows)==400
    binding=U.read(U.checked(native['native_parent_binding']));raw=np.load(U.checked(dict(path=binding['predictions_path'],sha256=binding['predictions_sha256'])))
    index=rows(U.checked(dict(path=binding['prediction_index'],sha256=binding['prediction_index_sha256'])));positions={q['row']:i for i,q in enumerate(index)}
    train_c=np.stack([raw[positions[q['row']]] for q in training]);assert train_c.shape==(400,1024)
    # Check actual selected OOF outputs against both original subject caches and model files.
    native_contract=U.read(R/'configs/split_manifest.json')
    T=set(native_contract['image_rows']['T']);Dfit=set(native_contract['d_fit_concepts']);Dcheck=set(native_contract['d_check_concepts'])
    assert all(q['array_row'] in T for q in training)
    coverage=np.zeros(400,np.int8)
    for source in binding['sources']:
        cache=torch.load(U.checked(dict(path=source['condition_cache_path'],sha256=source['condition_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
        U.checked(dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256']));lookup={int(row):i for i,row in enumerate(cache['array_rows'])}
        for i,q in enumerate(training):
            if q['fold']==int(source['role'][-1]):
                assert q['array_row'] in lookup
                assert np.array_equal(train_c[i],cache['condition'][lookup[q['array_row']]].numpy());coverage[i]+=1
    assert np.all(coverage==1)
    d09_path=Q/'candidates/D09_gain_calibration/stage400_recovered_v3/completion.json';d09=U.read(d09_path);inputs=U.read(U.checked(d09['inputs']));dp=U.read(U.checked(inputs['panel']))
    metadata={int(x['array_row']):x for x in csv.DictReader(U.checked(inputs['image_manifest']).open()) if x['dataset_split']=='train'}
    assert len(dp['Dfit'])==200 and len(dp['Dcheck'])==400
    assert all(q['concept_id'] in Dfit for q in dp['Dfit']) and all(q['concept_id'] in Dcheck for q in dp['Dcheck'])
    fit_manifest=Q/'candidates/D09_gain_calibration/stage80/Dfit_inference_manifest.jsonl';fit_rows=rows(fit_manifest)
    assert len(fit_rows)==600;fit_parent=fit_rows[::3];dc_rows=rows(U.checked(d09['check_manifest']))[::3]
    cache=torch.load(U.checked(inputs['native_cache']),weights_only=True,map_location='cpu',mmap=True);lookup={int(row):i for i,row in enumerate(cache['array_rows'])}
    fit_c=cache['condition'][[lookup[q['array_row']] for q in dp['Dfit']]].numpy().copy();dc_c=np.load(U.checked(d09['check_conditions']))[:,0].copy()
    assert np.array_equal(dc_c,cache['condition'][[lookup[q['array_row']] for q in dp['Dcheck']]].numpy())
    mapping={};count=0
    for name,queries,generated,c in [('T400',training,source_rows,train_c),('Dfit200',dp['Dfit'],fit_parent,fit_c),('Dcheck400',dp['Dcheck'],dc_rows,dc_c)]:
        records=[]
        for i,(q,row) in enumerate(zip(queries,generated)):
            for key,value in q.items():assert row[key]==value
            assert row['arm'] in ('native_parent','parent') and row['UNet_NFE']==4 and row['normal_inference_target_access'] is False
            assert hashlib.sha256(np.ascontiguousarray(c[i:i+1]).tobytes()).hexdigest()==row['condition_sha256']
            path=U.checked(dict(path=row['generated_path'],sha256=row['generated_sha256']))
            with Image.open(path) as image:
                assert image.size==(512,512) and image.mode=='RGB';assert hashlib.sha256(np.asarray(image).tobytes()).hexdigest()==row['generated_pixel_sha256']
            item=q|dict(generated=U.ref(path),generated_pixel_sha256=row['generated_pixel_sha256'],original_observation=row,condition_sha256=row['condition_sha256'])
            if name!='Dcheck400':
                target=metadata[q['array_row']];assert target['concept_id']==q['concept_id'] and target['stimulus_id']==q['stimulus_id']
                assert row['target_image_sha256']==target['image_sha256'];item['target']={key:target[key] for key in ('image_zip','image_member','image_sha256','image_bytes')}
            records.append(item);count+=1
        assert len(records)==len(queries);mapping[name]=records;U.save_array(a.output/(name+'_conditions.npy'),c)
    U.atomic(a.output/'mapping.json',mapping);U.atomic(a.output/'CPU_checks.json',checks)
    refs=[U.ref(native_path),native['panel'],native['inference_manifest'],native['native_parent_binding'],U.ref(fit_manifest),U.ref(d09_path),d09['inputs'],inputs['native_cache'],inputs['native_checkpoint'],inputs['panel'],inputs['image_manifest'],inputs['split'],d09['check_conditions'],d09['check_manifest']]
    for source in binding['sources']:
        for key,hashkey in [('condition_cache_path','condition_cache_sha256'),('checkpoint_path','checkpoint_sha256')]:refs.append(dict(path=source[key],sha256=source[hashkey]))
    result=dict(status='D19_ALL1000_NATIVE_IMAGE_CONDITION_IDENTITIES_AND_T400_DFIT200_PAIR_METADATA_VERIFIED',schema=V.SCHEMA,mapping=U.ref(a.output/'mapping.json'),
        conditions={name:U.ref(a.output/(name+'_conditions.npy')) for name in mapping},CPU_checks=U.ref(a.output/'CPU_checks.json'),inputs=refs,
        source=U.ref(__file__),D09_completion=U.ref(d09_path),T_training_queries=400,T_training_concepts=40,training_subjects=1,Dfit_diagnostic_queries=200,Dcheck_source_queries=400,
        all_original_image_bytes_pixels_condition_and_identity_hashes_verified=True,original_OOF_caches_and_checkpoints_checked=True,
        Dcheck_target_pixels_or_F_read=False,Dcheck_scores_read=False,source_sampler_terminal_latents_available=False,
        proposed_latent_distribution='same native VAE mode of real images and generated image files, same512 center-crop preprocess; distinct from sampler terminal latents',
        CUDA_initialized=False,new_GPU_jobs=0,new_images=0,new_fit=0,seconds=time.monotonic()-start)
    U.atomic(a.output/'completion.json',result);print(json.dumps({key:result[key] for key in ['status','seconds','T_training_queries','Dfit_diagnostic_queries','Dcheck_source_queries']}))

if __name__=='__main__':main()
