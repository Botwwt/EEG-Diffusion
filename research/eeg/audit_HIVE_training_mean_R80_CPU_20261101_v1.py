"""Audit complete native control images/conditions/noise before semantic scores."""
import argparse
import hashlib
from pathlib import Path
import time
import brainhive_training_mean_R80_contract_CPU_20261101_v1 as M


def constant_geometry_numpy(condition,weight,bias,raw,np):
    c=condition.astype(np.float64);w=weight.astype(np.float64);b=bias.astype(np.float64)
    u=(c-b)/w;d=raw.astype(np.float64)[None,:]-raw.astype(np.float64).mean()
    r2=(u*u).sum(-1,keepdims=True);d=d-(d*u).sum(-1,keepdims=True)/r2*u
    delta=.25*np.sqrt(r2)*d/np.sqrt(r2+(d*d).sum(-1,keepdims=True))
    scale=np.sqrt(r2)/np.sqrt(((u+delta)**2).sum(-1,keepdims=True))
    return c+w*((scale-1)*u+scale*delta)


def run(protocol,role):
    started=time.monotonic();s,parent,F=M.setup(protocol)
    import numpy as np
    import torch
    from PIL import Image
    assert not torch.cuda.is_initialized();root=M.OUT/role
    unused,component,saved=M.load_control(s,role,torch)
    source,bank,native=M.load_evaluation_inputs(parent,torch)
    closure=M.read(M.OUT/'actual_source_data_closure.json')
    references={key:[__import__('json').loads(x) for x in M.checked(pin).read_text().splitlines()]
                for key,pin in closure['source_manifests'].items()}
    profile=M.read(root/'native_profile/completion.json')
    assert profile['status']=='PASS_TRAIN_MEAN_ALL2000_CONDITIONS_AND10_FIVE_PATH_PIXEL_REPLAYS'
    assert profile['component']==component and profile['role']==role and profile['profile_indices']==s['profile_indices']
    certified=np.load(M.checked(profile['certified_conditions']),allow_pickle=False)
    assert certified.shape==(2000,1024) and certified.dtype==np.float32 and np.isfinite(certified).all()
    independent=constant_geometry_numpy(bank['conditions'].reshape(2000,1024),
        np.repeat(bank['gammas'],200,axis=0),np.repeat(bank['betas'],200,axis=0),saved['model']['direction.bias'].numpy(),np)
    assert np.allclose(certified,independent,**s['numerical_guard'])
    independent_max=float(np.abs(certified.astype(np.float64)-independent).max())
    gallery=[__import__('json').loads(x) for x in M.checked(parent['evaluation']['gallery_manifest']).read_text().splitlines()]
    assert len(gallery)==200;targets={row['stimulus_id']:row for row in gallery};assert len(targets)==200
    for row in gallery:assert M.ref(row['image_path'])['sha256']==row['image_sha256']
    reserved=set()
    for items in references.values():
        assert len(items)==2000
        for row in items:
            st=Path(row['generated_path']).stat();reserved.add((st.st_dev,st.st_ino))
    assert len(reserved)==6000
    def check_pixels(record):
        path=Path(record['generated_path']);assert M.ref(path)['sha256']==record['generated_sha256']
        st=path.stat();identity=st.st_dev,st.st_ino
        assert identity not in reserved;reserved.add(identity)
        with Image.open(path) as image:
            assert image.size==(512,512);pixels=np.asarray(image.convert('RGB'))
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==record['generated_pixel_sha256']
        return pixels
    def check_replays(folder,indices,labels):
        for index in indices:
            for label in labels:
                item=M.read(folder/f'{label}_{index:04d}.json');pixels=check_pixels(item)
                expected=references[M.ROLES[role]][index] if label=='source_head_replay' else native[index]
                assert item['expected_image_path']==expected['generated_path'] and item['expected_file_SHA256']==expected['generated_sha256']
                with Image.open(expected['generated_path']) as image:assert np.array_equal(pixels,np.asarray(image.convert('RGB')))
                for key in ('initial_noise_latents_sha256','ancestral_noise_observations'):assert item[key]==expected[key]
    check_replays(root/'native_profile',s['profile_indices'],('native_replay','source_head_replay','disabled_replay','zero_head_replay'))
    for pin in profile['records']:check_pixels(M.read(M.checked(pin)))
    rows=[];values=[]
    for lo in range(0,2000,200):
        folder=root/'generation'/f'{lo:04d}_{lo+200:04d}';done=M.read(folder/'completion.json')
        assert done['status']=='COMPLETE_TRAIN_MEAN_R80_FIXED200_IMAGES' and done['range']==[lo,lo+200]
        assert done['component']==component and done['role']==role and len(done['records'])==200
        values.append(np.load(M.checked(done['actual_conditions']),allow_pickle=False))
        check_replays(folder,[lo],('native_replay','source_head_replay'))
        for index,pin in enumerate(done['records'],lo):
            row=M.read(M.checked(pin));baseline=references['parent'][index];source_row=references[M.ROLES[role]][index]
            assert row['global_query_index']==index and row['source_protocol']==M.ref(protocol) and row['arm']==role
            assert row['component']==component and row['source_component']==saved['source_component'] and row['derivation_protocol']==M.MEAN
            for key in ('query_id','subject','stimulus_id','array_row','image_seed','repeat_protocol','EEG_source_stimulus_id'):
                assert row[key]==baseline[key]==source_row[key]
            for key in ('initial_noise_latents_sha256','ancestral_noise_observations'):assert row[key]==source_row[key]
            assert row['original_condition_sha256']==native[index]['condition_sha256']
            assert row['condition_sha256']==hashlib.sha256(certified[index:index+1].tobytes()).hexdigest()
            assert row['actual_A_input_fp16_sha256']==hashlib.sha256(certified[index:index+1].astype(np.float16).tobytes()).hexdigest()
            assert row['UNet_NFE']==4 and row['native_IP_K_calls']==row['native_IP_V_calls']==280
            assert row['strength']==1. and row['radius']==.25 and row['normal_inference_target_access'] is False
            assert row['new_training_updates']==0 and row['mean_constructed_from_training_only'] is True
            check_pixels(row)
            target=targets[row['stimulus_id']];assert baseline['target_image_sha256']==target['image_sha256']
            row.update(concept_id=baseline['concept_id'],target_image_sha256=target['image_sha256'],
                target_image_path=target['image_path'],score_target_path=target['image_path'],target_mapping_attached_after_inference=True)
            rows.append(row)
    assert len(rows)==2000 and np.array_equal(np.concatenate(values),certified) and not torch.cuda.is_initialized()
    output=root/'loss_control';output.mkdir()
    with (output/'manifest.jsonl').open('x') as stream:
        for row in rows:stream.write(__import__('json').dumps(row)+'\n')
    result=dict(status='PASS_TRAIN_MEAN_ALL2000_R80_ARTIFACTS_AND_NUMPY_GEOMETRY',queries=2000,subjects=10,gallery=200,
        role=role,manifest=M.ref(output/'manifest.jsonl'),protocol=M.ref(protocol),component=component,
        all2048000_generation_condition_values_bit_exact_with_profile=True,independent_NumPy_geometry_maxerror=independent_max,
        all_noise_equal_own_source_head=True,all_new_file_pixel_identities_verified=True,
        profile50_and_segment20_engineering_image_files_independently_checked=True,
        effect_scores_read=False,CUDA_initialized=False,seconds=time.monotonic()-started)
    M.write(output/'artifact_audit.json',result);print(__import__('json').dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--role',choices=tuple(M.ROLES),required=True)
    a=p.parse_args();run(a.protocol,a.role)
