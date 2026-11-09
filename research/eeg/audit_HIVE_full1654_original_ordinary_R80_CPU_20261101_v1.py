"""Complete2000 new image/condition/noise/target joins before any semantic scoring."""
import argparse
import hashlib
from pathlib import Path
import time
import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F
E=F.E;K=F.K


def run(protocol):
    started=time.monotonic();s=F.setup_fit(protocol);s['_protocol_pin']=K.ref(protocol)
    import numpy as np
    import torch
    from PIL import Image
    assert not torch.cuda.is_initialized();root=Path(s['output_root'])/'ordinary_full1654'
    unused,fit,audit=F.load_final(s,torch);profile=K.read(root/'native_profile/completion.json')
    certified=np.load(K.check(profile['certified_conditions']),allow_pickle=False)
    parent=[__import__('json').loads(x) for x in K.check(s['evaluation']['baseline_scoring_manifest']).read_text().splitlines()]
    ordinary=[__import__('json').loads(x) for x in K.check(s['evaluation']['ordinary_manifest']).read_text().splitlines()]
    gallery=[__import__('json').loads(x) for x in K.check(s['evaluation']['gallery_manifest']).read_text().splitlines()]
    assert len(parent)==len(ordinary)==2000 and len(gallery)==200
    targets={row['stimulus_id']:row for row in gallery}
    for row in gallery:assert K.ref(row['image_path'])['sha256']==row['image_sha256']
    rows=[];values=[];inodes=set()
    for lo in range(0,2000,200):
        folder=root/'generation'/f'{lo:04d}_{lo+200:04d}';done=K.read(folder/'completion.json')
        assert done['status']=='COMPLETE_ORIGINAL_ORDINARY_FULL1654_R80_200_IMAGES' and done['range']==[lo,lo+200]
        assert done['component']==fit['component'] and len(done['records'])==200
        values.append(np.load(K.check(done['actual_conditions']),allow_pickle=False))
        for index,pin in enumerate(done['records'],lo):
            row=K.read(K.check(pin));baseline=parent[index];old=ordinary[index]
            assert row['global_query_index']==index and row['source_protocol']==K.ref(protocol)
            for key in ('query_id','subject','stimulus_id','array_row','image_seed','repeat_protocol','EEG_source_stimulus_id'):
                assert row[key]==baseline[key]==old[key]
            for key in ('initial_noise_latents_sha256','ancestral_noise_observations'):assert row[key]==old[key]
            assert row['condition_sha256']==hashlib.sha256(certified[index:index+1].tobytes()).hexdigest()
            assert row['actual_A_input_fp16_sha256']==hashlib.sha256(certified[index:index+1].astype(np.float16).tobytes()).hexdigest()
            assert row['UNet_NFE']==4 and row['native_IP_K_calls']==row['native_IP_V_calls']==280
            assert row['strength']==1. and row['radius']==.25 and row['normal_inference_target_access'] is False
            assert K.ref(row['generated_path'])['sha256']==row['generated_sha256']
            info=Path(row['generated_path']).stat();identity=info.st_dev,info.st_ino
            assert identity not in inodes;inodes.add(identity)
            for other in (baseline,old):
                ref=Path(other['generated_path']).stat();assert identity!=(ref.st_dev,ref.st_ino)
            with Image.open(row['generated_path']) as image:
                assert image.size==(512,512);pixels=np.asarray(image.convert('RGB'))
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['generated_pixel_sha256']
            target=targets[row['stimulus_id']];assert baseline['target_image_sha256']==target['image_sha256']
            row.update(concept_id=baseline['concept_id'],target_image_sha256=target['image_sha256'],
                       target_image_path=target['image_path'],score_target_path=target['image_path'],target_mapping_attached_after_inference=True)
            rows.append(row)
    assert len(rows)==2000 and len(inodes)==2000 and np.array_equal(np.concatenate(values),certified)
    output=root/'loss_control';output.mkdir()
    with (output/'manifest.jsonl').open('x') as stream:
        for row in rows:stream.write(__import__('json').dumps(row)+'\n')
    result=dict(status='PASS_ORIGINAL_ORDINARY_FULL1654_ALL2000_R80_ARTIFACTS',queries=2000,subjects=10,gallery=200,
                manifest=K.ref(output/'manifest.jsonl'),protocol=K.ref(protocol),component=fit['component'],
                all2048000_condition_values_bit_exact=True,all_noise_equal_original_ordinary=True,
                all2000_file_pixel_identities_verified=True,effect_scores_read=False,CUDA_initialized=False,
                seconds=time.monotonic()-started)
    K.write(output/'artifact_audit.json',result);print(__import__('json').dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);run(parser.parse_args().protocol)
