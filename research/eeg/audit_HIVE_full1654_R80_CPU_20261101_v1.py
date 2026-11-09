"""All2000 file/pixel/identity/noise/condition joins, with complete denominator."""
import argparse,hashlib,json,time
from pathlib import Path
import brainhive_full1654_F_study_contract_20261101_v1 as G

def run(path,arm):
    start=time.monotonic();s=G.setup(path)
    import numpy as np,torch
    from PIL import Image
    assert not torch.cuda.is_initialized();root=Path(s['output_root'])/arm
    _,fit,_=G.load_final(s,arm,torch);profile=G.read(root/'native_profile/completion.json')
    certified=np.load(G.check(profile['certified_conditions']),allow_pickle=False)
    baseline=[json.loads(x) for x in G.check(s['evaluation']['baseline_scoring_manifest']).read_text().splitlines()]
    original_ordinary=[json.loads(x) for x in G.check(s['evaluation']['ordinary_manifest']).read_text().splitlines()]
    assert len(baseline)==len(original_ordinary)==2000
    gallery=[json.loads(x) for x in G.check(s['evaluation']['gallery_manifest']).read_text().splitlines()];targets={r['stimulus_id']:r for r in gallery};assert len(targets)==200
    for r in targets.values():assert G.ref(r['image_path'])['sha256']==r['image_sha256']
    inodes=set();paths=set();records=[];conditions=[]
    for lo in range(0,2000,200):
        out=root/'generation'/f'{lo:04d}_{lo+200:04d}';done=G.read(out/'completion.json')
        assert done['status']=='COMPLETE_FULL1654_R80_200_IMAGES' and done['range']==[lo,lo+200] and done['component']==fit['component'] and not done['effect_scores_read']
        conditions.append(np.load(G.check(done['actual_conditions']),allow_pickle=False))
        for i,pin in enumerate(done['records'],lo):
            r=G.read(G.check(pin));old=baseline[i];ordinary=original_ordinary[i]
            assert r['global_query_index']==i and r['arm']==arm and r['component']==fit['component'] and r['source_study']==G.ref(path)
            for key in ('query_id','subject','stimulus_id','array_row','image_seed','repeat_protocol','EEG_source_stimulus_id'):assert r[key]==old[key]==ordinary[key]
            assert r['strength']==1. and r['radius']==.25 and r['UNet_NFE']==4 and r['native_IP_K_calls']==r['native_IP_V_calls']==280
            for key in ('initial_noise_latents_sha256','ancestral_noise_observations'):assert r[key]==ordinary[key]
            assert not r['normal_inference_target_access'] and r['one_image_per_query']
            value=certified[i:i+1];assert r['condition_sha256']==hashlib.sha256(value.tobytes()).hexdigest() and r['actual_A_input_fp16_sha256']==hashlib.sha256(value.astype(np.float16).tobytes()).hexdigest()
            image=Path(r['generated_path']);assert G.ref(image)['sha256']==r['generated_sha256']
            st=image.stat();identity=st.st_dev,st.st_ino
            assert str(image) not in paths and identity not in inodes;paths.add(str(image));inodes.add(identity)
            for reference in (old,ordinary):
                other=Path(reference['generated_path']);v=other.stat();assert identity!=(v.st_dev,v.st_ino)
                assert G.ref(other)['sha256']==reference['generated_sha256']
            with Image.open(image) as im:assert im.size==(512,512);pixels=np.asarray(im.convert('RGB')).copy()
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==r['generated_pixel_sha256']
            target=targets[r['stimulus_id']];assert old['target_image_sha256']==target['image_sha256']
            r.update(concept_id=old['concept_id'],target_image_sha256=target['image_sha256'],target_image_path=target['image_path'],score_target_path=target['image_path'],target_mapping_attached_after_inference=True)
            records.append(r)
    assert len(records)==len(inodes)==2000 and np.array_equal(np.concatenate(conditions),certified)
    out=root/'loss_control';out.mkdir();manifest=out/'manifest.jsonl'
    with manifest.open('x') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    result=dict(status='PASS_FULL1654_ALL2000_R80_IMAGES_CONDITIONS_NOISE_IDENTITIES_AND_PIXELS',queries=2000,subjects=10,gallery=200,manifest=G.ref(manifest),study=G.ref(path),arm=arm,component=fit['component'],all2048000_generated_condition_values_bit_exact_to_CPU_verified_profile=True,all_noise_equal_original_ordinary=True,all_images_unique_inodes_and_disjoint_reference_paths=True,CUDA_initialized=False,effect_scores_read=False,seconds=time.monotonic()-start)
    G.write(out/'artifact_audit.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--arm',choices=G.ARMS,required=True);a=p.parse_args();run(a.study,a.arm)
