"""All2000 new images, certified raw head, positive ray and native norm audit."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
def ref(p):
    p=Path(p)
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha)
def checked(item):assert ref(item['path'])==item;return Path(item['path'])
def read(p):return json.loads(Path(p).read_bytes())
def run(path):
    e=read(path);s=read(checked(e['protocol']));root=Path(s['output_root']);out=root/'loss_control';began=time.monotonic()
    assert e['status']=='BOUND_FIXED_FROZEN_EUCLIDEAN_F_NORM_ONLY_R80_LIFECYCLE_v1' and not (out/'artifact_audit.json').exists()
    sys.path.insert(0,s['repository_root']);import brainhive_native_models_v1
    import torch,numpy as np
    from PIL import Image
    from brainhive_norm_only_independent_CPU_20261030_v1 import check_actual,independent_projection
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    source=read(checked(s['input_completion']));bank=np.load(checked(source['bank']),allow_pickle=False)
    c=bank['conditions'].reshape(2000,1024);gamma=np.repeat(bank['gammas'],200,axis=0);beta=np.repeat(bank['betas'],200,axis=0)
    raw=np.load(checked(s['certified_FP32_trace']),allow_pickle=False)['output'];assert raw.shape==(2000,1024) and raw.dtype==np.float32
    profile=read(checked(e['profile_completion']));cert=np.load(checked(profile['certified_norm_projection']),allow_pickle=False)
    actual=np.concatenate([np.load(checked(read(root/'generation_attempts'/f'{i:04d}_{i+200:04d}/completion.json')['corrected_conditions']),allow_pickle=False) for i in range(0,2000,200)])
    assert actual.shape==(2000,1024) and actual.dtype==np.float32 and np.array_equal(actual.view(np.uint32),cert['output'].view(np.uint32))
    numerical=check_actual(actual,c,raw,gamma,beta);unused,factors,ideal=independent_projection(c,raw,gamma,beta)
    assert np.max(abs(cert['positive_factor']-factors[:,0]))<1e-12
    baseline=[json.loads(x) for x in checked(s['baseline_scoring_manifest']).read_text().splitlines()];assert len(baseline)==2000
    gallery=[json.loads(x) for x in checked(s['gallery_manifest']).read_text().splitlines()];targets={x['stimulus_id']:x for x in gallery};assert len(targets)==200
    for row in gallery:assert ref(row['image_path'])['sha256']==row['image_sha256']
    rows=[];paths=set();inodes=set();baseline_inodes=set()
    for i,old in enumerate(baseline):
        row=read(out/'records'/f'{i:04d}.json')
        assert all(row[k]==old[k] for k in ('query_id','subject','stimulus_id','array_row','image_seed','repeat_protocol','EEG_source_stimulus_id'))
        assert row['control_arm']=='Norm_only' and row['geometry']=='POSITIVE_RAY_AFFINE_NORM_ONLY'
        assert row['loss_control_checkpoint']==s['loss_control'] and row['loss_control_strength']==1 and row['loss_control_radius']==.25
        assert row['norm_only_not_center_or_angle_constraint'] and not row['original_fullT_FP64_guard_passed']
        assert row['certified_singleton_condition_bit_exact'] and row['amended_numeric_verification']
        assert row['source_execution']==ref(path) and row['parent_checkpoint']==source['reports'][i//200]['model']
        assert row['UNet_NFE']==4 and row['native_IP_K_calls']==row['native_IP_V_calls']==280 and len(row['ancestral_noise_observations'])==4
        assert not row['normal_inference_target_access'] and row['one_image_per_query']
        assert row['raw_Euclidean_condition_sha256']==hashlib.sha256(raw[i:i+1].tobytes()).hexdigest()
        assert row['original_condition_sha256']==hashlib.sha256(c[i:i+1].tobytes()).hexdigest()
        assert row['condition_sha256']==hashlib.sha256(actual[i:i+1].tobytes()).hexdigest()
        assert row['actual_A_input_fp16_sha256']==hashlib.sha256(actual[i:i+1].astype(np.float16).tobytes()).hexdigest()
        assert row['positive_norm_factor']==float(cert['positive_factor'][i])
        assert row['gamma_sha256']==hashlib.sha256(bank['gammas'][i//200].tobytes()).hexdigest() and row['beta_sha256']==hashlib.sha256(bank['betas'][i//200].tobytes()).hexdigest()
        p=Path(row['generated_path']);st=p.stat();bst=Path(old['generated_path']).stat()
        assert ref(p)['sha256']==row['generated_sha256'] and ref(old['generated_path'])['sha256']==old['generated_sha256']
        assert str(p) not in paths and (st.st_dev,st.st_ino) not in inodes
        paths.add(str(p));inodes.add((st.st_dev,st.st_ino));baseline_inodes.add((bst.st_dev,bst.st_ino))
        with Image.open(p) as image:
            assert image.size==(512,512);pixels=np.asarray(image.convert('RGB')).copy()
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['generated_pixel_sha256']
        target=targets[row['stimulus_id']];assert old['target_image_sha256']==target['image_sha256']
        row.update(concept_id=old['concept_id'],target_image_sha256=target['image_sha256'],target_image_path=target['image_path'],score_target_path=target['image_path'],target_mapping_attached_after_inference=True)
        rows.append(row)
    assert len(inodes)==len(baseline_inodes)==2000 and not inodes.intersection(baseline_inodes)
    manifest=out/'manifest.jsonl'
    with manifest.open('x') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    result=dict(status='PASS_ALL2000_NORM_ONLY_IMAGES_RAW_HEAD_CERTIFICATE_AND_BRACKETED_ROOT',queries=2000,subjects=10,gallery=200,manifest=ref(manifest),source_execution=ref(path),numerical=numerical,all2000_raw_Euclidean_outputs_bit_exact_with_prior_certificate=True,all2000_certified_norm_only_outputs_bit_exact=True,independent_norm_projection_audited_queries=2000,actual_corrected_scalar_count=2048000,unique_loss_control_inodes=2000,baseline_inodes_disjoint=True,original_fullT_FP64_guard_passed=False,old_fullT_guard_not_replaced=True,norm_only_not_full_LayerNorm_or_angle_constraint=True,effect_scores_read=False,CUDA_initialized=False,seconds=time.monotonic()-began)
    with (out/'artifact_audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',required=True);run(p.parse_args().execution)
