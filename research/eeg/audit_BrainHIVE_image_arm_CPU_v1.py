"""Independent post-generation image/EEG/target join; no new sampling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import zipfile

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-name',required=True);a=p.parse_args()
    assert '/' not in a.run_name
    out=WORK/'images'/a.run_name
    completion=json.loads((out/'completion.json').read_text())
    assert completion['status']=='ALL_ACTUAL_LOSSLESS_IMAGES_SAVED_INFERENCE_ONLY'
    assert sha(out/'inference_manifest.jsonl')==completion['inference_manifest_sha256']
    targets={r['stimulus_id']:r for r in map(json.loads,(WORK/'data/generation_inputs'/(completion['split']+'_targets.jsonl')).read_text().splitlines())}
    rows=list(map(json.loads,(out/'inference_manifest.jsonl').read_text().splitlines()))
    assert len(rows)==len({r['query_id'] for r in rows})==completion['queries']
    assert len({r['generated_path'] for r in rows})==len(rows)
    assert len({r['generated_sha256'] for r in rows})==len(rows), 'Duplicate outputs within one arm need investigation'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2)
    inputs=json.loads((WORK/'data/generation_inputs'/(completion['split']+'.json')).read_text())
    ordered=inputs['inputs']
    positions={r['stimulus_id']:i for i,r in enumerate(ordered)}
    permutation=list(range(len(ordered)))
    rng=random.Random(1729)
    for i in range(len(permutation)-1,0,-1):
        j=rng.randrange(i)
        permutation[i],permutation[j]=permutation[j],permutation[i]
    assert all(i!=j for i,j in enumerate(permutation))
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    parent=WORK/'runs'/(completion['role']+'_source_dropout_v2')
    condition_lookup={}
    if completion['intervention']=='oracle':
        actual=torch.load(WORK/'data/fused_train_targets.pt',weights_only=True,map_location='cpu')
        for subject in inputs['subjects']:condition_lookup[subject]={int(q['array_row']):actual[q['array_row']:q['array_row']+1] for q in ordered}
    elif completion['intervention'] in ('correct','permuted') and completion['split']!='Tdiag':
        for subject in inputs['subjects']:
            exported=torch.load(parent/subject/'conditions.pt',weights_only=True,map_location='cpu')
            assert exported['model_sha256']==sha(parent/subject/'brain_final.pt')
            condition_lookup[subject]={int(q):exported['condition'][i:i+1] for i,q in enumerate(exported['array_rows'])}
    zero_eeg_expected={}
    if completion['intervention']=='missing':
        from brainhive_native_models_v1 import classes
        scope=classes()
        for subject in inputs['subjects']:
            model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval()
            model.load_state_dict(torch.load(parent/subject/'brain_final.pt',weights_only=True,map_location='cpu')['model'],strict=True)
            with torch.inference_mode():zero_eeg_expected[subject]=float(model(torch.zeros(1,17,250)).norm())
    missing_hashes={}
    cfg=json.loads((WORK/'assets/prior_H14/attn_adapter/config.json').read_text())
    expected_IP_calls=4*sum(n.endswith('attn2.processor') for n in cfg['attn_processor_keys'])
    from PIL import Image
    gallery=WORK/'gallery'/completion['split']
    gallery.mkdir(parents=True,exist_ok=True)
    real=[]
    for stimulus,target in targets.items():
        path=gallery/Path(stimulus).name
        if not path.exists():
            with zipfile.ZipFile(target['image_zip']) as z:data=z.read(target['image_member'])
            assert hashlib.sha256(data).hexdigest()==target['image_sha256']
            path.write_bytes(data)
        assert sha(path)==target['image_sha256']
        real.append(dict(stimulus_id=stimulus,image_path=str(path),image_sha256=target['image_sha256'],
                         concept_id=target['concept_id']))
    lookup={r['stimulus_id']:r for r in real}
    joined=[]
    for row in rows:
        target=targets[row['stimulus_id']]
        assert int(target['array_row'])==row['array_row']
        assert sha(row['generated_path'])==row['generated_sha256']
        with Image.open(row['generated_path']) as im:
            im.load();assert im.size==(512,512) and im.mode=='RGB'
        assert row['UNet_NFE']==4 and row['IP_conditioned_attention_calls']==expected_IP_calls
        assert row['VAE_decoder_calls']==1 and row['lossless_pixel_roundtrip']
        assert row['query_id']==f"{row['subject']}/{row['stimulus_id']}/{inputs['repeat_protocol']}"
        expected_seed=int(hashlib.sha256(f"1729|{row['subject']}|{row['stimulus_id']}|{inputs['repeat_protocol']}|{protocol['version']}".encode()).hexdigest()[:16],16)%(2**63-1)
        assert row['image_seed']==expected_seed
        q=ordered[positions[row['stimulus_id']]]
        assert row['array_row']==q['array_row']
        if completion['intervention']=='missing':
            assert row['EEG_source_stimulus_id']=='MISSING_ZERO_EEG'
            missing_hashes.setdefault(row['subject'],set()).add(row['condition_sha256'])
            assert abs(row['condition_norm']-zero_eeg_expected[row['subject']])<1e-3
        else:
            source=ordered[permutation[positions[row['stimulus_id']]]] if completion['intervention']=='permuted' else q
            assert row['EEG_source_stimulus_id']==source['stimulus_id']
            if row['subject'] in condition_lookup:
                value=condition_lookup[row['subject']][source['array_row']]
                assert hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()==row['condition_sha256']
        joined.append(row|dict(concept_id=target['concept_id'],target_image_sha256=target['image_sha256'],
                              score_target_path=lookup[row['stimulus_id']]['image_path'],
                              target_mapping_attached_after_inference=True))
    assert all(len(hashes)==1 for hashes in missing_hashes.values())
    manifest=out/'manifest.jsonl'
    assert not manifest.exists(),'Do not overwrite a completed join'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in joined))
    gallery_manifest=gallery/'manifest.jsonl'
    data=''.join(json.dumps(r)+'\n' for r in real)
    if gallery_manifest.exists():assert gallery_manifest.read_text()==data
    else:gallery_manifest.write_text(data)
    proof=dict(status='ALL_SAVED_IMAGES_TARGETS_AND_STIMULUS_ROWS_JOIN_VERIFIED',queries=len(rows),
        manifest_sha256=sha(manifest),gallery_manifest=str(gallery_manifest),gallery_manifest_sha256=sha(gallery_manifest),
        independent_raw_event_replay=False,sampled_new_images=False,scientific_image_gain_claimed=False,
        exact_cached_condition_row_SHA_verified=bool(condition_lookup),
        fixed_permutation_and_sampling_seed_verified=True,
        missing_condition_CPU_zero_EEG_norm_and_within_subject_constancy_verified=bool(zero_eeg_expected),
        required_IP_calls_per_image=expected_IP_calls)
    (out/'pairing_audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
