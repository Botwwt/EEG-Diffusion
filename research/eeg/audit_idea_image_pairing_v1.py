"""Independent CPU identity audit for actual new-encoder interface images."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import torch
from PIL import Image
from safetensors import safe_open

ROOT=Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def load(path):return json.loads(Path(path).read_text())


def tensor_sha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ['manifest','runtime-config','full-training-config','gallery-manifest','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--scores');args=parser.parse_args();started=time.monotonic();torch.set_num_threads(1)
    runtime_path=Path(args.runtime_config);runtime=load(runtime_path);config_path=Path(args.full_training_config);config=load(config_path)
    assert sha(config_path)==runtime['full_training_config_sha256']
    protocol=load(ROOT/'configs/split_manifest.json');assert sha(ROOT/'configs/split_manifest.json')==config['split_sha256']
    assert sha(ROOT/'audit/query_manifest_R4.csv')==runtime['query_manifest_sha256']
    queries=[q for q in csv.DictReader((ROOT/'audit/query_manifest_R4.csv').open()) if q['role']=='V']
    expected={q['query_id']:q for q in queries};assert len(expected)==16600
    by_stim={(q['subject_id'],q['stimulus_id']):q for q in queries}
    originals={int(q['array_row']):q for q in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if q['dataset_split']=='train'}
    gallery=[json.loads(line) for line in Path(args.gallery_manifest).read_text().splitlines()]
    g={q['stimulus_id']:q for q in gallery};assert len(g)==1660
    for row in gallery:assert sha(row['image_path'])==row['image_sha256']
    manifest=Path(args.manifest);rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows)==16600 and {r['query_id'] for r in rows}==expected.keys()
    assert len({r['generated_path'] for r in rows})==16600
    assert len({r['method'] for r in rows})==len({r['intervention'] for r in rows})==1
    assert all(r['partition']=='V' and r['repeat_protocol']=='R4' for r in rows)
    if args.scores:
        scores={r['query_id']:r for r in csv.DictReader(Path(args.scores).open())}
        assert len(scores)==16600 and scores.keys()==expected.keys()
    else:scores={}
    root=Path(config['output_root'])/'full';checkpoints={};conditions={}
    for subject in protocol['subjects']:
        proof=load(root/subject/'completion.json');assert proof['status']=='COMPLETED' and proof['prior'] is None
        assert sha(proof['encoder']['checkpoint'])==proof['encoder']['checkpoint_sha256']
        assert sha(proof['condition_cache'])==proof['condition_cache_sha256']
        bank=torch.load(proof['condition_cache'],map_location='cpu',weights_only=True,mmap=True)
        assert bank['encoder_sha256']==proof['encoder']['checkpoint_sha256'] and torch.equal(bank['array_rows'],torch.arange(16540))
        conditions[subject]=bank['condition'];checkpoints[subject]=proof['encoder']['checkpoint_sha256']
    first=rows[0];projection={}
    if first['projection_checkpoint']:
        assert sha(first['projection_checkpoint'])==first['projection_checkpoint_sha256']
        saved=torch.load(first['projection_checkpoint'],map_location='cpu',weights_only=True)
        projection={name:tensor_sha(value.to(torch.bfloat16)) for name,value in saved['state_dict'].items()}
    else:
        asset=Path('/dev/shm/eeg_diffusion_next_20261018_v1/adapter.safetensors')
        assert sha(asset)==load(ROOT/'configs/idea_first_error_validation_plan_v1.json')['native_projector_asset_sha256']
        with safe_open(str(asset),framework='pt',device='cpu') as file:
            for tail,source in [('image_embeds.weight','image_proj.proj.weight'),('image_embeds.bias','image_proj.proj.bias'),('norm.weight','image_proj.norm.weight'),('norm.bias','image_proj.norm.bias')]:
                projection['encoder_hid_proj.image_projection_layers.0.'+tail]=tensor_sha(file.get_tensor(source).to(torch.bfloat16))
    output=Path(args.output);assert not output.exists();output.mkdir(parents=True)
    failures=[];hashes=collections.defaultdict(list);verified=collections.Counter();cache_checked=0
    with (output/'per_query.csv').open('x',newline='') as file:
        fields=['query_id','subject','stimulus_id','concept_id','array_row','EEG_source_stimulus_id','generated_path','target_zip','target_member','target_sha256','score_target_path','native_cache_verified','status','error']
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for row in rows:
            q=expected[row['query_id']];target=g[row['stimulus_id']]
            record=dict(query_id=row['query_id'],subject=row['subject'],stimulus_id=row['stimulus_id'],concept_id=row['concept_id'],array_row=row['array_row'],
                EEG_source_stimulus_id=row['EEG_source_stimulus_id'],generated_path=row['generated_path'],target_zip=row['target_image_zip'],target_member=row['target_image_member'],
                target_sha256=row['target_image_sha256'],score_target_path='',native_cache_verified=False,status='OK',error='')
            try:
                original=originals[int(q['array_row'])]
                assert row['subject']==q['subject_id'] and row['stimulus_id']==q['stimulus_id']==original['stimulus_id']
                assert row['concept_id']==q['concept_id']==original['concept_id'] and row['array_row']==int(q['array_row'])
                for saved,key in [('target_image_zip','image_zip'),('target_image_member','image_member'),('target_image_sha256','image_sha256')]:
                    assert row[saved]==q[key]==original[key]
                assert row['target_image_sha256']==target['image_sha256']
                assert row['image_seed']==int(q['generation_seed'])
                assert row['encoder_sha256']==checkpoints[row['subject']] and row['full_training_config_sha256']==sha(config_path)
                assert row['image_interface_config_sha256']==sha(runtime_path)
                assert row['actual_deployed_projection_tensor_sha256']==projection
                assert row['projection_checkpoint']==first['projection_checkpoint']
                assert row['projection_checkpoint_sha256']==first['projection_checkpoint_sha256']
                assert row['steps']==20 and row['CFG']==5. and row['IP_scale']==1. and row['UNet_NFE']==40
                assert not row['target_image_or_visual_feature_read_at_inference'] and row['projected_native_image_tokens_per_query']==4
                source=row['EEG_source_stimulus_id']
                if row['intervention']=='correct':assert source==row['stimulus_id']
                if row['intervention']=='permuted':assert by_stim[(row['subject'],source)]['concept_id']!=row['concept_id']
                if row['intervention']=='missing':assert source is None and row['missing_input_definition']=='all-zero63x250 EEG through the actual frozen aligned encoder'
                else:
                    source_query=by_stim[(row['subject'],source)];value=conditions[row['subject']][int(source_query['array_row'])]
                    assert tensor_sha(value)==row['EEG_native_condition_sha256']==row['deployed_native_condition_sha256']
                    record['native_cache_verified']=True;cache_checked+=1
                path=Path(row['generated_path']);assert sha(path)==row['generated_sha256']
                with Image.open(path) as image:assert image.size==(512,512) and image.mode=='RGB';image.load()
                if scores:
                    score=scores[row['query_id']];assert score['stimulus_id']==row['stimulus_id'] and score['concept_id']==row['concept_id']
                    assert score['generated_path']==row['generated_path'] and score['score_target_path']==target['image_path']==score['target_image_path']
                    assert score['EEG_source_stimulus_id']==(source or '')
                    assert int(score['image_seed'])==row['image_seed'];record['score_target_path']=score['score_target_path']
                hashes[row['generated_sha256']].append(row['query_id']);verified[row['subject']]+=1
            except Exception as error:
                record.update(status='FAILED',error=repr(error));failures.append(record)
            writer.writerow(record)
    assert not torch.cuda.is_initialized()
    report=dict(status='PASSED_ALL16600_IDENTITY_AND_SAVED_FILES' if not failures else 'FAILED_SOME_QUERIES',expected_queries=16600,
        verified_queries=sum(verified.values()),failed_queries=len(failures),per_subject_verified=dict(verified),score_rows_checked=len(scores),
        native_EEG_cache_hash_verified_queries=cache_checked,duplicate_generated_hash_groups=sum(len(v)>1 for v in hashes.values()),
        manifest_sha256=sha(manifest),runtime_sha256=sha(runtime_path),full_training_config_sha256=sha(config_path),
        deployed_projection_tensor_sha256=projection,seconds=time.monotonic()-started,GPU_used=False,
        limits=['Release metadata/array/stimulus ownership is checked, not independent acquisition-event replay.',
                'Missing-input encoder output is not independently recomputed on CPU; its source code, checkpoint and recorded definition are checked.',
                'Correct pairing and image decoding do not certify correct objects or a mechanism gain.'])
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
