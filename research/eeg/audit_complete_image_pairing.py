"""Independent, CPU-only reconciliation of complete formal image manifests.

Reads the actual original metadata/query tables, frozen condition caches and
per-image receipts. Does not generate, score, select or repair any image. The
released EEG event labels are unavailable; this proves release-order identity,
not independent acquisition-event replay.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import time

import torch
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def load(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def seed(text):return int(hashlib.sha256(text.encode()).hexdigest()[:16],16)%(2**63-1)


def require(condition,message):
    if not condition:raise ValueError(message)


def expected_sources(queries,subject,partition,R,intervention):
    if intervention=='correct':return {q['stimulus_id']:q['stimulus_id'] for q in queries}
    if intervention=='missing':return {q['stimulus_id']:None for q in queries}
    require(intervention=='permuted','Unknown intervention')
    groups=collections.defaultdict(list)
    for q in queries:groups[q['concept_id']].append(q)
    concepts=sorted(groups)
    offset=1+int(hashlib.sha256(f'1729|EEG-permutation|{subject}|{partition}|{R}'.encode()).hexdigest()[:16],16)%(len(concepts)-1)
    result={}
    for position,c in enumerate(concepts):
        others=groups[concepts[(position+offset)%len(concepts)]]
        for target,source in zip(sorted(groups[c],key=lambda v:int(v['array_row'])),sorted(others,key=lambda v:int(v['array_row']))):
            require(target['concept_id']!=source['concept_id'],'Permutation retained concept')
            result[target['stimulus_id']]=source['stimulus_id']
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--scores')
    args=parser.parse_args();started=time.monotonic();torch.set_num_threads(1)
    manifest=Path(args.manifest).resolve();completion=load(manifest.parent/'completion.json')
    require(completion['status']=='COMPLETED' and completion['image_manifest_sha256']==sha(manifest),'Incomplete or changed image manifest')
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    require(len(rows) in [16600,2000] and len({v['query_id'] for v in rows})==len(rows),'Incomplete/duplicate query population')
    first=rows[0];partition=first['partition'];R=first['repeat_protocol'];method=first['method'];intervention=first['intervention']
    require(partition in ['V','benchmark'] and R in ['R1','R4','R80'],'Unsupported partition/protocol')
    for key in ['partition','repeat_protocol','method','intervention']:
        require(len({v[key] for v in rows})==1,'Mixed arm: '+key)
    protocol=load(ROOT/'configs/split_manifest.json');config=load(ROOT/'configs/full_reference_resolved.json')
    require(sha(ROOT/'configs/split_manifest.json')==config['split_sha256'],'Split changed')
    if partition=='benchmark':
        unlock=load(ROOT/'configs/FINAL_EVAL_UNLOCK.json')
        require(unlock['method_and_checkpoint_selection_frozen'] and unlock['split_sha256']==config['split_sha256'],'Final test remains locked')
    else:require(R=='R4','Unregistered development repeat protocol')
    query_path=ROOT/'audit/query_manifest_R4.csv';image_path=ROOT/'audit/image_manifest.csv'
    with query_path.open() as file:
        queries=[q for q in csv.DictReader(file) if (q['role']=='V' if partition=='V' else q['dataset_split']=='test')]
    with image_path.open() as file:
        originals={int(q['array_row']):q for q in csv.DictReader(file) if q['dataset_split']==('train' if partition=='V' else 'test')}
    require(len(queries)==len(rows),'Expected population size differs')
    expected={(q['subject_id'],q['stimulus_id']):q for q in queries}
    require(len(expected)==len(queries),'Duplicate original query identity')
    subjects=protocol['subjects'];counts=collections.Counter(q['subject'] for q in rows)
    require(set(counts)==set(subjects) and set(counts.values())=={len(rows)//10},'Incomplete subject coverage')
    destination=Path(args.output);destination.mkdir(parents=True,exist_ok=False)
    output=destination/'per_query.csv';errors=[];verified=collections.Counter();hash_groups=collections.defaultdict(list)
    paths=set();sources={};cached={};cache_proofs=[];checkpoints={};native_verified=0
    gallery_path=DATA/'evaluation/gallery'/partition/'manifest.jsonl'
    gallery={g['stimulus_id']:g for g in map(json.loads,gallery_path.read_text().splitlines())} if gallery_path.exists() else {}
    for target in gallery.values():require(sha(target['image_path'])==target['image_sha256'],'Scoring target file bytes changed')
    score_map={}
    if args.scores:
        require(bool(gallery),'A complete scoring gallery is required for score-path reconciliation')
        with Path(args.scores).open() as file:score_rows=list(csv.DictReader(file))
        require(len(score_rows)==len(rows) and len({q['query_id'] for q in score_rows})==len(rows),'Incomplete/duplicate score rows')
        score_map={q['query_id']:q for q in score_rows}
        require(set(score_map)=={q['query_id'] for q in rows},'Score query identities differ')
    for subject in subjects:
        subject_queries=[q for q in queries if q['subject_id']==subject]
        sources[subject]=expected_sources(subject_queries,subject,partition,R,intervention)
        proof=load(Path(config['output_root'])/'full'/subject/'completion.json')
        require(sha(proof['encoder']['checkpoint'])==proof['encoder']['checkpoint_sha256'],'Encoder checkpoint changed')
        checkpoints[subject]=dict(encoder=proof['encoder']['checkpoint_sha256'])
        if method=='REF_PRIOR_V':
            repair=load(ROOT/'configs/prior_velocity_repair_resolved.json')
            prior=load(Path(repair['output_root'])/'full'/subject/'completion.json')['prior']
        else:prior=proof['prior']
        if method!='REF_REG':
            require(sha(prior['checkpoint'])==prior['checkpoint_sha256'],'Prior checkpoint changed')
            checkpoints[subject]['prior']=prior['checkpoint_sha256']
        if partition=='V' and intervention!='missing':
            require(sha(proof['condition_cache'])==proof['condition_cache_sha256'],'Frozen condition cache changed')
            value=torch.load(proof['condition_cache'],weights_only=True,mmap=True,map_location='cpu')
            require(value['encoder_sha256']==proof['encoder']['checkpoint_sha256'],'Cache belongs to another encoder')
            require(value['condition'].dtype==torch.float32 and value['condition'].shape[1:]==(1024,),'Wrong native condition space')
            positions={int(row):i for i,row in enumerate(value['array_rows'])}
            # Cache is joined by the stimulus source's original row, never by
            # gallery sort position or numeric concept-prefix arithmetic.
            for query in subject_queries:
                source=sources[subject][query['stimulus_id']]
                source_query=expected[(subject,source)];index=int(source_query['array_row'])
                cached[(subject,query['stimulus_id'])]=tensor_hash(value['condition'][positions[index]])
            cache_proofs.append(dict(subject=subject,path=proof['condition_cache'],sha256=proof['condition_cache_sha256']))
            del value
    with output.open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['query_id','subject','stimulus_id','concept_id','array_row','EEG_source_stimulus_id',
            'target_zip','target_member','target_sha256','generated_path','generated_sha256','score_target_path',
            'image_seed','prior_seed','EEG_native_cache_hash_checked','status','error'])
        writer.writeheader()
        for position,row in enumerate(rows):
            record=dict(query_id=row['query_id'],subject=row['subject'],stimulus_id=row['stimulus_id'],concept_id=row['concept_id'],
                array_row=row['array_row'],EEG_source_stimulus_id=row['EEG_source_stimulus_id'],target_zip=row['target_image_zip'],
                target_member=row['target_image_member'],target_sha256=row['target_image_sha256'],generated_path=row['generated_path'],
                generated_sha256=row['generated_sha256'],score_target_path='',image_seed=row['image_seed'],prior_seed=row['prior_seed'],
                EEG_native_cache_hash_checked=False,status='OK',error='')
            try:
                subject=row['subject'];stimulus=row['stimulus_id'];query=expected[(subject,stimulus)]
                original=originals[int(query['array_row'])]
                require(original['stimulus_id']==stimulus and original['concept_id']==query['concept_id']==row['concept_id'],'Stimulus/concept/row mixed')
                require(row['array_row']==int(query['array_row']) and row['query_id']==f'{subject}/{stimulus}/{R}','Array row or query identity differs')
                for target_field,source_field in [('target_image_zip','image_zip'),('target_image_member','image_member'),('target_image_sha256','image_sha256')]:
                    require(row[target_field]==query[source_field]==original[source_field],'Target origin differs: '+target_field)
                require(row['EEG_source_stimulus_id']==sources[subject][stimulus],'Wrong EEG source stimulus')
                image_seed=seed(f'1729|{subject}|{stimulus}|{R}|{protocol["version"]}')
                require(row['image_seed']==image_seed and row['prior_seed']==seed(f'{image_seed}|prior'),'Wrong image/prior RNG stream')
                require(row['encoder_sha256']==checkpoints[subject]['encoder'],'Wrong encoder checkpoint')
                require(row['prior_sha256']==checkpoints[subject].get('prior'),'Wrong prior checkpoint')
                if (subject,stimulus) in cached:
                    require(row['EEG_native_condition_sha256']==cached[(subject,stimulus)],'EEG condition does not match source cache row')
                    record['EEG_native_cache_hash_checked']=True;native_verified+=1
                    if method=='REF_REG':require(row['deployed_native_condition_sha256']==cached[(subject,stimulus)],'Point native condition changed at deployment')
                require(row['steps']==20 and row['CFG']==5. and row['IP_scale']==1. and row['UNet_NFE']==40,'Sampling protocol differs')
                require(row['projected_native_image_tokens_per_query']==4 and not row['target_image_or_visual_feature_read_at_inference'],'Wrong/leaked image interface')
                path=Path(row['generated_path']).resolve()
                require(path.is_relative_to(manifest.parent) and path not in paths,'Output escapes arm or path reused')
                paths.add(path);require(sha(path)==row['generated_sha256'],'Saved file SHA differs')
                with Image.open(path) as image:image.load();require(image.size==(512,512) and image.mode=='RGB','Saved output shape/mode differs')
                receipt=load(manifest.parent/'receipts'/subject/f'{row["array_row"]:08d}.json')
                require(receipt==row,'Per-image receipt differs from manifest')
                hash_groups[row['generated_sha256']].append(row['query_id'])
                if gallery:
                    target=gallery[stimulus]
                    require(target['image_sha256']==row['target_image_sha256'] and target['array_row']==row['array_row'],'Scoring gallery target identity differs')
                    record['score_target_path']=target['image_path']
                if score_map:
                    score=score_map[row['query_id']]
                    for k in ['subject','stimulus_id','concept_id','generated_path']:
                        require(score[k]==row[k],'Scoring query identity differs: '+k)
                    require(score['EEG_source_stimulus_id']==(row['EEG_source_stimulus_id'] or ''),'Scoring EEG source differs')
                    require(score['image_seed']==str(image_seed),'Scoring random stream differs')
                    require(score['score_target_path']==score['target_image_path']==record['score_target_path'],'Scoring target path differs')
                verified[subject]+=1
            except Exception as error:
                record.update(status='FAILED_IDENTITY_OR_FILE_CHECK',error=repr(error));errors.append(record)
            writer.writerow(record)
            if (position+1)%2000==0:print(json.dumps(dict(stage='pairing_CPU',verified=sum(verified.values()),processed=position+1,expected=len(rows))),flush=True)
    duplicates={h:q for h,q in hash_groups.items() if len(q)>1}
    report=dict(status='PASSED_COMPLETE_RELEASE_IDENTITY_AND_SAVED_FILES' if not errors else 'FAILED_COMPLETE_IDENTITY_AUDIT_NO_REPAIR',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),expected_queries=len(rows),verified_queries=sum(verified.values()),per_subject_verified=dict(verified),
        method=method,intervention=intervention,partition=partition,repeat_protocol=R,failed_queries=len(errors),
        unique_generated_paths=len(paths),duplicate_generated_hash_groups=len(duplicates),duplicate_images_beyond_first=sum(len(v)-1 for v in duplicates.values()),
        native_EEG_cache_hash_verified_queries=native_verified,score_rows_checked=len(score_map),gallery_identity_checked=bool(gallery),
        manifest_sha256=sha(manifest),query_manifest_sha256=sha(query_path),image_manifest_sha256=sha(image_path),
        source_sha256=sha(__file__),cache_proofs=cache_proofs,checkpoints=checkpoints,
        GPU_used=False,images_regenerated=False,scores_computed=False,independent_acquisition_event_replay=False,
        limits=['Release-order/metadata identity is verified; original acquisition event labels are not present in the available release.',
            'Benchmark encoder outputs and missing-input encoder outputs are not numerically recomputed by this CPU identity audit.',
            'A prior-sampled deployed condition is not independently resampled here; cached encoder input hash is checked where available.',
            'Pairing/file correctness does not prove successful reconstruction or model capacity adequacy.'],seconds=time.monotonic()-started)
    (destination/'failures.json').write_text(json.dumps(errors,indent=2)+'\n')
    (destination/'duplicate_hash_groups.json').write_text(json.dumps(duplicates,indent=2)+'\n')
    (destination/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
