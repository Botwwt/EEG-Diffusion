"""Full artifact audit, actual frozen evaluator invocation, and fixed readout."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import brainhive_paired_study_contract_20261025_v1 as C


def manifest_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def own_rows(config,arm):
    output=C.DATA/'arms'/arm/'images';done=C.read(output/'completion.json')
    assert done['status']=='PAIRED_GENERATION_COMPLETE' and done['queries']==16600 and done['arm']==arm and done['split']=='V'
    assert done['study_core_sha256']==C.core(config)
    assert C.sha(done['manifest']['path'])==done['manifest']['sha256']
    rows=manifest_rows(done['manifest']['path']);assert len(rows)==len({r['query_id'] for r in rows})==16600
    from generate_BrainHIVE_paired_effect_20261025_v1 import infer_queries
    queries,_=infer_queries(config,'V')
    for index,(row,query) in enumerate(zip(rows,queries)):
        assert row['arm']==arm and row['global_query_index']==index
        for key in ('query_id','subject','stimulus_id','array_row','image_seed'):assert row[key]==query[key]
        assert row['EEG_source_stimulus_id']==row['stimulus_id']
        assert row['normal_inference_target_access'] is False and not row['teacher_or_negative_pair_at_inference']
        assert row['UNet_NFE']==4 and row['native_IP_K_calls']==row['native_IP_V_calls']==280
    return rows,done


def audit(config,arm):
    import numpy as np
    import torch
    from PIL import Image
    from audit_BrainHIVE_spatial_image_integrity_CPU_v1 import pixel_statistics
    rows,done=own_rows(config,arm);output=C.DATA/'arms'/arm/'artifact';output.mkdir(exist_ok=False)
    contract=C.bound(config['input_bindings']['input_contract'])
    sources={s['subject']:next(x for x in s['sources'] if x['role']=='full') for s in contract['subjects']}
    counts=collections.Counter();pixel_duplicates=collections.Counter();inodes=set();cache_key=None
    table=output/'per_query.csv.gz';failures=[]
    with gzip.open(table,'wt',newline='') as stream:
        writer=None
        for index,row in enumerate(rows):
            source=sources[row['subject']]
            if cache_key!=row['subject']:
                data=torch.load(source['condition_cache_path'],map_location='cpu',mmap=True,weights_only=True)
                lookup={int(value):i for i,value in enumerate(data['array_rows'])};cache_key=row['subject']
                assert data['model_sha256']==source['checkpoint_sha256']
            position=lookup[row['array_row']]
            assert position==row['condition_cache_row']
            assert row['condition_cache_sha256']==source['condition_cache_sha256']
            condition=data['condition'][position:position+1].contiguous()
            assert hashlib.sha256(condition.numpy().tobytes()).hexdigest()==row['condition_sha256']
            path=Path(row['generated_path']);expected=C.DATA/'arms'/arm/'images/pixels'/row['subject']/(Path(row['stimulus_id']).stem+'.webp')
            assert path==expected and C.sha(path)==row['generated_sha256']
            st=path.stat();inode=(st.st_dev,st.st_ino);assert inode not in inodes;inodes.add(inode)
            with Image.open(path) as image:
                image.load();assert image.size==(512,512);pixels=np.asarray(image.convert('RGB'))
            values=pixel_statistics(pixels,np);pixel_sha=hashlib.sha256(pixels.tobytes()).hexdigest();pixel_duplicates[pixel_sha]+=1
            record=dict(query_id=row['query_id'],subject=row['subject'],stimulus_id=row['stimulus_id'],
                array_row=row['array_row'],generated_sha256=row['generated_sha256'],pixel_sha256=pixel_sha,**values)
            if writer is None:writer=csv.DictWriter(stream,fieldnames=list(record));writer.writeheader()
            writer.writerow(record);counts['verified']+=1
            counts['constant']+=int(values['exact_spatial_constant']);counts['near_constant']+=int(values['near_spatial_constant'])
    assert counts['verified']==16600
    C.atomic(output/'completion.json',dict(status='FULL16600_PAIRED_ARTIFACT_AND_EEG_IDENTITY_AUDIT_COMPLETE',
        arm=arm,queries=16600,counts=dict(counts),duplicate_decoded_pixel_groups=sum(n>1 for n in pixel_duplicates.values()),
        full_denominator=16600,failures=failures,manifest=done['manifest'],table=C.ref(table),
        artifact_flags_are_not_perceptual_quality=True))


def score(config,arm):
    rows,done=own_rows(config,arm);root=C.DATA/'arms'/arm
    artifact=C.read(root/'artifact/completion.json');assert artifact['queries']==16600 and not artifact['failures']
    features=root/'features';feature_done=C.read(features/'completion.json');feature_cfg=C.read(features/'config.json')
    assert feature_done['status']=='COMPLETED' and feature_done['queries']==16600
    assert feature_done['manifest_sha256']==feature_cfg['manifest_sha256']==done['manifest']['sha256']
    assert feature_done['all_queries_used'] and not feature_done['query_subsampling']
    gallery_cfg=C.bound(config['input_bindings']['gallery_feature_config'])
    for key in ('DINO_weight_sha256','dino_preprocessing','SEED_source_commit','ObjectF1_grid','semantic_model_receipts'):
        assert feature_cfg[key]==gallery_cfg[key]
    output=root/'evaluation';output.mkdir(exist_ok=False);invocations=[]
    gallery=config['input_bindings']['gallery_manifest']['path']
    commands=[['score_full_manifest.py','--manifest',done['manifest']['path'],'--features',str(features),
        '--gallery-manifest',gallery,'--gallery-features',config['gallery_features_root'],'--output',str(output/'semantic')],
        ['score_low_level.py','--manifest',done['manifest']['path'],'--gallery-manifest',gallery,'--output',str(output/'low')]]
    for index,args in enumerate(commands):
        before=C.ref(done['manifest']['path']);log_path=output/('invocation_'+str(index)+'.log')
        with log_path.open('xb') as log:
            result=subprocess.run([sys.executable,'-B',*[str(C.ROOT/args[0]),*args[1:]]],cwd=C.ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=14000)
        invocation=dict(command=[sys.executable,'-B',str(C.ROOT/args[0]),*args[1:]],exit_code=result.returncode,
            log=C.ref(log_path),manifest_before=before,manifest_after=C.ref(done['manifest']['path']))
        invocations.append(invocation);C.atomic(output/'invocations.json',invocations)
        assert result.returncode==0 and invocation['manifest_before']==invocation['manifest_after']
    paths={};keys={r['query_id']:r for r in rows}
    for name in ('semantic','low'):
        path=output/name/'per_query.csv';seen=set()
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                identity=row['query_id'];assert identity in keys and identity not in seen;seen.add(identity)
                assert row['generated_path']==keys[identity]['generated_path']
                if name=='low':assert row['status']=='OK'
        assert len(seen)==16600
        paths[name]=dict(table=C.ref(path),summary=C.ref(output/name/'summary.json'))
    shards={str(p):C.sha(p) for p in features.rglob('*.pt')};assert shards
    C.atomic(output/'completion.json',dict(status='FULL16600_REAL_EVALUATOR_INVOCATIONS_CLOSED',arm=arm,queries=16600,
        manifest=done['manifest'],artifact=C.ref(root/'artifact/completion.json'),feature_completion=C.ref(features/'completion.json'),
        feature_config=C.ref(features/'config.json'),feature_shards=shards,outputs=paths,invocations=invocations,
        scalar_metric_values_not_used_to_select_or_branch=True))


def join_scores(manifest,semantic,low,metrics):
    lookup={r['query_id']:r for r in manifest};assert len(lookup)==len(manifest)
    joined={}
    for table in (semantic,low):
        seen=set()
        for row in table:
            key=row['query_id'];assert key in lookup and key not in seen;seen.add(key)
            actual=lookup[key];assert row['subject']==actual['subject'] and row['stimulus_id']==actual['stimulus_id']
            assert row['generated_path']==actual['generated_path']
            target_key='score_target_path' if 'score_target_path' in row else 'scoring_target_path'
            assert row[target_key]==actual['score_target_path']
            for metric in metrics:
                if metric in row:
                    value=float(row[metric]);assert math.isfinite(value);joined.setdefault(key,{})[metric]=value
        assert seen==set(lookup)
    assert all(set(v)==set(metrics) for v in joined.values())
    return [joined[row['query_id']] for row in manifest]


def compare(config):
    from compare_BrainHIVE_functional_V_CPU_v2 import METRICS,crossed_statistics
    deadline=time.monotonic()+1800;manifests={};values={};pins={};closures={};inodes=set()
    # All four invocation closures must exist before any scalar values are read.
    for arm in C.ARMS:
        root=C.DATA/'arms'/arm;closure=C.read(root/'evaluation/completion.json')
        assert closure['status']=='FULL16600_REAL_EVALUATOR_INVOCATIONS_CLOSED' and closure['queries']==16600
        for key in ('artifact','feature_completion','feature_config'):
            assert C.sha(closure[key]['path'])==closure[key]['sha256']
        for branch in closure['outputs'].values():
            for item in branch.values():assert C.sha(item['path'])==item['sha256'];pins[item['path']]=item['sha256']
        for p,s in closure['feature_shards'].items():
            assert C.sha(p)==s
            st=Path(p).stat();inode=(st.st_dev,st.st_ino);assert inode not in inodes;inodes.add(inode)
        for item in closure['invocations']:
            assert item['exit_code']==0 and C.sha(item['log']['path'])==item['log']['sha256']
            assert item['manifest_before']==item['manifest_after']==closure['manifest']
        rows,done=own_rows(config,arm);assert done['manifest']==closure['manifest']
        manifests[arm]=rows;closures[arm]=C.ref(root/'evaluation/completion.json')
        for row in rows:
            st=Path(row['generated_path']).stat();inode=(st.st_dev,st.st_ino)
            assert inode not in inodes and C.sha(row['generated_path'])==row['generated_sha256'];inodes.add(inode)
        post=C.read(root/'postfit_T32/completion.json');assert post['status']=='PAIRED_GENERATION_COMPLETE' and post['queries']==32
        assert C.sha(post['manifest']['path'])==post['manifest']['sha256']
    reference=manifests['O'];subjects=sorted({r['subject'] for r in reference});concepts=sorted({r['concept_id'] for r in reference})
    assert len(subjects)==10 and len(concepts)==166
    for arm in C.ARMS:
        for left,right in zip(reference,manifests[arm]):
            for key in ('query_id','subject','stimulus_id','array_row','image_seed','initial_UNet_input_sha256','target_image_sha256','score_target_path'):
                assert left[key]==right[key],(arm,key)
        tables={}
        for key,item in C.read(closures[arm]['path'])['outputs'].items():
            with Path(item['table']['path']).open(newline='') as stream:tables[key]=list(csv.DictReader(stream))
        values[arm]=join_scores(manifests[arm],tables['semantic'],tables['low'],METRICS)
    output=C.DATA/'comparison';output.mkdir(exist_ok=False);effects={}
    with gzip.open(output/'all66400_paired_differences.csv.gz','wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['contrast','query_id','subject','concept_id',*METRICS]);writer.writeheader()
        for arm,other in [('D','O'),('D','A'),('D','U'),('A','O')]:
            delta=[{k:a[k]-b[k] for k in METRICS} for a,b in zip(values[arm],values[other])]
            name=arm+'_minus_'+other
            effects[name],_=crossed_statistics(delta,reference,subjects,concepts,deadline=deadline)
            for row,value in zip(reference,delta):writer.writerow(dict(contrast=name,query_id=row['query_id'],subject=row['subject'],concept_id=row['concept_id'],**value))
    e=effects['D_minus_O']
    semantic_signal=e['DINO2AFC']['mean']>=.01 and e['DINO2AFC']['CI95_subject_and_concept_bootstrap'][0]>0 and e['SEED']['CI95_subject_and_concept_bootstrap'][0]>0 and e['ObjectF1']['mean']>=0
    low_guard=e['SSIM_Y']['mean']>=0 and e['PixCorr_RGB']['mean']>=0
    mechanism=all(effects[c][m]['CI95_subject_and_concept_bootstrap'][0]>0 for c in ('D_minus_A','D_minus_U') for m in ('DINO2AFC','SEED'))
    functional_path=C.DATA/'functional_T80/summary.json';functional_result='UNKNOWN';functional_ref=None
    if functional_path.exists():
        functional=C.read(functional_path)
        job=C.read(C.ROOT/'audit'/(config['functional_job_name']+'_job.json'))
        assert job['status']=='COMPLETED' and job['exit_code']==0
        assert functional['status']=='COMPLETE_FIXED_POSTFIT_PAIRED_T80' and functional['study_core_sha256']==C.core(config)
        functional_result=functional['paired_projection_increment']['result'];functional_ref=C.ref(functional_path)
    for p,s in pins.items():assert C.sha(p)==s
    C.atomic(output/'summary.json',dict(status='FOUR_FULL16600_PAIRED_EFFECT_COMPARISONS_COMPLETE',study_core_sha256=C.core(config),
        paired_differences=effects,evaluation_closures=closures,provisional_development_signal=bool(semantic_signal and low_guard and mechanism and functional_result=='SUPPORTED'),
        semantic_gate=bool(semantic_signal),mean_low_level_no_regression_guard=bool(low_guard),paired_mechanism_contrasts_positive=bool(mechanism),
        fixed_postfit_functional_result=functional_result,fixed_postfit_functional_summary=functional_ref,
        low_level_guard_is_not_human_quality_noninferiority=True,fixed_seed=1729,bootstrap_draws=1000,
        independent_confirmation=False,SOTA_established=False,automatic_new_candidate=False,all_differences=C.ref(output/'all66400_paired_differences.csv.gz')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--stage',choices=['audit','score','compare'],required=True);parser.add_argument('--arm',choices=C.ARMS)
    args=parser.parse_args();config=C.read(args.registration);C.validate(config);C.require_admission(config)
    if args.stage=='compare':compare(config)
    else:
        assert args.arm in C.ARMS
        {'audit':audit,'score':score}[args.stage](config,args.arm)


if __name__=='__main__':main()
