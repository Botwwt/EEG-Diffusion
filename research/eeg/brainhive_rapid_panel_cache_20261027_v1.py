"""CPU-only deterministic T400 panel and shared rapid-cache contracts.

The old eighty queries are an exact prefix. Sixteen additional concepts per
outer fold are selected without opening any prediction, pixel, or score file.
Both screening stages use the same complete four-hundred-target gallery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import generate_BrainHIVE_trialset_fastscreen_20261027_v1 as G

ROOT=Path(__file__).resolve().parent
SCHEMA='BrainHIVE_rapid_panel400_20261027_v1'
PANEL_STATUS='FIXED_SINGLE_SUBJECT_T400_WITH_REUSED_T80_PREFIX'
BASELINE_SCHEMA='BrainHIVE_rapid_native_baseline_cache_20261027_v1'
BASELINE_STATUS='COMPLETE_RAPID_NATIVE_BASELINE_CACHE'
FEATURE_SCHEMA='BrainHIVE_rapid_panel_DINO_feature_cache_20261027_v1'
FEATURE_STATUS='COMPLETE_RAPID_PANEL400_NATIVE_AND_TARGET_DINO_CACHE'
DINO_SHA='08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'
TRANSFORM='image_evaluation.dino_features: RGB/255; 256x256 bilinear antialias align_corners=False; real CPU resize/generated CUDA resize; ImageNet mean/std; normalized layer11 CLS+mean-patch concatenation; L2-normalized768'
sha,ref,read,rows,atomic=G.sha,G.ref,G.read,G.rows,G.atomic
IDENTITY_KEYS=('row','source_row','array_row','subject_id','concept_id','stimulus_id','fold','query_id','image_seed')


def verified(binding):
    path=Path(binding['path']);assert sha(path)==binding['sha256'],f'SHA mismatch: {path}'
    return path


def select(index,split,metadata,legacy):
    old=G.choose_panel(index,split,metadata)
    assert legacy['queries']==old['queries'] and legacy['selected_concepts']==old['selected_concepts']
    assert legacy['expected_queries']==80 and len(legacy['queries'])==80
    clean=[G.identity(row) for row in index]
    concepts={}
    for row in clean:concepts.setdefault(row['concept_id'],[]).append(row)
    additional={};selected=[dict(row) for row in legacy['queries']]
    for fold in (0,1):
        eligible=[name for name,group in concepts.items() if group[0]['fold']==fold and name not in old['selected_concepts'][str(fold)]]
        ranked=sorted(eligible,key=lambda name:(hashlib.sha256(f'{SCHEMA}|additional|1729|fold{fold}|{name}'.encode()).hexdigest(),name))[:16]
        assert len(ranked)==16
        additional[str(fold)]=ranked
        for concept in ranked:
            for item in sorted(concepts[concept],key=lambda row:row['array_row']):
                query=dict(item)
                # Preserve the established query/noise naming convention for
                # every query, including the immutable legacy prefix.
                label=f"{G.SCHEMA}|1729|{query['subject_id']}|{query['stimulus_id']}"
                query['query_id']=hashlib.sha256(label.encode()).hexdigest()
                query['image_seed']=int(hashlib.sha256((label+'|noise').encode()).hexdigest()[:15],16)
                selected.append(query)
    assert len(selected)==400 and len({row['query_id'] for row in selected})==400
    assert selected[:80]==legacy['queries']
    groups={}
    for query in selected:
        item=metadata[query['array_row']]
        assert item['concept_id']==query['concept_id'] and item['stimulus_id']==query['stimulus_id']
        groups.setdefault(query['concept_id'],[]).append(query)
    assert len(groups)==40 and all(len(group)==10 for group in groups.values())
    assert {fold:sum(group[0]['fold']==fold for group in groups.values()) for fold in (0,1)}=={0:20,1:20}
    return selected,additional


def semantic_digest(queries):
    return hashlib.sha256(json.dumps(queries,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def target_gallery(queries,metadata):
    return [query|dict(target_index=i,image_sha256=metadata[query['array_row']]['image_sha256'],
        image_zip=metadata[query['array_row']]['image_zip'],image_member=metadata[query['array_row']]['image_member'],
        image_bytes=int(metadata[query['array_row']]['image_bytes'])) for i,query in enumerate(queries)]


def validate_panel(path):
    panel=read(path)
    assert panel['schema']==SCHEMA and panel['status']==PANEL_STATUS
    assert panel['expected_queries']==len(panel['queries'])==len(panel['target_gallery'])==400
    assert panel['legacy_prefix_queries']==80 and panel['new_queries']==320
    for key in ('index','split','image_manifest','old_panel'):verified(panel[key])
    old=read(panel['old_panel']['path']);metadata=G.metadata(panel['image_manifest']['path'])
    selected,additional=select(rows(panel['index']['path']),read(panel['split']['path']),metadata,old)
    assert panel['queries']==selected and panel['additional_concepts']==additional
    assert panel['target_gallery']==target_gallery(selected,metadata)
    assert panel['selection_identity_sha256']==semantic_digest(selected)
    for count in (80,400):
        stage=panel['stages'][str(count)]
        assert stage==dict(query_count=count,query_ids=[row['query_id'] for row in selected[:count]],target_gallery_query_count=400)
    return panel


def prepare(args):
    assert not args.output.exists();args.output.mkdir(parents=True)
    legacy=read(args.old_panel)
    for key,path in [('index',args.index),('split',args.split),('image_manifest',args.image_manifest)]:
        assert sha(path)==legacy[key]['sha256']
    metadata=G.metadata(args.image_manifest)
    selected,additional=select(rows(args.index),read(args.split),metadata,legacy)
    panel=dict(schema=SCHEMA,status=PANEL_STATUS,created_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        expected_queries=400,legacy_prefix_queries=80,new_queries=320,subjects=1,folds=2,concepts_per_fold=20,
        stimuli_per_concept=10,selection_seed=1729,queries=selected,target_gallery=target_gallery(selected,metadata),
        additional_concepts=additional,legacy_concepts=legacy['selected_concepts'],selection_identity_sha256=semantic_digest(selected),
        stages={str(count):dict(query_count=count,query_ids=[row['query_id'] for row in selected[:count]],target_gallery_query_count=400) for count in (80,400)},
        index=ref(args.index),split=ref(args.split),image_manifest=ref(args.image_manifest),old_panel=ref(args.old_panel),
        source=ref(__file__),panel_helper=ref(G.__file__),results_or_predictions_read=False,pixel_files_opened=False,
        prefix80_already_exposed=True,new320_selected_before_future_effects=True,development_only=True,independent_confirmation=False)
    atomic(args.output/'panel.json',panel)
    contract=dict(schema='BrainHIVE_rapid_panel_cache_contract_20261027_v1',status='FIXED_BEFORE_NEW_BASELINE_GENERATION',
        panel=ref(args.output/'panel.json'),selection_identity_sha256=panel['selection_identity_sha256'],
        baseline_completion=dict(schema=BASELINE_SCHEMA,status=BASELINE_STATUS,queries=400,
            reused_existing_images=80,new_query_images=320,parity_images=1,total_new_images=321,UNet_evaluations_per_new_image=4,
            required_bindings=['panel','native_parent_binding','legacy_completion','input_provenance','manifest','host','loader_proof','actual_native_pixel_parity'],
            per_image_required=['query_id','source_row','condition_sha256','condition_FP16_sha256','tokens_sha256','initial_noise_latents_sha256',
                'initial_UNet_input_sha256','generated_sha256','generated_pixel_sha256','target_image_sha256']),
        feature_cache=dict(schema=FEATURE_SCHEMA,status=FEATURE_STATUS,feature_shape=[400,768],
            DINO_weight_sha256=DINO_SHA,transform=TRANSFORM,
            required_bindings=['source','image_evaluation_source','locked_metrics_source','baseline_completion','host','panel','feature_index'],
            feature_index_required=['feature_row','query_id','array_row','generated_sha256','generated_pixel_sha256','target_image_sha256'],
            arrays=['baseline_features.npy','target_features.npy'],feature_reuse_condition='all SHA/source/weight/transform/identity/image hashes match; otherwise no reuse'),
        stage_gallery_rule='Both80and400 stages use the same immutable400target features; never compare or reuse historical80gallery scores',
        normal_generation_target_access=False,no_EEG_refit=True,no_F_access=True,no_new_GPU_execution=True)
    atomic(args.output/'cache_contract.json',contract)
    validate_panel(args.output/'panel.json')
    print(json.dumps(dict(status='CPU_PANEL400_AND_CACHE_CONTRACT_PREPARED',panel=ref(args.output/'panel.json'),
        cache_contract=ref(args.output/'cache_contract.json'),selection_identity_sha256=panel['selection_identity_sha256'],
        queries=400,old_prefix_preserved=80,newly_selected_queries=320,actual_new_images_generated=0,starts_GPU=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for name in ('index','split','image-manifest','old-panel','output'):prep.add_argument('--'+name,type=Path,required=True)
    valid=sub.add_parser('validate');valid.add_argument('--panel',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args)
    else:
        panel=validate_panel(args.panel)
        print(json.dumps(dict(status='CPU_PANEL400_VALIDATED',queries=len(panel['queries']),selection_identity_sha256=panel['selection_identity_sha256'],starts_GPU=False)))


if __name__=='__main__':main()
