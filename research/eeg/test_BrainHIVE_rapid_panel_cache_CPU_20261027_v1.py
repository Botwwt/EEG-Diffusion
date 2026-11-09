"""CPU checks for deterministic panel metadata and zero-copy cache contracts."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import brainhive_rapid_panel_cache_20261027_v1 as K
import cache_BrainHIVE_rapid_DINO_features_20261027_v1 as F
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v1 as S


def run(panel_path):
    panel=K.validate_panel(panel_path);old=K.read(panel['old_panel']['path'])
    assert panel['queries'][:80]==old['queries']
    assert len(panel['queries'])==len(panel['target_gallery'])==400
    metadata=K.G.metadata(panel['image_manifest']['path']);split=K.read(panel['split']['path'])
    index=K.rows(panel['index']['path'])
    permutation=np.random.default_rng(1729).permutation(len(index))
    reordered=[index[int(i)] for i in permutation]
    selected,additional=K.select(reordered,split,metadata,old)
    assert selected==panel['queries'] and additional==panel['additional_concepts']
    assert all(set(additional[str(fold)]).isdisjoint(old['selected_concepts'][str(fold)]) for fold in (0,1))
    assert all(len(additional[str(fold)])==16 for fold in (0,1))
    assert {row['array_row'] for row in selected}<=set(split['image_rows']['T'])
    assert all(stage['target_gallery_query_count']==400 for stage in panel['stages'].values())
    assert panel['selection_identity_sha256']==K.semantic_digest(selected)
    broken=copy.deepcopy(old);broken['queries'][0]['image_seed']+=1
    try:K.select(index,split,metadata,broken);raise RuntimeError('legacy seed mutation accepted')
    except AssertionError:pass
    broken_metadata=copy.deepcopy(metadata);broken_metadata[selected[-1]['array_row']]['stimulus_id']='wrong.jpg'
    try:K.select(index,split,broken_metadata,old);raise RuntimeError('source image mismatch accepted')
    except AssertionError:pass
    # Exercise the actual common scorer loader against exported compatibility
    # manifests. These are explicitly synthetic features, not real GPU results.
    with tempfile.TemporaryDirectory(prefix='brainhive_rapid_cpu_') as directory:
        output=Path(directory);baseline=output/'baseline.npy';target=output/'target.npy'
        values=np.zeros((400,768),np.float32);values[np.arange(400),np.arange(400)]=1
        F.save_npy(baseline,values);F.save_npy(target,values)
        ordered=[]
        for i,query in enumerate(panel['queries']):
            ordered.append(query|dict(target_image_sha256=panel['target_gallery'][i]['image_sha256'],generated_sha256=f'{i:064x}',
                initial_noise_latents_sha256=f'{i+1000:064x}',initial_UNet_input_sha256=f'{i+2000:064x}',
                condition_sha256=f'{i+3000:064x}',condition_FP16_sha256=f'{i+4000:064x}',tokens_sha256=f'{i+5000:064x}',generated_pixel_sha256=f'{i+6000:064x}'))
        dummy=K.ref(__file__)
        provenance=dict(host={'CPU_synthetic_test':True},baseline_completion=dummy,DINO_weight=dummy,
            image_evaluation_source=dummy,locked_metrics_source=dummy)
        views=F.compatible_views(output,K.ref(panel_path),baseline,target,ordered,provenance)
        actual_panel=S.load_panel(panel_path)
        a=S.load_cache(views['target']['path'],panel_path,actual_panel,'targets',require_full=True)
        b=S.load_cache(views['native']['path'],panel_path,actual_panel,'generated',require_full=True)
        assert a['features'].shape==b['features'].shape==(400,768)
        assert a['document']['features']['path']==str(target.resolve()) and b['document']['features']['path']==str(baseline.resolve())
        stage=S.metric_arrays(b['features'][:80],a['features'],panel['queries'][:80],panel['target_gallery'])
        assert np.all(stage['DINO2AFC']==1) and np.all(stage['within_concept10_top1']==1)
        assert len(stage['target_cosine'])==80
        bad_rows=copy.deepcopy(a['rows']);bad_rows[0]['array_row']=-1
        bad_path=output/'bad_rows.json';K.atomic(bad_path,bad_rows)
        bad_doc=copy.deepcopy(a['document']);bad_doc['rows']=K.ref(bad_path)
        bad_manifest=output/'bad_cache.json';K.atomic(bad_manifest,bad_doc)
        try:S.load_cache(bad_manifest,panel_path,actual_panel,'targets',require_full=True);raise RuntimeError('cache identity mismatch accepted')
        except AssertionError:pass
    return dict(status='RAPID_PANEL_AND_FEATURE_CONTRACT_CPU_CHECKS_PASSED',checks=10,
        panel=K.ref(panel_path),selection_identity_sha256=panel['selection_identity_sha256'],
        prefix80_dict_and_seed_exact=True,all40_concepts_have10_stimuli=True,
        stage80_gallery_size=400,stage400_gallery_size=400,zero_copy_NPY_views_verified=True,
        model_weights_loaded=False,actual_images_generated=0,GPU_initialized=False,
        scope='actual metadata selection plus synthetic feature-cache/scorer integration; no actual native-cache pixel/GPU execution')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--panel',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();result=run(args.panel)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    result['source_pins']={path.name:K.ref(path) for path in (Path(__file__),Path(K.__file__),Path(F.__file__),Path(S.__file__),
        Path(__file__).with_name('generate_BrainHIVE_rapid_native_cache_20261027_v1.py'))}
    K.atomic(args.receipt,result);print(json.dumps(result))
