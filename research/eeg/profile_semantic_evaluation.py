"""Actual frozen evaluator load, timing, self-pair, and streaming equivalence checks."""
from semantic_evaluator import *
import csv
import gc
import hashlib
import io
import json
import time
import traceback
import zipfile
from scipy.spatial.distance import correlation

OUT=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runs/S0_semantic_evaluator')


def main():
    torch.set_num_threads(2);torch.manual_seed(1729)
    started=time.monotonic();assert not OUT.exists();OUT.mkdir(parents=True)
    throughput=json.loads((ROOT/'audit/generator_throughput_profile.json').read_text())
    case=throughput['cases'][0];rows=case['array_rows'][:8]
    manifest={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    targets=[];generated=[]
    with zipfile.ZipFile('/root/autodl-tmp/mced_research_bundle/external/things_eeg2_direct_v1/training_images.zip') as archive:
        for i,row in enumerate(rows):
            raw=archive.read(manifest[row]['image_member']);assert hashlib.sha256(raw).hexdigest()==manifest[row]['image_sha256']
            p=OUT/f'{i:08d}.jpg';p.write_bytes(raw);targets.append(p)
            generated.append(Path(case['files'][i]['path']))
    paths=targets+generated
    report=dict(status='IN_PROGRESS',formal_evaluation=False,array_rows=rows,queries=8,
                seed_source_commit='5327a134b90b04fb1b37066d98b309c9cc533a02',cases={})
    def save():
        (ROOT/'audit/semantic_evaluation_profile.json').write_text(json.dumps(report,indent=2)+'\n')
    model,preprocess=effnet_model();torch.cuda.synchronize();tick=time.monotonic()
    eff=effnet_features(paths,model,preprocess,batch_size=8);torch.cuda.synchronize()
    eff_seconds=time.monotonic()-tick
    score=correlation_similarity(eff[:8],eff[8:])
    reference=np.array([1-correlation(a,b) for a,b in zip(eff[:8],eff[8:])])
    assert np.allclose(score,reference,atol=1e-12)
    assert np.allclose(correlation_similarity(eff[:8],eff[:8]),1,atol=1e-12)
    # Official tensor-batch preprocessing versus streamed preprocessing on uniform native images.
    original=[]
    for p in targets:
        with Image.open(p) as im:original.append(torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1))
    if len({tuple(t.shape) for t in original})==1:
        with torch.inference_mode():
            full=model(preprocess(torch.stack(original).float().div_(255)).cuda())['avgpool'].flatten(1).cpu().numpy()
        difference=float(np.abs(full-eff[:8]).max());assert np.allclose(full,eff[:8],atol=2e-5,rtol=2e-4)
    else:difference=None
    report['cases']['EffNet']=dict(seconds=eff_seconds,seconds_per_image=eff_seconds/len(paths),
        oracle_scores=score.tolist(),self_scores=correlation_similarity(eff[:8],eff[:8]).tolist(),
        streamed_vs_official_feature_max_abs=difference,Pearson_vs_scipy_max_abs=float(abs(score-reference).max()))
    save();del model;gc.collect();torch.cuda.empty_cache()
    processor,model,text,info=caption_models();torch.cuda.synchronize();tick=time.monotonic()
    captions,cap=caption_features(paths,processor,model,text,batch_size=8);torch.cuda.synchronize()
    seconds=time.monotonic()-tick
    report['cases']['CapSim']=dict(seconds=seconds,seconds_per_image=seconds/len(paths),
        captions=captions,oracle_scores=np.sum(cap[:8]*cap[8:],axis=-1).tolist(),
        self_scores=np.sum(cap[:8]*cap[:8],axis=-1).tolist(),load_info=info)
    save();del processor,model,text;gc.collect();torch.cuda.empty_cache()
    detector,classes,cfg=detection_model();cfg.dump(str(OUT/'resolved_detector.py'))
    torch.cuda.synchronize();tick=time.monotonic()
    predictions=detection_features(paths,detector,classes,batch_size=4);torch.cuda.synchronize()
    seconds=time.monotonic()-tick
    folders={}
    for name,values in [('gt',predictions[:8]),('oracle',predictions[8:]),('self',predictions[:8]),('shuffle',predictions[1:8]+predictions[:1])]:
        p=OUT/name/'preds';p.mkdir(parents=True);folders[name]=p.parent
        for i,values_i in enumerate(values):(p/f'{i:08d}.json').write_text(json.dumps(values_i)+'\n')
    from seed.metrics import eval_object_f1
    self_scores=eval_object_f1(str(folders['self']),str(folders['gt']))
    assert np.array_equal(self_scores,np.ones(8))
    report['cases']['ObjectF1']=dict(seconds=seconds,seconds_per_image=seconds/len(paths),
        classes=list(classes),thresholds=np.arange(.05,.95,.05).tolist(),area_threshold_pixels=50,
        variant='official_public_code_default_grid',self_scores=self_scores.tolist(),
        oracle_scores=eval_object_f1(str(folders['oracle']),str(folders['gt'])).tolist(),
        shuffled_target_scores=eval_object_f1(str(folders['shuffle']),str(folders['gt'])).tolist(),
        predictions_count=[len(p['scores']) for p in predictions],minimum_saved_scores=[min(p['scores']) if p['scores'] else None for p in predictions])
    report.update(status='COMPLETED',total_seconds=time.monotonic()-started,formal_metric_calls=0,
        independent_semantic_evaluator_loaded=True,scope='S0 diverse T oracle images only; no EEG performance claim')
    save();print(json.dumps(dict(status=report['status'],seconds=report['total_seconds'],
        per_image_seconds={k:v['seconds_per_image'] for k,v in report['cases'].items()})),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'audit/semantic_evaluation_failure.json').write_text(traceback.format_exc());raise
