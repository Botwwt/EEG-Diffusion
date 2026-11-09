"""Prespecified CPU-only supplementary evaluation of Direction06's full T80.

Reuse frozen low-level image functions and the completed generic DINO CSV.
No image features, fit, seed selection, GPU forward or reduced denominator.
The exposed single-subject eight-concept panel supports development decisions.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

import score_low_level as L
import score_BrainHIVE_prediction_fastscreen_20261027_v1 as D
from score_BrainHIVE_trialset_fastscreen_20261027_v1 import paired_bootstrap

ROOT=Path(__file__).resolve().parent
SCHEMA='BrainHIVE_forward_neural_supplement_CPU_20261027_v1'
ARMS=('native_parent','neural_correct','neural_sameconcept_shuffled')
LOW_METRICS=('PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error','RGB8x8_MSE')
DINO_METRICS=('target_cosine','global80_two_afc','global80_top1','same_concept10_top1','same_concept10_two_afc')
MAX_SECONDS=120
OUTPUT_CAP=16<<20



def rgb8(rgb):
    assert rgb.shape==(256,256,3)
    return rgb.reshape(8,32,8,32,3).mean(axis=(1,3))


def low_values(generated_path,target_path,target_sha):
    rgb,y=L.pixels(generated_path)
    trgb,ty,tm=L.target_pixels(str(target_path),target_sha)
    values=dict(PixCorr_RGB=L.pixcorr(rgb,trgb),SSIM_Y=L.ssim_y(y,ty,tm),
                mean_RGB_absolute_error=float(np.abs(rgb.mean((0,1))-trgb.mean((0,1))).mean()),
                RGB8x8_MSE=float(np.square(rgb8(rgb)-rgb8(trgb)).mean()))
    assert all(math.isfinite(value) for value in values.values())
    return values


def compare(metrics,queries,candidate,baseline):
    result=paired_bootstrap({'mean':metrics[baseline],'each_mean':metrics[candidate]},queries,
                            repetitions=1000,seed=1729)
    return {name:{('difference' if key=='each_mean_minus_mean' else key):value for key,value in item.items()}
            for name,item in result.items()}


def bind_dino(path,generation_completion,manifest,queries,arms):
    done=D.read(path)
    assert done['status']=='COMPLETED_GENERIC_NATIVE_PREDICTION_DINO_SCREEN'
    assert done['generation_completion_sha256']==D.sha(generation_completion)
    assert done['generation_manifest_sha256']==D.sha(manifest)
    assert done['expected_images']==done['observed_images']==240
    summary_path=D.checked_ref(done['summary']);summary=D.read(summary_path)
    assert summary['arms']==list(arms) and summary['queries']==80 and summary['generated_images']==240
    per_query=D.checked_ref(summary['per_query'])
    assert D.sha(per_query)==done['per_query_sha256']
    with per_query.open(encoding='utf-8',newline='') as stream:records=list(csv.DictReader(stream))
    assert len(records)==240
    lookup={}
    identities={query['query_id']:query for query in queries}
    for row in records:
        key=(row['query_id'],row['arm']);assert key not in lookup
        assert row['query_id'] in identities and row['arm'] in arms
        query=identities[row['query_id']]
        for name in ('subject_id','concept_id','stimulus_id'):
            assert row[name]==str(query[name])
        assert int(row['array_row'])==int(query['array_row']) and int(row['fold'])==int(query['fold'])
        lookup[key]={name:float(row[name]) for name in DINO_METRICS}
        assert all(math.isfinite(value) for value in lookup[key].values())
    assert set(lookup)=={(query['query_id'],arm) for query in queries for arm in arms}
    metrics={arm:{name:np.asarray([lookup[(query['query_id'],arm)][name] for query in queries],dtype=np.float64)
                  for name in DINO_METRICS} for arm in arms}
    return metrics,dict(completion=dict(path=str(Path(path).resolve()),sha256=D.sha(path)),
                        summary=dict(path=str(summary_path.resolve()),sha256=D.sha(summary_path)),
                        per_query=dict(path=str(per_query.resolve()),sha256=D.sha(per_query)),
                        DINO_extraction_rerun=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-completion',type=Path,required=True)
    parser.add_argument('--dino-completion',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scientific-preregistration',type=Path,required=True)
    parser.add_argument('--scientific-preregistration-sha256',required=True)
    parser.add_argument('--max-seconds',type=float,default=120)
    args=parser.parse_args(argv)
    assert 0<args.max_seconds<=MAX_SECONDS and not args.output.exists()
    torch.set_num_threads(2)
    try:
        from threadpoolctl import threadpool_limits
        cpu_thread_limit=threadpool_limits(limits=2)
    except ImportError:
        cpu_thread_limit=None
    started=time.monotonic();args.output.mkdir(parents=True)
    expected=240;observed=0
    def check():
        if time.monotonic()-started>=args.max_seconds:raise TimeoutError('Finite CPU evaluation deadline')
        assert sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())<=OUTPUT_CAP
        assert not torch.cuda.is_initialized(),'CPU-only evaluation must never initialize CUDA'
    try:
        assert D.sha(args.scientific_preregistration)==args.scientific_preregistration_sha256
        check();numerical=L.numerical_checks()
        complete,manifest,panel,queries,ordered,targets,arms=D.validate_inputs(args.generation_completion,ARMS)
        assert complete.get('method_schema')=='BrainHIVE_forward_neural_native_screen_20261027_v1'
        assert tuple(arms)==ARMS and len(ordered)==240
        generation_sha,manifest_sha=D.sha(args.generation_completion),D.sha(manifest)
        dino,dino_binding=bind_dino(args.dino_completion,args.generation_completion,manifest,queries,arms)
        check()
        provenance=dict(schema=SCHEMA,source=dict(path=str(Path(__file__).resolve()),sha256=D.sha(__file__)),
            frozen_lowlevel_source_sha256=D.sha(L.__file__),generic_validation_source_sha256=D.sha(D.__file__),
            paired_bootstrap_source_sha256=D.sha(ROOT/'score_BrainHIVE_trialset_fastscreen_20261027_v1.py'),
            scientific_preregistration=dict(path=str(args.scientific_preregistration.resolve()),sha256=args.scientific_preregistration_sha256),
            generation_completion=dict(path=str(args.generation_completion.resolve()),sha256=generation_sha),
            generation_manifest=dict(path=str(manifest.resolve()),sha256=manifest_sha),DINO_existing=dino_binding,
            fixed_metrics=list(LOW_METRICS),fixed_pair=['neural_correct','neural_sameconcept_shuffled'],GPU_used=False,
            panel_already_exposed_development=True,independent_confirmation=False,
            metric_definition=dict(PixCorr_RGB='exact frozen score_low_level.pixcorr on flattened RGB256 float64',
                SSIM_Y='exact frozen score_low_level.ssim_y: PIL8-bit L256;11x11 Gaussian sigma1.5 population covariance valid interior',
                mean_RGB_absolute_error='exact frozen mean RGB absolute difference over channels in [0,1]',
                RGB8x8_MSE='RGB256 from frozen pixels(); exact non-overlapping32x32 block means ->8x8x3; squared difference mean; no alignment/crop'))
        D.atomic_json(args.output/'provenance.json',provenance)
        values={arm:{name:[] for name in LOW_METRICS} for arm in arms}
        per_query=args.output/'per_query.csv'
        with per_query.open('x',newline='',encoding='utf-8') as stream:
            fields=['query_id','subject_id','array_row','concept_id','stimulus_id','fold','arm','status',*LOW_METRICS]
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in ordered:
                check()
                assert D.sha(row['generated_path'])==row['generated_sha256']
                result=low_values(row['generated_path'],row['target_image_path'],row['target_image_sha256'])
                arm=row['arm']
                for name in LOW_METRICS:values[arm][name].append(result[name])
                writer.writerow({**{key:row[key] for key in fields if key not in LOW_METRICS and key!='status'},
                                 'status':'OK',**result})
                observed+=1
        assert observed==240
        low={arm:{name:np.asarray(data,dtype=np.float64) for name,data in values[arm].items()} for arm in arms}
        assert all(len(array)==80 for item in low.values() for array in item.values())
        pairs=[('neural_sameconcept_shuffled','native_parent'),('neural_correct','native_parent'),('neural_correct','neural_sameconcept_shuffled')]
        low_comparisons={a+'_minus_'+b:compare(low,queries,a,b) for a,b in pairs}
        dino_comparisons={a+'_minus_'+b:compare(dino,queries,a,b) for a,b in pairs}
        paired_path=args.output/'neural_correct_minus_neural_sameconcept_shuffled_per_query.csv'
        with paired_path.open('x',newline='',encoding='utf-8') as stream:
            fields=['query_id','subject_id','array_row','concept_id','stimulus_id','fold',
                    *['low_'+name for name in LOW_METRICS],*['DINO_'+name for name in DINO_METRICS]]
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for index,query in enumerate(queries):
                writer.writerow({**{key:query[key] for key in fields if key in query},
                    **{'low_'+name:float(low['neural_correct'][name][index]-low['neural_sameconcept_shuffled'][name][index]) for name in LOW_METRICS},
                    **{'DINO_'+name:float(dino['neural_correct'][name][index]-dino['neural_sameconcept_shuffled'][name][index]) for name in DINO_METRICS}})
        assert D.sha(args.generation_completion)==generation_sha and D.sha(manifest)==manifest_sha
        assert D.sha(args.dino_completion)==dino_binding['completion']['sha256']
        assert D.sha(args.scientific_preregistration)==args.scientific_preregistration_sha256
        assert D.sha(dino_binding['per_query']['path'])==dino_binding['per_query']['sha256']
        for row in ordered:
            check();assert D.sha(row['generated_path'])==row['generated_sha256']
        for index,target in enumerate(targets):assert D.sha(target)==ordered[index*len(arms)]['target_image_sha256']
        check()
        summary=dict(status='COMPLETE_FORWARD_NEURAL_SUPPLEMENTARY_CPU_SCREEN',schema=SCHEMA,
            subjects=1,concepts=8,queries=80,arms=list(arms),expected_images=240,scored_images=240,
            missing_or_excluded_images=0,full_registered_denominator=True,
            low_level_means={arm:{name:float(array.mean()) for name,array in item.items()} for arm,item in low.items()},
            low_level_paired_comparisons=low_comparisons,DINO_existing_paired_comparisons=dino_comparisons,
            metric_direction={'PixCorr_RGB':'higher','SSIM_Y':'higher','mean_RGB_absolute_error':'lower','RGB8x8_MSE':'lower'},
            numerical_checks=numerical,bootstrap=dict(seed=1729,draws=1000,unit='eight concepts with ten paired queries',
                existing_DINO_gallery_fixed80=True,descriptive_only=True),
            per_query=dict(path=str(per_query.resolve()),sha256=D.sha(per_query)),
            neural_correct_minus_neural_sameconcept_shuffled=dict(path=str(paired_path.resolve()),sha256=D.sha(paired_path)),
            provenance_sha256=D.sha(args.output/'provenance.json'),seconds=time.monotonic()-started,GPU_used=False,
            DINO_feature_extraction_rerun=False,panel_already_exposed_development=True,independent_confirmation=False,
            claim_limit='Supplementary low-level correspondence and fixed paired development decisions; no semantic correctness, perceptual-quality noninferiority, fullV/SOTA, or independent confirmation claim')
        D.atomic_json(args.output/'summary.json',summary)
        D.atomic_json(args.output/'completion.json',dict(status=summary['status'],schema=SCHEMA,
            summary=dict(path=str((args.output/'summary.json').resolve()),sha256=D.sha(args.output/'summary.json')),
            generation_manifest_sha256=manifest_sha,DINO_completion_sha256=dino_binding['completion']['sha256'],
            scored_images=240,queries=80,GPU_used=False,seconds=time.monotonic()-started))
        print(json.dumps(summary,ensure_ascii=False,allow_nan=False),flush=True)
        return 0
    except BaseException as error:
        D.atomic_json(args.output/'failure.json',dict(status='FAILED_CPU_SCREEN_FIXED_DENOMINATOR_PRESERVED',
            error=repr(error),traceback=traceback.format_exc(),expected_images=expected,scored_prefix=observed,
            seconds=time.monotonic()-started,GPU_used=False,no_effect_claim=True))
        raise


if __name__=='__main__':
    raise SystemExit(main())
