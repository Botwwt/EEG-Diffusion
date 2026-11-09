"""Full native R80 author-code image evaluation, one saved image per query."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

from brainhive_original_metrics_runtime_v1 import *
from brainhive_native_models_v1 import write
from PIL import Image
import skimage
import torchvision


def finite(value):
    value=float(value)
    return value if math.isfinite(value) else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-name',required=True,
        choices=['native_R80_correct_v1','native_R80_permuted_v1','native_R80_missing_v1']);args=parser.parse_args()
    assert b'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes()
    assert torch.cuda.is_available()
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic()
    folder=WORK/'images'/args.run_name
    audit=json.loads((folder/'pairing_audit.json').read_text())
    assert audit['status']=='ALL_SAVED_IMAGES_TARGETS_AND_STIMULUS_ROWS_JOIN_VERIFIED'
    manifest=folder/'manifest.jsonl';assert sha(manifest)==audit['manifest_sha256']
    rows=[json.loads(x) for x in manifest.read_text().splitlines()]
    assert len(rows)==2000 and len({r['query_id'] for r in rows})==2000
    subjects=sorted({r['subject'] for r in rows});assert len(subjects)==10
    gallery=[json.loads(x) for x in Path(audit['gallery_manifest']).read_text().splitlines()]
    assert len(gallery)==200 and sha(audit['gallery_manifest'])==audit['gallery_manifest_sha256']
    real={r['stimulus_id']:r for r in gallery};assert len(real)==200
    out=WORK/'evaluation'/args.run_name/'original_paper_metrics_v1';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'completion.json').exists(),'Never rescore a completed arm'
    evaluator=OriginalEvaluator()
    configuration=dict(run=args.run_name,manifest_sha256=sha(manifest),entry_sha256=sha(__file__),
        runtime_sha256=sha(ROOT/'brainhive_original_metrics_runtime_v1.py'),
        registration_sha256=sha(ROOT/'configs/BrainHIVE_original_image_metrics_registration_v1.json'),
        author_source_sha256=sha(ROOT/'reproduction_sources/BrainHIVE_f06db69f/main/utils_eval.py'),
        asset_receipt_sha256=sha(WORK/'audit/original_metric_assets_CPU_v3.json'),
        subjects=subjects,queries=2000,images_per_query=1,seed=1729,
        torch_version=torch.__version__,torchvision_version=torchvision.__version__,skimage_version=skimage.__version__,
        compatibility=['Reviewed original metric function bodies executed directly',
            'Dedicated offline Torch cache symlinks to fully verified public weights',
            'Exact pinned OpenAI build_model, no tokenizer needed for encode_image',
            'SwAV offline equivalent of official torchvision backbone loader; unused fc not evaluated',
            'Source keyword behavior tested on nonidentical images; native metric math unchanged'],
        protected_common_metrics_unchanged=True,source_checks=evaluator.checks)
    config=out/'configuration.json'
    if config.exists():assert json.loads(config.read_text())==configuration
    else:write(config,configuration)
    preprocess=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),transforms.ToTensor()])
    cache={}
    for stimulus,item in real.items():
        assert sha(item['image_path'])==item['image_sha256']
        with Image.open(item['image_path']) as image:cache[stimulus]=preprocess(image.convert('RGB'))
    records=[];subject_rows={}
    try:
        for subject in subjects:
            selected=[r for r in rows if r['subject']==subject]
            assert len(selected)==200 and {r['stimulus_id'] for r in selected}==set(real)
            saved=out/(subject+'.json')
            if saved.exists():
                old=json.loads(saved.read_text());assert old['configuration_sha256']==sha(config)
                assert [r['query_id'] for r in old['per_query']]==[r['query_id'] for r in selected]
                records.extend(old['per_query']);subject_rows[subject]=old['source_aggregate'];continue
            generated=[]
            for row in selected:
                target=real[row['stimulus_id']]
                assert row['target_image_sha256']==target['image_sha256']
                assert sha(row['generated_path'])==row['generated_sha256']
                with Image.open(row['generated_path']) as image:
                    image.load();assert image.size==(512,512)
                    generated.append(preprocess(image.convert('RGB')))
            images=torch.stack([cache[r['stimulus_id']] for r in selected])
            recon=torch.stack(generated)
            source,per_query=evaluator.evaluate(images,recon)
            part=[]
            for i,row in enumerate(selected):
                target=real[row['stimulus_id']]
                part.append(dict(query_id=row['query_id'],subject=subject,stimulus_id=row['stimulus_id'],
                    concept_id=row['concept_id'],EEG_source_stimulus_id=row['EEG_source_stimulus_id'],
                    generated_path=row['generated_path'],generated_sha256=row['generated_sha256'],
                    target_image_path=target['image_path'],score_target_path=target['image_path'],
                    target_sha256=target['image_sha256'],image_seed=row['image_seed'],
                    **{key:finite(per_query[key][i]) for key in NAMES}))
            aggregate={key:finite(source[key]) for key in NAMES}
            write(saved,dict(subject=subject,queries=200,configuration_sha256=sha(config),
                per_query=part,source_aggregate=aggregate,per_query_means_match_original_aggregate=True))
            records.extend(part);subject_rows[subject]=aggregate
            print(json.dumps(dict(stage='ORIGINAL_AUTHOR_IMAGE_METRICS',run=args.run_name,
                subject=subject,completed_queries=len(records),expected=2000,seconds=time.monotonic()-started)),flush=True)
            del images,recon,generated
            torch.cuda.empty_cache()
    finally:evaluator.close()
    assert len(records)==2000 and len({r['query_id'] for r in records})==2000
    with (out/'per_query.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    counts={key:sum(r[key] is not None for r in records) for key in NAMES}
    means={key:sum(subject_rows[s][key] for s in subjects)/10 if all(subject_rows[s][key] is not None for s in subjects) else None for key in NAMES}
    status='ALL2000_AUTHOR_CODE_QUERY_METRICS_COMPLETE' if all(v==2000 for v in counts.values()) else 'COMPLETE_WITH_UNDEFINED_METRICS_NO_SILENT_EXCLUSION'
    write(out/'completion.json',dict(status=status,queries=2000,subjects=10,means=means,
        per_subject=subject_rows,defined_query_counts=counts,configuration_sha256=sha(config),
        per_query_sha256=sha(out/'per_query.csv'),seconds=time.monotonic()-started,
        independent_H=False,exact_paper_multiple_seed_ten_image_mean=False,component_gain_established=False,
        limits=['User-required single training seed and one image/query instead of paper repeated means.',
                'Previously exposed R80 benchmark is an exploratory reproduction track.',
                'EffNet/SwAV here are original distances; do not relabel protected semantic similarities.',
                'All undefined values stay in coverage counts; no image was regenerated or selected.']))


if __name__=='__main__':main()
