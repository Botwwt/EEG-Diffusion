"""Author metric integration check on the complete 200 real-image gallery.

This reads no generated images, selects no model, and runs once before the
registered native image metrics. The derangement is fixed without scores.
"""
import json
import math
import os
from pathlib import Path
import time

from brainhive_original_metrics_runtime_v1 import *
from brainhive_native_models_v1 import write
from PIL import Image


def main():
    assert b'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes()
    assert torch.cuda.is_available()
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=WORK/'audit/original_real_image_checks_v1';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'completion.json').exists()
    manifest=WORK/'gallery/benchmark/manifest.jsonl'
    rows=[json.loads(x) for x in manifest.read_text().splitlines()]
    assert len(rows)==200 and len({r['stimulus_id'] for r in rows})==200
    rng=np.random.default_rng(1729)
    for attempt in range(100):
        permutation=rng.permutation(200)
        if np.all(permutation!=np.arange(200)):break
    else:raise AssertionError('No fixed derangement within finite attempts')
    registration=dict(gallery_manifest_sha256=sha(manifest),seed=1729,
        permutation=permutation.tolist(),entry_sha256=sha(__file__),
        runtime_sha256=sha(ROOT/'brainhive_original_metrics_runtime_v1.py'),
        generated_images_read=False,model_selection=False,
        interpretation='Identity tests integration; deranged real-image pairing is a metric negative control, not reconstruction quality.')
    write(out/'registration.json',registration)
    preprocess=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),transforms.ToTensor()])
    images=[]
    for row in rows:
        assert sha(row['image_path'])==row['image_sha256']
        with Image.open(row['image_path']) as image:images.append(preprocess(image.convert('RGB')))
    images=torch.stack(images)
    evaluator=OriginalEvaluator();started=time.monotonic();summaries={}
    try:
        for name,fake in [('identity',images),('deranged',images[permutation])]:
            values,per_query=evaluator.evaluate(images,fake)
            values={k:float(v) for k,v in values.items()}
            assert all(math.isfinite(v) for v in values.values())
            if name=='identity':
                for key in ['eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip']:
                    assert abs(values[key]-1)<1e-5,(key,values[key])
                for key in ['eval_effnet','eval_swav']:assert abs(values[key])<1e-5,(key,values[key])
            else:
                for key in ['eval_alex2','eval_alex5','eval_inception','eval_clip']:
                    assert 0<=values[key]<1,(key,values[key])
            records=[dict(query_row=i,target_stimulus_id=rows[i]['stimulus_id'],
                reconstruction_stimulus_id=rows[i if name=='identity' else int(permutation[i])]['stimulus_id'],
                **{k:float(per_query[k][i]) for k in NAMES}) for i in range(200)]
            write(out/(name+'.json'),dict(source_aggregate=values,per_query=records,
                registration_sha256=sha(out/'registration.json')))
            summaries[name]=values
            print(json.dumps(dict(stage='AUTHOR_METRIC_REAL_GALLERY_CHECK',case=name,seconds=time.monotonic()-started)),flush=True)
            torch.cuda.empty_cache()
    finally:evaluator.close()
    write(out/'completion.json',dict(status='ALL200_REAL_IMAGE_IDENTITY_AND_FIXED_DERANGED_CHECKS_PASSED',
        queries_per_case=200,means=summaries,registration_sha256=sha(out/'registration.json'),
        asset_receipt_sha256=sha(WORK/'audit/original_metric_assets_CPU_v3.json'),
        seconds=time.monotonic()-started,generated_images_read=False,model_selection=False))


if __name__=='__main__':main()
