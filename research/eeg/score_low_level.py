"""Supplementary CPU image details and exhaustive saved-file integrity audit.

Separate from the protected DINO/SEED evaluators; never used to choose a seed.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import collections
import csv
import hashlib
import json
import time
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from PIL import Image, __version__ as PIL_VERSION

ROOT=Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def pixcorr(x,y):
    a=x.ravel()-x.mean();b=y.ravel()-y.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else 0.


def moments(x):
    mean=gaussian_filter(x,sigma=1.5,truncate=3.5,mode='reflect')
    return mean,gaussian_filter(x*x,sigma=1.5,truncate=3.5,mode='reflect')-mean*mean


def ssim_y(x,y,y_moments=None):
    """Wang SSIM:11x11 Gaussian sigma1.5, population covariance, valid interior."""
    mx,vx=moments(x);my,vy=moments(y) if y_moments is None else y_moments
    covariance=gaussian_filter(x*y,sigma=1.5,truncate=3.5,mode='reflect')-mx*my
    c1=.01**2;c2=.03**2
    value=((2*mx*my+c1)*(2*covariance+c2))/((mx*mx+my*my+c1)*(vx+vy+c2))
    return float(value[5:-5,5:-5].mean())


def pixels(path):
    with Image.open(path) as image:
        image.load();assert image.size==(512,512) or min(image.size)>0
        rgb=image.convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
        return np.asarray(rgb,dtype=np.float64)/255.,np.asarray(rgb.convert('L'),dtype=np.float64)/255.


@lru_cache(maxsize=32)
def target_pixels(path,expected_sha):
    assert sha(path)==expected_sha
    rgb,y=pixels(path)
    return rgb,y,moments(y)


def numerical_checks():
    # Analytic constant-image expression verifies constants/data range independently.
    x=np.full((32,32),.2);y=np.full((32,32),.7)
    expected=(2*.2*.7+.01**2)/(.2**2+.7**2+.01**2)
    np.testing.assert_allclose(ssim_y(x,y),expected,rtol=0,atol=1e-10)
    rng=np.random.default_rng(1729);x=rng.random((32,32))
    np.testing.assert_allclose(ssim_y(x,x),1.,atol=1e-12)
    np.testing.assert_allclose(pixcorr(x,x),1.,atol=1e-12)
    np.testing.assert_allclose(pixcorr(x,-x),-1.,atol=1e-12)
    return dict(identity_SSIM=ssim_y(x,x),analytic_constant_pair_SSIM=expected,
                constant_pair_error=abs(ssim_y(np.full((32,32),.2),y)-expected))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    parser.add_argument('--gallery-manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();started=time.monotonic();checks=numerical_checks()
    rows=[json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    assert len(rows) in [16600,2000] and len({r['query_id'] for r in rows})==len(rows)
    assert len({r['subject'] for r in rows})==10
    gallery={r['stimulus_id']:r for r in map(json.loads,Path(args.gallery_manifest).read_text().splitlines())}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'summary.json').exists(),'Do not overwrite a completed audit'
    counts=collections.Counter();failures=[];values=[];duplicates=collections.defaultdict(list)
    with (out/'per_query.csv').open('x',newline='') as file:
        writer=None
        for i,row in enumerate(rows):
            record=dict(query_id=row['query_id'],subject=row['subject'],stimulus_id=row['stimulus_id'],
                concept_id=row['concept_id'],generated_path=row['generated_path'],status='OK')
            target=gallery[row['stimulus_id']]
            assert target['image_sha256']==row['target_image_sha256']
            record['scoring_target_path']=target['image_path']
            path=Path(row['generated_path'])
            try:
                if not path.exists():raise FileNotFoundError(path)
                actual_sha=sha(path);assert actual_sha==row['generated_sha256'],'Saved-file hash mismatch'
                with Image.open(path) as image:
                    image.load();assert image.size==(512,512),'Unexpected generated image resolution'
                rgb,y=pixels(path);trgb,ty,tm=target_pixels(target['image_path'],target['image_sha256'])
                record.update(PixCorr_RGB=pixcorr(rgb,trgb),SSIM_Y=ssim_y(y,ty,tm),
                    mean_RGB_absolute_error=float(np.abs(rgb.mean((0,1))-trgb.mean((0,1))).mean()),
                    generated_pixel_std=float(rgb.std()))
                duplicates[actual_sha].append(row['query_id']);values.append(record);counts['OK']+=1
            except Exception as exc:
                # Retain every query in the denominator; no hidden image regeneration.
                record.update(status=type(exc).__name__,PixCorr_RGB='',SSIM_Y='',
                    mean_RGB_absolute_error='',generated_pixel_std='')
                failures.append(record|dict(error=str(exc)));counts[record['status']]+=1
            if writer is None:writer=csv.DictWriter(file,fieldnames=list(record));writer.writeheader()
            writer.writerow(record)
            if (i+1)%1000==0:print(json.dumps(dict(stage='low_level_CPU',completed=i+1,expected=len(rows))),flush=True)
    groups={k:v for k,v in duplicates.items() if len(v)>1}
    (out/'duplicate_hash_groups.json').write_text(json.dumps(groups,indent=2)+'\n')
    (out/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    metrics=['PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error']
    subjects=sorted({r['subject'] for r in rows})
    report=dict(status='COMPLETED' if not failures else 'COMPLETED_WITH_FAILURES_NO_SILENT_EXCLUSION',
        expected_images=len(rows),actual_present=sum(Path(r['generated_path']).exists() for r in rows),
        verified_and_decoded=counts['OK'],missing_count=sum(r['status']=='FileNotFoundError' for r in failures),
        corrupt_or_mismatched_count=sum(r['status']!='FileNotFoundError' for r in failures),
        duplicate_hash_groups=len(groups),duplicate_images_beyond_first=sum(len(v)-1 for v in groups.values()),
        summary_over_valid_images={m:float(np.mean([r[m] for r in values])) if values else None for m in metrics},
        per_subject_over_valid_images={s:{m:float(np.mean([r[m] for r in values if r['subject']==s])) for m in metrics} for s in subjects},
        normalization='RGB256x256 PIL BICUBIC [0,1], then PIL8-bit L for SSIM; no crop or spatial alignment',
        SSIM='11x11 Gaussian sigma1.5, population covariance, K1=.01 K2=.03 L=1, valid interior5px cropped',
        PixCorr='Pearson correlation over all flattened RGB pixels; constant image denominator gives0',
        dependencies=dict(numpy=np.__version__,scipy=scipy.__version__,Pillow=PIL_VERSION),
        checks=checks,manifest_sha256=sha(args.manifest),gallery_manifest_sha256=sha(args.gallery_manifest),
        source_sha256=sha(__file__),GPU_used=False,seconds=time.monotonic()-started,
        sources=['https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf',
                 'https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity'],
        limits=['Supplementary low-level metrics, not the protected primary DINO/SEED metrics.',
            'Position/size/background/colour differences can strongly affect these scores.',
            'These metrics do not certify correct object, absence of noise, or perceptual quality noninferiority.',
            'Human quality ratings have not been collected. All failures remain in expected_images.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
