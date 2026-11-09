"""Independent NumPy reconstruction of all actual D25 profile noise transforms."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import brainhive_gain_calibration_20261027_v1 as U

def main():
    p=argparse.ArgumentParser();p.add_argument('--completion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(2)
    done=U.read(a.completion);assert done['status']=='D25_NATIVE_AND_FROZEN_ORDINARY_SAMPLER_SEPARABILITY_PROFILE_COMPLETE';spec=U.read(U.checked(done['spec']));freeze=U.read(U.checked(done['ordinary_freeze']));U.checked(freeze['checkpoint']);U.checked(freeze['source'])
    traces=torch.load(U.checked(done['noise_traces']),weights_only=True,map_location='cpu');verified=0;maximum=0.;parities=0
    for case in done['cases']:
        path=Path(case['image']['generated_path']);assert U.sha(path)==case['image']['generated_sha256']
        with Image.open(path) as im:assert im.size==(512,512) and hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()==case['image']['generated_pixel_sha256']
        arrays=traces[case['name']];assert len(arrays)==(4 if case['sampler_enabled'] else 0)
        if case['exact_legacy_pixel_parity']:parities+=1
        for i,entry in enumerate(arrays):
            z=entry['raw_noise'].numpy();field=entry['field'].numpy();actual=entry['used_noise'].numpy();assert z.shape==field.shape==actual.shape==(1,4,64,64)
            raw=case['observations']['ancestral_noise_observations'][i];used=case['observations']['actual_used_noise_observations'][i];assert hashlib.sha256(z.tobytes()).hexdigest()==raw['sha256'] and hashlib.sha256(actual.tobytes()).hexdigest()==used['sha256']
            if case['mode']=='isotropic' or case['synthetic_zero_response']:expected=z;assert np.array_equal(actual,z)
            else:
                d=field.astype(np.float64)
                if case['mode']=='orthogonal_control':d=np.stack((-d[:,1],d[:,0],-d[:,3],d[:,2]),axis=1)
                norm=np.linalg.norm(d,axis=1,keepdims=True);valid=norm>0;v=d/np.maximum(norm,np.finfo(np.float64).tiny);v=np.where(valid,v,0.)
                expected=np.where(valid,np.sqrt(4/3)*(z.astype(np.float64)-v*np.sum(z*v,axis=1,keepdims=True)),z).astype(z.dtype)
            error=float(np.linalg.norm(actual.astype(np.float64)-expected.astype(np.float64))/max(np.linalg.norm(expected.astype(np.float64)),1e-20));assert error<.002;maximum=max(maximum,error);verified+=1
    assert verified==28 and parities==4 and not torch.cuda.is_initialized()
    for item in spec['source_pins']:U.checked(item)
    result=dict(status='D25_ALL8_PROFILE_IMAGES_AND28_ACTUAL_NOISE_TRANSFORMS_CPU_VERIFIED',completion=U.ref(a.completion),source=U.ref(__file__),all_sources_unchanged=True,ordinary_unchanged=True,verified_noise_transforms=verified,exact_legacy_pixel_parity_cases=parities,maximum_independent_numpy_relative_error=maximum,new_models=0,new_images=0,new_GPU_jobs=0,seconds=time.monotonic()-start)
    U.atomic(a.output/'summary.json',result);print(json.dumps(result))

if __name__=='__main__':main()
