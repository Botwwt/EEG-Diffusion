"""Necessary NEW-host frozen VQ transfer/target parity on two fixed T images."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,hashlib,io,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from EEG_generative_backend_loader_20261029_v1 import build_vq_only,sha,SCALES
def check(item):
    path=Path(item['path']);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path);return path
def ref(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def digest(model):
    result=hashlib.sha256()
    for name,value in model.state_dict().items():
        result.update(name.encode());result.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()
def run(spec_path):
    started=time.monotonic();spec=json.loads(spec_path.read_bytes());out=Path(spec['output']);assert not out.exists();out.mkdir();base=peak();phase='PINNED_ORIGINAL_FIXED_T_INPUTS'
    try:
        for item in spec['source_pins']:check(item)
        assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==spec['GPU_UUID']=='GPU-62a1aee8-4740-dc59-93f4-d50644db650d'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        samples=json.loads(check(spec['fixed_T_samples']).read_bytes());assert [x['query_index'] for x in samples['images']]==[0,1]
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        transform=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BILINEAR),transforms.ToTensor()]);tensors=[]
        for item in samples['images']:
            data=check(item['original_image']).read_bytes();assert hashlib.sha256(data).hexdigest()==item['original_sha256'];image=Image.open(io.BytesIO(data)).convert('RGB');assert image.width==image.height
            tensors.append(transform(image).mul(2).sub(1))
        cpu=torch.stack(tensors);fixed=np.load(check(spec['fixed_transformed_tensor']),allow_pickle=False);expected=np.load(check(spec['fixed_old_codes']),allow_pickle=False)
        assert cpu.shape==(2,3,256,256) and np.array_equal(cpu.numpy(),fixed) and expected.shape==(2,680) and expected.dtype==np.int16
        torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        vae,transfer=build_vq_only(spec['author_source'],spec['vendor'],check(spec['VQ_checkpoint']));vae.cuda();torch.cuda.reset_peak_memory_stats();before=digest(vae)
        def guard():assert time.monotonic()-started<840 and torch.cuda.max_memory_allocated()<20<<30 and peak()-base<10<<30
        phase='ACTUAL_FULL_SIZE_PRETRAINED_FROZEN_VQ_TWO_T_PARITY'
        with torch.inference_mode():
            x=cpu.cuda();parts=vae.img_to_idxBl(x);first=torch.cat(parts,1);again=torch.cat(vae.img_to_idxBl(x),1)
            assert [p.shape[1] for p in parts]==[s*s for s in SCALES] and torch.equal(first,again)
            old=torch.from_numpy(expected.astype(np.int64)).cuda();assert torch.equal(first,old),'New-host VQ target codes differ from sealed OLD token bank first two rows'
            reconstruction=vae.idxBl_to_img(parts,same_shape=True,last_one=True);replayed=vae.idxBl_to_img(parts,same_shape=True,last_one=True);assert torch.equal(reconstruction,replayed)
        guard();assert digest(vae)==before and all(not p.requires_grad and p.grad is None for p in vae.parameters())
        np.save(out/'new_fixed_T_indices.npy',first.cpu().numpy().astype(np.int16),allow_pickle=False);images=[]
        for i in range(2):
            for name,value in [('source',cpu[i].add(1).mul(.5)),('VQ_reconstruction',reconstruction[i].cpu().add(1).mul(.5))]:
                pixels=value.float().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy();path=out/f'T{i}_{name}.png';Image.fromarray(pixels).save(path);images.append(dict(image=ref(path),pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),kind=name,engineering_only=True))
        write(out/'completion.json',dict(status='PASS_NEW_HOST_STRICT_FULL_VQ_TWO_T_PREPROCESSING_CODE_AND_RECONSTRUCTION_PARITY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(spec_path),transfer=transfer,first_two_T_queries=samples['images'],new_codes=ref(out/'new_fixed_T_indices.npy'),same_transformed_pixels_as_OLD=True,same_1360_codes_as_OLD=True,repeated_codes_exact=True,repeated_reconstruction_exact=True,frozen_weights_unchanged=True,VQ_tensor_digest=before,engineering_images=images,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base,seconds=time.monotonic()-started,optimizer_updates=0,formal_trained_models=0,new_research_images=0,VAR_loaded=False,EEG_reconstruction_effect_claimed=False))
        guard()
    except BaseException as error:
        write(out/'failure.json',dict(status='NEW_VQ_ONLY_ENGINEERING_FAILURE_PRESERVED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
