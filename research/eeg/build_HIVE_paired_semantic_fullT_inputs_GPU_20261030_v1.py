"""Full allowed-T paired inputs and image-target bank; zero model updates."""
import argparse,datetime,hashlib,importlib.util,io,json,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):assert ref(x['path'])==x;return Path(x['path'])
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024


def main(path):
    s=read(path);assert s['host']=='old' and s['status']=='REGISTERED_ALL_T_PAIRED_IMAGE_TARGET_AND_EEG_CACHE_ONLY'
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    for pin in s['source_pins']+s['payloads']:checked(pin)
    root=Path(s['output_root']);root.mkdir();start=time.monotonic();memory=rss();phase='IMPORT'
    sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0,s['profile_source_root'])
    import numpy as np,torch
    import torch.nn.functional as F
    from PIL import Image,ImageOps
    from brainhive_paired_semantic_effect_20261030_v1 import CLIP_pixels
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,s['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory))
    def budget():
        assert time.monotonic()-start<800 and rss()-memory<s['maximum_extra_CPU_bytes']
        assert torch.cuda.max_memory_allocated()<s['maximum_GPU_bytes'] and shutil.disk_usage(root).free>3<<30
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<s['output_cap_bytes']
    try:
        phase='LOAD_FIXED_FROZEN_CLIP'
        spec=importlib.util.spec_from_file_location('pinned_CLIP_visual',s['CLIP_source']['path']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        jit=torch.jit.load(s['CLIP_weights']['path'],map_location='cpu').eval()
        weights={k[len('visual.'):]:v for k,v in jit.state_dict().items() if k.startswith('visual.')}
        critic=mod.VisionTransformer(224,14,1024,24,16,768).eval().requires_grad_(False);critic.load_state_dict(weights,strict=True);critic.cuda().float();del weights,jit
        rows=read(s['rows']['path']);pairs=read(s['pairs']['path'])
        assert len(rows)==13230 and len(pairs)==6615
        targets=np.lib.format.open_memmap(root/'target_CLIP_L14.npy.pending',mode='w+',dtype=np.float32,shape=(13230,768))
        phase='ALL_REAL_T_IMAGE_FEATURES';archives={};image_bytes=0
        for lo in range(0,len(rows),s['batch_size']):
            budget();images=[]
            for item in rows[lo:lo+s['batch_size']]:
                if item['image_zip'] not in archives:archives[item['image_zip']]=zipfile.ZipFile(item['image_zip'])
                raw=archives[item['image_zip']].read(item['image_member']);image_bytes+=len(raw)
                assert len(raw)==int(item['image_bytes']) and hashlib.sha256(raw).hexdigest()==item['image_sha256']
                with Image.open(io.BytesIO(raw)) as image:
                    image=ImageOps.fit(image.convert('RGB'),(512,512),method=Image.Resampling.BICUBIC)
                    images.append(np.asarray(image).copy())
            batch=torch.from_numpy(np.stack(images)).permute(0,3,1,2).cuda().float()/255
            with torch.no_grad():features=F.normalize(critic(CLIP_pixels(batch)).float(),dim=-1)
            assert torch.isfinite(features).all();targets[lo:lo+len(images)]=features.cpu().numpy()
            del batch,features,images
            if lo%1024==0:print(json.dumps(dict(stage=phase,completed=lo+min(s['batch_size'],len(rows)-lo),total=13230,seconds=time.monotonic()-start)),flush=True)
        for z in archives.values():z.close()
        targets.flush();del targets
        (root/'target_CLIP_L14.npy.pending').rename(root/'target_CLIP_L14.npy')
        phase='ALL_REAL_T_PHYSICAL_MEAN4'
        raw=torch.load(s['EEG']['path'],map_location='cpu',weights_only=True,mmap=True);indices=read(s['input_bindings']['path'])['selected_indices']
        eeg=np.lib.format.open_memmap(root/'EEG_mean4_T.npy.pending',mode='w+',dtype=np.float32,shape=(13230,17,250))
        for lo in range(0,13230,256):
            budget();ids=[int(r['array_row']) for r in rows[lo:lo+256]];x=raw[ids][:,indices,:].numpy()
            assert x.shape==(len(ids),17,250) and np.isfinite(x).all();eeg[lo:lo+len(ids)]=x
        eeg.flush();del eeg,raw
        (root/'EEG_mean4_T.npy.pending').rename(root/'EEG_mean4_T.npy')
        targets=np.load(root/'target_CLIP_L14.npy',mmap_mode='r',allow_pickle=False)
        normerror=float(np.max(np.abs(np.linalg.norm(targets.astype(np.float64),axis=1)-1)))
        assert normerror<5e-6
        result=dict(status='COMPLETE_ALL13230_T_IMAGES_AND_PHYSICAL_MEAN4_PAIRED_CACHE_NO_FIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            protocol=ref(path),rows=s['rows'],pairs=s['pairs'],concepts=1323,unique_stimuli=13230,pair_count=6615,
            target_features=ref(root/'target_CLIP_L14.npy'),EEG_mean4=ref(root/'EEG_mean4_T.npy'),
            image_bytes_verified=image_bytes,every_image_identity_and_SHA_verified=True,all_EEG_values_finite=True,
            maximum_target_unit_norm_error=normerror,critic_frozen=True,CLIP_critic_shared_with_author_CLIP_metric=True,
            Dfit_Dcheck_V_or_official_test_read=False,new_models=0,new_optimizer_updates=0,new_research_images=0,
            formal_fit_admitted=False,training_seed=1729,NEW_contacted=False,seconds=time.monotonic()-start,
            extra_CPU_peak=rss()-memory,peak_GPU_allocated=torch.cuda.max_memory_allocated())
        budget();write(root/'completion.json',result);print(json.dumps(result))
    except BaseException as e:
        write(root/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();main(a.protocol)
