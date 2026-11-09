"""Acquire exact author-evaluation assets on server RAM; no GPU or local weights."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request
os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_native_models_v1 import ROOT,WORK,sha,write,torch
sys.path.insert(0,'/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runtime/BrainHIVE_original_metrics_0p24_v1')

ASSETS=Path('/dev/shm/BrainHIVE_original_metrics_20261020_v1')


def main():
    started=time.monotonic();torch.set_num_threads(2)
    completion=WORK/'audit/original_metric_assets_CPU_v2.json'
    assert not completion.exists(),'Preserve existing readiness report'
    pin=json.loads((ROOT/'audit/BrainHIVE_CLIP_metric_source_pin_20261020_v1.json').read_text())
    package=ROOT/'reproduction_sources'/('OpenAI_CLIP_'+pin['commit'][:12])
    for row in pin['files']:
        path=package/row['path'];data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==row['sha256']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob']
    assert shutil.disk_usage('/dev/shm').free>2<<30
    ASSETS.mkdir(exist_ok=True)
    downloads=[dict(name='AlexNet',filename='alexnet-owt-7be5be79.pth',bytes=244408911,
        sha_prefix='7be5be79',url='https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'),
        dict(name='CLIP_L14',filename='ViT-L-14.pt',bytes=932768134,
        sha256='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836',
        url='https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt')]
    def download(row):
        path=ASSETS/row['filename'];temporary=Path(str(path)+'.partial')
        if path.exists():
            actual=sha(path);assert path.stat().st_size==row['bytes']
            if 'sha256' in row:assert actual==row['sha256']
            else:assert actual.startswith(row['sha_prefix'])
            return row|dict(path=str(path),sha256=actual,reused_complete_payload=True)
        assert not path.exists() and not temporary.exists(),'Preserve earlier transfer; explicit resume if needed'
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        # Unique query avoids a CDN's cached 8-byte range probe response.
        request=urllib.request.Request(row['url']+'?brainhive_full_asset=20261020v1',headers={'Accept-Encoding':'identity'})
        total=0;digest=hashlib.sha256()
        with opener.open(request,timeout=60) as response,temporary.open('xb') as destination:
            assert response.status==200,response.status
            while True:
                assert time.monotonic()-started<1200
                chunk=response.read(8<<20)
                if not chunk:break
                total+=len(chunk);assert total<=row['bytes'];destination.write(chunk);digest.update(chunk)
            destination.flush();os.fsync(destination.fileno())
        actual=digest.hexdigest();assert total==row['bytes']
        if 'sha256' in row:assert actual==row['sha256']
        else:assert actual.startswith(row['sha_prefix'])
        temporary.rename(path)
        print(json.dumps(dict(stage='ORIGINAL_METRIC_PUBLIC_ASSET_READY',name=row['name'],bytes=total,sha256=actual)),flush=True)
        return row|dict(path=str(path),sha256=actual)
    with ThreadPoolExecutor(2) as pool:assets=list(pool.map(download,downloads))
    for name,path,prefix in [
        ('Inception','/root/rime_var_20261015/assets/evaluators/inception_v3_google-0cc3c7bd.pth','0cc3c7bd'),
        ('SwAV','/root/rime_var_20261015/assets/evaluators/swav_800ep_pretrain.pth.tar',None),
        ('EffNet','/dev/shm/eeg_sota_autoresearch_20261018_v1/semantic_evaluation/efficientnet_b1-c27df63c.pth','c27df63c')]:
        path=Path(path);actual=sha(path)
        if prefix:assert actual.startswith(prefix)
        assets.append(dict(name=name,path=str(path),bytes=path.stat().st_size,sha256=actual,preexisting=True))
    import torchvision
    from torchvision.models import alexnet,inception_v3,efficientnet_b1,resnet50
    lookup={r['name']:r for r in assets};checks={}
    for name,constructor in [('AlexNet',lambda:alexnet(weights=None)),
            ('Inception',lambda:inception_v3(weights=None,init_weights=False)),
            ('EffNet',lambda:efficientnet_b1(weights=None))]:
        model=constructor();state=torch.load(lookup[name]['path'],weights_only=True,map_location='cpu')
        model.load_state_dict(state,strict=True)
        assert all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point())
        checks[name]=dict(strict_load=True,parameters=sum(p.numel() for p in model.parameters()))
        del model,state
    model=resnet50(weights=None)
    state=torch.load(lookup['SwAV']['path'],weights_only=True,map_location='cpu')
    state={k.replace('module.',''):v for k,v in state.items()}
    coverage=model.load_state_dict(state,strict=False)
    assert set(coverage.missing_keys)=={'fc.weight','fc.bias'}
    assert all(k.startswith(('projection_head.','prototypes.')) for k in coverage.unexpected_keys)
    checks['SwAV']=dict(strict_backbone_load=True,missing_unused_head=coverage.missing_keys,
        unused_pretraining_heads=coverage.unexpected_keys,parameters=sum(p.numel() for p in model.parameters()))
    del model,state
    spec=importlib.util.spec_from_file_location('pinned_openai_CLIP_model',package/'clip/model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    scripted=torch.jit.load(lookup['CLIP_L14']['path'],map_location='cpu').eval()
    model=module.build_model(scripted.state_dict()).eval();del scripted
    assert model.visual.input_resolution==224 and model.visual.conv1.kernel_size==(14,14)
    assert len(model.visual.transformer.resblocks)==24
    checks['CLIP_L14']=dict(strict_original_build_model=True,parameters=sum(p.numel() for p in model.parameters()),
        source_commit=pin['commit'],visual_resolution=224,visual_layers=24,embedding_dimensions=768)
    del model
    import skimage
    from skimage.metrics import structural_similarity
    import numpy as np
    x=np.random.default_rng(1729).random((32,32)).astype('float32')
    kwargs=dict(gaussian_weights=True,sigma=1.5,use_sample_covariance=False,data_range=1.)
    y=.7*x+.3*np.roll(x,3,axis=0)
    source_value=structural_similarity(x,y,multichannel=True,**kwargs)
    explicit=structural_similarity(x,y,channel_axis=None,**kwargs)
    wrong_axis=structural_similarity(x,y,channel_axis=1,**kwargs)
    assert abs(source_value-explicit)<1e-12 and abs(explicit-wrong_axis)>1e-6
    assert structural_similarity(x,x,multichannel=True,**kwargs)==1.
    write(completion,dict(status='ALL5_AUTHOR_METRIC_ASSETS_STRICT_LOAD_VERIFIED_CPU',assets=assets,
        checks=checks,CLIP_source_root=str(package),seconds=time.monotonic()-started,
        torch_version=torch.__version__,torchvision_version=torchvision.__version__,skimage_version=skimage.__version__,
        source_SSIM_multichannel_keyword_on_2D_verified_equal_to_current_channel_axis_None=True,
        nonidentical_SSIM_fixture=dict(source_keyword=source_value,explicit_2D=explicit,incorrect_channel_axis=wrong_axis),
        GPU_initialized=False,images_evaluated=0,metric_scores_seen=False,local_large_files_written=False))
    print('ORIGINAL_METRIC_ASSETS_CPU_READY',flush=True)


if __name__=='__main__':main()
