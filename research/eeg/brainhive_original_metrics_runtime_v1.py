"""Execute the pinned author's actual metric functions with offline model loading.

Only loading/progress and per-query instrumentation differ. Aggregate metric
math stays in the verified original function bodies. Protected metrics are not
imported or changed here.
"""
import ast
from collections import OrderedDict
import functools
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from brainhive_native_models_v1 import ROOT,WORK,sha,torch
sys.path.insert(0,'/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runtime/BrainHIVE_original_metrics_0p24_v1')
import numpy as np
import scipy as sp
import scipy.spatial.distance
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import inception_v3,Inception_V3_Weights,efficientnet_b1,EfficientNet_B1_Weights,resnet50

NAMES=['eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav']
ASSETS=Path('/dev/shm/BrainHIVE_original_metrics_20261020_v1')


def source_scope():
    registration=json.loads((ROOT/'configs/BrainHIVE_original_image_metrics_registration_v1.json').read_text())
    for name,expected in registration['sources'].items():assert sha(ROOT/name.replace('\\','/'))==expected
    source=ROOT/'reproduction_sources/BrainHIVE_f06db69f/main/utils_eval.py'
    wanted=['two_way_identification','pixcorr','ssim','alexnet','inception','clip_','effnet','swav','eval_images']
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)]
    assert [n.name for n in nodes]==wanted
    scope=dict(__name__='pinned_BrainHIVE_original_metrics',np=np,torch=torch,transforms=transforms,
        tqdm=lambda value,*args,**kwargs:value,create_feature_extractor=create_feature_extractor,
        rgb2gray=rgb2gray,structural_similarity=structural_similarity,
        inception_v3=inception_v3,Inception_V3_Weights=Inception_V3_Weights,
        efficientnet_b1=efficientnet_b1,EfficientNet_B1_Weights=EfficientNet_B1_Weights,sp=sp)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    return scope


def numerical_source_checks():
    scope=source_scope()
    images=torch.eye(4).reshape(4,1,2,2)
    identity=lambda x:x.flatten(1)
    correct=scope['two_way_identification'](images,images,identity,lambda x:x)
    wrong=scope['two_way_identification'](torch.roll(images,1,0),images,identity,lambda x:x)
    second=.25*images+.75*torch.roll(images,1,0)
    rank2=scope['two_way_identification'](second,images,identity,lambda x:x)
    assert correct==1 and wrong==0 and abs(rank2-2/3)<1e-12
    asymmetric=images.clone();asymmetric[3]=images[0]
    counts,denominator=scope['two_way_identification'](asymmetric,images,identity,lambda x:x,return_avg=False)
    assert denominator==3 and list(counts)==[3,3,3,0]
    x=torch.full((1,3,32,32),.2,dtype=torch.float64);y=torch.full_like(x,.7)
    measured=scope['ssim'](x,y)
    analytic=(2*.2*.7+.01**2)/(.2**2+.7**2+.01**2)
    assert abs(measured-analytic)<1e-10
    rng=np.random.default_rng(1729);a=rng.random((32,32)).astype('float32');b=.7*a+.3*np.roll(a,3,axis=0)
    opts=dict(gaussian_weights=True,sigma=1.5,use_sample_covariance=False,data_range=1.)
    actual=structural_similarity(a,b,multichannel=True,**opts)
    explicit=structural_similarity(a,b,channel_axis=None,**opts)
    wrong_axis=structural_similarity(a,b,channel_axis=1,**opts)
    assert abs(actual-explicit)<1e-12 and abs(actual-wrong_axis)>1e-6
    return dict(identity_2AFC=correct,deranged_onehot_2AFC=wrong,analytic_second_rank_2AFC=rank2,
        asymmetric_generated_query_fixed_real_gallery_2AFC=float(np.mean(counts)/denominator),
        counterfactual_reversed_gallery_direction_analytic_score=2/3,
        constant_SSIM_source=float(measured),constant_SSIM_analytic=analytic,
        nonidentical_SSIM_source=float(actual),nonidentical_SSIM_explicit_2D=float(explicit),
        nonidentical_SSIM_wrong_axis=float(wrong_axis),benchmark_images_read=False)


class OriginalEvaluator:
    def __init__(self):
        self.scope=source_scope()
        self.ready=json.loads((WORK/'audit/original_metric_assets_CPU_v3.json').read_text())
        assert self.ready['status']=='ALL5_AUTHOR_METRIC_ASSETS_STRICT_LOAD_VERIFIED_CPU'
        self.assets={row['name']:row for row in self.ready['assets']}
        for row in self.assets.values():assert sha(row['path'])==row['sha256']
        self.models={};self.current={};self.fields={};self.checks=numerical_source_checks()
        self.scope['clip']=SimpleNamespace(load=self.load_clip)
        directory=ASSETS/'torch_cache/hub/checkpoints';directory.mkdir(parents=True,exist_ok=True)
        for name in ['AlexNet','Inception','EffNet']:
            source=Path(self.assets[name]['path']);target=directory/source.name
            if target.is_symlink() or target.exists():assert target.resolve()==source.resolve()
            else:target.symlink_to(source)
        torch.hub.set_dir(str(directory.parent))
        self.original_hub_load=torch.hub.load
        torch.hub.load=self.load_hub
        self.install_instrumentation()

    def close(self):
        torch.hub.load=self.original_hub_load

    def load_clip(self,name,device='cuda',**kwargs):
        assert name=='ViT-L/14' and not kwargs
        if 'CLIP_L14' not in self.models:
            package=Path(self.ready['CLIP_source_root'])
            receipt=json.loads((package/'_fetch_receipt.json').read_text())
            path=package/'clip/model.py';row=next(x for x in receipt['files'] if x['path']=='clip/model.py')
            assert sha(path)==row['sha256']
            spec=importlib.util.spec_from_file_location('pinned_openai_clip_metric_model',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            scripted=torch.jit.load(self.assets['CLIP_L14']['path'],map_location='cpu').eval()
            model=module.build_model(scripted.state_dict()).to(device).eval().requires_grad_(False)
            assert model.visual.input_resolution==224 and model.visual.conv1.kernel_size==(14,14)
            self.models['CLIP_L14']=model
        return self.models['CLIP_L14'],None  # source overrides its preprocess explicitly

    def load_hub(self,repo,name,*args,**kwargs):
        assert repo=='facebookresearch/swav:main' and name=='resnet50' and not args and not kwargs
        if 'SwAV' not in self.models:
            model=resnet50(weights=None)
            state=torch.load(self.assets['SwAV']['path'],weights_only=True,map_location='cpu')
            coverage=model.load_state_dict({k.replace('module.',''):v for k,v in state.items()},strict=False)
            assert set(coverage.missing_keys)=={'fc.weight','fc.bias'}
            assert all(k.startswith(('projection_head.','prototypes.')) for k in coverage.unexpected_keys)
            self.models['SwAV']=model.eval().requires_grad_(False)
        return self.models['SwAV']

    def store(self,key,values):
        assert key not in self.fields,key
        assert len(values)==self.current['queries']
        self.fields[key]=[float(v) for v in values]

    def install_instrumentation(self):
        original_two=self.scope['two_way_identification']
        @functools.wraps(original_two)
        def traced_two(recons,images,model,preprocess,feature_layer=None,return_avg=True,device=torch.device('cpu')):
            counts,negatives=original_two(recons,images,model,preprocess,feature_layer,False,device)
            assert negatives==len(images)-1
            key={'features.4':'eval_alex2','features.11':'eval_alex5'}.get(feature_layer,self.current['metric'])
            self.store(key,np.asarray(counts)/negatives)
            return np.mean(counts)/negatives if return_avg else (counts,negatives)
        self.scope['two_way_identification']=traced_two
        for name,key in [('pixcorr','eval_pixcorr'),('ssim','eval_ssim')]:
            original=self.scope[name]
            def low_level(images,recons,original=original,key=key):
                aggregate=original(images,recons)
                values=[original(images[i:i+1],recons[i:i+1]) for i in range(len(images))]
                mean=float(np.mean(values))
                assert (not np.isfinite(aggregate) and not np.isfinite(mean)) or abs(mean-float(aggregate))<1e-6
                self.store(key,values)
                return aggregate
            self.scope[name]=low_level
        for name,key in [('alexnet','alexnet'),('inception','eval_inception'),('clip_','eval_clip')]:
            original=self.scope[name]
            def tagged(*args,original=original,key=key,**kwargs):
                self.current['metric']=key
                return original(*args,**kwargs)
            self.scope[name]=tagged
        for name,key in [('effnet','eval_effnet'),('swav','eval_swav')]:
            original=self.scope[name]
            def distance(*args,original=original,key=key,**kwargs):
                function=sp.spatial.distance.correlation;values=[]
                def recorded(*a,**k):
                    result=function(*a,**k);values.append(result);return result
                sp.spatial.distance.correlation=recorded
                try:result=original(*args,**kwargs)
                finally:sp.spatial.distance.correlation=function
                self.store(key,values)
                assert (not np.isfinite(result) and not np.isfinite(np.mean(values))) or abs(float(result)-float(np.mean(values)))<1e-12
                return result
            self.scope[name]=distance

    @torch.inference_mode()
    def evaluate(self,images,recons,device='cuda'):
        assert images.shape==recons.shape and images.shape[0]==200 and images.shape[1:]==(3,512,512)
        self.fields={};self.current=dict(queries=len(images))
        result=self.scope['eval_images'](images,recons,device=torch.device(device))
        assert set(result)==set(self.fields)==set(NAMES)
        for key in NAMES:
            actual=float(np.mean(self.fields[key]));source=float(result[key])
            assert (not math.isfinite(actual) and not math.isfinite(source)) or abs(actual-source)<1e-6
        return result,self.fields
