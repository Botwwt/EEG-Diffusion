"""Frozen model extraction and explicit-gallery algebra. Numerical work: Slurm only."""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
R=Path('/dfs/share/maleiLab/wwt/eeg')
METRICS=['DINO','Alex2','Alex5','Inception','CLIP','EffNet','SwAV','low']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def load_model(name,device='cuda'):
    assets=R/'assets/evaluation_v272';ready=json.loads((assets/'completion.json').read_text());assert ready['status']=='PASS_ALL6_EXACT_HISTORICAL_WEIGHT_SHA_AND_CPU_STRICT_MODELS'
    source=R/'code/metric_sources_v272';lookup={r['name']:Path(r['path']) for r in ready['assets']}
    weights=lookup['AlexNet' if name.startswith('Alex') else ('CLIP_L14' if name=='CLIP' else ('DINOv3' if name=='DINO' else name))]
    row=next(v for v in ready['assets'] if Path(v['path'])==weights);assert sha(weights)==row['sha256']
    if name=='DINO':
        sys.path.insert(0,str(source));from dinov3.hub.backbones import dinov3_vits16
        model=dinov3_vits16(pretrained=False);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    elif name=='CLIP':
        spec=importlib.util.spec_from_file_location('pinned_clip276',source/'openai_clip_model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        scripted=torch.jit.load(str(weights),map_location='cpu');model=module.build_model(scripted.state_dict());del scripted
    else:
        from torchvision.models import alexnet,inception_v3,efficientnet_b1,resnet50
        model=alexnet(weights=None) if name.startswith('Alex') else (inception_v3(weights=None,init_weights=False) if name=='Inception' else (efficientnet_b1(weights=None) if name=='EffNet' else resnet50(weights=None)))
        state=torch.load(weights,map_location='cpu',weights_only=True)
        if name=='SwAV':
            diff=model.load_state_dict({k.replace('module.',''):v for k,v in state.items()},strict=False);assert set(diff.missing_keys)=={'fc.weight','fc.bias'} and all(k.startswith(('projection_head.','prototypes.')) for k in diff.unexpected_keys)
        else:model.load_state_dict(state,strict=True)
        key='features.4' if name=='Alex2' else ('features.11' if name=='Alex5' else 'avgpool');model=create_feature_extractor(model,return_nodes=[key])
    return model.to(device).eval().requires_grad_(False)

def pixels(path,paper=True):
    with Image.open(path) as im:
        im=im.convert('RGB')
        if paper:im=transforms.CenterCrop(512)(transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR)(im))
        return transforms.ToTensor()(im)

@torch.inference_mode()
def features(paths,name,model,real=False,device='cuda'):
    if name=='DINO':
        items=[]
        for path in paths:
            x=pixels(path,paper=False)[None]
            if not real:x=x.to(device)
            items.append(F.interpolate(x,(256,256),mode='bilinear',align_corners=False,antialias=True))
        x=torch.cat(items).to(device);mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        patch,cls=model.get_intermediate_layers((x-mean)/std,n=[11],return_class_token=True,norm=True)[0]
        y=F.normalize(torch.cat([cls,patch.mean(1)],-1),dim=-1)
    else:
        x=torch.stack([pixels(p) for p in paths]).to(device)
        size=256 if name.startswith('Alex') else {'Inception':342,'CLIP':224,'EffNet':255,'SwAV':224}[name]
        x=transforms.Resize(size,interpolation=transforms.InterpolationMode.BILINEAR)(x)
        if name=='CLIP':x=transforms.Normalize([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711])(x);y=model.encode_image(x)
        else:
            x=transforms.Normalize([.485,.456,.406],[.229,.224,.225])(x);key='features.4' if name=='Alex2' else ('features.11' if name=='Alex5' else 'avgpool');y=model(x)[key]
        y=y.float().flatten(1)
    assert torch.isfinite(y).all();return y.float()

def pearson_rows(x):
    x=x.double();x=x-x.mean(-1,keepdim=True);norm=x.norm(dim=-1,keepdim=True);assert (norm>0).all();return x/norm

@torch.inference_mode()
def ranked(q,g,positive,concepts,kind,return_scores=False):
    if kind=='DINO':q=F.normalize(q.float(),dim=-1);g=F.normalize(g.float(),dim=-1)
    else:q=pearson_rows(q);g=pearson_rows(g)
    scores=q@g.T
    if kind!='DINO':scores=scores.clamp(-1,1)
    positive=torch.as_tensor(positive,device=q.device);correct=scores.gather(1,positive[:,None]);mask=torch.arange(len(g),device=q.device)[None,:]!=positive[:,None]
    smaller=(scores<correct)&mask;ties=(scores==correct)&mask
    afc=(smaller.sum(1)+(ties.sum(1)*.5 if kind=='DINO' else 0))/(len(g)-1)
    same=torch.as_tensor(np.asarray(concepts)[:,None]==np.asarray(concepts)[np.asarray(positive.cpu())][None,:],device=q.device).T
    assert same.shape==scores.shape and (same.sum(1)==10).all()
    within=scores.masked_fill(~same,-torch.inf);top=scores.topk(5,dim=1).indices
    result=dict(two_afc=afc,top1=(top[:,0]==positive).float(),top5=(top==positive[:,None]).any(1).float(),correct_similarity=correct[:,0],same_concept_top1=(within.argmax(1)==positive).float(),same_concept_two_afc=((smaller&same).sum(1)+(ties&same).sum(1)*(.5 if kind=='DINO' else 0))/9,exact_negative_ties=ties.sum(1))
    result={k:v.cpu() for k,v in result.items()}
    return (result,scores.cpu()) if return_scores else result

def paper_source_scope():
    sys.path.insert(0,str(R/'package_cache/evaluation_runtime_v275'))
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity
    scope=dict(np=np,torch=torch,transforms=transforms,tqdm=lambda x,*a,**kw:x,rgb2gray=rgb2gray,structural_similarity=structural_similarity)
    p=R/'code/HIVE_original_utils_eval_v276.py';pins=json.loads((R/'code/EEG_scoring_policy_v276.json').read_text());assert sha(p)==pins['original_metric_source_sha256']
    nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['two_way_identification','pixcorr','ssim']];assert len(nodes)==3
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope);return scope
