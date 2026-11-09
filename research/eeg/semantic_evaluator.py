"""Frozen SEED models, streamed image features, and explicit paired-record handling.

Object F1 uses the pinned official implementation and its default .05:.05:.95
threshold grid. This is the public-code variant, not the paper's .01 grid.
EffNet is Pearson correlation similarity (1 - official correlation distance).
"""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/code/SEED_5327a134')
WEIGHTS=Path('/dev/shm/eeg_sota_autoresearch_20261018_v1/semantic_evaluation')
sys.path[:0]=['/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runtime/evaluation',
             '/root/mced_representation_runtime_v1/lib/python3.12/site-packages',str(SOURCE)]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
from PIL import Image
import torch
from torchvision.models import efficientnet_b1
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms


def effnet_model(device='cuda'):
    model=efficientnet_b1(weights=None)
    state=torch.load(WEIGHTS/'efficientnet_b1-c27df63c.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    model=create_feature_extractor(model,return_nodes=['avgpool']).eval().requires_grad_(False).to(device)
    preprocess=transforms.Compose([
        transforms.Resize(255,interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
    return model,preprocess


@torch.inference_mode()
def effnet_features(paths,model,preprocess,batch_size=32):
    output=[];device=next(model.parameters()).device
    for start in range(0,len(paths),batch_size):
        tensors=[]
        for path in paths[start:start+batch_size]:
            with Image.open(path) as im:arr=np.array(im.convert('RGB'),copy=True)
            tensors.append(preprocess(torch.from_numpy(arr).permute(2,0,1).float()/255))
        # The official metric also requires uniform shapes; do not silently crop.
        assert len({tuple(t.shape) for t in tensors})==1
        feature=model(torch.stack(tensors).to(device))['avgpool'].flatten(1)
        assert torch.isfinite(feature).all();output.append(feature.cpu().numpy())
    return np.concatenate(output)


def correlation_similarity(left,right):
    left=np.asarray(left,dtype=np.float64);right=np.asarray(right,dtype=np.float64)
    a=left-left.mean(-1,keepdims=True);b=right-right.mean(-1,keepdims=True)
    denominator=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    assert np.all(denominator>0), 'Undefined official Pearson similarity'
    return (a*b).sum(-1)/denominator


def caption_models(device='cuda'):
    from transformers import AutoProcessor,AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer
    processor=AutoProcessor.from_pretrained(WEIGHTS/'git_large_coco',local_files_only=True)
    model,info=AutoModelForCausalLM.from_pretrained(WEIGHTS/'git_large_coco',local_files_only=True,
        torch_dtype=torch.float32,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
    model=model.eval().requires_grad_(False).to(device)
    text=SentenceTransformer(str(WEIGHTS/'sentence_minilm'),device=device,local_files_only=True)
    text.eval().requires_grad_(False)
    return processor,model,text,info


@torch.inference_mode()
def caption_features(paths,processor,model,text,batch_size=16):
    captions=[];device=next(model.parameters()).device
    for start in range(0,len(paths),batch_size):
        images=[]
        for path in paths[start:start+batch_size]:
            with Image.open(path) as im:images.append(im.convert('RGB'))
        pixels=processor(images=images,return_tensors='pt').pixel_values.to(device)
        tokens=model.generate(pixel_values=pixels,max_new_tokens=50,do_sample=False)
        captions.extend(processor.batch_decode(tokens,skip_special_tokens=True))
    features=text.encode(captions,batch_size=128,show_progress_bar=False,
                         convert_to_numpy=True,normalize_embeddings=True)
    assert np.isfinite(features).all()
    return captions,features


def detection_model(device='cuda:0'):
    from mmengine.config import Config
    from mmdet.apis import DetInferencer
    from mmdet.evaluation import get_classes
    cfg=Config.fromfile(str(SOURCE/'mmdet/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py'))
    cfg.model.language_model.name=str(WEIGHTS/'bert_base')
    cfg.model.test_cfg.chunked_size=-1
    detector=DetInferencer(model=cfg,weights=str(WEIGHTS/'grounding_dino_swin-l_pretrain_all-56d69e78.pth'),
                           device=device,show_progress=False)
    detector.model.eval().requires_grad_(False)
    classes=tuple(get_classes('coco_gender'));assert len(classes)==82
    return detector,classes,cfg


@torch.inference_mode()
def detection_features(paths,detector,classes,batch_size=4):
    predictions=[]
    for start in range(0,len(paths),batch_size):
        out=detector([str(p) for p in paths[start:start+batch_size]],texts=[classes],
            batch_size=batch_size,no_save_vis=True,no_save_pred=True)
        for row in out['predictions']:
            assert len(row['labels'])==len(row['scores'])==len(row['bboxes'])
            scores=np.asarray(row['scores']);assert np.isfinite(scores).all()
            assert np.all(scores[:-1]>=scores[1:]), 'Official F1 assumes descending scores'
            assert all(0<=label<len(classes) for label in row['labels'])
            predictions.append(row)
    assert len(predictions)==len(paths)
    return predictions
