"""Small adapters around pinned, unchanged original ATM model/dataset functions.

The legacy modules eagerly load CLIP and training-time test data at import/main.
Select only their original AST definitions; do not edit model or loss bodies.
The prior epoch hook adds logging/checkpoints after an epoch, preserving updates.
"""
import ast
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import types

import numpy as np
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from einops.layers.torch import Rearrange,Reduce

MAIN='49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d'
MODEL_NAMES=['Config','iTransformer','PatchEmbedding','ResidualAdd','FlattenHead','Enc_eeg','Proj_eeg','ATMS']


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def check_source(source):
    source=Path(source).resolve()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source).decode().strip()==MAIN
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source).strip()
    return source


def load_model_training(source):
    source=check_source(source);sys.path.insert(0,str(source))
    from models.subject_layers.Transformer_EncDec import Encoder,EncoderLayer
    from models.subject_layers.SelfAttention_Family import FullAttention,AttentionLayer
    from models.subject_layers.Embed import DataEmbedding
    from models.loss import ClipLoss
    path=source/'Generation/ATMS_reconstruction.py';tree=ast.parse(path.read_text())
    wanted=set(MODEL_NAMES+['extract_id_from_string','train_model'])
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
    assert {n.name for n in nodes}==wanted
    scope={k:globals()[k] for k in ['torch','nn','Tensor','np','math','re','Rearrange','Reduce','F']}
    scope.update(Encoder=Encoder,EncoderLayer=EncoderLayer,FullAttention=FullAttention,
                 AttentionLayer=AttentionLayer,DataEmbedding=DataEmbedding,ClipLoss=ClipLoss)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return types.SimpleNamespace(**{name:scope[name] for name in wanted},source_sha256=digest(path))


def load_prior(source):
    source=check_source(source);path=source/'Generation/diffusion_prior.py'
    spec=importlib.util.spec_from_file_location('pinned_paper_main_prior',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def add_prior_epoch_hook(pipe,source,callback):
    """Append one callback after the original epoch print; no update expression changes."""
    path=check_source(source)/'Generation/diffusion_prior.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Pipe')
    original=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    adapted=copy.deepcopy(original)
    loops=[n for n in adapted.body if isinstance(n,ast.For)]
    assert len(loops)==1 and isinstance(loops[0].target,ast.Name) and loops[0].target.id=='epoch'
    assert isinstance(loops[0].body[-1],ast.Expr)
    assert isinstance(loops[0].body[-1].value,ast.Call) and loops[0].body[-1].value.func.id=='print'
    hook=ast.parse('_paper_epoch_hook(epoch, loss_epoch, optimizer, lr_scheduler)').body[0]
    loops[0].body.append(hook)
    stripped=copy.deepcopy(adapted);next(n for n in stripped.body if isinstance(n,ast.For)).body.pop()
    assert ast.dump(stripped,include_attributes=False)==ast.dump(original,include_attributes=False)
    module=sys.modules.get(type(pipe).__module__)
    # Modules loaded by spec need not be registered in sys.modules.
    scope=dict(type(pipe).train.__globals__);scope['_paper_epoch_hook']=callback
    exec(compile(ast.fix_missing_locations(ast.Module(body=[adapted],type_ignores=[])),str(path),'exec'),scope)
    pipe.train=types.MethodType(scope['train'],pipe)


def native_dataset(source,data_root,images_root,feature_root,work_root):
    """Execute the old loader class with its verified cache, and forbid test reads."""
    source=check_source(source);data_root=Path(data_root).resolve();images_root=Path(images_root).resolve()
    feature=Path(feature_root).resolve()/'ViT-H-14_features_train.pt';work=Path(work_root).resolve()
    work.mkdir(exist_ok=False);(work/feature.name).symlink_to(feature)
    eeg=(data_root/'sub-01/preprocessed_eeg_training.npy').resolve()
    path=source/'Generation/eegdatasets_leaveone.py';tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EEGDataset'];assert len(nodes)==1
    scope=dict(torch=torch,np=np,os=os,F=F,model_type='ViT-H-14',device='cpu',data_path=str(data_root),
        img_directory_training=str(images_root/'image_set/training_images'),
        img_directory_test='/OFFICIAL_TEST_LOCKED_DO_NOT_READ')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    original_np,original_torch=np.load,torch.load
    def load_eeg(path,*args,**kwargs):
        assert Path(path).resolve()==eeg,'Attempt to read EEG outside the declared training input'
        return original_np(path,*args,**kwargs)
    def load_feature(path,*args,**kwargs):
        assert Path(path).resolve()==feature,'Attempt to read a non-training feature cache'
        return original_torch(path,*args,**kwargs)
    previous=Path.cwd()
    try:
        np.load,torch.load=load_eeg,load_feature;os.chdir(work)
        result=scope['EEGDataset'](str(data_root),subjects=['sub-01'],train=True)
    finally:
        np.load,torch.load=original_np,original_torch;os.chdir(previous)
    assert len(result)==66160 and result.data.shape==(66160,63,250)
    return result
