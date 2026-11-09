"""Pinned author classes/optimizer and independent raw-output loss derivatives."""
import ast
import numpy as np

CLASSES = ('Spatio_Temporal_CNN','ResidualAdd','FlattenHead','MLP_Projector','ENIGMA')

def author_namespace(models_source, training_source):
    import torch
    from einops.layers.torch import Rearrange
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, np=np, Rearrange=Rearrange)
    tree = ast.parse(models_source)
    nodes = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in CLASSES]
    assert tuple(n.name for n in nodes) == CLASSES
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_original_ENIGMA_classes','exec'),scope)
    tree = ast.parse(training_source)
    names = ('ClipLoss','train_epoch','prepare_optimizer')
    nodes = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert tuple(n.name for n in nodes) == names
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_original_ENIGMA_training','exec'),scope)
    return scope

def torch_loss(pred, image, text):
    import torch.nn.functional as F
    target = F.normalize(image,dim=-1)
    txt = F.normalize(text,dim=-1)
    labels = __import__('torch').arange(len(pred),device=pred.device)
    z, zr = pred@target.T, target@pred.T
    v, vr = pred@txt.T, txt@pred.T
    mse = F.mse_loss(pred,image)
    ce = (F.cross_entropy(z,labels)+F.cross_entropy(zr,labels))/2
    tce = (F.cross_entropy(v,labels)+F.cross_entropy(vr,labels))/2
    return mse+.5*ce+.05*tce, dict(raw_MSE=mse,image_CE=ce,text_CE=tce,
        output_norm=pred.norm(dim=-1).mean())

def numpy_loss_gradient(pred, image, text):
    p,y,t = (np.asarray(x,dtype=np.float64) for x in (pred,image,text))
    n,d = p.shape
    loss = np.square(p-y).mean()
    gradient = 2*(p-y)/(n*d)
    parts = dict(raw_MSE=float(loss))
    for name, target, weight in (('image_CE',y,.5),('text_CE',t,.05)):
        target = target/np.linalg.norm(target,axis=-1,keepdims=True)
        z = p@target.T
        maximum = z.max(axis=1,keepdims=True)
        row = np.exp(z-maximum); rowsum = row.sum(1,keepdims=True)
        maximum2 = z.max(axis=0,keepdims=True)
        col = np.exp(z-maximum2); colsum = col.sum(0,keepdims=True)
        value = ((np.log(rowsum[:,0])+maximum[:,0]-z.diagonal()).mean()+
                 (np.log(colsum[0])+maximum2[0]-z.diagonal()).mean())/2
        dz = ((row/rowsum-np.eye(n))+(col/colsum-np.eye(n)))/(2*n)
        gradient += weight*dz@target
        loss += weight*value
        parts[name] = float(value)
    return dict(loss=float(loss),prediction_gradient=gradient,parts=parts)

def role_rows(T, folds, role):
    T,folds = np.asarray(T,np.int64),np.asarray(folds,np.int64)
    assert len(T)==len(folds)==13230 and set(folds)=={0,1}
    assert np.all(folds.reshape(-1,10)==folds.reshape(-1,10)[:,:1])
    if role=='full': return T.copy(),np.empty(0,np.int64)
    held = int(role[-1]); assert role in ('oof0','oof1')
    fit, export = T[folds!=held],T[folds==held]
    assert not set(fit)&set(export) and not set(fit//10)&set(export//10)
    return fit,export

def canonical_concept_join(cache_folders, split_groups):
    """Number prefixes are source-local identifiers, never image-row indices."""
    def semantic_name(folder):
        number,name=folder.split('_',1)
        assert len(number)==5 and number.isdigit() and name
        return name
    canonical=[v for group in ('T','D','V') for v in split_groups[group]]
    assert len(cache_folders)==len(canonical)==1654
    assert len(set(cache_folders))==len(set(canonical))==1654
    by_name={semantic_name(v):v for v in canonical}
    cache_names=[semantic_name(v) for v in cache_folders]
    assert len(by_name)==len(set(cache_names))==1654 and set(cache_names)==set(by_name)
    return [dict(class_index=i,cache_folder=v,canonical_split_concept=by_name[cache_names[i]],
                 semantic_name=cache_names[i]) for i,v in enumerate(cache_folders)]
