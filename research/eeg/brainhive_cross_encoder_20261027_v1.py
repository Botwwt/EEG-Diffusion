"""D15 fixed conditional linear visual readout; no candidate image inference yet."""
import numpy as np
import torch
import torch.nn.functional as F

SCHEMA='BrainHIVE_old_ATMS_conditional_visual_information_20261027_v1'
ARMS=('c_only','ATMS_only','c_plus_ATMS_innovation','shuffled_innovation')


def ridge(x,y,regularization=.1):
    assert regularization==.1 and x.ndim==y.ndim==2 and len(x)==len(y)
    gram=x.T@x/len(x);cross=x.T@y/len(x)
    lam=regularization*gram.diagonal().mean()
    assert torch.isfinite(lam) and lam>0
    chol=torch.linalg.cholesky(gram+lam*torch.eye(x.shape[1],device=x.device,dtype=x.dtype))
    coefficient=torch.cholesky_solve(cross,chol)
    assert torch.isfinite(coefficient).all()
    return coefficient,lam


def fit(c,a,y,permutation):
    bc,lc=ridge(c,y);ca,lca=ridge(c,a);ba,la=ridge(a,y)
    innovation=a-c@ca;residual=y-c@bc
    extra,le=ridge(innovation,residual);null,ln=ridge(innovation[permutation],residual)
    return dict(c_to_y=bc,c_to_a=ca,a_to_y=ba,innovation_to_residual=extra,shuffled_to_residual=null,
        regularization=dict(c_y=float(lc),c_a=float(lca),a_y=float(la),innovation=float(le),shuffled=float(ln)))


def predict(models,c,a,permutation):
    base=c@models['c_to_y'];e=a-c@models['c_to_a']
    return dict(c_only=base,ATMS_only=a@models['a_to_y'],c_plus_ATMS_innovation=base+e@models['innovation_to_residual'],
        shuffled_innovation=base+e[permutation]@models['shuffled_to_residual'])


def metrics(prediction,target,groups):
    assert prediction.shape==target.shape and prediction.ndim==2
    mse=(prediction-target).double().square().mean(-1)
    p=F.normalize(prediction,dim=-1);y=F.normalize(target,dim=-1);cosine=(p*y).sum(-1)
    loss=torch.empty(len(p),device=p.device);top=torch.empty_like(loss);margin=torch.empty_like(loss)
    for indices in groups:
        assert len(indices)==10;ix=torch.as_tensor(indices,device=p.device)
        similarity=p[ix]@y[ix].T;labels=torch.arange(10,device=p.device)
        loss[ix]=F.cross_entropy(similarity/.1,labels,reduction='none')
        top[ix]=(similarity.argmax(-1)==labels).float()
        margin[ix]=similarity.diagonal()-(similarity.sum(-1)-similarity.diagonal())/9
    return dict(MSE=mse.cpu().numpy(),target_cosine=cosine.cpu().numpy(),within10_log_loss=loss.cpu().numpy(),
        within10_top1=top.cpu().numpy(),within10_cosine_margin=margin.cpu().numpy())


def paired_difference(values,other,subjects,concepts):
    subjects=np.asarray(subjects);concepts=np.asarray(concepts)
    snames=sorted(set(subjects));cnames=sorted(set(concepts));rng=np.random.default_rng(1729)
    draws_s=rng.integers(len(snames),size=(1000,len(snames)));draws_c=rng.integers(len(cnames),size=(1000,len(cnames)))
    result={}
    for key in values:
        difference=np.asarray(values[key],np.float64)-np.asarray(other[key],np.float64)
        cells=np.array([[difference[(subjects==s)&(concepts==c)].mean() for c in cnames] for s in snames])
        assert np.isfinite(cells).all()
        samples=cells[draws_s[:,:,None],draws_c[:,None,:]].mean((1,2))
        result[key]=dict(difference=float(difference.mean()),descriptive_subject_concept_bootstrap95=np.quantile(samples,[.025,.975]).tolist(),
            positive_concepts=int((cells.mean(0)>0).sum()),concepts=len(cnames),subjects=len(snames))
    return result


def readout(values,subjects,concepts):
    joint=values['c_plus_ATMS_innovation']
    # Signs are base/control minus joint for errors, positive supports improvement.
    comparisons={arm:paired_difference(values[arm],joint,subjects,concepts) for arm in ('c_only','ATMS_only','shuffled_innovation')}
    comparisons_fine={arm:paired_difference(joint,values[arm],subjects,concepts) for arm in ('c_only','shuffled_innovation')}
    means={arm:{key:float(v.mean()) for key,v in x.items()} for arm,x in values.items()}
    supported=True
    for arm in ('c_only','shuffled_innovation'):
        effect=comparisons[arm]
        supported&=effect['MSE']['difference']>=.005*means[arm]['MSE'] and effect['MSE']['descriptive_subject_concept_bootstrap95'][0]>0
        supported&=effect['within10_log_loss']['difference']>0 and effect['within10_log_loss']['descriptive_subject_concept_bootstrap95'][0]>0
        supported&=means['c_plus_ATMS_innovation']['within10_top1']>=means[arm]['within10_top1']
    return dict(status='D15_ALL10_FULL_DFIT_FIXED_LINEAR_INCREMENT_READOUT',means=means,error_improvement_over_joint=comparisons,
        joint_minus_reference=comparisons_fine,conditional_increment_supported=bool(supported),
        decision=dict(action='SUPPORTS_SEPARATE_REGISTERED_IMAGE_TEST' if supported else 'STOP_THIS_LINEAR_COMPLEMENTARITY_INSTANCE',
            no_automatic_image_admission=True,no_hyperparameter_search=True),
        effect_scope='fixed linear conditional visual readout on exposed Dfit, not neural information theorem or image gain',
        Dcheck_read=False,actual_image_gain_established=False,independent_confirmation=False,SOTA_established=False)


def self_test():
    torch.set_num_threads(2);generator=torch.Generator().manual_seed(1729)
    c=torch.randn(256,8,generator=generator,dtype=torch.float64);a=torch.randn(256,8,generator=generator,dtype=torch.float64)
    y=c@torch.randn(8,4,generator=generator,dtype=torch.float64)+a@torch.randn(8,4,generator=generator,dtype=torch.float64)
    order=torch.randperm(256,generator=generator);models=fit(c,a,y,order)
    pred=predict(models,c,a,order);assert (pred['c_plus_ATMS_innovation']-y).square().mean()<(pred['c_only']-y).square().mean()
    b,lam=ridge(c,y);direct=torch.linalg.solve(c.T@c/256+lam*torch.eye(8,dtype=c.dtype),c.T@y/256)
    assert torch.allclose(b,direct,atol=1e-12,rtol=1e-12)
    subjects=np.repeat(['s1','s2'],20);concepts=np.tile(np.repeat(['a','b'],10),2)
    values={arm:{key:np.full(40,value) for key,value in [('MSE',1.),('within10_log_loss',2.),('within10_top1',.1),('target_cosine',.2),('within10_cosine_margin',0.)]} for arm in ARMS}
    assert not readout(values,subjects,concepts)['conditional_increment_supported']
    for key,value in [('MSE',.8),('within10_log_loss',1.8),('within10_top1',.2)]:values['c_plus_ATMS_innovation'][key][:]=value
    assert readout(values,subjects,concepts)['conditional_increment_supported']
    return dict(status='D15_LINEAR_SOLVE_INCREMENT_AND_NULL_POSITIVE_GATE_TESTS_PASSED',checks=4,real_fit=False,CUDA_initialized=torch.cuda.is_initialized())
