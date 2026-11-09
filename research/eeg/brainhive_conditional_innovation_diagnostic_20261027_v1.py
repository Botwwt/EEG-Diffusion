"""Fixed Gaussian working-model diagnostic in an explicit VAE subspace.

No claim that EEG or HIVE trajectories are Gaussian. Shrinkage is declared model
regularization, not a proof of a valid posterior or physiological noise model.
"""
import numpy as np
import torch

SCHEMA='BrainHIVE_conditional_VAE_innovation_diagnostic_20261027_v1'
ARMS=('c_only','paired_joint','sameconcept_wrong_train','sameconcept_wrong_inference')
TIMES=(999,749,499,249)
DIM=128
SHRINK=.1


def standardize(x):
    mean=x.mean(0);std=x.std(0,unbiased=False)
    return (x-mean)/std.clamp_min(1e-6),mean,std.clamp_min(1e-6),int((std<1e-6).sum())


def fit_joint(image,condition,eeg,shrink=SHRINK):
    assert shrink==SHRINK and len(image)==len(condition)==len(eeg)
    u,um,us,uc=standardize(image.double());c,cm,cs,cc=standardize(condition.double());e,em,es,ec=standardize(eeg.double())
    assert u.shape[1]==DIM and c.shape[1]>0 and e.shape[1]>0
    z=torch.cat([u,c,e],1);s=(z.T@z)/len(z)
    s.mul_(1-shrink);s.diagonal().add_(shrink)
    a=DIM;b=a+c.shape[1]
    suu=s[:a,:a];suc=s[:a,a:b];see=s[b:,b:];sec=s[b:,a:b];sue=s[:a,b:];scc=s[a:b,a:b]
    lc=torch.linalg.cholesky(scc)
    auc=torch.cholesky_solve(suc.T,lc).T;aec=torch.cholesky_solve(sec.T,lc).T
    p=suu-auc@suc.T;r=see-aec@sec.T;cross=sue-auc@sec.T
    p=(p+p.T)/2;r=(r+r.T)/2
    lr=torch.linalg.cholesky(r);gain=torch.cholesky_solve(cross.T,lr).T
    post=p-gain@cross.T;post=(post+post.T)/2
    torch.linalg.cholesky(p);torch.linalg.cholesky(post)
    error=float(torch.linalg.vector_norm(gain@r-cross)/torch.linalg.vector_norm(cross).clamp_min(1e-12))
    assert error<1e-8 and torch.isfinite(gain).all()
    return dict(image_mean=um,image_std=us,condition_mean=cm,condition_std=cs,eeg_mean=em,eeg_std=es,
                mean_image_from_c=auc,mean_eeg_from_c=aec,prior_cov=p,EEG_gain=gain,joint_cov=post,
                constant_coordinate_counts=[uc,cc,ec],solve_relative_residual=error,
                fit_rows=len(image),shrink=shrink,working_model='standardized empirical joint covariance shrink .1 toward identity')


def clean_means(model,c,e):
    c=(c.double()-model['condition_mean'])/model['condition_std']
    e=(e.double()-model['eeg_mean'])/model['eeg_std']
    base=c@model['mean_image_from_c'].T
    innovation=e-c@model['mean_eeg_from_c'].T
    joint=base+innovation@model['EEG_gain'].T
    return base,joint,innovation


def noisy_update(mean,cov,noisy,alpha,sigma,image_std):
    # u is standardized VAE PC, so projected native Gaussian noise has this
    # diagonal covariance. It is NOT sigma^2 I in whitened coordinates.
    noise=(sigma/image_std).square()
    system=alpha**2*cov+torch.diag(noise)
    gain=alpha*torch.cholesky_solve(cov.T,torch.linalg.cholesky(system)).T
    posterior=mean+(noisy-alpha*mean)@gain.T
    identity=torch.eye(DIM,device=cov.device,dtype=cov.dtype)-alpha*gain
    postcov=identity@cov@identity.T+(gain*noise[None])@gain.T
    postcov=(postcov+postcov.T)/2;torch.linalg.cholesky(postcov)
    return posterior,postcov


def score(pred,target,cov,image_std):
    residual=target-pred
    chol=torch.linalg.cholesky(cov);white=torch.linalg.solve_triangular(chol,residual.T,upper=False).T
    nll=.5*(white.square().sum(1)+2*torch.log(chol.diag()).sum()+DIM*np.log(2*np.pi))/DIM
    physical=(residual*image_std).square().mean(1)
    return dict(NLL_per_dim=nll,MSE_native_PC=physical,standardized_quadratic_per_dim=white.square().mean(1))


def bootstrap(values,concept_ids,subjects):
    values=np.asarray(values,np.float64);assert values.shape==(10,820)
    grouped=values.reshape(10,82,10).mean(2)
    assert len(concept_ids)==820 and len(subjects)==10
    g=np.random.default_rng(1729);draws=[]
    for _ in range(1000):
        ss=g.integers(0,10,10);cc=g.integers(0,82,82);draws.append(grouped[ss][:,cc].mean())
    return dict(mean=float(values.mean()),descriptive_subject_concept_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),
                subjects_positive=int((values.mean(1)>0).sum()),concepts_positive=int((grouped.mean(0)>0).sum()),
                exposed_development_only=True)


def readout(scores,concept_ids,subjects):
    # [subject,image,time,arm,metric]. Every registered time is equally weighted.
    assert scores.shape==(10,820,4,4,3) and np.isfinite(scores).all()
    means={a:dict(zip(('NLL_per_dim','MSE_native_PC','standardized_quadratic_per_dim'),scores[:,:,:,j].mean((0,1,2)).tolist())) for j,a in enumerate(ARMS)}
    contrasts={}
    for j,a in enumerate(ARMS):
        if a=='paired_joint':continue
        contrasts[a]={key:bootstrap((scores[:,:,:,j,k]-scores[:,:,:,1,k]).mean(2),concept_ids,subjects)
                      for k,key in enumerate(('NLL_improvement_per_dim','native_PC_MSE_reduction'))}
    base=contrasts['c_only'];wrong=contrasts['sameconcept_wrong_train'];paired=means['paired_joint']
    relative=base['native_PC_MSE_reduction']['mean']/means['c_only']['MSE_native_PC']
    increment=(relative>=.005 and base['native_PC_MSE_reduction']['descriptive_subject_concept_bootstrap95'][0]>0
               and all(contrasts[a]['NLL_improvement_per_dim']['mean']>=.001
                       and contrasts[a]['NLL_improvement_per_dim']['descriptive_subject_concept_bootstrap95'][0]>0
                       and contrasts[a]['native_PC_MSE_reduction']['mean']>0
                       for a in ('c_only','sameconcept_wrong_train'))
               and base['native_PC_MSE_reduction']['subjects_positive']>=7
               and .5<=paired['standardized_quadratic_per_dim']<=2.)
    return dict(status='COMPLETE_ALL10_T_AND_DFIT8200_CONDITIONAL_INNOVATION_SUBSPACE_READOUT',means=means,paired_improvements=contrasts,
                relative_PC_MSE_improvement_vs_c=relative,conditional_increment_supported_in_registered_subspace=bool(increment),
                action='SUPPORTS_FROZEN_DCHECK_DIAGNOSTIC_ONLY' if increment else 'STOP_THIS_WORKING_MODEL_NO_IMAGE_ADMISSION',
                Gaussian_model_validity_established=False,full16384_image_increment_established=False,actual_HIVE_trajectory_applicability_established=False,
                actual_image_gain_established=False,automatic_new_image_admission=False,novelty_established=False,SOTA_established=False)
