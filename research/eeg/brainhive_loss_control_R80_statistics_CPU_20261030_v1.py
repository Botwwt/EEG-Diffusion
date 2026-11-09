"""Fixed paired statistics, full denominator, no training-seed claims."""
import numpy as np

COMMON=('DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED')
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
COMPARISONS=(('ordinary','F_only'),('ordinary','token_only'),('F_only','parent'),('token_only','parent'),('ordinary','parent'))


def paired(values_left,values_right,subject_draws,concept_draws):
    left=np.asarray(values_left,dtype=np.float64);right=np.asarray(values_right,dtype=np.float64)
    if left.shape!=(10,200) or right.shape!=left.shape:raise ValueError('Full10x200 denominator required')
    if not np.isfinite(left).all() or not np.isfinite(right).all():raise ValueError('Undefined metric retains UNKNOWN, never drop queries')
    difference=left-right
    draws=np.asarray([difference[np.ix_(si,ci)].mean() for si,ci in zip(subject_draws,concept_draws)])
    weighted=np.asarray([np.einsum('s,sc,c->',np.bincount(si,minlength=10),difference,np.bincount(ci,minlength=200))/2000 for si,ci in zip(subject_draws,concept_draws)])
    error=float(np.max(abs(draws-weighted)));assert error<1e-12
    return dict(left_mean=float(left.mean()),right_mean=float(right.mean()),paired_left_minus_right=float(difference.mean()),descriptive_subject_concept_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),per_subject_difference=difference.mean(axis=1).tolist(),defined=2000,independent_weighted_bootstrap_maxerror=error)


def fixed_draws():
    rng=np.random.default_rng(1729);subjects=[];concepts=[]
    for unused in range(1000):subjects.append(rng.integers(10,size=10));concepts.append(rng.integers(200,size=200))
    return subjects,concepts
