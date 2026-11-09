"""Independent Dcheck numerical tools. These are not image quality metrics."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def energy_score(samples, observation):
    """Unbiased K-draw energy score, norm (not squared norm), lower is better.

    samples[B,K,D], observation[B,D]. Independent simulation draws are required.
    All ordered off-diagonal simulation pairs appear in the U-statistic.
    """
    assert samples.ndim==3 and observation.shape==(samples.shape[0],samples.shape[2])
    k=samples.shape[1];assert k>=2
    assert torch.isfinite(samples).all() and torch.isfinite(observation).all()
    first=(samples-observation[:,None]).norm(dim=-1).mean(1)
    pair=torch.cdist(samples,samples,p=2,compute_mode='donot_use_mm_for_euclid_dist')
    diagonal=pair.diagonal(dim1=1,dim2=2)
    assert torch.max(diagonal.abs())<1e-6
    return first-pair.sum((1,2))/(2*k*(k-1))


def binary_auc(labels, scores):
    """P(score(real)>score(sim)), with ties worth one half."""
    y=np.asarray(labels,dtype=np.int64);s=np.asarray(scores,dtype=np.float64)
    assert np.isfinite(s).all() and set(np.unique(y))=={0,1} and y.shape==s.shape
    order=np.argsort(s,kind='stable');rank=np.empty(len(s),dtype=np.float64)
    start=0
    while start<len(s):
        stop=start+1
        while stop<len(s) and s[order[stop]]==s[order[start]]:stop+=1
        rank[order[start:stop]]=(start+1+stop)/2;start=stop
    n1=int(y.sum());n0=len(y)-n1
    return float((rank[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def classification_summary(labels, logits):
    y=np.asarray(labels,dtype=np.int64);scores=np.asarray(logits,dtype=np.float64)
    prediction=scores>=0
    return dict(AUC_real_positive=binary_auc(y,scores),
        balanced_accuracy=float(((prediction[y==1]).mean()+(~prediction[y==0]).mean())/2),
        real_recall=float(prediction[y==1].mean()),simulated_recall=float((~prediction[y==0]).mean()),
        BCE=float((np.logaddexp(0,scores)-y*scores).mean()),samples=len(y),
        real_samples=int(y.sum()),simulated_samples=int((y==0).sum()))


def classifier_features(ci,ce,pca,subject_index,subject_count=10):
    assert ci.shape==ce.shape and ci.ndim==2
    center=pca['mean'];basis=pca['basis']
    image=(ci-center)@basis;condition=(ce-center)@basis
    scalars=torch.stack([ci.norm(dim=-1),ce.norm(dim=-1),(ce-ci).norm(dim=-1),
                         F.cosine_similarity(ci,ce)],dim=-1)
    subject=torch.zeros(len(ci),subject_count);subject[:,subject_index]=1
    # Full CE coordinates retain off-image-PCA deviations; products expose
    # content/condition associations to the controlled linear classifier.
    return torch.cat([image,ce,image*condition,scalars,subject],dim=-1)


def fit_two_sample(train_real,train_sim,test_real,test_sim,plan):
    """Balanced classifiers fit only on Dfit; one frozen capacity/optimizer.

    Positive=real CE, negative=simulated C. Both query conditions use the same
    target-side features and subject metadata. No Dcheck standardization fit.
    """
    assert train_real.shape==train_sim.shape and test_real.shape==test_sim.shape
    x=torch.cat([train_real,train_sim]);y=torch.cat([torch.ones(len(train_real)),torch.zeros(len(train_sim))])
    z=torch.cat([test_real,test_sim]);zy=torch.cat([torch.ones(len(test_real)),torch.zeros(len(test_sim))])
    assert torch.isfinite(x).all() and torch.isfinite(z).all()
    mean=x.mean(0);std=x.std(0,unbiased=False);std[std<1e-6]=1
    x=(x-mean)/std;z=(z-mean)/std
    reports={};states={}
    for architecture in plan['architectures']:
        torch.manual_seed(plan['seed'])
        if architecture=='linear':
            model=torch.nn.Linear(x.shape[1],1)
            # Neutral convex-classifier start avoids an arbitrary initial
            # orientation/threshold dominating the small positive controls.
            torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias)
        elif architecture=='mlp64':
            model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        else:raise ValueError(architecture)
        optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
        generator=torch.Generator().manual_seed(plan['seed'])
        for epoch in range(plan['epochs']):
            model.train();order=torch.randperm(len(x),generator=generator)
            for begin in range(0,len(order),plan['batch_size']):
                rows=order[begin:begin+plan['batch_size']]
                prediction=model(x[rows]).flatten();loss=F.binary_cross_entropy_with_logits(prediction,y[rows])
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode():
            tr=torch.cat([model(x[i:i+1024]).flatten() for i in range(0,len(x),1024)]).numpy()
            te=torch.cat([model(z[i:i+1024]).flatten() for i in range(0,len(z),1024)]).numpy()
        reports[architecture]=dict(parameters=sum(p.numel() for p in model.parameters()),
            feature_dimension=x.shape[1],fit=classification_summary(y.numpy(),tr),
            Dcheck=classification_summary(zy.numpy(),te),
            initialization='zero_weights_and_bias' if architecture=='linear' else 'torch_default_seed1729',
            optimizer_updates=plan['epochs']*math.ceil(len(x)/plan['batch_size']),
            training_example_exposures=plan['epochs']*len(x),
            fixed_plan=plan,uses_Dcheck_for_fitting_or_selection=False)
        states[architecture]=dict(model=model.state_dict(),mean=mean,std=std,plan=plan)
    return reports,states


def distribution_summary(real,sim):
    """Complete Dcheck residual moments; no inference of high-dimensional KL."""
    assert real.shape==sim.shape and real.ndim==2 and len(real)>1
    a=real.double();b=sim.double();ma=a.mean(0);mb=b.mean(0)
    ac=a-ma;bc=b-mb;n=len(a)
    ca=ac.T@ac/n;cb=bc.T@bc/n
    ta=torch.trace(ca);tb=torch.trace(cb)
    # Gram eigenvalues are sufficient when queries<dimensions (830 vs1024).
    def spectrum(centered):
        if centered.shape[0]<centered.shape[1]:matrix=centered@centered.T/n
        else:matrix=centered.T@centered/n
        e=torch.linalg.eigvalsh(matrix).flip(0).clamp_min(0)
        return e[:32].tolist()
    def norm_stats(value):
        v=value.norm(dim=-1)
        return dict(mean=float(v.mean()),std=float(v.std(unbiased=False)),
            q05=float(torch.quantile(v,.05)),q50=float(torch.quantile(v,.5)),q95=float(torch.quantile(v,.95)))
    return dict(real_bias_norm=float(ma.norm()),simulated_bias_norm=float(mb.norm()),
        bias_difference_norm=float((ma-mb).norm()),real_covariance_trace=float(ta),
        simulated_covariance_trace=float(tb),covariance_trace_ratio=float(tb/ta) if ta>0 else None,
        covariance_relative_Frobenius=float((ca-cb).norm()/ca.norm()) if ca.norm()>0 else None,
        real_covariance_eigenvalues_top32=spectrum(ac),simulated_covariance_eigenvalues_top32=spectrum(bc),
        real_error_norm=norm_stats(a),simulated_error_norm=norm_stats(b),queries=n)


def confusion_js(real_indices,sim_indices,positive_concepts,gallery_concepts):
    """Conditional wrong-concept confusion; common zero cells need no epsilon."""
    classes=sorted(set(gallery_concepts));mapping={c:i for i,c in enumerate(classes)}
    p=np.zeros((len(classes),len(classes)),dtype=np.float64);q=p.copy()
    for truth,ri,si in zip(positive_concepts,real_indices,sim_indices):
        row=mapping[truth];r=mapping[gallery_concepts[int(ri)]];s=mapping[gallery_concepts[int(si)]]
        if r!=row:p[row,r]+=1
        if s!=row:q[row,s]+=1
    p/=len(positive_concepts);q/=len(positive_concepts)
    # Add a correct-concept category per row, retaining total probability 1.
    total=np.zeros(len(classes))
    for c in positive_concepts:total[mapping[c]]+=1/len(positive_concepts)
    p[np.arange(len(classes)),np.arange(len(classes))]=total-p.sum(1)
    q[np.arange(len(classes)),np.arange(len(classes))]=total-q.sum(1)
    p=np.maximum(p,0);q=np.maximum(q,0);m=(p+q)/2
    js=0.
    for v in [p,q]:
        mask=v>0;js+=.5*np.sum(v[mask]*np.log(v[mask]/m[mask]))
    return dict(jensen_shannon_nats=float(js),real_concept_correct_fraction=float(np.trace(p)),
                simulated_concept_correct_fraction=float(np.trace(q)),
                query_classes=len(set(positive_concepts)),gallery_classes=len(classes))
