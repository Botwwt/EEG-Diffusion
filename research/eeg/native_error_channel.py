"""CPU error-channel estimators; no fitted scientific model is shipped here.

Fits a single subject/R in the raw native condition coordinates. Only T OOF
residuals and Dfit teacher/full pairs may enter fit(). Dcheck is never accepted
by fit(). Actual campaign provenance must be verified by the cache consumer.
"""
import hashlib
import math

import torch


def stream_seed(seed, *parts):
    value='|'.join([str(seed), *map(str,parts)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big') % (2**63-1)


def validate_matrix(x, n=None, d=None):
    assert isinstance(x,torch.Tensor) and x.device.type=='cpu'
    assert x.ndim==2 and x.dtype==torch.float32 and torch.isfinite(x).all()
    if n is not None:assert len(x)==n
    if d is not None:assert x.shape[1]==d


def canonical_pca(x, components):
    """All fitting inputs must be T image features, once per unique image."""
    validate_matrix(x);assert 1<=components<=min(x.shape[1],len(x)-1)
    mean=x.double().mean(0);centered=x.double()-mean
    covariance=centered.T@centered/len(x)
    eigenvalues,eigenvectors=torch.linalg.eigh(covariance)
    basis=eigenvectors[:,-components:].flip(1)
    signs=torch.sign(basis[basis.abs().argmax(0),torch.arange(components)])
    signs[signs==0]=1;basis*=signs
    return dict(mean=mean.float(),basis=basis.float(),
                eigenvalues=eigenvalues[-components:].flip(0).float(),whiten=False)


def calibrate_from_dfit(ci, oof, full, fold_weights):
    """OOF/full distribution calibration; does not regress on Dcheck."""
    validate_matrix(ci);validate_matrix(full,len(ci),ci.shape[1])
    assert oof.dtype==torch.float32 and oof.device.type=='cpu'
    assert oof.shape==(2,*ci.shape) and torch.isfinite(oof).all()
    w=torch.tensor(fold_weights,dtype=torch.float64)
    assert w.shape==(2,) and (w>0).all() and abs(float(w.sum())-1)<1e-8
    a=oof.double()-ci.double()[None];b=full.double()-ci.double()
    mu_oof=(a.mean(1)*w[:,None]).sum(0);mu_full=b.mean(0)
    trace_oof=(((a-mu_oof).square().sum(-1)).mean(1)*w).sum()
    trace_full=(b-mu_full).square().sum(-1).mean()
    assert trace_oof>0 and trace_full>0,'Degenerate Dfit error; do not silently invent variance'
    scale=torch.sqrt(trace_full/trace_oof)
    return dict(oof_bias=mu_oof.float(),full_bias=mu_full.float(),scale=float(scale),
                oof_trace=float(trace_oof),full_trace=float(trace_full),
                fold_weights=list(map(float,fold_weights)),fit_queries=len(ci),
                fit_role='Dfit',uses_Dcheck=False)


def apply_calibration(residual, calibration):
    return calibration['full_bias']+calibration['scale']*(residual-calibration['oof_bias'])


class NativeErrorChannel:
    VERSION='raw_native_1024_oof_dfit_affine_pca_knn_v1'

    @classmethod
    def fit(cls, t_ci, t_ce_oof, t_concepts, dfit_ci, dfit_oof, dfit_full,
            fold_weights, pca_components=32, neighbors=32, shrinkage=.1):
        validate_matrix(t_ci);n,d=t_ci.shape
        validate_matrix(t_ce_oof,n,d)
        assert len(t_concepts)==n and len(set(t_concepts))>=neighbors+1
        assert isinstance(neighbors,int) and neighbors>0 and 0<shrinkage<=1
        validate_matrix(dfit_ci,d=d)
        calibration=calibrate_from_dfit(dfit_ci,dfit_oof,dfit_full,fold_weights)
        residual=apply_calibration(t_ce_oof-t_ci,calibration)
        mu=residual.double().mean(0);centered=residual.double()-mu
        covariance=centered.T@centered/n
        trace=torch.trace(covariance);assert trace>0
        shrunk=(1-shrinkage)*covariance+shrinkage*(trace/d)*torch.eye(d,dtype=torch.float64)
        factor=torch.linalg.cholesky(shrunk).float()
        assert torch.isfinite(factor).all()
        pca=canonical_pca(t_ci,pca_components)
        coordinates=(t_ci-pca['mean'])@pca['basis']
        state=dict(version=cls.VERSION,dimensions=d,T_rows=n,T_concepts=list(t_concepts),
                   pca=pca,coordinates=coordinates.contiguous(),residual=residual.contiguous(),
                   bias=mu.float(),covariance_cholesky=factor,calibration=calibration,
                   shrinkage=float(shrinkage),neighbors=neighbors,
                   energy=float(residual.double().square().sum(-1).mean()),
                   covariance_trace=float(trace),fit_roles=['T_OOF','Dfit'],
                   Dcheck_used=False,extra_external_images=0)
        return cls(state)

    def __init__(self,state):
        assert state['version']==self.VERSION and state['fit_roles']==['T_OOF','Dfit']
        assert not state['Dcheck_used'] and state['extra_external_images']==0
        self.state=state;self.d=state['dimensions']
        self.concepts=state['T_concepts'];self.k=state['neighbors']
        self.keys=state['coordinates'];self.key_norm=self.keys.square().sum(-1)

    def nearest(self,ci,concepts):
        """K distinct source concepts, all different from the query concept.

        Query concepts are training/calibration metadata. They are never target
        concepts supplied at inference: the channel is used in adapter training.
        """
        validate_matrix(ci,d=self.d);assert len(ci)==len(concepts)
        p=self.state['pca'];query=(ci-p['mean'])@p['basis'];result=[]
        for start in range(0,len(ci),128):
            q=query[start:start+128]
            dist=q.square().sum(-1)[:,None]+self.key_norm[None]-2*q@self.keys.T
            # Stable sort resolves exact ties by original T array order.
            ordered=torch.argsort(dist,dim=-1,stable=True)
            for j,ranks in enumerate(ordered):
                forbidden=concepts[start+j];seen={forbidden};chosen=[]
                for index in ranks.tolist():
                    concept=self.concepts[index]
                    if concept not in seen:
                        chosen.append(index);seen.add(concept)
                        if len(chosen)==self.k:break
                assert len(chosen)==self.k
                result.append(chosen)
        return torch.tensor(result,dtype=torch.long)

    def sample(self,arm,ci,concepts,seeds,neighbors=None):
        validate_matrix(ci,d=self.d)
        assert len(ci)==len(concepts)==len(seeds)
        assert arm in ['Clean','Iso','Moment','Conditional']
        if arm=='Clean':return ci.clone(),dict(arm=arm,source_indices=None)
        if arm=='Conditional':
            neighbors=self.nearest(ci,concepts) if neighbors is None else neighbors
            assert neighbors.shape==(len(ci),self.k) and neighbors.dtype==torch.long
        delta=[];indices=[]
        for i,seed in enumerate(seeds):
            generator=torch.Generator(device='cpu').manual_seed(int(seed))
            if arm=='Conditional':
                selected=int(neighbors[i,torch.randint(self.k,(),generator=generator)])
                assert self.concepts[selected]!=concepts[i]
                assert len(set(self.concepts[j] for j in neighbors[i].tolist()))==self.k
                value=self.state['residual'][selected];indices.append(selected)
            else:
                z=torch.randn(self.d,generator=generator,dtype=torch.float32)
                if arm=='Iso':value=z*math.sqrt(self.state['energy']/self.d)
                else:value=self.state['bias']+self.state['covariance_cholesky']@z
            delta.append(value)
        output=ci+torch.stack(delta)
        assert torch.isfinite(output).all()
        return output,dict(arm=arm,source_indices=indices if arm=='Conditional' else None)
