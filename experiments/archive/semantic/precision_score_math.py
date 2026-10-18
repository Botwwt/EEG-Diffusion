"""Cross-modal condition-noise score identity; Gaussian corruption, arbitrary data.

The heat equation / score Fokker--Planck identity is classical. The candidate
uses a physical-repeat noise operator on the CONDITION axis, with matched
joint-score and isotropic controls. No unconditional quality guarantee follows.
"""
import torch


def precision_residual(model, x, z, t, v, chol, spatial_h=.2, variance_h=.05, drift_h=.02):
    """Full directional finite-difference trace, not a squared Hutchinson estimate.

    f=epsilon_x*, b=Sigma score_z=-L epsilon_z*/sqrt(v).
    d_v f = .5 tr(Sigma Hessian_z f) + J_z f b.
    Inputs x,z are held FIXED when changing v. All operations retain gradients.
    """
    batch,dim=z.shape;assert x.shape==z.shape and (v>variance_h).all()
    def both(xx,zz,vv,tt):return model(torch.cat([xx,zz],-1),tt,vv[:,None])
    center=both(x,z,v,t); f=center[:,:dim]; drift=-(center[:,dim:]@chol.T)/v.sqrt()[:,None]
    delta=chol.T*spatial_h
    directions=torch.cat([delta,-delta],0)
    xx=x[:,None,:].expand(-1,2*dim,-1).reshape(-1,dim)
    zz=(z[:,None,:]+directions[None,:,:]).reshape(-1,dim)
    vv=v[:,None].expand(-1,2*dim).reshape(-1);tt=t[:,None].expand(-1,2*dim).reshape(-1)
    probes=both(xx,zz,vv,tt)[:,:dim].reshape(batch,2*dim,dim)
    lap=(probes[:,:dim]+probes[:,dim:]-2*f[:,None,:]).sum(1)/(spatial_h**2)
    dv=(both(x,z,v+variance_h,t)[:,:dim]-both(x,z,v-variance_h,t)[:,:dim])/(2*variance_h)
    directional=(both(x,z+drift_h*drift,v,t)[:,:dim]-both(x,z-drift_h*drift,v,t)[:,:dim])/(2*drift_h)
    return dv-.5*lap-directional


def validate_math():
    """Exact AD check on a genuinely non-Gaussian joint mixture, then FD check."""
    torch.set_num_threads(2);dtype=torch.float64
    covariance=torch.tensor([[1.2,.4,-.2],[.4,.9,.15],[-.2,.15,.7]],dtype=dtype)
    sigma=torch.tensor([[.7,.2],[.2,.5]],dtype=dtype);L=torch.linalg.cholesky(sigma)
    means=torch.tensor([[1.1,-.7,.5],[-1.,.8,-.4]],dtype=dtype)
    def logp(x,z,v):
        C=covariance+torch.block_diag(torch.zeros(1,1,dtype=dtype),v*sigma)
        point=torch.cat([x,z]);diff=point[None,:]-means
        lp=-.5*(torch.einsum('ki,ij,kj->k',diff,torch.linalg.inv(C),diff)+torch.logdet(C))
        return torch.logsumexp(lp+torch.tensor([.4,.6],dtype=dtype).log(),0)
    residuals=[];bad=[]
    for seed in range(8):
        generator=torch.Generator().manual_seed(seed)
        x=torch.randn(1,dtype=dtype,generator=generator).requires_grad_()
        z=torch.randn(2,dtype=dtype,generator=generator).requires_grad_()
        v=torch.tensor(.8+seed*.1,dtype=dtype,requires_grad=True)
        value=logp(x,z,v);sx,sz=torch.autograd.grad(value,(x,z),create_graph=True)
        gradz=torch.autograd.grad(sx.sum(),z,create_graph=True)[0]
        H=torch.stack([torch.autograd.grad(gradz[j],z,create_graph=True,retain_graph=True)[0] for j in range(2)])
        dv=torch.autograd.grad(sx.sum(),v,retain_graph=True)[0]
        lap=(sigma*H).sum();drift=gradz@(sigma@sz)
        residuals.append(float((dv-.5*lap-drift).abs().detach()))
        bad.append(float((dv-.5*lap+drift).abs().detach()))
    assert max(residuals)<1e-10 and max(bad)>.01
    # The score relation also holds for epsilon_x, a factor independent of v,z.
    class Exact(torch.nn.Module):
        def forward(self,values,t,v):
            outputs=[]
            for point,variance in zip(values,v[:,0]):
                xx=point[:1].detach().requires_grad_();zz=point[1:].detach().requires_grad_()
                gx,gz=torch.autograd.grad(logp(xx,zz,variance),[xx,zz])
                outputs.append(torch.cat([-gx,-variance.sqrt()*(gz@L)]))
            return torch.stack(outputs)
    # x_dim1 and z_dim2 differ here, so independently check the mathematical FD.
    x=torch.tensor([.4],dtype=dtype);z=torch.tensor([.2,-.3],dtype=dtype);v=torch.tensor(1.2,dtype=dtype)
    def sx_at(xx,zz,vv):
        xx=xx.detach().requires_grad_();zz=zz.detach().requires_grad_()
        gx,gz=torch.autograd.grad(logp(xx,zz,vv),[xx,zz]);return gx.detach(),gz.detach()
    f,g=sx_at(x,z,v);direction=sigma@g;errors=[]
    for h in [.08,.04,.02]:
        lap=sum((sx_at(x,z+h*L[:,j],v)[0]+sx_at(x,z-h*L[:,j],v)[0]-2*f)/(h*h) for j in range(2))
        dv=(sx_at(x,z,v+h)[0]-sx_at(x,z,v-h)[0])/(2*h)
        drift=(sx_at(x,z+h*direction,v)[0]-sx_at(x,z-h*direction,v)[0])/(2*h)
        errors.append(float((dv-.5*lap-drift).abs()))
    assert errors[2]<errors[1]<errors[0] and errors[2]<.002
    # Observed R4 is the base distribution. Added noise is not a clean EEG oracle.
    return dict(status='completed',scope='synthetic-pass only',exact_nonGaussian_identity_max_error=max(residuals),
        wrong_sign_counterexample_max_error=max(bad),finite_difference_errors=errors,
        data_Gaussianity_required=False,independent_Gaussian_augmentation_required=True,
        actual_generation_improvement_proven=False)


if __name__=='__main__':
    import json
    print(json.dumps(validate_math()))
