"""Fixed-Q condition-space feasibility; classical constrained least squares."""
import numpy as np

def projector(c,w,b,gamma,beta,jacobian=False):
    w=w.reshape(4,2048,1024);b=b.reshape(4,2048)
    z=np.einsum('kdn,n->kd',w,c)+b;x=z-z.mean(1,keepdims=True)
    sd=np.sqrt((x*x).mean(1,keepdims=True)+1e-5);n=x/sd;p=n*gamma+beta
    if not jacobian:return p
    j=np.empty_like(w)
    for k in range(4):j[k]=(w[k]-w[k].mean(0,keepdims=True)-n[k,:,None]*(n[k]@w[k])[None]/2048)*(gamma/sd[k,0])[:,None]
    return p,j

def response(p,Q,wk,wv,jp=None):
    heads,d=Q.shape[0],Q.shape[-1]
    k=(p@wk.T).reshape(4,heads,d).transpose(1,0,2)
    v=(p@wv.T).reshape(4,heads,d).transpose(1,0,2)
    logits=Q@k.transpose(0,2,1)/np.sqrt(d);logits-=logits.max(-1,keepdims=True)
    a=np.exp(logits);a/=a.sum(-1,keepdims=True);o=a@v
    if jp is None:return o,k,v
    jk=np.stack([wk@jp[t] for t in range(4)]).reshape(4,heads,d,1024).transpose(1,0,2,3)
    jv=np.stack([wv@jp[t] for t in range(4)]).reshape(4,heads,d,1024).transpose(1,0,2,3)
    dl=np.einsum('hqd,htdn->hqtn',Q,jk)/np.sqrt(d)
    jo=np.einsum('hqt,htdn->hqdn',a,jv)+np.einsum('hqt,hqtd,hqtn->hqdn',a,v[:,None]-o[:,:,None],dl)
    return o,k,v,jo.reshape(-1,1024)

def solve(B,b,G=None,relative_eigen_cutoff=1e-10):
    if G is None:
        R=np.eye(B.shape[1]);eig=None
    else:
        eig,U=np.linalg.eigh((G+G.T)/2);assert eig[-1]>0 and eig[0]>-1e-10*eig[-1]
        keep=eig>relative_eigen_cutoff*eig[-1];R=U[:,keep]/np.sqrt(eig[keep])[None,:]
    H=B@R;u,s,vt=np.linalg.svd(H,full_matrices=False);keep=s>1e-10*s[0]
    z=(u[:,keep].T@b)/s[keep];v=R@(vt[keep].T@z)
    lam=u[:,keep]@((u[:,keep].T@b)/(s[keep]**2))
    metric=np.eye(B.shape[1]) if G is None else G
    return v,lam,dict(constraint_rank=int(keep.sum()),metric_rank=int(R.shape[1]),constraint_singular_values=s.tolist(),metric_eigenvalues=None if eig is None else eig.tolist(),
        relative_constraint_residual=float(np.linalg.norm(B@v-b)/max(np.linalg.norm(b),1e-12)),relative_KKT_residual=float(np.linalg.norm(metric@v-B.T@lam)/max(np.linalg.norm(metric@v),np.linalg.norm(B.T@lam),1e-12)))

def oracle(response0,response1,teachers):
    D=(teachers-teachers.mean(0)).reshape(10,-1);delta=(response1-response0).ravel()
    u,s,vt=np.linalg.svd(D,full_matrices=False);keep=s>1e-10*s[0];v=vt[keep].T@(vt[keep]@delta)
    exact=float(np.linalg.norm(D@(v-delta))/max(np.linalg.norm(D@delta),1e-12));assert exact<1e-8
    return response0+v.reshape(response0.shape),dict(rank=int(keep.sum()),margin_relative_error=exact,retained_update_energy=float(v@v/max(delta@delta,1e-30)))
