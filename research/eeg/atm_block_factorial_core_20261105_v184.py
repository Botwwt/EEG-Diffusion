"""Frozen two-by-two EEG trunk / visual-head intervention. No fit or selector."""
import numpy as np
ARMS=("N_trunk_N_head","N_trunk_P_head","P_trunk_N_head","P_trunk_P_head")
FACTORS=((0,0),(0,1),(1,0),(1,1))
def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def project(x,weights):
    z=(x@weights["image_embeds.weight"].T+weights["image_embeds.bias"]).reshape(-1,4,2048)
    z=z-z.mean(-1,keepdims=True)
    return z/np.sqrt((z*z).mean(-1,keepdims=True)+1e-5)*weights["norm.weight"]+weights["norm.bias"]
def measure(x,y,groups,weights,full_gallery):
    x=x.astype(np.float64);y=y.astype(np.float64)
    px,py=project(x,weights),project(y,weights)
    a,b=unit(x),unit(y);pa,pb=unit(px),unit(py)
    out=dict(F_cosine=(a*b).sum(-1),F_MSE=((x-y)**2).mean(-1),condition_norm=np.linalg.norm(x,axis=-1),
             P_cosine=(pa*pb).sum(-1).mean(-1),P_MSE=((px-py)**2).mean((1,2)))
    n=len(x)
    if full_gallery:
        sims={"F":a@b.T,"P":pa.reshape(n,-1)@pb.reshape(n,-1).T/4}
        for name,s in sims.items():
            true=np.diag(s)
            wins=(s<true[:,None]).astype(float)+.5*(s==true[:,None])
            np.fill_diagonal(wins,0)
            out[name+"_gallery_AFC"]=wins.sum(1)/(n-1)
            out[name+"_gallery_top1"]=(s.argmax(1)==np.arange(n)).astype(float)
            out[name+"_gallery_top5"]=(np.argsort(-s,axis=1,kind="stable")[:,:5]==np.arange(n)[:,None]).any(1).astype(float)
    if max(map(len,groups))>1:
        for name,aa,bb,div in (("F",a,b,1),("P",pa.reshape(n,-1),pb.reshape(n,-1),4)):
            top=np.zeros(n);afc=np.zeros(n)
            for ids in groups:
                assert len(ids)==10
                ss=aa[ids]@bb[ids].T/div;di=np.diag(ss)
                win=(ss<di[:,None]).astype(float)+.5*(ss==di[:,None])
                np.fill_diagonal(win,0)
                top[ids]=(ss.argmax(1)==np.arange(10));afc[ids]=win.sum(1)/9
            out[name+"_within10_top1"]=top;out[name+"_within10_AFC"]=afc
    assert all(np.isfinite(v).all() for v in out.values())
    return out
def summarize(perarm,groups):
    means={arm:{k:float(v.mean()) for k,v in perarm[arm].items()} for arm in ARMS}
    # All six contrasts and the factorial interaction, not a selected winner.
    coeffs={ARMS[j]+"-"+ARMS[i]:{ARMS[j]:1,ARMS[i]:-1} for i in range(4) for j in range(i+1,4)}
    coeffs["interaction_PP-PN-NP+NN"]={ARMS[3]:1,ARMS[2]:-1,ARMS[1]:-1,ARMS[0]:1}
    draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)))
    contrasts={}
    for title,coeff in coeffs.items():
        contrasts[title]={}
        for key in perarm[ARMS[0]]:
            d=sum(v*perarm[arm][key] for arm,v in coeff.items())
            gm=np.array([d[ids].mean() for ids in groups])
            contrasts[title][key]=dict(difference=float(d.mean()),descriptive_concept95=np.quantile(gm[draw].mean(1),[.025,.975]).tolist())
    return dict(means=means,contrasts=contrasts)
