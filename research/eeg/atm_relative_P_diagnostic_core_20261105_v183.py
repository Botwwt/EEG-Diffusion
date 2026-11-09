"""Fixed target diagnostic, not an admitted training recipe or novelty claim."""
import numpy as np
LOSSES = ("F_NCE", "P_MSE", "P_NCE")
SPLITS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))

def project(x, weights):
    import torch.nn.functional as F
    w, b, g, be = weights
    return F.layer_norm(F.linear(x, w, b).reshape(-1, 4, 2048), (2048,), g, be, 1e-5)

def losses(net, c, y, weights):
    import torch
    import torch.nn.functional as F
    scale = net.logit_scale.exp().clamp_max(100)
    sf = scale * F.normalize(c, dim=1) @ F.normalize(y, dim=1).T
    p, t = project(c, weights), project(y, weights).detach()
    # Fixed existing F temperature; P auxiliary cannot solve itself by changing temperature.
    sp = scale.detach() * (F.normalize(p, dim=2).flatten(1) @ F.normalize(t, dim=2).flatten(1).T) / 4
    labels = torch.arange(len(c), device=c.device)
    ce = lambda s: (F.cross_entropy(s, labels) + F.cross_entropy(s.T, labels)) / 2
    return (ce(sf), (p-t).square().mean(), ce(sp)), (sf, sp)

def repeat_mass(logits):
    """Input [2 coordinates,4 physical R1 plus physical R4,B,B]."""
    _, _, n, _ = logits.shape
    off = ~np.eye(n, dtype=bool)
    same = np.arange(n)[:, None]//10 == np.arange(n)[None, :]//10
    masks = (off, off & same, off & ~same)
    num = np.zeros((n, 2, 3, 2, 4), np.float64)
    den = np.zeros((n, 2, 3), np.float64)
    for space in range(2):
        s4 = logits[space, 4].astype(np.float64)
        er = np.exp(s4-s4.max(1, keepdims=True)); er /= er.sum(1, keepdims=True)
        ec = np.exp(s4-s4.max(0, keepdims=True)); ec /= ec.sum(0, keepdims=True)
        weights = (er/(2*n), ec/(2*n))
        for control in range(2):
            perm = np.arange(n) if control == 0 else (np.arange(n)//10)*10+(np.arange(n)%10+1)%10
            s = logits[space, :4, :, :][:, :, perm].astype(np.float64)
            diagonal = np.diagonal(s, axis1=1, axis2=2)
            for orientation in range(2):
                margins = (diagonal[:, :, None] if orientation == 0 else diagonal[:, None, :])-s
                axis = 1 if orientation == 0 else 0
                for ri, mask in enumerate(masks):
                    w = weights[orientation]*mask
                    if control == 0: den[:, space, ri] += w.sum(axis)
                    for a, b in SPLITS:
                        x = (margins[a[0]]+margins[a[1]])/2
                        y = (margins[b[0]]+margins[b[1]])/2
                        categories = ((x>0)&(y>0), (x<0)&(y<0), x*y<0, (x==0)|(y==0))
                        for k, cat in enumerate(categories):
                            num[:, space, ri, control, k] += (w*cat).sum(axis)/3
    np.testing.assert_allclose(num.sum(-1), np.repeat(den[..., None], 2, axis=-1), atol=2e-15, rtol=1e-12)
    np.testing.assert_allclose(num[:, :, 0], num[:, :, 1]+num[:, :, 2], atol=2e-15, rtol=1e-12)
    return num, den

def gradient_flat(grads, params):
    import torch
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for g, p in zip(grads, params)])

def cosine(gram):
    d = np.sqrt(np.maximum(np.diag(gram), 0))
    return gram/np.maximum(d[:, None]*d[None, :], 1e-300)
