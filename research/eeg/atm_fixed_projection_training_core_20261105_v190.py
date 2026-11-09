"""Fixed projection training control: native gradient clipping, masked update.

The projection remains in native train mode. Its parameters and Adam state are
fixed; gradients through it still reach EEG trunk weights. Not a novelty claim.
"""
def native_gradients(net,meta,x,target,seed,objective):
    import torch
    torch.cuda.manual_seed(seed)
    pred=net(x,torch.ones(len(x),device=x.device,dtype=torch.long))
    logits=net.logit_scale.exp().clamp_max(100)*torch.nn.functional.normalize(pred,dim=1)@torch.nn.functional.normalize(target,dim=1).T
    loss,_=objective(logits,'nce_continued')
    gs=torch.autograd.grad(loss,meta['parameters'],allow_unused=True)
    assert torch.isfinite(loss) and all(g is None or torch.isfinite(g).all() for g in gs)
    return loss,gs,logits

def assign_clip_mask(net,meta,grads,freeze_projection=True):
    import torch
    net.zero_grad(set_to_none=True)
    assert len(grads)==len(meta['parameters'])
    for p,g in zip(meta['parameters'],grads):p.grad=None if g is None else g.detach().clone()
    # Match continued C's clipping norm before changing the allowed update set.
    norm=torch.nn.utils.clip_grad_norm_(meta['parameters'],1.)
    assert torch.isfinite(norm)
    frozen=0;active=0
    for name,p in net.named_parameters():
        if name.startswith('proj_eeg.'):
            frozen+=p.numel()
            if freeze_projection:p.grad=None # skips weight decay AND Adam moments
        elif p.grad is not None:active+=p.numel()
    assert frozen>0 and active>0
    return dict(preclip_full_parameter_norm=float(norm),frozen_projection_parameters=frozen,
                active_other_parameters=active,logit_scale_trainable=True,
                projection_dropout_native_train_mode=True)
