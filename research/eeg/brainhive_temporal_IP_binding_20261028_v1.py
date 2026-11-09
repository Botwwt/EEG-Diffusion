"""Bind D37 residual attention to frozen native IP without changing its path.

The original processor still computes the whole native result. Temporary hooks
capture its real Q and image K; only an independently projected residual is added.
"""
import torch
from torch import nn
import torch.nn.functional as F
from brainhive_temporal_conditional_attention_20261028_v1 import attention_delta


class TemporalContext:
    def __init__(self):
        self.enabled=False
        self.detail=None
        self.mode='conditional'
        self.calls={}
        self.records={}

    def configure(self,detail,mode='conditional',enabled=True):
        assert mode in ['conditional','flat_residual']
        assert detail is None or detail.shape[1:]==(4,5,2048)
        self.detail=detail
        self.mode=mode
        self.enabled=enabled
        self.calls={}
        self.records={}


class TemporalIPProcessor(nn.Module):
    def __init__(self,original,context,name):
        super().__init__()
        assert type(original).__name__=='IPAttnProcessor2_0'
        self.original=original
        self.context=context
        self.name=name

    def __call__(self,attn,hidden_states,encoder_hidden_states=None,
                 attention_mask=None,temb=None,ip_hidden_states=None,*args,**kwargs):
        ctx=self.context
        if not ctx.enabled:
            return self.original(attn,hidden_states,encoder_hidden_states,
                                 attention_mask,temb,ip_hidden_states,*args,**kwargs)
        assert ip_hidden_states is not None and ctx.detail is not None
        assert not attn.training and not self.original.training
        assert isinstance(attn.to_out[0],nn.Linear)
        assert isinstance(attn.to_out[1],nn.Dropout) and not attn.to_out[1].training
        captured={}
        def capture(name):
            def hook(module,inputs,output):
                assert name not in captured
                captured[name]=output
            return hook
        hooks=[attn.to_q.register_forward_hook(capture('q')),
               self.original.to_k_ip.register_forward_hook(capture('k'))]
        try:
            base=self.original(attn,hidden_states,encoder_hidden_states,
                               attention_mask,temb,ip_hidden_states,*args,**kwargs)
        finally:
            for hook in hooks:hook.remove()
        q=captured['q'];k=captured['k'];B,N,inner=q.shape;H=attn.heads;D=inner//H
        assert inner%H==0 and k.shape==(B,4,inner)
        assert ctx.detail.shape==(B,4,5,2048)
        q=q.reshape(B,N,H,D).transpose(1,2)
        if attn.norm_q is not None:q=attn.norm_q(q)
        k=k.reshape(B,4,H,D).transpose(1,2)
        detail=ctx.detail.to(device=k.device,dtype=ip_hidden_states.dtype)
        dk=self.original.to_k_ip(detail).reshape(B,4,5,H,D).permute(0,3,1,2,4)
        dv=self.original.to_v_ip(detail).reshape(B,4,5,H,D).permute(0,3,1,2,4)
        first=self.name not in ctx.records
        result=attention_delta(q,k,dk,dv,mode=ctx.mode,query_chunk=256,
                               return_probabilities=first)
        if first:
            delta,prob=result
            root=(q.float()@k.float().transpose(-1,-2)/(D**.5)).softmax(-1)
            ctx.records[self.name]=dict(heads=H,head_dimension=D,query_positions=N,
                query_dtype=str(q.dtype),delta_RMS=float(delta.float().square().mean().sqrt()),
                native_marginal_RMS_error=float((prob['joint'].sum(-1)-root).square().mean().sqrt()),
                native_marginal_max_error=float((prob['joint'].sum(-1)-root).abs().max()))
        else:delta=result
        delta=delta.transpose(1,2).reshape(B,N,inner)
        # Native projection bias is already in base. Do not add it twice.
        delta=F.linear(delta,attn.to_out[0].weight,bias=None)*self.original.scale
        delta=attn.to_out[1](delta)
        if hidden_states.ndim==4:
            delta=delta.transpose(-1,-2).reshape_as(hidden_states)
        delta=delta/attn.rescale_output_factor
        assert delta.shape==base.shape and torch.isfinite(delta).all()
        ctx.calls[self.name]=ctx.calls.get(self.name,0)+1
        return base+delta


def bind_temporal_IP(unet,context):
    originals=dict(unet.attn_processors)
    replacement={name:TemporalIPProcessor(value,context,name)
                 if type(value).__name__=='IPAttnProcessor2_0' else value
                 for name,value in originals.items()}
    assert sum(isinstance(value,TemporalIPProcessor) for value in replacement.values())==70
    before=sum(p.numel() for p in unet.parameters())
    unet.set_attn_processor(dict(replacement))
    assert sum(p.numel() for p in unet.parameters())==before
    unet.eval()
    return originals
