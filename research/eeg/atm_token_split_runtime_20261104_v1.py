"""Frozen native P-token substitution with native K/V and actual-Q inspection."""
import hashlib,io
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from atm_token_mean_split_20261104_v1 import torch_tokens,attention64

class TokenSplitAudit:
    def __init__(self,pipe,projection):
        self.pipe=pipe;self.projection=projection;self.state=None;self.pending=None;self.rows=[];self.cases={};self.handles=[];self.original_sdpa=F.scaled_dot_product_attention
        self.layers=[(n,p) for n,p in pipe.unet.attn_processors.items() if hasattr(p,'to_k_ip')]
        assert len(self.layers)==70
        for index,(name,p) in enumerate(self.layers):
            assert type(p).__name__=='IPAdapterAttnProcessor2_0' and len(p.to_k_ip)==len(p.to_v_ip)==1 and list(p.num_tokens)==[4] and p.scale==[1.0]
            self.handles.append(p.to_k_ip[0].register_forward_hook(self.k_hook(index,name)))
            self.handles.append(p.to_v_ip[0].register_forward_hook(self.v_hook(index,name)))
        self.handles.append(projection.register_forward_hook(self.projection_hook));F.scaled_dot_product_attention=self.sdpa

    def begin(self,tag,query_index,old_tokens,new_tokens,mode,audit=True):
        assert self.state is None and self.pending is None
        base=old_tokens if mode in ('native_old','new_mean_old_center') else new_tokens
        mixed=torch_tokens(old_tokens,new_tokens,mode);style=new_tokens if mode=='new_mean_old_center' else old_tokens
        assert torch.isfinite(mixed).all() and mixed.float().norm()<=4*max(old_tokens.float().norm(),new_tokens.float().norm())
        cache=[]
        with torch.inference_mode():
            for name,p in self.layers:
                row={}
                for key,module in [('K',p.to_k_ip[0]),('V',p.to_v_ip[0])]:
                    # Extra linear probes are engineering-only, not production NFE.
                    row['base_'+key]=module(base).float().cpu().numpy().reshape(4,-1)
                    row['style_'+key]=module(style).float().cpu().numpy().reshape(4,-1)
                cache.append(row)
        self.state=dict(tag=tag,query_index=query_index,old=old_tokens,new=new_tokens,base=base,mixed=mixed,mode=mode,audit=audit,counts=[0]*70,projection_calls=0,cache=cache)

    def projection_hook(self,module,args,output):
        if self.state is None:return None
        s=self.state;s['projection_calls']+=1;assert output.numel()==8192 and torch.equal(output.reshape(1,4,2048),s['base'])
        if s['mode'] in ('native_old','native_new') or torch.equal(s['old'],s['new']):return output
        return s['mixed'].reshape(output.shape)

    def k_hook(self,index,name):
        def hook(module,args,output):
            if self.state is None or not self.state['audit']:return None
            assert self.pending is None and args[0].numel()==8192 and torch.equal(args[0].reshape(1,4,2048),self.state['mixed'])
            self.pending=dict(index=index,name=name,K_shape=tuple(output.shape))
        return hook

    def v_hook(self,index,name):
        def hook(module,args,output):
            if self.state is None or not self.state['audit']:return None
            assert self.pending is not None and self.pending['index']==index and args[0].numel()==8192 and torch.equal(args[0].reshape(1,4,2048),self.state['mixed'])
            self.pending['V_seen']=True
        return hook

    def sdpa(self,q,k,v,*args,**kwargs):
        output=self.original_sdpa(q,k,v,*args,**kwargs)
        if self.pending is None:return output
        assert self.state is not None and self.pending['V_seen'] and k.shape[-2]==v.shape[-2]==4 and q.shape[0]==1
        assert not args and kwargs.get('attn_mask') is None and kwargs.get('dropout_p',0)==0 and not kwargs.get('is_causal',False)
        s=self.state;index=self.pending['index'];step=s['counts'][index];assert step<4;s['counts'][index]+=1
        ids=np.unique(np.linspace(0,q.shape[-2]-1,min(8,q.shape[-2]),dtype=int));positions=torch.tensor(ids,device=q.device,dtype=torch.long)
        Q=q.index_select(-2,positions).float().cpu().numpy().astype(float);K=k.float().cpu().numpy().astype(float);V=v.float().cpu().numpy().astype(float)
        O=output.index_select(-2,positions).float().cpu().numpy().astype(float);reference,prob=attention64(Q,K,V);error=float(abs(O-reference).max()/max(1.,abs(reference).max()));assert error<.005
        heads,dim=q.shape[1],q.shape[-1];cache=s['cache'][index]
        def headed(a):return a.reshape(4,heads,dim).transpose(1,0,2)[None].astype(float)
        KB=headed(cache['base_K']);VB=headed(cache['base_V']);KS=headed(cache['style_K']);VS=headed(cache['style_V']);dk=KS.mean(-2,keepdims=True)-KB.mean(-2,keepdims=True);dv=VS.mean(-2,keepdims=True)-VB.mean(-2,keepdims=True)
        base_response,base_prob=attention64(Q,KB,VB);ideal_response,ideal_prob=attention64(Q,KB+dk,VB+dv)
        identity_error=float(abs(ideal_response-base_response-dv).max());assert identity_error<2e-12 and np.max(abs(ideal_prob-base_prob))<2e-12
        drift=float(abs(prob-base_prob).max());mean_mode=s['mode'] in ('old_mean_new_center','new_mean_old_center')
        if mean_mode:assert drift<.005
        row=dict(tag=s['tag'],query_index=s['query_index'],mode=s['mode'],layer=index,name=self.pending['name'],step=step,Q_shape=list(q.shape),Q_rows=ids.tolist(),scaled_native_GPU_vs_F64_SDPA_error=error,ideal_common_shift_identity_error=identity_error,
            actual_mean_swap_probability_drift=drift,probability_preservation_gate_applies=mean_mode,not_style_or_causal_semantic_claim=True)
        self.rows.append(row)
        if s['query_index'] in (0,7) and index in (0,69) and s['tag'] in ('old_mean_new_center','new_mean_old_center'):
            label=f"q{s['query_index']}_{s['tag']}_t{step}_l{index}";assert label not in self.cases
            self.cases[label]=dict(Q=Q,K=K,V=V,O=O,base_K=KB,base_V=VB,delta_K=dk,delta_V=dv,reference=reference)
        self.pending=None;return output

    def end(self):
        assert self.state is not None and self.pending is None;s=self.state
        assert s['projection_calls']==4
        if s['audit']:assert s['counts']==[4]*70
        self.state=None;return dict(tag=s['tag'],query_index=s['query_index'],projection_calls=4,actual_IP_attention_calls=sum(s['counts']),engineering_extra_linear_probes=70*4)

    def close(self):
        F.scaled_dot_product_attention=self.original_sdpa
        for h in self.handles:h.remove()
        self.handles=[]
