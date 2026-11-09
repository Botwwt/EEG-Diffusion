"""Same-Q native attention geometry, diagnostic only; no fitting or edits."""
def numpy_attention(q,k,v):
    import numpy as np
    logits=np.matmul(q,k.swapaxes(-1,-2))/q.shape[-1]**.5
    logits-=logits.max(-1,keepdims=True);p=np.exp(logits);p/=p.sum(-1,keepdims=True)
    return p@v

def numpy_measures(response,correct):
    import numpy as np
    diff=response[:4,None]-response[None,4:]
    mu=diff.mean(-2,keepdims=True);center=diff-mu
    distance=np.stack([(diff*diff).mean((-3,-2,-1)),(mu*mu).mean((-3,-2,-1)),(center*center).mean((-3,-2,-1))],-1)
    teacher=response[4:]-response[4:].mean(0,keepdims=True);tm=teacher.mean(-2,keepdims=True)
    scale=np.array([(teacher*teacher).mean(),(tm*tm).mean(),((teacher-tm)**2).mean()])
    delta=response[1:4]-response[:1];res=response[4+correct:5+correct]-response[:1]
    dm=delta.mean(-2,keepdims=True);rm=res.mean(-2,keepdims=True)
    energy=np.stack([(delta*delta).mean((-3,-2,-1)),(dm*dm).mean((-3,-2,-1)),((delta-dm)**2).mean((-3,-2,-1))],-1)
    align=np.stack([(delta*res).mean((-3,-2,-1)),(dm*rm).mean((-3,-2,-1)),((delta-dm)*(res-rm)).mean((-3,-2,-1))],-1)
    return distance,scale,energy,align

class SameQueryResponse:
    def __init__(self,pipe):
        import torch.nn.functional as F
        self.F=F;self.original=F.scaled_dot_product_attention;self.handles=[];self.state=None;self.pending=None;self.cases={};self.layouts=set()
        self.layers=[(n,p) for n,p in pipe.unet.attn_processors.items() if hasattr(p,'to_k_ip')]
        assert len(self.layers)==70
        for i,(n,p) in enumerate(self.layers):
            assert type(p).__name__=='IPAdapterAttnProcessor2_0' and list(p.num_tokens)==[4] and p.scale==[1.0]
            self.handles.append(p.to_k_ip[0].register_forward_hook(self.key_hook(i)))
            self.handles.append(p.to_v_ip[0].register_forward_hook(self.value_hook(i)))
        F.scaled_dot_product_attention=self.sdpa

    def begin(self,index,tokens):
        import torch
        assert self.state is None and self.pending is None and tokens.shape==(14,4,2048)
        cache=[]
        with torch.inference_mode():
            for name,p in self.layers:
                # Batch size one matches the actual native path for every condition.
                cache.append(tuple(torch.cat([module(tokens[j:j+1]) for j in range(14)]) for module in [p.to_k_ip[0],p.to_v_ip[0]]))
        self.state=dict(index=index,tokens=tokens,cache=cache,counts=[0]*70,rows=[])

    def key_hook(self,index):
        def hook(module,args,output):
            import torch
            if self.state is None:return
            assert self.pending is None and torch.equal(args[0].reshape(1,4,2048),self.state['tokens'][:1])
            self.layouts.add(('K',tuple(output.shape),tuple(self.state['cache'][index][0][:1].shape)))
            assert torch.equal(output.reshape_as(self.state['cache'][index][0][:1]),self.state['cache'][index][0][:1]), ('K',tuple(output.shape),index)
            self.pending=dict(index=index,V_seen=False)
        return hook

    def value_hook(self,index):
        def hook(module,args,output):
            import torch
            if self.state is None:return
            assert self.pending is not None and self.pending['index']==index
            self.layouts.add(('V',tuple(output.shape),tuple(self.state['cache'][index][1][:1].shape)))
            assert torch.equal(output.reshape_as(self.state['cache'][index][1][:1]),self.state['cache'][index][1][:1]), ('V',tuple(output.shape),index)
            self.pending['V_seen']=True
        return hook

    def sdpa(self,q,k,v,*args,**kwargs):
        import torch
        import numpy as np
        native=self.original(q,k,v,*args,**kwargs)
        if self.pending is None:return native
        assert self.pending['V_seen'] and q.shape[0]==1 and k.shape[-2]==v.shape[-2]==4
        assert not args and kwargs.get('attn_mask') is None and kwargs.get('dropout_p',0)==0 and not kwargs.get('is_causal',False)
        s=self.state;i=self.pending['index'];t=s['counts'][i];assert t<4;s['counts'][i]+=1
        positions=np.unique(np.linspace(0,q.shape[-2]-1,min(16,q.shape[-2]),dtype=np.int64))
        Q=q.index_select(-2,torch.tensor(positions,device=q.device)).float();h,d=q.shape[1],q.shape[-1]
        K,V=[z.reshape(14,4,h,d).transpose(1,2).float() for z in s['cache'][i]]
        response=torch.softmax((Q@K.transpose(-1,-2))/(d**.5),dim=-1)@V
        reference=native.index_select(-2,torch.tensor(positions,device=q.device)).float()
        error=float((response[:1]-reference).abs().max()/torch.maximum(reference.abs().max(),torch.ones((),device=q.device)))
        assert error<.005
        diff=response[:4,None]-response[None,4:];mu=diff.mean(-2,keepdim=True);center=diff-mu
        distance=torch.stack([(diff*diff).mean((-3,-2,-1)),(mu*mu).mean((-3,-2,-1)),(center*center).mean((-3,-2,-1))],-1)
        teacher=response[4:]-response[4:].mean(0,keepdim=True);tm=teacher.mean(-2,keepdim=True)
        scale=torch.stack([(teacher*teacher).mean(),(tm*tm).mean(),((teacher-tm)**2).mean()])
        correct=s['index']%10;delta=response[1:4]-response[:1];res=response[4+correct:5+correct]-response[:1]
        dm=delta.mean(-2,keepdim=True);rm=res.mean(-2,keepdim=True)
        energy=torch.stack([(delta*delta).mean((-3,-2,-1)),(dm*dm).mean((-3,-2,-1)),((delta-dm)**2).mean((-3,-2,-1))],-1)
        align=torch.stack([(delta*res).mean((-3,-2,-1)),(dm*rm).mean((-3,-2,-1)),((delta-dm)*(res-rm)).mean((-3,-2,-1))],-1)
        residual=(distance[1:,correct]-distance[:1,correct])-(energy-2*align)
        assert float(residual.abs().max())<2e-5*max(1.,float(distance.max()))
        assert float((distance[:,:,0]-distance[:,:,1]-distance[:,:,2]).abs().max())<2e-5*max(1.,float(distance.max()))
        row=(t,i,*[z.cpu().numpy() for z in [distance,scale,energy,align]],error,float(residual.abs().max()))
        s['rows'].append(row)
        if s['index'] in [0,79] and i in [0,69] and t in [0,3]:
            label=f'q{s["index"]}_t{t}_l{i}'
            self.cases[label]=dict(Q=Q.cpu().numpy(),K=K.cpu().numpy(),V=V.cpu().numpy(),native=reference.cpu().numpy(),positions=positions,
                distance=row[2],scale=row[3],energy=row[4],align=row[5])
        self.pending=None
        return native

    def end(self):
        assert self.state is not None and self.pending is None and self.state['counts']==[4]*70
        rows=self.state['rows'];assert len(rows)==280;self.state=None
        return rows

    def close(self):
        self.F.scaled_dot_product_attention=self.original
        for h in self.handles:h.remove()
        self.handles=[]

def summarize(distances,scales,energy,align):
    """Equal-time/equal-layer normalized distances; all fixed contrasts retained."""
    import numpy as np
    assert distances.shape==(80,4,70,4,10,3) and scales.shape==(80,4,70,3)
    assert np.isfinite(distances).all() and np.isfinite(scales).all()
    denom=np.maximum(scales,1e-12);scores=(distances/denom[:,:,:,None,None,:]).mean((1,2))
    per={};means={};arms=['NN','CC','CN','FH'];components=['full','sampled_position_mean','sampled_spatial_center'];correct=np.arange(80)%10
    for a,name in enumerate(arms):
        per[name]={}
        for k,comp in enumerate(components):
            sc=scores[:,a,:,k];hit=sc[np.arange(80),correct];other=np.arange(10)[None,:]!=correct[:,None]
            per[name][comp+'_normalized_MSE']=hit
            per[name][comp+'_top1']=(sc.argmin(1)==correct).astype(float)
            per[name][comp+'_AFC']=(((sc>hit[:,None])+.5*(sc==hit[:,None]))*other).sum(1)/9
        means[name]={k:float(v.mean()) for k,v in per[name].items()}
    draw=np.random.default_rng(1729).integers(0,8,(1000,8));contrasts={}
    for a in range(1,4):
        for b in range(a):
            name=arms[a]+'-'+arms[b];contrasts[name]={}
            for key in per[arms[a]]:
                delta=per[arms[a]][key]-per[arms[b]][key];ci=np.quantile(delta.reshape(8,10).mean(1)[draw].mean(1),[.025,.975])
                contrasts[name][key]=dict(difference=float(delta.mean()),CI95=ci.tolist())
    changes={}
    for a in range(3):
        changes[arms[a+1]+'-NN']={}
        for k,comp in enumerate(components):
            en=(energy[:,:,:,a,k]/denom[:,:,:,k]).mean((1,2));al=(align[:,:,:,a,k]/denom[:,:,:,k]).mean((1,2))
            actual=per[arms[a+1]][comp+'_normalized_MSE']-per['NN'][comp+'_normalized_MSE']
            err=float(abs(actual-en+2*al).max());assert err<.01*max(1.,float(abs(actual).max()))
            changes[arms[a+1]+'-NN'][comp]=dict(update_energy=float(en.mean()),twice_alignment=float(2*al.mean()),risk_difference=float(actual.mean()),max_identity_error=err)
    return dict(means=means,contrasts=contrasts,change_accounting=changes,scale_floor_count=int((scales<1e-12).sum()),
        risk_identity='delta teacher MSE = update energy - 2 alignment; arithmetic not causal attribution',
        limitations='Seen T80, eight concepts, one subject, single seed; 16 deterministic Q positions per layer/time under NN trajectories only. Not image improvement or heldout validation.'),per,draw,scores
