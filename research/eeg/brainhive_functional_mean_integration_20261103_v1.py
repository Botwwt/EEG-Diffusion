"""Frozen native attention output fields; no processor or weight replacement."""
import hashlib
import numpy as np
from brainhive_functional_mean_calibration_CPU_20261103_v1 import output_bias,wrong_field

ARMS=('native_shared_mean','ordinary_shared_mean','ordinary_full_mean_matched','ordinary_wrong_field','ordinary_shared_donor','ordinary_donor')

def background(arm):return 0 if arm=='native_shared_mean' else 1

class FunctionalMeanField:
    def __init__(self,pipe,projector,fields,torch):
        self.torch=torch;self.fields=fields;self.mode=None;self.capture_Q=False;self.record_rounding=False
        self.nodes={};self.bias64={'native':{},'ordinary':{}};self.bias={key:{} for key in self.bias64};self.handles=[]
        self.maximum_rounding_error=0.;self.maximum_rounding_bound=0.
        for name,processor in pipe.unet.attn_processors.items():
            if not hasattr(processor,'to_k_ip'):continue
            assert type(processor).__name__=='IPAttnProcessor2_0' and name.endswith('.processor')
            module=pipe.unet.get_submodule(name[:-len('.processor')])
            assert float(processor.scale)==1. and module.norm_q is None
            assert not module.training and not module.to_out[1].training
            self.nodes[name]=(module,processor)
            wv=processor.to_v_ip.weight.detach().cpu().double().numpy()
            wo=module.to_out[0].weight.detach().cpu().double().numpy()
            for bank in ('native','ordinary'):
                value=output_bias(fields[bank+'_shared_delta'],wv,wo,processor.scale,module.rescale_output_factor)
                self.bias64[bank][name]=value
                self.bias[bank][name]=torch.from_numpy(value.copy()).to(device=module.to_out[0].weight.device,dtype=module.to_out[0].weight.dtype)
            self.handles.append(module.register_forward_hook(self._output_hook(name)))
            self.handles.append(module.to_q.register_forward_hook(self._query_hook(name)))
        assert len(self.nodes)==70
        self.projector_handle=projector.register_forward_hook(self._token_hook)
        self.full_delta=torch.from_numpy(fields['ordinary_matched_full_delta'].copy()).to(device=next(projector.parameters()).device,dtype=next(projector.parameters()).dtype)
        self.reset(None)

    def reset(self,mode,capture_Q=False,record_rounding=False):
        assert mode is None or mode in ARMS or mode=='zero'
        self.mode=mode;self.capture_Q=capture_Q;self.record_rounding=record_rounding
        self.calls={name:0 for name in self.nodes};self.queries={name:[] for name in self.nodes};self.projector_calls=0

    def _query_hook(self,name):
        def hook(module,args,output):
            if self.capture_Q:
                assert output.ndim==3 and len(self.queries[name])<4
                ids=[0,output.shape[1]//2,output.shape[1]-1]
                self.queries[name].append(output[0,ids].detach().cpu().double().numpy())
        return hook

    def _output_hook(self,name):
        def hook(module,args,output):
            self.calls[name]+=1
            if self.mode in (None,'zero','ordinary_full_mean_matched','ordinary_donor'):return output
            bank='native' if self.mode=='native_shared_mean' else 'ordinary'
            value=self.bias[bank][name]
            if self.mode=='ordinary_wrong_field':value=self.torch.roll(value,len(value)//2)
            assert output.ndim in (3,4)
            b=value[None,None,:] if output.ndim==3 else value[None,:,None,None]
            changed=output+b
            if self.record_rounding:
                expected=output.float()+b.float();error=(changed.float()-expected).abs()
                finfo=self.torch.finfo(output.dtype);bound=finfo.eps*(output.float().abs()+b.float().abs())+finfo.tiny
                assert self.torch.isfinite(changed).all() and self.torch.all(error<=bound)
                self.maximum_rounding_error=max(self.maximum_rounding_error,float(error.max()))
                self.maximum_rounding_bound=max(self.maximum_rounding_bound,float(bound.max()))
            return changed
        return hook

    def _token_hook(self,module,args,output):
        self.projector_calls+=1
        assert output.shape[-2:]==(4,2048)
        if self.mode=='ordinary_full_mean_matched':return output+self.full_delta[None]
        return output

    def proof(self):
        assert set(self.calls.values())=={4} and self.projector_calls==1
        if self.capture_Q:assert all(len(v)==4 for v in self.queries.values())
        return dict(mode=self.mode,IP_modules=70,IP_module_calls=280,projector_calls=1,additional_UNet_calls=0,
                    all_native_keys_values_and_processors_unchanged=self.mode!='ordinary_full_mean_matched',
                    modified_module_output_bias=self.mode not in (None,'zero','ordinary_full_mean_matched','ordinary_donor'),
                    full_token_shift=self.mode=='ordinary_full_mean_matched')

    def field_arrays(self):
        return {bank+'__'+name:value for bank,rows in self.bias64.items() for name,value in rows.items()}

    def real_Q_algebra(self,tokens0,tokens1):
        """Independent float64 native-weight SDPA check on every captured Q."""
        assert all(len(v)==4 for v in self.queries.values())
        maximum=pair_error=key_error=0.;sites=0
        def attention(q,z,wk,wv,wo,heads,scale,rescale):
            dimension=wk.shape[0]//heads
            qq=q.reshape(len(q),heads,dimension).transpose(1,0,2)
            k=(z@wk.T).reshape(4,heads,dimension).transpose(1,0,2)
            v=(z@wv.T).reshape(4,heads,dimension).transpose(1,0,2)
            logits=qq@k.transpose(0,2,1)/np.sqrt(dimension)
            weights=np.exp(logits-logits.max(-1,keepdims=True));weights/=weights.sum(-1,keepdims=True)
            value=(weights@v).transpose(1,0,2).reshape(len(q),heads*dimension)@wo.T
            return value*scale/rescale,weights
        for name,(module,processor) in self.nodes.items():
            wk=processor.to_k_ip.weight.detach().cpu().double().numpy();wv=processor.to_v_ip.weight.detach().cpu().double().numpy();wo=module.to_out[0].weight.detach().cpu().double().numpy()
            for query in self.queries[name]:
                first,w0=attention(query,tokens0,wk,wv,wo,module.heads,processor.scale,module.rescale_output_factor)
                second,w1=attention(query,tokens1,wk,wv,wo,module.heads,processor.scale,module.rescale_output_factor)
                for bank in ('native','ordinary'):
                    delta=self.fields[bank+'_shared_delta'];bias=self.bias64[bank][name]
                    shifted0,a0=attention(query,tokens0+delta,wk,wv,wo,module.heads,processor.scale,module.rescale_output_factor)
                    shifted1,a1=attention(query,tokens1+delta,wk,wv,wo,module.heads,processor.scale,module.rescale_output_factor)
                    maximum=max(maximum,float(np.max(np.abs(shifted0-first-bias))),float(np.max(np.abs(shifted1-second-bias))))
                    pair_error=max(pair_error,float(np.max(np.abs((shifted0-shifted1)-(first-second)))))
                    key_error=max(key_error,float(np.max(np.abs(w0-a0))),float(np.max(np.abs(w1-a1))))
                sites+=1
        assert sites==280 and max(maximum,pair_error,key_error)<1e-9
        return dict(status='PASS_ALL70_LAYERS4_TIMES_REAL_Q_FLOAT64_ATTENTION_IDENTITIES',sites=sites,Q_positions_per_site=3,
                    common_bias_identity_error=maximum,paired_response_error=pair_error,key_gauge_error=key_error,
                    scope='Captured real native Q; originalFP16weights liftedCPU64 andexplicitsoftmax. Not proof of fulltrajectory orfloatingpoint exact paired preservation.')

    def close(self):
        self.projector_handle.remove()
        for h in self.handles:h.remove()
