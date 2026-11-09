"""Actual author reduced random VAR/VQ; exact rollback and causal contract tests."""
import datetime,hashlib,json,sys,time,types
from pathlib import Path
import torch
import torch.nn.functional as F
from unittest.mock import patch

L=Path(__file__).resolve().parent;S=L/'reproduction_sources/direction_switch_20261028_v1/avde_3be06a90ae69'
inventory=json.loads((S/'source_inventory.json').read_bytes())
for item in inventory['files']:assert hashlib.sha256((S/item['path']).read_bytes()).hexdigest()==item['sha256']
package=types.ModuleType('models');package.__path__=[str(S/'models')];assert 'models' not in sys.modules;sys.modules['models']=package
sys.path.insert(0,str(S));import dist;dist.__dict__['__device']='cpu'
from models.var_eeg import VAREEG
from models.vqvae import VQVAE
from EEG_anchored_attention_20261029_v1 import EEGAnchoredAttentionDecoder

torch.set_num_threads(2);torch.manual_seed(1729);start=time.monotonic();checks={};details={}
def make(l2=False):
    vae=VQVAE(vocab_size=32,z_channels=8,ch=32,test_mode=True,v_patch_nums=(1,2,3))
    decoder=VAREEG(vae,depth=2,embed_dim=64,num_heads=4,cond_drop_rate=0.,patch_nums=(1,2,3),attn_l2_norm=l2,flash_if_available=False,fused_if_available=False)
    decoder.init_weights(init_adaln_gamma=.1)
    return decoder,EEGAnchoredAttentionDecoder(decoder).eval()
d,m=make();global_eeg=torch.randn(2,1024);local=torch.randn(2,5,1024)
memory=m.centered_memory(global_eeg,local);anchor=global_eeg[:,None].expand_as(memory)
previous=torch.randn(2,d.L-d.first_l,d.Cvae)
parameter_ids={id(p) for p in d.parameters()};checks['no_new_learned_parameter_objects']={id(p) for p in m.parameters()}==parameter_ids
with torch.no_grad():
    native=d(previous,global_eeg);disabled=m(previous,global_eeg);zero=m(previous,global_eeg,eeg_memory=anchor)
    checks['disabled_logits_exact_native']=torch.equal(disabled,native)
    checks['zero_local_innovation_logits_exact_native']=torch.equal(zero,native)
    actual=m(previous,global_eeg,eeg_memory=memory)
    checks['local_EEG_visible_at_identical_global_c']=(actual-native).abs().max().item()>1e-7
    altered=previous.clone();altered[:,4:]+=torch.randn_like(altered[:,4:])*10
    checks['future_image_cannot_leak_to_earlier_scales']=torch.equal(actual[:,:5],m(altered,global_eeg,eeg_memory=memory)[:,:5])
    permuted=m(previous,global_eeg,eeg_memory=memory[:,torch.tensor([4,0,2,1,3])])
    checks['EEG_token_axis_permutation_is_invariant']=torch.allclose(actual,permuted,atol=2e-6,rtol=2e-5)
    native_image=d.autoregressive_infer_cfg(2,global_eeg,g_seed=1729,cfg=4.,top_k=5)
    checks['disabled_pixels_exact_native']=torch.equal(native_image,m.generate(global_eeg,seed=1729,top_k=5))
    checks['zero_local_innovation_pixels_exact_native']=torch.equal(native_image,m.generate(global_eeg,eeg_memory=anchor,seed=1729,top_k=5))
    image=m.generate(global_eeg,eeg_memory=memory,seed=1729,top_k=5)
    checks['EEG_only_generation_replay']=torch.equal(image,m.generate(global_eeg,eeg_memory=memory,seed=1729,top_k=5))
    checks['actual_VQ_decode_finite']=image.shape[:2]==(2,3) and bool(torch.isfinite(image).all())
    checks['native_cache_released_after_success']=all(not b.attn.caching and b.attn.cached_k is None for b in d.blocks)
    cached,indices,cache_lengths=[],[],[];old_logits=d.get_logits;old_embedding=d.vae_quant_proxy[0].embedding.forward
    def capture_logits(hidden,condition):
        value=old_logits(hidden,condition);cached.append(value[:2].clone());cache_lengths.append([b.attn.cached_k.shape[2] for b in d.blocks]);return value
    def capture_indices(value):indices.append(value.clone());return old_embedding(value)
    d.get_logits=capture_logits;d.vae_quant_proxy[0].embedding.forward=capture_indices
    try:m.generate(global_eeg,eeg_memory=memory,seed=1729,guidance=0.,top_k=5)
    finally:d.get_logits=old_logits;d.vae_quant_proxy[0].embedding.forward=old_embedding
    history=d.vae_quant_proxy[0].idxBl_to_var_input(indices)
    difference=(m(history,global_eeg,eeg_memory=memory)-torch.cat(cached,1)).abs().max().item()
    checks['cache_and_teacher_forward_same_generated_history']=difference<2e-6;details['cached_full_logit_max_difference']=difference
    checks['native_cache_contains_image_tokens_only']=cache_lengths==[[1,1],[5,5],[14,14]]
    def injected_failure(hidden,condition):raise RuntimeError('injected after cache initialization')
    d.get_logits=injected_failure
    try:
        try:m.generate(global_eeg,eeg_memory=memory,seed=1729,top_k=5)
        except RuntimeError:pass
        else:raise AssertionError('Expected injected failure')
    finally:d.get_logits=old_logits
    checks['native_cache_released_after_exception']=all(not b.attn.caching and b.attn.cached_k is None for b in d.blocks)
    # Independent float64 full-key/value response minus anchor response. This
    # exercises conditional LayerNorm, native query bias and per-head cosine K.
    for l2 in (False,True):
        decoder,model=(d,m) if not l2 else make(True)
        condition=decoder.embed_proj(global_eeg);conditioning=decoder.shared_ada_lin(condition)
        embeddings=model._memory_embeddings(global_eeg,memory,condition);block=decoder.blocks[0];attn=block.attn
        _,_,scale,_,shift,_=model._native_condition_parts(block,conditioning)
        u0=block.ln_wo_grad(embeddings[0]).mul(scale.add(1)).add(shift)
        ue=block.ln_wo_grad(embeddings[1]).mul(scale.add(1)).add(shift)
        image_input=torch.randn(2,4,decoder.C)
        actual_innovation=model._innovation(image_input,attn,model._prepare_layer(block,conditioning,embeddings))
        weights=attn.mat_qkv.weight.double();qkv_bias=torch.cat((attn.q_bias,attn.zero_k_bias,attn.v_bias)).double()
        def qkv(x):return F.linear(x.double(),weights,qkv_bias).view(x.shape[0],x.shape[1],3,attn.num_heads,attn.head_dim).permute(2,0,3,1,4)
        query=qkv(image_input)[0];keys=qkv(ue)[1];values=qkv(ue)[2];anchor_values=qkv(u0)[2]
        if l2:
            query=F.normalize(query,dim=-1)*attn.scale_mul_1H11.double().clamp_max(attn.max_scale_mul).exp()
            keys=F.normalize(keys,dim=-1)
        response=((query@keys.transpose(-1,-2))*attn.scale).softmax(-1)@values-anchor_values[:,:,:1]
        expected=F.linear(response.transpose(1,2).reshape(2,4,decoder.C),attn.proj.weight.double(),None)
        error=(expected-actual_innovation.double()).abs().max().item()
        checks['independent_full_response_minus_anchor_l2_'+str(l2)]=error<2e-6
        details['independent_innovation_max_error_l2_'+str(l2)]=error
        checks['zero_key_value_innovation_exact_l2_'+str(l2)]=all(bool((x==0).all()) for x in model._prepare_layer(block,conditioning,model._memory_embeddings(global_eeg,anchor,condition)))
m.train();d.cond_drop_rate=.5
draws=[]
def zero_draw(*shape,device=None,**kwargs):
    draws.append(shape);return torch.zeros(*shape,device=device)
with patch('torch.rand',zero_draw):
    dropped=m(previous,global_eeg,eeg_memory=memory)
d.cond_drop_rate=0.
baseline_zero=m(previous,torch.zeros_like(global_eeg))
checks['one_shared_condition_dropout_mask']=draws==[(2,1)]
checks['dropping_EEG_removes_global_and_local_together']=torch.equal(dropped,baseline_zero)
m.zero_grad(set_to_none=True)
memory_gradient=memory.detach().requires_grad_();m(previous,global_eeg,eeg_memory=memory_gradient).square().mean().backward()
checks['gradient_reaches_local_EEG']=bool(torch.isfinite(memory_gradient.grad).all() and memory_gradient.grad.norm()>0)
checks['gradient_reaches_existing_QKV']=bool(d.blocks[0].attn.mat_qkv.weight.grad.norm()>0)
record=dict(status='PASS' if all(checks.values()) else 'FAIL',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,details=details,seconds=time.monotonic()-start,author_commit=inventory['commit'],actual_author_VAR_VQ_modules=True,reduced_random_weight_model=True,attention_gate_test_initialization=.1,pretrained_weights_loaded=False,real_EEG_or_research_images_used=False,GPU_jobs=0,trained_models=0,original_decoder_parameters=sum(p.numel() for p in d.parameters()),new_learned_parameters=0,source_sha256=hashlib.sha256((L/'EEG_anchored_attention_20261029_v1.py').read_bytes()).hexdigest())
out=L/'audit/EEG_anchored_attention_20261029_v1';out.mkdir(exist_ok=True)
target=out/'CPU_contract_actual_v1.json'
if target.exists():
    prior=target.read_bytes()
    (out/('prior_'+hashlib.sha256(prior).hexdigest()[:12]+'.json')).write_bytes(prior)
target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(record));assert all(checks.values()),checks
