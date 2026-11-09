"""Real author VAR/VQ code, full10 scales/680 tokens, reduced RANDOM weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import types
import torch

L=Path(__file__).resolve().parent
SOURCE=L/'reproduction_sources/direction_switch_20261028_v1/avde_3be06a90ae69'


def digest(module):
    h=hashlib.sha256()
    for key,value in module.state_dict().items():
        value=value.detach().contiguous().cpu()
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def run(output):
    started=time.monotonic();output=Path(output);assert not output.exists()
    inventory=json.loads((SOURCE/'source_inventory.json').read_bytes())
    for item in inventory['files']:assert hashlib.sha256((SOURCE/item['path']).read_bytes()).hexdigest()==item['sha256']
    package=types.ModuleType('models');package.__path__=[str(SOURCE/'models')]
    assert 'models' not in sys.modules;sys.modules['models']=package;sys.path.insert(0,str(SOURCE))
    import dist
    dist.__dict__['__device']='cpu'
    from models.var_eeg import VAREEG
    from models.vqvae import VQVAE
    from EEG_native_marginal_core_20261029_v2 import NativeMarginalDecoder
    import EEG_multiscale_native_kernel_DRAFT_20261101_v1 as M
    torch.set_num_threads(2);torch.manual_seed(1729);assert not torch.cuda.is_initialized()
    patches=(1,2,3,4,5,6,8,10,13,16);K=7
    vae=VQVAE(vocab_size=32,z_channels=8,ch=32,test_mode=True,share_quant_resi=4,v_patch_nums=patches).eval().requires_grad_(False)
    d=VAREEG(vae,depth=2,embed_dim=64,num_heads=4,cond_drop_rate=0.,patch_nums=patches,
             attn_l2_norm=True,flash_if_available=False,fused_if_available=False)
    d.init_weights(init_adaln=.5,init_adaln_gamma=.1,init_head=.2)
    native=NativeMarginalDecoder(d,torch.randn(K+1,64)*.7).eval().requires_grad_(False)
    assert d.L==680 and d.first_l==1 and d.C==64 and d.V==32
    before=dict(VAR=digest(d),VQ=digest(vae));ids={id(p) for p in native.parameters()}
    checks=[];details={}
    # Independent scoring of a synthetic, complete image under each native class.
    codes=torch.randint(0,d.V,(1,d.L));parts=list(codes.split([p*p for p in patches],dim=1))
    previous=vae.quantize.idxBl_to_var_input(parts)
    assert previous.shape==(1,679,d.Cvae)
    condition=native.native_class_embedding[:K]
    with torch.inference_mode():
        full=native(previous.expand(K,-1,-1),condition)
        likelihoods=M.complete_scale_log_likelihoods(native,previous.expand(K,-1,-1),codes.expand(K,-1),condition)
        explicit=full.float().log_softmax(-1).gather(-1,codes.expand(K,-1)[:,:,None]).squeeze(-1).double()
        expected=torch.stack([explicit[:,lo:hi].sum(-1) for lo,hi in d.begin_ends],1)
        assert torch.equal(expected,likelihoods) and likelihoods.shape==(K,10)
        details['all680_per_scale_NLL_sum_maxerror']=float((likelihoods.sum(1)-explicit.sum(1)).abs().max())
        assert details['all680_per_scale_NLL_sum_maxerror']<1e-9
        checks.append('all7_native_classes_all10_scales_680_true_tokens_scored_as_sum_of_token_log_probabilities')
        maximum=0.
        for k,(lo,hi) in enumerate(d.begin_ends):
            actual=M.stage_logits(native,previous[:,:hi-1].expand(K,-1,-1),condition,k)
            error=float((actual-full[:,lo:hi]).abs().max());maximum=max(maximum,error)
            assert torch.allclose(actual,full[:,lo:hi],atol=2e-6,rtol=2e-5)
        details['all10_fresh_prefix_vs_full_forward_logit_maxerror']=maximum
        checks.append('each_complete_scale_fresh_prefix_matches_same_native_class_full680_teacher_forward')
        k=3;lo,hi=d.begin_ends[k]
        # Targets from this scale onward cannot affect inputs or its logits.
        altered=[value.clone() for value in parts]
        for j in range(k,len(altered)):altered[j]=(altered[j]+7)%d.V
        changed=vae.quantize.idxBl_to_var_input(altered)
        assert torch.equal(previous[:,:hi-1],changed[:,:hi-1])
        assert torch.equal(M.stage_logits(native,previous[:,:hi-1],condition[:1],k),M.stage_logits(native,changed[:,:hi-1],condition[:1],k))
        checks.append('current_and_future_VQ_answers_cannot_change_current_scale_causal_features_or_logits')
        # Native fixed-class cached generation supplies a real reference path.
        captured=[];indices=[];lengths=[];original_logits=d.get_logits;original_embedding=vae.quantize.embedding.forward
        def capture_logits(hidden,cond):
            value=original_logits(hidden,cond);captured.append(value.clone());lengths.append([b.attn.cached_k.shape[2] for b in d.blocks]);return value
        def capture_codes(value):indices.append(value.clone());return original_embedding(value)
        d.get_logits=capture_logits;vae.quantize.embedding.forward=capture_codes
        try:reference=native.generate(condition[:1],seed=1729,guidance=0.,top_k=5)
        finally:d.get_logits=original_logits;vae.quantize.embedding.forward=original_embedding
        assert lengths==[[end,end] for _,end in d.begin_ends]
        assert all(not block.attn.caching and block.attn.cached_k is None for block in d.blocks)
        history=vae.quantize.idxBl_to_var_input(indices);current_cond=torch.cat((condition[:1],native.native_class_embedding[-1:]))
        max_cached=0.
        for k,(lo,hi) in enumerate(d.begin_ends):
            fresh=M.stage_logits(native,history[:,:hi-1].repeat(2,1,1),current_cond,k)
            error=float((fresh-captured[k]).abs().max());max_cached=max(max_cached,error)
            assert torch.allclose(fresh,captured[k],atol=2e-6,rtol=2e-5)
        details['native_cached_vs_fresh_prefix_all10_logit_maxerror']=max_cached
        checks.append('fixed_native_class_cached_and_fresh_prefix_paths_match_on_identical_actual_generated_history')
        sequence=torch.zeros(1,10,dtype=torch.long)
        replay,proof=M.generate_class_chain(native,sequence,image_seed=1729,top_k=5,keep_tensor_trace=True)
        assert torch.equal(replay,reference)
        assert all(torch.equal(item['codes'],old) for item,old in zip(proof['records'],indices))
        assert proof['actual_generated_tokens']==680 and proof['full_prefix_transformer_calls']==10
        checks.append('constant_class_sequence_exactly_replays_all_native_random_model_tokens_and_VQ_decoded_pixels')
        switched=torch.tensor([[0,1,2,3,4,5,6,0,1,2]],dtype=torch.long)
        image,trace=M.generate_class_chain(native,switched,image_seed=1729,top_k=5,keep_tensor_trace=True)
        own_history=vae.quantize.idxBl_to_var_input([item['codes'] for item in trace['records']])
        for k,(lo,hi) in enumerate(d.begin_ends):
            cond=torch.cat((condition[switched[0,k]:switched[0,k]+1],native.native_class_embedding[-1:]))
            expected=M.stage_logits(native,own_history[:,:hi-1].repeat(2,1,1),cond,k)
            assert torch.allclose(trace['records'][k]['logits'],expected,atol=2e-6,rtol=2e-5)
        assert image.shape==(1,3,256,256) and torch.isfinite(image).all()
        checks.append('switched_class_generation_all10_actual_logits_equal_defined_fresh_current_class_kernels')
        future=switched.clone();future[:,1:]=6
        unused,future_trace=M.generate_class_chain(native,future,image_seed=1729,top_k=5,keep_tensor_trace=True)
        assert torch.equal(trace['records'][0]['codes'],future_trace['records'][0]['codes'])
        checks.append('future_class_choices_do_not_affect_earlier_tokens_with_private_image_rng')
        # Demonstrate stale old-class cache differs from the intended new-class kernel.
        k=3;accum=torch.zeros(1,d.Cvae,16,16);position=d.lvl_embed(d.lvl_1L)+d.pos_1LC
        old_cond=torch.cat((condition[:1],native.native_class_embedding[-1:]))
        new_cond=torch.cat((condition[1:2],native.native_class_embedding[-1:]))
        token=old_cond[:,None].expand(-1,1,-1)+d.pos_start+position[:,:1]
        with native._cached_attention():
            for j in range(k+1):
                cond=old_cond if j<k else new_cond;hidden=token
                for block in d.blocks:hidden=block(hidden,d.shared_ada_lin(cond),None)
                if j==k:stale=d.get_logits(hidden,cond);break
                emb=vae.quantize.embedding(parts[j]).transpose(1,2).reshape(1,d.Cvae,patches[j],patches[j])
                accum,following=vae.quantize.get_next_autoregressive_input(j,10,accum,emb)
                start,end=d.begin_ends[j+1]
                token=(d.word_embed(following.flatten(2).transpose(1,2))+position[:,start:end]).repeat(2,1,1)
        lo,hi=d.begin_ends[k];fresh=M.stage_logits(native,previous[:,:hi-1].repeat(2,1,1),new_cond,k)
        difference=float((stale-fresh).abs().max());assert difference>1e-8
        details['actual_author_random_decoder_stale_class_cache_logit_error']=difference
        checks.append('actual_author_random_decoder_counterexample_rejects_reusing_previous_class_KV_as_new_class_kernel')
    # Input and exception handling must leave the frozen module/cache unchanged.
    invalids=[lambda:M.stage_logits(native,previous,condition[:1],0),
              lambda:M.generate_class_chain(native,torch.full((1,10),K,dtype=torch.long),image_seed=1729)]
    for call in invalids:
        try:call()
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('Invalid target-prefix length or null class accepted')
    for block in d.blocks:block.attn.kv_caching(True)
    try:
        try:M.stage_logits(native,previous[:,:0],condition[:1],0)
        except RuntimeError:pass
        else:raise AssertionError('Existing cache accepted')
    finally:
        for block in d.blocks:block.attn.kv_caching(False)
    old=d.get_logits
    def failure(*args):raise RuntimeError('injected inside full-prefix native logits')
    d.get_logits=failure
    try:
        try:M.generate_class_chain(native,torch.zeros((1,10),dtype=torch.long),image_seed=1729)
        except RuntimeError:pass
        else:raise AssertionError('Expected exception')
    finally:d.get_logits=old
    assert all(not block.attn.caching and block.attn.cached_k is None and block.attn.cached_v is None for block in d.blocks)
    checks.append('invalid_prefix_null_class_stale_cache_and_injected_exception_are_rejected_without_native_cache_mutation')
    logits=torch.randn(2,10,K);one=torch.ones(2,9);zero=torch.zeros(2,9)
    global_rng=torch.get_rng_state().clone()
    a,trace_a=M.draw_class_chain(logits,one,generator=torch.Generator().manual_seed(99))
    b,trace_b=M.draw_class_chain(logits,zero,generator=torch.Generator().manual_seed(99))
    again,trace_again=M.draw_class_chain(logits,one,generator=torch.Generator().manual_seed(99))
    assert torch.equal(global_rng,torch.get_rng_state()) and torch.equal(a,again)
    assert (a==a[:,:1]).all() and trace_a['private_latent_rng_after']==trace_b['private_latent_rng_after']
    for key in ('initial_uniform','stay_uniform','refresh_uniform'):assert torch.equal(trace_a[key],trace_again[key]) and torch.equal(trace_a[key],trace_b[key])
    assert trace_a['uniform_draws_per_query']==19
    checks.append('private_latent_rng_replay_rho_endpoints_and_fixed19_uniform_draws_per_query_do_not_touch_image_or_global_rng')
    assert ids=={id(p) for p in native.parameters()} and before==dict(VAR=digest(d),VQ=digest(vae))
    assert not torch.cuda.is_initialized()
    checks.append('all_author_random_VAR_VQ_tensors_and_parameter_objects_frozen_no_CUDA_or_optimizer')
    details['native_reference_pixel_SHA256']=hashlib.sha256(reference.numpy().tobytes()).hexdigest()
    details['fresh_fixed_class_pixel_SHA256']=hashlib.sha256(replay.numpy().tobytes()).hexdigest()
    details['switched_class_pixel_SHA256']=hashlib.sha256(image.numpy().tobytes()).hexdigest()
    result=dict(status='PASS_REAL_AUTHOR_CODE_FULL680_REDUCED_RANDOM_NATIVE_KERNEL_DRAFT_CPU',checks=checks,details=details,
        seconds=time.monotonic()-started,author_commit=inventory['commit'],author_source_inventory_SHA256=hashlib.sha256((SOURCE/'source_inventory.json').read_bytes()).hexdigest(),
        actual_author_VAR_VQ_code=True,reduced_random_weight_model=True,depth=2,width=64,vocabulary=32,latent_classes=K,
        full_scales=list(patches),tokens=680,attention_l2_normalization=True,
        random_initialization_gate=.1,real_pretrained_weights_loaded=False,real_EEG_or_true_images_used=False,
        new_research_images=0,random_model_VQ_decodes=4,new_optimizer_updates=0,new_GPU_jobs=0,
        real_full_size_cache_numerics_or_cost_profile_NOT_yet_done=True,EEG_model_or_emission_bank_or_training_admitted=False,
        novelty_or_reconstruction_gain_established=False,
        new_source=dict(path=str(Path(M.__file__).resolve()),bytes=Path(M.__file__).stat().st_size,sha256=hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest()))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
