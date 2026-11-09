"""Mathematical checks for a proposed raw-EEG temporal attention interface."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import brainhive_temporal_conditional_attention_20261028_v1 as M

def run(output):
    torch.set_num_threads(2);torch.manual_seed(1729);dtype=torch.float64;checks=[]
    state={'proj_in.weight':torch.randn(1024,4250,dtype=dtype)/4250**.5,
           'proj_in.bias':torch.randn(1024,dtype=dtype)*.01,
           'block.layers.0.ff.w1.weight':torch.randn(1024,1024,dtype=dtype)/32,
           'block.layers.0.ff.w2.weight':torch.randn(1024,1024,dtype=dtype)/32,
           'block.layers.0.norm.weight':torch.rand(1024,dtype=dtype)+.5,
           'block.layers.0.norm.bias':torch.randn(1024,dtype=dtype)*.01}
    x=torch.randn(2,4250,dtype=dtype)
    c,r,note=M.temporal_tangents(x,state)
    h=F.linear(x,state['proj_in.weight'],state['proj_in.bias'])
    def f(h):
        z=h+F.linear(F.gelu(F.linear(h,state['block.layers.0.ff.w1.weight'])),state['block.layers.0.ff.w2.weight'])
        return F.layer_norm(z,(1024,),state['block.layers.0.norm.weight'],state['block.layers.0.norm.bias'],1e-6)
    assert (f(h)-c).abs().max()<1e-12
    assert (note['contributions'].sum(1)+state['proj_in.bias']-h).abs().max()<1e-12
    checks.append('native_first_layer_time_decomposition_and_complete_MLP_forward_match')
    errors=[]
    for t in range(5):
        _,expected=torch.autograd.functional.jvp(f,h,note['hidden_residuals'][:,t])
        errors.append(float((expected-r[:,t]).abs().max()))
    assert max(errors)<1e-11
    checks.append('all_five_output_coordinate_tangents_match_independent_autograd_JVP')
    assert r.sum(1).abs().max()<1e-11
    checks.append('temporal_innovations_sum_to_zero_without_changing_full_native_condition')
    wp=torch.randn(8192,1024,dtype=dtype)/32
    def prior(c):return F.linear(c,wp).reshape(len(c),4,2048)
    token,details,bound=M.frozen_prior_temporal_tokens(c,r,prior)
    analytic=torch.stack([F.linear(r[:,i],wp).reshape(2,4,2048) for i in range(5)],dim=2)
    analytic=analytic-analytic.mean(dim=2,keepdim=True)
    assert (bound['unbounded']-analytic).abs().max()<1e-11
    relative=details.square().sum(-1).mean(-1).sqrt()/token.norm(dim=-1)
    assert relative.max()<=.25+1e-12 and details.sum(2).abs().max()<1e-11
    checks.append('frozen_native_prior_JVP_matches_analytic_map_and_fixed_quarter_norm_budget')
    tzero,dzero,_=M.frozen_prior_temporal_tokens(c,torch.zeros_like(r),prior)
    assert dzero.count_nonzero()==0 and (tzero-prior(c)).abs().max()==0
    checks.append('zero_temporal_input_preserves_frozen_prior_tokens_without_new_parameters')
    q=torch.randn(2,2,7,8,dtype=dtype);k=torch.randn(2,2,4,8,dtype=dtype)
    dk=torch.randn(2,2,4,5,8,dtype=dtype);dv=torch.randn_like(dk)
    delta,p=M.attention_delta(q,k,dk,dv,return_probabilities=True)
    base=(q@k.transpose(-1,-2)/8**.5).softmax(-1)
    assert (p['joint'].sum(-1)-base).abs().max()<1e-14 and (p['conditional'].sum(-1)-1).abs().max()<1e-14
    checks.append('hierarchical_joint_preserves_original_four_token_marginal')
    flat,pflat=M.attention_delta(q,k,dk,dv,mode='flat_residual',return_probabilities=True)
    assert (pflat['marginal']-base).abs().max()>.01
    assert (flat-delta).norm()>.01
    checks.append('flat_residual_control_exposes_detail_partition_reweighting')
    zero=torch.zeros_like(dv)
    for mode in ['conditional','flat_residual']:assert torch.count_nonzero(M.attention_delta(q,k,dk,zero,mode=mode))==0
    checks.append('both_matched_operators_have_zero_value_exact_baseline_fallback')
    small=M.attention_delta(q,k,dk,dv,query_chunk=3)
    assert (small-delta).abs().max()<1e-14
    checks.append('query_chunking_preserves_operator')
    readout=M.TemporalReadout().double();tokens=torch.randn(2,4,2048,dtype=dtype)
    kd,vd=readout(r,tokens)
    assert sum(v.numel() for v in readout.parameters())==656128 and vd.count_nonzero()==0
    probe=torch.randn_like(vd);loss=(vd*probe).sum();loss.backward()
    assert readout.value.weight.grad.norm()>0
    checks.append('zero_initialized_values_keep_nonzero_trainable_value_gradient')
    readout.zero_grad(set_to_none=True)
    wk=torch.randn(16,2048,dtype=dtype)/2048**.5;wv=torch.randn_like(wk)/2048**.5
    base_key=F.linear(tokens,wk).reshape(2,4,2,8).permute(0,2,1,3)
    dk_embed,dv_embed=readout(r,tokens)
    detail_key=F.linear(dk_embed,wk).reshape(2,4,5,2,8).permute(0,3,1,2,4)
    detail_value=F.linear(dv_embed,wv).reshape(2,4,5,2,8).permute(0,3,1,2,4)
    response=M.attention_delta(q,base_key,detail_key,detail_value)
    assert response.count_nonzero()==0
    (response*torch.randn_like(response)).sum().backward()
    assert readout.value.weight.grad.norm()>1e-12
    checks.append('zero_value_gradient_survives_native_KV_projection_and_conditional_attention')
    with torch.no_grad():readout.value.weight.normal_(std=.01)
    ka,va=readout(r,tokens);kb,vb=readout(r[:,[4,3,2,1,0]],tokens)
    # Chronological positions are held fixed while measurements are reversed.
    assert (va-vb[:,:,list(reversed(range(5))),:]).norm()>1e-6
    checks.append('fixed_chronological_embedding_distinguishes_temporal_reversal')
    value=dict(status='D37_TEMPORAL_CONDITIONAL_ATTENTION_MATH_CHECKS_PASSED',tests=len(checks),checks=checks,
               maximum_native_tangent_JVP_error=max(errors),unused_synthetic_readout_fixture_parameters=656128,
               primary_candidate_additional_learned_parameters=0,
               actual_EEG_read=False,new_models_fit=0,new_images=0,CUDA_initialized=torch.cuda.is_initialized())
    assert not value['CUDA_initialized']
    with output.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
